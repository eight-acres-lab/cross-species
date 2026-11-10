#!/usr/bin/env python3
"""Post-hoc, CPU-only retrieval from frozen State population predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tied_ranking(scores, correct, *, tolerance=1e-10):
    """Expected metrics under uniform ordering within the correct score's tie."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= correct < len(scores):
        raise ValueError("Require finite candidate scores and a valid correct index")
    value = scores[correct]
    above = int(np.sum(scores > value + tolerance))
    tied = int(np.sum(np.abs(scores - value) <= tolerance))
    positions = np.arange(above + 1, above + tied + 1, dtype=float)
    return {"top1": 1. / tied if above == 0 else 0.,
            "rank": float(positions.mean()), "mrr": float((1. / positions).mean()),
            "correct_tie_size": tied}


def profile_scores(prediction, candidates):
    """All features are fixed; any constant candidate leaves the full pool undefined."""
    prediction, candidates = np.asarray(prediction, float), np.asarray(candidates, float)
    if candidates.ndim != 2 or prediction.shape != (candidates.shape[1],):
        raise ValueError("Prediction and candidates must share the complete feature axis")
    if not np.isfinite(prediction).all() or not np.isfinite(candidates).all():
        raise ValueError("Nonfinite profiles")
    p = prediction - prediction.mean()
    c = candidates - candidates.mean(axis=1, keepdims=True)
    pnorm, cnorm = np.linalg.norm(p), np.linalg.norm(c, axis=1)
    if pnorm <= 1e-12:
        return None, "constant_prediction"
    if np.any(cnorm <= 1e-12):
        return None, "constant_candidate"
    return np.clip(c @ p / (cnorm * pnorm), -1, 1), None


def evaluate_profiles(predicted, observed, queries, conditions, donors, references=None):
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    if predicted.shape != observed.shape or predicted.ndim != 2 or len(queries) != len(predicted):
        raise ValueError("Frozen mean profiles and query metadata disagree")
    lookup = {}
    for index, query in enumerate(queries):
        key = (query["condition"], query["donor"])
        if key in lookup:
            raise ValueError("Duplicate donor/cytokine candidate")
        lookup[key] = index
    if set(lookup) != {(c, d) for c in conditions for d in donors}:
        raise ValueError("Require the complete fixed donor-by-cytokine candidate grid")
    k = len(conditions)
    rows = []
    for query_index, query in enumerate(queries):
        donor = query["donor"]
        candidate_indices = [lookup[(condition, donor)] for condition in conditions]
        reference = np.zeros(predicted.shape[1]) if references is None else references[donor]
        scores, reason = profile_scores(predicted[query_index] - reference,
                                        observed[candidate_indices] - reference)
        row = {"condition": query["condition"], "donor": donor,
               "candidate_conditions": list(conditions), "candidate_count": k,
               "chance_top1": 1 / k, "chance_rank": (k + 1) / 2,
               "chance_mrr": float(np.sum(1 / np.arange(1, k + 1)) / k),
               "defined": scores is not None, "undefined_reason": reason}
        if scores is not None:
            row.update(tied_ranking(scores, conditions.index(query["condition"])))
            row["pearson_scores"] = scores.tolist()
        rows.append(row)
    return rows


def summarize(rows, allowed=None):
    selected = [r for r in rows if r["defined"] and
                (allowed is None or (r["condition"], r["donor"]) in allowed)]
    def metrics(subset):
        return {metric: float(np.mean([r[metric] for r in subset])) if subset else None
                for metric in ["top1", "rank", "mrr", "chance_top1", "chance_rank", "chance_mrr"]}
    by_condition = {c: metrics([r for r in selected if r["condition"] == c])
                    for c in sorted({r["condition"] for r in rows})}
    by_donor = {d: metrics([r for r in selected if r["donor"] == d])
               for d in sorted({r["donor"] for r in rows})}
    represented = [v for v in by_condition.values() if v["top1"] is not None]
    macro = {m: float(np.mean([v[m] for v in represented])) if represented else None
             for m in ["top1", "rank", "mrr", "chance_top1", "chance_rank", "chance_mrr"]}
    return {"defined_queries": len(selected), "total_queries": len(rows),
            "defined_cytokines": len(represented), "macro_by_cytokine": macro,
            "by_cytokine": by_condition, "by_donor": by_donor}


def training_references(payload, specification):
    labels = payload["cell_ids"].astype(str)
    if len(set(labels)) != len(labels):
        raise ValueError("Real-cell identifiers must be unique")
    index = {value: i for i, value in enumerate(labels)}
    train = specification["roles"]["train"]
    if set(train) & (set(specification["roles"]["test"]) | set(specification["roles"]["calib"])):
        raise ValueError("Training reference roles overlap held-out cytokines")
    populations = {(r["condition"], r["donor"]): r for r in specification["cell_populations"]}
    expression = payload["expression"]
    references, counts = {}, {}
    for donor in specification["donors"]:
        means, counts[donor] = [], {}
        for condition in train:
            ids = populations[(condition, donor)]["target_cell_ids"]
            rows = np.array([index[str(cell)] for cell in ids])
            if (len(rows) != len(set(rows)) or np.any(payload["is_control"][rows])
                    or np.any(payload["donor"][rows].astype(str) != donor)
                    or np.any(payload["cytokine"][rows].astype(str) != condition)):
                raise ValueError("Training-reference cell provenance disagrees with frozen roles")
            means.append(expression[rows].mean(axis=0, dtype=np.float64))
            counts[donor][condition] = len(rows)
        references[donor] = np.mean(means, axis=0)
    return references, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--cells-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prediction", nargs=3, action="append", metavar=("NAME", "NPZ", "QUERIES"),
                        help="Optional additional frozen procedure, including later mispaired control")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use a new or empty audit output directory")
    spec_path = args.formal_dir / "specification.json"
    spec = json.loads(spec_path.read_text())
    conditions, donors = spec["roles"]["test"], spec["donors"]
    if len(conditions) != 8 or len(set(conditions)) != 8:
        raise ValueError("This frozen audit requires all eight formal test cytokines")
    cells_hash = sha256(args.cells_npz)
    if cells_hash != spec["input_hashes"]["cells_npz"]:
        raise ValueError("Real-cell artifact differs from the original formal input")
    with np.load(args.cells_npz, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in ["expression", "gene_ids", "cell_ids", "cytokine", "donor", "is_control"]}
    reference, counts = training_references(payload, spec)
    entries = [(name, args.formal_dir / (name + "_test_predictions.npz"),
                args.formal_dir / (name + "_test_query_indices.json"))
               for name in ["released_checkpoint", "native_energy_sft"]]
    entries += [(name, Path(p), Path(q)) for name, p, q in args.prediction or []]
    if len({name for name, _, _ in entries}) != len(entries):
        raise ValueError("Duplicate procedure names")
    frozen_rule = {"status": "post-hoc secondary evaluation; no model fitting or setting selection",
        "candidate_pool": "All eight test cytokines within the same donor; exactly one candidate per cytokine",
        "profiles": "Population means over the fixed released 2000-feature axis, with no gene selection",
        "reference": "Equal mean of the 16 training-cytokine means within donor; actual frozen training cells only",
        "ranking": "Pearson over all features; absolute tie tolerance 1e-10; uniform tie credit and reciprocal occupied-rank average",
        "undefined": "Any constant prediction or candidate makes full query undefined; candidate pool never shrinks",
        "aggregation": "Donor average within cytokine, then cytokine macro; per-donor and per-cytokine metrics retained",
        "comparisons": "Raw-profile and training-reference-centered results; compare procedures on shared-defined queries",
        "chance": "Top1=1/8; MRR=H8/8, separately stored for each query"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specification = {"rule": frozen_rule, "conditions": conditions, "donors": donors,
        "reference_cell_counts": counts, "inputs": {"formal_specification": sha256(spec_path), "real_cells": cells_hash},
        "code_sha256": sha256(__file__)}
    (args.out_dir / "specification.json").write_text(json.dumps(specification, indent=2, allow_nan=False) + "\n")
    results, canonical_targets, canonical_queries, hashes = {}, None, None, {}
    for name, path, query_path in entries:
        queries = json.loads(query_path.read_text())
        with np.load(path, allow_pickle=False) as archive:
            predicted, target, features = archive["predictions"], archive["targets"], archive["feature_ids"]
        if (predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[2] != 2000
                or not np.array_equal(features.astype(str), payload["gene_ids"].astype(str))
                or not np.isfinite(predicted).all() or not np.isfinite(target).all()):
            raise ValueError("Invalid frozen prediction/population or feature-axis contract")
        if len(queries) != len(predicted):
            raise ValueError("Query identifiers disagree with frozen predictions")
        for row, query in enumerate(queries):
            target_rows = np.asarray(query["target_rows"], dtype=int)
            if (not np.array_equal(target[row], payload["expression"][target_rows])
                    or np.any(payload["donor"][target_rows].astype(str) != query["donor"])
                    or np.any(payload["cytokine"][target_rows].astype(str) != query["condition"])):
                raise ValueError("Frozen evaluation target provenance mismatch")
        if canonical_targets is not None and (queries != canonical_queries or not np.array_equal(target, canonical_targets)):
            raise ValueError("Procedures must use identical frozen evaluation populations/order")
        canonical_targets, canonical_queries = target, queries
        predicted_mean, target_mean = predicted.mean(1, dtype=np.float64), target.mean(1, dtype=np.float64)
        results[name] = {"raw_profiles": evaluate_profiles(predicted_mean, target_mean, queries, conditions, donors),
                         "training_reference_centered": evaluate_profiles(predicted_mean, target_mean, queries, conditions, donors, reference)}
        hashes[name] = {"predictions": sha256(path), "query_indices": sha256(query_path)}
    report = {"rule": frozen_rule, "feature_count": 2000, "queries": len(canonical_queries),
              "input_hashes": hashes, "models": {}, "comparisons": {},
              "limitations": ["Secondary rule fixed after native-energy results were examined; no independent confirmatory status.",
                  "Training reference is estimated and shared by prediction and all same-donor candidates; its sampling error can induce common covariance.",
                  "Donors and PBS inputs are shared across cytokine roles. Within-donor candidates remove donor identity as a retrieval shortcut but do not establish new-donor performance.",
                  "Raw-profile retrieval retains basal/common expression. Centered retrieval removes one training-label mean, not every common or technical component.",
                  "Finite cell means, one fit seed and eight cytokines; this audit gives no differential-expression, pathway, functional, animal or prospective validation."]}
    for mode in ["raw_profiles", "training_reference_centered"]:
        shared = set.intersection(*[{(r["condition"], r["donor"]) for r in value[mode] if r["defined"]} for value in results.values()])
        report["comparisons"][mode] = {"common_defined_queries": len(shared)}
        for name, value in results.items():
            report["models"].setdefault(name, {})[mode] = {"all_defined": summarize(value[mode]),
                "common_defined": summarize(value[mode], shared), "queries": value[mode]}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {mode: block["common_defined"] for mode, block in model.items()} for name, model in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
