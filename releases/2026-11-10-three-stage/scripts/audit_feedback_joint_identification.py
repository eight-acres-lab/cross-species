#!/usr/bin/env python3
"""Secondary identification scoring of frozen joint OOF predictions; no model refit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_feedback_nested_benchmark import file_hash, save_json
from run_feedback_specificity_benchmark import TrainingMeans, summarize_identification, tied_ranking

BASELINE = "chemical_context_plus_common"
AUGMENTED = "chemical_context_plus_common_residual"
LEARNED = (BASELINE, AUGMENTED)


def training_context_means(truth, strata, folds):
    """Reproduce the float32 context-mean scoring convention."""
    truth, strata, folds = np.asarray(truth, dtype=float), np.asarray(strata), np.asarray(folds)
    mean = np.empty_like(truth, dtype=np.float32)
    fallbacks = 0
    for fold in sorted(set(folds.tolist())):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit = TrainingMeans(truth[train], strata[train])
        values, fallback = fit.transform(strata[test])
        mean[test] = values
        fallbacks += int(fallback.sum())
    return mean.astype(float), fallbacks


def score_identification(predictions, truth, human_context, groups, strata, folds, learned_models=LEARNED):
    """Original Pearson/tie/undefined rules, with an explicit model intersection."""
    truth, human_context = np.asarray(truth, dtype=float), np.asarray(human_context, dtype=float)
    groups, strata, folds = np.asarray(groups).astype(str), np.asarray(strata).astype(str), np.asarray(folds)
    if truth.ndim != 2 or truth.shape != human_context.shape or not all(np.isfinite(v).all() for v in [truth, human_context]):
        raise ValueError("Finite equal-shaped truth and training context means required")
    if not all(len(v) == len(truth) for v in [groups, strata, folds]):
        raise ValueError("One compound/stratum/fold per condition required")
    if len(set(zip(groups.tolist(), strata.tolist()))) != len(truth):
        raise ValueError("Duplicate compound-stratum candidate")
    if not learned_models or not all(name in predictions for name in learned_models):
        raise ValueError("Learned model names must identify supplied frozen predictions")
    for value in predictions.values():
        if np.asarray(value).shape != truth.shape or not np.isfinite(value).all():
            raise ValueError("Finite frozen prediction on the full target axis required")
    rows = {name: [None] * len(truth) for name in predictions}
    for fold, stratum in sorted(set(zip(folds.tolist(), strata.tolist()))):
        pool = np.flatnonzero((folds == fold) & (strata == stratum))
        k = len(pool)
        chance = {"top1": 1./k, "rank": (k+1)/2., "mrr": float((1./np.arange(1, k+1)).mean())}
        candidate = truth[pool] - human_context[pool]
        candidate -= candidate.mean(axis=1, keepdims=True)
        cnorm = np.sqrt(np.square(candidate).sum(axis=1))
        invalid_candidates = not np.isfinite(cnorm).all() or np.any(cnorm <= 1e-12)
        for name, prediction in predictions.items():
            query = np.asarray(prediction[pool], dtype=float) - human_context[pool]
            query -= query.mean(axis=1, keepdims=True)
            qnorm = np.sqrt(np.square(query).sum(axis=1))
            scores = None
            if k >= 2 and not invalid_candidates:
                denom = qnorm[:, None] * cnorm[None, :]
                scores = np.divide(query @ candidate.T, denom, out=np.zeros((k, k)), where=denom > 0)
                scores = np.clip(scores, -1., 1.)
            for local, index in enumerate(pool):
                row = {"row": int(index), "compound": groups[index], "fold": int(fold), "stratum": stratum,
                       "candidates": k, "chance": chance, "defined": False, "reason": None}
                if k < 2:
                    row["reason"] = "fewer_than_two_candidates"
                elif invalid_candidates:
                    row["reason"] = "undefined_candidate_pearson"
                elif not np.isfinite(query[local]).all() or qnorm[local] <= 1e-12:
                    row["reason"] = "undefined_query_pearson"
                else:
                    row.update(tied_ranking(scores[local], local))
                    row["defined"] = True
                rows[name][index] = row
    common = np.asarray([all(rows[name][i]["defined"] for name in learned_models) for i in range(len(truth))])
    summary = {}
    for name, records in rows.items():
        natural = np.asarray([row["defined"] for row in records])
        summary[name] = {"all_defined_queries": summarize_identification(records, natural),
                         "common_defined_queries": summarize_identification(records, common & natural),
                         "undefined_reasons": {reason: sum(row["reason"] == reason for row in records)
                                               for reason in sorted({row["reason"] for row in records if row["reason"]})}}
    return summary, rows, common


def run_audit(run_dir):
    run_dir = Path(run_dir)
    names = ["identification.json", "identification_rows.json", "identification_context_means.npz",
             "identification_evaluation_specification.json", "identification_output_hashes.json"]
    if any((run_dir/name).exists() for name in names):
        raise ValueError("Secondary identification artifacts already exist; refusing overwrite")
    original_hashes = json.loads((run_dir/"output_hashes.json").read_text())
    for name in ["oof_predictions.npz", "report.json", "specification.json"]:
        if original_hashes.get(name) != file_hash(run_dir/name):
            raise ValueError("Frozen joint artifact hash mismatch: " + name)
    model_report = json.loads((run_dir/"report.json").read_text())
    rule_path = ROOT/"outputs/feedback_specificity_20261107/specification.json"
    rules = json.loads(rule_path.read_text())["identification_rules"]
    with np.load(run_dir/"oof_predictions.npz", allow_pickle=False) as payload:
        truth = payload["truth"].astype(float)
        predictions = {name: payload[name].astype(float) for name in LEARNED}
        sample_ids, genes = payload["sample_ids"].copy(), payload["human_gene_ids"].copy()
        groups, strata, folds = [payload[name].copy() for name in ["compounds", "strata", "outer_fold"]]
    specification = {
        "scope": "Post-hoc identification analysis of fixed joint out-of-fold predictions using the specified retrieval rules and complete gene axis.",
        "source_rule_file": str(rule_path.relative_to(ROOT)), "source_rule_sha256": file_hash(rule_path),
        "inherited_rules": rules,
        "model_intersection": "Common-defined queries for the two frozen joint procedures; report each model's full coverage separately.",
        "model_context_inputs": model_report["context_inputs"],
        "candidate_exposure_scope": "Candidates share outer fold and nominal time/dose level. Their planned absolute rat and human concentrations can differ; this is not a candidate pool matched for full exposure context. The training human context mean uses these same nominal strata.",
        "human_context_mean": "For each outer test fold, the human context mean is calculated by time and nominal dose using row weights from all training folds, with the global training mean as fallback. Values are stored in float32 and scored in float64.",
        "gene_axis": "Every supplied human target gene, unchanged; no test-based selection or weighting.",
        "models": list(LEARNED), "references": ["zero_response", "training_human_context_mean"],
        "rows": len(truth), "compounds": len(set(groups.tolist())), "target_genes": truth.shape[1],
        "input_hashes": {name: file_hash(run_dir/name) for name in ["oof_predictions.npz", "report.json", "specification.json"]},
        "code_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in [Path(__file__), ROOT/"scripts/run_feedback_specificity_benchmark.py"]},
        "limitations": ["Known candidate human profiles enter terminal scoring; this is not identification without human candidate measurements.",
                        "Chance is calculated from each query's K before compound averaging; per-query candidates are dependent.",
                        "These descriptive retrieval metrics do not identify unique biological mechanisms or establish prospective validation."],
    }
    save_json(run_dir/"identification_evaluation_specification.json", specification)
    human_context, fallback = training_context_means(truth, strata, folds)
    predictions["zero_response"] = np.zeros_like(truth)
    predictions["training_human_context_mean"] = human_context
    summary, rows, common = score_identification(predictions, truth, human_context, groups, strata, folds)
    report = {**specification, "human_context_fallback_conditions": fallback,
              "common_defined_queries": int(common.sum()), "identification": summary}
    save_json(run_dir/"identification.json", report)
    save_json(run_dir/"identification_rows.json", rows)
    np.savez_compressed(run_dir/"identification_context_means.npz", sample_ids=sample_ids, human_gene_ids=genes,
        outer_fold=folds, fold_training_human_context_mean=human_context.astype(np.float32), common_defined_queries=common)
    save_json(run_dir/"identification_output_hashes.json", {name: file_hash(run_dir/name) for name in names if name != "identification_output_hashes.json"})
    # Confirm that every original model/spec/result artifact remains unchanged.
    for name, digest in original_hashes.items():
        if file_hash(run_dir/name) != digest:
            raise ValueError("Original joint artifact changed during audit: " + name)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/feedback_joint_specificity_20261107"))
    args = parser.parse_args()
    report = run_audit(args.run_dir)
    for name in LEARNED:
        result = report["identification"][name]["common_defined_queries"]
        print(name, "top1", result["top1"], "chance", result["chance_top1"], "MRR", result["mrr"])


if __name__ == "__main__":
    main()
