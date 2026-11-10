#!/usr/bin/env python3
"""Fit and score frozen, partition-local cross-species pairing controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=8)
_known, _ = _pre.parse_known_args()
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(_known.threads)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species import source_pairing_controls as controls


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, payload):
    Path(path).write_text(json.dumps(plain(payload), indent=2, allow_nan=False) + "\n")


def save_fit(result, out, outer, bundle, seeds):
    prefix = "fold_%d" % outer
    save_json(out / (prefix + "_selection.json"), result["selection"])
    save_json(out / (prefix + "_audit.json"), result["audit"])
    state, predictions = result["states"], result["predictions"]
    test = result["test_indices"]
    # Deterministic arms and common transforms are stored once per outer fold.
    base = {"train_indices": result["train_indices"], "test_indices": test,
            "sample_ids": bundle["sample_ids"][test], "compounds": bundle["compounds"][test],
            "human_gene_ids": bundle["human_gene_ids"], "rat_gene_ids": bundle["rat_gene_ids"],
            "prior_prediction": predictions["prior"][0],
            "correct_prediction": predictions["correct_fit_correct_query"][0],
            "target_mean": state["target_mean"], "prior_coefficients": state["prior_coefficients"],
            "correct_coefficients": state["correct_coefficients"]}
    base.update({"transform__" + key: value for key, value in state["transforms"].items()})
    np.savez(out / (prefix + "_base.npz"), **base)
    for position, seed in enumerate(seeds):
        maps = result["audit"]["donor_maps"][position]
        np.savez(out / (prefix + "_seed_%d.npz" % seed),
                 **{name: predictions[name][position] for name in controls.ARMS[2:]},
                 wrong_coefficients=state["wrong_coefficients"][position],
                 wrong_alpha=np.asarray(result["selection"]["per_seed"][position]["wrong_alpha"]),
                 fit_donor_indices=np.asarray(maps["outer_fit"]["donor_indices"]),
                 query_donor_indices=np.asarray(maps["outer_query"]["donor_indices"]),
                 train_indices=result["train_indices"], test_indices=test)
    return {"outer": int(outer), "training_rows": len(result["train_indices"]),
            "query_rows": len(test), "query_compounds": len(set(bundle["compounds"][test].tolist())),
            "prior_alpha": result["selection"]["prior_alpha"],
            "correct_alpha": result["selection"]["correct_alpha"],
            "wrong_alphas": [row["wrong_alpha"] for row in result["selection"]["per_seed"]],
            "legacy_reference_checks": result["audit"]["legacy_reference_checks"]}


def score_saved(bundle, out, manifest):
    """The first access to outer held human values occurs after all fits are saved."""
    if (out / "report.json").exists() or (out / "row_and_compound_losses.npz").exists():
        raise ValueError("Existing scoring output must not be overwritten")
    n, seeds = len(bundle["sample_ids"]), manifest["permutation_seeds"]
    losses = {arm: np.full((len(seeds), n), np.nan) for arm in controls.ARMS}
    correlations = {kind: {arm: np.full((len(seeds), n), np.nan) for arm in controls.ARMS}
                    for kind in ("response_pearson", "train_mean_residual_pearson")}
    eligible = np.zeros(n, dtype=bool)
    for fold in manifest["outer_folds"]:
        prefix = "fold_%d" % fold
        with np.load(out / (prefix + "_base.npz"), allow_pickle=False) as base:
            test = base["test_indices"].copy()
            if eligible[test].any():
                raise ValueError("Repeated outer evaluation row")
            eligible[test] = True
            terminal_truth = bundle["target"][test]
            groups = bundle["compounds"][test]
            training_mean = base["target_mean"].copy()
            scored = controls.score_predictions({"prior": base["prior_prediction"][None],
                                                  "correct_fit_correct_query": base["correct_prediction"][None]},
                                                 terminal_truth, groups, training_mean)
            for arm in scored:
                losses[arm][:, test] = scored[arm]["row_losses"][0]
                for kind in correlations:
                    correlations[kind][arm][:, test] = scored[arm][kind][0]
        for position, seed in enumerate(seeds):
            with np.load(out / (prefix + "_seed_%d.npz" % seed), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["test_indices"], test)
                scored = controls.score_predictions({arm: saved[arm][None] for arm in controls.ARMS[2:]}, terminal_truth, groups, training_mean)
                for arm in scored:
                    losses[arm][position, test] = scored[arm]["row_losses"][0]
                    for kind in correlations:
                        correlations[kind][arm][position, test] = scored[arm][kind][0]
    rows = np.flatnonzero(eligible)
    compounds = np.asarray(bundle["compounds"])[rows]
    identifiers = sorted(set(compounds.tolist()))
    compound_losses, macros = {}, {}
    correlation_summary, group_correlations = {}, {}
    for arm in controls.ARMS:
        if not np.isfinite(losses[arm][:, rows]).all():
            raise ValueError("Incomplete terminal scoring")
        compound_losses[arm] = np.stack([losses[arm][:, rows][:, compounds == compound].mean(axis=1) for compound in identifiers], axis=1)
        macros[arm] = compound_losses[arm].mean(axis=1)
    for kind, arms in correlations.items():
        correlation_summary[kind], group_correlations[kind] = {}, {}
        for arm, values in arms.items():
            values = values[:, rows]
            by_group = []
            for compound in identifiers:
                selected = values[:, compounds == compound]
                counts = np.isfinite(selected).sum(axis=1)
                by_group.append(np.divide(np.nansum(selected, axis=1), counts,
                                          out=np.full(len(seeds), np.nan), where=counts > 0))
            group = np.stack(by_group, axis=1)
            counts = np.isfinite(group).sum(axis=1)
            macro = np.divide(np.nansum(group, axis=1), counts, out=np.full(len(seeds), np.nan), where=counts > 0)
            group_correlations[kind][arm] = group
            correlation_summary[kind][arm] = {"compound_macro_pearson_per_seed": macro,
                                              "defined_conditions_per_seed": np.isfinite(values).sum(axis=1),
                                              "defined_compounds_per_seed": counts,
                                              "total_conditions": len(rows), "total_compounds": len(identifiers)}
    comparisons = {}
    contrasts = [
        ("animal_increment", "prior", "correct_fit_correct_query"),
        ("query_pairing_given_correct_fit", "correct_fit_wrong_query", "correct_fit_correct_query"),
        ("fit_pairing_given_correct_query", "wrong_fit_correct_query", "correct_fit_correct_query"),
        ("full_pairing_vs_full_mismatch", "wrong_fit_wrong_query", "correct_fit_correct_query"),
        ("query_pairing_given_wrong_fit", "wrong_fit_wrong_query", "wrong_fit_correct_query"),
        ("wrong_fit_increment_over_prior", "prior", "wrong_fit_correct_query"),
    ]
    for name, reference, method in contrasts:
        delta = compound_losses[reference] - compound_losses[method]
        average = delta.mean(axis=1)
        comparisons[name] = {"reference": reference, "method": method,
                             "delta_reference_minus_method_per_seed": average,
                             "relative_mse_reduction_per_seed": average / macros[reference],
                             "fraction_compounds_hurt_per_seed": (delta < 0).mean(axis=1),
                             "compound_ids": identifiers, "compound_deltas_per_seed": delta,
                             "seed_variation_min_median_max": [float(np.min(average)), float(np.median(average)), float(np.max(average))]}
    np.savez(out / "row_and_compound_losses.npz", eligible_indices=rows,
             sample_ids=bundle["sample_ids"][rows], compounds=compounds,
             compound_ids=np.asarray(identifiers), permutation_seeds=np.asarray(seeds),
             **{arm + "__row_mse": losses[arm][:, rows] for arm in controls.ARMS},
             **{arm + "__compound_mse": compound_losses[arm] for arm in controls.ARMS},
             **{arm + "__" + kind: correlations[kind][arm][:, rows] for kind in correlations for arm in controls.ARMS},
             **{arm + "__compound_" + kind: group_correlations[kind][arm] for kind in correlations for arm in controls.ARMS})
    report = {"scope": "Retrospective direct rat-information pairing ablation; no independent external or prospective validation.",
              "regime": manifest["regime"], "fit_manifest_sha256": digest(out / "fit_manifest.json"),
              "input_bundle_sha256": manifest["input_bundle_sha256"], "protocol_sha256": manifest["protocol_sha256"],
              "original_rows": n, "rows": len(rows), "compounds": len(identifiers),
              "genes": len(bundle["human_gene_ids"]), "permutation_seeds": seeds,
              "arms": list(controls.ARMS), "macro_mse_per_seed": macros, "comparisons": comparisons,
              "correlation_summaries": correlation_summary,
              "correlation_definition": "Pearson across full genes per condition; residuals subtract the corresponding outer-fitting human gene mean from both prediction and truth. Finite conditions averaged within compound and finite compounds equally; constant-vector correlations remain undefined.",
              "eligibility": "Every arm shares each frozen partition's metadata-only eligible rows; original outer/inner fold memberships are retained.",
              "alpha_selection": "Prior and correct-fit alphas use shared correct-pair inner validation. Each wrong-fit seed independently selects alpha with separately deranged inner-fit and inner-validation source arrays. Wrong-fit/correct-query reuses that wrong-fit/wrong-query-selected alpha as a secondary diagnostic.",
              "mismatch_interpretation": "Current query chemical/context features remain their own. Source-exposure matching fixes time and rat concentration; paired-exposure also fixes the donor row's associated human concentration. Compound identity changes, so molecular/source compatibility is intentionally broken.",
              "metric": "Mean gene MSE per condition, mean conditions per compound, then equal mean over compounds. All arms use identical scored rows and genes.",
              "limitations": [
                  "Twenty nuisance permutations are not20 independent biological replications; size-two strata permit only one derangement.",
                  "Source rows excluded from a partition when its stratum lacks a full different-compound bijection; current results do not directly equal the prior678-row benchmark.",
                  "Original joint-RMA preprocessing, averaged replicate profiles, nominal exposure and previously examined public outcomes remain limitations.",
                  "One animal species and one broad in-vitro liver context; no species-selection or clinical claim.",
                  "Wrong-fit/correct-query is a distribution-shift diagnostic. No cross-regime winner or permutation seed is selected from terminal outcomes.",
                  "No p-values or population confidence intervals; seed ranges only describe the specified random mismatches.",
              ],
              "loss_array_sha256": digest(out / "row_and_compound_losses.npz")}
    save_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--regime", required=True, choices=("source_exposure", "paired_exposure"))
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=list(range(20261108, 20261128)))
    parser.add_argument("--threads", type=int, default=8)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fit-only", action="store_true")
    modes.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    torch.set_num_threads(args.threads)
    with np.load(args.bundle, allow_pickle=False) as payload:
        bundle = {key: payload[key].copy() for key in payload.files}
    metadata = json.loads(args.metadata.read_text())
    folds = controls.validate_bundle(bundle, metadata)
    input_hashes = {"input_bundle_sha256": digest(args.bundle), "metadata_sha256": digest(args.metadata), "protocol_sha256": digest(args.protocol)}
    out = args.out_dir
    if args.score_only:
        manifest = json.loads((out / "fit_manifest.json").read_text())
        for key, value in input_hashes.items():
            if manifest[key] != value:
                raise ValueError("Frozen scoring input changed: " + key)
        if manifest["regime"] != args.regime or manifest["permutation_seeds"] != args.permutation_seeds:
            raise ValueError("Scoring arguments differ from fitted run")
        for filename, expected in manifest["output_files"].items():
            if digest(out / filename) != expected:
                raise ValueError("Frozen fitted artifact changed: " + filename)
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Use a new empty output directory")
        out.mkdir(parents=True, exist_ok=True)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        snapshots = out / "code_snapshot"
        snapshots.mkdir()
        files = [Path(__file__), Path(controls.__file__), Path(controls.legacy.__file__)]
        for path in files:
            shutil.copy2(path, snapshots / path.name)
        shutil.copy2(args.protocol, out / "PROTOCOL.md")
        started = time.monotonic()
        summaries = []
        for outer in folds:
            fold_start = time.monotonic()
            result = controls.fit_outer(bundle, metadata, outer, args.regime, args.permutation_seeds, args.device)
            summaries.append(save_fit(result, out, outer, bundle, args.permutation_seeds))
            del result
            print(json.dumps({"outer": outer, "fit_saved": True, "seconds": time.monotonic() - fold_start,
                              "training_rows": summaries[-1]["training_rows"], "query_rows": summaries[-1]["query_rows"]}), flush=True)
        manifest = {**input_hashes, "status": "all_fits_and_predictions_frozen_before_terminal_scoring",
                    "bundle": str(args.bundle.resolve()), "metadata": str(args.metadata.resolve()),
                    "protocol": str(args.protocol.resolve()), "outer_folds": folds, "regime": args.regime,
                    "permutation_seeds": args.permutation_seeds, "alpha_grid": list(controls.DEFAULT_ALPHAS),
                    "fold_summaries": summaries,
                    "code_sha256": {path.name: digest(path) for path in files},
                    "runtime": {"device": args.device, "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                                "dtype": "float64", "torch": torch.__version__, "numpy": np.__version__,
                                "fit_and_save_seconds": time.monotonic() - started,
                                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                                "threads": args.threads},
                    "output_files": {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}}
        save_json(out / "fit_manifest.json", manifest)
    if not args.fit_only:
        report = score_saved(bundle, out, manifest)
        print(json.dumps({"rows": report["rows"], "compounds": report["compounds"],
                          "macro_mse_per_seed": plain(report["macro_mse_per_seed"])}, indent=2), flush=True)
    else:
        print(json.dumps({"status": "fit_only_complete", "out_dir": str(out), "human_outer_scoring_performed": False}), flush=True)


if __name__ == "__main__":
    main()
