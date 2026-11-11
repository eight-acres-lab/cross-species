#!/usr/bin/env python3
"""Independent CPU audit for the frozen rat-to-human pairing experiment.

This auditor imports NumPy ChemicalKernelInputs and SpectralRidge. The Torch
pairing-control module is excluded from the independent CPU calculation.
Large arrays and all calculations must be kept on the remote research host.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import numpy as np


ARMS = ("prior", "correct_fit_correct_query", "correct_fit_wrong_query",
        "wrong_fit_correct_query", "wrong_fit_wrong_query")
FACTORS = {"M": Decimal("1"), "mM": Decimal("0.001"),
           "uM": Decimal("0.000001"), "μM": Decimal("0.000001"),
           "µM": Decimal("0.000001"), "nM": Decimal("0.000000001")}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.numeric_values = 0
        self.max_abs_difference = 0.
        self.failed = []

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            self.failed.append(label)
            raise AssertionError(label)

    def close(self, left, right, label, tolerance=1e-8):
        a, b = np.asarray(left), np.asarray(right)
        self.require(a.shape == b.shape, label + ":shape")
        self.require(np.array_equal(np.isfinite(a), np.isfinite(b)), label + ":finite")
        finite = np.isfinite(a)
        difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        self.numeric_values += int(finite.sum())
        self.max_abs_difference = max(self.max_abs_difference, difference)
        self.require(difference <= tolerance, label + ":difference=" + str(difference))


def exposure_keys(metadata, regime):
    keys = []
    for row in metadata:
        parts = [Decimal(str(row["time_h"]))]
        species = ["rat", "human"] if regime == "paired_exposure" else ["rat"]
        for source in species:
            value = Decimal(str(row[source + "_dose_value"]))
            parts.append(value * FACTORS[row[source + "_dose_unit"].strip()])
        if not all(value.is_finite() for value in parts) or not all(value > 0 for value in parts[1:]):
            raise ValueError("Invalid declared exposure")
        keys.append(tuple(parts))
    return keys


def eligible_rows(indices, keys, compounds):
    buckets = {}
    for index in map(int, indices):
        buckets.setdefault(keys[index], []).append(index)
    admitted = set()
    for rows in buckets.values():
        counts = Counter(str(compounds[index]) for index in rows)
        if len(rows) >= 2 and 2 * max(counts.values()) <= len(rows):
            admitted.update(rows)
    return np.asarray([index for index in indices if int(index) in admitted], dtype=int)


def audit_map(record, expected_rows, keys, compounds, checker, label):
    query = np.asarray(record["query_indices"], dtype=int)
    donor = np.asarray(record["donor_indices"], dtype=int)
    local = np.asarray(record["local_permutation"], dtype=int)
    checker.require(np.array_equal(query, expected_rows), label + ":partition_rows")
    checker.require(len(set(query.tolist())) == len(query), label + ":unique_query")
    checker.require(set(query.tolist()) == set(donor.tolist()), label + ":closed_bijection")
    checker.require(np.array_equal(np.sort(local), np.arange(len(query))), label + ":local_bijection")
    checker.require(np.array_equal(query[local], donor), label + ":local_global_consistency")
    checker.require(all(keys[int(q)] == keys[int(d)] for q, d in zip(query, donor)), label + ":exact_Decimal_exposure")
    checker.require(all(compounds[q] != compounds[d] for q, d in zip(query, donor)), label + ":different_compound")
    return donor


def legacy_kernels(legacy, bundle, train, query, donor_train=None, donor_query=None):
    fit_source = train if donor_train is None else donor_train
    query_source = query if donor_query is None else donor_query
    inputs = legacy.ChemicalKernelInputs(bundle["source"][fit_source], bundle["context"][train],
        bundle["fingerprints"][train], bundle["descriptors"][train])
    kt = inputs.train_kernels()
    kq = inputs.cross(bundle["source"][query_source], bundle["context"][query],
        fingerprints=bundle["fingerprints"][query], descriptors=bundle["descriptors"][query])
    return kt, kq


def compound_macro(row_losses, groups):
    ids = sorted(set(map(str, groups)))
    losses = np.asarray(row_losses)
    per_group = np.stack([losses[..., np.asarray(groups).astype(str) == group].mean(axis=-1) for group in ids], axis=-1)
    return ids, per_group, per_group.mean(axis=-1)


def response_correlation(prediction, target):
    x, y = np.asarray(prediction), np.asarray(target)
    xc, yc = x - x.mean(axis=-1, keepdims=True), y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(xc * xc, axis=-1) * np.sum(yc * yc, axis=-1))
    good = (np.ptp(x, axis=-1) > 0) & (np.ptp(y, axis=-1) > 0) & (denominator > 0)
    out = np.full(denominator.shape, np.nan)
    np.divide(np.sum(xc * yc, axis=-1), denominator, out=out, where=good)
    return np.clip(out, -1, 1)


def finite_mean(values, axis=-1):
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count, out=np.full(count.shape, np.nan), where=count > 0)


def independent_alpha(legacy, bundle, inner_parts, alphas, wrong_maps=None, use_source=True):
    n = len(bundle["sample_ids"])
    losses = np.full((len(alphas), n), np.nan)
    name = "source_plus_chemical_context" if use_source else "chemical_context"
    for inner, train, valid in inner_parts:
        donors = wrong_maps[inner] if wrong_maps is not None else (None, None)
        kt, kv = legacy_kernels(legacy, bundle, train, valid, *donors)
        model = legacy.SpectralRidge(kt[name], bundle["target"][train])
        losses[:, valid] = model.row_losses(kv[name], bundle["target"][valid], alphas)
    selected_rows = np.flatnonzero(np.isfinite(losses).all(axis=0))
    _, _, macro = compound_macro(losses[:, selected_rows], bundle["compounds"][selected_rows])
    return float(alphas[int(np.argmin(macro))]), macro, selected_rows


def independent_predict(legacy, bundle, train, query, alpha, donor_train=None, donor_query=None, use_source=True):
    kt, kq = legacy_kernels(legacy, bundle, train, query, donor_train, donor_query)
    name = "source_plus_chemical_context" if use_source else "chemical_context"
    model = legacy.SpectralRidge(kt[name], bundle["target"][train])
    return model.predict(kq[name], alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    parser.add_argument("--legacy-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_script.parent))
    sys.path.insert(0, str(args.legacy_script.parent.parent / "src"))
    import run_feedback_nested_benchmark as legacy
    checker, started = Audit(), time.monotonic()
    checker.require(Path(legacy.__file__).resolve() == args.legacy_script.resolve(), "legacy_exact_import_path")
    bundle_manifest = json.loads((args.bundle_dir / "manifest.json").read_text())
    for name, expected in bundle_manifest["files"].items():
        checker.require(sha(args.bundle_dir / name) == expected, "bundle_hash:" + name)
    checker.require(sha(args.legacy_script) == bundle_manifest["legacy_kernel_script_sha256"], "legacy_frozen_code_hash")
    with np.load(args.bundle_dir / "bundle.npz", allow_pickle=False) as saved:
        bundle = {key: saved[key].copy() for key in saved.files}
    # The frozen input arrays retain float32 observations; the declared fit and
    # scoring arithmetic use float64. Cast observations without changing values.
    for key in ("source", "target", "context", "fingerprints", "descriptors"):
        bundle[key] = np.asarray(bundle[key], dtype=np.float64)
    metadata = json.loads((args.bundle_dir / "metadata.json").read_text())
    checker.require(bundle["source"].shape == bundle["target"].shape == (678, 10358), "strict_cohort_shape")
    checker.require(len(set(bundle["compounds"].tolist())) == 126, "strict_compound_count")
    checker.require(len(set(bundle["sample_ids"].tolist())) == 678, "unique_sample_ids")
    checker.require([row["sample_id"] for row in metadata] == bundle["sample_ids"].tolist(), "metadata_sample_order")
    original_outer = legacy.group_folds(bundle["compounds"], 5, 20261107)
    checker.require(np.array_equal(original_outer, bundle["outer_fold"]), "original_outer_folds")
    for outer in range(5):
        base_train = np.flatnonzero(original_outer != outer)
        inner = legacy.group_folds(bundle["compounds"][base_train], 3, 20261107 + 1009 * (outer + 1))
        checker.require(np.array_equal(bundle["inner_fold_by_outer"][outer, base_train], inner), "original_inner_folds:" + str(outer))
    result_records = []
    for folder in args.result_dir:
        fit = json.loads((folder / "fit_manifest.json").read_text())
        report = json.loads((folder / "report.json").read_text())
        regime = fit["regime"]
        keys = exposure_keys(metadata, regime)
        seeds, alphas = fit["permutation_seeds"], fit["alpha_grid"]
        checker.require(seeds == list(range(20261108, 20261128)), regime + ":frozen_seeds")
        checker.require(alphas == [.001, .01, .1, 1., 10., 100., 1000.], regime + ":alpha_grid")
        checker.require(fit["input_bundle_sha256"] == bundle_manifest["files"]["bundle.npz"], regime + ":bundle_hash")
        checker.require(fit["metadata_sha256"] == bundle_manifest["files"]["metadata.json"], regime + ":metadata_hash")
        checker.require(sha(folder / "PROTOCOL.md") == fit["protocol_sha256"], regime + ":protocol_hash")
        for name, expected in fit["output_files"].items():
            checker.require(sha(folder / name) == expected, regime + ":fit_output_hash:" + name)
        checker.require(sha(folder / "row_and_compound_losses.npz") == report["loss_array_sha256"], regime + ":loss_hash")
        expected_losses = {arm: np.full((len(seeds), 678), np.nan) for arm in ARMS}
        expected_correlations = {kind: {arm: np.full((len(seeds), 678), np.nan) for arm in ARMS}
                                 for kind in ("response_pearson", "train_mean_residual_pearson")}
        eligible = np.zeros(678, dtype=bool)
        map_count = 0
        cpu_cases = []
        for outer in fit["outer_folds"]:
            prefix, label = "fold_" + str(outer), regime + ":outer" + str(outer)
            audit = json.loads((folder / (prefix + "_audit.json")).read_text())
            selection = json.loads((folder / (prefix + "_selection.json")).read_text())
            outer_labels = bundle["outer_fold"]
            inner_labels = bundle["inner_fold_by_outer"][outer]
            train = eligible_rows(np.flatnonzero(outer_labels != outer), keys, bundle["compounds"])
            query = eligible_rows(np.flatnonzero(outer_labels == outer), keys, bundle["compounds"])
            checker.require(audit["outer_fit"]["kept_indices"] == train.tolist(), label + ":outer_fit_eligibility")
            checker.require(audit["outer_query"]["kept_indices"] == query.tolist(), label + ":outer_query_eligibility")
            checker.require(not eligible[query].any(), label + ":nonoverlapping_queries")
            eligible[query] = True
            inner_parts = []
            for inner_record in audit["inner"]:
                inner = inner_record["fold"]
                ti = eligible_rows(np.flatnonzero((outer_labels != outer) & (inner_labels != inner)), keys, bundle["compounds"])
                vi = eligible_rows(np.flatnonzero((outer_labels != outer) & (inner_labels == inner)), keys, bundle["compounds"])
                checker.require(ti.tolist() == inner_record["fit"]["kept_indices"], label + ":inner_fit_eligibility:" + str(inner))
                checker.require(vi.tolist() == inner_record["validation"]["kept_indices"], label + ":inner_valid_eligibility:" + str(inner))
                checker.require(not set(bundle["compounds"][ti]) & set(bundle["compounds"][vi]), label + ":inner_compound_isolation:" + str(inner))
                checker.require(not set(bundle["compounds"][ti]) & set(bundle["compounds"][query]), label + ":outer_compound_isolation:" + str(inner))
                inner_parts.append((inner, ti, vi))
            with np.load(folder / (prefix + "_base.npz"), allow_pickle=False) as saved:
                checker.require(np.array_equal(saved["train_indices"], train), label + ":saved_train_rows")
                checker.require(np.array_equal(saved["test_indices"], query), label + ":saved_test_rows")
                checker.require(np.array_equal(saved["sample_ids"], bundle["sample_ids"][query]), label + ":saved_sample_ids")
                checker.require(np.array_equal(saved["human_gene_ids"], bundle["human_gene_ids"]), label + ":human_gene_axis")
                checker.require(np.array_equal(saved["rat_gene_ids"], bundle["rat_gene_ids"]), label + ":animal_gene_axis")
                checker.close(saved["target_mean"], bundle["target"][train].mean(axis=0), label + ":same_human_training_labels")
                base_predictions = {"prior": saved["prior_prediction"].copy(),
                                    "correct_fit_correct_query": saved["correct_prediction"].copy()}
                target_mean = bundle["target"][train].mean(axis=0)
            for arm, values in base_predictions.items():
                expected_losses[arm][:, query] = np.mean(np.square(values - bundle["target"][query]), axis=1)
                expected_correlations["response_pearson"][arm][:, query] = response_correlation(values, bundle["target"][query])
                expected_correlations["train_mean_residual_pearson"][arm][:, query] = response_correlation(values - target_mean, bundle["target"][query] - target_mean)
            if outer == 0:
                for use_source, alpha_key, trial_key, arm in [
                    (False, "prior_alpha", "prior_inner_selection", "prior"),
                    (True, "correct_alpha", "correct_inner_selection", "correct_fit_correct_query")]:
                    alpha, macro, validation = independent_alpha(legacy, bundle, inner_parts, alphas, use_source=use_source)
                    checker.require(alpha == selection[alpha_key], label + ":CPU_alpha:" + arm)
                    checker.close(macro, [row["compound_macro_mse"] for row in selection[trial_key]["trials"]], label + ":CPU_inner_trials:" + arm)
                    checker.require(validation.tolist() == selection[trial_key]["validation_indices"], label + ":CPU_inner_cohort:" + arm)
                    cpu_prediction = independent_predict(legacy, bundle, train, query, alpha, use_source=use_source)
                    checker.close(cpu_prediction, base_predictions[arm], label + ":CPU_outer_prediction:" + arm)
            checker.require([row["seed"] for row in audit["donor_maps"]] == seeds, label + ":all_mapping_seeds")
            for position, seed_maps in enumerate(audit["donor_maps"]):
                seed = seed_maps["seed"]
                seed_label = label + ":seed" + str(seed)
                p = audit_map(seed_maps["outer_fit"], train, keys, bundle["compounds"], checker, seed_label + ":fit")
                q = audit_map(seed_maps["outer_query"], query, keys, bundle["compounds"], checker, seed_label + ":query")
                map_count += 2
                inner_by_fold = {row["fold"]: row for row in seed_maps["inner"]}
                wrong_maps = {}
                for inner, ti, vi in inner_parts:
                    record = inner_by_fold[inner]
                    wrong_maps[inner] = (audit_map(record["fit"], ti, keys, bundle["compounds"], checker, seed_label + ":innerfit" + str(inner)),
                                         audit_map(record["query"], vi, keys, bundle["compounds"], checker, seed_label + ":innervalid" + str(inner)))
                    map_count += 2
                with np.load(folder / (prefix + "_seed_" + str(seed) + ".npz"), allow_pickle=False) as saved:
                    checker.require(np.array_equal(saved["train_indices"], train) and np.array_equal(saved["test_indices"], query), seed_label + ":same_all_arm_rows")
                    checker.require(np.array_equal(saved["fit_donor_indices"], p) and np.array_equal(saved["query_donor_indices"], q), seed_label + ":saved_maps")
                    seed_selection = selection["per_seed"][position]
                    checker.require(seed_selection["seed"] == seed, seed_label + ":selection_seed")
                    checker.require(float(saved["wrong_alpha"]) == seed_selection["wrong_alpha"], seed_label + ":saved_alpha")
                    for arm in ARMS[2:]:
                        expected_losses[arm][position, query] = np.mean(np.square(saved[arm] - bundle["target"][query]), axis=1)
                        expected_correlations["response_pearson"][arm][position, query] = response_correlation(saved[arm], bundle["target"][query])
                        expected_correlations["train_mean_residual_pearson"][arm][position, query] = response_correlation(saved[arm] - target_mean, bundle["target"][query] - target_mean)
                    if outer == 0 and position in (0, len(seeds) - 1):
                        alpha, macro, validation = independent_alpha(legacy, bundle, inner_parts, alphas, wrong_maps=wrong_maps)
                        checker.require(alpha == seed_selection["wrong_alpha"], seed_label + ":CPU_wrong_alpha")
                        checker.close(macro, [row["compound_macro_mse"] for row in seed_selection["wrong_inner_selection"]["trials"]], seed_label + ":CPU_wrong_inner_trials")
                        checker.require(validation.tolist() == seed_selection["wrong_inner_selection"]["validation_indices"], seed_label + ":CPU_wrong_inner_cohort")
                        configurations = [("correct_fit_wrong_query", selection["correct_alpha"], None, q),
                                          ("wrong_fit_correct_query", alpha, p, None),
                                          ("wrong_fit_wrong_query", alpha, p, q)]
                        for arm, chosen, fit_donors, query_donors in configurations:
                            cpu_prediction = independent_predict(legacy, bundle, train, query, chosen, fit_donors, query_donors)
                            checker.close(cpu_prediction, saved[arm], seed_label + ":CPU_raw_permuted_prediction:" + arm)
                        cpu_cases.append({"outer": outer, "seed": seed, "wrong_alpha": alpha, "all_inner_alpha_trials_and_three_query_arms_verified": True})
        rows = np.flatnonzero(eligible)
        checker.require(len(rows) == report["rows"], regime + ":query_cohort_size")
        checker.require(report["arms"] == list(ARMS), regime + ":all_arms_reported")
        with np.load(folder / "row_and_compound_losses.npz", allow_pickle=False) as saved:
            checker.require(np.array_equal(saved["eligible_indices"], rows), regime + ":scored_row_order")
            checker.require(np.array_equal(saved["sample_ids"], bundle["sample_ids"][rows]), regime + ":scored_ids")
            ids, group_results, macro_results = None, {}, {}
            for arm in ARMS:
                checker.close(saved[arm + "__row_mse"], expected_losses[arm][:, rows], regime + ":all_condition_losses:" + arm)
                ids, grouped, macro = compound_macro(expected_losses[arm][:, rows], bundle["compounds"][rows])
                checker.require(saved["compound_ids"].tolist() == ids, regime + ":compound_axis:" + arm)
                checker.close(saved[arm + "__compound_mse"], grouped, regime + ":all_compound_losses:" + arm)
                checker.close(report["macro_mse_per_seed"][arm], macro, regime + ":all_seed_macro:" + arm)
                group_results[arm], macro_results[arm] = grouped, macro
                for kind in expected_correlations:
                    values = expected_correlations[kind][arm][:, rows]
                    checker.close(saved[arm + "__" + kind], values, regime + ":all_condition_correlations:" + kind + ":" + arm)
                    group = np.stack([finite_mean(values[:, bundle["compounds"][rows] == compound]) for compound in ids], axis=1)
                    checker.close(saved[arm + "__compound_" + kind], group, regime + ":all_compound_correlations:" + kind + ":" + arm)
                    summary = report["correlation_summaries"][kind][arm]
                    checker.close(np.asarray(summary["compound_macro_pearson_per_seed"], dtype=float), finite_mean(group), regime + ":macro_correlations:" + kind + ":" + arm)
                    checker.require(summary["defined_conditions_per_seed"] == np.isfinite(values).sum(axis=1).tolist(), regime + ":correlation_defined_conditions:" + kind + ":" + arm)
                    checker.require(summary["defined_compounds_per_seed"] == np.isfinite(group).sum(axis=1).tolist(), regime + ":correlation_defined_compounds:" + kind + ":" + arm)
            for name, record in report["comparisons"].items():
                reference, method = record["reference"], record["method"]
                delta = group_results[reference] - group_results[method]
                mean = delta.mean(axis=1)
                checker.require(record["compound_ids"] == ids, regime + ":comparison_axis:" + name)
                checker.close(record["compound_deltas_per_seed"], delta, regime + ":comparison_group_delta:" + name)
                checker.close(record["delta_reference_minus_method_per_seed"], mean, regime + ":comparison_macro_delta:" + name)
                checker.close(record["relative_mse_reduction_per_seed"], mean / macro_results[reference], regime + ":comparison_relative:" + name)
                checker.close(record["fraction_compounds_hurt_per_seed"], (delta < 0).mean(axis=1), regime + ":comparison_hurt:" + name)
                checker.close(record["seed_variation_min_median_max"], [mean.min(), np.median(mean), mean.max()], regime + ":comparison_seed_range:" + name)
        result_records.append({"regime": regime, "remote_result_directory": str(folder.resolve()),
                               "rows": len(rows), "compounds": len(ids), "all_partition_maps_checked": map_count,
                               "CPU_raw_feature_permutation_refits": cpu_cases,
                               "all_folds_all_seeds_all_condition_compound_losses_recomputed": True,
                               "fit_manifest_sha256": sha(folder / "fit_manifest.json"),
                               "report_sha256": sha(folder / "report.json")})
    output = {"status": "pass", "checks_passed": checker.checks, "checks_failed": len(checker.failed),
              "numeric_values_compared": checker.numeric_values, "max_abs_difference": checker.max_abs_difference,
              "legacy_script_sha256": sha(args.legacy_script), "auditor_sha256": sha(__file__),
              "bundle_manifest_sha256": sha(args.bundle_dir / "manifest.json"),
              "elapsed_seconds": time.monotonic() - started, "regimes": result_records,
              "method": "Independent NumPy ChemicalKernelInputs/SpectralRidge calculations use permuted raw animal input rows without importing the GPU pairing module.",
              "limitations": ["Numerical/control verification does not establish biological generalization.",
                              "CPU full refits sample outer0 and first/last permutation seed in each regime; all saved maps and all loss arrays are checked.",
                              "Existing retrospective, shared preprocessing and fixed-condition-mean limitations remain."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
