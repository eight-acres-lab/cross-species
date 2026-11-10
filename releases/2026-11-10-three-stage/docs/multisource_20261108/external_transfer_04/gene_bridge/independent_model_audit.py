"""Independent legacy-CPU reconstruction of frozen external-study predictions.

No functions from the new transfer runner or Torch pairing implementation are
imported. The legacy kernel/Ridge implementation serves as the numerical model
reference; scoring and fold aggregation are written independently below.
"""
from __future__ import annotations

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "8"

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
NAMES = {"chemical_context": "chemical_context", "source_plus_context": "source_plus_chemical_context"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def document(path):
    return json.loads(Path(path).read_text())


class Audit:
    def __init__(self):
        self.checks = 0
        self.values = 0
        self.maximum = 0.

    def require(self, predicate, name):
        if not predicate:
            raise AssertionError(name)
        self.checks += 1

    def exact(self, x, y, name):
        self.require(np.array_equal(x, y), name)

    def close(self, x, y, name, tolerance=1e-8):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.require(x.shape == y.shape, name + " shape")
        self.require(np.array_equal(np.isfinite(x), np.isfinite(y)), name + " finite status")
        finite = np.isfinite(x)
        maximum = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.
        self.maximum = max(self.maximum, maximum)
        self.values += x.size
        self.require(maximum <= tolerance, name + " numeric mismatch " + str(maximum))


def macro_losses(row_losses, labels):
    groups = sorted(set(labels.tolist()))
    values = np.asarray([np.mean(row_losses[labels == group]) for group in groups])
    return groups, values, float(values.mean())


def select(losses, labels):
    totals = [macro_losses(row, labels)[2] for row in losses]
    return ALPHAS[int(np.argmin(totals))], np.asarray(totals)


def observed_metrics(predicted, target):
    residual = np.asarray(predicted, float) - np.asarray(target, float)
    output = {"mse": float(np.dot(residual, residual) / residual.size),
              "predicted_rms": math.sqrt(float(np.dot(predicted, predicted) / predicted.size)),
              "human_rms": math.sqrt(float(np.dot(target, target) / target.size))}
    p, y = predicted - np.sum(predicted) / len(predicted), target - np.sum(target) / len(target)
    denominator = math.sqrt(float(np.dot(p, p) * np.dot(y, y)))
    output["pearson"] = float(np.dot(p, y) / denominator) if denominator > 0 else None
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/external_transfer_04"))
    parser.add_argument("--legacy-dir", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/crossspecies_direct_03/code/scripts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    a = Audit()
    sys.path.insert(0, str(args.legacy_dir))
    import run_feedback_nested_benchmark as legacy
    root, run = args.root, args.root / "run_v1"
    inputs = root / "inputs_v1"
    fit, report, selection = document(run / "fit_manifest.json"), document(run / "report.json"), document(run / "selection.json")
    input_manifest = document(inputs / "manifest.json")
    a.require(fit["stage"] == "external_predictions_frozen_without_external_human_endpoint_access", "fit stage")
    a.require(fit["external_human_endpoint_read"] is False and report["external_human_labels_used_in_fit"] is False, "declared external-label isolation")
    a.require(sha(legacy.__file__) == fit["code_sha256"]["run_feedback_nested_benchmark.py"], "legacy model source version")
    for filename, expected in fit["output_files"].items():
        a.require(sha(run / filename) == expected, "frozen fit hash " + filename)
    for filename, expected in input_manifest["files"].items():
        a.require(sha(inputs / filename) == expected, "frozen input hash " + filename)
    a.require(sha(inputs / "manifest.json") == fit["input_manifest_sha256"] == report["input_manifest_sha256"], "input manifest chain")
    a.require(sha(run / "fit_manifest.json") == report["fit_manifest_sha256"], "report fit linkage")
    a.require(sha(run / "PROTOCOL.md") == fit["protocol_sha256"] == report["protocol_sha256"], "protocol chain")
    a.require(sha(inputs / "external_truth.npz") == report["external_endpoint_sha256"], "terminal endpoint chain")
    train, query = load(inputs / "train.npz"), load(inputs / "external_inputs.npz")
    for data, names in [(train, ("source", "target", "context", "fingerprints", "descriptors")),
                        (query, ("source", "context", "fingerprints", "descriptors"))]:
        for name in names:
            data[name] = data[name].astype(np.float64)
    oof, final, stored_predictions = load(run / "oof_predictions.npz"), load(run / "final_fit.npz"), load(run / "external_predictions.npz")
    truth = load(inputs / "external_truth.npz")
    a.require("target" not in query, "query object excludes external human response")
    a.exact(oof["truth"], train["target"], "OOF truth unchanged")
    a.exact(truth["task_ids"], query["task_ids"], "external truth task order")
    a.exact(stored_predictions["task_ids"], query["task_ids"], "external prediction task order")
    a.exact(final["human_gene_ids"], train["human_gene_ids"], "model target gene order")
    a.exact(final["rat_gene_ids"], train["rat_gene_ids"], "model source gene order")
    n = len(train["sample_ids"])

    def kernel_for(indices):
        return legacy.ChemicalKernelInputs(train["source"][indices], train["context"][indices],
                                          train["fingerprints"][indices], train["descriptors"][indices])

    def kernel_query(model, data, indices=None, source=None):
        indices = np.arange(len(data["source"])) if indices is None else indices
        return model.cross(data["source"][indices] if source is None else source,
                           data["context"][indices], fingerprints=data["fingerprints"][indices],
                           descriptors=data["descriptors"][indices])

    cv = {name: np.full((len(ALPHAS), n), np.nan) for name in NAMES}
    outer_zero_selection = None
    for fold in range(5):
        ti, vi = np.flatnonzero(train["outer_fold"] != fold), np.flatnonzero(train["outer_fold"] == fold)
        recorded = selection["outer_folds"][fold]
        a.exact(recorded["fit_indices"], ti, "outer fit rows")
        a.exact(recorded["query_indices"], vi, "outer held rows")
        a.require(not set(train["compounds"][ti]) & set(train["compounds"][vi]), "whole-compound holdout")
        if fold == 0:
            inside = train["inner_fold_by_outer"][fold]
            inner_losses = {name: np.full((len(ALPHAS), n), np.nan) for name in NAMES}
            for inner in range(3):
                inner_fit, inner_valid = ti[inside[ti] != inner], ti[inside[ti] == inner]
                declared = recorded["inner_partitions"][inner]
                a.exact(declared["fit_indices"], inner_fit, "inner original fit partition")
                a.exact(declared["validation_indices"], inner_valid, "inner original validation partition")
                a.require(not set(train["compounds"][inner_fit]) & set(train["compounds"][inner_valid]), "inner whole compounds")
                transform = kernel_for(inner_fit)
                kt, kv = transform.train_kernels(), kernel_query(transform, train, inner_valid)
                for name, original_name in NAMES.items():
                    ridge = legacy.SpectralRidge(kt[original_name], train["target"][inner_fit])
                    for ai, alpha in enumerate(ALPHAS):
                        predicted = ridge.predict(kv[original_name], alpha)
                        inner_losses[name][ai, inner_valid] = np.mean((predicted - train["target"][inner_valid]) ** 2, axis=1)
            outer_zero_selection = {}
            for name, losses in inner_losses.items():
                alpha, curve = select(losses[:, ti], train["compounds"][ti])
                a.close(curve, [row["compound_macro_mse"] for row in recorded["selection"][name]["trials"]], "nested7alpha macro " + name)
                a.require(alpha == recorded["selection"][name]["alpha"], "nested selected alpha " + name)
                outer_zero_selection[name] = alpha
        transform = kernel_for(ti)
        kt, kv = transform.train_kernels(), kernel_query(transform, train, vi)
        for name, original_name in NAMES.items():
            ridge = legacy.SpectralRidge(kt[original_name], train["target"][ti])
            for ai, alpha in enumerate(ALPHAS):
                predicted = ridge.predict(kv[original_name], alpha)
                cv[name][ai, vi] = np.mean((predicted - train["target"][vi]) ** 2, axis=1)
            if fold == 0:
                predicted = ridge.predict(kv[original_name], outer_zero_selection[name])
                a.close(predicted, oof[name][vi], "nested outer0 predicted genes " + name)
        print(json.dumps({"independent_training_fold_checked": fold}), flush=True)
    final_alphas = {}
    for name, losses in cv.items():
        a.close(losses, oof[name + "__final_cv_row_losses"], "all5fold7alpha row losses " + name)
        alpha, curve = select(losses, train["compounds"])
        a.close(curve, [row["compound_macro_mse"] for row in selection["final"][name]["trials"]], "final selection curve " + name)
        a.require(alpha == selection["final"][name]["alpha"], "final selected alpha " + name)
        a.close(final[name + "__alpha"], alpha, "saved final alpha")
        final_alphas[name] = alpha
        row_loss = np.mean((oof[name] - train["target"]) ** 2, axis=1)
        labels, values, macro = macro_losses(row_loss, train["compounds"])
        recorded = fit["internal_portable_model"]["oof_metrics"][name]
        a.exact(recorded["group_ids"], labels, "OOF compound order")
        a.close(recorded["group_losses"], values, "allOOF compound scores")
        a.close(recorded["macro_loss"], macro, "allOOF macro score")
    baseline = fit["internal_portable_model"]["oof_metrics"]["chemical_context"]["macro_loss"]
    augmented = fit["internal_portable_model"]["oof_metrics"]["source_plus_context"]["macro_loss"]
    a.close(fit["internal_portable_model"]["relative_gain_from_rat"], 1 - augmented / baseline, "internal relative gain")
    a.require(report["internal_portable_model"] == fit["internal_portable_model"], "internal report copy")
    all_rows = np.arange(n)
    full = kernel_for(all_rows)
    kt, kq = full.train_kernels(), kernel_query(full, query)
    for name, values in full.saved_transforms().items():
        a.close(values, final["transform__" + name], "full fit transform " + name)
    a.close(full.source_train, final["training_source_standardized"], "full source zscore")
    a.close(full.context_train, final["training_context_standardized"], "full context zscore")
    human_mean = np.sum(train["target"], axis=0) / n
    numerator = math.fsum((train["source"] * train["target"]).ravel())
    denominator = math.fsum(np.square(train["source"]).ravel())
    beta = max(0., numerator / denominator) if denominator > 1e-12 else 0.
    a.close(final["scalar_nointercept_beta"], beta, "scalar beta independent fsum")
    a.close(final["human_mean"], human_mean, "human historical mean")
    predictions = {"zero": np.zeros_like(query["source"]), "human_mean": np.stack([human_mean] * 3),
                   "rat_copy": query["source"].copy(), "scalar_nointercept": query["source"] * beta}
    for name, original_name in NAMES.items():
        model = legacy.SpectralRidge(kt[original_name], train["target"])
        predictions[name] = model.predict(kq[original_name], final_alphas[name])
        a.close(model.coefficients(final_alphas[name]), final[name + "__dual_coef"], "final coefficients " + name)
        if name == "source_plus_context":
            for key, perm in [("fused_wrong_source_120", [1, 2, 0]), ("fused_wrong_source_201", [2, 0, 1])]:
                wrong = kernel_query(full, query, source=query["source"][perm])[original_name]
                predictions[key] = model.predict(wrong, final_alphas[name])
    computed_macro = {}
    for name, predicted in predictions.items():
        a.close(predicted, stored_predictions[name], "all external predictions " + name)
        scores = []
        for task_index, task in enumerate(query["task_ids"]):
            row = report["external_tasks"][task_index]
            a.require(row["task_id"] == task, "terminal task order")
            metrics = observed_metrics(predicted[task_index], truth["target"][task_index].astype(float))
            for key, value in metrics.items():
                expected = row["rules"][name][key]
                if value is None:
                    a.require(expected is None, "undefined terminal Pearson")
                else:
                    a.close(value, expected, "terminal metric " + str(task) + name + key)
            scores.append(metrics["mse"])
        computed_macro[name] = float(np.mean(scores))
        a.close(computed_macro[name], report["macro_mse"][name], "terminal macro MSE " + name)
    benefits, harms = 0, 0
    for row in report["external_tasks"]:
        prior, fused = row["rules"]["chemical_context"]["mse"], row["rules"]["source_plus_context"]["mse"]
        a.close(row["source_relative_mse_reduction_vs_chemical_context"], 1 - fused / prior, "per-task relative gain")
        a.require(row["source_hurts_vs_chemical_context"] == (fused > prior), "per-task harm label")
        benefits += fused < prior
        harms += fused > prior
    a.require(report["rat_information_benefit_tasks"] == benefits and report["rat_information_hurt_tasks"] == harms, "external benefit/harm counts")
    a.close(report["rat_information_relative_mse_reduction"], 1 - computed_macro["source_plus_context"] / computed_macro["chemical_context"], "external macro gain")
    # This data support check uses molecular features, never external human labels.
    fingerprints = train["fingerprints"]
    for i, task in enumerate(query["task_ids"]):
        intersection = fingerprints @ query["fingerprints"][i]
        union = fingerprints.sum(axis=1) + query["fingerprints"][i].sum() - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        nearest = int(np.argmax(similarity))
        support = report["external_support"][i]
        a.require(support["task_id"] == task and support["nearest_training_compound_tanimoto"] == train["compounds"][nearest], "chemical support nearest")
        a.close(support["maximum_training_tanimoto"], similarity[nearest], "chemical support similarity")
        a.close(support["training_context_min"], train["context"].min(axis=0), "training context min")
        a.close(support["training_context_max"], train["context"].max(axis=0), "training context max")
        a.exact(support["outside_training_context_range"], ((query["context"][i] < train["context"].min(axis=0)) | (query["context"][i] > train["context"].max(axis=0))), "support range labels")
    result = {"status": "pass", "checks_passed": a.checks, "checks_failed": 0,
              "numeric_values_compared": a.values, "max_abs_difference": a.maximum,
              "training_rows": n, "genes": len(train["human_gene_ids"]), "external_tasks": query["task_ids"].tolist(),
              "external_rules_checked": list(predictions), "final_alphas_independently_reproduced": final_alphas,
              "cpu_refit_scope": "Both kernel arms: all5 training-cohort folds/all7alpha row-loss curves; all3inner folds of outer0 and its selected nested OOF predictions; both full-training final fits; all8 external rules.",
              "scoring_scope": "All saved OOF condition/compound/macro scores and all three external task MSE/Pearson/RMS/relative-gain values independently recomputed.",
              "production_transfer_or_torch_functions_imported": False,
              "legacy_source_sha256": sha(legacy.__file__), "input_manifest_sha256": sha(inputs / "manifest.json"),
              "fit_manifest_sha256": sha(run / "fit_manifest.json"), "report_sha256": sha(run / "report.json"),
              "auditor_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - start,
              "limitations": ["This verifies numeric reproducibility and declared task linkage, not external-blindness or generalization beyond these three previously inspected drugs.",
                              "Independent CPU reference shares the legacy kernel definition used by production; no new-runner or Torch fit functions are imported.",
                              "Human-truth file-access isolation is tested separately; this audit reads the terminal truth only to verify frozen predictions."]}
    output = args.output or root / "verification/independent_model_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
