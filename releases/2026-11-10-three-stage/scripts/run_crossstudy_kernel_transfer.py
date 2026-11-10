#!/usr/bin/env python3
"""Train the fixed portable rat-to-human rule on TG-GATEs, then score another study.

The external human endpoint file is accessed only by score_saved, after all
training, tuning and external predictions have been saved and hashed.
"""
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
_args, _ = _pre.parse_known_args()
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = str(_args.threads)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_feedback_nested_benchmark as legacy
from cross_species.source_pairing_controls import TorchRidge

ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
KERNEL_NAMES = {"chemical_context": "chemical_context", "source_plus_context": "source_plus_chemical_context"}
PRIMARY_RULES = ("zero", "human_mean", "rat_copy", "scalar_nointercept", "chemical_context", "source_plus_context")
WRONG_RULES = ("fused_wrong_source_120", "fused_wrong_source_201")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def validate_inputs(train, external):
    n, genes = np.asarray(train["source"]).shape
    if n < 10 or genes < 1 or np.shape(train["target"]) != (n, genes):
        raise ValueError("Invalid TG-GATEs source/target axes")
    for data, rows, label in ((train, n, "training"), (external, 3, "external")):
        if np.shape(data["source"]) != (rows, genes) or np.shape(data["context"]) != (rows, 3):
            raise ValueError("Expected portable three-continuous-context inputs: " + label)
        if np.shape(data["fingerprints"]) != (rows, 881) or np.shape(data["descriptors"]) != (rows, 10):
            raise ValueError("Molecular feature schema mismatch: " + label)
        for name in ("source", "context", "fingerprints"):
            if not np.isfinite(data[name]).all():
                raise ValueError("Nonfinite " + label + " " + name)
        if not np.isin(data["fingerprints"], [0, 1]).all() or np.isinf(data["descriptors"]).any():
            raise ValueError("Invalid fingerprints or descriptors")
        for key in ("human_gene_ids", "rat_gene_ids"):
            if np.shape(data[key]) != (genes,) or len(set(data[key].tolist())) != genes:
                raise ValueError("Gene axes must be unique and complete")
            np.testing.assert_array_equal(data[key], train[key])
    if not np.isfinite(train["target"]).all():
        raise ValueError("Training human responses must be finite")
    if len(set(train["sample_ids"].tolist())) != n or np.shape(train["compounds"]) != (n,):
        raise ValueError("Training identity axes differ")
    if np.shape(external["task_ids"]) != (3,) or len(set(external["task_ids"].tolist())) != 3:
        raise ValueError("Exactly three declared external tasks required")
    outer = np.asarray(train["outer_fold"])
    inner = np.asarray(train["inner_fold_by_outer"])
    if outer.shape != (n,) or set(outer.tolist()) != set(range(5)) or inner.shape != (5, n):
        raise ValueError("Expected frozen five outer folds and per-outer inner membership")
    for fold in range(5):
        if np.any(inner[fold, outer == fold] != -1) or set(inner[fold, outer != fold].tolist()) != {0, 1, 2}:
            raise ValueError("Three inner folds must remain inside the outer training set")
    for compound in set(train["compounds"].tolist()):
        rows = train["compounds"] == compound
        if len(set(outer[rows].tolist())) != 1:
            raise ValueError("Compound crosses outer folds")
        for fold in range(5):
            if len(set(inner[fold, rows].tolist())) != 1:
                raise ValueError("Compound crosses inner folds")


def make_inputs(train, indices):
    return legacy.ChemicalKernelInputs(train["source"][indices], train["context"][indices],
                                      train["fingerprints"][indices], train["descriptors"][indices])


def cross(inputs, data, indices=None, source_override=None):
    if indices is None:
        indices = np.arange(len(data["source"]))
    source = data["source"][indices] if source_override is None else source_override
    return inputs.cross(source, data["context"][indices], fingerprints=data["fingerprints"][indices],
                        descriptors=data["descriptors"][indices])


def choose_alpha(losses, compounds):
    if losses.shape != (len(ALPHAS), len(compounds)) or not np.isfinite(losses).all():
        raise ValueError("Every declared validation task must have a finite loss for each alpha")
    groups = sorted(set(compounds.tolist()))
    macro = np.asarray([np.mean([row[compounds == group].mean() for group in groups]) for row in losses])
    index = int(np.argmin(macro))
    return float(ALPHAS[index]), {"trials": [{"alpha": alpha, "compound_macro_mse": float(value)}
                                           for alpha, value in zip(ALPHAS, macro)],
                                 "rows": len(compounds), "compounds": len(groups)}


def cpu_reference(train_kernel, query_kernel, y_train, y_valid, gpu_model, gpu_query, alpha):
    reference = legacy.SpectralRidge(train_kernel, y_train)
    pred = reference.predict(query_kernel, alpha)
    actual = gpu_model.predict(gpu_query, alpha).detach().cpu().numpy()
    prediction_gap = float(np.max(np.abs(pred - actual)))
    old_losses = reference.row_losses(query_kernel, y_valid, ALPHAS)
    new_losses = gpu_model.row_losses(gpu_query, y_valid, ALPHAS)
    loss_gap = float(np.max(np.abs(old_losses - new_losses)))
    if max(prediction_gap, loss_gap) > 1e-8:
        raise ValueError("Legacy CPU and portable GPU kernel prediction mismatch")
    return {"prediction_max_abs_difference": prediction_gap, "row_loss_max_abs_difference": loss_gap,
            "tolerance": 1e-8, "passed": True}


def fit_training(train, external, device="cpu"):
    """Fit all rules using training labels only; external human truth is absent."""
    validate_inputs(train, external)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Explicit double conversion avoids dtype-dependent means and norms.
    train = {name: np.asarray(value, dtype=np.float64) if name in
             {"source", "target", "context", "fingerprints", "descriptors"} else value
             for name, value in train.items()}
    external = {name: np.asarray(value, dtype=np.float64) if name in
                {"source", "context", "fingerprints", "descriptors"} else value
                for name, value in external.items()}
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
    n, genes = train["target"].shape
    oof = {name: np.empty((n, genes), dtype=np.float64) for name in KERNEL_NAMES}
    final_cv_losses = {name: np.empty((len(ALPHAS), n)) for name in KERNEL_NAMES}
    outer_records, validation = [], []
    for outer in range(5):
        fit = np.flatnonzero(train["outer_fold"] != outer)
        test = np.flatnonzero(train["outer_fold"] == outer)
        inside = train["inner_fold_by_outer"][outer]
        inner_losses = {name: np.full((len(ALPHAS), n), np.nan) for name in KERNEL_NAMES}
        inner_records = []
        for inner in range(3):
            ti = fit[inside[fit] != inner]
            vi = fit[inside[fit] == inner]
            inputs = make_inputs(train, ti)
            train_kernels, validation_kernels = inputs.train_kernels(), cross(inputs, train, vi)
            for name, kernel_name in KERNEL_NAMES.items():
                model = TorchRidge(tensor(train_kernels[kernel_name]), train["target"][ti])
                query = tensor(validation_kernels[kernel_name])
                inner_losses[name][:, vi] = model.row_losses(query, train["target"][vi], ALPHAS)
                if outer == 0 and inner == 0:
                    check = cpu_reference(train_kernels[kernel_name], validation_kernels[kernel_name],
                                          train["target"][ti], train["target"][vi], model, query, 1.)
                    validation.append({"outer": outer, "inner": inner, "arm": name, **check})
            inner_records.append({"inner": inner, "fit_indices": ti.tolist(), "validation_indices": vi.tolist()})
        selection = {name: choose_alpha(values[:, fit], train["compounds"][fit]) for name, values in inner_losses.items()}
        inputs = make_inputs(train, fit)
        train_kernels, query_kernels = inputs.train_kernels(), cross(inputs, train, test)
        for name, kernel_name in KERNEL_NAMES.items():
            model = TorchRidge(tensor(train_kernels[kernel_name]), train["target"][fit])
            query = tensor(query_kernels[kernel_name])
            oof[name][test] = model.predict(query, selection[name][0]).detach().cpu().numpy()
            # These are TG-GATEs training-cohort validation labels, never external-study labels.
            final_cv_losses[name][:, test] = model.row_losses(query, train["target"][test], ALPHAS)
        outer_records.append({"outer": outer, "fit_indices": fit.tolist(), "query_indices": test.tolist(),
                              "inner_partitions": inner_records,
                              "selection": {name: {"alpha": value[0], **value[1]} for name, value in selection.items()}})
    final_selection = {name: choose_alpha(values, train["compounds"]) for name, values in final_cv_losses.items()}
    all_rows = np.arange(n)
    inputs = make_inputs(train, all_rows)
    train_kernels, external_kernels = inputs.train_kernels(), cross(inputs, external)
    human_mean = train["target"].mean(axis=0)
    denominator = float(np.square(train["source"]).sum())
    numerator = float(np.sum(train["source"] * train["target"]))
    beta = max(0., numerator / denominator) if denominator > 1e-12 else 0.
    predictions = {"zero": np.zeros((3, genes)), "human_mean": np.repeat(human_mean[None], 3, axis=0),
                   "rat_copy": external["source"].copy(), "scalar_nointercept": beta * external["source"]}
    state = {"human_mean": human_mean, "scalar_nointercept_beta": np.asarray(beta),
             "scalar_fit_numerator": np.asarray(numerator), "scalar_fit_denominator": np.asarray(denominator),
             "training_source_standardized": inputs.source_train, "training_context_standardized": inputs.context_train,
             "human_gene_ids": train["human_gene_ids"], "rat_gene_ids": train["rat_gene_ids"]}
    state.update({"transform__" + key: value for key, value in inputs.saved_transforms().items()})
    for name, kernel_name in KERNEL_NAMES.items():
        alpha = final_selection[name][0]
        model = TorchRidge(tensor(train_kernels[kernel_name]), train["target"])
        predictions[name] = model.predict(tensor(external_kernels[kernel_name]), alpha).detach().cpu().numpy()
        state[name + "__dual_coef"] = model.coefficients(alpha).detach().cpu().numpy()
        state[name + "__alpha"] = np.asarray(alpha)
        reference = legacy.SpectralRidge(train_kernels[kernel_name], train["target"])
        cpu = reference.predict(external_kernels[kernel_name], alpha)
        gap = float(np.max(np.abs(cpu - predictions[name])))
        if gap > 1e-8:
            raise ValueError("Final external prediction differs from legacy CPU reference")
        validation.append({"stage": "final_external_prediction", "arm": name,
                           "max_abs_difference": gap, "tolerance": 1e-8, "passed": True})
        if name == "source_plus_context":
            for rule, donor in zip(WRONG_RULES, ([1, 2, 0], [2, 0, 1])):
                query = cross(inputs, external, source_override=external["source"][donor])[kernel_name]
                predictions[rule] = model.predict(tensor(query), alpha).detach().cpu().numpy()
                cpu_wrong = reference.predict(query, alpha)
                wrong_gap = float(np.max(np.abs(cpu_wrong - predictions[rule])))
                if wrong_gap > 1e-8:
                    raise ValueError("Wrong-source external prediction differs from legacy CPU")
                validation.append({"stage": "external_cross_condition_diagnostic", "arm": rule,
                                   "donor_indices": donor, "max_abs_difference": wrong_gap,
                                   "tolerance": 1e-8, "passed": True})
    # These simple-reference checks are independent elementwise formulas.
    scalar_gap = float(np.max(np.abs(predictions["scalar_nointercept"] - external["source"] * beta)))
    validation.append({"stage": "simple_rules", "zero_exact": not np.any(predictions["zero"]),
                       "copy_exact": bool(np.array_equal(predictions["rat_copy"], external["source"])),
                       "human_mean_exact": bool(np.array_equal(predictions["human_mean"][0], human_mean)),
                       "scalar_formula_max_abs_difference": scalar_gap})
    tanimoto = legacy.tanimoto(external["fingerprints"], train["fingerprints"])
    support = []
    for i, task in enumerate(external["task_ids"]):
        nearest = int(np.argmax(tanimoto[i]))
        support.append({"task_id": str(task), "context": external["context"][i].tolist(),
                        "training_context_min": train["context"].min(axis=0).tolist(),
                        "training_context_max": train["context"].max(axis=0).tolist(),
                        "outside_training_context_range": ((external["context"][i] < train["context"].min(axis=0)) |
                                                           (external["context"][i] > train["context"].max(axis=0))).tolist(),
                        "nearest_training_compound_tanimoto": str(train["compounds"][nearest]),
                        "maximum_training_tanimoto": float(tanimoto[i, nearest])})
    internal_metrics = {name: legacy.compound_macro_loss(values, train["target"], train["compounds"])
                        for name, values in oof.items()}
    baseline = internal_metrics["chemical_context"]["macro_loss"]
    internal_gain = 1 - internal_metrics["source_plus_context"]["macro_loss"] / baseline if baseline > 0 else None
    return {"oof_predictions": oof, "external_predictions": predictions, "final_fit": state,
            "selection": {"outer_folds": outer_records,
                          "final": {name: {"alpha": result[0], **result[1]} for name, result in final_selection.items()},
                          "final_rule": "Select each kernel arm alpha from pooled TG-GATEs five-fold compound-macro validation losses on the fixed seven-point grid; external human outcomes unused."},
            "final_cv_losses": final_cv_losses, "validation": validation, "external_support": support,
            "internal_oof_metrics": internal_metrics, "internal_relative_gain_from_rat": internal_gain}


def metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    p, y = prediction - prediction.mean(axis=1, keepdims=True), truth - truth.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(p * p, axis=1) * np.sum(y * y, axis=1))
    correlation = np.divide(np.sum(p * y, axis=1), denominator, out=np.full(len(truth), np.nan), where=denominator > 0)
    return {"mse": np.mean((prediction - truth) ** 2, axis=1), "pearson": correlation,
            "predicted_rms": np.sqrt(np.mean(prediction ** 2, axis=1)),
            "human_rms": np.sqrt(np.mean(truth ** 2, axis=1))}


def score_saved(out, input_dir, manifest):
    """Read the external human endpoint only after frozen fit outputs are verified."""
    if (out / "report.json").exists():
        raise ValueError("Scored report already exists; preserve frozen results")
    for filename, expected in manifest["output_files"].items():
        if sha(out / filename) != expected:
            raise ValueError("Frozen fit output changed: " + filename)
    inputs_manifest = json.loads((input_dir / "manifest.json").read_text())
    if sha(input_dir / "manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Input manifest changed")
    expected_truth = inputs_manifest["files"]["external_truth.npz"]
    if sha(input_dir / "external_truth.npz") != expected_truth:
        raise ValueError("External endpoint checksum differs")
    truth = read_npz(input_dir / "external_truth.npz")
    predictions = read_npz(out / "external_predictions.npz")
    np.testing.assert_array_equal(predictions["task_ids"], truth["task_ids"])
    target = np.asarray(truth["target"], dtype=float)
    if target.shape != predictions["zero"].shape or not np.isfinite(target).all():
        raise ValueError("External target shape or finite-value contract failed")
    scores = {name: metrics(predictions[name], target) for name in (*PRIMARY_RULES, *WRONG_RULES)}
    rows = []
    for i, task in enumerate(predictions["task_ids"]):
        row = {"task_id": str(task), "rules": {name: {key: float(value[i]) for key, value in score.items()}
                                                for name, score in scores.items()}}
        prior_mse = scores["chemical_context"]["mse"][i]
        source_mse = scores["source_plus_context"]["mse"][i]
        row["source_relative_mse_reduction_vs_chemical_context"] = 1 - source_mse / prior_mse if prior_mse > 0 else None
        row["source_hurts_vs_chemical_context"] = bool(source_mse > prior_mse)
        rows.append(row)
    macro = {name: float(score["mse"].mean()) for name, score in scores.items()}
    selection = json.loads((out / "selection.json").read_text())
    report = {"scope": "Retrospective external-study and new-chemical transfer of a TG-GATEs-trained rat-to-human kernel rule; only three external compounds.",
              "model": "Portable three-continuous-context chemical kernel ridge; fixed half-prior/half-rat-response fusion; TG-GATEs-only training and tuning.",
              "input_manifest_sha256": manifest["input_manifest_sha256"], "fit_manifest_sha256": sha(out / "fit_manifest.json"),
              "external_endpoint_sha256": expected_truth, "protocol_sha256": manifest["protocol_sha256"],
              "primary_rules": list(PRIMARY_RULES), "diagnostic_rules": list(WRONG_RULES),
              "external_tasks": rows, "macro_mse": macro,
              "rat_information_relative_mse_reduction": 1 - macro["source_plus_context"] / macro["chemical_context"] if macro["chemical_context"] else None,
              "rat_information_benefit_tasks": int(np.sum(scores["source_plus_context"]["mse"] < scores["chemical_context"]["mse"])),
              "rat_information_hurt_tasks": int(np.sum(scores["source_plus_context"]["mse"] > scores["chemical_context"]["mse"])),
              "training_only_final_alpha_selection": selection["final"], "external_human_labels_used_in_fit": False,
              "internal_portable_model": manifest["internal_portable_model"],
              "external_support": manifest["external_support"], "runtime": manifest["runtime"],
              "interpretation_limits": [
                  "Study identity is external but these public outcomes were previously inspected by this project; not a new blind or prospective test.",
                  "Portable model removes TG-GATEs-specific Low/Middle/High indicators and refits on an ID/presence-only common gene axis; it is not an unchanged historical seven-context model checkpoint.",
                  "Only three chemicals from two stress pathways; no reliable population uncertainty or source-selection claim.",
                  "TG-GATEs uses joint-RMA microarray condition means; external RNA-seq logCPM responses differ in measurement and preprocessing. No external human scale correction is fitted.",
                  "CDDO-Me100nM is below the TG-GATEs1uM minimum source and target concentration; retain this predeclared extrapolation task.",
                  "External wrong-source permutations also change source time/dose; these are cross-condition diagnostics, not pure identity-mismatch controls.",
                  "Replicate/lot identities and cross-arm pairing follow each data audit; genes and culture replicates do not increase the number of independent compounds.",
              ]}
    save_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fit-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Positive thread count required")
    torch.set_num_threads(args.threads)
    input_manifest = json.loads((args.input_dir / "manifest.json").read_text())
    common = {"input_manifest_sha256": sha(args.input_dir / "manifest.json"), "protocol_sha256": sha(args.protocol)}
    if args.score_only:
        manifest = json.loads((args.out_dir / "fit_manifest.json").read_text())
        for key, expected in common.items():
            if manifest[key] != expected:
                raise ValueError("Frozen scoring contract changed: " + key)
    else:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError("Use a fresh output directory")
        # Do not read or checksum external_truth here; its expected digest is metadata only.
        for filename in ("train.npz", "external_inputs.npz"):
            if sha(args.input_dir / filename) != input_manifest["files"][filename]:
                raise ValueError("Input checksum mismatch: " + filename)
        train = read_npz(args.input_dir / "train.npz")
        external = read_npz(args.input_dir / "external_inputs.npz")
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = fit_training(train, external, args.device)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_dir / "oof_predictions.npz", **result["oof_predictions"],
                 truth=train["target"], sample_ids=train["sample_ids"], compounds=train["compounds"],
                 outer_fold=train["outer_fold"], human_gene_ids=train["human_gene_ids"],
                 **{name + "__final_cv_row_losses": value for name, value in result["final_cv_losses"].items()})
        np.savez(args.out_dir / "external_predictions.npz", **result["external_predictions"],
                 task_ids=external["task_ids"], human_gene_ids=external["human_gene_ids"], rat_gene_ids=external["rat_gene_ids"])
        np.savez(args.out_dir / "final_fit.npz", **result["final_fit"])
        save_json(args.out_dir / "selection.json", result["selection"])
        save_json(args.out_dir / "cpu_reference_validation.json", result["validation"])
        snapshot = args.out_dir / "code_snapshot"
        snapshot.mkdir()
        code_paths = [Path(__file__), Path(legacy.__file__), Path(sys.modules[TorchRidge.__module__].__file__)]
        for path in code_paths:
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(args.protocol, args.out_dir / "PROTOCOL.md")
        manifest = {**common, "stage": "external_predictions_frozen_without_external_human_endpoint_access",
                    "input_dir": str(args.input_dir.resolve()), "external_human_endpoint_read": False,
                    "context_features": ["time_h", "log10_rat_M", "log10_human_M"],
                    "alpha_grid": list(ALPHAS), "training_rows": len(train["sample_ids"]),
                    "training_compounds": len(set(train["compounds"].tolist())), "genes": len(train["human_gene_ids"]),
                    "external_task_ids": external["task_ids"].tolist(), "external_support": result["external_support"],
                    "internal_portable_model": {"oof_metrics": result["internal_oof_metrics"],
                                                "relative_gain_from_rat": result["internal_relative_gain_from_rat"],
                                                "scope": "Nested TG-GATEs internal OOF diagnostic; final all-training alpha is selected separately on all five TG-GATEs validation folds."},
                    "code_sha256": {path.name: sha(path) for path in code_paths},
                    "runtime": {"device": args.device,
                                "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                                "dtype": "float64", "threads": args.threads, "torch": torch.__version__, "numpy": np.__version__,
                                "fit_and_save_seconds": time.monotonic() - started,
                                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None},
                    "output_files": {str(path.relative_to(args.out_dir)): sha(path) for path in sorted(args.out_dir.rglob("*")) if path.is_file()}}
        save_json(args.out_dir / "fit_manifest.json", manifest)
        print(json.dumps({"stage": manifest["stage"], "training_rows": manifest["training_rows"],
                          "genes": manifest["genes"], "runtime": manifest["runtime"]}), flush=True)
    if not args.fit_only:
        report = score_saved(args.out_dir, args.input_dir, manifest)
        print(json.dumps(plain({"macro_mse": report["macro_mse"],
                               "external_tasks": report["external_tasks"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
