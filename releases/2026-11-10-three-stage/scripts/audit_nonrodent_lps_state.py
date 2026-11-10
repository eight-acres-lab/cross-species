#!/usr/bin/env python3
"""Independently score a globally frozen 20-fit/40-population LPS experiment.

No model fitting, model selection, gene selection, or prediction is performed.
All prediction and provenance hashes are checked before opening assay arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

CONTRACT_SHA = "77d6afe4ea55a8b0fc98507faa1f87aacb01caf46505397a8d85fa67f20433c9"
ARMS = ("H", "H+Rabbit", "H+Hextra", "H+RabbitA0")
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_REPLICATES = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes frozen run")
    return path


def folder_name(fold, arm):
    return f"fold_{fold}_{arm.replace('+', '_')}"


def near(a, b, label, rtol=1e-9, atol=1e-11):
    require(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False), label)


def primary_mse(prediction, target):
    p = np.asarray(prediction, dtype=np.float64).mean(axis=0)
    t = np.asarray(target, dtype=np.float64).mean(axis=0)
    return float(np.mean(np.square(p - t)))


def fitting_scale(expression, records):
    return float(np.mean([primary_mse(expression[r["control_rows"]],
                                      expression[r["target_rows"]]) for r in records]))


def correlation(x, y, centered):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if centered:
        x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0 else float(np.clip(np.dot(x, y) / denominator, -1, 1))


class NativeEnergy:
    """Pinned native GeomLoss operator, including its self-distance epsilon."""

    def __init__(self, device):
        import torch
        import geomloss
        from geomloss import SamplesLoss
        require(importlib.metadata.version("geomloss") == "0.3.1", "Expected audited GeomLoss0.3.1")
        root = Path(geomloss.__file__).resolve().parent
        expected = {
            "_legacy/samples_loss.py": "0503d0badfbbf87e5b859e278500191316cef71fd0f89f6a3d8345f80193e849",
            "_legacy/kernel_samples.py": "5b38eaed54fa31e700eabbe1e6def385c47884781ce6eea31a3fec2eb05fdf5f",
            "_legacy/utils.py": "65c39e3476d8a773d4f1fe0d6342765eec595991f838259bd73150b550dbecec",
        }
        for name, digest in expected.items():
            require(sha256(root / name) == digest, "Native energy implementation changed: " + name)
        torch.set_num_threads(4)
        self.torch, self.device = torch, device
        self.loss = SamplesLoss(loss="energy", blur=0.05, backend="tensorized")
        self.metadata = {
            "library": "geomloss", "version": "0.3.1", "source_hashes": expected,
            "loss": "energy", "blur": 0.05, "backend": "tensorized", "dtype": "float64",
            "device": device, "mmd_num_chunks": 1, "gene_normalization": "none",
            "formula": "mean(Dxy)-0.5*mean(Dxx)-0.5*mean(Dyy), uniform population weights; V-statistic includes diagonals",
            "distance": "native sqrt(clamp_min(squared Euclidean distance,1e-8)); diagonal/duplicate distance1e-4",
            "scale": "one half of the common 2*cross-within-within energy-distance convention",
            "precision_note": "Same pinned native operator evaluated in float64; training used float32.",
            "official_state_source_sha256": "45828e6f5ec4da52d1980fd54881e095ad1e8aaece4689c303fe10a671a57b63",
        }

    def __call__(self, prediction, target):
        with self.torch.no_grad():
            p = self.torch.as_tensor(prediction, dtype=self.torch.float64, device=self.device)
            t = self.torch.as_tensor(target, dtype=self.torch.float64, device=self.device)
            value = float(self.loss(p, t).mean().cpu())
        require(np.isfinite(value), "Nonfinite native energy")
        return value


def verify_global_freeze(args):
    """This function reads metadata and hashes bytes; it never opens assay NPZ."""
    run, prepared = args.run.resolve(), args.prepared.resolve()
    require(sha256(args.contract) == CONTRACT_SHA, "Contract v3 SHA mismatch")
    contract = read_json(args.contract)
    freeze_path = run / "ALL_TARGET_PREDICTIONS_FROZEN.json"
    require(freeze_path.is_file(), "All target predictions must be frozen before scoring")
    frozen = read_json(freeze_path)
    selected_path, spec_path = run / "ALL_FITS_SELECTED.json", run / "specification.json"
    require(sha256(selected_path) == frozen["fits_selected_sha256"], "Selected-fit manifest changed")
    selected, spec = read_json(selected_path), read_json(spec_path)
    require(sha256(spec_path) == selected["specification_sha256"], "Run specification changed")
    require(spec["contract_sha256"] == CONTRACT_SHA and spec["contract"] == contract, "Run used another contract")
    require(frozen["target_outcomes_scored"] is False and selected["target_scored"] is False,
            "Freeze metadata lacks unscored-target declaration")
    require(spec["fold_specific_target_outcomes_used_in_that_fold_fitting_or_selection"] is False,
            "Fold-specific target labels were used before selection")
    require(spec["target_outcomes_scored_before_global_freeze"] is False, "Premature target scoring")
    manifest_path, selections_path = prepared / "manifest.json", prepared / "selections.json"
    manifest, selections = read_json(manifest_path), read_json(selections_path)
    require(manifest == spec["prepared_manifest"], "Prepared manifest differs from frozen run")
    bound = {Path(x["path"]).name: x["sha256"] for x in contract["provenance"]}
    require(sha256(manifest_path) == bound["manifest.json"], "Prepared manifest not contract-bound")
    require(sha256(selections_path) == bound["selections.json"], "Selections not contract-bound")
    for key in ("arrays", "selections"):
        item = manifest[key]
        require(sha256(prepared / Path(item["path"]).name) == item["sha256"], "Prepared artifact changed: " + key)
    for path, digest in spec["inputs"].items():
        if Path(path).name in {args.contract.name, "manifest.json", "selections.json", "lps_primary_arrays.npz"}:
            actual = args.contract if Path(path).name == args.contract.name else prepared / Path(path).name
            require(sha256(actual) == digest, "Input differs from frozen specification: " + path)
    fits = selected["fits"]
    require(len(fits) == 20, "Expected exactly20 selected fits")
    require({(x["fold"], x["arm"]) for x in fits} == {(f, a) for f in range(5) for a in ARMS},
            "Missing, duplicated or unexpected fold/arm")
    expected_predictions = {f"{folder_name(f, a)}/target_{i}.npz"
                            for f in range(5) for a in ARMS for i in range(2)}
    require(set(frozen["artifacts"]) == expected_predictions, "Require all40 target populations before scoring")
    for name, digest in frozen["artifacts"].items():
        require(sha256(checked_path(run, name)) == digest, "Frozen target prediction changed: " + name)
    for fit in fits:
        folder = run / folder_name(fit["fold"], fit["arm"])
        require(read_json(folder / "training.json") == fit, "Training record differs from selected manifest")
        require(sha256(folder / "selected.pt") == fit["weight_sha256"], "Selected checkpoint changed")
        require(fit["target_outcomes_scored"] is False, "Fit record indicates premature target scoring")
        require(len(fit["training_history"]) == 200, "Incomplete fixed optimization budget")
        require([x["step"] for x in fit["calibration_history"]] == [0, 50, 100, 200], "Checkpoint grid changed")
        winner = min(fit["calibration_history"], key=lambda x: (x["macro_mse"], x["step"]))
        require(winner["step"] == fit["selected_step"], "Checkpoint selected outside frozen rule")
        near(winner["macro_mse"], fit["selected_calibration_mse"], "Selected calibration value differs")
        require([x["alpha"] for x in fit["shrink_grid"]] == [0, .25, .5, .75, 1], "Shrink grid changed")
        alpha = min(fit["shrink_grid"], key=lambda x: (x["macro_mse"], x["alpha"]))["alpha"]
        require(alpha == fit["selected_shrink"], "Shrink selected outside frozen rule")
        require(fit["unused_old_perturbation_columns_unchanged"] and fit["unused_old_batch_rows_unchanged"],
                "Unused old coordinates changed")
    audit = {"contract_sha256": CONTRACT_SHA, "specification_sha256": sha256(spec_path),
             "selected_fits_sha256": sha256(selected_path), "global_freeze_sha256": sha256(freeze_path),
             "prepared_manifest_sha256": sha256(manifest_path), "selections_sha256": sha256(selections_path),
             "prediction_artifacts_verified": 40, "selected_weight_files_verified": 20,
             "all_verified_before_opening_assay_arrays": True, "frozen_unix_time": frozen["unix_time"],
             "verification_unix_time": time.time(), "runner_code_hashes_from_frozen_specification": spec["code"]}
    return contract, manifest, selections, fits, frozen, audit


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def validate_rows(data, record, species="human"):
    for part, label in (("control", "Baseline"), ("target", "LPS")):
        rows = np.asarray(record[part + "_rows"])
        require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer), "Invalid row-index dtype")
        require(len(rows) == len(set(rows)) and len(rows) > 0, "Duplicated/empty population rows")
        require(np.all(rows >= 0) and np.all(rows < len(data[species + "_cell_ids"])), "Row index outside assay")
        require(data[species + "_cell_ids"][rows].astype(str).tolist() == record[part + "_cell_ids"], "Row-to-cell-ID mismatch")
        require(np.all(data[species + "_donors"][rows].astype(str) == record["donor"]), "Donor row mismatch")
        if species == "human":
            require(np.all(data["human_condition"][rows].astype(str) == label), "Human condition mismatch")


def audit_fits(data, selections, fits, run):
    """Post-global-freeze audit; calibration values are checked, never reselected."""
    human, rabbit = data["human_expression"], data["rabbit_expression"]
    report = []
    for fit in fits:
        fold = selections["folds"][fit["fold"]]
        for role in ("human_base", "human_extra", "human_cal", "human_test"):
            for rec in fold[role]:
                validate_rows(data, rec)
                for part, label in (("control", "Baseline"), ("target", "LPS")):
                    eligible = np.flatnonzero((data["human_donors"].astype(str) == rec["donor"]) &
                                               (data["human_condition"].astype(str) == label))
                    if part == "target" or role in ("human_cal", "human_test"):
                        require(set(rec[part + "_rows"]) == set(eligible), "An author-QC calibration/test or LPS cell was omitted")
                    else:
                        require(len(rec[part + "_rows"]) == min(128, len(eligible)), "Training control cap changed")
        combined = [r["donor"] for role in ("human_base", "human_extra", "human_cal", "human_test") for r in fold[role]]
        require(len(combined) == len(set(combined)) == 10, "Fold donor leakage")
        scales = {"H": fitting_scale(human, fold["human_base"]),
                  "Rabbit": fitting_scale(rabbit, fold["rabbit_source"]),
                  "Hextra": fitting_scale(human, fold["human_extra"])}
        for k, v in scales.items():
            near(v, fit["source_scale"][k], "Training-only source scale mismatch")
        coefficient = 0 if fit["arm"] == "H" else .5 * scales["H"] / max(scales["Hextra" if fit["arm"] == "H+Hextra" else "Rabbit"], 1e-8)
        near(coefficient, fit["source_coefficient"], "Source coefficient mismatch; A0 must use trueRabbit scale")
        folder = run / folder_name(fit["fold"], fit["arm"])
        calibration = []
        for i, rec in enumerate(fold["human_cal"]):
            out = load_npz(folder / f"calibration_{i}.npz")
            require(out["control_rows"].tolist() == rec["control_rows"] and out["target_rows"].tolist() == rec["target_rows"], "Calibration rows changed")
            calibration.append((out, human[rec["control_rows"]], human[rec["target_rows"]]))
        near(np.mean([primary_mse(o["prediction"], t) for o, c, t in calibration]),
             fit["selected_calibration_mse"], "Selected calibration predictions disagree with frozen score")
        for row in fit["shrink_grid"]:
            value = np.mean([primary_mse(c + row["alpha"] * (o["prediction"] - c), t)
                             for o, c, t in calibration])
            near(value, row["macro_mse"], "Calibration shrink score mismatch")
        if fit["arm"] in ("H+Rabbit", "H+RabbitA0"):
            later = [row for row in fit["animal_gradient_witness"] if row["step"] >= 3]
            require(any(any(v is not None and np.isfinite(v) and v > 0 for k, v in row["norms"].items()
                            if "v_proj" in k or "basal_encoder" in k) for row in later), "No later shared source-data gradient")
        report.append({"fold": fit["fold"], "arm": fit["arm"], "selected_step": fit["selected_step"],
                       "selected_shrink": fit["selected_shrink"], "source_scales": scales,
                       "source_coefficient": coefficient, "a0_retains_real_rabbit_amplitude_scalar": fit["arm"] == "H+RabbitA0",
                       "calibration_prediction_sha256": [sha256(folder / f"calibration_{i}.npz") for i in range(2)]})
    return report


def metric_values(prediction, target, control, energy, contrast=None, reconstruction=None):
    p, t, c = [np.asarray(x, dtype=np.float64) for x in (prediction, target, control)]
    delta, truth = p.mean(0) - c.mean(0), t.mean(0) - c.mean(0)
    mse = primary_mse(p, t)
    near(mse, float(np.square(delta - truth).mean()), "Mean-response/treated-MSE identity failed")
    result = {"mean_response_mse": mse, "native_energy": energy(p, t),
              "response_pearson": correlation(delta, truth, True),
              "response_direction_cosine": correlation(delta, truth, False),
              "predicted_response_l2": float(np.linalg.norm(delta)), "true_response_l2": float(np.linalg.norm(truth)),
              "predicted_response_mean_abs": float(np.abs(delta).mean()),
              "unclamped_native_contrast_mse": None, "clamped_coordinate_fraction": None,
              "native_control_reconstruction_mse": None}
    if contrast is not None:
        contrast = np.asarray(contrast, dtype=np.float64)
        result["unclamped_native_contrast_mse"] = float(np.square(contrast.mean(0) - truth).mean())
        result["clamped_coordinate_fraction"] = float(np.mean(c + contrast < 0))
    if reconstruction is not None:
        result["native_control_reconstruction_mse"] = primary_mse(reconstruction, control)
    return result


def score_predictions(args, data, selections, fits, frozen, energy):
    human, ix = data["human_expression"], data["native_gene_indices"]
    require(ix.shape == (4633,) and len(set(ix)) == 4633 and np.all(np.diff(ix) > 0), "Measured-axis corruption")
    records, seen = [], set()
    for fit in fits:
        fold = selections["folds"][fit["fold"]]
        base_delta = np.mean([human[r["target_rows"]].mean(0, dtype=np.float64) - human[r["control_rows"]].mean(0, dtype=np.float64)
                              for r in fold["human_base"]], axis=0)
        for i, rec in enumerate(fold["human_test"]):
            name = f"{folder_name(fit['fold'], fit['arm'])}/target_{i}.npz"
            path = checked_path(args.run, name)
            require(sha256(path) == frozen["artifacts"][name], "Prediction changed after global verification")
            out = load_npz(path)
            require(str(out["donor"].item()) == rec["donor"], "Target prediction donor mismatch")
            require(out["control_rows"].tolist() == rec["control_rows"] and out["target_rows"].tolist() == rec["target_rows"], "Frozen target row IDs changed")
            require(np.array_equal(out["native_gene_indices"], ix), "Prediction gene axis differs")
            c = human[rec["control_rows"]]
            # First target-truth indexing occurs only after the global40-file gate.
            target = human[rec["target_rows"]]
            for field in ("prediction", "contrast", "native_control", "calibrated_shrink", "zero_response", "base_mean_response", "half_response"):
                require(out[field].shape == c.shape and np.isfinite(out[field]).all(), "Invalid frozen prediction field: " + field)
            require(np.array_equal(out["zero_response"], c), "Zero-response input differs")
            require(np.array_equal(out["prediction"], np.maximum(c + out["contrast"], 0)), "Anchored clamp readout differs")
            require(np.array_equal(out["calibrated_shrink"], c + fit["selected_shrink"] * (out["prediction"] - c)), "Shrink prediction differs from frozen alpha")
            require(np.array_equal(out["half_response"], c + .5 * (out["prediction"] - c)), "Half-response prediction differs")
            near(out["base_mean_response"], np.maximum(c + base_delta, 0), "Base-human mean-response baseline differs")
            if fit["selected_step"] == 0:
                require(np.array_equal(out["prediction"], c), "Zero-step selected model must predict zero response")
            variants = [(fit["arm"], out["prediction"], out["contrast"], out["native_control"]),
                        (fit["arm"] + "|cal_shrink", out["calibrated_shrink"], None, None)]
            if fit["arm"] == "H":
                variants.extend([("Zero_response", out["zero_response"], None, None),
                                 ("Hbase_mean_response", out["base_mean_response"], None, None),
                                 ("Half_H_response", out["half_response"], None, None)])
            for variant, prediction, contrast, reconstruction in variants:
                key = (rec["donor"], variant)
                require(key not in seen, "Repeated out-of-fold biological donor score")
                seen.add(key)
                row = {"fold": fit["fold"], "donor": rec["donor"], "variant": variant,
                       "n_control": len(c), "n_target": len(target), "n_genes": 4633,
                       "selected_step": fit["selected_step"], "selected_alpha": fit["selected_shrink"],
                       "source_coefficient": fit["source_coefficient"],
                       "prediction_sha256": frozen["artifacts"][name]}
                row.update(metric_values(prediction, target, c, energy, contrast, reconstruction))
                records.append(row)
    require(len(records) == 110 and len({r["donor"] for r in records}) == 10, "Incomplete11-variant/10-donor scores")
    return records


COMPARISONS = {
    "delta_A": ("H", "H+Rabbit"), "delta_H": ("H", "H+Hextra"),
    "source_information_gain": ("H+RabbitA0", "H+Rabbit"),
    "gain_over_H_shrink": ("H|cal_shrink", "H+Rabbit"),
    "gain_over_zero": ("Zero_response", "H+Rabbit"),
    "gain_over_Hbase_mean": ("Hbase_mean_response", "H+Rabbit"),
    "gain_over_half_H": ("Half_H_response", "H+Rabbit"),
    "animal_minus_extra_human_error": ("H+Rabbit", "H+Hextra"),
    "shrunk_delta_A": ("H|cal_shrink", "H+Rabbit|cal_shrink"),
    "shrunk_delta_H": ("H|cal_shrink", "H+Hextra|cal_shrink"),
    "shrunk_source_information_gain": ("H+RabbitA0|cal_shrink", "H+Rabbit|cal_shrink"),
}


def summarize(records, fits):
    donors = sorted({r["donor"] for r in records})
    lookup = {(r["donor"], r["variant"]): r for r in records}
    variants = sorted({r["variant"] for r in records})
    boot_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 10, size=(BOOTSTRAP_REPLICATES, 10))
    group = {}
    for v in variants:
        rows = [lookup[d, v] for d in donors]
        metrics = {}
        for key in ("mean_response_mse", "native_energy", "response_pearson", "response_direction_cosine",
                    "unclamped_native_contrast_mse", "clamped_coordinate_fraction", "native_control_reconstruction_mse"):
            finite = [r[key] for r in rows if r[key] is not None]
            metrics[key] = {"mean": None if not finite else float(np.mean(finite)), "defined_donors": len(finite)}
        mse = np.asarray([r["mean_response_mse"] for r in rows])
        metrics["mean_response_mse"]["descriptive_bootstrap_95pct"] = np.quantile(mse[boot_indices].mean(1), [.025, .975]).tolist()
        group[v] = metrics
    paired, paired_rows = {}, []
    for name, (left, right) in COMPARISONS.items():
        differences = np.asarray([lookup[d, left]["mean_response_mse"] - lookup[d, right]["mean_response_mse"] for d in donors])
        per_fold = {str(f): float(np.mean([differences[i] for i, d in enumerate(donors) if lookup[d, left]["fold"] == f])) for f in range(5)}
        comparator_mean = float(np.mean([lookup[d, left]["mean_response_mse"] for d in donors]))
        paired[name] = {"left": left, "right": right, "definition": "left MSE minus right MSE",
                        "mean_difference": float(differences.mean()), "left_comparator_mean_mse": comparator_mean,
                        "percent_of_left_mean": None if comparator_mean == 0 else float(100 * differences.mean() / comparator_mean),
                        "positive_donors": int(np.sum(differences > 0)), "zero_donors": int(np.sum(differences == 0)),
                        "negative_donors": int(np.sum(differences < 0)), "fold_mean_differences": per_fold,
                        "descriptive_paired_bootstrap_95pct": np.quantile(differences[boot_indices].mean(1), [.025, .975]).tolist()}
        secondary = {}
        for metric in ("native_energy", "response_pearson", "response_direction_cosine"):
            valid = [(lookup[d, left][metric], lookup[d, right][metric]) for d in donors
                     if lookup[d, left][metric] is not None and lookup[d, right][metric] is not None]
            secondary[metric] = {"left_minus_right_mean": None if not valid else float(np.mean([a - b for a, b in valid])),
                                 "defined_paired_donors": len(valid),
                                 "better_direction": "negative favors left" if metric == "native_energy" else "positive favors left"}
        paired[name]["secondary_paired_metrics"] = secondary
        for d, difference in zip(donors, differences):
            row = {"donor": d, "fold": lookup[d, left]["fold"], "comparison": name,
                   "left_variant": left, "right_variant": right,
                   "left_mse": lookup[d, left]["mean_response_mse"], "right_mse": lookup[d, right]["mean_response_mse"],
                   "mse_difference": float(difference)}
            for metric in ("native_energy", "response_pearson", "response_direction_cosine"):
                lv, rv = lookup[d, left][metric], lookup[d, right][metric]
                row[metric + "_left_minus_right"] = None if lv is None or rv is None else lv - rv
            paired_rows.append(row)
    gate = {name + "_positive": paired[name]["mean_difference"] > 0
            for name in ("delta_A", "delta_H", "source_information_gain", "gain_over_H_shrink", "gain_over_zero", "gain_over_Hbase_mean")}
    gate["at_least8_of10_donors_improve"] = paired["delta_A"]["positive_donors"] >= 8
    gate["all5_fold_mean_gains_positive"] = all(x > 0 for x in paired["delta_A"]["fold_mean_differences"].values())
    gate["all_rabbit_selected_steps_positive"] = all(f["selected_step"] > 0 for f in fits if f["arm"] == "H+Rabbit")
    delta_a, delta_h = paired["delta_A"]["mean_difference"], paired["delta_H"]["mean_difference"]
    ratio = None if delta_h <= 0 else delta_a / delta_h
    summary = {"biological_test_units": 10, "outer_folds": 5, "fits": 20, "variants": group,
               "paired_comparisons": paired, "positive_candidate_gate": {"passes": all(gate.values()), "checks": gate,
               "meaning": "Descriptive prespecified stability screen; no confirmatory p-value or independence claim."},
               "shrink_claim_eligibility": {a: all(f["selected_step"] > 0 and f["selected_shrink"] > 0 for f in fits if f["arm"] == a) for a in ARMS},
               "animal_fraction_of_extra_human_gain": {"value": ratio, "denominator_delta_H": delta_h,
               "denominator_descriptive_interval": paired["delta_H"]["descriptive_paired_bootstrap_95pct"],
               "stability": "Undefined when delta_H<=0; otherwise descriptive and sensitive to a small denominator. No equivalence or noninferiority conclusion."},
               "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "unit": "human donor",
               "shared_resample_indices_across_variants": True, "interval": "percentile95%",
               "interpretation": "Descriptive conditional on the20 fixed trained/selected models; excludes retraining and selection uncertainty. Overlapping CV training sets remain dependent."},
               "p_values": "Not computed; naive CV sign/permutation tests and cell/gene pseudo-replication are not confirmatory."}
    return summary, paired_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prepared", "contract", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--verify-only", action="store_true", help="Verify full frozen provenance without opening assay NPZ or scoring")
    args = parser.parse_args()
    contract, manifest, selections, fits, frozen, provenance = verify_global_freeze(args)
    if args.verify_only:
        print(json.dumps(provenance, indent=2))
        return
    require(not args.out.exists(), "Use a new audit output directory; preserve prior audit artifacts")
    energy = NativeEnergy(args.device)
    # No assay expression has been opened before this line.
    data = load_npz(args.prepared / "lps_primary_arrays.npz")
    fit_audit = audit_fits(data, selections, fits, args.run)
    records = score_predictions(args, data, selections, fits, frozen, energy)
    summary, paired_rows = summarize(records, fits)
    provenance.update({"scorer_sha256": sha256(__file__), "audit_started_after_full_freeze": True,
                       "native_energy": energy.metadata, "numpy_version": np.__version__,
                       "fit_audit": fit_audit, "limitations": contract["interpretation"]["limitations"],
                       "historical_evidence_preserved": contract["scope"]["historical_negative_evidence"]})
    args.out.mkdir(parents=True)
    write_csv(args.out / "donor_scores.csv", records)
    write_csv(args.out / "paired_donor_differences.csv", paired_rows)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "audit_provenance.json", provenance)
    outputs = {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out / "AUDIT_COMPLETE.json", {"contract_sha256": CONTRACT_SHA,
               "global_prediction_freeze_sha256": provenance["global_freeze_sha256"],
               "scorer_sha256": sha256(__file__), "outputs": outputs, "unix_time": time.time(),
               "donors": 10, "score_rows": len(records), "paired_rows": len(paired_rows)})
    print(json.dumps({"audit_output": str(args.out), "donors": 10,
                      "positive_candidate_gate": summary["positive_candidate_gate"]}, indent=2))


if __name__ == "__main__":
    main()
