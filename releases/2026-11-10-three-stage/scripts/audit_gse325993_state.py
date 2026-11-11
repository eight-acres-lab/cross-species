#!/usr/bin/env python3
"""Independently score the globally frozen36fit GSE325993 comparison.

Verify all prediction/weight/calibration hashes before opening assay arrays.
Three sampling seeds remain computational repeats; biological target N is2.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np

CONTRACT_SHA = "f69e4a440bd35d1ea66b727e5f41fa02e7e412c3b5986b1a6331a883dfc5e9fd"
ARMS = ("H", "Hextra_cells", "Monkey56", "Monkey56A0", "Monkey82", "Monkey82A0")
SEEDS = (20261110, 20261111, 20261112)
SOURCES = ("Monkey56", "Monkey82")
METRICS = ("mean_response_mse", "native_energy", "response_pearson", "response_cosine",
           "unclamped_contrast_mse", "clamped_coordinate_fraction", "native_control_reconstruction_mse")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def folder_name(fold, seed, arm):
    return f"fold_{fold}_seed_{seed}_{arm}"


def safe_path(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Frozen path escapes run directory")
    return path


def near(a, b, label, rtol=1e-9, atol=1e-11):
    require(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False), label)


def mse(prediction, target):
    p, t = [np.asarray(x, dtype=np.float64).mean(0) for x in (prediction, target)]
    return float(np.square(p - t).mean())


def corr(x, y, centered=False):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if centered:
        x, y = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if den == 0 else float(np.clip(np.dot(x, y) / den, -1., 1.))


class NativeEnergy:
    def __init__(self, device):
        import torch
        import geomloss
        from geomloss import SamplesLoss
        require(importlib.metadata.version("geomloss") == "0.3.1", "Expected audited GeomLoss0.3.1")
        root = Path(geomloss.__file__).resolve().parent
        hashes = {
            "_legacy/samples_loss.py": "0503d0badfbbf87e5b859e278500191316cef71fd0f89f6a3d8345f80193e849",
            "_legacy/kernel_samples.py": "5b38eaed54fa31e700eabbe1e6def385c47884781ce6eea31a3fec2eb05fdf5f",
            "_legacy/utils.py": "65c39e3476d8a773d4f1fe0d6342765eec595991f838259bd73150b550dbecec",
        }
        for name, digest in hashes.items():
            require(sha256(root / name) == digest, "Native energy source changed: " + name)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch, self.device = torch, device
        self.loss = SamplesLoss(loss="energy", blur=.05, backend="tensorized")
        self.metadata = {"library": "GeomLoss0.3.1", "source_hashes": hashes, "loss": "energy",
                         "blur": .05, "backend": "tensorized", "dtype": "float64", "device": device,
                         "formula": "meanDxy-0.5meanDxx-0.5meanDyy; uniform1/N and1/M V-statistic",
                         "distance": "sqrt(clamp_min(squared Euclidean,1e-8)); native self-distance1e-4 retained",
                         "gene_normalization": "none; full6357observed coordinates",
                         "precision_note": "Native energy/metrics float64. Model, C/P and scalar-shrink arithmetic float32; Hbase-mean delta remains float64 before addition/clamp."}

    def __call__(self, p, t):
        with self.torch.no_grad():
            p = self.torch.as_tensor(p, dtype=self.torch.float64, device=self.device)
            t = self.torch.as_tensor(t, dtype=self.torch.float64, device=self.device)
            value = float(self.loss(p, t).mean().cpu())
        require(np.isfinite(value), "Nonfinite native energy")
        return value


def global_gate(args):
    """Metadata and byte hashes only: expression NPZ arrays stay unopened."""
    require(sha256(args.contract) == CONTRACT_SHA, "Frozen contract SHA mismatch")
    contract = read(args.contract)
    require(tuple(contract["design"]["arms"]) == ARMS and tuple(contract["design"]["seeds"]) == SEEDS, "Wrong declared grid")
    run, prepared, code_root = args.run.resolve(), args.prepared.resolve(), args.code_root.resolve()
    require(code_root.is_dir(), "Frozen code snapshot directory is missing")
    marker_path = run / "ALL_TARGET_PREDICTIONS_FROZEN.json"
    require(marker_path.exists(), "Complete all36target predictions before scoring")
    marker = read(marker_path)
    require(marker["contract_sha256"] == CONTRACT_SHA and marker["fits"] == 36 and marker["target_units"] == 2,
            "Global frozen marker has wrong scope")
    require(marker["target_outcomes_scored"] is False, "Premature target scoring indicated")
    fit_path, spec_path = run / "ALL_FITS_SELECTED.json", run / "specification.json"
    require(sha256(fit_path) == marker["fits_selected_sha256"], "Fit manifest changed")
    fit_manifest, spec = read(fit_path), read(spec_path)
    require(fit_manifest["target_outcomes_scored"] is False, "Fit manifest indicates scored targets")
    require(sha256(spec_path) == fit_manifest["specification_sha256"], "Frozen specification changed")
    require(spec["contract_sha256"] == CONTRACT_SHA and spec["contract"] == contract, "Run used another contract")
    require(spec["fold_specific_test_outcomes_used_for_training_or_selection"] is False, "Fold-specific label leakage flag")
    require(spec["all_fits_and_selections_before_target_inference"] is True and spec["actual_primary_target_units"] == 2,
            "Run isolation/unit metadata differ")
    require(spec["checkpoint_sha256"] == contract["model"]["checkpoint_sha256"], "Original checkpoint changed")
    require(spec["code"][contract["model"]["adapter"]] == contract["model"]["adapter_sha256"], "Adapter differs from contract")
    runner_relative = "scripts/run_gse325993_state.py"
    require(runner_relative in spec["code"], "Frozen runner code binding is missing")
    for relative, digest in spec["code"].items():
        require(not Path(relative).is_absolute(), "Code snapshot entries must be relative paths")
        actual_path = safe_path(code_root, relative)
        require(actual_path.is_file(), "Frozen code snapshot file missing: " + relative)
        require(sha256(actual_path) == digest, "Frozen code snapshot changed: " + relative)
    binding = contract["data"]
    require(prepared == Path(binding["prepared_root"]).resolve(), "Prepared data root changed")
    for name, digest in binding["files"].items():
        require(sha256(prepared / name) == digest, "Bound input changed: " + name)
    manifest = read(prepared / "manifest.json")
    require(manifest == spec["prepared_manifest"], "Prepared manifest differs from run")
    sm, splits = read(args.splits / "manifest.json"), read(args.splits / "splits.json")
    require(sm == spec["split_manifest"], "Split manifest differs from run")
    require(sm["code_sha256"] == spec["code"][runner_relative], "Split runner hash differs from frozen training runner")
    require(sm["contract_sha256"] == CONTRACT_SHA and sm["arrays_sha256"] == binding["arrays_sha256"], "Split bindings changed")
    require(sm["prepared_manifest_sha256"] == binding["prepared_manifest_sha256"], "Split prepared manifest mismatch")
    require(sm["splits_sha256"] == spec["splits_sha256"] == sha256(args.splits / "splits.json"), "Split JSON changed")
    require(sm["metadata_only"] is True and sm["expression_values_read"] is False, "Split tool was not metadata-only")
    fits = fit_manifest["fits"]
    expected_grid = {(f, s, a) for f in (0, 1) for s in SEEDS for a in ARMS}
    require(len(fits) == 36 and {(x["fold"], x["seed"], x["arm"]) for x in fits} == expected_grid, "Incomplete/duplicated36fit grid")
    expected_paths = {folder_name(f, s, a) + "/target.npz" for f, s, a in expected_grid}
    require(set(marker["artifacts"]) == expected_paths, "Missing/unexpected target artifacts")
    for name, digest in marker["artifacts"].items():
        require(sha256(safe_path(run, name)) == digest, "Frozen prediction changed: " + name)
    for fit in fits:
        folder = run / folder_name(fit["fold"], fit["seed"], fit["arm"])
        require(read(folder / "training.json") == fit, "Selected/training record disagreement")
        require(sha256(folder / "selected.pt") == fit["selected_weight_sha256"], "Selected weights changed")
        require(sha256(folder / "calibration.npz") == fit["calibration_predictions_sha256"], "Calibration predictions changed")
        require(fit["target_outcomes_scored"] is False and fit["old_pert_columns_and_batch_rows_unchanged"] is True,
                "Fit provenance/interface assertion failed")
        require([x["step"] for x in fit["training_history"]] == list(range(1, 201)), "Incorrect optimization steps")
        require([x["step"] for x in fit["calibration_history"]] == [0, 50, 100, 200], "Checkpoint grid differs")
        winner = min(fit["calibration_history"], key=lambda x: (x["mse"], x["step"]))
        require(winner["step"] == fit["selected_step"], "Checkpoint selection differs from calibration-only rule")
        near(winner["mse"], fit["selected_calibration_mse"], "Selected calibration value differs")
        require([x["alpha"] for x in fit["shrink_grid"]] == [0, .25, .5, .75, 1], "Shrink grid differs")
        winner_alpha = min(fit["shrink_grid"], key=lambda x: (x["mse"], x["alpha"]))["alpha"]
        require(winner_alpha == fit["selected_shrink"], "Shrink selected outside frozen rule")
    provenance = {"contract_sha256": CONTRACT_SHA, "global_freeze_sha256": sha256(marker_path),
                  "selected_fits_sha256": sha256(fit_path), "specification_sha256": sha256(spec_path),
                  "split_manifest_sha256": sha256(args.splits / "manifest.json"), "splits_sha256": sm["splits_sha256"],
                  "input_files": binding["files"], "all36target_and36weight_and36calibration_hashes_verified": True,
                  "expression_arrays_opened_during_global_gate": False, "verification_unix_time": time.time(),
                  "frozen_unix_time": marker["unix_time"], "training_code_hashes": spec["code"],
                  "code_snapshot_root": str(code_root), "actual_code_snapshot_files_verified": len(spec["code"]),
                  "split_runner_hash_matches_training_runner": True}
    return contract, manifest, splits, fits, marker, provenance


def fold_split(splits, seed, fold):
    seeds = [x for x in splits["seed_splits"] if x["seed"] == seed]
    require(len(seeds) == 1, "Duplicate/missing seed split")
    folds = [x for x in seeds[0]["folds"] if x["fold"] == fold]
    require(len(folds) == 1, "Duplicate/missing outer fold")
    return folds[0]


def validate_record(data, record, species, target_condition="POU5F1", budget=None):
    for part, condition in (("control", "NTC"), ("target", target_condition)):
        rows = np.asarray(record[part + "_rows"])
        require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer), "Invalid row indices")
        require(len(rows) > 0 and len(rows) == len(set(rows)), "Duplicated/empty actual cell population")
        require(np.all(rows >= 0) and np.all(rows < len(data[species + "_cell_ids"])), "Row out of bounds")
        require(data[species + "_cell_ids"][rows].astype(str).tolist() == record[part + "_cell_ids"], "Cell ID/row mismatch")
        require(np.all(data[species + "_cell_lines"][rows].astype(str) == record["line"]), "Cell line mismatch")
        require(np.all(data[species + "_condition"][rows].astype(str) == condition), "Cell condition mismatch")
        require(np.all(data[species + "_experiment"][rows].astype(str) == "exp2"), "Unexpected experiment")
        if budget is not None:
            require(len(rows) == budget, "Changed per-condition cell budget")


def validate_splits(contract, splits, data):
    require([x["seed"] for x in splits["seed_splits"]] == list(SEEDS), "Seed order differs")
    require(splits["expression_values_read"] is False and splits["expression_values_used_for_pool_selection"] is False,
            "Pool selection used expression")
    for seed in SEEDS:
        for expected in contract["design"]["outer_folds"]:
            fold = fold_split(splits, seed, expected["fold"])
            require(all(fold[k] == expected[k] for k in ("fold", "train_line", "test_line")), "Outer split differs")
            all_train = []
            for role in ("base", "extra", "cal"):
                rec = fold["human"][role]
                require(rec["line"] == fold["train_line"], "Training/calibration line leakage")
                validate_record(data, rec, "human", budget=128)
                all_train.extend(rec["control_rows"] + rec["target_rows"])
            require(len(all_train) == len(set(all_train)) == 768, "Human base/extra/cal cells overlap")
            rec = fold["test"]
            require(rec["line"] == fold["test_line"], "Test line changed")
            validate_record(data, rec, "human")
            for part, cond in (("control", "NTC"), ("target", "POU5F1")):
                expected_rows = np.flatnonzero((data["human_cell_lines"].astype(str) == rec["line"]) &
                                               (data["human_condition"].astype(str) == cond))
                require(rec[part + "_rows"] == expected_rows.tolist(), "Whole-line target population/order changed")
                require(not set(all_train) & set(expected_rows), "Target cells entered training/calibration")
            for source in SOURCES:
                real, null = [fold["animal"][source][k] for k in ("real", "null")]
                require(real["line"] == null["line"] == contract["design"]["line_ids"][source], "Source line differs")
                validate_record(data, real, "cynomolgus", budget=128)
                validate_record(data, null, "cynomolgus", target_condition="NTC", budget=128)
                require(real["control_rows"] == null["control_rows"], "A/A0 input cells differ")
                require(not set(null["control_rows"]) & set(null["target_rows"]), "A0 repeats input cells as targets")
                other = fold_split(splits, seed, 1 - expected["fold"])
                require(fold["animal"][source] == other["animal"][source], "Source pools change across outer folds")


def reconstruct_baselines(controls, prediction, alpha, base_delta):
    """Frozen factors only; preserve runner float32 shrink/float64 mean semantics."""
    c, p = np.asarray(controls, dtype=np.float32), np.asarray(prediction, dtype=np.float32)
    delta = np.asarray(base_delta, dtype=np.float64)
    return {"calibrated_shrink": c + np.float32(alpha) * (p - c),
            "zero_response": c,
            "half_response": c + np.float32(.5) * (p - c),
            "three_quarter_response": c + np.float32(.75) * (p - c),
            "base_mean_response": np.maximum(c + delta, 0)}


def score_metrics(prediction, target, controls, energy, contrast=None, reconstruction=None):
    p, t, c = [np.asarray(x, dtype=np.float64) for x in (prediction, target, controls)]
    delta, truth = p.mean(0) - c.mean(0), t.mean(0) - c.mean(0)
    error = mse(p, t)
    near(error, float(np.square(delta - truth).mean()), "Response/treated mean MSE identity failed")
    result = {"mean_response_mse": error, "native_energy": energy(p, t),
              "response_pearson": corr(delta, truth, True), "response_cosine": corr(delta, truth),
              "predicted_response_l2": float(np.linalg.norm(delta)), "true_response_l2": float(np.linalg.norm(truth)),
              "unclamped_contrast_mse": None, "clamped_coordinate_fraction": None,
              "native_control_reconstruction_mse": None}
    if contrast is not None:
        contrast = np.asarray(contrast, dtype=np.float64)
        result["unclamped_contrast_mse"] = float(np.square(contrast.mean(0) - truth).mean())
        result["clamped_coordinate_fraction"] = float(np.mean(c + contrast < 0))
    if reconstruction is not None:
        result["native_control_reconstruction_mse"] = mse(reconstruction, controls)
    return result


def audit_and_score(args, contract, splits, fits, marker, data, energy):
    validate_splits(contract, splits, data)
    ix = data["native_gene_indices"]
    require(ix.shape == (6357,) and len(set(ix)) == 6357 and np.all(np.diff(ix) > 0), "Measured gene mask differs")
    require(np.array_equal(data["human_gene_ids"], data["cynomolgus_gene_ids"]), "Human/Liftoff coordinates differ")
    human, animal = data["human_expression"], data["cynomolgus_expression"]
    require(human.shape == (6169, 6357) and animal.shape == (3563, 6357), "Expression dimensions changed")
    for species in ("human", "cynomolgus"):
        require(np.isfinite(data[species + "_expression"]).all() and np.min(data[species + "_expression"]) >= 0, "Invalid normalized expression")
        require(np.all(data[species + "_coding_library_size"] > 0), "Invalid coding denominator")
    records, fit_audits, seen = [], [], set()
    for fit in fits:
        fold = fold_split(splits, fit["seed"], fit["fold"])
        require(fit["train_line"] == fold["train_line"] and fit["test_line"] == fold["test_line"], "Fit line identities differ")
        folder = args.run / folder_name(fit["fold"], fit["seed"], fit["arm"])
        base, extra, cal = [fold["human"][k] for k in ("base", "extra", "cal")]
        scale = lambda x, r: mse(x[r["control_rows"]], x[r["target_rows"]])
        scales = {"H": scale(human, base), "Hextra_cells": scale(human, extra),
                  **{s: scale(animal, fold["animal"][s]["real"]) for s in SOURCES}}
        for key, value in scales.items():
            near(value, fit["source_scales"][key], "Training source scale mismatch")
        source = fit["arm"].removesuffix("A0")
        coefficient = 0. if source == "H" else .5 * scales["H"] / max(scales[source], 1e-8)
        near(coefficient, fit["source_coefficient"], "Source/A0 coefficient differs")
        cal_pred = load_npz(folder / "calibration.npz")
        require(cal_pred["control_rows"].tolist() == cal["control_rows"] and cal_pred["target_rows"].tolist() == cal["target_rows"], "Calibration cells changed")
        near(mse(cal_pred["prediction"], human[cal["target_rows"]]), fit["selected_calibration_mse"], "Calibration metric mismatch")
        cc = human[cal["control_rows"]]
        for row in fit["shrink_grid"]:
            near(mse(cc + np.float32(row["alpha"]) * (cal_pred["prediction"] - cc), human[cal["target_rows"]]), row["mse"], "Calibration shrink differs")
        if fit["arm"].startswith("Monkey"):
            witnesses = fit["source_gradient_witness"]
            require(any(r["step"] >= 3 and any(v["weighted_source_norm"] is not None and np.isfinite(v["weighted_source_norm"]) and v["weighted_source_norm"] > 0
                        for k, v in r["parameters"].items() if "v_proj" in k or "basal_encoder" in k) for r in witnesses), "No later shared source-data gradient")
        name = folder_name(fit["fold"], fit["seed"], fit["arm"]) + "/target.npz"
        path = safe_path(args.run, name)
        require(sha256(path) == marker["artifacts"][name], "Target changed after global verification")
        out = load_npz(path)
        test = fold["test"]
        require(str(out["donor_line"].item()) == test["line"], "Target prediction line differs")
        require(out["control_rows"].tolist() == test["control_rows"] and out["target_rows"].tolist() == test["target_rows"], "Target cell IDs changed")
        require(np.array_equal(out["native_gene_indices"], ix), "Target prediction mask differs")
        controls = human[test["control_rows"]]
        # First held-out target indexing for metrics occurs after global_gate.
        target = human[test["target_rows"]]
        for key in ("prediction", "contrast", "native_control"):
            require(out[key].shape == controls.shape and np.isfinite(out[key]).all(), "Bad target field: " + key)
        require(np.array_equal(out["prediction"], np.maximum(controls + out["contrast"], 0)), "Anchored readout differs")
        require(float(out["selected_shrink"]) == fit["selected_shrink"], "Frozen alpha factor differs")
        base_delta = human[base["target_rows"]].mean(0, dtype=np.float64) - human[base["control_rows"]].mean(0, dtype=np.float64)
        require(out["base_mean_delta"].shape == (6357,) and np.isfinite(out["base_mean_delta"]).all(), "Invalid mean-response factor")
        near(out["base_mean_delta"], base_delta, "Frozen Hbase-only mean-response factor differs")
        baselines = reconstruct_baselines(controls, out["prediction"], fit["selected_shrink"], out["base_mean_delta"])
        if fit["selected_step"] == 0:
            require(np.array_equal(out["prediction"], controls), "Zero-step fit differs from zero response")
        variants = [(fit["arm"], out["prediction"], out["contrast"], out["native_control"]),
                    (fit["arm"] + "|cal_shrink", baselines["calibrated_shrink"], None, None)]
        if fit["arm"] == "H":
            variants += [("Zero_response", baselines["zero_response"], None, None),
                         ("Hbase_mean_response", baselines["base_mean_response"], None, None),
                         ("Half_H_response", baselines["half_response"], None, None),
                         ("Three_quarter_H_response", baselines["three_quarter_response"], None, None)]
        for variant, prediction, contrast, reconstruction in variants:
            key = (fit["fold"], fit["seed"], variant)
            require(key not in seen, "Repeated line-seed-variant score")
            seen.add(key)
            row = {"fold": fit["fold"], "seed": fit["seed"], "train_line": fit["train_line"], "target_line": test["line"],
                   "variant": variant, "n_control": len(controls), "n_target": len(target), "n_genes": 6357,
                   "selected_step": fit["selected_step"], "selected_alpha": fit["selected_shrink"],
                   "source_coefficient": coefficient, "prediction_sha256": marker["artifacts"][name]}
            row.update(score_metrics(prediction, target, controls, energy, contrast, reconstruction))
            records.append(row)
        fit_audits.append({"fold": fit["fold"], "seed": fit["seed"], "arm": fit["arm"],
                           "selected_step": fit["selected_step"], "selected_alpha": fit["selected_shrink"],
                           "training_scales": scales, "source_coefficient": coefficient,
                           "A0_retains_real_source_amplitude_scalar": fit["arm"].endswith("A0")})
    require(len(records) == 96, "Expected16variants x2lines x3seeds")
    return records, fit_audits


def metric_average(rows):
    result = {}
    for metric in METRICS:
        values = [r[metric] for r in rows]
        result[metric] = None if any(v is None for v in values) else float(np.mean(values))
        result[metric + "_defined_count"] = sum(v is not None for v in values)
    return result


def summarize(records, fits, contract):
    variants = sorted({r["variant"] for r in records})
    lines = contract["design"]["line_ids"]["human"]
    lookup = {(r["target_line"], r["seed"], r["variant"]): r for r in records}
    summaries, line_rows = {}, []
    for variant in variants:
        per_line = []
        for line in lines:
            rows = [lookup[line, seed, variant] for seed in SEEDS]
            avg = metric_average(rows)
            avg.update({"target_line": line, "variant": variant, "sampling_seed_count": 3,
                        "seed_mse_min": min(r["mean_response_mse"] for r in rows),
                        "seed_mse_max": max(r["mean_response_mse"] for r in rows),
                        "seed_mse_sd_descriptive": float(np.std([r["mean_response_mse"] for r in rows], ddof=1))})
            per_line.append(avg); line_rows.append(avg)
        summaries[variant] = {"per_line": per_line, "macro": metric_average(per_line), "biological_target_lines": 2}
    paired_rows, comparisons = [], {}
    requests = [("extra_human_gain", "H", "Hextra_cells")]
    for source in SOURCES:
        for name, comparator in (("gain_over_H", "H"), ("gain_over_extra_human", "Hextra_cells"),
                ("gain_over_A0", source + "A0"), ("gain_over_Hcal_shrink", "H|cal_shrink"),
                ("gain_over_zero", "Zero_response"), ("gain_over_Hbase_mean", "Hbase_mean_response"),
                ("gain_over_halfH", "Half_H_response"), ("gain_over_threequarterH", "Three_quarter_H_response")):
            requests.append((source + ":" + name, comparator, source))
        requests.append((source + ":shrunk_gain_over_H", "H|cal_shrink", source + "|cal_shrink"))
        requests.append((source + ":shrunk_gain_over_A0", source + "A0|cal_shrink", source + "|cal_shrink"))
    for name, left, right in requests:
        line_differences = []
        for line in lines:
            seed_differences = []
            for seed in SEEDS:
                l, r = lookup[line, seed, left], lookup[line, seed, right]
                diff = l["mean_response_mse"] - r["mean_response_mse"]
                seed_differences.append(diff)
                paired_rows.append({"comparison": name, "target_line": line, "seed": seed,
                                    "left_variant": left, "right_variant": right,
                                    "left_mse": l["mean_response_mse"], "right_mse": r["mean_response_mse"],
                                    "left_minus_right_mse": diff})
            line_differences.append({"target_line": line, "seed_mean_difference": float(np.mean(seed_differences)),
                                     "positive_seeds": sum(x > 0 for x in seed_differences),
                                     "seed_differences": seed_differences})
        mean_difference = float(np.mean([x["seed_mean_difference"] for x in line_differences]))
        denominator = summaries[left]["macro"]["mean_response_mse"]
        comparisons[name] = {"left": left, "right": right, "definition": "leftMSE minus rightMSE; positive favorsright",
                             "per_line": line_differences, "macro_difference": mean_difference,
                             "positive_target_lines": sum(x["seed_mean_difference"] > 0 for x in line_differences),
                             "comparator_macro_mse": denominator,
                             "percent_of_comparator": None if denominator == 0 else 100 * mean_difference / denominator}
    sources = {}
    human_gain = comparisons["extra_human_gain"]["macro_difference"]
    for source in SOURCES:
        criteria = {"both_target_lines_improve_over_H": comparisons[source + ":gain_over_H"]["positive_target_lines"] == 2,
                    "exceeds_own_A0_mean": comparisons[source + ":gain_over_A0"]["macro_difference"] > 0,
                    "exceeds_Hcal_shrink_mean": comparisons[source + ":gain_over_Hcal_shrink"]["macro_difference"] > 0,
                    "exceeds_halfH_mean": comparisons[source + ":gain_over_halfH"]["macro_difference"] > 0,
                    "exceeds_threequarterH_mean": comparisons[source + ":gain_over_threequarterH"]["macro_difference"] > 0,
                    "all_six_source_fits_selected_nonzero_step": all(f["selected_step"] > 0 for f in fits if f["arm"] == source)}
        sources[source] = {"source_line": contract["design"]["line_ids"][source], "descriptive_candidate_checks": criteria,
                           "all_checks_pass": all(criteria.values()),
                           "shrink_diagnostic_nonzero_in_all_seeds_and_lines": all(f["selected_shrink"] > 0 and f["selected_step"] > 0 for f in fits if f["arm"] == source),
                           "extra_human_gain_denominator": human_gain,
                           "fraction_of_positive_extra_human_gain": None if human_gain <= 0 else comparisons[source + ":gain_over_H"]["macro_difference"] / human_gain}
    summary = {"scope": contract["scope"], "biological_target_N": 2, "sampling_seeds": list(SEEDS),
               "aggregation": "scoreeachseed; average3seeds WITHINeachhumanline; average2line means; source56/82 remainseparate",
               "seed_predictions_ensembled_before_scoring": False, "variants": summaries,
               "paired_comparisons": comparisons, "source_results_in_fixed_order": sources,
               "extra_human_control": comparisons["extra_human_gain"],
               "inference_limits": "Descriptive N=2line benchmark conditional on author-filteredcohort. Seeds/cells/captures are not additional biologicalN. No pvalues, confirmatory confidenceintervals, bootstrap significance, noninferiority orbest-source selection."}
    return summary, line_rows, paired_rows


def csv_out(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "prepared", "splits", "run", "out", "code-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    contract, manifest, splits, fits, marker, provenance = global_gate(args)
    if args.verify_only:
        print(json.dumps(provenance, indent=2)); return
    require(not args.out.exists(), "Preserve existing results; use new audit output directory")
    energy = NativeEnergy(args.device)
    # First assay-array opening occurs after every frozen prediction hash passed.
    data = load_npz(args.prepared / contract["data"]["arrays_file"])
    records, fit_audit = audit_and_score(args, contract, splits, fits, marker, data, energy)
    summary, line_rows, paired_rows = summarize(records, fits, contract)
    provenance.update({"scorer_sha256": sha256(__file__), "energy": energy.metadata,
                       "fit_scale_calibration_audit": fit_audit,
                       "baseline_reconstruction": "C/P andalpha-shrink float32; frozenbase_delta float64 retained forC+delta/clamp; allmetrics float64",
                       "upstream_public_cohort_limitation": contract["scope"]["upstream_eligibility_limitation"],
                       "numpy_version": np.__version__})
    args.out.mkdir(parents=True)
    csv_out(args.out / "line_seed_scores.csv", records)
    csv_out(args.out / "line_mean_scores.csv", line_rows)
    csv_out(args.out / "paired_line_seed_differences.csv", paired_rows)
    save(args.out / "summary.json", summary)
    save(args.out / "audit_provenance.json", provenance)
    hashes = {p.name: sha256(p) for p in args.out.iterdir() if p.is_file()}
    save(args.out / "AUDIT_COMPLETE.json", {"contract_sha256": CONTRACT_SHA, "scorer_sha256": sha256(__file__),
         "outputs": hashes, "fits": 36, "line_seed_variant_rows": len(records), "biological_target_N": 2,
         "global_freeze_sha256": provenance["global_freeze_sha256"], "unix_time": time.time()})
    print(json.dumps({"output": str(args.out), "biological_target_N": 2,
                      "sources": summary["source_results_in_fixed_order"]}, indent=2))


if __name__ == "__main__":
    main()
