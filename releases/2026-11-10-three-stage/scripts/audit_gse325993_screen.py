#!/usr/bin/env python3
"""Independent exp3 scoring of all36 frozen exp2 models; no adaptation.

All old/new hash chains and every metadata mapping are checked before loading
the separate415-cell POU5F1 truth array. Reuse only pinned pure metric helpers.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_gse325993_state as metric_helpers

SCREEN_CONTRACT_SHA = "c0e51cbe559ed22e483d0323f344adfa211b2294acf606b01b37a1386c8058ef"
METRIC_HELPER_SHA = "8ebdcc1cd681eb0a701de22622ff89c952ae6ccf329d5fe2166dd114f14e36e3"
ARMS = ("H", "Hextra_cells", "Monkey56", "Monkey56A0", "Monkey82", "Monkey82A0")
SEEDS = (20261110, 20261111, 20261112)
SOURCES = ("Monkey56", "Monkey82")
METRICS = ("mean_response_mse", "native_energy", "response_pearson", "response_cosine")
require, sha256, near = metric_helpers.require, metric_helpers.sha256, metric_helpers.near


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def safe_path(root, relative):
    require(not Path(relative).is_absolute(), "Expected relative frozen artifact path")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), "Frozen artifact escapes its root")
    return result


def folder_name(fold, seed, arm):
    return f"fold_{fold}_seed_{seed}_{arm}"


def load_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        require(not any("expression" in k for k in archive.files), "Metadata archive contains expression")
        return {k: archive[k].copy() for k in archive.files}


def read_factors(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in ("native_gene_indices", "donor_line", "selected_shrink", "base_mean_delta")}


def global_gate(args):
    """Validate byte hashes and old small factors; no exp3 expression is loaded."""
    require(sha256(args.contract) == SCREEN_CONTRACT_SHA, "Reviewed screen contract SHA mismatch")
    contract = read(args.contract); old = contract["frozen_exp2"]; binding = contract["data"]
    require(tuple(contract["design"]["arms"]) == ARMS and tuple(contract["design"]["seeds"]) == SEEDS, "Wrong model grid")
    require(contract["design"]["new_fits"] == contract["design"]["adaptation_steps"] == 0, "Adaptation is outside contract")
    old_root = Path(old["run_root"]).resolve()
    require(args.exp2_run.resolve() == old_root and args.prepared.resolve() == Path(binding["prepared_root"]).resolve(), "Bound run/input root differs")
    old_paths = {key: old_root / old[key + "_file"] for key in ("global_freeze", "fits", "specification")}
    for key, path in old_paths.items():
        require(sha256(path) == old[key + "_sha256"], "Original exp2 manifest changed: " + key)
    old_marker, old_fits, old_spec = [read(old_paths[k]) for k in ("global_freeze", "fits", "specification")]
    require(old_marker["fits_selected_sha256"] == old["fits_sha256"] and old_fits["specification_sha256"] == old["specification_sha256"], "Original hash chain differs")
    require(old_marker["contract_sha256"] == old_spec["contract_sha256"] == old["contract_sha256"], "Original contract differs")
    require(old_spec["code"] == old["code_files"], "Original code declarations differ")
    for relative, digest in old["code_files"].items():
        require(sha256(safe_path(Path(old["code_root"]), relative)) == digest, "Original actual code snapshot changed")
    runner_key = "scripts/run_gse325993_state.py"
    require(old_spec["split_manifest"]["code_sha256"] == old_spec["code"][runner_key], "Old split/training runner binding differs")
    grid = {(f, s, a) for f in (0, 1) for s in SEEDS for a in ARMS}
    fits = old_fits["fits"]
    require(len(fits) == 36 and {(f["fold"], f["seed"], f["arm"]) for f in fits} == grid, "Original36fit grid incomplete")
    factors = {}
    for fit in fits:
        key = fit["fold"], fit["seed"], fit["arm"]; rel = folder_name(*key)
        folder = safe_path(old_root, rel)
        require(read(folder / "training.json") == fit, "Old training/selection record changed")
        require(sha256(folder / "selected.pt") == fit["selected_weight_sha256"], "Old model weights changed")
        factor_path = folder / "target.npz"
        require(sha256(factor_path) == old_marker["artifacts"][rel + "/target.npz"], "Old frozen factor archive changed")
        factor = read_factors(factor_path)
        require(float(factor["selected_shrink"]) == fit["selected_shrink"], "Old alpha changed")
        require(str(factor["donor_line"].item()) == fit["test_line"], "Old target line changed")
        require(factor["base_mean_delta"].dtype == np.float64 and factor["base_mean_delta"].shape == (6357,), "Invalid old mean-response factor")
        factors[key] = factor
    require(all(np.array_equal(factors[next(iter(factors))]["native_gene_indices"], x["native_gene_indices"]) for x in factors.values()), "Old gene masks differ")
    for name, digest in binding["files"].items():
        require(sha256(safe_path(args.prepared, name)) == digest, "New input artifact changed: " + name)
    marker_path = args.run / "ALL_SCREEN_PREDICTIONS_FROZEN.json"
    require(marker_path.exists(), "Freeze all36new predictions before scoring")
    marker = read(marker_path)
    require(marker["models"] == 36 and marker["target_units"] == 2 and marker["contract_sha256"] == SCREEN_CONTRACT_SHA, "Wrong new global freeze scope")
    require(marker["original_exp2_freeze_sha256"] == old["global_freeze_sha256"], "Wrong original model freeze")
    require(marker["target_outcomes_scored"] is False and marker["target_expression_arrays_opened"] is False and marker["parameter_updates"] == 0, "New prediction phase violated isolation")
    require(sha256(args.run / "specification.json") == marker["specification_sha256"], "New specification changed")
    require(sha256(args.run / "prediction_records.json") == marker["records_sha256"], "New prediction record manifest changed")
    spec = read(args.run / "specification.json"); records = read(args.run / "prediction_records.json")
    require(spec["contract"] == contract and spec["contract_sha256"] == SCREEN_CONTRACT_SHA, "New run contract differs")
    require(spec["parameter_updates"] == 0 and spec["calibration_or_alpha_selection"] is False and spec["exp3_target_expression_arrays_opened"] is False, "New run adapted or read target expression")
    require(spec["exp2_global_freeze_sha256"] == old["global_freeze_sha256"] and spec["exp2_fits_sha256"] == old["fits_sha256"] and spec["exp2_specification_sha256"] == old["specification_sha256"], "New run original bindings differ")
    for key in ("manifest", "query_expression", "query_metadata", "cohort_metadata"):
        require(spec["input_hashes"][key] == binding[key + "_sha256"], "New predictor input binding differs: " + key)
    require(sha256(safe_path(args.code_root, "scripts/predict_gse325993_screen.py")) == spec["code_sha256"], "Actual new predictor snapshot changed")
    for relative, digest in spec["interface_code_sha256"].items():
        require(digest == old["code_files"][relative] and sha256(safe_path(args.code_root, relative)) == digest, "Actual inference interface changed")
    helper_path = Path(metric_helpers.__file__).resolve()
    require(helper_path == safe_path(args.code_root, "scripts/audit_gse325993_state.py"), "Metric helper loaded outside reviewed snapshot")
    require(sha256(helper_path) == METRIC_HELPER_SHA, "Pinned pure metric helper changed")
    expected_paths = {folder_name(*key) + "/" + filename for key in grid for filename in ("prediction.npy", "metadata.npz", "record.json")}
    require(set(marker["artifacts"]) == expected_paths, "New36prediction/metadata/record grid incomplete")
    for relative, digest in marker["artifacts"].items():
        require(sha256(safe_path(args.run, relative)) == digest, "Frozen screen artifact changed: " + relative)
    require(len(records) == 36 and {(r["fold"], r["seed"], r["arm"]) for r in records} == grid, "Prediction record grid incomplete")
    old_lookup = {(f["fold"], f["seed"], f["arm"]): f for f in fits}
    for record in records:
        key = record["fold"], record["seed"], record["arm"]; relative = folder_name(*key)
        fit = old_lookup[key]; folder = safe_path(args.run, relative)
        require(read(folder / "record.json") == record, "Per-fit/global prediction record differs")
        require(record["original_weight_sha256"] == fit["selected_weight_sha256"], "Prediction used wrong old model")
        require(Path(record["original_weight_path"]).resolve() == old_root / relative / "selected.pt", "Prediction weight path differs")
        require(record["original_factors_sha256"] == old_marker["artifacts"][relative + "/target.npz"], "Prediction used wrong old factors")
        require(record["selected_alpha_from_exp2"] == fit["selected_shrink"] and record["selected_step_from_exp2"] == fit["selected_step"], "Model/alpha reselection")
        require(record["test_line"] == fit["test_line"] and record["parameter_updates"] == 0 and record["target_expression_read"] is False, "Prediction line/isolation differs")
        require(record["strict_missing_keys"] == record["strict_unexpected_keys"] == [], "Model was not strictly loaded")
        require(record["prediction_sha256"] == marker["artifacts"][relative + "/prediction.npy"] and record["metadata_sha256"] == marker["artifacts"][relative + "/metadata.npz"], "Prediction record SHA differs")
    provenance = {"contract_sha256": SCREEN_CONTRACT_SHA, "old_exp2_global_freeze_sha256": old["global_freeze_sha256"],
                  "new_global_freeze_sha256": sha256(marker_path), "old_models_verified": 36, "old_factor_archives_verified": 36,
                  "new_artifacts_verified": 108, "actual_old_and_new_code_verified": True,
                  "exp3_expression_loaded_during_hash_gate": False, "metric_helper_sha256": METRIC_HELPER_SHA,
                  "verification_unix_time": time.time(), "source_weight_and_alpha_updates": 0}
    return contract, records, fits, factors, marker, provenance


def validate_metadata(args, contract, records, factors):
    """Check every full/query/target coordinate before opening POU5F1 values."""
    b = contract["data"]
    cohort = load_metadata(args.prepared / b["cohort_metadata_file"])
    query = load_metadata(args.prepared / b["query_metadata_file"])
    target = load_metadata(args.prepared / b["target_metadata_file"])
    expected_ix = factors[next(iter(factors))]["native_gene_indices"]
    require(len(cohort["cell_ids"]) == 12136 and np.array_equal(cohort["full_row_indices"], np.arange(12136)), "Full-cohort coordinates differ")
    for m in (cohort, query, target):
        require(len(set(m["cell_ids"].astype(str))) == len(m["cell_ids"]), "Repeated metadata cell IDs")
        require(np.array_equal(m["native_gene_indices"], expected_ix), "Exp3/native6357mask differs")
        require(set(m["experiment"].astype(str)) == {"exp3"}, "Wrong experiment")
    for m, cond, n in ((query, "NTC", 11721), (target, "POU5F1", 415)):
        rows = np.flatnonzero(cohort["condition"].astype(str) == cond)
        require(len(m["cell_ids"]) == n and np.array_equal(m["full_row_indices"], rows), "Subset/full-cohort row mapping differs")
        for name in ("cell_ids", "cell_lines", "condition", "capture", "guide", "experiment"):
            require(np.array_equal(m[name].astype(str), cohort[name][rows].astype(str)), "Subset metadata mismatch: " + name)
    require(not set(query["full_row_indices"]) & set(target["full_row_indices"]), "NTC and POU5F1 overlap")
    require(len(set(query["capture"].astype(str))) == 23 and "exp3_chip3_lane8" not in set(cohort["capture"].astype(str)), "Capture inclusion differs")
    mappings = {}
    for line, counts in contract["design"]["known_eligible_counts"].items():
        qr = np.flatnonzero(query["cell_lines"].astype(str) == line)
        tr = np.flatnonzero(target["cell_lines"].astype(str) == line)
        require(len(qr) == counts["NTC"] and len(tr) == counts["POU5F1"], "Line-specific whole population budget differs")
        mappings[line] = {"query_rows": qr, "target_subset_rows": tr,
                          "control_full_rows": query["full_row_indices"][qr], "target_full_rows": target["full_row_indices"][tr]}
    require(set(cohort["cell_lines"].astype(str)) == set(mappings), "Unexpected target line")
    for r in records:
        key = r["fold"], r["seed"], r["arm"]; folder = args.run / folder_name(*key)
        m = load_metadata(folder / "metadata.npz"); line = str(m["donor_line"].item())
        require(line == r["test_line"] and line in mappings, "Prediction metadata line differs")
        expected = mappings[line]
        for field, value in (("query_rows", expected["query_rows"]), ("control_rows", expected["control_full_rows"]), ("target_rows", expected["target_full_rows"])):
            require(np.array_equal(m[field], value), "Prediction full-to-subset mapping differs: " + field)
        require(np.array_equal(m["control_cell_ids"].astype(str), query["cell_ids"][expected["query_rows"]].astype(str)), "Query cell IDs differ")
        require(np.array_equal(m["target_cell_ids"].astype(str), target["cell_ids"][expected["target_subset_rows"]].astype(str)), "POU5F1 target cell IDs differ")
        require(np.array_equal(m["native_gene_indices"], expected_ix), "Prediction gene axis differs")
        require(float(m["selected_shrink"]) == float(factors[key]["selected_shrink"]), "New alpha factor differs")
        require(np.array_equal(m["base_mean_delta"], factors[key]["base_mean_delta"]), "New base-mean factor differs")
        require(r["prediction_shape"] == [len(expected["query_rows"]), 6357] and r["dtype"] == "float32" and r["uncompressed_npy"] is True, "Prediction storage scope differs")
    return mappings, query, target


def metric_values(prediction, target, control, energy):
    p, t, c = [np.asarray(x, dtype=np.float64) for x in (prediction, target, control)]
    delta, truth = p.mean(0) - c.mean(0), t.mean(0) - c.mean(0)
    error = metric_helpers.mse(p, t)
    near(error, float(np.square(delta - truth).mean()), "Mean-response MSE identity failed")
    return {"mean_response_mse": error, "native_energy": energy(p, t),
            "response_pearson": metric_helpers.corr(delta, truth, True), "response_cosine": metric_helpers.corr(delta, truth),
            "predicted_response_l2": float(np.linalg.norm(delta)), "true_response_l2": float(np.linalg.norm(truth))}


def score_predictions(args, contract, records, factors, marker, mappings):
    b = contract["data"]
    energy = metric_helpers.NativeEnergy(args.device)
    query = np.load(args.prepared / b["query_expression_file"], mmap_mode="r", allow_pickle=False)
    # First POU5F1 expression opening occurs after all hash AND metadata gates.
    target = np.load(args.prepared / b["target_expression_file"], mmap_mode="r", allow_pickle=False)
    require(query.dtype == target.dtype == np.float32 and query.shape == (11721, 6357) and target.shape == (415, 6357), "Query/target expression dimensions differ")
    require(np.isfinite(query).all() and np.isfinite(target).all() and np.min(query) >= 0 and np.min(target) >= 0, "Invalid normalized query/target")
    result, seen = [], set()
    for r in records:
        key = r["fold"], r["seed"], r["arm"]; relative = folder_name(*key) + "/prediction.npy"
        require(sha256(safe_path(args.run, relative)) == marker["artifacts"][relative], "P changed after global freeze verification")
        p = np.load(args.run / relative, mmap_mode="r", allow_pickle=False)
        line = r["test_line"]; mapping = mappings[line]
        c = np.array(query[mapping["query_rows"]], copy=True)
        # Explicit target-local rows, NEVER full-cohort indices on415-row array.
        t = np.array(target[mapping["target_subset_rows"]], copy=True)
        require(p.dtype == np.float32 and p.shape == c.shape and np.isfinite(p).all() and np.min(p) >= 0, "Invalid frozen P")
        f = factors[key]; alpha = float(f["selected_shrink"])
        variants = [(r["arm"], p), (r["arm"] + "|cal_shrink", c + np.float32(alpha) * (p - c))]
        if r["arm"] == "H":
            baselines = metric_helpers.reconstruct_baselines(c, p, alpha, f["base_mean_delta"])
            variants.extend([("Zero_response", baselines["zero_response"]), ("Hbase_mean_response", baselines["base_mean_response"]),
                             ("Half_H_response", baselines["half_response"]), ("Three_quarter_H_response", baselines["three_quarter_response"])])
        for variant, prediction in variants:
            identity = line, r["seed"], variant
            require(identity not in seen, "Repeated line-seed-variant")
            seen.add(identity)
            row = {"target_line": line, "fold": r["fold"], "seed": r["seed"], "variant": variant,
                   "query_cells": len(c), "target_cells": len(t), "genes": 6357,
                   "selected_step_from_exp2": r["selected_step_from_exp2"], "selected_alpha_from_exp2": alpha,
                   "model_weight_sha256": r["original_weight_sha256"], "prediction_sha256": r["prediction_sha256"]}
            row.update(metric_values(prediction, t, c, energy)); result.append(row)
        print(json.dumps({"phase": "screen_scores", "fold": r["fold"], "seed": r["seed"], "arm": r["arm"]}), flush=True)
    require(len(result) == 96, "Missing full96line-seed-variant scores")
    return result, energy.metadata


def aggregate(rows):
    out = {}
    for k in METRICS:
        values = [r[k] for r in rows]
        out[k] = None if any(v is None for v in values) else float(np.mean(values))
        out[k + "_defined_count"] = sum(v is not None for v in values)
    return out


def summarize(rows, contract):
    lines = contract["design"]["human_lines"]; variants = sorted({r["variant"] for r in rows})
    lookup = {(r["target_line"], r["seed"], r["variant"]): r for r in rows}
    summaries, line_rows = {}, []
    for v in variants:
        per_line = []
        for line in lines:
            rr = [lookup[line, seed, v] for seed in SEEDS]
            item = {"target_line": line, "variant": v, **aggregate(rr),
                    "seed_mse_min": min(r["mean_response_mse"] for r in rr), "seed_mse_max": max(r["mean_response_mse"] for r in rr)}
            per_line.append(item); line_rows.append(item)
        summaries[v] = {"per_line": per_line, "macro": aggregate(per_line)}
    requests = [("extra_human_gain", "H", "Hextra_cells")]
    for source in SOURCES:
        for label, comparator in (("H", "H"), ("extra_human", "Hextra_cells"), ("A0", source + "A0"),
                                  ("Hcal_shrink", "H|cal_shrink"), ("zero", "Zero_response"), ("base_mean", "Hbase_mean_response"),
                                  ("halfH", "Half_H_response"), ("threequarterH", "Three_quarter_H_response")):
            requests.append((source + ":gain_over_" + label, comparator, source))
        requests += [(source + ":shrunk_gain_over_H", "H|cal_shrink", source + "|cal_shrink"),
                     (source + ":shrunk_gain_over_A0", source + "A0|cal_shrink", source + "|cal_shrink")]
    comparisons, paired = {}, []
    for name, left, right in requests:
        line_differences = []
        for line in lines:
            diffs = []
            for seed in SEEDS:
                l, r = lookup[line, seed, left], lookup[line, seed, right]
                difference = l["mean_response_mse"] - r["mean_response_mse"]; diffs.append(difference)
                paired.append({"comparison": name, "target_line": line, "seed": seed, "left": left, "right": right,
                               "left_mse": l["mean_response_mse"], "right_mse": r["mean_response_mse"], "left_minus_right_mse": difference})
            line_differences.append({"target_line": line, "seed_mean_difference": float(np.mean(diffs)),
                                     "positive_seeds": sum(x > 0 for x in diffs), "seed_differences": diffs})
        difference = float(np.mean([x["seed_mean_difference"] for x in line_differences])); comparator = summaries[left]["macro"]["mean_response_mse"]
        comparisons[name] = {"left": left, "right": right, "definition": "left minus right MSE; positive favorsright",
                             "per_line": line_differences, "macro_difference": difference,
                             "positive_target_lines": sum(x["seed_mean_difference"] > 0 for x in line_differences),
                             "percent_of_comparator": None if comparator == 0 else 100 * difference / comparator}
    extra_gain = comparisons["extra_human_gain"]["macro_difference"]
    source_results = {s: {"both_lines_improve_over_H": comparisons[s + ":gain_over_H"]["positive_target_lines"] == 2,
                         "mean_gain_over_H": comparisons[s + ":gain_over_H"]["macro_difference"],
                         "mean_gain_over_A0": comparisons[s + ":gain_over_A0"]["macro_difference"],
                         "mean_gain_over_threequarterH": comparisons[s + ":gain_over_threequarterH"]["macro_difference"],
                         "extra_human_gain_denominator": extra_gain,
                         "fraction_of_positive_extra_human_gain": None if extra_gain <= 0 else comparisons[s + ":gain_over_H"]["macro_difference"] / extra_gain}
                      for s in SOURCES}
    summary = {"scope": contract["scope"], "validation_experiment": "exp3", "exp2_outcomes_were_known": True,
               "parameter_updates": 0, "biological_target_N": 2, "inherited_seeds": list(SEEDS),
               "aggregation": "scoreeachmodel; average3seedmetricswithin eachhumanline, then2line means; source56/82separate",
               "variants": summaries, "paired_comparisons": comparisons, "source_results_in_fixed_order": source_results,
               "inference_limits": contract["interpretation"], "p_values": None, "confidence_intervals": None}
    return summary, line_rows, paired


def csv_out(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "prepared", "exp2-run", "run", "code-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    contract, records, fits, factors, marker, provenance = global_gate(args)
    mappings, query_metadata, target_metadata = validate_metadata(args, contract, records, factors)
    provenance["full_to_query_and_415_target_mapping_verified_before_target_expression"] = True
    if args.verify_only:
        print(json.dumps(provenance, indent=2)); return
    require(not args.out.exists(), "Use a new audit output directory")
    rows, energy = score_predictions(args, contract, records, factors, marker, mappings)
    summary, line_rows, paired = summarize(rows, contract)
    provenance.update({"scorer_sha256": sha256(__file__), "energy": energy, "numpy_version": np.__version__,
                       "old_base_delta_and_alpha_recomputed_or_reselected": False})
    args.out.mkdir(parents=True)
    csv_out(args.out / "line_seed_scores.csv", rows); csv_out(args.out / "line_mean_scores.csv", line_rows)
    csv_out(args.out / "paired_line_seed_differences.csv", paired)
    save(args.out / "summary.json", summary); save(args.out / "audit_provenance.json", provenance)
    save(args.out / "AUDIT_COMPLETE.json", {"contract_sha256": SCREEN_CONTRACT_SHA, "scorer_sha256": sha256(__file__),
         "models": 36, "biological_target_N": 2, "score_rows": 96, "new_global_freeze_sha256": provenance["new_global_freeze_sha256"],
         "outputs": {f.name: sha256(f) for f in args.out.iterdir() if f.is_file()}, "unix_time": time.time()})
    print(json.dumps({"audit": str(args.out), "models": 36, "biological_target_N": 2, "adaptation_steps": 0}))


if __name__ == "__main__":
    main()
