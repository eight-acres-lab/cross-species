#!/usr/bin/env python3
"""Apply every frozen exp2 State model to exp3 NTC queries, then seal outputs.

This is an inference-only entry point. It never opens exp3 target expression,
scores an outcome, selects a model, adjusts an alpha, or updates parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP2_FREEZE_SHA = "25f88bed76ef7fbb84056658424ece4c0f4f5283c387b02533bf1f111474ca33"
EXP2_FITS_SHA = "f2d05b411272743402445d13790c36f8af8641e01f01b62f12f5b2a68e46d06c"
EXP2_SPEC_SHA = "cefed6002669c7355b5aab53d3e85487bf0ca282d046556faf077b5fe02c0b65"
EXP2_CONTRACT_SHA = "f69e4a440bd35d1ea66b727e5f41fa02e7e412c3b5986b1a6331a883dfc5e9fd"
ARMS = ("H", "Hextra_cells", "Monkey56", "Monkey56A0", "Monkey82", "Monkey82A0")
SEEDS = (20261110, 20261111, 20261112)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def chunks(n, size=64):
    return [(start, min(start + size, n)) for start in range(0, n, size)]


def line_rows(metadata, line):
    rows = np.flatnonzero(metadata["cell_lines"].astype(str) == line)
    require(len(rows) > 0, "Query line has no NTC cells: " + line)
    return rows


def verify_old_models(exp2_run):
    """Verify the complete old36model grid before applying any one model."""
    files = {"ALL_TARGET_PREDICTIONS_FROZEN.json": EXP2_FREEZE_SHA,
             "ALL_FITS_SELECTED.json": EXP2_FITS_SHA, "specification.json": EXP2_SPEC_SHA}
    for name, digest in files.items():
        require(sha256(exp2_run / name) == digest, "Original exp2 freeze changed: " + name)
    marker, fits, spec = [read(exp2_run / name) for name in files]
    require(marker["fits_selected_sha256"] == EXP2_FITS_SHA and fits["specification_sha256"] == EXP2_SPEC_SHA,
            "Original freeze chain is inconsistent")
    require(spec["contract_sha256"] == marker["contract_sha256"] == EXP2_CONTRACT_SHA, "Original exp2 contract differs")
    require(len(fits["fits"]) == marker["fits"] == len(marker["artifacts"]) == 36, "Original freeze is not complete")
    expected = {(fold, seed, arm) for fold in (0, 1) for seed in SEEDS for arm in ARMS}
    observed = [(f["fold"], f["seed"], f["arm"]) for f in fits["fits"]]
    require(len(set(observed)) == 36 and set(observed) == expected, "Do not omit or select exp2 models")
    prepared = []
    for fit in fits["fits"]:
        folder = exp2_run / f"fold_{fit['fold']}_seed_{fit['seed']}_{fit['arm']}"
        weight = folder / "selected.pt"
        old_target = folder / "target.npz"
        require(sha256(weight) == fit["selected_weight_sha256"], "Original selected weights changed")
        factor_hash = sha256(old_target)
        require(factor_hash == marker["artifacts"][str(old_target.relative_to(exp2_run))], "Original frozen factors changed")
        require(read(folder / "training.json") == fit, "Original fit record differs from global selection")
        # Only these small frozen factors are loaded; original prediction arrays
        # and original target RNA values are not read into the inference model.
        with np.load(old_target, allow_pickle=False) as z:
            alpha = float(z["selected_shrink"])
            delta = z["base_mean_delta"].copy()
            ix = z["native_gene_indices"].copy()
            line = str(z["donor_line"].item())
        require(alpha == fit["selected_shrink"] and alpha in (0, .25, .5, .75, 1), "Frozen alpha differs")
        require(delta.dtype == np.float64 and delta.shape == (6357,) and np.isfinite(delta).all(), "Invalid frozen training-only mean response")
        require(ix.dtype == np.int64 and ix.shape == (6357,) and np.all(np.diff(ix) > 0), "Invalid frozen native mask")
        require(line == fit["test_line"] == spec["contract"]["design"]["outer_folds"][fit["fold"]]["test_line"], "Held-out line assignment changed")
        prepared.append({"fit": fit, "folder": folder, "weight": weight, "alpha": alpha, "delta": delta,
                         "indices": ix, "line": line, "old_factors_sha256": factor_hash})
    require(all(np.array_equal(prepared[0]["indices"], p["indices"]) for p in prepared), "Exp2 model axes differ")
    return spec, prepared


def verified_query_inputs(args, contract, expected_indices):
    """Open the independent query-only expression artifact and metadata only."""
    binding = contract["data"]
    require(args.prepared.resolve() == Path(binding["prepared_root"]).resolve(), "Prepared query root differs from contract")
    names = {
        "manifest": (binding["manifest_file"], binding["manifest_sha256"]),
        "query_expression": (binding["query_expression_file"], binding["query_expression_sha256"]),
        "query_metadata": (binding["query_metadata_file"], binding["query_metadata_sha256"]),
        "cohort_metadata": (binding["cohort_metadata_file"], binding["cohort_metadata_sha256"]),
    }
    paths, hashes = {}, {}
    for key, (name, expected) in names.items():
        require(Path(name).name == name and isinstance(expected, str), "Unfinalized or unsafe query binding")
        path = args.prepared / name
        require(sha256(path) == expected, "Query artifact hash mismatch: " + name)
        paths[key], hashes[key] = path, expected
    with np.load(paths["query_metadata"], allow_pickle=False) as z:
        query = {key: z[key].copy() for key in z.files}
    with np.load(paths["cohort_metadata"], allow_pickle=False) as z:
        cohort = {key: z[key].copy() for key in z.files}
    for meta in (query, cohort):
        require("expression" not in meta and "human_expression" not in meta, "Metadata artifact contains expression values")
        require(np.array_equal(meta["native_gene_indices"], expected_indices), "Screen axis differs from frozen6357training coordinates")
        require(len(set(meta["cell_ids"].astype(str))) == len(meta["cell_ids"]), "Duplicate screen cell IDs")
        require(set(meta["experiment"].astype(str)) == {"exp3"}, "Wrong validation screen")
    require(set(query["condition"].astype(str)) == {"NTC"}, "Query artifact must contain NTC expression only")
    full_rows = query["full_row_indices"]
    require(full_rows.dtype.kind in "iu" and len(set(full_rows.tolist())) == len(full_rows), "Query-to-cohort rows are invalid")
    require(np.array_equal(cohort["full_row_indices"], np.arange(len(cohort["cell_ids"]))), "Cohort row coordinates changed")
    expected_query = np.flatnonzero(cohort["condition"].astype(str) == "NTC")
    require(np.array_equal(full_rows, expected_query), "Use all NTC cells in stored cohort order")
    for key in ("cell_ids", "cell_lines", "condition", "guide", "capture", "experiment"):
        require(np.array_equal(query[key].astype(str), cohort[key][full_rows].astype(str)), "Query/cohort identity mismatch: " + key)
    require(set(cohort["condition"].astype(str)) == {"NTC", "POU5F1"}, "Unexpected validation conditions")
    require(len(cohort["cell_ids"]) == contract["design"]["human_cells"] == 12136 and
            len(query["cell_ids"]) == contract["design"]["NTC_query_cells"] == 11721, "Public screen population changed")
    require(len(set(cohort["capture"].astype(str))) == contract["design"]["public_capture_count"] == 23, "Public capture set changed")
    require("exp3_chip3_lane8" not in set(cohort["capture"].astype(str)), "Undeposited capture was added")
    for line, counts in contract["design"]["known_eligible_counts"].items():
        for condition, count in counts.items():
            require(int(np.sum((cohort["cell_lines"].astype(str) == line) & (cohort["condition"].astype(str) == condition))) == count,
                    "Screen line/condition count changed")
    values = np.load(paths["query_expression"], mmap_mode="r", allow_pickle=False)
    require(values.dtype == np.float32 and values.shape == (len(query["cell_ids"]), 6357), "Query expression shape/dtype differs")
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid normalized NTC expression")
    return values, query, cohort, hashes


def run(args):
    require(platform.system() == "Linux", "Run real screen inference on the remote Linux runtime")
    require(not args.out.exists(), "Use a new screen prediction directory")
    require(sha256(args.contract) == args.contract_sha256, "Reviewed screen contract hash mismatch")
    contract = read(args.contract)
    require(args.exp2_run.resolve() == Path(contract["frozen_exp2"]["run_root"]).resolve(), "Original run path differs")
    old_binding = contract["frozen_exp2"]
    require(old_binding["global_freeze_sha256"] == EXP2_FREEZE_SHA and old_binding["fits_sha256"] == EXP2_FITS_SHA
            and old_binding["specification_sha256"] == EXP2_SPEC_SHA and old_binding["contract_sha256"] == EXP2_CONTRACT_SHA,
            "Screen contract identifies a different original freeze")
    require(tuple(contract["design"]["arms"]) == ARMS and tuple(contract["design"]["seeds"]) == SEEDS, "Screen model grid differs")
    require(contract["design"]["validation_experiment"] == "exp3", "Wrong validation experiment")
    require(contract["design"]["models"] == 36 and contract["design"]["new_fits"] == 0
            and contract["design"]["adaptation_steps"] == 0, "Only the complete frozen inference grid is supported")
    old_spec, models = verify_old_models(args.exp2_run)
    require(old_binding["code_files"] == old_spec["code"], "Declared original code hashes differ from the frozen specification")
    old_code_root = Path(old_binding["code_root"]).resolve()
    for relative, digest in old_spec["code"].items():
        code_path = (old_code_root / relative).resolve()
        require(code_path.is_relative_to(old_code_root), "Original code path escapes the frozen snapshot")
        require(sha256(code_path) == digest, "Actual original code snapshot changed: " + relative)
    require(sha256(args.checkpoint) == old_spec["checkpoint_sha256"], "Original pretrained checkpoint differs")
    for relative in ("src/cross_species/state_assay_adapter.py", "src/cross_species/named_gene_state.py"):
        require(sha256(ROOT / relative) == old_spec["code"][relative], "Frozen model interface changed: " + relative)
    indices = models[0]["indices"]
    query_values, query, cohort, input_hashes = verified_query_inputs(args, contract, indices)
    expected_lines = {m["line"] for m in models}
    require(set(query["cell_lines"].astype(str)) == set(cohort["cell_lines"].astype(str)) == expected_lines, "Screen human line set differs")
    query_rows = {line: line_rows(query, line) for line in expected_lines}
    target_rows = {line: np.flatnonzero((cohort["cell_lines"].astype(str) == line) & (cohort["condition"].astype(str) == "POU5F1")) for line in expected_lines}
    require(all(len(rows) > 0 for rows in target_rows.values()), "A screen target line lacks POU5F1 metadata")
    prediction_bytes = sum(len(query_rows[m["line"]]) * 6357 * 4 for m in models)
    require(prediction_bytes == contract["storage"]["expected_P_data_bytes_without_headers"], "Prediction storage budget differs")
    disk_root = args.out.parent
    while not disk_root.exists():
        disk_root = disk_root.parent
    free = shutil.disk_usage(disk_root).free
    require(free > prediction_bytes + 1024**3, "Insufficient room for all36uncompressed prediction arrays plus1GiB margin")
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from cross_species import state_assay_adapter as adapter
    require(args.device == "cuda" and torch.cuda.is_available(), "Use reviewed remote GPU runtime")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True)
    specification = {
        "contract": contract, "contract_sha256": args.contract_sha256,
        "exp2_run": str(args.exp2_run.resolve()), "exp2_global_freeze_sha256": EXP2_FREEZE_SHA,
        "exp2_fits_sha256": EXP2_FITS_SHA, "exp2_specification_sha256": EXP2_SPEC_SHA,
        "input_hashes": input_hashes, "code_sha256": sha256(__file__),
        "interface_code_sha256": {p: sha256(ROOT / p) for p in ("src/cross_species/state_assay_adapter.py", "src/cross_species/named_gene_state.py")},
        "prediction_bytes_expected": prediction_bytes, "initial_free_bytes": free,
        "models": 36, "human_lines": sorted(expected_lines), "exp3_target_expression_arrays_opened": False,
        "parameter_updates": 0, "calibration_or_alpha_selection": False, "target_outcomes_scored": False,
    }
    save(args.out / "specification.json", specification)
    artifacts, records = {}, []
    ix = torch.as_tensor(indices, device=args.device)
    mask = torch.zeros(6546, dtype=torch.bool, device=args.device)
    mask[ix] = True
    for item in models:
        fit, line = item["fit"], item["line"]
        folder = args.out / f"fold_{fit['fold']}_seed_{fit['seed']}_{fit['arm']}"
        folder.mkdir()
        native, _ = adapter.load_native_model(args.checkpoint, device="cpu", source_root=args.state_source)
        model = adapter.extend_assay_model(native).to(args.device).eval()
        del native
        state = torch.load(item["weight"], map_location="cpu", weights_only=True)
        status = model.load_state_dict(state, strict=True)
        del state
        model.requires_grad_(False)
        rows = query_rows[line]
        prediction_path = folder / "prediction.npy"
        output = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.float32, shape=(len(rows), 6357))
        started = time.monotonic()
        with torch.no_grad():
            for start, end in chunks(len(rows)):
                observed = torch.as_tensor(np.array(query_values[rows[start:end]], copy=True), device=args.device)
                controls = torch.zeros((end - start, 6546), dtype=torch.float32, device=args.device)
                controls[:, ix] = observed
                prediction = adapter.predict_assay_contrast(model, controls, mask, context="human_iPSC")["prediction"][:, ix]
                require(bool(torch.isfinite(prediction).all()), "Nonfinite frozen-model prediction")
                output[start:end] = prediction.cpu().numpy()
        output.flush()
        del output
        np.savez_compressed(folder / "metadata.npz", query_rows=rows,
                            control_rows=query["full_row_indices"][rows], control_cell_ids=query["cell_ids"][rows],
                            target_rows=target_rows[line], target_cell_ids=cohort["cell_ids"][target_rows[line]],
                            native_gene_indices=indices, donor_line=np.asarray(line),
                            selected_shrink=np.asarray(item["alpha"], dtype=np.float64), base_mean_delta=item["delta"])
        record = {"fold": fit["fold"], "seed": fit["seed"], "arm": fit["arm"], "test_line": line,
                  "selected_step_from_exp2": fit["selected_step"], "selected_alpha_from_exp2": item["alpha"],
                  "original_weight_path": str(item["weight"]), "original_weight_sha256": fit["selected_weight_sha256"],
                  "original_factors_sha256": item["old_factors_sha256"], "query_cells": len(rows),
                  "prediction_shape": [len(rows), 6357], "dtype": "float32", "uncompressed_npy": True,
                  "prediction_sha256": sha256(prediction_path), "metadata_sha256": sha256(folder / "metadata.npz"),
                  "strict_missing_keys": status.missing_keys, "strict_unexpected_keys": status.unexpected_keys,
                  "parameter_updates": 0, "target_expression_read": False, "elapsed_seconds": time.monotonic() - started}
        save(folder / "record.json", record)
        records.append(record)
        for filename in ("prediction.npy", "metadata.npz", "record.json"):
            path = folder / filename
            artifacts[str(path.relative_to(args.out))] = sha256(path)
        print(json.dumps({"phase": "screen_prediction_complete", "fold": fit["fold"], "seed": fit["seed"], "arm": fit["arm"], "cells": len(rows)}), flush=True)
        del model
        torch.cuda.empty_cache()
    require(len(records) == 36, "All36screen predictions required before freeze")
    save(args.out / "prediction_records.json", records)
    save(args.out / "ALL_SCREEN_PREDICTIONS_FROZEN.json", {
        "models": 36, "target_units": 2, "artifacts": artifacts, "contract_sha256": args.contract_sha256,
        "specification_sha256": sha256(args.out / "specification.json"), "records_sha256": sha256(args.out / "prediction_records.json"),
        "original_exp2_freeze_sha256": EXP2_FREEZE_SHA, "target_expression_arrays_opened": False,
        "target_outcomes_scored": False, "parameter_updates": 0, "unix_time": time.time()})
    print(json.dumps({"phase": "all36screen_predictions_frozen", "scores_computed": False, "parameter_updates": 0}))


def self_test():
    require(chunks(130) == [(0, 64), (64, 128), (128, 130)], "Do not pad final inference chunk")
    require(chunks(64) == [(0, 64)] and chunks(1) == [(0, 1)], "Unexpected batch boundary")
    meta = {"cell_lines": np.asarray(["a", "b", "a", "b", "a"])}
    require(np.array_equal(line_rows(meta, "a"), [0, 2, 4]), "Preserve stored query order")
    require(np.array_equal(line_rows(meta, "b"), [1, 3]), "Line filtering changed order")
    alpha, delta = .25, np.array([.1, -.2], dtype=np.float64)
    require(delta.dtype == np.float64 and alpha in (0, .25, .5, .75, 1), "Factors must retain original precision")
    require(len({(f, s, a) for f in (0, 1) for s in SEEDS for a in ARMS}) == 36, "Incomplete model grid")
    print(json.dumps({"artificial_checks": "PASS", "checks": ["un-padded64chunks", "stable line/query ordering", "36-model grid", "factor precision"], "real_data_opened": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("predict")
    for name in ("contract", "exp2-run", "prepared", "checkpoint", "state-source", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--contract-sha256", required=True)
    command.add_argument("--device", default="cuda")
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "predict":
        run(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
