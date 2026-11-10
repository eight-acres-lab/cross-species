#!/usr/bin/env python3
"""Post-hoc State diagnostic with donor-preserving incorrect cytokine pairing.

This preserves the original conditioning-label/PBS schedule and exact multiset
of treated-cell observations. It changes only which same-donor treated population
is associated with each conditioning cytokine during fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.official_state import load_official_state, load_perturbation_map, sha256_file
from cross_species.state_finetuning import fit_arm, evaluate_model
from run_official_state_benchmark import write_json, paired_interval


def mispair_schedule(schedule, *, seed=20261108):
    """A fixed condition-derangement bijection of schedule slots within each donor."""
    result = [{**record, "control_rows": np.array(record["control_rows"]).copy(),
               "target_rows": np.array(record["target_rows"]).copy()} for record in schedule]
    mappings = []
    for donor in sorted({record["donor"] for record in schedule}):
        indices = [index for index, record in enumerate(schedule) if record["donor"] == donor]
        count = len(indices)
        costs = np.empty((count, count), dtype=float)
        for i, destination in enumerate(indices):
            for j, origin in enumerate(indices):
                forbidden = schedule[destination]["condition"] == schedule[origin]["condition"]
                key = f"{seed}|state-mispair|{donor}|{destination}|{origin}"
                tie = int(hashlib.sha256(key.encode()).hexdigest()[:13], 16) / float(16 ** 13)
                costs[i, j] = 1e6 * forbidden + tie
        destinations, origins = linear_sum_assignment(costs)
        if (costs[destinations, origins] >= 1e6).any():
            raise ValueError("No complete wrong-condition derangement within donor " + donor)
        for i, j in zip(destinations, origins):
            destination, origin = indices[i], indices[j]
            result[destination]["target_rows"] = np.array(schedule[origin]["target_rows"]).copy()
            result[destination]["mispaired_target_condition"] = schedule[origin]["condition"]
            result[destination]["target_from_original_step"] = origin + 1
            mappings.append({"step": destination + 1, "target_from_step": origin + 1, "donor": donor,
                "conditioning_cytokine": schedule[destination]["condition"], "actual_target_cytokine": schedule[origin]["condition"]})
    return result, sorted(mappings, key=lambda row: row["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "perturbation-map", "cells-npz", "native-run", "out-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use an empty output directory")
    torch.set_num_threads(args.threads)
    # Match the original State formal FP32 setting; do not inherit flow TF32 flags.
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda": torch.backends.cuda.matmul.allow_tf32 = False
    parent_path = args.native_run/"specification.json"
    parent = json.loads(parent_path.read_text())
    for name, path in [("checkpoint", args.checkpoint), ("perturbation_map", args.perturbation_map), ("cells_npz", args.cells_npz)]:
        if sha256_file(path) != parent["input_hashes"][name]:
            raise ValueError("Input differs from the original State formal run: " + name)
    with np.load(args.cells_npz, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    original = [{**record, "control_rows": np.array(record["control_rows"]), "target_rows": np.array(record["target_rows"])}
                for record in parent["training_schedule"]]
    schedule, mapping = mispair_schedule(original, seed=parent["seed"])
    for old, new in zip(original, schedule):
        if old["condition"] != new["condition"] or not np.array_equal(old["control_rows"], new["control_rows"]):
            raise ValueError("Conditioning/PBS schedule changed")
        if not np.all(payload["donor"][new["target_rows"]].astype(str) == new["donor"]):
            raise ValueError("A target crossed donor strata")
        if np.any(payload["cytokine"][new["target_rows"]].astype(str) == new["condition"]):
            raise ValueError("An accidental correct cytokine/response pairing remains")
        if new["condition"] not in parent["roles"]["train"] or new["mispaired_target_condition"] not in parent["roles"]["train"]:
            raise ValueError("A held-out condition entered the diagnostic")
    original_multiset = np.sort(np.concatenate([row["target_rows"] for row in original]))
    new_multiset = np.sort(np.concatenate([row["target_rows"] for row in schedule]))
    if not np.array_equal(original_multiset, new_multiset):
        raise ValueError("The exact target-cell observation multiset changed")
    lookup = {str(value): index for index, value in enumerate(payload["cell_ids"])}
    populations = {(record["condition"], record["donor"]): {
        "controls": np.array([lookup[value] for value in record["control_cell_ids"]]),
        "targets": np.array([lookup[value] for value in record["target_cell_ids"]])} for record in parent["cell_populations"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specification = {"status": "Post-hoc historical diagnostic; the original correctly paired test scores were already examined before this control was designed.",
        "parent_specification_sha256": sha256_file(parent_path), "parent_roles": parent["roles"],
        "mispairing": "Metadata-hash-tied minimum-cost bijection of schedule entries within each donor; every assigned target cytokine differs from its conditioning label.",
        "retained": "Identical conditioning-label/PBS sequence, exact target-cell observation multiset, 200 updates, native loss, original checkpoint, original FP32 settings, AdamW hyperparameters and seed.",
        "mapping": mapping, "training_schedule": schedule, "actual_target_cell_count": len(np.unique(new_multiset)),
        "input_hashes": parent["input_hashes"], "optimizer": parent["optimizer"], "updates": parent["updates_per_arm"],
        "anchor_weight": 0, "no_parameter_selection": True, "test_conditions_unchanged": parent["roles"]["test"],
        "comparison": "Primary pairing utility is mismatched loss minus correctly paired loss; positive means correct pairing has lower error.",
        "scope": "Tests the utility of correct cytokine/response association within this continuation recipe. It does not isolate the original 512-to64 set-size change or old-data-only continuation.",
        "script_sha256": sha256_file(__file__)}
    write_json(args.out_dir/"specification.json", specification)
    model, metadata = load_official_state(args.checkpoint, device=args.device)
    reference, _ = load_official_state(args.checkpoint, device=args.device)
    representations = load_perturbation_map(args.perturbation_map, model.pert_dim)
    fit = fit_arm(model, reference, payload["expression"], schedule, representations, weight=0,
                  device=args.device, lr=parent["optimizer"]["lr"], seed=parent["seed"])
    write_json(args.out_dir/"fit.json", fit)
    torch.save({"state_dict": {key:value.detach().cpu() for key,value in model.state_dict().items()},
                "specification_sha256": sha256_file(args.out_dir/"specification.json")}, args.out_dir/"mismatched.pt")
    result = evaluate_model(model, payload["expression"], populations, parent["roles"]["test"], parent["donors"], representations,
        device=args.device, cell_count=parent["cell_count_per_population"], bandwidth=parent["rbf_bandwidth_train_only"], save_predictions=True)
    np.savez_compressed(args.out_dir/"test_predictions.npz", predictions=result.pop("predictions"), targets=result.pop("targets"), feature_ids=payload["gene_ids"])
    write_json(args.out_dir/"test_query_indices.json", result.pop("query_indices"))
    previous = json.loads((args.native_run/"report.json").read_text())
    report = {"specification_sha256": sha256_file(args.out_dir/"specification.json"), "status": specification["status"],
              "checkpoint_metadata": metadata, "fit": fit, "mismatched_test": result,
              "original_correct_test": previous["test"]["native_energy_sft"], "released_checkpoint_test": previous["test"]["released_checkpoint"],
              "mismatched_vs_released": paired_interval(previous["test"]["released_checkpoint"], result),
              "correct_vs_mismatched": paired_interval(result, previous["test"]["native_energy_sft"]),
              "correct_vs_mismatched_direction": "Positive gain means correct pairing has lower test native energy than mismatching.",
              "scope": specification["scope"]}
    if args.device == "cuda": report["cuda"] = {"device": torch.cuda.get_device_name(), "runtime": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(args.out_dir/"report.json", report)
    print(json.dumps({"mismatched_test":result["macro"], "correct_vs_mismatched":report["correct_vs_mismatched"]},indent=2),flush=True)


if __name__ == "__main__": main()
