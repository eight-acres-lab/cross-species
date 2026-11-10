#!/usr/bin/env python3
"""Custom State latent-Gaussian REINFORCE versus matched pathwise optimization.

This implements a newly defined policy adaptation of released State weights. It
is not an implementation of native State RL, DiffusionNFT, GRPO, or PPO.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.official_state import load_official_state, load_perturbation_map, sha256_file
from cross_species.state_policy import StateLatentPolicy, fit_policy, evaluate_policy
from run_official_state_benchmark import write_json, paired_interval, serializable


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
    native_specification = args.native_run / "specification.json"
    native = json.loads(native_specification.read_text())
    for key, path in [("checkpoint", args.checkpoint), ("perturbation_map", args.perturbation_map), ("cells_npz", args.cells_npz)]:
        if sha256_file(path) != native["input_hashes"][key]:
            raise ValueError("Input differs from the frozen native benchmark: " + key)
    with np.load(args.cells_npz, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    lookup = {str(value): index for index, value in enumerate(payload["cell_ids"])}
    if len(lookup) != len(payload["expression"]):
        raise ValueError("Duplicate actual cell IDs")
    populations = {(record["condition"], record["donor"]): {
        "controls": np.array([lookup[value] for value in record["control_cell_ids"]]),
        "targets": np.array([lookup[value] for value in record["target_cell_ids"]])}
        for record in native["cell_populations"]}
    schedule = [{**record, "control_rows": np.array(record["control_rows"]), "target_rows": np.array(record["target_rows"])}
                for record in native["training_schedule"]]
    if any(record["condition"] not in native["roles"]["train"] for record in schedule):
        raise ValueError("Training schedule contains a held-out cytokine")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specification = {"native_specification_sha256": sha256_file(native_specification),
        "roles": native["roles"], "donors": native["donors"], "cell_count": native["cell_count_per_population"],
        "updates_per_arm": len(schedule), "seed": native["seed"], "sigma": .05, "last_official_blocks_trainable": 2,
        "candidate_beta_kl_per_dimension": [0., .01, .1, 1.], "rollouts_per_update_and_evaluation": 4,
        "estimators": ["pathwise", "reinforce"], "optimizer": native["optimizer"],
        "policy": "Z_population ~ independent Normal(mu_official_predecoder, .05); joint action is all N x384 latent entries; only final two official transformer blocks are trainable; original decoder frozen.",
        "objective": "E[native population energy] + beta * KL(N(mu,.05),N(mu_original,.05))/(N*384)",
        "reinforce": "Negative reward-weighted joint log-probability sum with detached actions and detached leave-one-rollout-out baseline; no advantage-standardization or clipping; exact likelihood-ratio estimator.",
        "pathwise": "Same objective, official blocks, four noise draws, target populations, update count and seeds; differentiate energy through reparameterized latents and frozen decoder.",
        "selection": "Choose beta independently per estimator using only cytokine-macro calibration expected native energy; exact ties prefer smaller beta.",
        "primary": "Four-rollout expected native population energy, cytokine macro after donor average, same metadata-derived evaluation seeds.",
        "baseline": "Zero-update original Gaussian policy, same sigma and rollout count, released weights.",
        "scope": "Human-only known-donor/cell-context technical benchmark; custom stochastic adaptation of State, no claim of native State/NFT/GRPO/PPO reproduction or animal feedback.",
        "limitations": ["Gaussian latent noise is exploration, not biological variability.", "decode(mu) differs from E[decode(Z)]; deterministic outputs are decoded latent-mean outputs.",
                        "Same rollouts and updates do not imply equal FLOPs; walltime is reported.", "Single optimization seed; eight held-out cytokines with reused donors/controls; intervals conditional.",
                        "Differentiable human-label reward has a direct pathwise supervised comparator; an RL advantage is not presumed."],
        "script_sha256": sha256_file(__file__), "policy_module_sha256": sha256_file(ROOT / "src/cross_species/state_policy.py"),
        "inputs": native["input_hashes"]}
    write_json(args.out_dir / "specification.json", specification)
    progress = args.out_dir / "progress.jsonl"
    def log(event, **values):
        row = {"event": event, "utc_epoch": time.time(), **values}
        with progress.open("a") as file:
            file.write(json.dumps(row, default=serializable, allow_nan=False) + "\n")
        print(json.dumps(row, default=serializable, allow_nan=False), flush=True)
    representations = load_perturbation_map(args.perturbation_map, 91)
    reference, metadata = load_official_state(args.checkpoint, device=args.device)
    reference.eval().requires_grad_(False)
    calib, selections = {}, {}
    for estimator in specification["estimators"]:
        calib[estimator] = {}
        for beta in specification["candidate_beta_kl_per_dimension"]:
            log("arm_started", estimator=estimator, beta=beta)
            model, _ = load_official_state(args.checkpoint, device=args.device)
            policy = StateLatentPolicy(model, last_blocks=2, sigma=.05).to(args.device)
            fit = fit_policy(policy, reference, payload["expression"], schedule, representations,
                estimator=estimator, beta=beta, device=args.device, lr=native["optimizer"]["lr"],
                seed=native["seed"], rollouts=4)
            if fit["updated_parameter_l2_change"] <= 0:
                raise ValueError("No actual official-block policy update was measured")
            result = evaluate_policy(policy, payload["expression"], populations, native["roles"]["calib"], native["donors"], representations,
                device=args.device, cell_count=specification["cell_count"], bandwidth=native["rbf_bandwidth_train_only"], rollouts=4, seed=native["seed"])
            name = estimator + "_beta_" + format(beta, "g")
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()}, "beta": beta,
                        "estimator": estimator, "specification_sha256": sha256_file(args.out_dir / "specification.json")}, args.out_dir / (name + ".pt"))
            write_json(args.out_dir / (name + "_fit.json"), fit)
            write_json(args.out_dir / (name + "_calibration.json"), result)
            calib[estimator][beta] = result
            log("arm_completed", estimator=estimator, beta=beta, calibration=result["macro"], seconds=fit["seconds"],
                trainable_parameters=fit["trainable_parameters"], witness_delta=fit["updated_parameter_l2_change"])
            del policy, model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        selections[estimator] = min(calib[estimator], key=lambda beta: (calib[estimator][beta]["macro"]["expected_native_energy"], beta))
    selection = {"selected_beta": selections, "test_evaluated_before_selection": False,
                 "calibration": {estimator: {str(beta): result["macro"] for beta, result in values.items()} for estimator, values in calib.items()}}
    write_json(args.out_dir / "selection.json", selection)
    log("calibration_selection_frozen", selected_beta=selections)
    results = {}
    for name in ["zero_update_gaussian_reference", "pathwise", "reinforce"]:
        model, _ = load_official_state(args.checkpoint, device=args.device)
        policy = StateLatentPolicy(model, last_blocks=2, sigma=.05).to(args.device)
        if name != "zero_update_gaussian_reference":
            archive = torch.load(args.out_dir / (name + "_beta_" + format(selections[name], "g") + ".pt"), map_location="cpu", weights_only=True)
            policy.load_state_dict(archive["state_dict"], strict=True)
        result = evaluate_policy(policy, payload["expression"], populations, native["roles"]["test"], native["donors"], representations,
            device=args.device, cell_count=specification["cell_count"], bandwidth=native["rbf_bandwidth_train_only"], rollouts=4,
            seed=native["seed"], save_predictions=True)
        np.savez_compressed(args.out_dir / (name + "_test_predictions.npz"), sampled_predictions=result.pop("sampled_predictions"),
            deterministic_latent_mean_predictions=result.pop("deterministic_predictions"), targets=result.pop("targets"), feature_ids=payload["gene_ids"])
        write_json(args.out_dir / (name + "_test_query_indices.json"), result.pop("query_indices"))
        results[name] = result
        log("test_procedure_completed", procedure=name, metrics=result["macro"])
        del policy, model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report = {"specification_sha256": sha256_file(args.out_dir / "specification.json"), "checkpoint_metadata": metadata,
              "selection": selection, "test": results,
              "versus_zero_update": {name: paired_interval(results["zero_update_gaussian_reference"], results[name], metric="expected_native_energy") for name in ["pathwise", "reinforce"]},
              "reinforce_vs_pathwise": paired_interval(results["pathwise"], results["reinforce"], metric="expected_native_energy"),
              "claim": "Custom State last-two-block Gaussian-policy likelihood-ratio optimization versus same-objective pathwise supervision, entirely human labels."}
    if args.device == "cuda":
        report["cuda"] = {"device": torch.cuda.get_device_name(), "runtime": torch.version.cuda,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(args.out_dir / "report.json", report)
    log("completed")


if __name__ == "__main__":
    main()
