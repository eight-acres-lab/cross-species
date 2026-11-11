#!/usr/bin/env python3
"""Released-scDFM continuation and audited NFT-objective adaptation on real cells."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.official_scdfm import load_official_scdfm
from cross_species.official_state import sha256_file
from cross_species.scdfm_feedback import fit_feedback, evaluate_flow
from run_official_state_benchmark import write_json, serializable, paired_interval


def make_roles(frame, seed):
    original_train = sorted(set(frame.loc[frame["original_role"].eq("official_train") & frame["condition"].ne("control"), "condition"]))
    development = "Alvespimycin+Pirarubicin"
    if development not in original_train or len(original_train) != 24:
        raise ValueError("Unexpected official training-condition metadata")
    key = lambda value: hashlib.sha256((str(seed) + "|scdfm-feedback|" + value).encode()).hexdigest()
    remaining = sorted(set(original_train) - {development}, key=key)
    test = sorted(set(frame.loc[frame["original_role"].eq("official_test"), "condition"]))
    if len(test) != 7:
        raise ValueError("All seven original chemical-condition holdouts must be retained")
    return {"train": [development] + remaining[:17], "calib": remaining[17:], "test": test,
            "development_forced_into_train": development,
            "calibration_exposure": "These six conditions were present in released-checkpoint pretraining; they are excluded only from this additional fitting."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["checkpoint", "vendor-root", "processed-dir", "out-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--training-cells", type=int, default=4)
    parser.add_argument("--evaluation-cells", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--development-only", action="store_true", help="Fit training rows only, then stop without any held-out scoring")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use an empty output directory")
    if min(args.training_cells, args.evaluation_cells) < 2 or min(args.updates, args.rollouts, args.ode_steps) <= 0:
        parser.error("Invalid fixed population or optimization budget")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.processed_dir / "preprocessing_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256_file(args.processed_dir / name) != expected["sha256"]:
            raise ValueError("Input hash mismatch: " + name)
    frame = pd.read_csv(args.processed_dir / "metadata.csv", keep_default_na=False)
    matrix = sparse.load_npz(args.processed_dir / "expression_lognorm_hvg5000.npz")
    all_genes = np.load(args.processed_dir / "gene_names.npy", allow_pickle=False)
    token_ids = np.load(args.processed_dir / "gene_token_ids.npy", allow_pickle=False)
    genes = np.load(args.processed_dir / "official_test_hvg1000_gene_names.npy", allow_pickle=False)
    lookup = {str(value): index for index, value in enumerate(all_genes)}
    columns = np.array([lookup[str(value)] for value in genes])
    if len(columns) != 1000 or matrix.shape != (len(frame), len(all_genes)) or len(np.unique(frame["original_cell_id"])) != len(frame):
        raise ValueError("Unexpected real-cell or author-released evaluation-axis contract")
    roles = make_roles(frame, args.seed)
    control_rows = np.flatnonzero(frame["condition"].eq("control").to_numpy())
    target_rows = {condition: np.flatnonzero(frame["condition"].eq(condition).to_numpy()) for role in ["train", "calib", "test"] for condition in roles[role]}
    drug_ids = {condition: frame.loc[rows[0], ["drug_id_1", "drug_id_2"]].to_numpy(dtype="int64") for condition, rows in target_rows.items()}
    rng = np.random.default_rng(args.seed)
    schedule = []
    for update in range(args.updates):
        condition = str(rng.choice(roles["train"]))
        schedule.append({"update": update, "condition": condition,
                         "control_rows": rng.choice(control_rows, args.training_cells, replace=False),
                         "target_rows": rng.choice(target_rows[condition], args.training_cells, replace=False), "drug_ids": drug_ids[condition]})
    observed_targets = np.unique(np.concatenate([record["target_rows"] for record in schedule]))
    sample = rng.choice(observed_targets, min(128, len(observed_targets)), replace=False)
    distances = pdist(matrix[sample][:, columns].toarray())
    positive_distances = distances[distances > 0]
    if not len(positive_distances):
        raise ValueError("Training-only RBF bandwidth is undefined")
    bandwidth = float(np.median(positive_distances))
    populations = {}
    for condition, rows in target_rows.items():
        if min(len(rows), len(control_rows)) < args.evaluation_cells:
            raise ValueError("Insufficient distinct real evaluation cells")
        key = int(hashlib.sha256((str(args.seed)+"|scdfm-population|"+condition).encode()).hexdigest()[:15], 16)
        choice = np.random.default_rng(key)
        populations[condition] = {"control_rows": choice.choice(control_rows, args.evaluation_cells, replace=False),
                                 "target_rows": choice.choice(rows, args.evaluation_cells, replace=False),
                                 "drug_ids": drug_ids[condition], "rbf_bandwidth": bandwidth}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specification = {"scope": "Single released 60M scDFM checkpoint, human A549 chemical-condition continuation; no animal/source measurement or scaling claim.",
        "roles": roles, "available_original_cells": len(frame), "actual_optimizer_target_cell_count": len(observed_targets),
        "actual_optimizer_control_cell_count": len(np.unique(np.concatenate([record["control_rows"] for record in schedule]))),
        "training_schedule": schedule, "evaluation_populations": populations,
        "training_target_cell_ids": frame.loc[observed_targets, "original_cell_id"].tolist(), "gene_names": genes.tolist(),
        "gene_columns": columns, "gene_token_ids": token_ids[columns], "updates_per_method": args.updates,
        "methods": ["native_fm", "nft", "terminal_gradient"], "rollouts_per_update": args.rollouts, "training_cells_per_population": args.training_cells,
        "evaluation_cells_per_population": args.evaluation_cells, "test_rollouts": args.rollouts, "calibration_rollouts": 1,
        "ode": {"solver": "fixed Euler", "steps": args.ode_steps, "source_solver": "Fixed Euler integration; the source implementation uses RK4 on 20 time-grid points."},
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0, "clip_norm": 1.},
        "nft": {"source_commit": "cbb14f84b8312b620390dfcbe2fab69c1383b104", "source": "NVlabs/DiffusionNFT/scripts/train_nft_sd3.py:910-970",
                "mixture_beta": 1., "advantage_clip": 5., "reference_velocity_mse_weight": 1e-4,
                "old_refresh": "NFT decay_type1: min(.001*global_update,.5) EMA, after each completed generation/update round",
                "time_convention": "scDFM tau=noise-to-data; NFT t=1-tau,v=-u; generated raw terminal is clean target, fresh independent corruption noise.",
                "advantage": "One fixed condition/basal population prompt per round; center K population rewards, divide population-std+1e-4, clip to5.",
                "double_weights": "Official detached float64 mean-absolute reconstruction denominators, clamp_min1e-5."},
        "reward": "Negative energy U between clipped generated population and the identical real fitting target population; one scalar per joint population.",
        "primary": "Test expected energy U; macro over seven chemical conditions, four fixed generation seeds; cells and genes are not independent tasks.",
        "reference_penalty": "All methods use the same 1e-4 current-vs-original velocity MSE and generation/corruption rules; each method's old-buffer values differ as its policy changes.",
        "native_fm": "Four real-target native Gaussian FM + one-step MMD(.5) losses per update; same four old-policy rollout seeds are generated for monitoring and the common reference penalty.",
        "terminal_gradient": "Same terminal energy and reference penalty; replay same four initial noise seeds with live-model differentiable Euler and activation checkpointing. Extra replay/backward compute is reported, not equated to NFT FLOPs.",
        "modes": "live/old/original reference all eval-mode, retaining live autograd, to disable scDFM .1 dropout and preserve neutral-advantage cancellation.",
        "precision": "float32 weights and activations; TF32 matrix multiplications enabled/high precision; NFT weight reductions float64.",
        "numeric_compatibility": "Frozen old/reference velocities use detached leaf parameter views in torch.func.functional_call with autograd enabled, then outputs detach immediately. Original frozen parameters remain unchanged and receive no gradients. This matches live CUDA kernels: development initial velocity difference and neutral-advantage parameter gradients were exactly zero with TF32 enabled or disabled.",
        "selection": "No hyperparameter or update selection; all coefficients fixed before formal fitting. Calibration is descriptive and evaluated before final test.",
        "seed": args.seed, "development_only": args.development_only, "input_hashes": {"checkpoint": sha256_file(args.checkpoint),
             "preprocessing_manifest": sha256_file(args.processed_dir / "preprocessing_manifest.json")},
        "preprocessing": manifest, "rbf_bandwidth_fit_rows_only": bandwidth,
        "limitations": ["Original 5000-HVG and test1000-HVG axes follow author transductive preprocessing; not a new training-only feature-selection claim.",
                        "Original seven test labels may have been used for checkpoint selection by source authors; this is not independently reconstructed pretraining isolation.",
                        "Six additional-calibration conditions were seen in original pretraining.", "N4 fitting populations and one A549 context; one optimization seed; no biological-replicate uncertainty.",
                        "63k is original dataset size, not this continuation's actual label count.", "NFT loss is an adaptation to scDFM; no SD3/image-experiment reproduction, likelihood-ratio, PPO or exact-KL claim.",
                        "Energy U may be negative; intervals condition on fitted predictions and the fixed feature/readout contract."],
        "code_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in [Path(__file__), ROOT/"src/cross_species/scdfm_feedback.py", ROOT/"src/cross_species/scdfm_nft.py", ROOT/"src/cross_species/official_scdfm.py"]}}
    write_json(args.out_dir / "specification.json", specification)
    progress = args.out_dir / "progress.jsonl"
    def log(event, **values):
        row = {"event": event, "time": time.time(), **values}
        with progress.open("a") as file: file.write(json.dumps(row, allow_nan=False, default=serializable)+"\n")
        print(json.dumps(row, allow_nan=False, default=serializable), flush=True)
    reference, metadata = load_official_scdfm(args.checkpoint, args.vendor_root, args.processed_dir/"mask_k30_signed_pad4.pt", device=args.device)
    reference.eval().requires_grad_(False)
    write_json(args.out_dir/"checkpoint_metadata.json", metadata)
    fits, calibration = {}, {}
    for method in specification["methods"]:
        log("method_started", method=method)
        model, _ = load_official_scdfm(args.checkpoint, args.vendor_root, args.processed_dir/"mask_k30_signed_pad4.pt", device=args.device)
        old, _ = load_official_scdfm(args.checkpoint, args.vendor_root, args.processed_dir/"mask_k30_signed_pad4.pt", device=args.device)
        fits[method] = fit_feedback(model, old, reference, matrix, schedule, columns, token_ids, method=method,
            device=args.device, steps=args.ode_steps, rollouts=args.rollouts, lr=args.lr, seed=args.seed)
        write_json(args.out_dir/(method+"_fit.json"), fits[method])
        torch.save({"state_dict": {key:value.detach().cpu() for key,value in model.state_dict().items()},
                    "old_state_dict": {key:value.detach().cpu() for key,value in old.state_dict().items()},
                    "specification_sha256": sha256_file(args.out_dir/"specification.json")}, args.out_dir/(method+".pt"))
        log("method_completed", method=method, updates=fits[method]["updates"], wall_seconds=fits[method]["wall_seconds"], witness_delta=fits[method]["updated_parameter_l2_change"])
        if not args.development_only:
            selected = {condition:populations[condition] for condition in roles["calib"]}
            result = evaluate_flow(model, matrix, selected, columns, token_ids, device=args.device,
                                   steps=args.ode_steps, rollouts=1, seed=args.seed)
            for key in ["predictions","raw_predictions","targets"]: result.pop(key)
            calibration[method] = result
            write_json(args.out_dir/(method+"_calibration.json"), result)
        del model, old
        if args.device == "cuda": torch.cuda.empty_cache()
    if args.development_only:
        write_json(args.out_dir/"report.json", {"development_only": True, "fits": fits, "no_held_out_scores": True})
        log("development_completed_without_heldout_scoring")
        return
    write_json(args.out_dir/"selection.json", {"selection": "All methods and coefficients fixed; no method selected.", "test_evaluated_before_selection": False, "calibration": calibration})
    results = {}
    for method in ["released_checkpoint"] + specification["methods"]:
        log("test_started", method=method)
        model, _ = load_official_scdfm(args.checkpoint, args.vendor_root, args.processed_dir/"mask_k30_signed_pad4.pt", device=args.device)
        if method != "released_checkpoint":
            archive = torch.load(args.out_dir/(method+".pt"), map_location="cpu", weights_only=True)
            model.load_state_dict(archive["state_dict"], strict=True)
        result = evaluate_flow(model, matrix, {condition:populations[condition] for condition in roles["test"]}, columns, token_ids,
            device=args.device, steps=args.ode_steps, rollouts=args.rollouts, seed=args.seed)
        np.savez_compressed(args.out_dir/(method+"_test_predictions.npz"), predictions=result.pop("predictions"), raw_predictions=result.pop("raw_predictions"),
                            targets=result.pop("targets"), condition_order=np.array(result["condition_order"]), gene_names=genes)
        results[method] = result
        write_json(args.out_dir/(method+"_test.json"), result)
        log("test_completed", method=method, metrics=result["macro"])
        del model
        if args.device == "cuda": torch.cuda.empty_cache()
    report = {"specification_sha256": sha256_file(args.out_dir/"specification.json"), "checkpoint_metadata": metadata, "fit": fits,
              "calibration": calibration, "test": results,
              "versus_released": {method: paired_interval(results["released_checkpoint"], results[method], metric="energy_u", seed=args.seed) for method in specification["methods"]},
              "nft_vs_native_fm": paired_interval(results["native_fm"], results["nft"], metric="energy_u", seed=args.seed),
              "nft_vs_terminal_gradient": paired_interval(results["terminal_gradient"], results["nft"], metric="energy_u", seed=args.seed),
              "scope": specification["scope"]}
    if args.device == "cuda": report["cuda"] = {"device": torch.cuda.get_device_name(), "runtime": torch.version.cuda,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(args.out_dir/"report.json", report)
    log("completed")


if __name__ == "__main__":
    main()
