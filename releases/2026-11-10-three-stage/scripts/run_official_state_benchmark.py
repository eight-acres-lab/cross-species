#!/usr/bin/env python3
"""Fixed-budget human-context State energy continuation and reference anchoring.

The released State transformer is loaded strictly. This is supervised population
fine-tuning; no policy-gradient or cross-species experiment is defined here.
"""
import argparse
import ast
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.official_state import load_official_state, load_perturbation_map, sha256_file
from cross_species.state_finetuning import condition_split, population_index, training_schedule, fit_arm, evaluate_model


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=serializable) + "\n")


def official_test_conditions(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip().startswith("test =")]
    if len(lines) != 1:
        raise ValueError("Expected the single released B-cell test-list override")
    return ast.literal_eval(lines[0].split("=", 1)[1].strip())


def paired_interval(baseline, method, metric="native_energy", seed=20261108):
    conditions = sorted(baseline["by_condition"])
    gains = np.array([baseline["by_condition"][name][metric] - method["by_condition"][name][metric] for name in conditions])
    draws = np.random.default_rng(seed).integers(0, len(gains), size=(2000, len(gains)))
    return {"metric": metric, "positive_means_improvement": True, "conditions": conditions,
            "per_condition_gain": gains.tolist(), "macro_gain": float(gains.mean()),
            "fixed_prediction_cytokine_bootstrap_95_interval": np.quantile(gains[draws].mean(axis=1), [.025, .975]).tolist(),
            "warning": "Descriptive conditional interval from four or eight cytokines; shared donors and controls, one training seed, no population-coverage guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ["checkpoint", "perturbation-map", "cells-npz", "official-split", "out-dir"]:
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--cells-manifest", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cell-count", type=int, default=16, choices=[16, 64])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true", help="Write metadata/data specification without loading the model or scoring any role")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Use an empty output directory; completed or partial runs are preserved")
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.set_num_threads(args.threads)
    manifest_path = args.cells_manifest or args.cells_npz.with_name(args.cells_npz.stem + "_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    revision = "827df657cd87063026d2abefd73e99036755ac1b"
    prefix = "https://huggingface.co/arcinstitute/ST-HVG-Parse/resolve/" + revision + "/"
    if (manifest.get("sha256") != sha256_file(args.cells_npz) or manifest.get("source_revision") != revision
            or not manifest.get("source_url", "").startswith(prefix)):
        raise ValueError("Real-cell data must match the pinned same-release extraction manifest")
    with np.load(args.cells_npz, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    expression = payload["expression"]
    if expression.ndim != 2 or not np.isfinite(expression).all() or len(np.unique(payload["cell_ids"])) != len(expression):
        raise ValueError("Require distinct actual cells and finite expression")
    if set(payload["cell_type"].astype(str)) != {"B_Intermediate_Memory"}:
        raise ValueError("Only the frozen released B-cell context is eligible")
    if not np.array_equal(payload["gene_ids"].astype(str), np.arange(2000).astype(str)):
        raise ValueError("Require exact same-release anonymous numeric feature order")
    perturbations = load_perturbation_map(args.perturbation_map, 91)
    allowed = set(official_test_conditions(args.official_split)) & set(perturbations)
    donors = sorted(set(payload["donor"].astype(str)))
    if len(donors) != 12:
        raise ValueError("The frozen known-context protocol retains all twelve donors")
    available = sorted(set(payload["cytokine"].astype(str)) & allowed)
    # Eligibility depends on cell counts and metadata only, never expression magnitude or loss.
    complete = []
    for condition in available:
        try:
            population_index(payload, [condition], donors, args.cell_count)
        except ValueError:
            continue
        complete.append(condition)
    roles = condition_split(complete, allowed, seed=args.seed)
    used = roles["train"] + roles["calib"] + roles["test"]
    populations = population_index(payload, used, donors, args.cell_count)
    schedule = training_schedule(populations, roles["train"], donors, steps=args.steps,
                                 cell_count=args.cell_count, seed=args.seed)
    train_rows = np.concatenate([populations[(condition, donor)]["targets"] for condition in roles["train"] for donor in donors])
    # A training-only secondary-kernel bandwidth, fixed for every arm and role.
    sample = np.random.default_rng(args.seed).choice(train_rows, min(128, len(train_rows)), replace=False)
    from scipy.spatial.distance import pdist
    distances = pdist(expression[sample])
    bandwidth = float(np.median(distances[distances > 0]))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Training-only RBF bandwidth is undefined")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = payload["cell_ids"].astype(str)
    frozen_populations = [{"condition": condition, "donor": donor,
                           "control_cell_ids": labels[rows["controls"][:args.cell_count]],
                           "target_cell_ids": labels[rows["targets"][:args.cell_count]]}
                          for (condition, donor), rows in populations.items()]
    specification = {"scope": "Human-only post-training cytokine-held-out groups within released B-cell evaluation context; known donors; anonymous native features.",
        "method": "Released State native primary-energy supervised continuation plus deterministic frozen-reference prediction-MSE anchoring.",
        "not_established": ["RL/NFT/PPO", "animal feedback", "global unseen perturbations", "donor generalization", "named-gene biology"],
        "roles": roles, "donors": donors, "cell_count_per_population": args.cell_count,
        "available_distinct_training_target_cells": len(np.unique(train_rows)),
        "optimizer_used_distinct_training_target_cells": len(np.unique(np.concatenate([item["target_rows"] for item in schedule]))),
        "candidate_anchor_weights": [0., .01, .1, 1.], "selection": "Minimum calibration cytokine-macro native energy; exact ties prefer smaller lambda.",
        "updates_per_arm": args.steps, "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": .0005, "clip_norm": 10.},
        "seed": args.seed, "rbf_bandwidth_train_only": bandwidth,
        "cell_populations": frozen_populations, "training_schedule": schedule,
        "input_hashes": {"checkpoint": sha256_file(args.checkpoint), "cells_npz": sha256_file(args.cells_npz),
                         "cells_manifest": sha256_file(manifest_path), "perturbation_map": sha256_file(args.perturbation_map),
                         "official_split": sha256_file(args.official_split)},
        "script_sha256": sha256_file(__file__), "module_sha256": sha256_file(ROOT / "src/cross_species/state_finetuning.py"),
        "python": platform.python_version(), "torch": str(torch.__version__), "device": args.device,
        "data_manifest": manifest,
        "limitations": ["Condition groups, not cells or coordinates, define held-out tasks.", "PBS populations and donors are shared across roles.",
                        "Model set size differs from original training size512.", "Primary native energy is a finite-population V-statistic; energy U/MMD U are secondary and may be negative.",
                        "One optimization seed and few held-out cytokines; this is a technical human-context benchmark."]}
    write_json(args.out_dir / "specification.json", specification)
    (args.out_dir / "specification.sha256").write_text(sha256_file(args.out_dir / "specification.json") + "\n")
    if args.prepare_only:
        print(json.dumps({"prepared_only": True, "roles": roles, "cell_count": args.cell_count,
                          "specification_sha256": sha256_file(args.out_dir / "specification.json")}, indent=2))
        return
    progress = args.out_dir / "progress.jsonl"
    def log(event, **values):
        record = {"event": event, "utc_epoch": time.time(), **values}
        with progress.open("a") as file:
            file.write(json.dumps(record, default=serializable, allow_nan=False) + "\n")
        print(json.dumps(record, default=serializable, allow_nan=False), flush=True)
    reference, metadata = load_official_state(args.checkpoint, device=args.device)
    reference.eval().requires_grad_(False)
    write_json(args.out_dir / "checkpoint_metadata.json", metadata)
    calib = {}
    for weight in specification["candidate_anchor_weights"]:
        log("arm_started", anchor_weight=weight)
        model, _ = load_official_state(args.checkpoint, device=args.device)
        fit = fit_arm(model, reference, expression, schedule, perturbations, weight=weight,
                      device=args.device, lr=args.lr, seed=args.seed)
        calibration = evaluate_model(model, expression, populations, roles["calib"], donors, perturbations,
                                     device=args.device, cell_count=args.cell_count, bandwidth=bandwidth)
        name = "lambda_" + format(weight, "g")
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "anchor_weight": weight, "specification_sha256": sha256_file(args.out_dir / "specification.json")}, args.out_dir / (name + ".pt"))
        write_json(args.out_dir / (name + "_fit.json"), fit)
        write_json(args.out_dir / (name + "_calibration.json"), calibration)
        calib[weight] = calibration
        log("arm_completed", anchor_weight=weight, updates=fit["updates"], calibration=calibration["macro"], seconds=fit["seconds"])
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    selected = min(calib, key=lambda weight: (calib[weight]["macro"]["native_energy"], weight))
    selection = {"selected_anchor_weight": selected, "calibration_native_energy": {str(weight): result["macro"]["native_energy"] for weight, result in calib.items()},
                 "test_evaluated_before_selection": False, "selection_metric": "calibration cytokine-macro native energy"}
    write_json(args.out_dir / "selection.json", selection)
    log("calibration_selection_frozen", **selection)
    results = {}
    for name, weight in [("released_checkpoint", None), ("native_energy_sft", 0.), ("calibration_selected_anchoring", selected)]:
        model, _ = load_official_state(args.checkpoint, device=args.device)
        if weight is not None:
            archive = torch.load(args.out_dir / ("lambda_" + format(weight, "g") + ".pt"), map_location="cpu", weights_only=True)
            model.load_state_dict(archive["state_dict"], strict=True)
        result = evaluate_model(model, expression, populations, roles["test"], donors, perturbations,
                                device=args.device, cell_count=args.cell_count, bandwidth=bandwidth, save_predictions=True)
        np.savez_compressed(args.out_dir / (name + "_test_predictions.npz"), predictions=result.pop("predictions"),
                            targets=result.pop("targets"), feature_ids=payload["gene_ids"])
        write_json(args.out_dir / (name + "_test_query_indices.json"), result.pop("query_indices"))
        results[name] = result
        log("test_procedure_completed", procedure=name, metrics=result["macro"])
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    report = {"specification_sha256": sha256_file(args.out_dir / "specification.json"), "checkpoint_metadata": metadata,
              "selection": selection, "test": results, "calibration": {str(weight): result for weight, result in calib.items()},
              "paired_comparisons": {name: paired_interval(results["released_checkpoint"], result) for name, result in results.items() if name != "released_checkpoint"},
              "anchored_vs_sft": paired_interval(results["native_energy_sft"], results["calibration_selected_anchoring"]),
              "statement": "Completed human-only supervised population fine-tuning benchmark; no RL or animal-feedback result."}
    if args.device == "cuda":
        report["cuda"] = {"device": torch.cuda.get_device_name(), "runtime": torch.version.cuda,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    write_json(args.out_dir / "report.json", report)
    log("completed", selected_anchor_weight=selected)


if __name__ == "__main__":
    main()
