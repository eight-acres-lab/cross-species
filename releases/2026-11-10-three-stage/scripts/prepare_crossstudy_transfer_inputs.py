#!/usr/bin/env python3
"""Prepare a shared-ID, human/rat external-study transfer contract remotely.

External human responses are isolated in a scoring-only file. Feature admission
uses exact IDs and observation presence, never response size or prediction loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-dir", type=Path, required=True)
    ap.add_argument("--panel-dir", type=Path, required=True)
    ap.add_argument("--bridge-dir", type=Path, required=True)
    ap.add_argument("--task-manifest", type=Path, required=True)
    ap.add_argument("--chemical-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise ValueError("Use a new input version")
    training_manifest = json.loads((args.training_dir / "manifest.json").read_text())
    for name in ["bundle.npz", "metadata.json"]:
        if sha(args.training_dir / name) != training_manifest["files"][name]:
            raise ValueError("Training input checksum mismatch: " + name)
    panel_manifest = json.loads((args.panel_dir / "manifest.json").read_text())
    for name, expected in panel_manifest["files"].items():
        if sha(args.panel_dir / name) != expected:
            raise ValueError("Frozen external panel changed: " + name)
    bridge_audit = json.loads((args.bridge_dir / "audit.json").read_text())
    bridge_path = args.bridge_dir / "GSE330151_bridge.tsv"
    if sha(bridge_path) != bridge_audit["files"][bridge_path.name]:
        raise ValueError("Gene bridge changed")
    with bridge_path.open() as handle:
        bridge = list(csv.DictReader(handle, delimiter="\t"))
    chemistry_manifest = json.loads((args.chemical_dir / "manifest.json").read_text())
    if sha(args.chemical_dir / "features.npz") != chemistry_manifest["feature_sha256"]:
        raise ValueError("External chemistry changed")
    if not chemistry_manifest["all_three_absent_from_strict_training_CIDs_and_InChIKeys"]:
        raise ValueError("External chemical identity overlaps training or remains unresolved")
    with np.load(args.training_dir / "bundle.npz", allow_pickle=False) as z:
        training = {key: z[key].copy() for key in z.files}
    with np.load(args.panel_dir / "panel.npz", allow_pickle=False) as z:
        panel = {key: z[key].copy() for key in ["feature_ids", "species_order", "native_gene_ids", "arm_logcpm", "contrast_means"]}
    with np.load(args.chemical_dir / "features.npz", allow_pickle=False) as z:
        chemistry = {key: z[key].copy() for key in z.files}
    tasks = json.loads(args.task_manifest.read_text())
    if [row["perturbation_code"] for row in tasks] != ["CDDO", "KI696", "THAP"]:
        raise ValueError("Expected the three metadata-selected tasks in fixed order")
    arm_metadata = json.loads((args.panel_dir / "arm_metadata.json").read_text())
    contrast_metadata = json.loads((args.panel_dir / "contrast_metadata.json").read_text())
    contrast_position = {row["contrast_id"]: i for i, row in enumerate(contrast_metadata)}
    species_position = {str(value): i for i, value in enumerate(panel["species_order"])}
    arm_indices = []
    for task in tasks:
        for side in ("human", "rat"):
            row = task["sides"][side]
            for role in ("treated", "control"):
                indices = row[role + "_arm_rows"]
                samples = row[role + "_sample_ids"]
                if len(indices) != 3 or [arm_metadata[i]["sample_id"] for i in indices] != samples:
                    raise ValueError("Original sample metadata bridge changed")
                if any(arm_metadata[i]["species"] != side for i in indices):
                    raise ValueError("Wrong species in external contrast")
                arm_indices.extend(indices)
    arm_indices = sorted(set(arm_indices))
    if len(arm_indices) != 36:
        raise ValueError("Unexpected external unique sample count")
    tg_indices, panel_indices = [], []
    for row in bridge:
        ti, pi = int(row["tg_axis_index_zero_based"]), int(row["GSE330151_panel_index_zero_based"])
        if (str(training["human_gene_ids"][ti]) != row["human_entrez"]
                or str(training["rat_gene_ids"][ti]) != row["rat_entrez"]
                or str(panel["feature_ids"][pi]) != row["human_ensembl"]
                or str(panel["native_gene_ids"][species_position["human"], pi]) != row["human_ensembl"]
                or str(panel["native_gene_ids"][species_position["rat"], pi]) != row["rat_ensembl"]):
            raise ValueError("Exact training/panel identifier bridge failed")
        tg_indices.append(ti); panel_indices.append(pi)
    tg_indices, panel_indices = np.asarray(tg_indices), np.asarray(panel_indices)
    for values in (tg_indices, panel_indices):
        if len(values) != len(set(values.tolist())):
            raise ValueError("Repeated feature bridge position")
    observed = np.isfinite(panel["arm_logcpm"][np.ix_(arm_indices, panel_indices)]).all(axis=0)
    excluded = [dict(row, exclusion="not_finite_in_all_selected_human_rat_arms") for row, ok in zip(bridge, observed) if not ok]
    bridge = [row for row, ok in zip(bridge, observed) if ok]
    tg_indices, panel_indices = tg_indices[observed], panel_indices[observed]
    if len(bridge) < 1000:
        raise ValueError("Too few complete jointly bridged genes")
    source, target, context, reconstruction_errors = [], [], [], []
    for task in tasks:
        responses = {}
        for side in ("human", "rat"):
            row = task["sides"][side]
            response = (panel["arm_logcpm"][row["treated_arm_rows"]][:, panel_indices].mean(axis=0)
                        - panel["arm_logcpm"][row["control_arm_rows"]][:, panel_indices].mean(axis=0))
            old = panel["contrast_means"][contrast_position[row["contrast_id"]], panel_indices]
            error = float(np.max(np.abs(response - old)))
            if error > 1e-10:
                raise ValueError("External response differs from original arm-level reconstruction")
            reconstruction_errors.append({"contrast": row["contrast_id"], "max_abs_difference": error})
            responses[side] = response
        source.append(responses["rat"]); target.append(responses["human"])
        context.append([float(task["time_h"]), np.log10(task["rat_M"]), np.log10(task["human_M"])])
    code_to_chemistry = {row["panel_perturbation"]: row["compound"] for row in chemistry_manifest["accepted"]}
    chemical_positions = {str(value): i for i, value in enumerate(chemistry["compounds"])}
    qindices = [chemical_positions[code_to_chemistry[row["perturbation_code"]]] for row in tasks]
    if not np.all(chemistry["verified_mask"][qindices]):
        raise ValueError("Unverified query molecule")
    train = {key: value for key, value in training.items() if key not in ["source", "target", "context", "human_gene_ids", "rat_gene_ids"]}
    train.update(source=training["source"][:, tg_indices], target=training["target"][:, tg_indices],
                 context=training["context"][:, :3], human_gene_ids=training["human_gene_ids"][tg_indices],
                 rat_gene_ids=training["rat_gene_ids"][tg_indices])
    queries = {"source": np.asarray(source), "context": np.asarray(context),
               "fingerprints": chemistry["fingerprints"][qindices], "descriptors": chemistry["descriptors"][qindices],
               "task_ids": np.asarray([row["perturbation_code"] for row in tasks]),
               "human_gene_ids": train["human_gene_ids"], "rat_gene_ids": train["rat_gene_ids"]}
    args.out_dir.mkdir(parents=True)
    np.savez_compressed(args.out_dir / "train.npz", **train)
    np.savez_compressed(args.out_dir / "external_inputs.npz", **queries)
    np.savez_compressed(args.out_dir / "external_truth.npz", target=np.asarray(target), task_ids=queries["task_ids"])
    np.savez_compressed(args.out_dir / "external_arms.npz", arm_logcpm=panel["arm_logcpm"][np.ix_(arm_indices, panel_indices)],
                        original_arm_indices=np.asarray(arm_indices), panel_gene_indices=panel_indices,
                        sample_ids=np.asarray([arm_metadata[i]["sample_id"] for i in arm_indices]))
    with (args.out_dir / "axis.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bridge[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(bridge)
    write_json(args.out_dir / "excluded_genes.json", excluded)
    write_json(args.out_dir / "task_metadata.json", tasks)
    manifest = {"scope": "Retrospective external-study transfer on three pre-inspected human/rat tasks; no independent prospective or previously unseen dataset claim.",
                "training_rows": len(train["source"]), "training_compounds": len(set(train["compounds"])),
                "external_tasks": queries["task_ids"].tolist(), "external_unique_measurement_columns": 36,
                "shared_genes": len(bridge), "id_bridge_candidates": len(observed), "nonfinite_genes_excluded": len(excluded),
                "context_feature_names": ["time_h", "log10_rat_M", "log10_human_M"],
                "nominal_category_handling": "No cross-study nominal-dose categories invented; models are refit on three transferable continuous context features using training labels only.",
                "feature_selection": "Exact IDs and complete observation across required human/rat arms. No global multispecies mask, target variance, magnitude, differential expression or prediction-based filtering.",
                "quantities": {"train": "treatment-control RMA log2 condition means", "external": "mean treated log2(CPM+1) minus mean matched vehicle log2(CPM+1)"},
                "response_reconstruction": reconstruction_errors,
                "chemical_identity": chemistry_manifest["accepted"],
                "input_provenance": {"training_manifest": sha(args.training_dir / "manifest.json"), "panel_manifest": sha(args.panel_dir / "manifest.json"),
                                     "bridge_audit": sha(args.bridge_dir / "audit.json"), "bridge_tsv": sha(bridge_path),
                                     "task_manifest": sha(args.task_manifest), "chemical_manifest": sha(args.chemical_dir / "manifest.json")},
                "prepare_script_sha256": sha(__file__),
                "limitations": ["External human responses were examined in earlier project comparisons; this is a fixed subsequent external-study stress test.",
                                "Only three drugs and two pathway families; donor independence is unverified.",
                                "Microarray RMA and RNA-seq logCPM share log2 response form but have different measurement and low-expression behavior.",
                                "CDDO-Me100nM is below the minimum1uM TG training dose; all three molecular identities are absent from training.",
                                "Current common ID axis inherits the existing multi-species-curated panel axis; genes absent from that panel are not reconstructed."]}
    manifest["files"] = {p.name: sha(p) for p in args.out_dir.iterdir() if p.is_file()}
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["training_rows", "training_compounds", "external_tasks", "shared_genes", "nonfinite_genes_excluded"]}, indent=2))


if __name__ == "__main__":
    main()
