"""Independently reconstruct the external-study input contract on the server.

Uses original ID bridges and curated arm values directly, without importing the
production input preparer or any model code. Output contains integrity facts,
not external target values or response magnitude summaries.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


class Checks:
    def __init__(self):
        self.passed = 0
        self.numeric = 0
        self.maximum = 0.

    def check(self, condition, name):
        if not condition:
            raise AssertionError(name)
        self.passed += 1

    def exact(self, actual, expected, name):
        self.check(np.array_equal(actual, expected, equal_nan=True)
                   if np.asarray(actual).dtype.kind == "f" and np.asarray(expected).dtype.kind == "f"
                   else np.array_equal(actual, expected), name)
        if np.asarray(actual).dtype.kind in "fiu":
            self.numeric += np.asarray(actual).size

    def close(self, actual, expected, name, tolerance=1e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape, name + ": shape")
        self.check(np.array_equal(np.isfinite(actual), np.isfinite(expected)), name + ": finite mask")
        mask = np.isfinite(actual)
        difference = float(np.max(np.abs(actual[mask] - expected[mask]))) if mask.any() else 0.
        self.maximum = max(self.maximum, difference)
        self.numeric += actual.size
        self.check(difference <= tolerance, name + ": values")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/external_transfer_04"))
    parser.add_argument("--training-dir", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/crossspecies_direct_03/bundle_v1"))
    parser.add_argument("--panel-dir", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/multisource/data/GSE330151"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, checks = args.root, Checks()
    prepared = root / "inputs_v1"
    metadata = root / "metadata"
    bridge_dir, chemistry_dir = metadata / "gene_bridge", metadata / "chemical_features"
    task_path = metadata / "hepatocyte_audit/gse330151_task_manifest.json"
    manifest = read_json(prepared / "manifest.json")
    tg_manifest, panel_manifest = read_json(args.training_dir / "manifest.json"), read_json(args.panel_dir / "manifest.json")
    bridge_audit, chemistry_manifest = read_json(bridge_dir / "audit.json"), read_json(chemistry_dir / "manifest.json")
    for folder, file_map in [(prepared, manifest["files"]), (args.training_dir, tg_manifest["files"]),
                             (args.panel_dir, panel_manifest["files"]), (bridge_dir, bridge_audit["files"])]:
        for name, expected in file_map.items():
            checks.check(sha(folder / name) == expected, "frozen hash " + str(folder / name))
    checks.check(sha(chemistry_dir / "features.npz") == chemistry_manifest["feature_sha256"], "external feature hash")
    provenance = {"training_manifest": args.training_dir / "manifest.json", "panel_manifest": args.panel_dir / "manifest.json",
                  "bridge_audit": bridge_dir / "audit.json", "bridge_tsv": bridge_dir / "GSE330151_bridge.tsv",
                  "task_manifest": task_path, "chemical_manifest": chemistry_dir / "manifest.json"}
    for name, path in provenance.items():
        checks.check(manifest["input_provenance"][name] == sha(path), "input provenance " + name)
    with np.load(args.training_dir / "bundle.npz", allow_pickle=False) as source:
        tg = {name: source[name].copy() for name in source.files}
    with np.load(args.panel_dir / "panel.npz", allow_pickle=False) as source:
        # Explicit original arm input; response arrays are not used to construct values.
        panel = {name: source[name].copy() for name in ["feature_ids", "species_order", "native_gene_ids", "arm_logcpm"]}
    arms = read_json(args.panel_dir / "arm_metadata.json")
    contrasts = read_json(args.panel_dir / "contrast_metadata.json")
    tasks = read_json(task_path)
    with (bridge_dir / "GSE330151_bridge.tsv").open() as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    task_ids = ["CDDO", "KI696", "THAP"]
    checks.check([row["perturbation_code"] for row in tasks] == task_ids, "fixed task order")
    unit_to_m = {"M": Decimal(1), "mM": Decimal(".001"), "uM": Decimal(".000001"), "μM": Decimal(".000001"), "nM": Decimal(".000000001")}
    arm_selection, actual_spec = set(), {}
    context = []
    for task in tasks:
        code = task["perturbation_code"]
        dose = {}
        for species in ["human", "rat"]:
            key = "GSE330151|" + species + "|" + code
            matching = [row for row in contrasts if row["contrast_id"] == key]
            checks.check(len(matching) == 1, "unique original contrast " + key)
            reference = matching[0]
            treated = [i for i, row in enumerate(arms) if row["species"] == species and row["contrast_id"] == key and row["role"] == "treated"]
            controls = [i for i, row in enumerate(arms) if row["species"] == species and row["contrast_id"] == key and row["role"] == "vehicle"]
            checks.check(len(treated) == len(controls) == 3, "three arms " + key)
            checks.check(not set(treated) & set(controls), "no self-control " + key)
            side = task["sides"][species]
            checks.check(side["contrast_id"] == key, "task contrast identity " + key)
            for role, indices in [("treated", treated), ("control", controls)]:
                checks.check(indices == side[role + "_arm_rows"], "independently derived arm indices " + key + role)
                checks.check([arms[i]["sample_id"] for i in indices] == side[role + "_sample_ids"], "sample identities " + key + role)
                checks.check(indices == reference[role + "_arm_rows"], "contrast metadata agrees " + key + role)
                checks.check(all(float(arms[i]["time_h"]) == float(task["time_h"]) for i in indices), "matched time " + key + role)
                arm_selection.update(indices)
            observed_doses = {Decimal(str(arms[i]["dose_value"])) * unit_to_m[arms[i]["dose_unit"]] for i in treated}
            checks.check(len(observed_doses) == 1, "single original exposure " + key)
            dose[species] = next(iter(observed_doses))
            checks.check(dose[species] == Decimal(str(task[species + "_M"])), "original exposure matches task " + key)
            checks.check(side["units"] == "log2(CPM+1)", "declared arm quantity " + key)
            actual_spec[(code, species)] = (treated, controls)
        context.append([float(task["time_h"]), np.log10(float(dose["rat"])), np.log10(float(dose["human"]))])
    arm_selection = sorted(arm_selection)
    checks.check(len(arm_selection) == 36, "exact36 unique humanrat measurements")
    checks.check({arms[i]["species"] for i in arm_selection} == {"human", "rat"}, "other species excluded")
    species_order = panel["species_order"].tolist()
    tg_index, panel_index = [], []
    for row in candidates:
        t, p = int(row["tg_axis_index_zero_based"]), int(row["GSE330151_panel_index_zero_based"])
        checks.check(str(tg["human_gene_ids"][t]) == row["human_entrez"] and str(tg["rat_gene_ids"][t]) == row["rat_entrez"], "training ID linkage")
        checks.check(str(panel["feature_ids"][p]) == row["human_ensembl"] and
                     str(panel["native_gene_ids"][species_order.index("rat"), p]) == row["rat_ensembl"], "external ID linkage")
        tg_index.append(t); panel_index.append(p)
    tg_index, panel_index = np.asarray(tg_index), np.asarray(panel_index)
    observed = np.isfinite(panel["arm_logcpm"][np.ix_(arm_selection, panel_index)]).all(axis=0)
    selected_axis = [row for row, keep in zip(candidates, observed) if keep]
    ti, pi = tg_index[observed], panel_index[observed]
    checks.check(len(ti) == len(set(ti)) and len(pi) == len(set(pi)), "unique common axis")
    with (prepared / "axis.tsv").open() as handle:
        actual_axis = list(csv.DictReader(handle, delimiter="\t"))
    checks.check(actual_axis == selected_axis, "ID/presence-only axis exactly reproduced")
    expected_exclusions = [dict(row, exclusion="not_finite_in_all_selected_human_rat_arms") for row, keep in zip(candidates, observed) if not keep]
    checks.check(read_json(prepared / "excluded_genes.json") == expected_exclusions, "all missing feature exclusions exact")
    source_values, target_values = [], []
    for code in task_ids:
        for species, destination in [("rat", source_values), ("human", target_values)]:
            treated, controls = actual_spec[(code, species)]
            treated_values = np.stack([panel["arm_logcpm"][i, pi] for i in treated])
            control_values = np.stack([panel["arm_logcpm"][i, pi] for i in controls])
            destination.append(np.mean(treated_values, axis=0) - np.mean(control_values, axis=0))
    with np.load(prepared / "train.npz", allow_pickle=False) as output:
        checks.check(set(output.files) == set(tg), "training field set preserved")
        for name in tg:
            expected = tg[name][:, ti] if name in {"source", "target"} else tg[name][ti] if name in {"human_gene_ids", "rat_gene_ids"} else tg[name][:, :3] if name == "context" else tg[name]
            checks.exact(output[name], expected, "training field " + name)
    with np.load(prepared / "external_inputs.npz", allow_pickle=False) as output:
        checks.check(set(output.files) == {"source", "context", "fingerprints", "descriptors", "task_ids", "human_gene_ids", "rat_gene_ids"}, "external inference interface excludes human truth")
        checks.close(output["source"], source_values, "rat responses rebuilt from original arms")
        checks.close(output["context"], context, "three continuous external contexts")
        checks.exact(output["task_ids"], task_ids, "inference task order")
        checks.exact(output["human_gene_ids"], tg["human_gene_ids"][ti], "inference human axis")
        checks.exact(output["rat_gene_ids"], tg["rat_gene_ids"][ti], "inference rat axis")
        with np.load(chemistry_dir / "features.npz", allow_pickle=False) as chemistry:
            for task_index, code in enumerate(task_ids):
                identity = next(row for row in chemistry_manifest["accepted"] if row["panel_perturbation"] == code)
                positions = np.flatnonzero(chemistry["compounds"] == identity["compound"])
                checks.check(len(positions) == 1, "exact chemical identifier " + code)
                pos = int(positions[0])
                checks.check(bool(chemistry["verified_mask"][pos]), "verified chemical row " + code)
                checks.exact(output["fingerprints"][task_index], chemistry["fingerprints"][pos], "chemical fingerprint order " + code)
                checks.exact(output["descriptors"][task_index], chemistry["descriptors"][pos], "chemical descriptor order " + code)
                checks.check(not identity["same_cid_in_strict_training"] and not identity["same_inchikey_in_strict_training"], "identity-overlap flag " + code)
    with np.load(prepared / "external_truth.npz", allow_pickle=False) as output:
        checks.check(set(output.files) == {"target", "task_ids"}, "scoring truth separate fields")
        checks.close(output["target"], target_values, "human responses rebuilt from original arms")
        checks.exact(output["task_ids"], task_ids, "truth task order")
    with np.load(prepared / "external_arms.npz", allow_pickle=False) as output:
        checks.exact(output["original_arm_indices"], arm_selection, "saved original arm positions")
        checks.exact(output["panel_gene_indices"], pi, "saved original feature positions")
        checks.exact(output["sample_ids"], [arms[i]["sample_id"] for i in arm_selection], "saved exact GSM order")
        checks.exact(output["arm_logcpm"], panel["arm_logcpm"][np.ix_(arm_selection, pi)], "all saved arm values")
    checks.check(manifest["shared_genes"] == len(ti) and manifest["id_bridge_candidates"] == len(candidates), "manifest feature counts")
    checks.check(manifest["nonfinite_genes_excluded"] == int((~observed).sum()), "manifest exclusions")
    checks.check(manifest["training_rows"] == len(tg["source"]) == 678, "training678 rows")
    checks.check(manifest["context_feature_names"] == ["time_h", "log10_rat_M", "log10_human_M"], "portable contexts declared")
    result = {"status": "pass", "checks_passed": checks.passed, "checks_failed": 0,
              "numeric_values_compared": checks.numeric, "max_abs_difference": checks.maximum,
              "training_conditions": len(tg["source"]), "training_compounds": len(set(tg["compounds"])),
              "id_bridge_candidates": len(candidates), "shared_genes": len(ti),
              "nonfinite_human_rat_genes_excluded": int((~observed).sum()),
              "external_tasks": task_ids, "external_unique_sample_columns": len(arm_selection),
              "other_species_mask_used": False, "human_truth_excluded_from_inference_file": True,
              "external_human_values_or_magnitude_summaries_emitted": False,
              "method": "Direct reconstruction from original TG bundle, curated external arm values, sample metadata, exact-ID bridge and identity-verified chemistry; no import of production preparation functions.",
              "prepared_manifest_sha256": sha(prepared / "manifest.json"),
              "auditor_sha256": sha(__file__),
              "limitations": ["This is an input-contract verification, not performance or biological-generalization evidence.",
                              "Cross-platform expression units and prior inspection of external labels remain as declared in the frozen protocol.",
                              "Source-input finiteness can use human observation presence; it does not use human response magnitude or predictions."]}
    output = args.output or root / "verification/independent_input_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
