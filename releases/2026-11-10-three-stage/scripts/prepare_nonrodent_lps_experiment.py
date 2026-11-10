#!/usr/bin/env python3
"""Freeze metadata-only selections and coding-normalized arrays for rabbit→human LPS.

Execute on the remote data host only. Source h5ad files remain immutable.
No expression-dependent selection, response ranking, fitting or scoring occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


NAMESPACE = "nonrodent_singlecell_20261110/v1"
EXPECTED_PAIRS = [
    ["202937150118_R06C01", "202937150118_R07C01"],
    ["202937150091_R02C01", "202937150118_R08C01"],
    ["202937150118_R03C01", "202937150118_R01C01"],
    ["202937150118_R02C01", "202937150118_R05C01"],
    ["202937150118_R04C01", "202937150091_R01C01"],
]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def ordered(values, namespace):
    return sorted(map(str, values), key=lambda item: (hashlib.sha256(f"{namespace}|{item}".encode()).hexdigest(), item))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population_record(donor, controls, targets, id_to_row, **extra):
    require(not (set(controls) & set(targets)), "input and target cells overlap")
    require(len(set(controls)) == len(controls), "duplicate control cell")
    require(len(set(targets)) == len(targets), "duplicate target cell")
    return {
        "donor": donor,
        "control_rows": [int(id_to_row[c]) for c in controls],
        "target_rows": [int(id_to_row[c]) for c in targets],
        "control_cell_ids": controls,
        "target_cell_ids": targets,
        "n_control": len(controls), "n_target": len(targets),
        "n_unique_cells": len(controls) + len(targets),
        **extra,
    }


def coding_mask(table_path, var_names, expected_n):
    table = pd.read_csv(table_path, sep="\t")
    require(np.array_equal(table.native_index.to_numpy(), np.arange(len(var_names))), "coding mask row index mismatch")
    require(table.original_gene_id.astype(str).tolist() == list(map(str, var_names)), "coding mask gene axis mismatch")
    flag = table.include_in_coding_denominator.astype(str).str.lower()
    require(flag.isin(["true", "false"]).all(), "invalid mask booleans")
    mask = flag.eq("true").to_numpy()
    require(int(mask.sum()) == expected_n, "unexpected coding denominator universe")
    return mask


def normalized(counts, mask, feature_indices):
    require(sparse.issparse(counts), "expected sparse original count matrix")
    require(np.isfinite(counts.data).all(), "nonfinite counts")
    require((counts.data >= 0).all(), "negative counts")
    require((counts.data == np.floor(counts.data)).all(), "fractional counts")
    coding_library = np.asarray(counts[:, mask].sum(axis=1, dtype=np.float64)).ravel()
    require((coding_library > 0).all(), "zero coding library: fail entire preparation; do not discard cells")
    require(mask[feature_indices].all(), "observed output gene outside coding denominator")
    values = counts[:, feature_indices].astype(np.float64).toarray()
    values *= 10000.0 / coding_library[:, None]
    np.log1p(values, out=values)
    result = values.astype(np.float32)
    require(np.isfinite(result).all() and (result >= 0).all(), "invalid normalized expression")
    return result, coding_library


def validate_selections(selections, arrays):
    human_ids = arrays["human_cell_ids"].astype(str)
    rabbit_ids = arrays["rabbit_cell_ids"].astype(str)
    human_donors = arrays["human_donors"].astype(str)
    human_condition = arrays["human_condition"].astype(str)
    rabbit_donors = arrays["rabbit_donors"].astype(str)
    rabbit_time = arrays["rabbit_time_h"]
    roles = {key: Counter() for key in ["human_base", "human_extra", "human_cal", "human_test"]}
    for fold in selections["folds"]:
        all_role_donors = []
        for key in roles:
            donor_set = {record["donor"] for record in fold[key]}
            all_role_donors.extend(donor_set)
            roles[key].update(donor_set)
            require(len(donor_set) == (4 if key == "human_base" else 2), "wrong role size")
            for record in fold[key]:
                for field, condition in [("control", "Baseline"), ("target", "LPS")]:
                    rows = np.asarray(record[field + "_rows"])
                    require(human_ids[rows].tolist() == record[field + "_cell_ids"], "human row↔cell ID mismatch")
                    require((human_donors[rows] == record["donor"]).all(), "human donor mismatch")
                    require((human_condition[rows] == condition).all(), "human condition mismatch")
                if key in ["human_base", "human_extra"]:
                    require(record["n_control"] <= 128, "training control cap violated")
                else:
                    require(record["n_control"] == int(((human_donors == record["donor"]) & (human_condition == "Baseline")).sum()), "endpoint control cells dropped")
                require(record["n_target"] == int(((human_donors == record["donor"]) & (human_condition == "LPS")).sum()), "LPS target cells dropped")
        require(len(all_role_donors) == len(set(all_role_donors)) == 10, "human role leakage")
        for extra, source, null in zip(fold["human_extra"], fold["rabbit_source"], fold["rabbit_null"]):
            require(source["mapped_human_extra_donor"] == null["mapped_human_extra_donor"] == extra["donor"], "source pairing mismatch")
            for key in ["n_control", "n_target", "n_unique_cells"]:
                require(source[key] == null[key] == extra[key], "source budget mismatch")
            require(source["control_rows"] == null["control_rows"], "A/A0 input pool mismatch")
            require(not set(source["control_rows"]) & set(null["target_rows"]), "A0 copies input cells")
            for record, target_time in [(source, 6), (null, 0)]:
                for field, time in [("control", 0), ("target", target_time)]:
                    rows = np.asarray(record[field + "_rows"])
                    require(rabbit_ids[rows].tolist() == record[field + "_cell_ids"], "rabbit row↔cell ID mismatch")
                    require((rabbit_donors[rows] == record["donor"]).all(), "rabbit donor mismatch")
                    require((rabbit_time[rows] == time).all(), "rabbit time mismatch")
    for key, counts in roles.items():
        require(len(counts) == 10 and set(counts.values()) == ({2} if key == "human_base" else {1}), "unbalanced donor role rotation")
    return {key: dict(counts) for key, counts in roles.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/hy-tmp/cross-species-feedback-20261108/nonrodent_singlecell_20261110"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    require(str(root).startswith("/hy-tmp/"), "Preparation is remote-only under /hy-tmp")
    output = (args.output or root / "prepared").resolve()
    require(str(output).startswith("/hy-tmp/"), "Expression outputs must remain remote")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "manifest.json").exists(), "Existing frozen manifest: choose a new named output rather than overwrite")
    paths = {
        "human_counts": root / "human_lps/lawlor2021_cd14_baseline_lps_counts.h5ad",
        "rabbit_counts": root / "animal_lps/prepared/rabbit.native_counts.h5ad",
        "shared_gene_map": root / "animal_lps/orthology/rabbit.human_state_measured.tsv",
        "human_coding_mask": root / "normalization/human.coding_denominator_mask.tsv",
        "rabbit_coding_mask": root / "normalization/rabbit.coding_denominator_mask.tsv",
        "coding_manifest": root / "normalization/coding_denominator_manifest.json",
        "script": Path(__file__).resolve(),
    }
    source_hashes = {key: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for key, path in paths.items()}
    print("Read frozen source matrices and metadata", flush=True)
    human = ad.read_h5ad(paths["human_counts"])
    rabbit = ad.read_h5ad(paths["rabbit_counts"])
    require(human.shape == (1644, 32738), "unexpected human source shape")
    require(rabbit.shape == (34528, 19293), "unexpected rabbit source shape")
    require(human.obs_names.is_unique and rabbit.obs_names.is_unique, "nonunique source IDs")
    require(human.obs.Celltype_Annotation.astype(str).eq("CD14_Mono").all(), "non-target human annotation")
    human_ids = human.obs_names.astype(str).to_numpy()
    human_donors = human.obs.donor_id.astype(str).to_numpy()
    human_conditions = human.obs.condition.astype(str).to_numpy()
    donor_order = ordered(set(human_donors), NAMESPACE + "/donor_roles")
    donor_pairs = [donor_order[i:i + 2] for i in range(0, len(donor_order), 2)]
    require(donor_pairs == EXPECTED_PAIRS, "donor roles differ from frozen review")
    rabbit_order = ordered(rabbit.obs.donor_id.astype(str).unique(), NAMESPACE + "/rabbit_roles")
    require(rabbit_order == ["rabbit2", "rabbit1", "rabbit3"], "rabbit order differs from frozen review")
    rabbit_donors_used = rabbit_order[:2]
    human_lookup = {cell: row for row, cell in enumerate(human_ids)}
    cell_salt = NAMESPACE + "/cell_order"
    human_pools = {}
    for donor in donor_order:
        human_pools[donor] = {
            condition: ordered(human_ids[(human_donors == donor) & (human_conditions == condition)], cell_salt + "/human")
            for condition in ["Baseline", "LPS"]
        }
    rabbit_source_ids = rabbit.obs_names.astype(str).to_numpy()
    rabbit_source_donors = rabbit.obs.donor_id.astype(str).to_numpy()
    rabbit_source_time = rabbit.obs.time_h.astype(int).to_numpy()
    rabbit_pools = {donor: {
        time: ordered(rabbit_source_ids[(rabbit_source_donors == donor) & (rabbit_source_time == time)], cell_salt + "/rabbit")
        for time in [0, 6]
    } for donor in rabbit_donors_used}
    # Reserve disjoint maximal C/C' blocks across every fold, using metadata sizes only.
    max_controls = {rabbit_donors_used[pos]: max(min(128, len(human_pools[pair[pos]]["Baseline"])) for pair in donor_pairs) for pos in [0, 1]}
    selections = {
        "schema_version": "nonrodent_lps_selection_v1",
        "namespaces": {"donor": NAMESPACE + "/donor_roles", "rabbit": NAMESPACE + "/rabbit_roles", "cell": cell_salt},
        "hash_order": "SHA256 UTF-8 namespace|full_ID ascending; ties full_ID ascending",
        "human_donor_pairs": donor_pairs,
        "rabbit_donor_order": rabbit_order,
        "rabbit_donors_used": rabbit_donors_used,
        "human_training_control_cap": 128,
        "human_target_rule": "all author-eligible LPS cells, every role",
        "cal_test_control_rule": "all author-eligible Baseline cells",
        "null_pool_rule": "Additional 0h cells after each rabbit's maximum input-control prefix across all folds; globally disjoint C and Cprime pools",
        "rabbit_null_offsets": max_controls,
        "folds": [],
    }
    selected_rabbit = set()
    for fold_index in range(5):
        role_pairs = {"human_test": fold_index, "human_cal": (fold_index + 1) % 5, "human_extra": (fold_index + 2) % 5}
        fold = {"fold": fold_index}
        for role in ["human_base", "human_extra", "human_cal", "human_test"]:
            donors = [donor for i, pair in enumerate(donor_pairs) if i not in role_pairs.values() for donor in pair] if role == "human_base" else donor_pairs[role_pairs[role]]
            fold[role] = []
            for donor in donors:
                controls = human_pools[donor]["Baseline"]
                if role in ["human_base", "human_extra"]:
                    controls = controls[:128]
                fold[role].append(population_record(donor, controls, human_pools[donor]["LPS"], human_lookup, study="Lawlor2021"))
        fold["rabbit_source"], fold["rabbit_null"] = [], []
        for donor, extra in zip(rabbit_donors_used, fold["human_extra"]):
            nc, nt = extra["n_control"], extra["n_target"]
            controls = rabbit_pools[donor][0][:nc]
            targets = rabbit_pools[donor][6][:nt]
            null_targets = rabbit_pools[donor][0][max_controls[donor]:max_controls[donor] + nt]
            require(len(controls) == nc and len(targets) == len(null_targets) == nt, "insufficient source pool")
            selected_rabbit.update(controls + targets + null_targets)
            # Temporary identity→identity mapping replaced by global array row indices below.
            for key, target_ids, target_time, role in [("rabbit_source", targets, 6, "LPS"), ("rabbit_null", null_targets, 0, "null_control")]:
                fold[key].append({"donor": donor, "mapped_human_extra_donor": extra["donor"], "control_cell_ids": controls, "target_cell_ids": target_ids,
                                  "n_control": nc, "n_target": nt, "n_unique_cells": nc + nt, "input_time_h": 0, "target_time_h": target_time,
                                  "training_condition": "LPS", "target_biological_condition": role, "study": "Hagai2018"})
        totals = {key: sum(record["n_unique_cells"] for record in fold[key]) for key in ["human_base", "human_extra", "rabbit_source", "rabbit_null"]}
        require(totals["human_extra"] == totals["rabbit_source"] == totals["rabbit_null"], "unequal incremental independent cell budget")
        fold["budget_unique_cells"] = totals
        selections["folds"].append(fold)
    rabbit_selected_ids = ordered(selected_rabbit, cell_salt + "/rabbit_array")
    rabbit_lookup = {cell: row for row, cell in enumerate(rabbit_selected_ids)}
    rabbit_original_rows = rabbit.obs_names.get_indexer(rabbit_selected_ids)
    require((rabbit_original_rows >= 0).all(), "selected rabbit ID absent from original")
    for fold in selections["folds"]:
        for role in ["rabbit_source", "rabbit_null"]:
            for record in fold[role]:
                record["control_rows"] = [rabbit_lookup[cell] for cell in record["control_cell_ids"]]
                record["target_rows"] = [rabbit_lookup[cell] for cell in record["target_cell_ids"]]
    gene_map = pd.read_csv(paths["shared_gene_map"], sep="\t").sort_values("state_index", kind="stable")
    require(len(gene_map) == 4633, "unexpected primary gene count")
    for col in ["state_index", "human_ensembl", "animal_ensembl"]:
        require(gene_map[col].is_unique, "ambiguous shared gene axis")
    native_indices = gene_map.state_index.to_numpy(dtype=np.int64)
    require(((native_indices >= 0) & (native_indices < 6546)).all(), "native index outside model axis")
    hi = human.var_names.get_indexer(gene_map.human_ensembl)
    ri = rabbit.var_names.get_indexer(gene_map.animal_ensembl)
    require((hi >= 0).all() and (ri >= 0).all(), "unobserved shared gene")
    hm = coding_mask(paths["human_coding_mask"], human.var_names, 19123)
    rm = coding_mask(paths["rabbit_coding_mask"], rabbit.var_names, 19293)
    hx, hd = normalized(human.X, hm, hi)
    rx, rd = normalized(rabbit.X[rabbit_original_rows], rm, ri)
    arrays = {
        "human_expression": hx, "human_cell_ids": human_ids.astype(str),
        "human_donors": human_donors.astype(str), "human_condition": human_conditions.astype(str),
        "rabbit_expression": rx, "rabbit_cell_ids": np.asarray(rabbit_selected_ids, dtype=str),
        "rabbit_donors": rabbit_source_donors[rabbit_original_rows].astype(str),
        "rabbit_time_h": rabbit_source_time[rabbit_original_rows].astype(np.int16),
        "native_gene_indices": native_indices,
        "human_gene_ids": gene_map.human_ensembl.to_numpy(dtype=str),
        "rabbit_gene_ids": gene_map.animal_ensembl.to_numpy(dtype=str),
        "human_coding_library_size": hd,
        "rabbit_coding_library_size": rd,
    }
    role_counts = validate_selections(selections, arrays)
    print("Metadata selections and all source budgets validated", flush=True)
    arrays_path = output / "lps_primary_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    with np.load(arrays_path, allow_pickle=False) as recovered:
        require(set(recovered.files) == set(arrays), "roundtrip field mismatch")
        for key in arrays:
            require(np.array_equal(recovered[key], arrays[key]), "roundtrip mismatch: " + key)
        validate_selections(selections, recovered)
    selections_path = output / "selections.json"
    selections_path.write_text(json.dumps(selections, indent=2) + "\n")
    gene_map.to_csv(output / "primary_gene_axis.tsv", sep="\t", index=False)
    pd.DataFrame({"cell_id": human_ids, "donor": human_donors, "condition": human_conditions, "coding_library_size": hd}).to_csv(output / "human_cell_normalization_qc.tsv", sep="\t", index=False)
    pd.DataFrame({"cell_id": rabbit_selected_ids, "donor": arrays["rabbit_donors"], "time_h": arrays["rabbit_time_h"], "coding_library_size": rd,
                  "original_source_row": rabbit_original_rows}).to_csv(output / "rabbit_selected_cell_normalization_qc.tsv", sep="\t", index=False)
    for key, path in paths.items():
        require(sha256(path) == source_hashes[key]["sha256"], "source mutated during preparation: " + key)
    manifest = {
        "schema_version": "nonrodent_lps_prepared_v1",
        "status": "metadata_count_normalization_and_roundtrip_checks_passed",
        "scope": "No fitting, model prediction, treatment-effect ranking, response-based selection, or outcome scores",
        "source_files": source_hashes,
        "arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path), "bytes": arrays_path.stat().st_size,
                   "fields": {key: {"shape": list(values.shape), "dtype": str(values.dtype)} for key, values in arrays.items()}},
        "selections": {"path": str(selections_path), "sha256": sha256(selections_path)},
        "primary_gene_axis": {"path": str(output / "primary_gene_axis.tsv"), "sha256": sha256(output / "primary_gene_axis.tsv"),
                              "n_observed": 4633, "native_full_axis_size": 6546, "input_loss_score_mask_identical_across_all_arms": True,
                              "excluded_coordinates": "Computational zero inputs only; never counted as observed target genes or scored"},
        "normalization": {"formula": "float32(log1p(10000 * original_UMI / full_observed_author_Ensembl84_coding_UMI_sum))",
                          "human_coding_gene_count": 19123, "rabbit_coding_gene_count": 19293,
                          "applied_before_selecting_4633_genes": True, "zero_coding_library_rule": "abort entire preparation; do not drop cells",
                          "human_library_min_max": [float(hd.min()), float(hd.max())], "rabbit_selected_library_min_max": [float(rd.min()), float(rd.max())]},
        "donor_role_counts": role_counts,
        "fold_budgets": [{"fold": f["fold"], **f["budget_unique_cells"]} for f in selections["folds"]],
        "counts": {"human_all_cells": len(human_ids), "human_donors": len(donor_order), "rabbit_selected_union_cells": len(rabbit_selected_ids),
                   "rabbit_donors": rabbit_donors_used, "rabbit_time_h": [0, 6]},
        "checks": {"raw_h5ad_immutable": True, "exact_source_budget_matching": True, "human_roles_disjoint_every_fold": True,
                   "no_duplicate_cells_within_populations": True, "A0_input_target_disjoint_all_folds": True,
                   "all_human_LPS_cells_kept_every_role": True, "all_cal_test_controls_kept": True,
                   "all_target_donors_appear_exactly_once": True, "npz_roundtrip_exact": True, "no_object_dtype_arrays": True},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": manifest["counts"], "fold_budgets": manifest["fold_budgets"], "npz_sha256": manifest["arrays"]["sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
