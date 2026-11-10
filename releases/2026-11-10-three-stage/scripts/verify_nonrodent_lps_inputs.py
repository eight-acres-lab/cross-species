#!/usr/bin/env python3
"""Independent read-only validation from native counts to frozen LPS arrays.

This script imports no experiment preparation or training code. It recomputes all
selected expression values, coding masks, exact gene joins, and selection budgets.
It never opens model predictions or calculates a treatment effect or outcome.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def require(value, label):
    if not value:
        raise AssertionError(label)


def table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def order(ids, namespace):
    return sorted(ids, key=lambda x: (hashlib.sha256((namespace + "|" + x).encode()).hexdigest(), x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True, type=Path); ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(); base = args.root; prep = base / "prepared"
    manifest = json.loads((prep / "manifest.json").read_text())
    selection = json.loads((prep / "selections.json").read_text())
    refs = {k: Path(v["path"]) for k, v in manifest["source_files"].items()}
    frozen = {str(p): digest(p) for p in [*refs.values(), prep / "lps_primary_arrays.npz", prep / "selections.json", prep / "manifest.json", prep / "primary_gene_axis.tsv"]}
    additional = [base / "animal_lps/native_gene_id_bridge.tsv", base / "animal_lps/orthology/rabbit.raw.tsv",
                  base / "animal_lps/orthology/rabbit.source.json", base / "animal_lps/orthology/rabbit.coverage.json",
                  base / "normalization/coding_Homo_sapiens.GRCh38.84.chr.txt",
                  base / "normalization/coding_Oryctolagus_cuniculus.OryCun2.0.84.txt"]
    frozen.update({str(p): digest(p) for p in additional})
    ortho_source = json.loads((base / "animal_lps/orthology/rabbit.source.json").read_text())
    ortho_coverage = json.loads((base / "animal_lps/orthology/rabbit.coverage.json").read_text())
    require(ortho_source["biomart_release"] == 116, "pinned orthology release")
    require(digest(base / "animal_lps/orthology/rabbit.raw.tsv") == ortho_source["sha256"], "raw orthology hash")
    require(digest(base / "animal_lps/native_gene_id_bridge.tsv") == ortho_coverage["state_axis_sha256"], "pinned model bridge hash")
    for k, v in manifest["source_files"].items():
        require(frozen[v["path"]] == v["sha256"], "source hash: " + k)
    for k in ["arrays", "selections", "primary_gene_axis"]:
        require(digest(manifest[k]["path"]) == manifest[k]["sha256"], "prepared hash: " + k)
    z = np.load(prep / "lps_primary_arrays.npz", allow_pickle=False)
    human = ad.read_h5ad(refs["human_counts"]); rabbit = ad.read_h5ad(refs["rabbit_counts"])
    axis = table(prep / "primary_gene_axis.tsv")
    require(len(axis) == 4633, "fixed axis size")
    for column in ["human_ensembl", "animal_ensembl", "state_index"]:
        require(axis[column].is_unique, "unique axis " + column)
    require((axis.orthology_type == "ortholog_one2one").all() and (axis.confidence == "1").all(), "one2one/high confidence")
    indices = axis.state_index.astype(int).to_numpy()
    require(np.all(np.diff(indices) > 0) and indices.min() >= 0 and indices.max() < 6546, "model index order/range")
    require(np.array_equal(indices, z["native_gene_indices"]), "array model coordinates")
    require(np.array_equal(axis.human_ensembl, z["human_gene_ids"]) and np.array_equal(axis.animal_ensembl, z["rabbit_gene_ids"]), "array native IDs")
    bridge = table(base / "animal_lps/native_gene_id_bridge.tsv")
    bridge = bridge[(bridge.human_ensembl_gene_id != "") & (bridge.author_human_axis_observed == "True")]
    require(bridge.human_ensembl_gene_id.is_unique, "bridge unique Ensembl IDs")
    model_index = dict(zip(bridge.human_ensembl_gene_id, bridge.native_index.astype(int)))
    require(all(model_index[h] == i for h, i in zip(axis.human_ensembl, indices)), "verified checkpoint coordinates")
    raw = pd.read_csv(base / "animal_lps/orthology/rabbit.raw.tsv", sep="\t", header=None, dtype=str, keep_default_na=False,
                      names=["h", "hs", "r", "rs", "type", "confidence"])
    raw = raw[(raw.type == "ortholog_one2one") & (raw.confidence == "1") & (raw.hs != "")].drop_duplicates()
    # Independently count alternative partners; all ambiguous relations are rejected.
    for a, b in [("h", "r"), ("r", "h"), ("hs", "h")]:
        choices = raw.groupby(a)[b].nunique(); raw = raw[raw[a].isin(choices[choices == 1].index)]
    native_h, native_r = set(human.var_names), set(rabbit.var_names)
    expected = {(row.h, row.r, model_index[row.h]) for row in raw.itertuples() if row.h in model_index and row.h in native_h and row.r in native_r}
    actual = set(zip(axis.human_ensembl, axis.animal_ensembl, indices))
    require(expected == actual, "complete common measured axis reconstructed from raw orthology")

    normalization = {}
    coding_manifest = json.loads(refs["coding_manifest"].read_text())
    for species, obj, gene_column, refname in [("human", human, "human_ensembl", "coding_Homo_sapiens.GRCh38.84.chr.txt"),
                                              ("rabbit", rabbit, "animal_ensembl", "coding_Oryctolagus_cuniculus.OryCun2.0.84.txt")]:
        require(obj.obs_names.is_unique and obj.var_names.is_unique, species + " native unique axes")
        masktab = table(refs[species + "_coding_mask"])
        require(masktab.original_gene_id.tolist() == obj.var_names.tolist(), species + " mask axis")
        require(masktab.native_index.astype(int).tolist() == list(range(obj.n_vars)), species + " mask coordinates")
        coding_path = base / "normalization" / refname
        require(digest(coding_path) == coding_manifest["species"][species]["coding_reference_sha256"], species + " coding reference hash")
        coding = {line.split("\t", 1)[0].split(".", 1)[0] for line in coding_path.read_text().splitlines()}
        mask = np.array([g.split(".", 1)[0] in coding for g in obj.var_names])
        require(np.array_equal(mask, masktab.include_in_coding_denominator.eq("True")), species + " annotation mask regenerated")
        ids = z[species + "_cell_ids"].tolist(); require(len(set(ids)) == len(ids), species + " selected cell IDs unique")
        cells = obj.obs_names.get_indexer(ids); require(np.all(cells >= 0), species + " cell lookup")
        if species == "human":
            require(set(ids) == set(obj.obs_names), "all author human cells retained")
        genes = obj.var_names.get_indexer(axis[gene_column]); require(np.all(genes >= 0), species + " measured gene lookup")
        counts = obj.X[cells]
        require(np.isfinite(counts.data).all() and (counts.data >= 0).all() and np.equal(counts.data, np.floor(counts.data)).all(), species + " native counts")
        libraries = np.asarray(counts[:, mask].sum(axis=1)).ravel().astype(np.float64)
        require((libraries > 0).all() and np.array_equal(libraries, z[species + "_coding_library_size"]), species + " coding denominators exact")
        original = counts[:, genes].toarray().astype(np.float64)
        reconstructed = np.log1p((10000.0 * original) / libraries[:, None]).astype(np.float32)
        observed = z[species + "_expression"]
        require(observed.dtype == np.float32 and np.isfinite(observed).all(), species + " float32 finite")
        difference = np.abs(reconstructed.astype(np.float64) - observed.astype(np.float64))
        require(np.array_equal(reconstructed, observed), species + " every normalized value exact float32 reproduction")
        obs = obj.obs.iloc[cells]
        require(np.array_equal(obs.donor_id.astype(str), z[species + "_donors"]), species + " native donor identities")
        if species == "human":
            require(np.array_equal(obs.condition.astype(str), z["human_condition"]), "native human condition")
        else:
            require(np.array_equal(obs.time_h.to_numpy(), z["rabbit_time_h"]), "native rabbit time")
        normalization[species] = {"cells": len(cells), "output_genes": len(genes), "values_compared": int(observed.size),
                                  "all_float32_values_exact": True, "max_abs_difference": float(difference.max()),
                                  "denominator_genes": int(mask.sum()), "all_coding_denominators_exact": True,
                                  "coding_reference_genes_missing_from_native_axis": len(coding - {g.split(".",1)[0] for g in obj.var_names})}

    namespaces = selection["namespaces"]
    hd = z["human_donors"]; hc = z["human_condition"]; hi = z["human_cell_ids"]
    rd = z["rabbit_donors"]; rt = z["rabbit_time_h"]; ri = z["rabbit_cell_ids"]
    donor_order = order(set(hd), namespaces["donor"])
    require(sum(selection["human_donor_pairs"], []) == donor_order, "donor hash order")
    require(selection["rabbit_donor_order"] == order(set(rabbit.obs.donor_id.astype(str)), namespaces["rabbit"]), "rabbit donor hash order")
    require(selection["rabbit_donors_used"] == selection["rabbit_donor_order"][:2], "rabbit donors fixed first two")
    for position, donor in enumerate(selection["rabbit_donors_used"]):
        max_control_prefix = max(min(selection["human_training_control_cap"], int(np.sum((hd == pair[position]) & (hc == "Baseline")))) for pair in selection["human_donor_pairs"])
        require(selection["rabbit_null_offsets"][donor] == max_control_prefix, "null offset equals independently computed maximum control prefix")
    test_counts = Counter(); global_c, global_cprime, all_rabbit = set(), set(), set(); fold_results = []
    human_roles = ["human_base", "human_extra", "human_cal", "human_test"]
    for fold in selection["folds"]:
        roles = {k: {item["donor"] for item in fold[k]} for k in human_roles}
        require([len(roles[k]) for k in human_roles] == [4, 2, 2, 2], "per-fold donor role counts")
        require(set.union(*roles.values()) == set(hd), "per-fold donor coverage")
        require(all(not roles[a] & roles[b] for a, b in itertools.combinations(human_roles, 2)), "per-fold donor role disjoint")
        test_counts.update(roles["human_test"])
        for role in human_roles:
            for item in fold[role]:
                for which, condition in [("control", "Baseline"), ("target", "LPS")]:
                    ix = np.asarray(item[which + "_rows"], dtype=int); ids = item[which + "_cell_ids"]
                    require(len(set(ix)) == len(ix) and np.all((ix >= 0) & (ix < len(hi))), "human rows unique/range")
                    require(hi[ix].tolist() == ids and np.all(hd[ix] == item["donor"]) and np.all(hc[ix] == condition), "human row donor/condition identity")
                    eligible = order(hi[(hd == item["donor"]) & (hc == condition)].tolist(), namespaces["cell"] + "/human")
                    if which == "control" and role in ["human_base", "human_extra"]:
                        eligible = eligible[:selection["human_training_control_cap"]]
                    require(ids == eligible and len(ix) == item["n_" + which], "complete/capped hash-sorted human selection")
                require(not set(item["control_rows"]) & set(item["target_rows"]), "human control target disjoint")
                require(item["n_unique_cells"] == item["n_control"] + item["n_target"], "human cell budget accounting")
        extra = {x["donor"]: x for x in fold["human_extra"]}
        for source, null in zip(fold["rabbit_source"], fold["rabbit_null"]):
            require(source["donor"] == null["donor"] and source["mapped_human_extra_donor"] == null["mapped_human_extra_donor"], "paired animal source/null")
            require(source["control_cell_ids"] == null["control_cell_ids"], "animal input control reuse identical")
            require(source["training_condition"] == null["training_condition"] == "LPS", "same encoded intervention")
            require(source["target_biological_condition"] == "LPS" and null["target_biological_condition"] == "null_control", "source/null biological label")
            comparator = extra[source["mapped_human_extra_donor"]]
            for item, target_time in [(source, 6), (null, 0)]:
                require(all(item[k] == comparator[k] for k in ["n_control", "n_target", "n_unique_cells"]), "source budgets equal human-extra per donor")
                require(item["target_time_h"] == target_time and item["input_time_h"] == 0, "declared animal times")
                for which, time_h in [("control", 0), ("target", target_time)]:
                    ix = np.asarray(item[which + "_rows"], dtype=int); ids = item[which + "_cell_ids"]
                    require(len(set(ix)) == len(ix) and np.all((ix >= 0) & (ix < len(ri))), "rabbit rows unique/range")
                    require(ri[ix].tolist() == ids and np.all(rd[ix] == item["donor"]) and np.all(rt[ix] == time_h), "rabbit row identity/time")
                    eligible = order(rabbit.obs_names[(rabbit.obs.donor_id.astype(str) == item["donor"]) & (rabbit.obs.time_h == time_h)].tolist(), namespaces["cell"] + "/rabbit")
                    offset = selection["rabbit_null_offsets"][item["donor"]] if item is null and which == "target" else 0
                    require(ids == eligible[offset:offset + item["n_" + which]], "rabbit exact deterministic prefix/offset")
                    all_rabbit.update(ids)
                require(not set(item["control_cell_ids"]) & set(item["target_cell_ids"]), "animal within-pool disjoint")
            global_c.update(source["control_cell_ids"]); global_cprime.update(null["target_cell_ids"])
        budgets = {role: sum(x["n_unique_cells"] for x in fold[role]) for role in ["human_base", "human_extra", "rabbit_source", "rabbit_null"]}
        require(budgets == fold["budget_unique_cells"] and budgets["human_extra"] == budgets["rabbit_source"] == budgets["rabbit_null"], "complete fold budget")
        fold_results.append({"fold": fold["fold"], "donor_roles_disjoint": True, "control_and_target_budgets_equal": True, "budgets": budgets})
    require(set(test_counts) == set(hd) and set(test_counts.values()) == {1}, "every human donor targeted once")
    require(not global_c & global_cprime, "global animal C/Cprime separation")
    require(all_rabbit == set(ri), "array rabbit union exactly matches selections")
    require(order(all_rabbit, namespaces["cell"] + "/rabbit_array") == ri.tolist(), "rabbit array deterministic order")
    require(set(rd) == set(selection["rabbit_donors_used"]), "rabbit two donors")
    require(all(digest(p) == h for p, h in frozen.items()), "all frozen inputs immutable")
    report = {"status": "passed", "scope": "Independent raw-count, coding-mask, gene-coordinate and selection validation; no preparation/training imports, predictions, outcome metrics or changes to frozen inputs",
              "script_sha256": digest(__file__), "source_hashes_verified_before_and_after": frozen,
              "normalization_formula": "float32(log1p(10000 * native_UMI / complete_observed_author_coding_UMI_sum))",
              "reference_versions": {"coding_gene_annotation": "author Ensembl84 all species", "animal_count_alignment": "CellRanger2.0/Ensembl84", "human_count_alignment": "CellRanger2.1.0/GRCh37 hg19", "orthology": "Ensembl116; exact stable-ID join", "model": "6546 named coordinates; verified human native gene bridge"},
              "normalization": normalization, "total_values_recomputed": sum(r["values_compared"] for r in normalization.values()),
              "gene_axis": {"size": 4633, "complete_exact_common_axis_reconstructed": True, "unique_high_confidence_one2one": True, "same_coordinates_all_arms": True},
              "folds": fold_results, "global_A0_input_target_disjoint": True,
              "global_rabbit_input_controls": len(global_c), "global_rabbit_null_targets": len(global_cprime),
              "each_target_donor_used_once": True, "selection_reproduced_from_hashes_and_native_metadata": True,
              "selection_cell_namespace_suffixes": {"human": "/human", "rabbit": "/rabbit", "rabbit_array": "/rabbit_array"},
              "frozen_h5ads_arrays_contracts_unchanged": True}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "normalization": normalization, "total_values_recomputed": report["total_values_recomputed"], "folds": len(fold_results)}, indent=2))


if __name__ == "__main__":
    main()
