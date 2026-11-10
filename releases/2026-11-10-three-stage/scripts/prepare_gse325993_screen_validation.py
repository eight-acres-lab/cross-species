#!/usr/bin/env python3
"""Prepare public exp3 human validation inputs under the frozen exp2 transform.

No fitting, tuning, expression-dependent selection, effect computation or scoring.
NTC inference expression is physically separated from POU5F1 scoring expression.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import anndata as ad
import numpy as np
import pandas as pd

EXPECTED_EXP2_MANIFEST = "6e10f92ac12da0680dc266a8cdd886cd81ffa4a19ca78c13dbd0feef66659938"
EXPECTED_EXP2_AXIS = "b3c58343f4518a877ee349cd519d57ae72709a2b43a2286acd4ac32ddc781b02"
EXPECTED_EXP2_MASK = "4375d81c3f4371a37a34d06892bbe0803b3279b81b8da80ecaa48a3ce7f9884f"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def row_fingerprints(obj):
    x = obj.X.tocsr().astype(np.int32); x.sort_indices()
    result = defaultdict(list)
    for i, cell in enumerate(obj.obs_names):
        lo, hi = x.indptr[i:i + 2]
        fingerprint = hashlib.sha256(x.indices[lo:hi].astype(np.int32).tobytes() + x.data[lo:hi].tobytes()).hexdigest()
        result[fingerprint].append(str(cell))
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args(); root = args.root; p = root / "gse325993_public_exp3"; out = p / "normalized"; out.mkdir(exist_ok=True)
    native_dir = p / "prepared"; native_dir.mkdir(exist_ok=True)
    old = root / "gse325993_public_exp2/normalized"
    assert sha(old / "manifest.json") == EXPECTED_EXP2_MANIFEST
    assert sha(old / "state_observed_gene_axis.tsv") == EXPECTED_EXP2_AXIS
    assert sha(old / "human.gencode32_coding_mask.tsv") == EXPECTED_EXP2_MASK
    exp2_manifest = json.loads((old / "manifest.json").read_text())
    axis = pd.read_csv(old / "state_observed_gene_axis.tsv", sep="\t")
    mask_table = pd.read_csv(old / "human.gencode32_coding_mask.tsv", sep="\t", dtype=str)
    assert len(axis) == 6357 and axis.native_index.is_unique and axis.human_ensembl_gene_id.is_unique
    mask = mask_table.in_coding_denominator.eq("True").to_numpy()
    assert int(mask.sum()) == 19394
    metadata_source = root / "gse325993_public_exp1/audit/public_cell_guide_and_line_metadata.tsv.gz"
    assert sha(metadata_source) == exp2_manifest["source_files"]["author_demux_and_guide_metadata"]["sha256"]
    metadata = pd.read_csv(metadata_source, sep="\t")
    human = metadata[(metadata.species == "human") & (metadata.experiment == "exp3")].copy()
    # This eligibility rule is frozen from exp2 and evaluated from guide UMIs only.
    eligible = human.author_demux_matched & human.passes_simple_dominant_rule & human.condition.isin(["NTC", "POU5F1"])
    assert len(human) == 12136 and eligible.all()
    sources = {r["file"]: r for r in json.loads((p / "download_manifest_verified.json").read_text())}
    assert len(sources) == 23
    authored_captures = set((root / "gse325993_public_exp1/author_metadata/ids.txt").read_text().splitlines())
    expected_captures = {x for x in authored_captures if x.startswith("exp3_")}
    assert expected_captures == set(human.capture) and len(expected_captures) == 23
    assert "exp3_chip3_lane8" not in authored_captures
    parts = []; file_qc = []; genes = None
    for name, info in sorted(sources.items()):
        assert "_HSapiens_exp3_" in name and name.endswith(".h5ad")
        f = p / name; assert sha(f) == info["sha256"] and f.stat().st_size == info["bytes"]
        a = ad.read_h5ad(f); a.X = a.X.tocsr(); a.X.sort_indices()
        assert np.isfinite(a.X.data).all() and (a.X.data >= 0).all() and (a.X.data == np.floor(a.X.data)).all()
        assert a.obs_names.is_unique and a.var_names.is_unique and not a.layers and a.raw is None
        assert a.var_names.tolist() == mask_table.gene_id.tolist()
        genes = a.var_names.tolist()
        capture = re.search(r"_(exp3_chip\d+_lane\d+)_", name)[1]
        m = human[human.capture == capture].set_index("barcode")
        assert m.index.is_unique and set(m.index) == set(a.obs_names)
        obs = m.loc[a.obs_names].copy(); obs["original_barcode"] = a.obs_names
        obs["source_matrix_file"] = name; obs["donor_id"] = obs.individual
        obs["cell_type"] = "induced pluripotent stem cell"; obs["time_days"] = 5; obs["study_id"] = "GSE325993"
        obs.index = ["human|" + capture + "|" + cell for cell in a.obs_names]
        a.obs = obs; a.X = a.X.astype(np.int32)
        a.obs["native_total_umi"] = np.asarray(a.X.sum(axis=1)).ravel()
        a.obs["native_detected_genes"] = np.diff(a.X.indptr)
        file_qc.append({**info, "cells": a.n_obs, "genes": a.n_vars, "capture": capture,
                        "exact_guide_and_author_demux_bridge": True, "integer_nonnegative_counts": True})
        parts.append(a)
    merged = ad.concat(parts, join="inner", merge="same", index_unique=None)
    assert merged.shape == (12136, 36606) and merged.obs_names.is_unique and merged.var_names.tolist() == genes
    assert merged.obs.groupby(["individual", "condition"], observed=True).size().to_dict() == {
        ("bv", "NTC"): 7238, ("bv", "POU5F1"): 230, ("cz", "NTC"): 4483, ("cz", "POU5F1"): 185}
    current_fingerprints = row_fingerprints(merged)
    assert all(len(ids) == 1 for ids in current_fingerprints.values())
    exp2_native_path = root / "gse325993_public_exp2/prepared/gse325993_exp2_human.native_counts.h5ad"
    assert sha(exp2_native_path) == exp2_manifest["source_files"]["human_counts"]["sha256"]
    exp2_native = ad.read_h5ad(exp2_native_path)
    assert exp2_native.var_names.tolist() == merged.var_names.tolist()
    old_fingerprints = row_fingerprints(exp2_native)
    identical_between_screens = set(old_fingerprints) & set(current_fingerprints)
    assert not identical_between_screens and not set(exp2_native.obs_names) & set(merged.obs_names)
    bare_barcode_overlap = len(set(exp2_native.obs.original_barcode) & set(merged.obs.original_barcode))
    del exp2_native, old_fingerprints, current_fingerprints, parts
    merged.uns["provenance"] = {"source": "GSE325993 public human exp3 GEX matrices", "counts_location": "X", "counts_units": "integer UMI",
                                "normalization": "none", "new_expression_filter": "none", "new_guide_filter": "same frozen exp2 strict dominant-UMI eligibility; all public exp3 human cells pass",
                                "independent_unit": "new screen of same two genotype lines; capture channels are technical units",
                                "upstream_NTC_caveat": exp2_manifest["upstream_cohort_caveat"]}
    native_path = native_dir / "gse325993_exp3_human.native_counts.h5ad"; merged.write_h5ad(native_path, compression="gzip")
    merged.obs.to_csv(native_dir / "human.cell_metadata.tsv.gz", sep="\t")
    counts = merged.obs.groupby(["individual", "cell_line", "experiment", "capture", "condition", "dominant_guide"], observed=True).size().rename("n_cells").reset_index()
    counts.to_csv(out / "perline_capture_condition_guide_counts.tsv", sep="\t", index=False)
    denominators = np.asarray(merged.X[:, mask].sum(axis=1)).ravel().astype(np.float64)
    assert np.isfinite(denominators).all() and (denominators > 0).all()
    cols = merged.var_names.get_indexer(axis.human_ensembl_gene_id); assert (cols >= 0).all()
    expression = merged.X[:, cols].toarray().astype(np.float64)
    expression *= 10000.; expression /= denominators[:, None]; np.log1p(expression, out=expression)
    expression = expression.astype(np.float32)
    metadata_arrays = {"cell_ids": merged.obs_names.to_numpy(dtype=str), "full_row_indices": np.arange(merged.n_obs, dtype=np.int64),
                       "coding_library_size": denominators, "native_gene_indices": axis.native_index.to_numpy(np.int64),
                       "human_gene_ids": axis.human_ensembl_gene_id.to_numpy(dtype=str)}
    for key, col in [("donors", "individual"), ("cell_lines", "cell_line"), ("condition", "condition"), ("guide", "dominant_guide"), ("capture", "capture"), ("experiment", "experiment")]:
        metadata_arrays[key] = merged.obs[col].astype(str).to_numpy(dtype=str)
    assert all(v.dtype != object for v in metadata_arrays.values())
    cohort_meta = out / "cohort_metadata.npz"; np.savez_compressed(cohort_meta, **metadata_arrays)
    all_arrays = {"human_expression": expression, "native_gene_indices": metadata_arrays["native_gene_indices"], "human_gene_ids": metadata_arrays["human_gene_ids"]}
    for k, v in metadata_arrays.items():
        if k not in ["native_gene_indices", "human_gene_ids"]: all_arrays["human_" + k] = v
    full_arrays = out / "gse325993_exp3_human_arrays.npz"; np.savez_compressed(full_arrays, **all_arrays)
    artifacts = {}
    for label, condition in [("query_ntc", "NTC"), ("target_POU5F1", "POU5F1")]:
        take = metadata_arrays["condition"] == condition
        matrix_path = out / (label + "_expression.npy"); np.save(matrix_path, expression[take], allow_pickle=False)
        subset = {k: v if k in ["native_gene_indices", "human_gene_ids"] else v[take] for k, v in metadata_arrays.items()}
        meta_path = out / (label + "_metadata.npz"); np.savez_compressed(meta_path, **subset)
        assert np.array_equal(np.load(matrix_path, mmap_mode="r", allow_pickle=False), expression[take])
        with np.load(meta_path, allow_pickle=False) as z: assert all(np.array_equal(z[k], v) for k, v in subset.items())
        artifacts[label] = {"expression": record(matrix_path), "metadata": record(meta_path), "cells": int(take.sum()), "condition": condition,
                            "shape": [int(take.sum()), len(axis)], "expression_dtype": "float32", "metadata_contains_expression": False}
    assert artifacts["query_ntc"]["cells"] == 11721 and artifacts["target_POU5F1"]["cells"] == 415
    with np.load(full_arrays, allow_pickle=False) as z: assert all(np.array_equal(z[k], v) for k, v in all_arrays.items())
    shutil.copyfile(old / "state_observed_gene_axis.tsv", out / "state_observed_gene_axis.tsv")
    shutil.copyfile(old / "human.gencode32_coding_mask.tsv", out / "human.gencode32_coding_mask.tsv")
    for name, r in sources.items(): assert sha(p / name) == r["sha256"]
    assert sha(old / "manifest.json") == EXPECTED_EXP2_MANIFEST
    assert sha(exp2_native_path) == exp2_manifest["source_files"]["human_counts"]["sha256"]
    anomalies = metadata[(metadata.experiment == "exp3") & ~metadata.passes_simple_dominant_rule]
    report = {"schema_version": "gse325993_exp3_human_validation_v1", "status": "passed", "scope": "Input-only independent-screen validation; no models, tuning, new gene/cell selection, effects or scoring",
              "script": record(Path(__file__)), "exp2_reference_manifest": record(old / "manifest.json"),
              "source_files": file_qc, "author_demux_metadata": record(metadata_source),
              "native_counts": record(native_path), "arrays": {**record(full_arrays), "shape": list(expression.shape), "dtype": str(expression.dtype), "use": "independent scorer only; inference reads only query_ntc artifact"},
              "cohort_metadata": {**record(cohort_meta), "contains_expression": False}, "query_ntc": artifacts["query_ntc"], "target_POU5F1": artifacts["target_POU5F1"],
              "axis": {**record(out / "state_observed_gene_axis.tsv"), "exact_exp2_axis": True, "genes": 6357},
              "coding_mask": {**record(out / "human.gencode32_coding_mask.tsv"), "exact_exp2_mask": True, "genes": 19394,
                              "GENCODE32_reference_protein_coding_genes": 19994, "reference_coding_genes_missing_from_native_axis": 600,
                              "native_axis_features": 36606, "output_axis_protein_coding": 6271, "output_axis_lncRNA": 86},
              "normalization": exp2_manifest["normalization"],
              "eligibility": {"all_public_exp3_human_cells": 12136, "retained": 12136, "excluded": 0, "new_expression_filter": False,
                              "rule": "Exact author demux + same exp2 strict dominant guideUMI threshold + NTC/POU5F1 target identity", "other_species_anomalies_not_in_scope": anomalies[["species", "individual", "capture", "barcode", "dominant_guide", "condition"]].to_dict("records")},
              "perline_condition_counts": merged.obs.groupby(["individual", "cell_line", "condition"], observed=True).size().rename("n_cells").reset_index().to_dict("records"),
              "capture_provenance": {"n_public_captures": 23, "missing_from_geo_and_author_capture_list": "exp3_chip3_lane8", "missing_reason": "not stated; not inferred to have zero valid cells", "author_capture_list": record(root / "gse325993_public_exp1/author_metadata/ids.txt")},
              "reuse_audit": {"exp3_duplicate_entire_count_vectors": 0, "exp2_exp3_identical_entire_count_vectors": 0, "exp2_exp3_overlap_full_capture_cell_ids": 0, "bare_barcodes_shared_across_screens": bare_barcode_overlap, "identity_rule": "experiment_chip_lane_barcode withspecies; barebarcode collisions do not constitute cell reuse"},
              "screen_limitations": ["Same two human genotype lines, new screen; no new donors or new study", "exp2 TF27 versus exp3 TF76 guide library; POU5F1 target same, guide composition changes", "Frozen exp2 weights, hyperparameters and gene coordinates; these inputs do not authorize adaptation"],
              "metadata_only_guide_composition": record(p / "metadata_guide_composition.json"),
              "pre_download_eligibility": record(p / "metadata_eligibility_frozen_before_download.json"),
              "upstream_cohort_caveat": exp2_manifest["upstream_cohort_caveat"], "raw_exp2_and_exp3_sources_unchanged": True,
              "array_roundtrip_exact": True, "free_disk_bytes_after_preparation": shutil.disk_usage(p).free}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cells": 12136, "query_cells": 11721, "target_cells": 415,
                      "manifest_sha256": sha(out / "manifest.json"), "full_arrays": report["arrays"], "query_ntc": report["query_ntc"],
                      "target_POU5F1": report["target_POU5F1"], "free_disk_bytes": report["free_disk_bytes_after_preparation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
