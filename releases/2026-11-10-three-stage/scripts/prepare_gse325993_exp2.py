#!/usr/bin/env python3
"""Verify public exp2 counts and attach exact author cell/guide metadata.

No normalization, expression selection, treatment-effect computation or model.
Published source h5ad files are immutable; new native-count h5ads are derivatives.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import anndata as ad
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args(); base = args.root; p = base / "gse325993_public_exp2"; out = p / "prepared"; out.mkdir(exist_ok=True)
    meta = pd.read_csv(base / "gse325993_public_exp1/audit/public_cell_guide_and_line_metadata.tsv.gz", sep="\t")
    meta = meta[meta.experiment == "exp2"]
    assert meta.author_demux_matched.all() and meta.passes_simple_dominant_rule.all()
    manifest = json.loads((p / "download_manifest_verified.json").read_text())
    source = {r["file"]: r for r in manifest}; audit = {}; file_qc = []; axis_sets = {}
    for species, token in [("human", "HSapiens"), ("cynomolgus", "MFascicularis")]:
        parts = []; genes = None; row_hashes = defaultdict(list); barcode_hashes = defaultdict(list)
        for f in sorted(p.glob(f"*_{token}_exp2_*.h5ad")):
            assert sha(f) == source[f.name]["sha256"]
            a = ad.read_h5ad(f); a.X = a.X.tocsr(); a.X.sort_indices()
            assert np.isfinite(a.X.data).all() and (a.X.data >= 0).all() and (a.X.data == np.floor(a.X.data)).all()
            assert a.obs_names.is_unique and a.var_names.is_unique and not a.layers and a.raw is None
            if genes is None: genes = a.var_names.tolist()
            assert genes == a.var_names.tolist()
            capture = re.search(r"_(exp2_chip\d+_lane\d+)_", f.name)[1]
            m = meta[(meta.species == species) & (meta.capture == capture)].set_index("barcode")
            assert set(m.index) == set(a.obs_names)
            obs = m.loc[a.obs_names].copy(); obs["original_barcode"] = a.obs_names
            obs["source_matrix_file"] = f.name; obs["donor_id"] = obs.individual
            obs["cell_type"] = "induced pluripotent stem cell"; obs["time_days"] = 5
            obs["study_id"] = "GSE325993"; obs.index = [species + "|" + capture + "|" + c for c in a.obs_names]
            a.X = a.X.astype(np.int32); a.obs = obs
            a.obs["native_total_umi"] = np.asarray(a.X.sum(axis=1)).ravel()
            a.obs["native_detected_genes"] = np.diff(a.X.indptr)
            for i, cell in enumerate(a.obs_names):
                lo, hi = a.X.indptr[i:i + 2]
                h = hashlib.sha256(a.X.indices[lo:hi].tobytes() + a.X.data[lo:hi].tobytes()).hexdigest()
                row_hashes[h].append(cell); barcode_hashes[a.obs.original_barcode.iloc[i]].append((capture, h, a.obs.dominant_guide.iloc[i]))
            file_qc.append({"file": f.name, "species": species, "capture": capture, "n_cells": a.n_obs, "n_genes": a.n_vars,
                            "raw_counts_integer_nonnegative": True, "all_cells_exact_author_demux_and_guide_match": True,
                            "sha256": source[f.name]["sha256"]})
            parts.append(a)
        assert len(parts) == 8
        merged = ad.concat(parts, join="inner", merge="same", index_unique=None)
        assert merged.obs_names.is_unique and merged.var_names.tolist() == genes
        assert set(merged.obs.condition) == {"NTC", "POU5F1"}
        expected = meta[meta.species == species]
        assert merged.n_obs == len(expected)
        duplicates = [ids for ids in row_hashes.values() if len(ids) > 1]
        same_barcode = {k: v for k, v in barcode_hashes.items() if len(v) > 1}
        identical_repeated_barcodes = {k: v for k, v in same_barcode.items() if len({x[1] for x in v}) < len(v)}
        assert not duplicates and not identical_repeated_barcodes
        merged.uns["provenance"] = {"source": "GSE325993 exp2 public CellRanger count h5ads", "counts_location": "X",
                                     "counts_units": "raw integer UMI", "normalization": "none", "new_expression_filter": "none",
                                     "author_demux_source_sha256": "a4a3c7792d0a2e851403ef0ec1b3c02d6392c7d63834e7782aac52fb66831f4b",
                                     "author_cell_join": "exact species plus experiment_chip_lane_barcode",
                                     "gene_coordinate_semantics": "human GENCODE32 IDs; macaque uses author human-ID Liftoff annotation on macFas6",
                                     "capture_unit": "10x chip lane/GEM channel; flow-cell sequencing lanes combined within CellRanger capture",
                                     "scope": "metadata/count QC; no effect estimates or predictions"}
        file = out / f"gse325993_exp2_{species}.native_counts.h5ad"; merged.write_h5ad(file, compression="gzip")
        merged.obs.to_csv(out / f"{species}.cell_metadata.tsv.gz", sep="\t")
        merged.var.to_csv(out / f"{species}.native_gene_metadata.tsv", sep="\t")
        perline = merged.obs.groupby(["species", "individual", "cell_line", "experiment", "condition"], observed=True).size().rename("n_cells").reset_index()
        perline.to_csv(out / f"{species}.line_condition_counts.tsv", sep="\t", index=False)
        reloaded = ad.read_h5ad(file)
        assert np.array_equal(reloaded.X.data, merged.X.data) and np.array_equal(reloaded.X.indices, merged.X.indices) and np.array_equal(reloaded.X.indptr, merged.X.indptr)
        audit[species] = {"cells": merged.n_obs, "genes": merged.n_vars, "nnz": merged.X.nnz, "n_lines": merged.obs.individual.nunique(),
                          "h5ad": str(file), "sha256": sha(file), "count_values_exact_after_serialization": True,
                          "allcells_gene_and_metadata_qc_passed": True, "entire_count_vector_duplicate_cells": len(duplicates),
                          "repeated_bare_barcodes_across_captures": len(same_barcode),
                          "repeated_bare_barcodes_with_identical_count_vectors": len(identical_repeated_barcodes),
                          "line_condition_counts": perline.to_dict("records")}
        axis_sets[species] = set(genes)
        print(json.dumps(audit[species]), flush=True)
    bridge = pd.read_csv(base / "animal_lps/native_gene_id_bridge.tsv", sep="\t").fillna("")
    common = bridge[(bridge.human_ensembl_gene_id != "") & bridge.human_ensembl_gene_id.isin(set.intersection(*axis_sets.values()))]
    assert common.human_ensembl_gene_id.is_unique and common.native_index.is_unique
    common.to_csv(out / "common_state_gene_id_axis.tsv", sep="\t", index=False)
    for f, r in source.items(): assert sha(p / f) == r["sha256"]
    report = {"status": "passed", "scope": "native counts and metadata only", "source_files": file_qc,
              "species": audit, "state_common_exact_id_genes": len(common), "source_h5ads_unchanged": True,
              "line_unit_caveat": "Two human genotype lines. Incremental human controls may use disjoint cells within the training line; they are not independent donor additions.",
              "capture_caveat": "Eight GEM channels belong to the same exp2 chip/screen, so they are technical capture units and do not create eight biological replicates."}
    (out / "exp2_preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
