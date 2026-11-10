#!/usr/bin/env python3
"""Freeze all public exp2 cells on the named State axis using coding CP10k.

No split construction, expression-based gene/cell/guide selection, model fitting,
prediction or outcome measurement is performed. Native h5ads stay immutable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def record(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args(); base = args.root; study = base / "gse325993_public_exp2"; out = study / "normalized"; out.mkdir(exist_ok=True)
    annotation = study / "annotation/gencode32_gene_annotation.tsv"
    ann_manifest = study / "annotation/gencode32_source_manifest.json"
    ann_source = json.loads(ann_manifest.read_text()); assert sha(annotation) == ann_source["table_sha256"]
    ann = pd.read_csv(annotation, sep="\t", dtype=str, keep_default_na=False)
    assert ann.gene_id_stable.is_unique
    type_by_id = dict(zip(ann.gene_id_stable, ann.gene_type))
    protein_coding = set(ann.loc[ann.gene_type.eq("protein_coding"), "gene_id_stable"])
    axis_path = study / "prepared/common_state_gene_id_axis.tsv"
    axis = pd.read_csv(axis_path, sep="\t").sort_values("native_index", kind="stable")
    assert len(axis) == 6357 and axis.native_index.is_unique and axis.human_ensembl_gene_id.is_unique
    axis = axis[["native_index", "gene_symbol", "human_ensembl_gene_id"]].copy()
    axis["gencode32_gene_type"] = axis.human_ensembl_gene_id.map(type_by_id).fillna("not_in_gencode32")
    axis.to_csv(out / "state_observed_gene_axis.tsv", sep="\t", index=False)
    arrays = {"native_gene_indices": axis.native_index.to_numpy(np.int64),
              "human_gene_ids": axis.human_ensembl_gene_id.to_numpy(dtype=str),
              "cynomolgus_gene_ids": axis.human_ensembl_gene_id.to_numpy(dtype=str)}
    sources = {"gencode32_annotation": record(annotation), "gencode32_source_manifest": record(ann_manifest), "common_state_axis": record(axis_path),
               "script": record(Path(__file__)), "native_count_preparation_audit": record(study / "prepared/exp2_preparation_audit.json"),
               "author_demux_and_guide_metadata": record(base / "gse325993_public_exp1/audit/public_cell_guide_and_line_metadata.tsv.gz"),
               "upstream_NTC_filter_boundary": record(base / "gse325993_public_exp1/audit/control_guide_upstream_filter_boundary.json")}
    reports = {}
    for species in ["human", "cynomolgus"]:
        source = study / f"prepared/gse325993_exp2_{species}.native_counts.h5ad"
        sources[species + "_counts"] = record(source)
        a = ad.read_h5ad(source)
        assert a.obs_names.is_unique and a.var_names.is_unique and (a.X.data >= 0).all() and np.equal(a.X.data, np.floor(a.X.data)).all()
        mask = np.asarray([g in protein_coding for g in a.var_names], dtype=bool)
        inserts = [g for g in a.var_names if not g.startswith("ENSG")]
        assert len(inserts) == 5 and not any(g in protein_coding for g in inserts)
        mask_frame = pd.DataFrame({"native_index": np.arange(a.n_vars), "gene_id": a.var_names,
                                   "gencode32_gene_type": [type_by_id.get(g, "not_in_gencode32") for g in a.var_names],
                                   "in_coding_denominator": mask})
        mask_path = out / f"{species}.gencode32_coding_mask.tsv"; mask_frame.to_csv(mask_path, sep="\t", index=False)
        denominators = np.asarray(a.X[:, mask].sum(axis=1)).ravel().astype(np.float64)
        assert np.isfinite(denominators).all() and (denominators > 0).all()
        columns = a.var_names.get_indexer(axis.human_ensembl_gene_id)
        assert (columns >= 0).all()
        values = a.X[:, columns].toarray().astype(np.float64)
        values *= 10000.0
        values /= denominators[:, None]
        np.log1p(values, out=values)
        arrays[species + "_expression"] = values.astype(np.float32)
        arrays[species + "_cell_ids"] = a.obs_names.to_numpy(dtype=str)
        for destination, column in [("donors", "individual"), ("cell_lines", "cell_line"), ("condition", "condition"),
                                    ("guide", "dominant_guide"), ("capture", "capture"), ("experiment", "experiment")]:
            arrays[species + "_" + destination] = a.obs[column].astype(str).to_numpy(dtype=str)
        arrays[species + "_coding_library_size"] = denominators
        pd.DataFrame({"cell_id": a.obs_names, "donor_id": a.obs.individual.to_numpy(), "cell_line": a.obs.cell_line.to_numpy(),
                      "condition": a.obs.condition.to_numpy(), "guide": a.obs.dominant_guide.to_numpy(), "capture": a.obs.capture.to_numpy(),
                      "coding_library_size": denominators}).to_csv(out / f"{species}.normalization_metadata.tsv.gz", sep="\t", index=False)
        reports[species] = {"cells": a.n_obs, "native_genes": a.n_vars, "observed_state_genes": len(axis),
                            "coding_denominator_genes": int(mask.sum()), "reference_coding_ids_absent_native": len(protein_coding - set(a.var_names)),
                            "native_gene_ids_absent_gencode32": int(mask_frame.gencode32_gene_type.eq("not_in_gencode32").sum()),
                            "excluded_insert_features": inserts,
                            "state_output_genes_outside_coding_denominator": int((~axis.human_ensembl_gene_id.isin(protein_coding)).sum()),
                            "denominator_min_median_max": [float(denominators.min()), float(np.median(denominators)), float(denominators.max())],
                            "cell_filter": "none; all author-public exact-demux cells retained", "mask": record(mask_path)}
        assert sha(source) == sources[species + "_counts"]["sha256"]
    assert arrays["human_expression"].shape == (6169, 6357) and arrays["cynomolgus_expression"].shape == (3563, 6357)
    assert all(v.dtype != object for v in arrays.values())
    target = out / "gse325993_exp2_arrays.npz"; np.savez_compressed(target, **arrays)
    with np.load(target, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
    report = {"schema_version": "gse325993_exp2_normalized_v1", "status": "all_count_normalization_and_roundtrip_checks_passed",
              "scope": "All9732 publicexp2cells; no train/cal/test split, model, outcomes or new expression/guide filtering",
              "source_files": sources,
              "arrays": {**record(target), "fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}},
              "normalization": {"formula": "float32(log1p(10000*native_UMI/full_native_observed_GENCODE32_protein_coding_UMI_sum))",
                                "definition": "GENCODE32 gene_type==protein_coding, exact deposited gene IDs; macaque uses author's human-ID Liftoff GTF",
                                "zero_denominator_policy": "abort, never drop cells", "exclude_five_insert_features": True,
                                "all6357_output_genes_retained": True, "coding_reference": ann_source},
              "species": reports, "axis": {**record(out / "state_observed_gene_axis.tsv"), "native_model_axis": 6546, "measured_common_axis": 6357,
                                            "gene_type_counts": axis.gencode32_gene_type.value_counts().to_dict(),
                                            "outside_coding_denominator_semantics": "Output counts may include noncoding genes, expressed per coding-UMI library. Output axis is unchanged."},
              "upstream_cohort_caveat": "Public NTCguide ID sets equal author's final RNA-effect-filtered lists acrossbothlines; comparison conditional on author-curated publiccohort. No exporter trace was found, so boundary inference remains explicit.",
              "frozen_native_count_files_unchanged": True, "npz_roundtrip_exact": True}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"arrays": report["arrays"]["path"], "sha256": report["arrays"]["sha256"], "species": reports, "output_types": report["axis"]["gene_type_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
