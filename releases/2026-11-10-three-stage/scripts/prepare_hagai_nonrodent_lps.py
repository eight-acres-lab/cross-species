#!/usr/bin/env python3
"""Acquire and preserve E-MTAB-6754 pig/rabbit author UMI counts on remote disk.

No expression-based selection, normalization, donor imputation, or prediction is
performed. Pig course IDs are deliberately not asserted to be donor IDs: the
paper's Supplementary Table 2 documents two complete biological individuals,
one with two technical courses, without the course-to-individual key.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

BASE = "https://www.ebi.ac.uk/biostudies/files/E-MTAB-6754/"
PATTERN = re.compile(r"^(pig|rabbit)([123])_(unst|lps[246])_filtered_by_cell_cluster0\.txt\.gz$")
SPECIES = {"pig": "Sus scrofa", "rabbit": "Oryctolagus cuniculus"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata(sdrf, manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    available = {f["path"] for f in json.loads(manifest.read_text())["files"]}
    samples = {}
    for r in csv.DictReader(sdrf.open(), delimiter="\t"):
        sample = r["Source Name"]
        species, replicate = re.match(r"([a-z]+)(\d+)", sample).groups()
        condition = sample.split("_", 1)[1].lower()
        course = species + replicate
        filename = f"{course}_{condition}_filtered_by_cell_cluster0.txt.gz"
        row = dict(sample_id=sample, course_id=course, species=species,
                   scientific_name=r["Characteristics[organism]"],
                   author_cell_type=r["Characteristics[cell type]"],
                   tissue=r["Characteristics[organism part]"],
                   sex=r["Characteristics[sex]"], age=r["Characteristics[age]"],
                   stimulus=r["Factor Value[stimulus]"], time_h=int(r["Factor Value[time]"]),
                   ena_sample=r["Comment[ENA_SAMPLE]"], ena_experiment=r["Comment[ENA_EXPERIMENT]"],
                   ena_run=r["Comment[ENA_RUN]"],
                   deposited_technical_replicate_group=r["Comment[technical replicate group]"],
                   selected_author_matrix=filename if filename in available else "",
                   author_github_individual_label=replicate,
                   donor_id="unresolved_pig_course_to_individual" if species == "pig" else course,
                   conservative_split_group="pig_complete_courses_unresolved" if species == "pig" and replicate != "4" else course,
                   biological_unit_status=("two complete individuals across pig1-3; one duplicated technical course; exact key unpublished" if species == "pig" and replicate != "4" else
                                           "third pig individual; 0/2h only; absent from author cluster0 files" if species == "pig" else
                                           "independent individual; Supplementary Table 2"))
        if sample in samples and samples[sample] != row:
            raise ValueError("Conflicting sample metadata: " + sample)
        samples[sample] = row
    with (output / "all_species_sample_metadata.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(samples.values()))), delimiter="\t")
        w.writeheader(); w.writerows(samples.values())
    save(output / "donor_provenance.json", {
        "primary_article": "https://doi.org/10.1038/s41586-018-0657-2",
        "supplement": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0657-2/MediaObjects/41586_2018_657_MOESM1_ESM.pdf",
        "supplement_pdf_page": 13, "supplement_printed_page": 12,
        "rabbit": {"complete_courses": 3, "biological_individuals": 3},
        "pig": {"complete_courses": 3, "biological_individuals_in_complete_courses_per_supplement": 2,
                "additional_partial_individual": "pig4,0/2h; no author cluster0 matrix",
                "course_to_individual_key": "unresolved; do not infer from expression",
                "source_conflict": "author GitHub samples_matching_files.csv labels pig1-4 as individual1-4, whereas journal Supplementary Table2 documents three pigs with one technical duplicate",
                "split_rule": "keep pig1,pig2,pig3 in one conservative block"},
        "cell_annotation": "author mononuclear phagocyte; author largest cluster0 membership from processed file inclusion",
        "author_processing": "CellRanger2.0, Ensembl84, original integer UMIs after author cell QC/cluster selection",
        "interpretation": "bone-marrow phagocytes,100ng/mL,2/4/6h; distinct context from human PBMC CD14,10ng/mL,24h"
    })
    return samples


def download(manifest, directory):
    j = json.loads(manifest.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 4 * j["total_bytes"]:
        raise ValueError("Insufficient disk headroom")
    def fetch(item):
        path = directory / item["path"]
        if not path.exists() or path.stat().st_size != item["size"]:
            tmp = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(j["base_url"] + item["path"], tmp)
            if tmp.stat().st_size != item["size"]:
                raise ValueError("Download size mismatch: " + path.name)
            tmp.replace(path)
        return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        audit = list(pool.map(fetch, j["files"]))
    save(directory / "download_manifest_verified.json", audit)


def read_matrix(path):
    import numpy as np
    from scipy import sparse
    genes, indices, values, ptr = [], [], [], [0]
    with gzip.open(path, "rt") as f:
        barcodes = next(f).split()
        if len(set(barcodes)) != len(barcodes):
            raise ValueError("Duplicate within-library barcodes")
        for line in f:
            gene, counts = line.split(None, 1)
            x = np.fromstring(counts, sep=" ", dtype=np.float64)
            if x.size != len(barcodes) or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
                raise ValueError("Invalid count row: " + path.name + ":" + gene)
            if x.max(initial=0) > np.iinfo(np.int32).max:
                raise ValueError("Count overflow")
            ix = np.flatnonzero(x)
            genes.append(gene); indices.append(ix.astype(np.int32)); values.append(x[ix].astype(np.int32)); ptr.append(ptr[-1] + ix.size)
    if len(set(genes)) != len(genes):
        raise ValueError("Duplicate native gene IDs")
    matrix = sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), np.asarray(ptr)), shape=(len(genes), len(barcodes))).T.tocsr()
    return matrix, genes, barcodes


def prepare(args):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    samples = metadata(args.sdrf, args.manifest, args.output)
    per_sample, report = [], {}
    for species in SPECIES:
        matrices, observations, native_genes = [], [], None
        paths = sorted(p for p in args.input.glob("*.txt.gz") if PATTERN.match(p.name) and p.name.startswith(species))
        if len(paths) != 12:
            raise ValueError(f"Expected twelve author matrices for {species}, found {len(paths)}")
        for path in paths:
            m = PATTERN.match(path.name)
            course, condition = species + m[2], m[3]
            sample = course + "_" + (condition.upper() if condition != "unst" else condition)
            row = samples[sample]
            X, genes, barcodes = read_matrix(path)
            if native_genes is None:
                native_genes = genes
            if genes != native_genes:
                raise ValueError("Author native gene axes differ; cannot fill missing genes with zero")
            totals = np.asarray(X.sum(axis=1)).ravel()
            n_genes = np.diff(X.indptr)
            obs = pd.DataFrame({k: [v] * len(barcodes) for k, v in row.items()}, index=[sample + ":" + b for b in barcodes])
            obs["author_barcode"] = barcodes; obs["author_cluster"] = "0"
            obs["condition"] = "control" if condition == "unst" else "LPS"
            obs["dose_ng_ml"] = 0 if condition == "unst" else 100
            obs["total_umi"] = totals; obs["n_genes_by_counts"] = n_genes
            matrices.append(X); observations.append(obs)
            summary = {**row, "n_cells": X.shape[0], "n_genes": X.shape[1], "nnz": X.nnz,
                       "umi_min": int(totals.min()), "umi_median": float(np.median(totals)), "umi_max": int(totals.max()),
                       "detected_gene_min": int(n_genes.min()), "detected_gene_median": float(np.median(n_genes)),
                       "sha256": sha(path), "count_dtype": "int32", "author_counts_preserved": True}
            per_sample.append(summary); print(json.dumps({k: summary[k] for k in ["sample_id", "n_cells", "n_genes", "nnz"]}), flush=True)
        var = pd.DataFrame({"native_gene_id": native_genes, "native_annotation_release": "Ensembl84"}, index=native_genes)
        obj = ad.AnnData(X=sparse.vstack(matrices, format="csr"), obs=pd.concat(observations), var=var)
        obj.uns["counts_location"] = "X"; obj.uns["counts_units"] = "author integer UMI counts"
        obj.uns["study"] = "E-MTAB-6754"; obj.uns["processing"] = "No additional filtering or normalization; author QC and cluster0 retained"
        obj.uns["donor_provenance"] = json.loads((args.output / "donor_provenance.json").read_text())
        target = args.output / f"{species}.native_counts.h5ad"
        obj.write_h5ad(target, compression="gzip")
        var.to_csv(args.output / f"{species}.native_genes.tsv", sep="\t", index=False)
        obj.obs.to_csv(args.output / f"{species}.cell_metadata.tsv.gz", sep="\t")
        report[species] = {"n_cells": obj.n_obs, "n_genes": obj.n_vars, "nnz": obj.X.nnz, "h5ad": str(target), "sha256": sha(target), "counts_integer_nonnegative": True}
        del obj, matrices
    pd.DataFrame(per_sample).to_csv(args.output / "sample_count_qc.tsv", sep="\t", index=False)
    save(args.output / "counts_qc_report.json", report)


def mapping(args):
    """Read a pinned six-attribute BioMart source; reject all ambiguous IDs/symbols."""
    import pandas as pd
    p = args.output; p.mkdir(parents=True, exist_ok=True)
    cols = ["human_ensembl", "human_symbol", "animal_ensembl", "animal_symbol", "orthology_type", "confidence"]
    source = pd.read_csv(args.mapping, sep="\t", names=cols, dtype=str, keep_default_na=False)
    eligible = source[(source.orthology_type == "ortholog_one2one") & (source.confidence == "1") & (source.human_symbol != "")].drop_duplicates()
    for key, value in [("human_ensembl", "animal_ensembl"), ("animal_ensembl", "human_ensembl"), ("human_symbol", "human_ensembl")]:
        counts = eligible.groupby(key)[value].nunique()
        eligible = eligible[eligible[key].isin(counts[counts == 1].index)]
    eligible = eligible.drop_duplicates(["human_ensembl", "animal_ensembl"]).sort_values("human_ensembl")
    native = pd.read_csv(args.native_genes, sep="\t", dtype=str).native_gene_id
    eligible["native_measured"] = eligible.animal_ensembl.isin(native)
    axis = pd.read_csv(args.state_genes, sep="\t", dtype=str)
    if "human_ensembl_gene_id" not in axis:
        raise ValueError("State bridge must provide verified human Ensembl IDs")
    axis = axis.fillna("")
    resolved = axis[axis.human_ensembl_gene_id != ""]
    if resolved.human_ensembl_gene_id.duplicated().any():
        raise ValueError("Ambiguous human Ensembl IDs in model bridge")
    state_order = dict(zip(resolved.human_ensembl_gene_id, resolved.native_index.astype(int)))
    eligible["state_index"] = eligible.human_ensembl.map(state_order).fillna(-1).astype(int)
    observed = dict(zip(resolved.human_ensembl_gene_id, resolved.author_human_axis_observed.eq("True")))
    eligible["human_measured"] = eligible.human_ensembl.map(observed).fillna(False).astype(bool)
    eligible.to_csv(p / f"{args.species}.human_one2one.tsv", sep="\t", index=False)
    selected = eligible[eligible.native_measured & (eligible.state_index >= 0)].sort_values("state_index")
    selected.to_csv(p / f"{args.species}.state_measured.tsv", sep="\t", index=False)
    common = selected[selected.human_measured]
    common.to_csv(p / f"{args.species}.human_state_measured.tsv", sep="\t", index=False)
    save(p / f"{args.species}.coverage.json", {"species": args.species, "mapping_release": 116,
        "mapping_sha256": sha(args.mapping), "native_axis_sha256": sha(args.native_genes), "state_axis_sha256": sha(args.state_genes),
        "native_genes": len(native), "unique_high_confidence_one2one_pairs": len(eligible),
        "native_measured_one2one_pairs": int(eligible.native_measured.sum()), "state_axis_genes": len(axis), "state_measured_genes": len(selected),
        "state_human_animal_measured_genes": len(common),
        "rule": "exact native and human Ensembl ID equality; verified model symbol-to-human-Ensembl bridge; high-confidence one2one; unique human and animal IDs and unique human symbol; no expression selection; no retired-ID inference"})


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="action", required=True)
    m = sub.add_parser("metadata"); m.add_argument("--sdrf", type=Path, required=True); m.add_argument("--manifest", type=Path, required=True); m.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("download"); d.add_argument("--manifest", type=Path, required=True); d.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare"); p.add_argument("--sdrf", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("mapping"); q.add_argument("--mapping", type=Path, required=True); q.add_argument("--native-genes", type=Path, required=True); q.add_argument("--state-genes", type=Path, required=True); q.add_argument("--species", choices=list(SPECIES), required=True); q.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.action == "metadata": metadata(a.sdrf, a.manifest, a.output)
    elif a.action == "download": download(a.manifest, a.output)
    elif a.action == "prepare": prepare(a)
    else: mapping(a)


if __name__ == "__main__":
    main()
