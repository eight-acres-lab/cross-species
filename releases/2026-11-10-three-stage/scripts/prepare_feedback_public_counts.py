#!/usr/bin/env python3
"""Prepare reviewed public raw-count contrasts, separately by study and species.

All genes contribute to each sample's CPM denominator. Outputs are observed
within-source treatment contrasts; this script does not construct training roles
or cross-study/cross-species pairs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
QC_EXCLUDED = {"IL1beta_6", "IL1betaTNF_6", "TNF_6", "Unstimulated_3"}
HUMAN_130_DONORS = {"LONZA_NHLF1", "LONZA_NHLF2", "LONZA_NHLF3", "NHLF_014", "NHLF_C018"}
MOUSE_130_DONORS = {"K28", "K29", "K30", "WT32", "WT33", "WT34"}


@dataclass(frozen=True)
class CountSpec:
    study: str
    species: str
    filename: str
    sample_count: int
    header_prefix: tuple[str, ...]
    delimiter: str = "\t"
    implicit_gene_index: bool = False


SPECS = (
    CountSpec("GSE166927", "human", "GSE166927_featurecounts_counts.txt.gz", 24, ("gene_id",)),
    CountSpec("GSE246216", "human", "GSE246216_RNA-seq_counts.csv.gz", 42, ("GeneID", "GeneName"), ","),
    CountSpec("GSE130983", "human", "GSE130983_Counts_Human_Stim.txt.gz", 15, ("Geneid", "Length"), implicit_gene_index=True),
    CountSpec("GSE130983", "mouse", "GSE130983_Counts_Mouse_Fib_KO_and_Stim.txt.gz", 18, ("Length",), implicit_gene_index=True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source(manifest: dict, manifest_dir: Path, spec: CountSpec) -> dict:
    records = [r for r in manifest.get("files", [])
               if r.get("accession") == spec.study and r.get("filename") == spec.filename]
    if len(records) != 1:
        raise ValueError(f"Expected one manifest record for {spec.study}/{spec.filename}")
    record = records[0]
    if record.get("status") not in {"cached", "downloaded"}:
        raise ValueError(f"Source manifest status is not cached/downloaded: {spec.filename}")
    expected = record.get("sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Missing valid source SHA256: {spec.filename}")
    path = Path(record.get("path") or spec.study + "/" + spec.filename)
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve()
    if not path.is_relative_to(manifest_dir.resolve()) or path.name != spec.filename or path.parent.name != spec.study:
        raise ValueError(f"Source path does not match the manifest source directory: {path}")
    actual = sha256(path)
    if actual.lower() != expected.lower() or (record.get("expected_sha256") and actual != record["expected_sha256"]):
        raise ValueError(f"Source SHA256 mismatch: {spec.filename}")
    if record.get("bytes") is not None and path.stat().st_size != record["bytes"]:
        raise ValueError(f"Source byte count mismatch: {spec.filename}")
    return {"path": str(path), "sha256": actual, "manifest_status": record["status"],
            "url": record.get("url"), "bytes": path.stat().st_size}


def read_raw_counts(path: Path, spec: CountSpec):
    """Return native genes, actual sample headers and a genes-by-samples matrix."""
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=spec.delimiter)
        header = next(reader, [])
        if tuple(header[:-spec.sample_count]) != spec.header_prefix:
            raise ValueError(f"Unexpected count header annotations: {path}")
        samples = header[-spec.sample_count:]
        if len(samples) != spec.sample_count or any(not x for x in samples) or len(set(samples)) != len(samples):
            raise ValueError(f"Missing/duplicate sample headers: {path}")
        allowed_widths = {len(header), len(header) + 1} if spec.implicit_gene_index else {len(header)}
        width, genes, values, seen = None, [], [], set()
        for line_number, fields in enumerate(reader, 2):
            if not fields or len(fields) not in allowed_widths or (width is not None and len(fields) != width):
                raise ValueError(f"Inconsistent count row width: {path}:{line_number}")
            width = len(fields)
            gene = fields[0].strip()
            if not gene or gene in seen:
                raise ValueError(f"Empty/duplicate native gene ID: {path}:{line_number}")
            seen.add(gene)
            try:
                counts = np.asarray(fields[-spec.sample_count:], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"Nonnumeric counts: {path}:{line_number}") from exc
            if (not np.isfinite(counts).all() or (counts < 0).any()
                    or (counts != np.floor(counts)).any() or (counts > 2**53).any()):
                raise ValueError(f"Counts must be finite nonnegative exact integers: {path}:{line_number}")
            genes.append(gene)
            values.append(counts)
    if not genes:
        raise ValueError(f"Empty count matrix: {path}")
    return np.asarray(genes), samples, np.stack(values)


def log_cpm(counts: np.ndarray):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Expected a finite nonnegative genes-by-samples matrix")
    libraries = counts.sum(axis=0)
    if not np.isfinite(libraries).all() or (libraries <= 0).any():
        raise ValueError("Every full-matrix sample library must be positive and finite")
    return np.log2(1 + counts / libraries[None, :] * 1e6), libraries


def sample_annotations(spec: CountSpec, samples: list[str]) -> list[dict]:
    records = []
    for sample in samples:
        row = {"study": spec.study, "species": spec.species, "sample_id": sample,
               "donor": "", "donor_kind": "source_donor_label", "tissue": "",
               "disease_context": "", "cell_line": "", "genotype": "not_reported",
               "replicate_label": "", "excluded_reason": ""}
        if spec.study == "GSE166927":
            match = re.fullmatch(r"(IL1betaTNF|IL1beta|TNF|Unstimulated)_([1-6])", sample)
            if match is None:
                raise ValueError(f"Unknown GSE166927 sample: {sample}")
            perturbation, replicate = match.groups()
            row.update(perturbation=perturbation, replicate_label=replicate,
                       donor_kind="single_cell_line_assay_replicate_not_independent_donor",
                       tissue="colon", cell_line="CCD18-Co", control_key="CCD18-Co",
                       is_control=perturbation == "Unstimulated", task_context="CCD18-Co")
            if sample in QC_EXCLUDED:
                row["excluded_reason"] = "GSM metadata explicitly qc:EXCLUDE"
        elif spec.study == "GSE246216":
            match = re.fullmatch(r"(CRC-S[456]|NSCLC-S[123])-(CXCL12|IL1a|IL6|LIF|SPP1|TNFa|WT)", sample)
            if match is None:
                raise ValueError(f"Unknown GSE246216 sample: {sample}")
            donor, perturbation = match.groups()
            disease = donor.split("-")[0]
            tissue = "colon" if disease == "CRC" else "lung"
            row.update(donor=donor, perturbation=perturbation, tissue=tissue, disease_context=disease,
                       control_key=donor, is_control=perturbation == "WT", task_context=tissue)
        elif spec.study == "GSE130983":
            match = re.fullmatch(r"(.+)[-_](BSL|IL11|TGFB)_S(\d+)", sample)
            if match is None:
                raise ValueError(f"Unknown GSE130983 sample: {sample}")
            donor, perturbation, replicate = match.groups()
            permitted = HUMAN_130_DONORS if spec.species == "human" else MOUSE_130_DONORS
            if donor not in permitted:
                raise ValueError(f"Unknown GSE130983 donor label: {donor}")
            genotype = "not_reported" if spec.species == "human" else "Il11ra1_KO" if donor.startswith("K") else "WT"
            row.update(donor=donor, perturbation=perturbation, tissue="lung", genotype=genotype,
                       replicate_label=replicate, control_key=donor, is_control=perturbation == "BSL",
                       task_context="lung" if spec.species == "human" else "lung_" + genotype)
            if spec.species == "mouse" and perturbation == "IL11":
                row["excluded_reason"] = "Processed count column present; corresponding GSM provenance unresolved"
        else:
            raise ValueError(f"Study outside reviewed count scope: {spec.study}")
        records.append(row)
    if spec.study == "GSE166927":
        expected = {f"{condition}_{i}" for condition in ["IL1betaTNF", "IL1beta", "TNF", "Unstimulated"] for i in range(1, 7)}
        if set(samples) != expected:
            raise ValueError("GSE166927 requires the reviewed set of 24 samples")
    else:
        groups = collections.defaultdict(list)
        for row in records:
            groups[row["donor"]].append(row["perturbation"])
        expected_donors = ({"CRC-S4", "CRC-S5", "CRC-S6", "NSCLC-S1", "NSCLC-S2", "NSCLC-S3"}
                           if spec.study == "GSE246216" else HUMAN_130_DONORS if spec.species == "human" else MOUSE_130_DONORS)
        conditions = ({"CXCL12", "IL1a", "IL6", "LIF", "SPP1", "TNFa", "WT"}
                      if spec.study == "GSE246216" else {"BSL", "IL11", "TGFB"})
        if set(groups) != expected_donors or any(set(v) != conditions or len(v) != len(conditions) for v in groups.values()):
            raise ValueError("Expected one reviewed count sample per donor/condition")
    return records


def prepare_counts(spec: CountSpec, genes: np.ndarray, samples: list[str], counts: np.ndarray, source: dict):
    annotations = sample_annotations(spec, samples)
    transformed, libraries = log_cpm(counts)
    if counts.shape != (len(genes), len(samples)):
        raise ValueError("Count axes do not match gene/sample identifiers")
    controls = collections.defaultdict(list)
    for i, row in enumerate(annotations):
        if row["is_control"] and not row["excluded_reason"]:
            controls[row["control_key"]].append(i)
    responses, metadata = [], []
    for i, row in enumerate(annotations):
        if row["is_control"] or row["excluded_reason"]:
            continue
        reference = controls[row["control_key"]]
        required_controls = 5 if spec.study == "GSE166927" else 1
        if len(reference) != required_controls:
            raise ValueError(f"Missing/nonunique reviewed control reference: {row['sample_id']}")
        responses.append(transformed[:, i] - transformed[:, reference].mean(axis=1))
        metadata.append({key: value for key, value in row.items() if key not in {"is_control", "excluded_reason", "control_key", "task_context"}})
        metadata[-1].update(
            task_id=f"{spec.study}:{spec.species}:{row['task_context']}:{row['perturbation']}",
            control_refs=json.dumps([samples[j] for j in reference], separators=(",", ":")),
            control_reference_type="mean_of_five_qc_retained_controls" if len(reference) > 1 else "same_donor_control",
            control_dependency_group=f"{spec.study}:{spec.species}:{row['control_key']}",
            dependency_note=("All treated rows share the same five-control mean; replicate numbers are not donor matches."
                             if spec.study == "GSE166927" else "Responses for this donor share its control; retain donor grouping in uncertainty estimates."),
            source_file=Path(source["path"]).name, source_sha256=source["sha256"],
        )
    responses = np.asarray(responses, dtype=np.float32)
    if not len(metadata) or not np.isfinite(responses).all():
        raise ValueError("No finite treatment responses generated")
    provenance = {
        "study": spec.study, "species": spec.species, "source": source,
        "transform": "Per-sample CPM using every deposited gene row, then log2(CPM + 1).",
        "response": "Treated log2CPM minus the mean of its explicitly matched control log2CPM profiles.",
        "native_gene_ids_preserved": True, "feature_selection": "none",
        "cross_study_or_species_pairing": "not_performed", "training_split_assignment": "not_performed",
        "raw_count_samples": len(samples), "retained_samples": sum(not r["excluded_reason"] for r in annotations),
        "gene_rows": len(genes), "response_rows": len(metadata),
        "unique_perturbations": sorted({r["perturbation"] for r in metadata}),
        "context_specific_task_counts": dict(collections.Counter(r["task_id"] for r in metadata)),
        "donor_labels": sorted({r["donor"] for r in metadata if r["donor"]}),
        "genotype_response_counts": dict(collections.Counter(r["genotype"] for r in metadata)),
        "control_samples_used": sorted({samples[j] for indices in controls.values() for j in indices}),
        "excluded_samples": [{"sample_id": r["sample_id"], "reason": r["excluded_reason"]} for r in annotations if r["excluded_reason"]],
        "all_sample_library_counts_full_matrix": {sample: int(size) for sample, size in zip(samples, libraries)},
        "dependency_note": metadata[0]["dependency_note"],
    }
    return responses, metadata, provenance


def prepare_all(manifest_path: Path, out_dir: Path, studies=None, overwrite=False):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    chosen = [spec for spec in SPECS if studies is None or spec.study in studies]
    if not chosen:
        raise ValueError("No reviewed raw-count studies selected")
    for spec in chosen:
        destination = out_dir / spec.study / spec.species
        if not overwrite and any((destination / filename).exists() for filename in ["responses.npz", "metadata.tsv", "provenance.json"]):
            raise FileExistsError(f"Processed output already exists: {destination}; use --overwrite explicitly")
    sources = [verified_source(manifest, manifest_path.parent, spec) for spec in chosen]
    reports = []
    for spec, source in zip(chosen, sources):
        genes, samples, counts = read_raw_counts(Path(source["path"]), spec)
        responses, metadata, provenance = prepare_counts(spec, genes, samples, counts, source)
        destination = out_dir / spec.study / spec.species
        destination.mkdir(parents=True, exist_ok=True)
        provenance.update(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                          preparation_script_sha256=sha256(Path(__file__)),
                          prepared_utc=datetime.now(timezone.utc).isoformat())
        np.savez_compressed(destination / "responses.npz", responses=responses, genes=genes,
                            sample_ids=np.asarray([r["sample_id"] for r in metadata]),
                            task_ids=np.asarray([r["task_id"] for r in metadata]))
        with (destination / "metadata.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(metadata)
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        reports.append({"study": spec.study, "species": spec.species, "output": str(destination),
                        **{key: provenance[key] for key in ["raw_count_samples", "retained_samples", "gene_rows", "response_rows", "unique_perturbations", "context_specific_task_counts", "genotype_response_counts", "excluded_samples"]}})
    summary = {"sources": reports, "total_response_rows": sum(r["response_rows"] for r in reports),
               "E-MTAB-5919": "Left unchanged; processed-expression measurement units require further verification."}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts/feedback_public_sources_2026-11-07/download_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/feedback_processed_sources_2026-11-07")
    parser.add_argument("--study", action="append", choices=sorted({spec.study for spec in SPECS}))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare_all(args.manifest, args.out_dir, args.study, args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
