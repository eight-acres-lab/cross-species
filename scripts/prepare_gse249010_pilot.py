#!/usr/bin/env python3
"""Prepare paired response vectors from the GSE249010 processed TPM matrix."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_soft(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    accession = ""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                accession = line.split(" = ", 1)[1]
                current = {"geo_accession": accession}
                records[accession] = current
            elif current is not None and line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                if ":" in value:
                    key, val = value.split(":", 1)
                    current[key.strip()] = val.strip()
            elif current is not None and line.startswith("!Sample_organism_ch1 = "):
                current["organism"] = line.split(" = ", 1)[1]
    return records


def parse_matrix(
    path: Path,
    drop_genes: frozenset[str] = frozenset(),
    renormalize_tpm: bool = False,
    pseudocount: float = 1.0,
) -> tuple[list[str], list[str], np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        sample_ids = header[1:]
        genes: list[str] = []
        rows: list[np.ndarray] = []
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            genes.append(fields[0])
            rows.append(np.asarray([0.0 if value in {"NA", "", "null"} else float(value) for value in fields[1:]], dtype=np.float32))
    tpm = np.asarray(rows, dtype=np.float32)
    if drop_genes:
        keep = [i for i, gene in enumerate(genes) if gene not in drop_genes]
        if len(keep) == len(genes):
            raise ValueError("none of the requested genes are on the axis")
        genes = [genes[i] for i in keep]
        tpm = tpm[keep]
    if renormalize_tpm:
        # Dropping genes breaks the sum-to-1e6 property that makes TPM comparable
        # between samples; restore it over the genes that remain.
        totals = tpm.sum(axis=0, keepdims=True)
        if not np.all(totals > 0):
            raise ValueError("a sample has no positive TPM after dropping genes")
        tpm = tpm * (1e6 / totals)
    # The matrix is gene x sample; downstream code uses sample x gene.
    # The pseudocount sets where log2 stops behaving like a logarithm: genes whose
    # TPM is small relative to it have their differences compressed toward zero.
    # Varying it is a direct test of whether that compression drives a result.
    return sample_ids, genes, np.log2(np.maximum(tpm, 0.0) + pseudocount).T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--soft", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--drop-genes",
        default="",
        help="comma-separated gene IDs to remove before any transform; used by the "
             "pre-registered globin sensitivity arm and empty for the primary arm",
    )
    parser.add_argument(
        "--renormalize-tpm",
        action="store_true",
        help="after dropping genes, rescale each sample to sum to 1e6 so TPM stays comparable",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="the constant added before log2; 1.0 is the published arm. Varying it tests whether "
             "low-expression compression drives a result",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = parse_soft(Path(args.soft))
    drop = frozenset(g.strip() for g in args.drop_genes.split(",") if g.strip())
    if args.renormalize_tpm and not drop:
        raise SystemExit("--renormalize-tpm without --drop-genes would change the primary arm")
    sample_ids, genes, expression = parse_matrix(
        Path(args.matrix), drop, args.renormalize_tpm, args.pseudocount
    )
    if expression.shape[0] != len(sample_ids):
        raise ValueError("matrix header and row widths disagree")
    by_barcode = {}
    for rec in records.values():
        match = re.search(r"\[([^\]]+)\]", rec.get("title", ""))
        if match:
            rec["barcode"] = match.group(1)
            by_barcode[rec["barcode"]] = rec
    missing = [sid for sid in sample_ids if sid not in by_barcode]
    if missing:
        raise ValueError(f"missing SOFT metadata for {len(missing)} matrix samples")
    metadata = []
    response_rows: list[np.ndarray] = []
    response_meta: list[dict[str, str]] = []
    index = {sid: i for i, sid in enumerate(sample_ids)}
    # For every individual and time point, subtract the matched no-LPS control.
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for sid in sample_ids:
        rec = dict(by_barcode[sid])
        rec["sample_id"] = sid
        metadata.append(rec)
        groups[(rec.get("subject", ""), rec.get("time", ""), rec.get("organism", ""))].append(rec)
    for (subject, time, organism), group in sorted(groups.items()):
        controls = [rec for rec in group if rec.get("dose", "") == "0"]
        if len(controls) != 1:
            continue
        control = expression[index[controls[0]["sample_id"]]]
        for rec in group:
            dose = rec.get("dose", "")
            if dose == "0":
                continue
            response_rows.append(expression[index[rec["sample_id"]]] - control)
            response_meta.append({
                "sample_id": rec["sample_id"],
                "species": rec.get("organism", ""),
                "subject": subject,
                "time_h": time,
                "dose_ng_ml": dose,
                "task_id": f"LPS|{dose}ngml|{time}h",
            })
    response_matrix = np.asarray(response_rows, dtype=np.float32)
    with (output / "sample_metadata.tsv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for rec in metadata for key in rec})
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(metadata)
    with (output / "response_metadata.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_meta[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(response_meta)
    np.save(output / "responses.npy", response_matrix)
    np.save(output / "genes.npy", np.asarray(genes))
    print({"samples": len(sample_ids), "genes": len(genes), "response_rows": len(response_meta), "species": sorted({r['species'] for r in response_meta})})


if __name__ == "__main__":
    main()
