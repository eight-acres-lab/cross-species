#!/usr/bin/env python3
"""Build a rat->human one-to-one ortholog map for the Open TG-GATEs gene columns.

The deposited rat matrix is indexed by RAT Entrez gene IDs (14,132) and the human
matrix by HUMAN Entrez gene IDs (20,414); the two ID spaces are disjoint (only 45
numerically identical strings, which are coincidences, not orthologs). A
target-free, rat-side-defined responsive-gene panel therefore needs an explicit
ortholog map, which no file in this repository provided.

This script downloads three Ensembl BioMart tables and writes the resulting
high-confidence one-to-one map, with the exact queries and file digests recorded
next to it.

CPU only, numpy + stdlib only. Network: www.ensembl.org BioMart (public).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

MARTSERVICE = "http://www.ensembl.org/biomart/martservice"

QUERIES = {
    "rat_ensembl_to_entrez": (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'count="" datasetConfigVersion="0.6">'
        '<Dataset name="rnorvegicus_gene_ensembl" interface="default">'
        '<Attribute name="ensembl_gene_id"/><Attribute name="entrezgene_id"/>'
        "</Dataset></Query>"
    ),
    "rat_to_human_homology": (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'count="" datasetConfigVersion="0.6">'
        '<Dataset name="rnorvegicus_gene_ensembl" interface="default">'
        '<Attribute name="ensembl_gene_id"/>'
        '<Attribute name="hsapiens_homolog_ensembl_gene"/>'
        '<Attribute name="hsapiens_homolog_orthology_type"/>'
        '<Attribute name="hsapiens_homolog_orthology_confidence"/>'
        "</Dataset></Query>"
    ),
    "human_ensembl_to_entrez": (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'count="" datasetConfigVersion="0.6">'
        '<Dataset name="hsapiens_gene_ensembl" interface="default">'
        '<Attribute name="ensembl_gene_id"/><Attribute name="entrezgene_id"/>'
        "</Dataset></Query>"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(query: str, out_path: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        result = subprocess.run(
            ["curl", "-sL", "-m", "900", "-G", "--data-urlencode", f"query={query}",
             "-o", str(out_path), "-w", "%{http_code}", MARTSERVICE],
            capture_output=True, text=True, check=False,
        )
        code = result.stdout.strip()
        if code == "200" and out_path.exists() and out_path.stat().st_size > 1000:
            head = out_path.read_text(errors="replace")[:200]
            if "Query ERROR" not in head:
                return
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"BioMart query failed for {out_path.name}")


def header_columns(path: Path) -> list[str]:
    with path.open() as handle:
        return handle.readline().rstrip("\n").split("\t")[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/tg_gates_rat_human_v1/data")
    parser.add_argument("--output-dir", default="artifacts/tg_gates_rat_human_v1/ortholog_map")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "biomart_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: raw_dir / f"{name}.tsv" for name in QUERIES}
    if not args.skip_download:
        for name, query in QUERIES.items():
            fetch(query, paths[name])

    rat_entrez_of_ensembl: dict[str, set[str]] = {}
    with paths["rat_ensembl_to_entrez"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2 or not fields[1]:
                continue
            rat_entrez_of_ensembl.setdefault(fields[0], set()).add(fields[1].split(".")[0])

    human_entrez_of_ensembl: dict[str, set[str]] = {}
    with paths["human_ensembl_to_entrez"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2 or not fields[1]:
                continue
            human_entrez_of_ensembl.setdefault(fields[0], set()).add(fields[1].split(".")[0])

    rat_genes = set(header_columns(data_dir / "pairs_rat_X.tsv"))
    human_genes = set(header_columns(data_dir / "pairs_human_Y.tsv"))

    # Restrict to high-confidence one-to-one orthology, then require that the
    # resulting Entrez pair is unique in BOTH directions among matrix columns.
    rat_to_human: dict[str, set[str]] = {}
    n_homology_rows = 0
    n_one2one = 0
    with paths["rat_to_human_homology"].open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or not fields[1]:
                continue
            n_homology_rows += 1
            if fields[2] != "ortholog_one2one" or fields[3] != "1":
                continue
            n_one2one += 1
            for rat_entrez in rat_entrez_of_ensembl.get(fields[0], ()):
                if rat_entrez not in rat_genes:
                    continue
                for human_entrez in human_entrez_of_ensembl.get(fields[1], ()):
                    if human_entrez not in human_genes:
                        continue
                    rat_to_human.setdefault(rat_entrez, set()).add(human_entrez)

    human_to_rat: dict[str, set[str]] = {}
    for rat_entrez, targets in rat_to_human.items():
        for human_entrez in targets:
            human_to_rat.setdefault(human_entrez, set()).add(rat_entrez)

    pairs = sorted(
        (rat_entrez, next(iter(targets)))
        for rat_entrez, targets in rat_to_human.items()
        if len(targets) == 1 and len(human_to_rat[next(iter(targets))]) == 1
    )

    map_path = out_dir / "rat_to_human_one2one.tsv"
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rat_entrez", "human_entrez"])
        writer.writerows(pairs)

    payload = {
        "purpose": (
            "Rat->human one-to-one ortholog map over the Open TG-GATEs matrix columns, needed for "
            "a target-free (rat-side) responsive-gene panel. No such map existed in the repository."
        ),
        "source": "Ensembl BioMart, " + MARTSERVICE,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries": QUERIES,
        "raw_files": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "filters": (
            "hsapiens_homolog_orthology_type == ortholog_one2one AND "
            "hsapiens_homolog_orthology_confidence == 1; then the Entrez-level pair must be unique "
            "in both directions among the matrix columns"
        ),
        "counts": {
            "rat_matrix_genes": len(rat_genes),
            "human_matrix_genes": len(human_genes),
            "biomart_homology_rows": n_homology_rows,
            "biomart_one2one_high_confidence_rows": n_one2one,
            "mapped_pairs": len(pairs),
        },
        "output": str(map_path),
        "output_sha256": sha256(map_path),
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
