#!/usr/bin/env python3
"""Curate GSE300171 without changing the author's DESeq2 count scale.

The compatibility array ``arm_logcpm`` stores log2(author-normalized count + 1)
for this panel. Its quantity is declared in every metadata record and manifest.
No CPM re-normalization, expression-based gene filter, or donor pairing is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.multisource_data import (
    CuratedPanel, file_hash, save_curated_panel, unique_ortholog_pairs, write_json,
)

SPECIES = ("human", "rat", "cyno", "dog")
FILE_TAG = {"human": "human", "rat": "rat", "cyno": "monkey", "dog": "dog"}
PREFIX = {"human": "HUM", "rat": "RAT", "cyno": "MON", "dog": "DOG"}
ORGANISM = {"human": "Homo sapiens", "rat": "Rattus norvegicus", "cyno": "Macaca fascicularis", "dog": "Canis lupus familiaris"}
LOTS = {"human": "HU8373", "rat": "FIM", "cyno": "JCW", "dog": "BTQ"}
SEX = {"human": "female", "rat": "male", "cyno": "male", "dog": "male"}
DRUGS = {"BOS": "bosentan", "CPZ": "chlorpromazine", "FIA": "fialuridine"}
UNIT = "log2(author DESeq2 normalized count + 1)"


def read_soft(path):
    records, current = [], None
    with Path(path).open() as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                current = {"gsm": line.split(" = ", 1)[1]}
                records.append(current)
            elif current is not None and line.startswith("!Sample_") and " = " in line:
                key, value = line[len("!Sample_"):].split(" = ", 1)
                current.setdefault(key, []).append(value)
    if not records or len({r["gsm"] for r in records}) != len(records):
        raise ValueError("Missing or duplicate GEO sample records")
    return records


def read_normalized_counts(path):
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        samples = header[1:]
        if not samples or any(not x for x in samples) or len(samples) != len(set(samples)):
            raise ValueError("Invalid count sample axis")
        genes, values = [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Count row width mismatch")
            genes.append(row[0].split(".")[0])
            values.append([np.nan if x in {"", "NA", "NaN"} else float(x) for x in row[1:]])
    counts = np.asarray(values, dtype=float)
    if not genes or len(genes) != len(set(genes)) or any(not x for x in genes):
        raise ValueError("Missing or duplicate normalized gene IDs")
    if np.isinf(counts).any() or (counts[np.isfinite(counts)] < 0).any():
        raise ValueError("Normalized counts must be nonnegative or explicit NA")
    return np.asarray(genes), samples, counts


def read_qc(path):
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_ID", "species", "treatment", "day", "replicate", "mapped_counts", "uniquely_mapped_reads"}
        if required - set(reader.fieldnames or []):
            raise ValueError("QC table lacks required columns")
        rows = list(reader)
    if len(rows) != len({r["sample_ID"] for r in rows}):
        raise ValueError("Duplicate QC sample IDs")
    return rows


def fourway_axis(mapping_dir):
    mapping_dir = Path(mapping_dir)
    frozen = json.loads((mapping_dir / "manifest.json").read_text())
    if frozen.get("biomart_gene_release") != 116:
        raise ValueError("Expected the previously frozen Ensembl release116")
    maps, audit = {}, {}
    for species in SPECIES[1:]:
        path = mapping_dir / (species + ".tsv")
        if file_hash(path) != frozen["files"][path.name]:
            raise ValueError("Frozen mapping checksum mismatch")
        with path.open() as handle:
            maps[species], audit[species] = unique_ortholog_pairs(list(csv.reader(handle, delimiter="\t")))
    ids = sorted(set.intersection(*(set(x) for x in maps.values())))
    return [{"human": h, **{s: maps[s][h] for s in SPECIES[1:]}} for h in ids], {
        "release": 116, "source_manifest_sha256": file_hash(mapping_dir / "manifest.json"),
        "pair_files": {s: file_hash(mapping_dir / (s + ".tsv")) for s in maps},
        "filtering": audit, "fourway_genes": len(ids),
        "rule": "Human anchored; high-confidence one-to-one; bidirectionally unique; same frozen release116 pair queries.",
    }


def sample_metadata(species, samples, qc_rows, gsm_records):
    qc = {r["sample_ID"]: r for r in qc_rows}
    if set(qc) != set(samples):
        raise ValueError("QC sample set differs from normalized matrix")
    geo = {}
    for record in gsm_records:
        title = (record.get("title") or [""])[0]
        match = re.fullmatch(r"(HUM|RAT|MON|DOG)-(BOS|CPZ|FIA|CON|D0|D4|D8|D14)-(\d+)_(S\d+)", title)
        if not match:
            raise ValueError("Unrecognized original GEO title: " + title)
        key = "_".join([match[1], match[2], match[4]])
        if key in geo:
            raise ValueError("Ambiguous GEO library identity")
        geo[key] = (record, match)
    output = []
    for library in samples:
        if library not in geo:
            raise ValueError("Matrix library absent from exact GEO title bridge: " + library)
        q = qc[library]
        record, title = geo[library]
        if title[1] != PREFIX[species] or q["species"] != PREFIX[species] or record.get("organism_ch1") != [ORGANISM[species]]:
            raise ValueError("Species identity differs among matrix/QC/GEO")
        group = title[2]
        expected_treatment = group if group in {*DRUGS, "CON"} else "NS"
        expected_day = "D14" if group in {*DRUGS, "CON"} else group
        if q["treatment"] != expected_treatment or q["day"] != expected_day or q["replicate"] != title[3]:
            raise ValueError("QC experimental labels disagree with GEO title")
        characteristics = record.get("characteristics_ch1", [])
        expected_geo = DRUGS.get(group, "untreated" if group == "CON" else group)
        actual_treatment = [x.split(":", 1)[1].strip() for x in characteristics if x.startswith("treatment:")]
        # The CON specimens are explicitly labelled DMSO control in deposited GEO.
        if group == "CON":
            if actual_treatment != ["DMSO control"]:
                raise ValueError("Drug-control identity not supported by GEO characteristics")
        elif actual_treatment != [expected_geo]:
            raise ValueError("Treatment identity not supported by GEO characteristics")
        mapped, unique = float(q["mapped_counts"]), float(q["uniquely_mapped_reads"])
        if not np.isfinite([mapped, unique]).all() or mapped < 1e6 or unique < 50:
            raise ValueError("QC-passed file contains an out-of-threshold sample")
        is_drug = group in DRUGS
        is_vehicle = group in {"CON", "D0"}
        kind = "drug" if group in {*DRUGS, "CON"} else "culture_state"
        perturbation = group if is_drug else ("vehicle" if group == "CON" else "CULTURE_" + group)
        contrast = "GSE300171|" + species + "|" + perturbation
        source_id = "GSE300171|" + species + "|primary_hepatocyte_PhysioMimix_LC12"
        row = {
            "study": "GSE300171", "species": species, "source_id": source_id,
            "cell_model": "primary_hepatocyte_PhysioMimix_LC12", "tissue": "liver", "platform": "PhysioMimix LC12 MPS",
            "perturbation": perturbation, "perturbation_name": DRUGS.get(group), "contrast_kind": kind,
            "role": "vehicle" if is_vehicle else "treated", "contrast_id": contrast,
            "library": library, "sample_id": record["gsm"], "original_sample_title": title[0],
            "replicate_label": q["replicate"], "biological_id": None,
            "source_lot": LOTS[species], "source_sex": SEX[species],
            "independent_unit_status": "assay_replicate_from_one_reported_species_lot; exact_experiment_identity_unresolved",
            "donor_identity_verified": False, "source_lot_provenance": "Original publication Table1; GEO does not repeat lot IDs per specimen",
            "culture_day": int(expected_day[1:]), "time_h": 240. if kind == "drug" else int(expected_day[1:]) * 24.,
            "time_reference": "Drug/vehicle exposure interval from culture day4 to14" if kind == "drug" else "Culture duration; D0 before plating",
            "dose_value": None, "dose_unit": None, "planned_exposure_known": False,
            "dose_status": "Per-GSM concentration absent; publication reports several experiments/doses without a verified mapping to each RNA-seq library" if is_drug else "No administered drug in this arm",
            "units": UNIT, "count_units": "Author DESeq2 normalized gene counts, not raw counts",
            "transform": "log2(normalized_count+1); no additional CPM/library normalization",
            "mapped_counts": mapped, "uniquely_mapped_reads_percent": unique,
            "control_match_rule": "Drug groups use deposited CON group at D14; culture-state groups use D0 before plating. No replicate-label pairing.",
            "control_group": "CON" if kind == "drug" else "D0",
            "response_dependency_group": "GSE300171|" + species + "|" + ("drug_CON_D14" if kind == "drug" else "culture_D0"),
        }
        if is_vehicle:
            targets = list(DRUGS) if group == "CON" else ["CULTURE_D4", "CULTURE_D8", "CULTURE_D14"]
            row["control_for_contrasts"] = ["GSE300171|" + species + "|" + x for x in targets]
        output.append(row)
    return output


def assemble(axis, matrices, annotations, manifest):
    features = np.asarray([r["human"] for r in axis])
    raw_blocks, log_blocks, arms, count_audit = [], [], [], {}
    for species in SPECIES:
        genes, samples, counts = matrices[species]
        by_library = {r["library"]: r for r in annotations if r["species"] == species}
        if set(by_library) != set(samples):
            raise ValueError("Sample metadata and matrix disagree")
        index = {g: i for i, g in enumerate(genes)}
        selected = np.asarray([index.get(row[species], -1) for row in axis])
        present = selected >= 0
        values = np.full((len(samples), len(axis)), np.nan)
        values[:, present] = counts[selected[present]].T
        raw_blocks.append(values)
        log_blocks.append(np.log2(values + 1))
        arms.extend(dict(by_library[library]) for library in samples)
        count_audit[species] = {"native_genes": len(genes), "samples": len(samples), "native_missing_values": int(np.isnan(counts).sum()),
                                "mapped_axis_genes_present": int(present.sum()), "count_measurement": "DESeq2 normalized counts", "recomputed_library_size": False}
    raw, log = np.concatenate(raw_blocks), np.concatenate(log_blocks)
    treated_rows, response, baseline, observations, contrasts, means, eligibility = [], [], [], [], [], [], []
    for cid in sorted({r["contrast_id"] for r in arms if r["role"] == "treated"}):
        t = [i for i, r in enumerate(arms) if r["contrast_id"] == cid and r["role"] == "treated"]
        c = [i for i, r in enumerate(arms) if cid in r.get("control_for_contrasts", []) and r["role"] == "vehicle"]
        if not t or not c:
            raise ValueError("Missing treated or explicit control arm")
        if any(arms[i]["species"] != arms[t[0]]["species"] for i in c):
            raise ValueError("Cross-species control assignment")
        ref = log[c].mean(axis=0)
        record = dict(arms[t[0]])
        record.update(treated_arm_rows=t, control_arm_rows=c, treated_replicates=len(t), control_replicates=len(c),
                      estimand="Mean treatment log2(author normalized count+1) minus mean explicitly assigned control arm",
                      measurement_resampling_eligible=len(t) >= 2 and len(c) >= 2,
                      uncertainty_limitation="One source lot per species; replicate/litter/experiment independence and treatment-control pairing unverified")
        contrasts.append(record)
        means.append(log[t].mean(axis=0) - ref)
        for i in t:
            treated_rows.append(i); response.append(log[i] - ref); baseline.append(ref)
            item = dict(arms[i]); item.update(control_arm_rows=c, control_reference_type="unpaired_assay_control_arm_mean", donor_pairing_verified=False)
            observations.append(item)
    for record in contrasts:
        related = [r for r in contrasts if r["perturbation"] == record["perturbation"] and r["contrast_kind"] == record["contrast_kind"]]
        indices = sorted({i for r in related for i in r["treated_arm_rows"] + r["control_arm_rows"]})
        eligibility.append(np.isfinite(log[indices]).all(axis=0))
    manifest = {**manifest, "counts_audit": count_audit,
                "compatibility_array_quantity": {"arm_counts": "Author DESeq2 normalized counts", "arm_logcpm": UNIT},
                "units": UNIT, "normalization": "Use deposited per-species DESeq2 normalization; log2(+1) only. No refit or CPM renormalization.",
                "normalization_scope_warning": "Author normalization used all available within-species experimental conditions before downstream holdout; this is inherited retrospective preprocessing.",
                "axis_rule": "Four-way release116 human-anchored map; one fixed complete axis across all93 arms for any source combination. Per-contrast masks also require all four species.",
                "independence_status": "Each species derives from one reported supplier lot. Assay replicates and exact experiment identities remain distinct from independent donor replication.",
                "independence_warning": "Preserve assay-level measurements and shared control references. No paired-donor or cross-species biological pairing is assumed.",
                "study_transfer_warning": "This is a perfused 3D MPS with repeated ten-day exposures, unlike acute2D hepatocyte panels; study, culture system and perturbation differences remain coupled.",
                "drug_concentrations": "Unresolved at the RNA-seq sample level; no highest-dose inference from paper-wide Table2.",
                "culture_state_warning": "D4/D8/D14-versus-D0 are culture-state contrasts from a separate experiment and one process, not three independent drug perturbations."}
    return CuratedPanel(features, np.asarray(SPECIES), np.asarray([[r[s] for r in axis] for s in SPECIES]), raw, log, arms,
                        np.asarray(treated_rows), np.asarray(response), np.asarray(baseline), observations,
                        np.asarray(means), contrasts, np.asarray(eligibility), np.isfinite(log).all(axis=0), manifest)


def verify_inputs(source_dir, delivery_manifest):
    manifest = json.loads(Path(delivery_manifest).read_text())
    expected = {r["filename"]: r for r in manifest["files"] if r["accession"] == "GSE300171"}
    expected_names = {f"GSE300171_{prefix}_{FILE_TAG[s]}.csv.gz" for s in SPECIES for prefix in ["normcounts", "qc_hash"]}
    if set(expected) != expected_names:
        raise ValueError("Expected eight reviewed GSE300171 source files")
    result = {}
    for name in sorted(expected_names):
        path = Path(source_dir) / name
        actual = file_hash(path)
        if actual != expected[name]["sha256"] or path.stat().st_size != expected[name]["bytes"]:
            raise ValueError("Source checksum/size mismatch: " + name)
        result[name] = {"sha256": actual, "bytes": path.stat().st_size, "url": expected[name]["url"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE300171_sources")
    parser.add_argument("--mapping-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/orthology_release_freeze")
    parser.add_argument("--delivery-manifest", type=Path, default=ROOT / "docs/multisource_20261108/discovery/download_delivery_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE300171")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Choose a fresh output directory")
    sources = verify_inputs(args.source_dir, args.delivery_manifest)
    soft = args.source_dir / "GSE300171_GSM_full.soft"
    paper = args.source_dir / "PMC12624441.xml"
    if not soft.exists() or not paper.exists():
        raise ValueError("Reviewed full GEO metadata and original-publication XML are required")
    gsm = read_soft(soft)
    axis, map_audit = fourway_axis(args.mapping_dir)
    matrices, metadata = {}, []
    for species in SPECIES:
        tag = FILE_TAG[species]
        matrices[species] = read_normalized_counts(args.source_dir / f"GSE300171_normcounts_{tag}.csv.gz")
        qc = read_qc(args.source_dir / f"GSE300171_qc_hash_{tag}.csv.gz")
        metadata.extend(sample_metadata(species, matrices[species][1], qc, gsm))
    if len(metadata) != 93:
        raise ValueError("Expected all93 deposited assay records")
    manifest = {"dataset": "GSE300171", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_files": sources, "GEO_metadata_sha256": file_hash(soft), "publication_xml_sha256": file_hash(paper),
                "publication": {"doi": "10.1021/acsptsci.5c00554", "pmid": "41262563", "pmcid": "PMC12624441"},
                "mapping": map_audit, "script_sha256": file_hash(Path(__file__)),
                "hypothesis_or_performance_used_for_filtering": False}
    panel = assemble(axis, matrices, metadata, manifest)
    save_curated_panel(panel, args.out_dir)
    with (args.out_dir / "fourway_axis.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, SPECIES, delimiter="\t"); writer.writeheader(); writer.writerows(axis)
    final_manifest = json.loads((args.out_dir / "manifest.json").read_text())
    final_manifest["files"]["fourway_axis.tsv"] = file_hash(args.out_dir / "fourway_axis.tsv")
    coverage = json.loads((args.out_dir / "coverage.json").read_text())
    coverage["independent_perturbations"] = sorted(DRUGS)
    coverage["culture_state_contrasts"] = ["CULTURE_D4", "CULTURE_D8", "CULTURE_D14"]
    coverage["culture_state_is_independent_drug"] = False
    write_json(args.out_dir / "coverage.json", coverage)
    final_manifest["coverage"] = coverage
    final_manifest["files"]["coverage.json"] = file_hash(args.out_dir / "coverage.json")
    write_json(args.out_dir / "manifest.json", final_manifest)
    summary = {"arm_samples": len(panel.arm_metadata), "responses": len(panel.response),
               "drug_observations": sum(r["contrast_kind"] == "drug" for r in panel.observation_metadata),
               "culture_state_observations": sum(r["contrast_kind"] == "culture_state" for r in panel.observation_metadata),
               "contrasts": len(panel.contrast_metadata), "genes": len(panel.feature_ids), "complete_genes": int(panel.global_eligible_mask.sum()),
               "species": list(SPECIES), "drug_concentration_status": "unresolved; no inferred dosing value",
               "donors_per_species": "one supplier lot; exact donor count not verified from lot alone",
               "output": str(args.out_dir)}
    write_json(args.out_dir / "curation_summary.json", summary)
    final_manifest["files"]["curation_summary.json"] = file_hash(args.out_dir / "curation_summary.json")
    write_json(args.out_dir / "manifest.json", final_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
