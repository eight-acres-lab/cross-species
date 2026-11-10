#!/usr/bin/env python3
"""Curate native GSE105160 count contrasts with explicit donor/control links.

The single synthetic poly(I:C) stimulus supports source comparison, not validation
of a new-perturbation selection rule. Alternative human-genome alignments are kept
as technical sensitivity data, never as additional biological samples.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.multisource_data import CuratedPanel, file_hash, normalize_counts, save_curated_panel, load_curated_panel, assemble_matched_panel, unique_ortholog_pairs, biomart_query

STUDY = "GSE105160"
SPECIES = {
    "human": ("Homo sapiens", None, "Homo sapiens"),
    "mouse": ("Mus musculus", "mmusculus", "Mus musculus"),
    "rhesus": ("Macaca mulatta", "mmulatta", "Macaca mulatta"),
    "pigtail_macaque": ("Macaca nemestrina", "mnemestrina", "Macaca nemestrina"),
    "baboon": ("Papio anubis", "panubis", "Papio anubis"),
    "chimpanzee": ("Pan troglodytes", "ptroglodytes", "Pan troglodytes"),
    "bonobo": ("Pan paniscus", "ppaniscus", "Pan paniscus"),
    "gorilla": ("Gorilla gorilla", "ggorilla", "Gorilla gorilla"),
    "orangutan": ("Pongo abelii", "pabelii", "Pongo abelii"),
    "squirrel_monkey": ("Saimiri sciureus", "sbboliviensis", "Saimiri boliviensis boliviensis"),
}
TAXON_KEY = {record[0]: key for key, record in SPECIES.items()}
DEFAULT_ARCHIVE = ROOT / "outputs/multisource_20261108/curated_panels/GSE105160_input/GSE105160_RAW.tar"
DISCOVERY = ROOT / "docs/multisource_20261108/discovery"
EXPECTED_ARCHIVE_HASH = "71fbedebd0b182af5051a2725de837217ab85e97c176dbee2ca9c36b7adb01b7"


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _download(url, path):
    if path.exists():
        raise ValueError("Unsealed mapping cache already contains " + str(path))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "GSE105160-native-orthology-audit/1.0"}), timeout=120) as response:
                body = response.read(20_000_001)
            if len(body) > 20_000_000:
                raise ValueError("Unexpected mapping response size")
            path.write_bytes(body)
            return body.decode("utf-8")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_mapping(directory, endpoint="https://useast.ensembl.org/biomart/martservice"):
    directory = Path(directory)
    manifest_file = directory / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        for name, expected in manifest["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("Mapping cache hash mismatch: " + name)
        with (directory / "axis.tsv").open() as handle:
            return list(csv.DictReader(handle, delimiter="\t")), manifest
    directory.mkdir(parents=True, exist_ok=True)
    registry = _download(endpoint + "?type=registry", directory / "registry_before.xml")
    release = re.search(r'database="ensembl_mart_(\d+)"', registry)
    if release is None:
        raise ValueError("BioMart release not reported")
    attrs = _download(endpoint + "?type=attributes&dataset=hsapiens_gene_ensembl", directory / "attributes.tsv")
    pair_maps, stats = {}, {}
    for key, (taxon, prefix, reference_taxon) in SPECIES.items():
        if key == "human":
            continue
        if prefix + "_homolog_ensembl_gene\t" not in attrs:
            raise ValueError("Reference species unavailable in BioMart: " + key)
        xml = biomart_query(prefix)
        (directory / (key + ".query.xml")).write_text(xml)
        body = _download(endpoint + "?" + urllib.parse.urlencode({"query": xml}), directory / (key + ".tsv"))
        mapping, counts = unique_ortholog_pairs([line.split("\t") for line in body.splitlines() if line.strip()])
        if len(mapping) < 1000:
            raise ValueError("Unexpectedly small orthology result for " + key)
        pair_maps[key] = mapping
        stats[key] = {**counts, "sample_taxon": taxon, "reference_homology_taxon": reference_taxon}
        print(json.dumps({"species": key, "release": int(release.group(1)), "one_to_one_pairs": len(mapping)}), flush=True)
        time.sleep(.5)
    after = _download(endpoint + "?type=registry", directory / "registry_after.xml")
    end_release = re.search(r'database="ensembl_mart_(\d+)"', after)
    if end_release is None or end_release.group(1) != release.group(1):
        raise ValueError("BioMart gene release changed during mapping acquisition")
    common = sorted(set.intersection(*(set(mapping) for mapping in pair_maps.values())))
    axis = [{"human": gene, **{s: pair_maps[s][gene] for s in pair_maps}} for gene in common]
    with (directory / "axis.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SPECIES), delimiter="\t")
        writer.writeheader(); writer.writerows(axis)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
                "biomart_release": int(release.group(1)), "mapping_rule": "Human anchored, high-confidence one-to-one, unique in both directions; exact native stable identifiers only",
                "ten_way_human_anchors": len(axis), "pair_stats": stats,
                "annotation_limitations": ["Author counts use older Ensembl releases; exact stable-ID overlap is measured, retired IDs are not guessed.",
                    "Saimiri sciureus samples were aligned by the authors to SaiBol1.0. Orthology describes that reference genome; sample taxon is retained as deposited.",
                    "Orangutan author build is ppyg2; current Pongo abelii mapping is joined by exact native gene IDs. Annotation coverage is reported separately."],
                "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}}
    save_json(manifest_file, manifest)
    return axis, manifest


def parse_count_stream(stream):
    """Read two-column deposited gene counts; special alignment counters stay separate."""
    header = stream.readline().rstrip("\r\n").split("\t")
    if len(header) != 2 or header[0].lower() not in {"geneid", "gene_id"}:
        raise ValueError("Unexpected native count header")
    genes, values, counters = [], [], {}
    for line in stream:
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != 2:
            raise ValueError("Malformed count row")
        gene, text = fields
        value = float(text)
        if not np.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("Counts must be finite nonnegative integers")
        if gene.startswith("__"):
            counters[gene] = int(value)
            continue
        if not gene.startswith("ENS"):
            raise ValueError("Unrecognized native stable identifier: " + gene)
        genes.append(gene.split(".")[0]); values.append(value)
    if len(genes) != len(set(genes)) or not genes:
        raise ValueError("Duplicate or empty gene axis")
    return np.asarray(genes), np.asarray(values), header[1], counters


def load_annotations(sample_map, member_map):
    with Path(sample_map).open() as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))
    by_gsm = {row["gsm"]: row for row in samples}
    if len(by_gsm) != len(samples) or len(samples) != 180:
        raise ValueError("Expected 180 uniquely identified primary samples")
    for sample in samples:
        if sample["species"] not in TAXON_KEY or sample["condition"] not in {"mock", "polyIC"}:
            raise ValueError("Unsupported sample species or condition")
        expected_dose = .05 if sample["condition"] == "polyIC" else 0.
        if float(sample["time_h"]) != 24 or float(sample["polyIC_dose_ug_per_cm2"]) != expected_dose:
            raise ValueError("Unexpected sampled exposure metadata")
    with Path(member_map).open() as handle:
        members = list(csv.DictReader(handle, delimiter="\t"))
    native = [row for row in members if row["primary_native_mapping"] == "True"]
    if len(native) != 180 or len({row["metadata_gsm"] for row in native}) != 180:
        raise ValueError("Each primary GSM must have exactly one native count member")
    for row in native:
        sample = by_gsm[row["metadata_gsm"]]
        for field in ["species", "donor", "condition", "culture_replicate"]:
            if row[field] != sample[field]:
                raise ValueError("Member/sample annotation disagreement: " + field)
    return by_gsm, native, members


def build_panel(axis, matrices, annotations, provenance):
    """Keep cultures nested within donors; use each donor's own mock population."""
    species_order = list(SPECIES)
    features = np.asarray([row["human"] for row in axis])
    blocks, logs, metadata, normalization, coverage = [], [], [], {}, {}
    for species in species_order:
        genes, samples, counts = matrices[species]
        logcpm, libraries, quality = normalize_counts(counts)
        normalization[species] = quality
        index = {gene: i for i, gene in enumerate(genes)}
        wanted = [row[species] for row in axis]
        found = np.asarray([index.get(g, -1) for g in wanted])
        ok = found >= 0
        aligned = np.full((len(samples), len(axis)), np.nan)
        aligned_log = aligned.copy()
        aligned[:, ok] = counts[found[ok]].T
        aligned_log[:, ok] = logcpm[found[ok]].T
        blocks.append(aligned); logs.append(aligned_log)
        coverage[species] = {"native_gene_rows": len(genes), "mapped_axis_present": int(ok.sum()), "missing_axis_ids": [g for g, present in zip(wanted, ok) if not present]}
        for i, sample in enumerate(samples):
            original = annotations[sample]
            metadata.append({"study": STUDY, "sample_id": sample, "gsm": sample, "library": sample,
                "source_id": STUDY + "|" + species + "|primary_dermal_fibroblast", "species": species,
                "sample_taxon": SPECIES[species][0], "reference_homology_taxon": SPECIES[species][2],
                "cell_model": "primary_dermal_fibroblast", "perturbation": "polyIC", "time_h": float(original["time_h"]),
                "dose_value": float(original["polyIC_dose_ug_per_cm2"]) if original["condition"] == "polyIC" else 0.,
                "dose_unit": "ug/cm2", "role": "treated" if original["condition"] == "polyIC" else "vehicle",
                "replicate_label": original["culture_replicate"], "biological_id": original["donor"],
                "independent_unit_status": "Three deposited donor identities per species; cultures are nested replicates, not extra donors",
                "contrast_id": STUDY + "|" + species + "|polyIC", "library_size_fixed_native_support": float(libraries[i]),
                "control_match_rule": "All mock cultures belonging to the same deposited donor identity",
                "species_references_not_independent": True})
    raw, log = np.concatenate(blocks), np.concatenate(logs)
    donor_roles = {}
    for i, row in enumerate(metadata):
        donor_roles.setdefault((row["species"], row["biological_id"]), {}).setdefault(row["role"], []).append(i)
    for donor, arms in donor_roles.items():
        if set(arms) != {"treated", "vehicle"} or any(len(rows) != 3 for rows in arms.values()):
            raise ValueError("Expected three independent culture records per donor and arm: " + str(donor))
        if any(len({metadata[i]["replicate_label"] for i in rows}) != 3 for rows in arms.values()):
            raise ValueError("Culture labels must be unique within each donor and arm")
    observations, treated, responses, baselines, donor_means, donor_meta = [], [], [], [], [], []
    for (species, donor), arms in sorted(donor_roles.items()):
        controls = arms["vehicle"]
        baseline = log[controls].mean(axis=0)
        dr = []
        for i in arms["treated"]:
            response = log[i] - baseline
            item = dict(metadata[i]); item.update(control_arm_rows=controls, donor_pairing_verified=True,
                control_reference_type="same_donor_three_mock_culture_mean", response_dependency_group=STUDY + "|" + species + "|" + donor)
            observations.append(item); treated.append(i); responses.append(response); baselines.append(baseline); dr.append(response)
        donor_means.append(np.mean(dr, axis=0))
        donor_meta.append({"study": STUDY, "species": species, "biological_id": donor, "perturbation": "polyIC",
                           "treated_arm_rows": arms["treated"], "control_arm_rows": controls,
                           "response_definition": "Mean of three treated culture logCPM minus mean of three own-donor mock cultures"})
    means, contrast_meta = [], []
    for species in species_order:
        ti = [i for i, row in enumerate(metadata) if row["species"] == species and row["role"] == "treated"]
        ci = [i for i, row in enumerate(metadata) if row["species"] == species and row["role"] == "vehicle"]
        row = dict(metadata[ti[0]])
        row.update(treated_arm_rows=ti, control_arm_rows=ci, treated_replicates=len(ti), control_replicates=len(ci),
                   donors=sorted({metadata[i]["biological_id"] for i in ti}), n_donors=3,
                   estimand="Equal-donor mean of paired treated-minus-own-donor-mock log2CPM responses")
        contrast_meta.append(row)
        means.append(np.mean([v for v, m in zip(donor_means, donor_meta) if m["species"] == species], axis=0))
    global_mask = np.isfinite(log).all(axis=0)
    panel = CuratedPanel(features, np.asarray(species_order), np.asarray([[row[s] for row in axis] for s in species_order]), raw, log, metadata,
        np.asarray(treated), np.asarray(responses), np.asarray(baselines), observations, np.asarray(means), contrast_meta,
        np.tile(global_mask, (len(species_order), 1)), global_mask,
        {**provenance, "normalization": normalization, "native_mapping_coverage": coverage,
         "axis_rule": "One ten-way complete fixed axis for every source subset; missing native IDs are not zero-filled",
         "independence_status": "Thirty deposited donor identities across ten species; three within-donor cultures per arm. Donor identities do not independently establish animal lineage/litter relationships.",
         "independence_warning": "Resample donors jointly with all their cultures and own controls; cultures are nested. One stimulus supplies no held-out perturbation validation.",
         "single_stimulus": True, "technical_alignment_versions_are_not_samples": True})
    return panel, np.asarray(donor_means), donor_meta


def build_alignment_sensitivity(panel_dir, archive_path, sample_map, member_map, out_dir):
    """Reconcile both mappings of the same cells, without judging prediction quality."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Technical sensitivity output must be empty")
    panel = load_curated_panel(panel_dir)
    if file_hash(archive_path) != EXPECTED_ARCHIVE_HASH:
        raise ValueError("Archive changed")
    samples, _, members = load_annotations(sample_map, member_map)
    target_genes = panel.feature_ids[panel.global_eligible_mask]
    lookup = {row["sample_id"]: i for i, row in enumerate(panel.arm_metadata)}
    included_species = [s for s in panel.species_order if s != "mouse"]
    aligned_blocks, native_blocks, ids, metadata, coverage = [], [], [], [], {}
    with tarfile.open(archive_path, "r:") as archive:
        for species in included_species:
            entries = sorted([r for r in members if r["mapping"] == "HumanMapped" and TAXON_KEY[r["species"]] == species], key=lambda r: r["metadata_gsm"])
            if len(entries) != 18:
                raise ValueError("Expected both mappings for all same biological records")
            gene_axis, vectors = None, []
            for row in entries:
                with gzip.open(archive.extractfile(row["member"]), "rt") as handle:
                    genes, vector, _, _ = parse_count_stream(handle)
                if gene_axis is None:
                    gene_axis = genes
                elif not np.array_equal(gene_axis, genes):
                    raise ValueError("HumanMapped gene axis varies")
                vectors.append(vector)
            raw = np.stack(vectors, axis=1)
            log, _, quality = normalize_counts(raw)
            index = {g: i for i, g in enumerate(gene_axis)}
            if not all(g in index for g in target_genes):
                raise ValueError("HumanMapped data lack a primary global-axis gene")
            chosen = np.asarray([index[g] for g in target_genes])
            aligned_blocks.append(log[chosen].T)
            sample_ids = [r["metadata_gsm"] for r in entries]
            native_blocks.append(panel.arm_logcpm[[lookup[s] for s in sample_ids]][:, panel.global_eligible_mask])
            ids.extend(sample_ids)
            metadata.extend([panel.arm_metadata[lookup[s]] for s in sample_ids])
            coverage[species] = {"native_axis_for_human_alignment": len(gene_axis),
                "count_fraction_on_primary_axis": (raw[chosen].sum(0) / raw.sum(0)).tolist(),
                "normalization": quality}
    mapped, native = np.concatenate(aligned_blocks), np.concatenate(native_blocks)
    groups = sorted({(r["species"], r["biological_id"]) for r in metadata})
    mapped_donors, native_donors = [], []
    for species, donor in groups:
        treated = [i for i, r in enumerate(metadata) if r["species"] == species and r["biological_id"] == donor and r["role"] == "treated"]
        control = [i for i, r in enumerate(metadata) if r["species"] == species and r["biological_id"] == donor and r["role"] == "vehicle"]
        if len(treated) != 3 or len(control) != 3:
            raise ValueError("Invalid donor pairing in technical sensitivity")
        mapped_donors.append(mapped[treated].mean(0) - mapped[control].mean(0))
        native_donors.append(native[treated].mean(0) - native[control].mean(0))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "same_sample_alignment.npz", gene_ids=target_genes, sample_ids=np.asarray(ids),
        native_arm_logcpm=native, humanmapped_arm_logcpm=mapped,
        donor_species=np.asarray([s for s, _ in groups]), donor_ids=np.asarray([d for _, d in groups]),
        native_donor_response=np.asarray(native_donors), humanmapped_donor_response=np.asarray(mapped_donors))
    save_json(out_dir / "arm_metadata.json", metadata)
    save_json(out_dir / "manifest.json", {"primary_panel_manifest_sha256": file_hash(Path(panel_dir) / "manifest.json"),
        "count_archive_sha256": file_hash(archive_path), "script_sha256": file_hash(Path(__file__)),
        "species": list(included_species), "arm_samples": len(ids), "donor_responses": len(groups), "genes": len(target_genes),
        "scope": "Technical realignment sensitivity of identical biological samples on the primary fixed native orthology axis; mouse lacks HumanMapped data and is not fabricated. No prediction performance evaluated here.",
        "coverage": coverage, "files": {p.name: file_hash(p) for p in out_dir.iterdir() if p.is_file()}})
    print(json.dumps({"technical_alignment_samples": len(ids), "donors": len(groups), "genes": len(target_genes), "out_dir": str(out_dir)}), flush=True)


def read_pig_counts(path):
    """Validate all adjacent duplicated Atlas run columns and retain one copy."""
    with Path(path).open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:2] != ["Gene ID", "Gene Name"] or (len(header) - 2) % 2:
            raise ValueError("Unexpected pig Atlas header")
        labels = header[2:]
        if any(labels[i] != labels[i + 1] for i in range(0, len(labels), 2)):
            raise ValueError("Pig duplicate run columns are not adjacent matching labels")
        samples = labels[::2]
        if len(samples) != len(set(samples)):
            raise ValueError("Pig unique run axis contains repeated identifiers")
        genes, values = [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Malformed pig count row")
            numbers = np.asarray(row[2:], dtype=float)
            if not np.isfinite(numbers).all() or (numbers < 0).any() or not np.equal(numbers, np.floor(numbers)).all():
                raise ValueError("Pig counts must be finite nonnegative integers")
            if not np.array_equal(numbers[::2], numbers[1::2]):
                raise ValueError("Duplicate pig count columns differ; cannot choose or sum")
            if not row[0].startswith("ENSSSCG"):
                raise ValueError("Expected stable pig gene identifier")
            genes.append(row[0].split(".")[0]); values.append(numbers[::2])
    if len(genes) != len(set(genes)):
        raise ValueError("Duplicate pig genes")
    return np.asarray(genes), samples, np.stack(values)


def pig_annotations(rows):
    annotations = []
    by_run = {r["sample_id"]: r for r in rows}
    if len(by_run) != len(rows):
        raise ValueError("Duplicate pig sample metadata")
    for row in rows:
        control = row["is_control"] == "True"
        genotype = row["genotype"]
        code = genotype.split(" ", 1)[0]
        if row["species"] != "Sus scrofa" or code not in {"A", "H", "V"}:
            raise ValueError("Unexpected pig species/genotype metadata")
        counterpart = by_run.get(row["control_run"])
        if counterpart is None or counterpart["is_control"] != "True" or counterpart["individual"] != row["individual"] or counterpart["genotype"] != genotype:
            raise ValueError("Pig treatment must have explicit same-individual vehicle")
        if float(row["time_h"]) != 2:
            raise ValueError("Unexpected pig sampling time")
        dex, lps = float(row["dexamethasone_nM"]), float(row["LPS_ug_per_mL"])
        expected = {"vehicle": (0, 0), "dexamethasone": (5, 0), "lipopolysaccharide": (0, 10), "dexamethasone and lipopolysaccharide": (5, 10)}
        if row["stimulus"] not in expected or (dex, lps) != expected[row["stimulus"]]:
            raise ValueError("Pig stimulus/dose metadata conflict")
        source_id = "E-MTAB-9808|pig|PBMC|GRAla610Val_" + code
        item = {"study": "E-MTAB-9808", "sample_id": row["sample_id"], "library": row["sample_id"], "species": "pig",
            "sample_taxon": "Sus scrofa", "cell_model": "cultured_PBMC", "source_id": source_id,
            "genotype": genotype, "biological_id": row["individual"], "sex": row["sex"],
            "perturbation": row["stimulus"], "time_h": 2., "dexamethasone_nM": dex, "LPS_ug_per_mL": lps,
            "dose_value": None, "dose_unit": "Separate ligand-specific fields; do not collapse combination to a scalar dose",
            "role": "vehicle" if control else "treated", "replicate_label": row["sample_id"],
            "contrast_id": source_id + "|" + row["stimulus"], "control_libraries": [row["control_run"]],
            "independent_unit_status": "Individual pig; all three treated populations share that individual's own vehicle", 
            "control_match_rule": "Exact within-individual, same genotype vehicle Run"}
        annotations.append(item)
    return annotations


def prepare_pig(counts_path, paired_map, mapping_dir, out_dir):
    expected = "28ce42635107bd7d816490e9f94a114b836f0190fa9b0235b436af89e53dd6b7"
    if file_hash(counts_path) != expected:
        raise ValueError("Pig raw-count file differs from reviewed official download")
    mapping_dir = Path(mapping_dir)
    mapping_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = mapping_dir / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        for name, digest in manifest["files"].items():
            if file_hash(mapping_dir / name) != digest:
                raise ValueError("Pig mapping cache hash mismatch")
    else:
        endpoint = "https://useast.ensembl.org/biomart/martservice"
        registry = _download(endpoint + "?type=registry", mapping_dir / "registry_before.xml")
        release = re.search(r'database="ensembl_mart_(\d+)"', registry)
        xml = biomart_query("sscrofa")
        (mapping_dir / "pig.query.xml").write_text(xml)
        _download(endpoint + "?" + urllib.parse.urlencode({"query": xml}), mapping_dir / "pig.tsv")
        after = _download(endpoint + "?type=registry", mapping_dir / "registry_after.xml")
        after_release = re.search(r'database="ensembl_mart_(\d+)"', after)
        if release is None or after_release is None or release.group(1) != after_release.group(1):
            raise ValueError("Pig mapping release changed or unavailable")
        manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
                    "biomart_release": int(release.group(1)), "files": {p.name: file_hash(p) for p in mapping_dir.iterdir() if p.is_file()}}
        save_json(manifest_file, manifest)
    raw_mapping = [line.split("\t") for line in (mapping_dir / "pig.tsv").read_text().splitlines() if line.strip()]
    mapping, statistics = unique_ortholog_pairs(raw_mapping)
    axis = [{"human": h, "pig": p} for h, p in sorted(mapping.items())]
    genes, samples, counts = read_pig_counts(counts_path)
    with Path(paired_map).open() as handle:
        annotations = pig_annotations(list(csv.DictReader(handle, delimiter="\t")))
    if len(samples) != 96 or len({r["biological_id"] for r in annotations}) != 24:
        raise ValueError("Unexpected pig sample/individual count")
    panel = assemble_matched_panel(axis, {"pig": (genes, samples, counts)}, annotations,
        {"study": "E-MTAB-9808", "created_utc": datetime.now(timezone.utc).isoformat(),
         "source_url": "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9808",
         "input_files": {str(counts_path): file_hash(counts_path), str(paired_map): file_hash(paired_map)},
         "mapping_manifest_sha256": file_hash(manifest_file), "mapping_release": manifest["biomart_release"], "mapping_stats": statistics,
         "duplicate_columns": "Every one of 96 adjacent duplicate run pairs checked equal over all native genes; one retained, never summed",
         "source_only": True, "human_targets_available": False,
         "scope": "Cultured pig PBMC source-only study with three genotype strata and three noninfectious ligand exposures. Human condition matching is pending; no human response or cross-species predictive performance inferred.",
         "script_sha256": file_hash(Path(__file__))})
    save_curated_panel(panel, out_dir)
    print(json.dumps({"study": "E-MTAB-9808", "unique_arms": len(panel.arm_metadata), "individuals": 24,
        "treated_responses": len(panel.observation_metadata), "genotype_specific_contrasts": len(panel.contrast_metadata),
        "human_anchored_genes": len(panel.feature_ids), "complete_genes": int(panel.global_eligible_mask.sum()), "out_dir": str(out_dir)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--sample-map", type=Path, default=DISCOVERY / "gse105160_donor_sample_map.tsv")
    parser.add_argument("--member-map", type=Path, default=DISCOVERY / "gse105160_archive_member_map.tsv")
    parser.add_argument("--mapping-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE105160_orthology")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE105160")
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--technical-only", action="store_true")
    parser.add_argument("--technical-out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE105160_alignment_sensitivity")
    parser.add_argument("--pig-only", action="store_true")
    parser.add_argument("--pig-counts", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/E-MTAB-9808_input/E-MTAB-9808-raw-counts.tsv")
    parser.add_argument("--pig-paired-map", type=Path, default=DISCOVERY / "e_mtab_9808_paired_sample_map.tsv")
    parser.add_argument("--pig-mapping-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/E-MTAB-9808_orthology")
    parser.add_argument("--pig-out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/E-MTAB-9808")
    args = parser.parse_args()
    if args.pig_only:
        prepare_pig(args.pig_counts, args.pig_paired_map, args.pig_mapping_dir, args.pig_out_dir)
        return
    if args.technical_only:
        build_alignment_sensitivity(args.out_dir, args.archive, args.sample_map, args.member_map, args.technical_out_dir)
        return
    axis, mapping_manifest = fetch_mapping(args.mapping_dir)
    if args.mapping_only:
        print(json.dumps({"mapping_anchors": len(axis), "directory": str(args.mapping_dir)})); return
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output must be empty")
    actual_hash = file_hash(args.archive)
    if actual_hash != EXPECTED_ARCHIVE_HASH:
        raise ValueError("Count archive checksum differs from independently reviewed official download")
    annotations, native_members, all_members = load_annotations(args.sample_map, args.member_map)
    matrices, library_records = {}, []
    with tarfile.open(args.archive, "r:") as archive:
        human_aligned = {r["metadata_gsm"]: r for r in all_members if r["mapping"] == "HumanMapped"}
        for species, (taxon, _, _) in SPECIES.items():
            members = sorted([r for r in native_members if r["species"] == taxon], key=lambda row: row["metadata_gsm"])
            if len(members) != 18:
                raise ValueError("Expected 18 native arm profiles for " + species)
            gene_axis, samples, vectors = None, [], []
            for row in members:
                info = archive.getmember(row["member"])
                if not info.isfile() or info.size > 2_000_000:
                    raise ValueError("Unexpected count member type or size")
                with gzip.open(archive.extractfile(info), "rt") as stream:
                    genes, vector, header, counters = parse_count_stream(stream)
                matched_header = None
                if row["mapping"] == "SpeciesMapped" and row["metadata_gsm"] in human_aligned:
                    other = human_aligned[row["metadata_gsm"]]
                    with gzip.open(archive.extractfile(other["member"]), "rt") as handle:
                        matched_header = handle.readline().rstrip("\r\n").split("\t")[1]
                    if header != matched_header:
                        raise ValueError("Native/HumanMapped original library headers disagree for " + row["metadata_gsm"])
                if gene_axis is None:
                    gene_axis = genes
                elif not np.array_equal(gene_axis, genes):
                    raise ValueError("Native gene order varies across samples for " + species)
                samples.append(row["metadata_gsm"]); vectors.append(vector)
                library_records.append({**row, "original_library_header": header, "alignment_counters": counters,
                    "genes": len(genes), "gene_count_total": float(vector.sum()),
                    "native_humanmapped_header_matches": None if matched_header is None else header == matched_header})
            matrices[species] = gene_axis, samples, np.stack(vectors, axis=1)
    provenance = {"study": STUDY, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE105160",
                  "input_files": {str(path): file_hash(path) for path in [args.archive, args.sample_map, args.member_map]},
                  "mapping_manifest": str(args.mapping_dir / "manifest.json"), "mapping_manifest_sha256": file_hash(args.mapping_dir / "manifest.json"),
                  "annotation_limitations": mapping_manifest["annotation_limitations"],
                  "analysis_scope": "Single synthetic poly(I:C) response, source comparability and measured-source combination descriptions; no unseen-drug prediction or reliable source-selection validation",
                  "script_sha256": file_hash(Path(__file__))}
    panel, donor_response, donor_metadata = build_panel(axis, matrices, annotations, provenance)
    save_curated_panel(panel, args.out_dir)
    np.savez_compressed(args.out_dir / "donor_responses.npz", response=donor_response,
                        species=np.asarray([r["species"] for r in donor_metadata]), donors=np.asarray([r["biological_id"] for r in donor_metadata]),
                        gene_ids=panel.feature_ids, global_eligible_mask=panel.global_eligible_mask)
    save_json(args.out_dir / "donor_metadata.json", donor_metadata)
    save_json(args.out_dir / "native_archive_records.json", library_records)
    # Extend the sealed manifest to include all added provenance artifacts.
    manifest = json.loads((args.out_dir / "manifest.json").read_text())
    manifest["coverage"]["verified_donors"] = 30
    manifest["coverage"]["verified_donor_definition"] = "Deposited donor/source-line labels and matched own-donor mock cultures; not an independent audit of biological procurement"
    coverage = json.loads((args.out_dir / "coverage.json").read_text())
    coverage.update({"verified_donors": 30, "donor_definition": manifest["coverage"]["verified_donor_definition"]})
    save_json(args.out_dir / "coverage.json", coverage)
    manifest["files"] = {p.name: file_hash(p) for p in args.out_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
    save_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps({"study": STUDY, "arm_profiles": len(panel.arm_metadata), "donors": len(donor_metadata), "response_observations": len(panel.observation_metadata),
                      "ortholog_anchors": len(panel.feature_ids), "global_comparison_genes": int(panel.global_eligible_mask.sum()), "out_dir": str(args.out_dir)}), flush=True)


if __name__ == "__main__":
    main()
