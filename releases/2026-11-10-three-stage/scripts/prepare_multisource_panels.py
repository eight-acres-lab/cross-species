#!/usr/bin/env python3
"""Curate real multi-source panels on an explicit frozen orthology axis."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.multisource_data import (SPECIES, assemble_panel, assemble_matched_panel, fetch_fiveway_mapping,
    file_hash, load_fiveway_mapping, read_counts_csv, save_curated_panel, unique_ortholog_pairs, write_json)

SPECIES_NAMES = {"Homo sapiens": "human", "Mus musculus": "mouse", "Rattus norvegicus": "rat",
                 "Macaca fascicularis": "cyno", "Canis lupus familiaris": "dog"}


def g330_metadata(path):
    rows = []
    with Path(path).open() as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            species = SPECIES_NAMES[record["species"]]
            text = record["treatment"]
            dose = re.search(r'([0-9.]+)\s*(nM|µM|μM|uM|mM)\b', text)
            # Vehicle rows inherit the intended treatment dose only as contrast context.
            intended = {"CDDO": (100., "nM"), "KI696": (10., "uM"), "THAP": (1., "uM")}[record["agent"]]
            if record["role"] == "treated":
                if dose is None or float(dose.group(1)) != intended[0]:
                    raise ValueError("Treatment dose is not independently supported by sample text")
                unit = dose.group(2).replace("µ", "u").replace("μ", "u")
                if unit != intended[1]:
                    raise ValueError("Treatment concentration units disagree")
            context = "primary_hepatocyte"
            contrast = "GSE330151|" + species + "|" + record["agent"]
            rows.append({"study": "GSE330151", "species": species, "source_id": "GSE330151|" + species + "|" + context,
                "cell_model": context, "tissue": "liver", "perturbation": record["agent"],
                "time_h": float(record["incubation_h"]), "dose_value": intended[0], "dose_unit": intended[1],
                "dose_role": "intended treatment concentration; vehicles use matched carrier only",
                "role": record["role"], "contrast_id": contrast, "library": record["library"],
                "sample_id": record["geo_accession"], "replicate_label": record["replicate_index_from_title"],
                "biological_id": None, "independent_unit_status": "labelled_replicate_donor_unverified",
                "donor_identity_verified": False, "quality": "source numeric/design audit passed; independence unresolved",
                "control_match_rule": "GSM title explicitly names matched compound and exposure time",
                "units": "log2(CPM+1)", "count_units": "Salmon/tximport fractional estimated counts",
                "title": record["title"], "treatment_text": text})
    if len(rows) != 90:
        raise ValueError("Expected all 90 original G330151 samples")
    return rows


def prepare_g130(out_dir, mapping_path):
    from prepare_feedback_public_counts import SPECS, read_raw_counts, sample_annotations, verified_source
    original = ROOT / "artifacts/feedback_public_sources_2026-11-07"
    manifest = json.loads((original / "download_manifest.json").read_text())
    cached_meta = ROOT / "docs/data_discovery_2026-11-07/agent_expanded_screen.json"
    gsm_rows = json.loads(cached_meta.read_text())["extra_sample_metadata"]["GSE130983"]["rows"]
    gsm = {r["title"][0]: r for r in gsm_rows}
    mouse_mapping = mapping_path.parent / "mouse.tsv"
    if not mouse_mapping.exists():
        raise ValueError("G130 requires the fresh complete human–mouse pair query beside the frozen five-way axis")
    with mouse_mapping.open() as stream:
        pairs, filtering = unique_ortholog_pairs(list(csv.reader(stream, delimiter="\t")))
    axis = [{"human": h, "mouse": pairs[h]} for h in sorted(pairs)]
    matrices, metadata, sources = {}, [], []
    for spec in [s for s in SPECS if s.study == "GSE130983"]:
        source = verified_source(manifest, original, spec)
        genes, samples, counts = read_raw_counts(Path(source["path"]), spec)
        matrices[spec.species] = (genes, samples, counts)
        sources.append(source)
        for row in sample_annotations(spec, samples):
            excluded = bool(row["excluded_reason"])
            if not excluded and row["sample_id"] not in gsm:
                raise ValueError("Admitted sample lacks an exact GSM title")
            native_gsm = gsm.get(row["sample_id"], {})
            protocol = " ".join(native_gsm.get("treatment_protocol_ch1", []))
            treated = not row["is_control"] and not excluded
            if treated and ("5ng/ml" not in protocol or "24h" not in protocol):
                raise ValueError("Admitted exposure lacks reviewed dose/time provenance")
            group = "GSE130983|" + spec.species + "|lung_fibroblast|" + row["genotype"]
            control = next(a["sample_id"] for a in sample_annotations(spec, samples)
                           if a["is_control"] and a["donor"] == row["donor"])
            metadata.append({"study": "GSE130983", "species": spec.species, "source_id": group,
                "cell_model": "primary_lung_fibroblast", "tissue": "lung", "genotype": row["genotype"],
                "perturbation": row["perturbation"], "time_h": 24. if treated else None,
                "dose_value": 5. if treated else None, "dose_unit": "ng/mL" if treated else None,
                "control_time_status": "Author baseline; elapsed baseline time not independently reported",
                "role": "excluded" if excluded else "vehicle" if row["is_control"] else "treated",
                "contrast_id": group + "|" + row["perturbation"], "library": row["sample_id"],
                "sample_id": native_gsm.get("geo_accession", [row["sample_id"]])[0],
                "biological_id": row["donor"], "replicate_label": row["replicate_label"],
                "independent_unit_status": "author_source_line_label_shared_own_control",
                "donor_identity_verified": False, "control_libraries": [control], "units": "log2(CPM+1)",
                "count_units": "Deposited nonnegative integer gene counts", "quality": row["excluded_reason"] or "Exact GSM title/exposure and count hash verified",
                "excluded_reason": row["excluded_reason"], "treatment_text": protocol})
    panel = assemble_matched_panel(axis, matrices, metadata, {"contract_version": "multisource_v1", "study": "GSE130983",
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "source_files": sources,
        "mapping_pair_sha256": file_hash(mouse_mapping), "mapping_filter_counts": filtering,
        "sample_metadata_sha256": file_hash(cached_meta), "script_sha256": file_hash(__file__),
        "data_module_sha256": file_hash(ROOT / "src/cross_species/multisource_data.py"),
        "limitations": ["Mouse IL11 six raw-count columns retained as excluded arms; no response or cross-species IL11 claim.",
            "Human five and mouse WT3/KO3 are author source-line labels; no litter/batch independence established.",
            "WT/KO source contexts remain separate; only TGFB has admitted human–mouse coverage.",
            "Baseline timing is not a verified time-matched 24h vehicle. Shared controls link human IL11/TGFB responses."]})
    save_curated_panel(panel, out_dir / "GSE130983")
    print(json.dumps({"study": "GSE130983", "raw_arm_rows": len(panel.arm_metadata), "responses": len(panel.observation_metadata),
        "source_context_contrasts": len(panel.contrast_metadata), "fixed_complete_genes": int(panel.global_eligible_mask.sum())}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels")
    parser.add_argument("--g330-dir", type=Path, default=ROOT / "outputs/biorxiv_sprint_20261107/gse330151_fetch")
    parser.add_argument("--orthologs", type=Path, help="Existing frozen five-way TSV; use --fetch-orthologs to obtain a new freeze")
    parser.add_argument("--fetch-orthologs", action="store_true")
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--panel", choices=["GSE330151", "GSE130983"], default="GSE330151")
    args = parser.parse_args()
    if args.orthologs and args.fetch_orthologs:
        parser.error("Select an explicit mapping or fetch the same-release mapping")
    if args.fetch_orthologs:
        axis, mapping_manifest = fetch_fiveway_mapping(args.out_dir / "orthology_release_freeze")
        mapping_path = args.out_dir / "orthology_release_freeze/axis.tsv"
    else:
        mapping_path = args.orthologs or args.out_dir / "orthology_release_freeze/axis.tsv"
        axis = load_fiveway_mapping(mapping_path)
        frozen = mapping_path.parent / "manifest.json"
        mapping_manifest = json.loads(frozen.read_text()) if frozen.exists() else {"provided_axis_sha256": file_hash(mapping_path), "release": "not_documented"}
    if args.mapping_only:
        print(json.dumps({"fiveway_genes": len(axis), "mapping": str(mapping_path)}, indent=2)); return
    if args.panel == "GSE130983":
        prepare_g130(args.out_dir, mapping_path); return
    acquisition = json.loads((args.g330_dir / "fetch_manifest.json").read_text())
    files = acquisition["files"]
    for entry in files:
        if entry.get("status") not in {"cached", "downloaded"}:
            raise ValueError("Acquisition manifest does not admit a complete file")
        name = entry.get("filename", entry.get("name", ""))
        path = args.g330_dir / "GSE330151" / name
        if not path.exists() and entry.get("path"):
            path = Path(entry["path"])
        if not path.exists() or file_hash(path) != entry["sha256"]:
            raise ValueError("Acquisition hash mismatch: " + name)
    matrices = {}
    for species in SPECIES:
        path = args.g330_dir / "GSE330151" / ("GSE330151_" + species + "_expression_matrix.genes.raw_counts.csv.gz")
        matrices[species] = read_counts_csv(path)
    panel = assemble_panel(axis, matrices, g330_metadata(args.g330_dir / "sample_design.tsv"),
        {"contract_version": "multisource_v1", "study": "GSE330151", "prepared_utc": datetime.now(timezone.utc).isoformat(),
         "mapping": mapping_manifest, "mapping_axis_sha256": file_hash(mapping_path),
         "input_manifest_sha256": file_hash(args.g330_dir / "fetch_manifest.json"),
         "sample_design_sha256": file_hash(args.g330_dir / "sample_design.tsv"),
         "script_sha256": file_hash(__file__), "data_module_sha256": file_hash(ROOT / "src/cross_species/multisource_data.py"),
         "limitations": ["Three compounds, two stress pathways; broad unseen-perturbation source selection unestablished.",
             "Donor identities and cross-arm pairing unverified; use separate-arm replicate uncertainty.",
             "Dog missing genes require a fixed incomplete native-library support; missing counts were not zero-filled.",
             "Mouse strain conflict between GSM and article remains unresolved; no donor identity inferred."]})
    save_curated_panel(panel, args.out_dir / "GSE330151")
    print(json.dumps(panel.manifest["normalization"], indent=2))
    print(json.dumps({"arm_rows": len(panel.arm_metadata), "contrasts": len(panel.contrast_metadata),
                      "axis_genes": len(panel.feature_ids), "fixed_eligible_genes": int(panel.global_eligible_mask.sum())}, indent=2))


if __name__ == "__main__":
    main()
