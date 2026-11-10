"""Frozen, replicate-preserving cross-species response panels; no fitted model."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np

SPECIES = ("human", "mouse", "rat", "cyno", "dog")
BIOMART_PREFIX = {"mouse": "mmusculus", "rat": "rnorvegicus", "cyno": "mfascicularis", "dog": "clfamiliaris"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unique_ortholog_pairs(rows):
    """Strict high-confidence one-to-one pairs, with sticky conflict rejection."""
    candidates, returned = {}, 0
    for row in rows:
        if len(row) != 4:
            raise ValueError("BioMart orthology response needs exactly four attributes")
        human, partner, kind, confidence = [x.strip() for x in row]
        returned += 1
        if (not human.startswith("ENSG") or not partner or kind != "ortholog_one2one"
                or confidence != "1"):
            continue
        candidates.setdefault(human, set()).add(partner)
    one = {h: next(iter(p)) for h, p in candidates.items() if len(p) == 1}
    reverse = {}
    for h, p in one.items():
        reverse.setdefault(p, set()).add(h)
    accepted = {h: p for h, p in one.items() if len(reverse[p]) == 1}
    return accepted, {"returned_rows": returned, "candidate_human_ids": len(candidates),
        "human_conflicts": sum(len(p) != 1 for p in candidates.values()),
        "partner_conflicts": sum(len(h) != 1 for h in reverse.values()), "accepted_pairs": len(accepted)}


def biomart_query(prefix):
    attributes = ["ensembl_gene_id", prefix + "_homolog_ensembl_gene",
                  prefix + "_homolog_orthology_type", prefix + "_homolog_orthology_confidence"]
    return ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">'
        '<Dataset name="hsapiens_gene_ensembl" interface="default">' +
        ''.join('<Attribute name="' + value + '"/>' for value in attributes) + '</Dataset></Query>')


def fetch_fiveway_mapping(cache_dir, endpoint="https://useast.ensembl.org/biomart/martservice"):
    """Fetch four sequential same-release queries once; reuse only a hash-verified freeze."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["files"].items():
            if file_hash(cache_dir / name) != expected:
                raise ValueError("Frozen orthologue cache hash mismatch: " + name)
        return load_fiveway_mapping(cache_dir / "axis.tsv"), manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    files, pair_maps, steps = {}, {}, {}
    def acquire(name, url):
        target = cache_dir / name
        if target.exists():
            raise ValueError("Incomplete mapping cache exists; use a fresh directory: " + str(target))
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read()
        target.write_bytes(body)
        files[name] = file_hash(target)
        return body.decode("utf-8")
    registry = acquire("registry.xml", endpoint + "?type=registry")
    release = re.search(r'database="ensembl_mart_(\d+)"', registry)
    if release is None:
        raise ValueError("BioMart registry did not identify the gene database release")
    attrs = acquire("attributes.tsv", endpoint + "?type=attributes&dataset=hsapiens_gene_ensembl")
    for species, prefix in BIOMART_PREFIX.items():
        if prefix + "_homolog_ensembl_gene\t" not in attrs:
            raise ValueError("Current BioMart does not expose expected species: " + species)
        query = biomart_query(prefix)
        (cache_dir / (species + ".query.xml")).write_text(query)
        files[species + ".query.xml"] = file_hash(cache_dir / (species + ".query.xml"))
        body = acquire(species + ".tsv", endpoint + "?" + urllib.parse.urlencode({"query": query}))
        rows = [line.split("\t") for line in body.splitlines() if line.strip()]
        pairs, stats = unique_ortholog_pairs(rows)
        if len(pairs) < 1000:
            raise ValueError("Unexpectedly small valid orthology result: " + species)
        pair_maps[species], steps[species] = pairs, stats
        print({"orthology_species": species, "release": release.group(1), **stats}, flush=True)
    # Verify the service did not move between sequential pair requests.
    end_registry = acquire("registry_after.xml", endpoint + "?type=registry")
    end_release = re.search(r'database="ensembl_mart_(\d+)"', end_registry)
    if end_release is None or end_release.group(1) != release.group(1):
        raise ValueError("BioMart release changed during acquisition")
    shared = sorted(set.intersection(*(set(m) for m in pair_maps.values())))
    axis = [{"human": h, **{s: pair_maps[s][h] for s in BIOMART_PREFIX}} for h in shared]
    with (cache_dir / "axis.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, SPECIES, delimiter="\t")
        writer.writeheader(); writer.writerows(axis)
    files["axis.tsv"] = file_hash(cache_dir / "axis.tsv")
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint,
        "biomart_gene_release": int(release.group(1)), "rule": "Human anchored, ortholog_one2one, confidence=1, unique both directions; fresh four pair queries from one release",
        "pair_filter_counts": steps, "fiveway_axis_genes": len(axis), "files": files,
        "old_release_maps_used": False, "symbols_used_for_orthology": False}
    write_json(manifest_path, manifest)
    return axis, manifest


def load_fiveway_mapping(path):
    with Path(path).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if set(reader.fieldnames or []) != set(SPECIES):
            raise ValueError("Five-way axis must name all five native species axes")
        axis = list(reader)
    if not axis:
        raise ValueError("Empty orthology axis")
    for species in SPECIES:
        ids = [row[species] for row in axis]
        if any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Orthology axis is not bidirectionally unique: " + species)
    return sorted(axis, key=lambda row: row["human"])


def read_counts_csv(path):
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        samples = header[1:]
        if not samples or len(samples) != len(set(samples)):
            raise ValueError("Duplicate/empty count sample axis")
        genes, values = [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Malformed raw-count row width")
            genes.append(row[0].split(".")[0])
            values.append([np.nan if v in ("", "NA", "NaN") else float(v) for v in row[1:]])
    matrix = np.asarray(values, dtype=np.float64)
    if len(set(genes)) != len(genes) or any(not g for g in genes):
        raise ValueError("Duplicate/empty native gene identifiers")
    if np.isinf(matrix).any() or np.any(matrix[np.isfinite(matrix)] < 0):
        raise ValueError("Counts must be nonnegative estimated counts or documented NA")
    return np.asarray(genes), samples, matrix


def normalize_counts(counts):
    """CPM on fixed complete native-gene support; never fill missing counts."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or np.isinf(counts).any() or np.any(counts[np.isfinite(counts)] < 0):
        raise ValueError("Invalid estimated counts")
    support = np.isfinite(counts).all(axis=1)
    if not support.any():
        raise ValueError("No complete native genes for library normalization")
    libraries = counts[support].sum(axis=0)
    if (libraries <= 0).any() or not np.isfinite(libraries).all():
        raise ValueError("Nonpositive library size")
    normalized = np.log2(1 + counts / libraries[None, :] * 1e6)
    observed_totals = np.nansum(counts, axis=0)
    return normalized, libraries, {"native_genes": len(counts), "normalization_support_genes": int(support.sum()),
        "incomplete_native_genes": int((~support).sum()), "incomplete_count_entries": int(np.isnan(counts).sum()),
        "libraries": libraries.tolist(), "observed_count_totals": observed_totals.tolist(),
        "known_count_fraction_outside_fixed_support": ((observed_totals - libraries) / observed_totals).tolist(),
        "rule": "Use every native gene complete across this species' sampled columns; define library support before orthology selection. Missing values remain NaN. Complete matrices use their whole native matrix."}


@dataclass
class CuratedPanel:
    feature_ids: np.ndarray
    species_order: np.ndarray
    native_gene_ids: np.ndarray
    arm_counts: np.ndarray
    arm_logcpm: np.ndarray
    arm_metadata: list
    treated_rows: np.ndarray
    response: np.ndarray
    source_baseline: np.ndarray
    observation_metadata: list
    contrast_means: np.ndarray
    contrast_metadata: list
    eligible_mask: np.ndarray
    global_eligible_mask: np.ndarray
    manifest: dict

    def arm_indices(self, contrast_id, role):
        return np.asarray([i for i, row in enumerate(self.arm_metadata)
                           if (row.get("contrast_id") == contrast_id or
                               contrast_id in row.get("control_for_contrasts", [])) and row["role"] == role], dtype=int)

    def comparison_view(self):
        """One fixed complete gene axis for every source subset and perturbation."""
        return self.feature_ids[self.global_eligible_mask], self.response[:, self.global_eligible_mask]


def assemble_panel(axis, matrices, metadata, manifest):
    """Preserve arm replicates; form source responses against unpaired arm means."""
    native = np.asarray([[row[s] for row in axis] for s in SPECIES])
    features = np.asarray([r["human"] for r in axis])
    arm_counts, arm_log, records, normalization = [], [], [], {}
    for species in SPECIES:
        genes, samples, counts = matrices[species]
        if set(samples) != {row["library"] for row in metadata if row["species"] == species}:
            raise ValueError("Metadata libraries must exactly match matrix headers")
        log, _, quality = normalize_counts(counts)
        normalization[species] = quality
        index = {g: i for i, g in enumerate(genes)}
        wanted = [r[species] for r in axis]
        selected = np.asarray([index.get(g, -1) for g in wanted])
        present = selected >= 0
        aligned_counts = np.full((len(samples), len(axis)), np.nan)
        aligned_log = aligned_counts.copy()
        aligned_counts[:, present] = counts[selected[present]].T
        aligned_log[:, present] = log[selected[present]].T
        by_library = {r["library"]: r for r in metadata if r["species"] == species}
        for j, sample in enumerate(samples):
            row = dict(by_library[sample]); row["library_size_fixed_native_support"] = quality["libraries"][j]
            row["sample_gene_missing_count_on_axis"] = int(np.isnan(aligned_counts[j]).sum())
            records.append(row)
        arm_counts.append(aligned_counts); arm_log.append(aligned_log)
    raw, log = np.concatenate(arm_counts), np.concatenate(arm_log)
    contrasts = sorted({r["contrast_id"] for r in records})
    treated_rows, responses, baselines, obs, means, cm, eligibility = [], [], [], [], [], [], []
    for contrast in contrasts:
        t = [i for i, r in enumerate(records) if r["contrast_id"] == contrast and r["role"] == "treated"]
        c = [i for i, r in enumerate(records) if r["contrast_id"] == contrast and r["role"] == "vehicle"]
        if len(t) < 2 or len(c) < 2:
            raise ValueError("Require at least two labelled replicates per independent arm")
        if len({records[i]["time_h"] for i in t + c}) != 1:
            raise ValueError("Vehicle and treatment times disagree")
        baseline = log[c].mean(axis=0)
        contrast_mean = log[t].mean(axis=0) - baseline
        source_row = dict(records[t[0]])
        source_row.update(treated_arm_rows=t, control_arm_rows=c, treated_replicates=len(t), control_replicates=len(c),
                          estimand="Mean treated log2CPM minus mean vehicle log2CPM; unpaired arms")
        cm.append(source_row); means.append(contrast_mean)
        for i in t:
            row = dict(records[i]); row.update(control_arm_rows=c, control_reference_type="unpaired_vehicle_arm_mean",
                response_dependency_group=contrast, donor_pairing_verified=False)
            treated_rows.append(i); responses.append(log[i] - baseline); baselines.append(baseline); obs.append(row)
    for row in cm:
        # The same mask for every candidate source in this perturbation/context.
        relevant = [i for i, r in enumerate(records) if r["study"] == row["study"]
                    and r["perturbation"] == row["perturbation"] and r["time_h"] == row["time_h"]]
        eligibility.append(np.isfinite(log[relevant]).all(axis=0))
    eligible = np.asarray(eligibility)
    manifest = {**manifest, "normalization": normalization,
        "axis_rule": "Human anchored; all source subsets use the frozen global complete mask. Per-contrast masks also require all five sources and both arms.",
        "donor_rule": "Replicate labels do not prove donor identity; no pairing across arms, compounds, or species.",
        "independence_warning": "Responses share a vehicle mean. Preserve original arm samples; uncertainty resamples the two arms separately and does not treat response rows as independent donor pairs."}
    return CuratedPanel(features, np.asarray(SPECIES), native, raw, log, records, np.asarray(treated_rows),
        np.asarray(responses), np.asarray(baselines), obs, np.asarray(means), cm, eligible,
        np.isfinite(log).all(axis=0), manifest)


def assemble_matched_panel(axis, matrices, annotations, manifest):
    """Known own-control samples retain shared-control links and source-line labels."""
    species_order = list(matrices)
    features = np.asarray([r["human"] for r in axis])
    raw_blocks, log_blocks, records, normalization = [], [], [], {}
    for species in species_order:
        genes, samples, counts = matrices[species]
        by_library = {r["library"]: dict(r) for r in annotations if r["species"] == species}
        if set(samples) != set(by_library):
            raise ValueError("Matched annotations disagree with count sample axis")
        log, _, quality = normalize_counts(counts)
        normalization[species] = quality
        index = {g: i for i, g in enumerate(genes)}
        wanted = np.asarray([index.get(r[species], -1) for r in axis])
        present = wanted >= 0
        aligned = np.full((len(samples), len(axis)), np.nan)
        aligned_log = aligned.copy()
        aligned[:, present] = counts[wanted[present]].T
        aligned_log[:, present] = log[wanted[present]].T
        raw_blocks.append(aligned); log_blocks.append(aligned_log)
        for j, sample in enumerate(samples):
            row = by_library[sample]
            row["library_size_fixed_native_support"] = quality["libraries"][j]
            records.append(row)
    raw, log = np.concatenate(raw_blocks), np.concatenate(log_blocks)
    lookup = {(r["species"], r["library"]): i for i, r in enumerate(records)}
    responses, baselines, obs, treated_rows = [], [], [], []
    for i, row in enumerate(records):
        if row["role"] != "treated":
            continue
        refs = row["control_libraries"]
        controls = [lookup[row["species"], ref] for ref in refs]
        if len(controls) != 1 or records[controls[0]]["role"] != "vehicle":
            raise ValueError("Require one explicit own-control sample")
        if any(records[j]["biological_id"] != row["biological_id"] for j in controls):
            raise ValueError("Own-control biological/source-line labels disagree")
        baseline = log[controls].mean(0)
        baselines.append(baseline); responses.append(log[i] - baseline); treated_rows.append(i)
        item = dict(row); item.update(control_arm_rows=controls, control_reference_type="author_labelled_own_control",
            response_dependency_group=row["study"] + "|" + row["species"] + "|" + row["biological_id"],
            donor_pairing_verified=True)
        obs.append(item)
        records[controls[0]].setdefault("control_for_contrasts", []).append(row["contrast_id"])
    cm, means, eligible = [], [], []
    for contrast in sorted({r["contrast_id"] for r in obs}):
        observations = [i for i, r in enumerate(obs) if r["contrast_id"] == contrast]
        t = [treated_rows[i] for i in observations]
        c = sorted({j for i in observations for j in obs[i]["control_arm_rows"]})
        row = dict(obs[observations[0]])
        row.update(treated_arm_rows=t, control_arm_rows=c, treated_replicates=len(t), control_replicates=len(c),
            estimand="Mean of author-labelled within-source-line treatment-minus-own-control log2CPM responses")
        cm.append(row); means.append(np.asarray(responses)[observations].mean(0))
        same_pert = [i for i, r in enumerate(obs) if r["perturbation"] == row["perturbation"]]
        arm_rows = sorted({treated_rows[i] for i in same_pert} | {j for i in same_pert for j in obs[i]["control_arm_rows"]})
        eligible.append(np.isfinite(log[arm_rows]).all(0))
    admitted = [i for i, r in enumerate(records) if r["role"] != "excluded"]
    return CuratedPanel(features, np.asarray(species_order), np.asarray([[r[s] for r in axis] for s in species_order]),
        raw, log, records, np.asarray(treated_rows), np.asarray(responses), np.asarray(baselines), obs,
        np.asarray(means), cm, np.asarray(eligible), np.isfinite(log[admitted]).all(0),
        {**manifest, "normalization": normalization,
         "independence_status": "Author source-line labels preserved; controls shared across perturbations; animal litter/batch independence unverified.",
         "independence_warning": "Resample source-line IDs jointly across their treatments and shared baseline; never independently resample paired response components."})


def save_curated_panel(panel, directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Curated panel output must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(panel, name) for name in ["feature_ids", "species_order", "native_gene_ids", "arm_counts",
        "arm_logcpm", "treated_rows", "response", "source_baseline", "contrast_means", "eligible_mask", "global_eligible_mask"]}
    np.savez_compressed(directory / "panel.npz", **arrays)
    for name in ["arm_metadata", "observation_metadata", "contrast_metadata"]:
        write_json(directory / (name + ".json"), getattr(panel, name))
        rows = getattr(panel, name)
        fields = sorted(set().union(*(set(row) for row in rows)))
        with (directory / (name + ".tsv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t"); writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    coverage = {"arm_samples": len(panel.arm_metadata), "treated_observations": len(panel.observation_metadata),
        "source_perturbation_context_contrasts": len(panel.contrast_metadata), "species": panel.species_order.tolist(),
        "independent_perturbations": sorted({r["perturbation"] for r in panel.contrast_metadata}),
        "human_anchored_genes_before_missingness": len(panel.feature_ids),
        "fixed_complete_comparison_genes": int(panel.global_eligible_mask.sum()),
        "verified_donors": 0,
        "labelled_biological_source_units": len({(r["species"], r.get("biological_id")) for r in panel.observation_metadata if r.get("biological_id")}),
        "biological_independence_status": panel.manifest.get("independence_status", "Unverified; labelled arm replicates retained"),
        "nonempty_source_ids": sorted({r["source_id"] for r in panel.contrast_metadata if "source_id" in r}),
        "perturbation_source_coverage": {p: sorted({r.get("source_id", r["species"]) for r in panel.contrast_metadata if r["perturbation"] == p})
            for p in sorted({r["perturbation"] for r in panel.contrast_metadata})},
        "per_contrast": [{"contrast_id": r["contrast_id"], "treated_replicates": r["treated_replicates"],
            "control_replicates": r["control_replicates"], "complete_fiveway_genes": int(panel.eligible_mask[i].sum())}
            for i, r in enumerate(panel.contrast_metadata)]}
    write_json(directory / "coverage.json", coverage)
    files = {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}
    write_json(directory / "manifest.json", {**panel.manifest, "files": files, "coverage": coverage})


def load_curated_panel(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("Curated panel hash mismatch: " + name)
    with np.load(directory / "panel.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    records = {name: json.loads((directory / (name + ".json")).read_text())
               for name in ["arm_metadata", "observation_metadata", "contrast_metadata"]}
    return CuratedPanel(**arrays, **records, manifest=manifest)
