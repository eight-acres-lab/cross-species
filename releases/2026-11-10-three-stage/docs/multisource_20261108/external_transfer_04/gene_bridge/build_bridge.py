"""Build a strict frozen Entrez/Ensembl bridge without response-value access.

Only mapping tables, matrix header identifiers, and curated-panel ID arrays are
inspected. Existing expression matrices and value-based masks are never loaded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/tg_gates_rat_human_v1/ortholog_map"
PANELS = ROOT / "outputs/multisource_20261108/curated_panels"
NEW = PANELS / "orthology_release_freeze"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for b in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def identity_maps(path):
    forward, reverse = defaultdict(set), defaultdict(set)
    data = rows(path)
    for row in data:
        if len(row) != 2:
            raise ValueError("Unexpected ID-map row width")
        ensembl, entrez = row
        if not ensembl or not entrez:
            continue
        if not entrez.isdigit():
            raise ValueError("Non-integer Entrez ID")
        forward[ensembl].add(entrez)
        reverse[entrez].add(ensembl)
    return forward, reverse, {
        "raw_rows": len(data), "ensembl_ids_with_entrez": len(forward),
        "entrez_ids_with_ensembl": len(reverse),
        "ensembl_ids_with_multiple_entrez": sum(len(v) > 1 for v in forward.values()),
        "entrez_ids_with_multiple_ensembl": sum(len(v) > 1 for v in reverse.values()),
    }


def high_conf_pairs(path, rat_first=False):
    forward, reverse = defaultdict(set), defaultdict(set)
    raw_rows = 0
    for row in rows(path):
        raw_rows += 1
        if len(row) != 4:
            raise ValueError("Unexpected ortholog-map row width")
        a, b, relation, confidence = row
        human, rat = (b, a) if rat_first else (a, b)
        if relation == "ortholog_one2one" and confidence == "1" and human and rat:
            forward[human].add(rat)
            reverse[rat].add(human)
    good = {(human, next(iter(rats))) for human, rats in forward.items()
            if len(rats) == 1 and len(reverse[next(iter(rats))]) == 1}
    return good, {"raw_rows": raw_rows, "high_confidence_human_ids": len(forward),
                  "human_partner_conflicts": sum(len(v) > 1 for v in forward.values()),
                  "rat_partner_conflicts": sum(len(v) > 1 for v in reverse.values()),
                  "accepted_unique_pairs": len(good)}


def panel_metadata(study):
    folder = PANELS / study
    manifest = json.loads((folder / "manifest.json").read_text())
    for filename in ["panel.npz", "contrast_metadata.json"]:
        expected = manifest["files"][filename]
        if sha(folder / filename) != expected:
            raise ValueError("Frozen panel file changed: " + study + "/" + filename)
    # NPZ members are individually loaded; no count/expression/response member is read.
    with np.load(folder / "panel.npz", allow_pickle=False) as saved:
        features = saved["feature_ids"].copy()
        species = saved["species_order"].tolist()
        native = saved["native_gene_ids"].copy()
    metadata = json.loads((folder / "contrast_metadata.json").read_text())
    human_native, rat_native = native[species.index("human")], native[species.index("rat")]
    if not np.array_equal(features, human_native):
        raise ValueError("Curated human axis differs from its native identifiers")
    if len(features) != len(set(features)) or len(rat_native) != len(set(rat_native)):
        raise ValueError("Curated human/rat axis not unique")
    selected = [i for i, row in enumerate(metadata) if row["species"] in {"human", "rat"}
                and row.get("contrast_kind", "drug") == "drug"]
    if len(selected) != 6 or {metadata[i]["species"] for i in selected} != {"human", "rat"}:
        raise ValueError("Expected both species and three drug comparisons")
    return {str(h): {"index": i, "rat_ensembl": str(rat_native[i])}
            for i, h in enumerate(features)}, {
        "study": study, "manifest_sha256": sha(folder / "manifest.json"),
        "panel_file_sha256": manifest["files"]["panel.npz"],
        "contrast_metadata_sha256": manifest["files"]["contrast_metadata.json"],
        "axis_genes": len(features), "human_rat_drug_complete_genes": None,
        "selected_contrasts": [metadata[i]["contrast_id"] for i in selected],
        "npz_members_read": ["feature_ids", "species_order", "native_gene_ids"],
        "expression_count_response_values_read": False,
        "presence_rule": "Exact human/rat identifiers present on the panel axis. Value-finiteness remains a separate remote check across selected human/rat arms. Existing masks combine all species and are not used.",
    }


def write_tsv(path, data, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((OLD / "provenance.json").read_text())
    new_manifest = json.loads((NEW / "manifest.json").read_text())
    provenance = {}
    for name, entry in old_manifest["raw_files"].items():
        path = ROOT / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Old frozen raw mapping hash mismatch: " + name)
        provenance[name] = {"path": str(path), "sha256": entry["sha256"], "bytes": path.stat().st_size}
    if sha(OLD / "rat_to_human_one2one.tsv") != old_manifest["output_sha256"]:
        raise ValueError("Old fixed training ortholog axis changed")
    for name in ["rat.tsv", "rat.query.xml", "registry.xml", "registry_after.xml"]:
        if sha(NEW / name) != new_manifest["files"][name]:
            raise ValueError("New frozen mapping hash mismatch: " + name)
        provenance["release116_" + name] = {"path": str(NEW / name), "sha256": new_manifest["files"][name]}
    h_forward, h_reverse, h_stats = identity_maps(OLD / "biomart_raw/human_ensembl_to_entrez.tsv")
    r_forward, r_reverse, r_stats = identity_maps(OLD / "biomart_raw/rat_ensembl_to_entrez.tsv")
    old_pairs, old_stats = high_conf_pairs(OLD / "biomart_raw/rat_to_human_homology.tsv", rat_first=True)
    new_pairs, new_stats = high_conf_pairs(NEW / "rat.tsv")
    with (OLD / "rat_to_human_one2one.tsv").open() as handle:
        axis = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda row: int(row["human_entrez"]))
    if len(axis) != 10358:
        raise ValueError("Training axis size changed")
    header_positions = {}
    for species, filename in [("human", "pairs_human_Y.tsv"), ("rat", "pairs_rat_X.tsv")]:
        path = OLD.parent / "data" / filename
        with path.open() as handle:
            ids = handle.readline().rstrip("\n\r").split("\t")[1:]
        header_positions[species] = {identifier: index for index, identifier in enumerate(ids)}
        if len(ids) != len(header_positions[species]):
            raise ValueError("Repeated native matrix header gene")
    panel_index, panel_audit = {}, {}
    for study in ["GSE330151", "GSE300171"]:
        panel_index[study], panel_audit[study] = panel_metadata(study)
    decisions, bridge = [], []
    for index, entry in enumerate(axis):
        h_ent, r_ent = entry["human_entrez"], entry["rat_entrez"]
        if h_ent not in header_positions["human"] or r_ent not in header_positions["rat"]:
            raise ValueError("Original paired Entrez axis no longer matches matrix header IDs")
        item = {"tg_axis_index_zero_based": index, "human_entrez": h_ent, "rat_entrez": r_ent,
                "human_ensembl": "", "rat_ensembl": "", "status": "admitted", "exclusion_reason": ""}
        failures = []
        for label, ent, forward, reverse in [("human", h_ent, h_forward, h_reverse), ("rat", r_ent, r_forward, r_reverse)]:
            candidates = reverse.get(ent, set())
            if not candidates:
                failures.append(label + "_entrez_has_no_ensembl")
            elif len(candidates) != 1:
                failures.append(label + "_entrez_maps_multiple_ensembl")
            else:
                ens = next(iter(candidates))
                item[label + "_ensembl"] = ens
                if forward[ens] != {ent}:
                    failures.append(label + "_ensembl_maps_multiple_entrez")
        pair = item["human_ensembl"], item["rat_ensembl"]
        if not failures:
            if pair not in old_pairs:
                failures.append("pair_missing_or_conflicting_in_original_high_conf_orthology")
            if pair not in new_pairs:
                failures.append("pair_missing_or_conflicting_in_release116_high_conf_orthology")
        item["status"] = "excluded" if failures else "admitted"
        item["exclusion_reason"] = ";".join(failures)
        for study in panel_index:
            panel_row = panel_index[study].get(pair[0])
            present = bool(panel_row and panel_row["rat_ensembl"] == pair[1])
            item[study + "_panel_index_zero_based"] = panel_row["index"] if present else ""
            item[study + "_id_pair_present"] = bool(present and not failures)
        decisions.append(item)
        if not failures:
            bridge.append(item.copy())
    # All four identities must be globally unique in the accepted bridge.
    for key in ["human_entrez", "rat_entrez", "human_ensembl", "rat_ensembl"]:
        if len(bridge) != len({r[key] for r in bridge}):
            raise ValueError("Bridge is not one-to-one: " + key)
    fields = list(decisions[0])
    write_tsv(OUT / "bridge.tsv", bridge, fields)
    write_tsv(OUT / "training_axis_decisions.tsv", decisions, fields)
    panel_outputs = {}
    for study in panel_index:
        admitted = [r for r in bridge if r[study + "_id_pair_present"]]
        name = study + "_bridge.tsv"
        write_tsv(OUT / name, admitted, fields)
        panel_outputs[study] = {"admitted_gene_pairs": len(admitted),
                                "fraction_of_training_axis": len(admitted) / len(axis),
                                "human_rat_arm_finiteness_checked": False,
                                "file": name, "sha256": sha(OUT / name)}
    for name, query in old_manifest["queries"].items():
        (OUT / ("original_" + name + ".query.xml")).write_text(query + "\n")
    audit = {
        "date": "2026-11-08", "status": "metadata_gene_bridge_ready",
        "original_training_axis_pairs": len(axis), "globally_unambiguous_id_pairs": len(bridge),
        "training_axis_excluded_pairs": len(axis) - len(bridge),
        "exclusion_combination_counts": dict(Counter(r["exclusion_reason"] for r in decisions if r["status"] == "excluded")),
        "human_id_mapping_counts": h_stats, "rat_id_mapping_counts": r_stats,
        "original_orthology_counts": old_stats, "release116_orthology_counts": new_stats,
        "original_vs_new_high_confidence_pair_overlap": len(old_pairs & new_pairs),
        "panels": panel_audit, "panel_bridges": panel_outputs,
        "source_files": provenance,
        "training_entrez_axis_sha256": old_manifest["output_sha256"],
        "old_provenance_sha256": sha(OLD / "provenance.json"),
        "release116_manifest_sha256": sha(NEW / "manifest.json"),
        "release_notes": {"ensembl_to_entrez_retrieved_utc": old_manifest["retrieved_utc"],
                          "ensembl_to_entrez_release_recorded": None,
                          "new_orthology_retrieved_utc": new_manifest["retrieved_utc"],
                          "new_orthology_release": new_manifest["biomart_gene_release"],
                          "inference": "The deposited mapping tables define the gene bridge; homology relations are checked independently against the high-confidence orthology table."},
        "selection": "Only exact IDs, bidirectional cardinality, orthology consistency and identifier-axis presence; no global/multi-species masks, gene symbol/alias inference or response-magnitude selection.",
        "expression_values_read": False, "network_fetch_performed": False,
        "limitations": [
            "Exact bridge does not harmonize RMA log2 and RNA-seq transformed response scales or establish cell/exposure comparability.",
            "Mapping excludes all globally ambiguous ID associations even if one candidate would match a panel; coverage is deliberately constrained.",
            "Panel axes were originally curated through multi-species orthology. This bridge cannot recover extra human-rat native genes absent from those panel axes.",
            "Human-rat arm finiteness remains unchecked here. The downstream remote preparation must verify or record missing values on all human/rat arms needed for all selected tasks.",
            "If models are retrained on a shared axis, this is a new fixed-feature version, not application of the original 10358-feature weights.",
        ],
        "files": {name: sha(OUT / name) for name in ["bridge.tsv", "training_axis_decisions.tsv", "GSE330151_bridge.tsv", "GSE300171_bridge.tsv"]},
        "script_sha256": sha(__file__),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ["globally_unambiguous_id_pairs", "training_axis_excluded_pairs", "exclusion_combination_counts", "panel_bridges"]}, indent=2))


if __name__ == "__main__":
    main()
