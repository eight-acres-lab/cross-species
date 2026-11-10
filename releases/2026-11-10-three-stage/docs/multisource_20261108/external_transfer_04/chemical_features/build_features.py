"""Freeze three externally named chemical features from cached primary PubChem metadata.

Reuses the TG-GATEs fingerprint decoder and descriptor definition. No expression
values or target-response outcomes enter identity matching or feature construction.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_feedback_compound_features import DESCRIPTORS, fingerprint, normalized

OUT = Path(__file__).resolve().parent
SPECS = [
    {"panel_perturbation": "CDDO", "compound": "bardoxolone methyl", "cid": 400769,
     "study_name_evidence": "bardoxolone (CDDO-Me)", "extra_required_exact_synonym": "CDDO-Me",
     "excluded_related_cid": 400010, "exclusion_reason": "Bardoxolone/CDDO free acid is a distinct structure; study explicitly specifies methyl ester CDDO-Me."},
    {"panel_perturbation": "KI696", "compound": "KI696", "cid": 118170767,
     "study_name_evidence": "Ki696", "extra_required_exact_synonym": "KI696",
     "excluded_related_cid": 118169940, "exclusion_reason": "KI696 isomer is a distinct stereoisomer name/structure, not the exact KI696 name returned by PubChem."},
    {"panel_perturbation": "THAP", "compound": "thapsigargin", "cid": 446378,
     "study_name_evidence": "thapsigargin", "extra_required_exact_synonym": "thapsigargin"},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    props = {int(row["CID"]): row for row in json.loads((OUT / "candidate_properties.json").read_text())["PropertyTable"]["Properties"]}
    syns = {int(row["CID"]): row["Synonym"] for row in json.loads((OUT / "candidate_synonyms.json").read_text())["InformationList"]["Information"]}
    resolution = {row["name"]: row for row in json.loads((OUT / "name_resolution.json").read_text())}
    arms_path = ROOT / "outputs/multisource_20261108/curated_panels/GSE330151/arm_metadata.json"
    arms = json.loads(arms_path.read_text())
    feature_rows, numerical, identities, evidence = [], [], [], []
    for spec in SPECS:
        cid = spec["cid"]
        if resolution[spec["compound"]]["data"]["IdentifierList"]["CID"] != [cid]:
            raise ValueError("Name lookup is no longer uniquely tied to the frozen CID")
        exact = [name for name in syns[cid] if normalized(name) == normalized(spec["compound"])]
        extra = [name for name in syns[cid] if normalized(name) == normalized(spec["extra_required_exact_synonym"])]
        if not exact or not extra:
            raise ValueError("Missing exact verified synonym")
        rows = [row for row in arms if row["species"] in {"human", "rat"}
                and row["perturbation"] == spec["panel_perturbation"] and row["role"] == "treated"]
        if len(rows) != 6 or not all(spec["study_name_evidence"].casefold() in row["treatment_text"].casefold() for row in rows):
            raise ValueError("Curated official sample evidence does not match the selected identity")
        p = props[cid]
        fp = fingerprint(p["Fingerprint2D"])
        values = np.asarray([float(p[name]) if name in p else float("nan") for name in DESCRIPTORS])
        if len(fp) != 881 or not np.isin(fp, [0, 1]).all() or not fp.any() or not np.isfinite(values).all():
            raise ValueError("Incomplete selected chemical features")
        feature_rows.append(fp)
        numerical.append(values)
        identities.append({**spec, "pubchem_title": p["Title"], "molecular_formula": p["MolecularFormula"],
                           "inchikey": p["InChIKey"], "isomeric_smiles": p["SMILES"],
                           "connectivity_smiles": p["ConnectivitySMILES"], "matched_synonym": exact[0],
                           "matched_explicit_study_form": extra[0], "fingerprint_set_bits": int(fp.sum()),
                           "descriptors": dict(zip(DESCRIPTORS, values.tolist())),
                           "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/" + str(cid),
                           "experimental_lot_purity_stereochemistry_verified": False})
        evidence += [{"panel_perturbation": spec["panel_perturbation"], "sample_id": row["sample_id"],
                      "species": row["species"], "treatment_text": row["treatment_text"],
                      "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + row["sample_id"]} for row in rows]
    np.savez_compressed(OUT / "features.npz", compounds=np.asarray([spec["compound"] for spec in SPECS]),
                        fingerprints=np.stack(feature_rows), descriptors=np.asarray(numerical),
                        descriptor_names=np.asarray(DESCRIPTORS), verified_mask=np.ones(3, dtype=bool))
    # Exact representation and saved-array roundtrip checks.
    with np.load(OUT / "features.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"compounds", "fingerprints", "descriptors", "descriptor_names", "verified_mask"}:
            raise ValueError("TG-GATEs chemical feature schema mismatch")
        np.testing.assert_array_equal(saved["fingerprints"], np.stack(feature_rows))
        np.testing.assert_array_equal(saved["descriptors"], numerical)
    training_report = ROOT / "outputs/feedback_nested_chemical_20261107_absolute_v1/report.json"
    strict_names = json.loads(training_report.read_text())["molecular_features"]["actual_model_compounds_after_exposure_filter"]
    training_features_manifest = ROOT / "artifacts/feedback_compound_features_20261107/manifest.json"
    original = json.loads(training_features_manifest.read_text())
    accepted_by_name = {normalized(row["compound"]): row for row in original["accepted"]}
    strict_records = [accepted_by_name[normalized(name)] for name in strict_names]
    strict_cids, strict_keys = {row["cid"] for row in strict_records}, {row["inchikey"] for row in strict_records}
    for item in identities:
        item["same_cid_in_strict_training"] = item["cid"] in strict_cids
        item["same_inchikey_in_strict_training"] = item["inchikey"] in strict_keys
        if item["same_cid_in_strict_training"] or item["same_inchikey_in_strict_training"]:
            raise ValueError("External identity overlaps strict training chemical identity")
    facts = {"date": "2026-11-08", "study": "GSE330151", "source": "PubChem PUG REST with exact name/synonym matching to curated official GEO text",
             "feature_schema_matches_TG_GATES": True, "fingerprint": "PubChem881-bit, exact existing length-prefixed base64 decoder",
             "descriptor_order": DESCRIPTORS, "accepted": identities, "all_three_absent_from_strict_training_CIDs_and_InChIKeys": True,
             "preprocessing": "Raw descriptors kept; imputation/standardization must use TG-GATEs training statistics only. External descriptors do not refit any scaler.",
             "representation_limit": "PubChem881 and these physicochemical descriptors may not encode stereoisomer differences; identity is separately verified by names/InChIKey and study wording.",
             "input_files": {"candidate_properties.json": sha(OUT / "candidate_properties.json"),
                             "candidate_synonyms.json": sha(OUT / "candidate_synonyms.json"),
                             "name_resolution.json": sha(OUT / "name_resolution.json"),
                             "GSE330151/arm_metadata.json": sha(arms_path),
                             "TG_strict_report": sha(training_report), "TG_features_manifest": sha(training_features_manifest)},
             "feature_sha256": sha(OUT / "features.npz"), "build_script_sha256": sha(__file__),
             "existing_decoder_script_sha256": sha(ROOT / "scripts/fetch_feedback_compound_features.py"),
             "sample_evidence": evidence}
    (OUT / "manifest.json").write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n")
    (OUT / "feature.json").write_text(json.dumps({"compounds": [spec["compound"] for spec in SPECS], "fingerprints": np.stack(feature_rows).tolist(),
                                                 "descriptors": np.asarray(numerical).tolist(), "descriptor_names": DESCRIPTORS,
                                                 "verified_mask": [True] * 3}, indent=2) + "\n")
    print(json.dumps({"accepted": [(row["compound"], row["cid"]) for row in identities], "fingerprint_shape": [3, 881],
                      "descriptor_shape": [3, 10], "overlap_with_strict_training": False}, indent=2))


if __name__ == "__main__":
    main()
