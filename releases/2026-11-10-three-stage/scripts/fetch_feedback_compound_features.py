#!/usr/bin/env python3
"""Fetch auditable PubChem molecular priors; never use response outcomes.

Existing CID candidates are accepted only after a normalized exact synonym match.
Unresolved names remain excluded rather than receiving zero fingerprints.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np


DESCRIPTORS = ["MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", "Complexity", "Charge", "CovalentUnitCount"]


def normalized(name):
    text = unicodedata.normalize("NFKC", str(name)).casefold().replace("_", " ")
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(encoded):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) < 5:
        raise ValueError("Fingerprint has no length prefix/data")
    length = int.from_bytes(raw[:4], "big")
    if length != 881:
        raise ValueError(f"Unexpected PubChem fingerprint length: {length}")
    bits = np.unpackbits(np.frombuffer(raw[4:], dtype=np.uint8))
    if len(bits) < length:
        raise ValueError("Truncated fingerprint")
    return bits[:length].copy()


class Client:
    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.last = 0.

    def get(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self.path / (key + ".json")
        if cached.exists():
            return json.loads(cached.read_text())
        for attempt in range(3):
            time.sleep(max(0, 1.0 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                with urlopen(Request(url, headers={"User-Agent": "cross-species-molecular-prior/0.1"}), timeout=30) as response:
                    body = response.read(25_000_001)
                if len(body) > 25_000_000:
                    raise ValueError("Response exceeds metadata cap")
                data = json.loads(body)
                cached.write_text(json.dumps(data))
                (self.path / (key + ".source.json")).write_text(json.dumps({"url": url, "sha256": hashlib.sha256(body).hexdigest(), "retrieved_unix": time.time()}, indent=2))
                return data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cid-seeds", type=Path, default=Path("artifacts/biology/dili/compound_druglikeness.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/feedback_compound_features_20261107"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(args.out_dir / "http_cache")
    seeds = json.loads(args.cid_seeds.read_text())["per_compound"]
    cids = sorted({int(item["cid"]) for item in seeds.values() if item.get("cid")})
    properties, synonyms, errors = {}, {}, []
    prefix = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
    for start in range(0, len(cids), 25):
        chunk = cids[start:start + 25]
        joined = ",".join(map(str, chunk))
        for kind in ["properties", "synonyms"]:
            url = prefix + joined + ("/property/" + ",".join(["Title", "InChIKey", "Fingerprint2D"] + DESCRIPTORS) + "/JSON" if kind == "properties" else "/synonyms/JSON")
            try:
                data = client.get(url)
                if kind == "properties":
                    properties.update({int(row["CID"]): row for row in data["PropertyTable"]["Properties"]})
                else:
                    synonyms.update({int(row["CID"]): row.get("Synonym", []) for row in data["InformationList"]["Information"]})
            except Exception as exc:
                errors.append({"kind": kind, "cids": chunk, "error": str(exc)})
        print(f"PubChem checked {min(start + 25, len(cids))}/{len(cids)} CID candidates", flush=True)
    accepted, rejected = [], []
    vectors, numeric = [], []
    for compound, seed in sorted(seeds.items()):
        cid = int(seed["cid"]) if seed.get("cid") else None
        record = properties.get(cid)
        exact = [s for s in synonyms.get(cid, []) if normalized(s) == normalized(compound)]
        reason = None
        if normalized(compound) in {"tnfalpha", "tnf alpha", "tnfa", "lps"}:
            reason = "Biological reagent is not represented by a unique verified small-molecule structure; a name-matched PubChem record is insufficient"
        elif record is None:
            reason = "CID properties unavailable"
        elif not exact:
            reason = "No normalized exact synonym match; candidate identity not promoted"
        elif not record.get("Fingerprint2D"):
            reason = "Fingerprint unavailable"
        if reason:
            rejected.append({"compound": compound, "cid_candidate": cid, "reason": reason})
            continue
        try:
            bits = fingerprint(record["Fingerprint2D"])
            values = [float(record[k]) if k in record else float("nan") for k in DESCRIPTORS]
            vectors.append(bits)
            numeric.append(values)
            accepted.append({"compound": compound, "cid": cid, "title": record.get("Title"), "inchikey": record.get("InChIKey"), "matched_synonym": exact[0], "missing_descriptors": [name for name, value in zip(DESCRIPTORS, values) if not np.isfinite(value)]})
        except Exception as exc:
            rejected.append({"compound": compound, "cid_candidate": cid, "reason": str(exc)})
    if not accepted:
        raise RuntimeError("No verified molecular features; inspect API errors")
    np.savez_compressed(args.out_dir / "features.npz", compounds=np.array([r["compound"] for r in accepted]),
                        fingerprints=np.stack(vectors), descriptors=np.asarray(numeric), descriptor_names=np.array(DESCRIPTORS), verified_mask=np.ones(len(accepted), dtype=bool))
    manifest = {"source": "PubChem PUG REST", "date": "2026-11-07", "seed_file": str(args.cid_seeds), "seed_sha256": hashlib.sha256(args.cid_seeds.read_bytes()).hexdigest(),
                "verification": "Normalized exact matching of study compound name to CID synonyms; this validates the database name mapping, not the experimental reagent's stereochemistry/salt or purity.",
                "fingerprint": "PubChem 881-bit substructure fingerprint, decoded from the 4-byte-length-prefixed base64 representation.",
                "accepted": accepted, "excluded": rejected, "api_errors": errors,
                "descriptor_handling": "Missing numerical descriptors remain NaN; any imputation must fit on training compounds only. Fingerprints are never imputed for missing identities.",
                "features_sha256": hashlib.sha256((args.out_dir / "features.npz").read_bytes()).hexdigest()}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"accepted_compounds": len(accepted), "excluded_compounds": len(rejected), "api_errors": len(errors), "feature_path": str(args.out_dir / "features.npz")}), flush=True)


if __name__ == "__main__":
    main()
