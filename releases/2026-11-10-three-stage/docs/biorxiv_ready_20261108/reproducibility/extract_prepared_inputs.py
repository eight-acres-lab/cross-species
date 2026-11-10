#!/usr/bin/env python3
"""Verify and extract the optional private prepared-input archive into a fresh root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).with_name("prepared_inputs_manifest.json"))
    ap.add_argument("--work-root", type=Path, required=True)
    args = ap.parse_args(); manifest = json.loads(args.manifest.read_text())
    if sha(args.archive) != manifest["archive_sha256"]:
        raise ValueError("Prepared archive checksum mismatch")
    root = args.work_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Extract into a new empty work root")
    root.mkdir(parents=True, exist_ok=True)
    expected = {row["path"]: row for row in manifest["files"]}
    with tarfile.open(args.archive, "r:gz") as tar:
        if set(tar.getnames()) != set(expected):
            raise ValueError("Archive members differ from manifest")
        for member in tar:
            destination = root / member.name
            if not member.isfile() or root not in destination.resolve().parents:
                raise ValueError("Unsafe archive member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, destination.open("xb") as output:
                shutil.copyfileobj(stream, output)
            if sha(destination) != expected[member.name]["sha256"]:
                raise ValueError("Extracted member checksum mismatch")
    print(json.dumps({"files_extracted_and_verified": len(expected), "work_root": str(root)}))


if __name__ == "__main__":
    main()
