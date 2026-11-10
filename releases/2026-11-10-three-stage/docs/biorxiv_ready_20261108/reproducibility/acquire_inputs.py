#!/usr/bin/env python3
"""Download only the explicitly catalogued public input files and verify hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import time
from urllib.request import Request, urlopen


def sha(path, method="sha256"):
    h = hashlib.new(method)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path(__file__).with_name("data_acquisition.json"))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--group", choices=["tg_pairs", "blood", "g330"], required=True)
    ap.add_argument("--fetch", action="store_true", help="Without this flag, print the immutable download plan only")
    args = ap.parse_args()
    records = [r for r in json.loads(args.catalog.read_text())["files"] if r["group"] == args.group]
    if not args.fetch:
        print(json.dumps(records, indent=2)); return
    work = args.work_root.resolve()
    for row in records:
        destination = work / row["relative_destination"]
        if work not in destination.resolve().parents:
            raise ValueError("Destination escapes work root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha(destination) != row["sha256"]:
                raise ValueError("Existing input differs: " + str(destination))
            print("verified cached", destination, flush=True); continue
        partial = destination.with_name(destination.name + ".part")
        if partial.exists():
            raise ValueError("A partial transfer exists; inspect/remove it before a fresh transfer: " + str(partial))
        started = time.monotonic(); total = 0
        with urlopen(Request(row["url"], headers={"User-Agent": "CrossSpeciesReproduction/1.0"}), timeout=60) as response:
            with partial.open("xb") as stream:
                while True:
                    if time.monotonic() - started > 3600:
                        raise TimeoutError("One-hour per-file transfer limit reached")
                    block = response.read(1048576)
                    if not block:
                        break
                    total += len(block)
                    if total > row["max_bytes"]:
                        raise ValueError("Recorded size cap exceeded")
                    stream.write(block)
        if sha(partial) != row["sha256"]:
            raise ValueError("Downloaded SHA256 differs: " + str(partial))
        if row.get("md5") and sha(partial, "md5") != row["md5"]:
            raise ValueError("Provider MD5 differs")
        partial.rename(destination)
        print("downloaded and verified", destination, total, flush=True)


if __name__ == "__main__":
    main()
