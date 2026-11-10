#!/usr/bin/env python3
"""Verify the SHA-256 inventory of the published files using only the standard library."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    failures = []
    for item in data['files']:
        path = ROOT / item['path']
        if not path.is_file():
            failures.append(f'Missing: {item["path"]}')
            continue
        contents = path.read_bytes()
        if len(contents) != item['bytes'] or hashlib.sha256(contents).hexdigest() != item['sha256']:
            failures.append(f'Changed: {item["path"]}')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'PASS: {len(data["files"])} released files match their SHA-256 checksums.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
