"""Collect Rhaister's per-split metrics into one artifact.

Rhaister prints its metrics to stdout as `RESULT key=value` lines and writes no
machine-readable output of its own, so this artifact is a transcription of those lines
rather than a recomputation. That is the one place in this repository where a number
comes from a run log, and it is why this script exists: the transcription has to be
repeatable and checkable against the logs, which are kept beside the artifact.

The released logs retain the RESULT lines only. Re-running this script on the
released results directory reproduces the deposited JSON byte for byte.

Usage:
    python3 collect_rhaister_metrics.py <results-dir> <out.json>
"""
import json
import sys
from pathlib import Path

SPLITS = ['split_0',
          'cluster_K6_c0', 'cluster_K6_c1',
          'cluster_K10_c0', 'cluster_K10_c1',
          'randpanel_0', 'randpanel_1', 'randpanel_2', 'randpanel_3']


def parse(path):
    """Pull the RESULT lines out of one run's stdout."""
    out = {}
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith('RESULT '):
            continue
        key, _, value = line[len('RESULT '):].partition('=')
        out[key.strip()] = float(value)
    return out


def main(results_dir, out_json):
    results, missing = {}, []
    for split in SPLITS:
        path = Path(results_dir) / f'{split}.out'
        if not path.exists():
            missing.append(split)
            continue
        metrics = parse(path)
        if not metrics:
            missing.append(f'{split} (no RESULT lines)')
            continue
        results[split] = metrics

    # indent=1 and no trailing newline, matching how every control script in this
    # repository writes its artifact.
    json.dump(results, open(out_json, 'w'), indent=1)
    print(f'{len(results)}/{len(SPLITS)} splits -> {out_json}')
    if missing:
        print('MISSING: ' + ', '.join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
