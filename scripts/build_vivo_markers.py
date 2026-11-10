#!/usr/bin/env python3
"""Every serum marker in the rat in vivo annotation, as a fold change against each
compound-day's own matched control. ALT alone gives 18 positives out of 143, which is too
few to separate an AUC of 0.6 from chance, so the alternatives and a composite are built
here to see whether any endpoint has more power.

Non-finite values are dropped explicitly: two NaNs among 6,765 were enough to make a sorted
array report a median below its own first quartile.
"""
import collections, csv, json, math, os, statistics, sys

import argparse
from pathlib import Path
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('annotations', help='TG-GATEs rat_vivo_repeat_exp.txt')
ap.add_argument('out', nargs='?', default='artifacts/biology/dili/rat_vivo_markers.json')
args = ap.parse_args()
P, OUT = args.annotations, args.out
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
MARK = ['ALT(IU/L)', 'AST(IU/L)', 'TBIL(mg/dL)', 'ALP(IU/L)', 'GTP(IU/L)', 'LIVER(g)']
DAYS = ('4 day', '8 day', '15 day', '29 day')
LEVELS = ('Low', 'Middle', 'High')


def fin(s):
    try:
        x = float(s)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


rows = list(csv.DictReader(open(P, encoding='utf-8', errors='replace'), delimiter='\t'))
out = collections.defaultdict(dict)
for m in MARK:
    cell = collections.defaultdict(list)
    for r in rows:
        v = fin(r.get(m))
        if v is not None:
            cell[(r['COMPOUND_NAME'].strip(), r['SACRI_PERIOD'].strip(),
                  r['DOSE_LEVEL'].strip())].append(v)
    for c in sorted({k[0] for k in cell}):
        best, at = 0.0, None
        for d in DAYS:
            ctl = cell.get((c, d, 'Control'))
            if not ctl:
                continue
            base = statistics.mean(ctl)
            if not (base > 0):
                continue
            for lv in LEVELS:
                tr = cell.get((c, d, lv))
                if tr:
                    f = statistics.mean(tr) / base
                    if f > best:
                        best, at = f, f'{lv}/{d}'
        if at:
            out[c][m] = {'fold_max': round(best, 4), 'at': at}

# The composite takes ALT and AST only. Bilirubin and GGT have more apparent responders
# but their absolute values sit at the resolution of the assay -- bilirubin's first quartile
# is 0.0 mg/dL and GGT's median is 1 IU/L -- so a control mean near zero turns a one-step
# change into a large fold. Choosing the endpoint with the most positives would be choosing
# the one whose positives are mostly rounding.
for c, d in out.items():
    parts = [d[m]['fold_max'] for m in ('ALT(IU/L)', 'AST(IU/L)') if m in d]
    if parts:
        d['COMPOSITE'] = {'fold_max': round(max(parts), 4), 'at': 'max of ALT and AST'}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=1, sort_keys=True)
print(f'compounds: {len(out)}')
for m in MARK + ['COMPOSITE']:
    have = [v[m]['fold_max'] for v in out.values() if m in v]
    if not have:
        continue
    have.sort()
    line = f'{m:14s} n={len(have):3d} median={have[len(have)//2]:6.2f}x  '
    for t in (1.5, 2.0, 3.0):
        line += f'>={t:g}x: {sum(1 for x in have if x >= t):3d}  '
    print(line)
print(f'wrote {OUT}')
