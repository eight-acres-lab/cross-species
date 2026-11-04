#!/usr/bin/env python3
"""Does the donor ceiling move with how many cells were sequenced behind each measurement?

This paper builds its scale on donor splits and says, in Section 2.2, that a split by cell
climbs towards one as depth rises where a split by donor does not. That is the distinction the
whole construction rests on, and in the manuscript it is an argument rather than a measurement.

The Parse panel makes it testable. Each (cell type, donor, perturbation) measurement carries
the number of cells behind it, and those counts vary several-fold across the panel. If the
leave-one-donor-out agreement rises with cell count, then part of what is being called a
ceiling is sequencing depth and a deeper experiment would raise it. If it does not, the
ceiling is donor-to-donor biological variation and is not bought down by depth.

Reported both ways; the correlation is whatever it is.

Run on the compute host:
    python3 run_rhaister_depth_vs_ceiling.py <pdex.parquet> <donor_ceiling.json> <out.json>
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

import argparse
from pathlib import Path
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('pdex', help='Parse pdex/all_celltypes_pdex.parquet')
ap.add_argument('ceiling', nargs='?', default='artifacts/rhaister_benchmark/donor_ceiling.json')
ap.add_argument('out', nargs='?', default='artifacts/rhaister_benchmark/depth_vs_ceiling.json')
args = ap.parse_args()
SRC, CEIL, OUT = args.pdex, args.ceiling, args.out
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
COLS = ['cell_type', 'donor', 'target', 'target_membership']

f = pq.ParquetFile(SRC)
n_groups = f.metadata.num_row_groups
print(f'{f.metadata.num_rows} rows in {n_groups} row groups; membership is constant within a '
      f'(cell type, donor, perturbation) group, so one row per group is enough', flush=True)

seen = {}
for g in range(n_groups):
    t = f.read_row_group(g, columns=COLS).to_pandas()
    for ct, dn, tg, mb in t.drop_duplicates(['cell_type', 'donor', 'target']).itertuples(
            index=False, name=None):
        seen[(ct, tg, dn)] = int(mb)
    if (g + 1) % 500 == 0:
        print(f'  {g + 1}/{n_groups} groups, {len(seen)} units', flush=True)
print(f'{len(seen)} (cell type, perturbation, donor) units', flush=True)

# median cells behind a (cell type, perturbation) cell, which is the unit the ceiling is on
cells = {}
for (ct, tg, dn), mb in seen.items():
    cells.setdefault((ct, tg), []).append(mb)

ceil = json.load(open(CEIL))['by_cell_type']
rows = []
for ct, v in ceil.items():
    for tg, pc in v['per_cytokine'].items():
        key = (ct, tg)
        if key in cells:
            rows.append(dict(cell_type=ct, target=tg, ceiling=pc['mean'],
                             median_cells=float(np.median(cells[key])),
                             min_cells=int(min(cells[key]))))
print(f'{len(rows)} units joined to a ceiling', flush=True)

c = np.array([r['ceiling'] for r in rows])
m = np.array([r['median_cells'] for r in rows])
overall = float(spearmanr(m, c).statistic)

# Within a cell type the biology is held roughly fixed, so this is the cleaner test: across
# perturbations of one cell type, does the one measured on more cells agree better?
within = {}
for ct in sorted({r['cell_type'] for r in rows}):
    sel = [r for r in rows if r['cell_type'] == ct]
    if len(sel) < 10:
        continue
    within[ct] = float(spearmanr([r['median_cells'] for r in sel],
                                 [r['ceiling'] for r in sel]).statistic)

res = dict(n_units=len(rows),
           cells_range=[float(m.min()), float(m.max())],
           ceiling_range=[float(c.min()), float(c.max())],
           spearman_cells_vs_ceiling_overall=overall,
           spearman_within_cell_type=within,
           within_median=float(np.median(list(within.values()))) if within else None,
           by_unit=rows)
json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT, flush=True)
print(f'cells per unit span {m.min():.0f} to {m.max():.0f}', flush=True)
print(f'Spearman(cells, ceiling) pooled     = {overall:+.3f}', flush=True)
if within:
    print(f'median within cell type             = {np.median(list(within.values())):+.3f}',
          flush=True)
    print('  ' + '  '.join(f'{k.split("_")[0]}={v:+.2f}' for k, v in sorted(within.items())),
          flush=True)
