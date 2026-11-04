#!/usr/bin/env python3
"""The donor-level ceiling of the Parse PBMC panel, in the unit Rhaister reports.

Svensson et al. score their model by the across-gene Pearson correlation between predicted and
observed response, averaged over the held-out cell type's observations. That is a raw
correlation: it says how close the prediction came, not how close anything could have come.
This computes the second number from their own data.

MATCHING THE UNIT MATTERS. Their metric is row-wise: a row is one (donor, cytokine)
observation, not a donor-averaged profile — n_samples is 744 for their own split against 90
cytokines times 12 donors in the cell type, so the rows are per donor. A ceiling computed from
two donor-half averages would therefore be in the wrong unit and would read far too high. The
ceiling for a per-donor prediction is what the consensus of the other donors achieves on the
held-out one: leave one donor out, average the rest, correlate across genes. No predictor
without access to that donor can systematically beat its peers' consensus.

Donors, not cells. A split by cell measures sequencing depth as much as biology and climbs
towards one as depth rises; a donor split does not, which is why the ceiling it gives is below
one and worth dividing by.

Run on the compute host, where the panel lives:
    python3 run_rhaister_donor_ceiling.py <delta.parquet> <out.json>
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq

import argparse
from pathlib import Path
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('delta', help='Parse cell_eval/all_celltypes_delta.parquet')
ap.add_argument('out', nargs='?', default='artifacts/rhaister_benchmark/donor_ceiling.json')
args = ap.parse_args()
SRC, OUT = args.delta, args.out
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
MIN_DONORS = 6


def pear(a, b):
    """Row-wise Pearson across genes, as rhaister.state_metrics._rowwise_pearson does it."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


f = pq.ParquetFile(SRC)
meta = f.read(columns=['cell_type', 'donor', 'cytokine']).to_pandas()
genes = [n for n in f.schema_arrow.names if n not in ('cell_type', 'donor', 'cytokine')]
print(f'{f.metadata.num_rows} rows, {len(genes)} genes, '
      f'{meta.donor.nunique()} donors, {meta.cell_type.nunique()} cell types', flush=True)

out = {'min_donors': MIN_DONORS, 'n_genes': len(genes),
       'unit': 'leave-one-donor-out, row-wise Pearson across genes, matching '
               'rhaister pdex_static/pearson_delta_mean',
       'by_cell_type': {}}

for ct in sorted(meta.cell_type.unique()):
    rows = meta.index[meta.cell_type == ct].to_numpy()
    X = f.read(columns=genes).take(rows).to_pandas().to_numpy(dtype=np.float32)
    sub = meta.iloc[rows].reset_index(drop=True)
    per_row = []
    per_cytokine = {}
    for cy in sorted(sub.cytokine.unique()):
        idx = np.flatnonzero((sub.cytokine == cy).to_numpy())
        if len(idx) < MIN_DONORS:
            continue
        rs = []
        for i in idx:
            others = idx[idx != i]
            r = pear(X[i], np.nanmean(X[others], axis=0))
            if np.isfinite(r):
                rs.append(r)
        if rs:
            per_cytokine[cy] = dict(mean=float(np.mean(rs)), n_donors=int(len(idx)))
            per_row.extend(rs)
    if not per_row:
        continue
    out['by_cell_type'][ct] = dict(
        n_cytokines=len(per_cytokine),
        n_rows=len(per_row),
        n_donors_median=int(np.median([v['n_donors'] for v in per_cytokine.values()])),
        # the mean over rows, which is exactly how their metric aggregates
        ceiling_mean=float(np.mean(per_row)),
        ceiling_median=float(np.median(per_row)),
        ceiling_q05=float(np.quantile(per_row, 0.05)),
        ceiling_q95=float(np.quantile(per_row, 0.95)),
        per_cytokine=per_cytokine)
    print(f'  {ct:26s} rows={len(per_row):5d} ceiling {np.mean(per_row):.4f}', flush=True)

cts = out['by_cell_type']
out['overall'] = dict(
    ceiling_mean=float(np.mean([v['ceiling_mean'] for v in cts.values()])),
    ceiling_min=float(min(v['ceiling_mean'] for v in cts.values())),
    ceiling_max=float(max(v['ceiling_mean'] for v in cts.values())),
    n_cell_types=len(cts))
json.dump(out, open(OUT, 'w'), indent=1)
print('wrote', OUT, flush=True)
print('overall', round(out['overall']['ceiling_mean'], 4),
      'range', round(out['overall']['ceiling_min'], 4),
      '-', round(out['overall']['ceiling_max'], 4), flush=True)
