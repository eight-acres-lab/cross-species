#!/usr/bin/env python3
"""Is the donor ceiling a limit of the data, or of the twelve donors that were collected?

A leave-one-donor-out ceiling is computed against the consensus of the other donors, and a
consensus built from more donors is a better predictor. So the ceiling reported for a panel is
conditioned on that panel's donor count, and the honest question is whether it has levelled off
by the count in hand or is still climbing.

This rebuilds the consensus from k = 2 to n-1 donors, drawn at random, and reports the curve.
If the last steps are flat the ceiling is a property of the biology; if it is still rising, the
number is a statement about twelve donors and has to be read that way.

Run on the compute host:
    python3 run_rhaister_ceiling_saturation.py <delta.parquet> <out.json> [cell_type ...]
"""
import json, sys
import numpy as np
import pyarrow.parquet as pq

import argparse
from pathlib import Path
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('delta', help='Parse cell_eval/all_celltypes_delta.parquet')
ap.add_argument('out', nargs='?', default='artifacts/rhaister_benchmark/ceiling_saturation.json')
ap.add_argument('cell_types', nargs='*')
args = ap.parse_args()
SRC, OUT = args.delta, args.out
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
WANT = args.cell_types or ['B_Intermediate_Memory', 'CD4_Memory', 'CD14_Mono', 'NK']
DRAWS = 12
SEED = 0


def pear(a, b):
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
rng = np.random.default_rng(SEED)
out = {'draws': DRAWS, 'cell_types': {}}

for ct in WANT:
    rows = meta.index[meta.cell_type == ct].to_numpy()
    if len(rows) == 0:
        print(f'  {ct}: not in the panel'); continue
    X = f.read(columns=genes).take(rows).to_pandas().to_numpy(dtype=np.float32)
    sub = meta.iloc[rows].reset_index(drop=True)
    curve = {}
    for cy in sorted(sub.cytokine.unique()):
        idx = np.flatnonzero((sub.cytokine == cy).to_numpy())
        if len(idx) < 4:
            continue
        for k in range(2, len(idx)):
            rs = []
            for _ in range(DRAWS):
                held = rng.choice(idx)
                pool = idx[idx != held]
                pick = rng.choice(pool, size=k, replace=False)
                r = pear(X[held], np.nanmean(X[pick], axis=0))
                if np.isfinite(r):
                    rs.append(r)
            if rs:
                curve.setdefault(k, []).append(float(np.mean(rs)))
    if not curve:
        continue
    ks = sorted(curve)
    means = [float(np.mean(curve[k])) for k in ks]
    out['cell_types'][ct] = dict(
        k=ks, ceiling=means,
        last_step=float(means[-1] - means[-2]) if len(means) > 1 else None,
        total_rise=float(means[-1] - means[0]))
    print(f'  {ct:24s} ' + ' '.join(f'k{k}={m:.3f}' for k, m in zip(ks, means)), flush=True)
    print(f'  {"":24s} last step {means[-1] - means[-2]:+.4f}, '
          f'total rise {means[-1] - means[0]:+.4f}', flush=True)

json.dump(out, open(OUT, 'w'), indent=1)
print('wrote', OUT, flush=True)
