"""Build cluster-structured panels for Rhaister, plus size-matched random controls.

Rhaister predicts an unseen perturbation in a held-out context by expressing it as a weighted
combination of the PANEL perturbations that remain observed in that context. Their split picks
the panel without regard to what the test perturbations look like. This asks what happens when
the panel contains nothing resembling the test set: the panel is drawn from one response
cluster and everything else is held out.

Clusters come from the response profiles themselves, averaged over contexts, so the grouping
uses only the measured responses and not any held-out quantity. Panel size is held at 28 and
the validation set at 4, matching parse/split_0 exactly, so cluster and random runs differ in
nothing but which cytokines sit in the panel.
"""
import json, sys
from pathlib import Path
import numpy as np
import polars as pl

import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir', type=Path, required=True, help='Parse data directory containing cell_eval/')
ap.add_argument('--split-dir', type=Path, required=True, help='Rhaister splits/parse directory containing split_0/')
ap.add_argument('--summary', type=Path, default=Path('artifacts/rhaister_benchmark/splits_made.json'))
args = ap.parse_args()
ROOT, OUT = args.data_dir, args.split_dir
args.summary.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261026)
N_PANEL, N_VAL = 28, 4

base = (OUT / 'split_0' / 'split.toml').read_text()
CELLTYPES = sorted({l.split('"')[1].split('.')[1]
                    for l in base.splitlines() if l.startswith('[fewshot."')})
DONORS = ['Donor%d' % i for i in range(1, 13)]
print(f'{len(CELLTYPES)} cell types, {len(DONORS)} donors', flush=True)

print('loading responses', flush=True)
df = pl.read_parquet(ROOT / 'cell_eval/all_celltypes_delta.parquet')
genes = [c for c in df.columns if c not in ('cell_type', 'donor', 'cytokine')]
prof = (df.group_by('cytokine').agg([pl.col(g).mean() for g in genes])
          .sort('cytokine'))
cyto = prof['cytokine'].to_list()
M = prof.select(genes).to_numpy().astype(np.float32)
print(f'{len(cyto)} cytokines x {M.shape[1]} genes', flush=True)

Z = M - M.mean(1, keepdims=True)
Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
C = Z @ Z.T

def agglom(C, k):
    D = 1 - C; mem = [[i] for i in range(len(C))]
    while len(mem) > k:
        best, bi, bj = np.inf, 0, 1
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                d = D[np.ix_(mem[a], mem[b])].mean()
                if d < best: best, bi, bj = d, a, b
        mem[bi] += mem[bj]; mem.pop(bj)
    return mem

def write_split(name, panel):
    """panel: list of cytokine names kept observed in the held-out context."""
    rest = [c for c in cyto if c not in set(panel)]
    val, test = rest[:N_VAL], rest[N_VAL:]
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    L = ['[training]', 'parse = "train"', '', '[zeroshot]', '', '[fewshot]']
    for ct in CELLTYPES:
        L += [f'[fewshot."parse.{ct}"]',
              'donors = [' + ', '.join(f"'{x}'" for x in DONORS) + ']',
              'val = [' + ', '.join(f"'{x}'" for x in val) + ']',
              'test = [' + ', '.join(f"'{x}'" for x in test) + ']', '']
    (d / 'split.toml').write_text('\n'.join(L))
    return len(panel), len(val), len(test)

# cluster panels: grow a cluster until it reaches N_PANEL, take that as the panel
made = {}
for K in (6, 10):
    lab = np.zeros(len(cyto), int)
    for j, g in enumerate(agglom(C, K)):
        for i in g: lab[i] = j
    sizes = [(j, int((lab == j).sum())) for j in range(K)]
    sizes.sort(key=lambda z: -z[1])
    for rank, (j, sz) in enumerate(sizes[:2]):
        idx = list(np.where(lab == j)[0])
        # top up from the nearest cytokines to that cluster's centroid, if short
        if len(idx) < N_PANEL:
            cen = Z[idx].mean(0); cen /= np.linalg.norm(cen) + 1e-12
            order = np.argsort(-(Z @ cen))
            for i in order:
                if i not in idx: idx.append(int(i))
                if len(idx) == N_PANEL: break
        panel = [cyto[i] for i in idx[:N_PANEL]]
        nm = f'cluster_K{K}_c{rank}'
        made[nm] = write_split(nm, panel)
        print(f'{nm}: panel {made[nm][0]}, val {made[nm][1]}, test {made[nm][2]}', flush=True)

for r in range(4):
    panel = [cyto[i] for i in rng.choice(len(cyto), N_PANEL, replace=False)]
    nm = f'randpanel_{r}'
    made[nm] = write_split(nm, panel)
    print(f'{nm}: panel {made[nm][0]}, val {made[nm][1]}, test {made[nm][2]}', flush=True)

json.dump({k: {'n_panel': v[0], 'n_val': v[1], 'n_test': v[2]} for k, v in made.items()},
          args.summary.open('w'), indent=1)
print('\nwrote', len(made), 'splits', flush=True)
