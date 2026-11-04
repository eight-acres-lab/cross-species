"""Does the random-vs-cluster reversal survive at ~100x the compound count?

LINCS L1000 Phase 2: 345,976 Level 3 wells, 948 plates, 41 human cell lines, 1,826 compounds.
Nothing cross-species is testable here -- every line is human and each is one donor -- so the
species ordering and the donor-level ceiling are out of reach. What IS testable, at two orders
of magnitude more perturbations than the 134-compound TG-GATEs panel, is the one claim left
after Rhaister published the ceiling and the trivial-baseline results:

    does the choice between a random split and a cluster split reverse the verdict on
    whether a fitted model beats a predictor that ignores its input?

Structure mirrors the TG-GATEs task exactly. Units are (compound, dose, time). Inputs are cell
line A's response, targets are cell line B's response to the same unit. Whole COMPOUNDS are
held out, so no dose or timepoint of a held-out compound leaks into training. Clusters are
built from the A side only, which is the model's input, so the grouping cannot see the target.

It also tests, unprompted, a prediction the contamination detector makes: LINCS pools DMSO per
plate, the same shared-control geometry that pushed the fraction of ceiling above 1 on two
unrelated datasets.
"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import h5py

import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-dir', type=Path, required=True, help='GSE70138 Level 3 GCTX and metadata')
ap.add_argument('--output-dir', type=Path, default=Path('artifacts/lincs_scale'))
args = ap.parse_args()
L, OUT = args.data_dir, args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
GCTX = L / 'GSE70138_Broad_LINCS_Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx'
rng = np.random.default_rng(20261026)
def log(*a): print(*a, flush=True)

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50: return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan

def rowwise(P, T):
    return np.array([pear(P[i], T[i]) for i in range(len(P))])

def ridge(xtr, ytr, xte, alpha=1000.):
    xm, xs = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True); xs[xs < 1e-8] = 1.
    a = ((xtr-xm)/xs).astype(np.float32); b = ((xte-xm)/xs).astype(np.float32)
    ym = ytr.mean(0, keepdims=True); yc = (ytr-ym).astype(np.float32)
    g = a @ a.T; g.flat[::len(g)+1] += alpha
    return (b @ a.T) @ np.linalg.solve(g, yc) + ym

log('metadata')
inst = {}
with gzip.open(L/'GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz','rt') as f:
    for r in csv.DictReader(f, delimiter='\t'): inst[r['inst_id']] = r
with gzip.open(L/'GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz','rt') as f:
    lmset = {r['pr_gene_id'] for r in csv.DictReader(f, delimiter='\t') if r.get('pr_is_lm')=='1'}

h = h5py.File(GCTX,'r')
cids = [x.decode() for x in h['0/META/COL/id'][:]]
rids = [x.decode() for x in h['0/META/ROW/id'][:]]
DS = h['0/DATA/0/matrix']
lm = np.array([i for i,r in enumerate(rids) if r in lmset])
log(f'{len(inst):,} wells | matrix {DS.shape} | {len(lm)} landmark genes')

byplate = defaultdict(list)
for i,c in enumerate(cids):
    r = inst.get(c)
    if r: byplate[r['det_plate']].append((i,r))
cc = np.array([sum(1 for _,r in ws if r['pert_type'].startswith('ctl')) for ws in byplate.values()])
tc = np.array([sum(1 for _,r in ws if not r['pert_type'].startswith('ctl')) for ws in byplate.values()])
log(f'plates {len(byplate):,} | ctrl/plate med {np.median(cc):.0f} | treated/plate med {np.median(tc):.0f} '
    f'| ratio {np.median(tc/np.maximum(cc,1)):.1f}:1')

def unit_of(r): return (r['pert_id'], r['pert_dose'], r['pert_time'])
cell_units = defaultdict(set)
for c in cids:
    r = inst.get(c)
    if r and r['pert_type']=='trt_cp': cell_units[r['cell_id']].add(unit_of(r))
top = sorted(cell_units, key=lambda z:-len(cell_units[z]))[:12]
best, bn = None, 0
for i,a in enumerate(top):
    for b in top[i+1:]:
        n = len(cell_units[a] & cell_units[b])
        if n > bn: bn, best = n, (a,b)
A, B = best
units = sorted(cell_units[A] & cell_units[B])
log(f'pair {A} / {B}: {bn:,} shared (compound,dose,time) units, '
    f'{len({u[0] for u in units}):,} distinct compounds')

def build(cell, units):
    want = set(units)
    ctrl, trt = defaultdict(list), defaultdict(list)
    for i,c in enumerate(cids):
        r = inst.get(c)
        if not r or r['cell_id']!=cell: continue
        if r['pert_type'].startswith('ctl'): ctrl[r['det_plate']].append(i)
        elif r['pert_type']=='trt_cp' and unit_of(r) in want: trt[(r['det_plate'],unit_of(r))].append(i)
    need = sorted({i for v in ctrl.values() for i in v} | {i for v in trt.values() for i in v})
    pos = {g:k for k,g in enumerate(need)}
    log(f'  {cell}: reading {len(need):,} wells')
    M = DS[np.array(need),:][:,lm].astype(np.float32)
    cm = {pl: M[[pos[i] for i in v]].mean(0) for pl,v in ctrl.items() if v}
    out = defaultdict(list)
    for (pl,u),v in trt.items():
        if pl in cm: out[u].append(M[[pos[i] for i in v]].mean(0) - cm[pl])
    return {k: np.vstack(v) for k,v in out.items() if v}

log('building responses')
RA, RB = build(A, units), build(B, units)
common = sorted(set(RA) & set(RB))
log(f'{len(common):,} units present in both lines')

# ---- ceiling from cross-plate replicates, and the above-one check ----
def sh(reps):
    if len(reps) < 2: return np.nan
    k = len(reps)//2
    return pear(reps[:k].mean(0), reps[k:].mean(0))
rel = {u: (sh(RA[u]), sh(RB[u])) for u in common}
use = [u for u in common if all(np.isfinite(x) and x > 0.1 for x in rel[u])]
frac = {}
for u in use:
    ce = np.sqrt(rel[u][0]*rel[u][1])
    if ce > 0: frac[u] = pear(RA[u].mean(0), RB[u].mean(0))/ce
fv = np.array(list(frac.values()))
above = int((fv > 1).sum())
log(f'ceiling computable on {len(use):,} units | fraction median {np.median(fv):+.3f} '
    f'max {fv.max():+.3f} | ABOVE 1: {above} ({100*above/len(fv):.1f}%)')

# ---- the split test ----
X = np.vstack([RA[u].mean(0) for u in common])
Y = np.vstack([RB[u].mean(0) for u in common])
cmpd = np.array([u[0] for u in common])
strat = np.array([f'{u[1]}|{u[2]}' for u in common])
comps = sorted(set(cmpd))
log(f'design matrix {X.shape} over {len(comps):,} compounds')

P = X - X.mean(1, keepdims=True)
P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
prof = np.vstack([P[cmpd==c].mean(0) for c in comps])
prof /= (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-12)
C = prof @ prof.T

def kmeans(C, k, iters=40):
    D = 1 - C; n = len(C)
    cen = rng.choice(n, k, replace=False)
    for _ in range(iters):
        lab = np.argmin(D[:, cen], axis=1)
        new = []
        for j in range(k):
            m = np.where(lab == j)[0]
            new.append(cen[j] if len(m)==0 else m[np.argmin(D[np.ix_(m,m)].sum(1))])
        if list(new) == list(cen): break
        cen = np.array(new)
    return np.argmin(D[:, cen], axis=1)

def run(test_comps):
    te = np.isin(cmpd, list(test_comps)); tr = ~te
    if te.sum() < 20 or tr.sum() < 200: return None
    center = Y[tr].mean(0)
    rp = ridge(X[tr], Y[tr], X[te])
    pm = np.zeros((te.sum(), Y.shape[1]), np.float32)
    st_tr, st_te = strat[tr], strat[te]
    for j,s in enumerate(st_te):
        sel = Y[tr][st_tr == s]
        pm[j] = sel.mean(0) if len(sel) else center
    return (float(np.nanmean(rowwise(rp - center, Y[te] - center))),
            float(np.nanmean(rowwise(pm - center, Y[te] - center))), int(te.sum()))

res = {}
for K in (4, 6, 8):
    lab = kmeans(C, K)
    recs = []
    for j in range(K):
        cs = [comps[i] for i in np.where(lab == j)[0]]
        cl = run(cs)
        if not cl: continue
        rnd = [run(rng.choice(comps, len(cs), replace=False).tolist()) for _ in range(5)]
        rnd = [r for r in rnd if r]
        recs.append({'cluster': j, 'n_comp': len(cs), 'n_rows': cl[2],
                     'cl_ridge': cl[0], 'cl_meta': cl[1],
                     'rnd_ridge': float(np.mean([r[0] for r in rnd])),
                     'rnd_meta': float(np.mean([r[1] for r in rnd]))})
        log(f'  K={K} c{j}: {len(cs):4d} cmpd / {cl[2]:5d} rows | '
            f'整簇 ridge {cl[0]:+.4f} meta {cl[1]:+.4f} | '
            f'随机 ridge {recs[-1]["rnd_ridge"]:+.4f} meta {recs[-1]["rnd_meta"]:+.4f}')
    w = np.array([r['n_rows'] for r in recs], float)
    agg = {t: {m: float(np.average([r[f'{t}_{m}'] for r in recs], weights=w)) for m in ('ridge','meta')}
           for t in ('cl','rnd')}
    log(f'  K={K} 行加权 | 整簇 ridge {agg["cl"]["ridge"]:+.4f} meta {agg["cl"]["meta"]:+.4f} '
        f'| 随机 ridge {agg["rnd"]["ridge"]:+.4f} meta {agg["rnd"]["meta"]:+.4f}')
    res[f'K={K}'] = {'per_cluster': recs, 'row_weighted': agg}

json.dump({'pair': [A, B], 'n_units': len(common), 'n_compounds': len(comps),
           'n_genes': int(len(lm)), 'plates': len(byplate),
           'ctrl_per_plate_median': float(np.median(cc)),
           'treated_per_plate_median': float(np.median(tc)),
           'ceiling': {'n_usable': len(use), 'median': float(np.median(fv)),
                       'max': float(fv.max()), 'n_above_one': above,
                       'pct_above_one': float(100*above/len(fv))},
           'splits': res}, open(OUT/'scale_test.json','w'), indent=1)
log('wrote scale_test.json')
h.close()
