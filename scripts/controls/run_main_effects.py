"""Direct check: do the species main effects transfer worse than the interaction?

Double-centring raised every cross-time correlation. Removing a component usually lowers one,
so the natural reading is that the removed part was itself failing to transfer and dragging
the total down. That is an inference. This measures the removed part directly: correlate the
per-species main effects across timepoints, and compare against the interaction's own
correlation over the same pairs.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
src = open('scripts/run_rho_reliability_and_phylogeny.py').read().split('if __name__')[0]
M = {}; exec(compile(src, 'mod', 'exec'), M)
M['load_bridge']('artifacts/reference/ensg_symbol_bridge.tsv')
rows, data, genes = M['load_blood']('artifacts/gse249010_v1/processed')
times = sorted({r['time_h'] for r in rows}, key=int)

def matrix(t):
    kf = (lambda tt: (lambda r: r['time_h'] == tt))(t)
    d = M['rho_matrix'](M['species_vectors'](rows, data, kf))
    sp = sorted({s for pr in d for s in pr}); idx = {s: i for i, s in enumerate(sp)}
    A = np.full((len(sp), len(sp)), np.nan)
    for (a, b), v in d.items(): A[idx[a], idx[b]] = A[idx[b], idx[a]] = v
    return sp, A

def parts(A):
    n = len(A)
    rm = np.array([np.nanmean(np.delete(A[i], i)) for i in range(n)])
    gm = np.nanmean(A[~np.eye(n, dtype=bool)])
    R = np.full_like(A, np.nan)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(A[i, j]): R[i, j] = A[i, j] - rm[i] - rm[j] + gm
    return rm, R

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
def spear(a, b):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pear(r(a), r(b))

mats = {t: matrix(t) for t in times}
mains = {}; inter = {}
for t in times:
    sp, A = mats[t]; rm, R = parts(A)
    mains[t] = (sp, rm); inter[t] = (sp, R)

SH = {'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
      'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
      'Mus musculus':'mouse','Rattus norvegicus':'rat','Homo sapiens':'human'}
sp0 = mains[times[0]][0]
print('每个物种的主效应（该物种与其他所有物种的平均 ρ）')
print('%-9s %8s %8s %8s'%('species', *[t+'h' for t in times]))
for i, s in enumerate(sp0):
    print('%-9s %8.3f %8.3f %8.3f'%(SH.get(s, s[:9]), *[mains[t][1][i] for t in times]))

print('\n跨时间点：主效应 vs 配对交互')
print('%-12s %14s %14s'%('比较', '主效应 Spearman', '交互 Spearman(跨目)'))
out = {'main_effects': {t: {SH.get(s, s): float(mains[t][1][i]) for i, s in enumerate(sp0)} for t in times},
       'cross_time': []}
for a, b in (('2','6'), ('2','24'), ('6','24')):
    ms = spear(mains[a][1], mains[b][1])
    Ra, Rb = inter[a][1], inter[b][1]
    n = len(sp0)
    td = np.array([[M['taxdist'](sp0[i], sp0[j]) for j in range(n)] for i in range(n)], dtype=float)
    mask = (~np.eye(n, dtype=bool)) & np.isfinite(Ra) & np.isfinite(Rb) & (td >= 4)
    iu = np.triu(mask, 1)
    isp = spear(Ra[iu], Rb[iu])
    out['cross_time'].append({'pair': f'{a}h vs {b}h', 'main_effect_spearman': ms,
                              'interaction_cross_order_spearman': isp, 'n_cross_order_pairs': int(iu.sum())})
    print('%-12s %14.3f %14.3f'%(f'{a}h vs {b}h', ms, isp))

json.dump(out, open('artifacts/biology/concordance/rho_main_effects.json','w'), indent=1, default=float)
print('\nwrote artifacts/biology/concordance/rho_main_effects.json')
