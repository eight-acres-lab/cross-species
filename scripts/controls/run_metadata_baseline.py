"""Does the ridge map beat a metadata-only predictor when a whole cluster is held out?

On a random compound split a predictor that knows only the dose/time stratum -- nothing about
which compound it is looking at -- reaches 96.6% of the fitted model. If that also holds when
an entire perturbation-response cluster is held out, then what survives cluster holdout is the
generic stress consensus and the map has learned nothing transferable about perturbations.
If instead the ridge pulls away from the metadata baseline under cluster holdout, the margin
is the part that is genuinely about the perturbation.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
from run_tggates_knn_validation import load_matrix, matched_change, row_pearson

D = Path('artifacts/tg_gates_rat_human_v1/data')
OUT = Path('artifacts/tg_gates_rat_human_v1/cluster_holdout')
ALPHA = 1000.0
rng = np.random.default_rng(20261023)

def ridge_predict(xtr, ytr, xte, alpha):
    xm = xtr.mean(axis=0, keepdims=True); xs = xtr.std(axis=0, keepdims=True); xs[xs < 1e-8] = 1.
    a = ((xtr - xm) / xs).astype(np.float32, copy=False); b = ((xte - xm) / xs).astype(np.float32, copy=False)
    ym = ytr.mean(axis=0, keepdims=True); yc = (ytr - ym).astype(np.float32, copy=False)
    gram = a @ a.T; gram.flat[::len(gram) + 1] += alpha
    return (b @ a.T) @ np.linalg.solve(gram, yc) + ym

ids, x = load_matrix(D / 'pairs_rat_X.tsv'); yids, y = load_matrix(D / 'pairs_human_Y.tsv')
md = {r['pair_id']: r for r in csv.DictReader((D / 'pairs_metadata.tsv').open(), delimiter='\t')}
rows = [md[i] for i in ids]
x, _ = matched_change(x, ids, rows); y, _ = matched_change(y, ids, rows)
ti = np.where(np.array([r['dose_level'].lower() != 'control' for r in rows]))[0]
comp = np.array([rows[i]['compound'] for i in ti])
strat = np.array([f"{rows[i]['dose_level']}|{rows[i]['time_h']}" for i in ti])
compounds = sorted(set(comp))

prof = np.vstack([x[ti[comp == c]].mean(axis=0) for c in compounds])
P = prof - prof.mean(axis=1, keepdims=True); P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
C = P @ P.T
def agglom(C, k):
    Dm = 1.0 - C; mem = [[i] for i in range(len(C))]
    while len(mem) > k:
        best, bi, bj = np.inf, 0, 1
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                d = Dm[np.ix_(mem[a], mem[b])].mean()
                if d < best: best, bi, bj = d, a, b
        mem[bi] = mem[bi] + mem[bj]; mem.pop(bj)
    return mem

def macro(pred, truth, center): return float(np.nanmean(row_pearson(pred - center, truth - center)))

def both(test_comps):
    tec = set(test_comps)
    m = np.isin(comp, list(tec))
    tr, te = ti[~m], ti[m]
    if len(te) < 5 or len(tr) < 50: return None
    center = y[tr].mean(axis=0)
    r = macro(ridge_predict(x[tr], y[tr], x[te], ALPHA), y[te], center)
    # metadata-only: predict the training mean of the same dose/time stratum
    str_tr, str_te = strat[~m], strat[m]
    pm = np.zeros((len(te), y.shape[1]), dtype=np.float32)
    for j, s in enumerate(str_te):
        sel = tr[str_tr == s]
        pm[j] = y[sel].mean(axis=0) if len(sel) else center
    return r, macro(pm, y[te], center), int(len(te))

print('%-6s %-28s %9s %9s %9s'%('K','holdout','ridge','meta-only','ridge−meta')); sys.stdout.flush()
out = {}
for K in (4, 6, 8):
    groups = agglom(C, K); recs = []
    for gi, g in enumerate(groups):
        cs = [compounds[i] for i in g]
        cl = both(cs)
        if not cl: continue
        rr, rm, rn = [], [], []
        for _ in range(20):
            z = both(rng.choice(compounds, len(cs), replace=False).tolist())
            if z: rr.append(z[0]); rm.append(z[1])
        recs.append(dict(k=K, cluster=gi, n_comp=len(cs), n_rows=cl[2],
                         cl_ridge=cl[0], cl_meta=cl[1],
                         rnd_ridge=float(np.mean(rr)), rnd_meta=float(np.mean(rm))))
        print('%-6s %-28s %+9.4f %+9.4f %+9.4f'%(K, f'cluster {gi} ({len(cs)}c/{cl[2]}r)',
              cl[0], cl[1], cl[0]-cl[1])); sys.stdout.flush()
    w = np.array([r['n_rows'] for r in recs], float)
    for tag, a, b in (('整簇留出', 'cl_ridge', 'cl_meta'), ('随机留出', 'rnd_ridge', 'rnd_meta')):
        ra = np.average([r[a] for r in recs], weights=w); rb = np.average([r[b] for r in recs], weights=w)
        print('   K=%d %s  行加权: ridge %+.4f  meta-only %+.4f  ridge−meta %+.4f  (meta 占 %.0f%%)'
              % (K, tag, ra, rb, ra - rb, 100 * rb / ra)); sys.stdout.flush()
    out[f'K={K}'] = recs
json.dump(out, open(OUT / 'metadata_control.json', 'w'), indent=1)
print('\nwrote', OUT / 'metadata_control.json')
