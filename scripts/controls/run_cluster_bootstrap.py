"""Is the per-cluster variation in transferability real, or sampling noise?

Three of six held-out clusters had a model beat the metadata baseline, twice by about +0.10.
Those wins rest on 9 to 248 rows. This resamples the HELD-OUT COMPOUNDS as clusters, the
same convention the project's scaling run uses for test uncertainty, and reports an interval
on (best model - metadata) for each held-out group. Predictions are fitted once and held
fixed; only the test set is resampled, so this is uncertainty about the held-out sample and
not about the training draw.

Clusters with fewer than three compounds get no interval: resampling rows inside one compound
would treat correlated rows as independent and report a confidence we do not have.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
from run_tggates_knn_validation import load_matrix, matched_change, row_pearson

D = Path('artifacts/tg_gates_rat_human_v1/data')
OUT = Path('artifacts/tg_gates_rat_human_v1/cluster_holdout')
rng = np.random.default_rng(20261023)
NB = 2000

def ridge(xtr, ytr, xte, alpha=1000.):
    xm = xtr.mean(0, keepdims=True); xs = xtr.std(0, keepdims=True); xs[xs < 1e-8] = 1.
    a = ((xtr - xm) / xs).astype(np.float32); b = ((xte - xm) / xs).astype(np.float32)
    ym = ytr.mean(0, keepdims=True); yc = (ytr - ym).astype(np.float32)
    g = a @ a.T; g.flat[::len(g) + 1] += alpha
    return (b @ a.T) @ np.linalg.solve(g, yc) + ym

def krr(xtr, ytr, xte, mult, lam=1.0):
    xm = xtr.mean(0, keepdims=True); xs = xtr.std(0, keepdims=True); xs[xs < 1e-8] = 1.
    A = ((xtr - xm) / xs).astype(np.float32); B = ((xte - xm) / xs).astype(np.float32)
    na = (A * A).sum(1); nb = (B * B).sum(1)
    d2 = np.maximum(na[:, None] + na[None, :] - 2 * A @ A.T, 0)
    med = np.median(d2[np.triu_indices(len(A), 1)]) or 1.0
    g_ = mult / med
    K = np.exp(-g_ * d2); Kt = np.exp(-g_ * np.maximum(nb[:, None] + na[None, :] - 2 * B @ A.T, 0))
    ym = ytr.mean(0, keepdims=True); yc = (ytr - ym).astype(np.float32)
    K.flat[::len(K) + 1] += lam
    return Kt @ np.linalg.solve(K, yc) + ym

ids, x = load_matrix(D / 'pairs_rat_X.tsv'); yids, y = load_matrix(D / 'pairs_human_Y.tsv')
md = {r['pair_id']: r for r in csv.DictReader((D / 'pairs_metadata.tsv').open(), delimiter='\t')}
rows = [md[i] for i in ids]
x, _ = matched_change(x, ids, rows); y, _ = matched_change(y, ids, rows)
ti = np.where(np.array([r['dose_level'].lower() != 'control' for r in rows]))[0]
comp = np.array([rows[i]['compound'] for i in ti])
strat = np.array([f"{rows[i]['dose_level']}|{rows[i]['time_h']}" for i in ti])
compounds = sorted(set(comp))
prof = np.vstack([x[ti[comp == c]].mean(0) for c in compounds])
P = prof - prof.mean(1, keepdims=True); P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
C = P @ P.T
def agglom(C, k):
    Dm = 1 - C; mem = [[i] for i in range(len(C))]
    while len(mem) > k:
        best, bi, bj = np.inf, 0, 1
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                d = Dm[np.ix_(mem[a], mem[b])].mean()
                if d < best: best, bi, bj = d, a, b
        mem[bi] += mem[bj]; mem.pop(bj)
    return mem

out = []
for gi, g in enumerate(agglom(C, 6)):
    cs = [compounds[i] for i in g]
    m = np.isin(comp, cs); tr, te = ti[~m], ti[m]
    if len(te) < 5: continue
    center = y[tr].mean(0)
    te_comp = comp[m]
    preds = {'ridge': ridge(x[tr], y[tr], x[te]),
             'krr_g0.5': krr(x[tr], y[tr], x[te], 0.5),
             'krr_g2': krr(x[tr], y[tr], x[te], 2.0)}
    st_tr, st_te = strat[~m], strat[m]
    pm = np.zeros((len(te), y.shape[1]), np.float32)
    for j, s in enumerate(st_te):
        sel = tr[st_tr == s]; pm[j] = y[sel].mean(0) if len(sel) else center
    sc = {k: row_pearson(v - center, y[te] - center) for k, v in preds.items()}
    sc['meta'] = row_pearson(pm - center, y[te] - center)
    best = max(preds, key=lambda k: np.nanmean(sc[k]))
    diff_rows = sc[best] - sc['meta']
    pt = float(np.nanmean(diff_rows))
    rec = {'cluster': gi, 'n_comp': len(cs), 'n_rows': int(len(te)), 'best_model': best,
           'best': float(np.nanmean(sc[best])), 'meta': float(np.nanmean(sc['meta'])), 'diff': pt}
    if len(cs) >= 3:
        uc = sorted(set(te_comp)); idxby = {c: np.where(te_comp == c)[0] for c in uc}
        bs = []
        for _ in range(NB):
            pick = rng.choice(uc, len(uc), replace=True)
            sel = np.concatenate([idxby[c] for c in pick])
            bs.append(np.nanmean(diff_rows[sel]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rec |= {'lo': float(lo), 'hi': float(hi), 'excludes_zero': bool(lo > 0 or hi < 0)}
    else:
        rec |= {'lo': None, 'hi': None, 'excludes_zero': None}
    out.append(rec)
    ci = ('[%+.3f, %+.3f] %s' % (rec['lo'], rec['hi'], '不含零' if rec['excludes_zero'] else '含零')) \
         if rec['lo'] is not None else '化合物不足 3 个，不给区间'
    print('%3dc/%4dr  %-9s best %+.3f  meta %+.3f  差 %+.3f  %s'
          % (rec['n_comp'], rec['n_rows'], best, rec['best'], rec['meta'], pt, ci)); sys.stdout.flush()
json.dump(out, open(OUT / 'bootstrap.json', 'w'), indent=1)
print('\nwrote', OUT / 'bootstrap.json')
