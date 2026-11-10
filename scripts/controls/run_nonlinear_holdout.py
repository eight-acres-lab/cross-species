"""Does a NON-linear map extrapolate to unseen perturbation types where ridge did not?

Ridge lost to a predictor that ignores its input when a whole perturbation cluster was held
out. That bounds linear extrapolation only. Here the same holdout runs against three
structurally different non-linear families, all with hyperparameters fixed in advance on a
grid and never chosen on the held-out fold:

  - RBF kernel ridge, gamma set by the median pairwise distance in TRAINING data
  - k-nearest neighbours in rat space, a non-parametric predictor
  - an MLP fitted in a PCA latent space learned from TRAINING data only

If every family also loses to metadata-only under cluster holdout, the failure is a property
of the data rather than of the estimator, and "accumulate cells so a model can learn across
them" needs a different argument. If a non-linear family wins, that is the case for needing
models at all.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
from run_tggates_knn_validation import load_matrix, matched_change, row_pearson

D = Path('artifacts/tg_gates_rat_human_v1/data')
OUT = Path('artifacts/tg_gates_rat_human_v1/cluster_holdout')
rng = np.random.default_rng(20261023)
N_RANDOM = 5

def macro(pred, truth, center): return float(np.nanmean(row_pearson(pred - center, truth - center)))

def ridge(xtr, ytr, xte, alpha=1000.):
    xm = xtr.mean(0, keepdims=True); xs = xtr.std(0, keepdims=True); xs[xs < 1e-8] = 1.
    a = ((xtr - xm) / xs).astype(np.float32); b = ((xte - xm) / xs).astype(np.float32)
    ym = ytr.mean(0, keepdims=True); yc = (ytr - ym).astype(np.float32)
    g = a @ a.T; g.flat[::len(g) + 1] += alpha
    return (b @ a.T) @ np.linalg.solve(g, yc) + ym

def krr_rbf(xtr, ytr, xte, mult, lam=1.0):
    """RBF kernel ridge. gamma from the median training distance, scaled by `mult`."""
    xm = xtr.mean(0, keepdims=True); xs = xtr.std(0, keepdims=True); xs[xs < 1e-8] = 1.
    A = ((xtr - xm) / xs).astype(np.float32); B = ((xte - xm) / xs).astype(np.float32)
    na = (A * A).sum(1); nb = (B * B).sum(1)
    d2 = np.maximum(na[:, None] + na[None, :] - 2 * A @ A.T, 0)
    med = np.median(d2[np.triu_indices(len(A), 1)]) or 1.0
    gamma = mult / med
    K = np.exp(-gamma * d2)
    Kt = np.exp(-gamma * np.maximum(nb[:, None] + na[None, :] - 2 * B @ A.T, 0))
    ym = ytr.mean(0, keepdims=True); yc = (ytr - ym).astype(np.float32)
    K.flat[::len(K) + 1] += lam
    return Kt @ np.linalg.solve(K, yc) + ym

def knn(xtr, ytr, xte, k):
    xm = xtr.mean(0, keepdims=True)
    A = (xtr - xm); B = (xte - xm)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = B @ A.T
    kk = min(k, len(A))
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    return np.stack([ytr[i].mean(0) for i in idx])

def mlp(xtr, ytr, xte, latent=64, hidden=256, epochs=400, lr=3e-3, seed=0):
    """MLP in a PCA latent space. Both bases come from the training fold only."""
    r = np.random.default_rng(seed)
    xm = xtr.mean(0, keepdims=True); ym = ytr.mean(0, keepdims=True)
    Xc = (xtr - xm).astype(np.float32); Yc = (ytr - ym).astype(np.float32)
    def basis(M, k):
        G = M @ M.T
        w, V = np.linalg.eigh(G)
        o = np.argsort(w)[::-1][:k]
        w = np.maximum(w[o], 1e-12)
        return ((V[:, o].T @ M) / np.sqrt(w)[:, None]).astype(np.float32)
    Vx = basis(Xc, latent); Vy = basis(Yc, latent)
    Zx = Xc @ Vx.T; Zy = Yc @ Vy.T
    zs = Zx.std(0, keepdims=True); zs[zs < 1e-8] = 1.; Zx = Zx / zs
    ys = Zy.std(0, keepdims=True); ys[ys < 1e-8] = 1.; Zyn = Zy / ys
    W1 = r.normal(0, (2 / latent) ** .5, (latent, hidden)).astype(np.float32); b1 = np.zeros(hidden, np.float32)
    W2 = r.normal(0, (2 / hidden) ** .5, (hidden, latent)).astype(np.float32); b2 = np.zeros(latent, np.float32)
    ms = [np.zeros_like(p) for p in (W1, b1, W2, b2)]; vs = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
    n = len(Zx)
    for t in range(1, epochs + 1):
        H = np.maximum(Zx @ W1 + b1, 0); P = H @ W2 + b2
        G = (P - Zyn) * (2.0 / n)
        gW2 = H.T @ G; gb2 = G.sum(0)
        Gh = (G @ W2.T) * (H > 0)
        gW1 = Zx.T @ Gh; gb1 = Gh.sum(0)
        for i, (p, g) in enumerate(zip((W1, b1, W2, b2), (gW1, gb1, gW2, gb2))):
            ms[i] = .9 * ms[i] + .1 * g; vs[i] = .999 * vs[i] + .001 * g * g
            p -= lr * (ms[i] / (1 - .9 ** t)) / (np.sqrt(vs[i] / (1 - .999 ** t)) + 1e-8)
    Zt = ((xte - xm) @ Vx.T) / zs
    Pt = np.maximum(Zt @ W1 + b1, 0) @ W2 + b2
    return (Pt * ys) @ Vy + ym

print('loading'); sys.stdout.flush()
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

MODELS = {
    'ridge_linear':  lambda a, b, c: ridge(a, b, c),
    'krr_rbf_g0.5':  lambda a, b, c: krr_rbf(a, b, c, 0.5),
    'krr_rbf_g2':    lambda a, b, c: krr_rbf(a, b, c, 2.0),
    'knn_k20':       lambda a, b, c: knn(a, b, c, 20),
    'mlp_pca64':     lambda a, b, c: mlp(a, b, c),
}

def fit_all(test_comps):
    m = np.isin(comp, list(set(test_comps)))
    tr, te = ti[~m], ti[m]
    if len(te) < 5 or len(tr) < 50: return None
    center = y[tr].mean(0)
    out = {}
    for name, fn in MODELS.items():
        try: out[name] = macro(fn(x[tr], y[tr], x[te]), y[te], center)
        except Exception as e: out[name] = float('nan')
    st_tr, st_te = strat[~m], strat[m]
    pm = np.zeros((len(te), y.shape[1]), np.float32)
    for j, s in enumerate(st_te):
        sel = tr[st_tr == s]; pm[j] = y[sel].mean(0) if len(sel) else center
    out['meta_only'] = macro(pm, y[te], center)
    out['_rows'] = int(len(te))
    return out

res = {}
for K in (6,):
    groups = agglom(C, K); recs = []
    print(f'\n=== K={K} ==='); sys.stdout.flush()
    for gi, g in enumerate(groups):
        cs = [compounds[i] for i in g]
        cl = fit_all(cs)
        if not cl: continue
        rnd = []
        for ri in range(N_RANDOM):
            rnd.append(fit_all(rng.choice(compounds, len(cs), replace=False).tolist()))
            print(f'      random {ri+1}/{N_RANDOM}', flush=True)
        rnd = [r for r in rnd if r]
        recs.append({'cluster': gi, 'n_comp': len(cs), 'n_rows': cl['_rows'], 'cluster_holdout': cl,
                     'random_holdout': {k: float(np.nanmean([r[k] for r in rnd])) for k in MODELS | {'meta_only': 0}}})
        print(f'  cluster {gi} ({len(cs)}c/{cl["_rows"]}r)  ' +
              '  '.join(f'{k}={cl[k]:+.3f}' for k in list(MODELS) + ['meta_only'])); sys.stdout.flush()
    res[f'K={K}'] = recs
    w = np.array([r['n_rows'] for r in recs], float)
    print(f'  --- K={K} 行加权 ---')
    for k in list(MODELS) + ['meta_only']:
        a = np.average([r['cluster_holdout'][k] for r in recs], weights=w)
        b = np.average([r['random_holdout'][k] for r in recs], weights=w)
        print(f'    {k:14s} 整簇 {a:+.4f}   随机 {b:+.4f}'); sys.stdout.flush()
json.dump(res, open(OUT / 'nonlinear.json', 'w'), indent=1)
print('\nwrote', OUT / 'nonlinear.json')
