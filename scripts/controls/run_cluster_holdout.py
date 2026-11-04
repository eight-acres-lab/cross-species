"""Does the learned rat->human map generalise to perturbation types it never saw?

The strategic question is whether accumulating more context cells buys a model anything, or
whether each cell is an island. The existing scaling run held out compounds at random, so a
held-out compound usually resembled something in training. This holds out a whole CLUSTER of
compounds instead, and compares against random holdouts of the SAME SIZE.

The clustering uses only the RAT side, which is the model's input, never the human side it
predicts, so the grouping cannot leak the target. Holding out an input cluster asks exactly
the question: can the map extrapolate to perturbations unlike anything it was trained on?

  cluster holdout ~ random holdout  ->  cross-type structure is learnable; cells compound
  cluster holdout collapses to ~0   ->  each perturbation type is an island
"""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
from run_tggates_knn_validation import load_matrix, matched_change, row_pearson

D = Path('artifacts/tg_gates_rat_human_v1/data')
OUT = Path('artifacts/tg_gates_rat_human_v1/cluster_holdout'); OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261023
ALPHA = 1000.0          # the grid point the published run reports
rng = np.random.default_rng(SEED)

def ridge_predict(xtr, ytr, xte, alpha):
    xm = xtr.mean(axis=0, keepdims=True); xs = xtr.std(axis=0, keepdims=True); xs[xs < 1e-8] = 1.
    a = ((xtr - xm) / xs).astype(np.float32, copy=False)
    b = ((xte - xm) / xs).astype(np.float32, copy=False)
    ym = ytr.mean(axis=0, keepdims=True); yc = (ytr - ym).astype(np.float32, copy=False)
    gram = a @ a.T; gram.flat[::len(gram) + 1] += alpha
    return (b @ a.T) @ np.linalg.solve(gram, yc) + ym

print('loading'); sys.stdout.flush()
ids, x = load_matrix(D / 'pairs_rat_X.tsv')
yids, y = load_matrix(D / 'pairs_human_Y.tsv')
assert ids == yids
md = {r['pair_id']: r for r in csv.DictReader((D / 'pairs_metadata.tsv').open(), delimiter='\t')}
rows = [md[i] for i in ids]
x, _ = matched_change(x, ids, rows)
y, _ = matched_change(y, ids, rows)
treated = np.array([r['dose_level'].lower() != 'control' for r in rows])
ti = np.where(treated)[0]
comp = np.array([rows[i]['compound'] for i in ti])
compounds = sorted(set(comp))
print(f'{len(ti)} treated rows, {len(compounds)} compounds, {x.shape[1]} genes'); sys.stdout.flush()

# ---- per-compound RAT profile, then cluster on correlation distance ----
prof = np.vstack([x[ti[comp == c]].mean(axis=0) for c in compounds])
P = prof - prof.mean(axis=1, keepdims=True)
P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
C = P @ P.T                                    # compound-by-compound correlation, rat side only

def ward_like(C, k):
    """Average-linkage agglomeration on 1 - correlation."""
    Dm = 1.0 - C
    members = [[i] for i in range(len(C))]
    while len(members) > k:
        best, bi, bj = np.inf, 0, 1
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                d = Dm[np.ix_(members[a], members[b])].mean()
                if d < best: best, bi, bj = d, a, b
        members[bi] = members[bi] + members[bj]; members.pop(bj)
    return members

def macro(pred, truth, center):
    return float(np.nanmean(row_pearson(pred - center, truth - center)))

def run(test_comps):
    tec = set(test_comps)
    tr = ti[~np.isin(comp, list(tec))]
    te = ti[np.isin(comp, list(tec))]
    if len(te) < 5 or len(tr) < 50: return None
    center = y[tr].mean(axis=0)
    pred = ridge_predict(x[tr], y[tr], x[te], ALPHA)
    const = np.broadcast_to(center, y[te].shape)
    return {'n_test_compounds': len(tec), 'n_test_rows': int(len(te)), 'n_train_rows': int(len(tr)),
            'ridge': macro(pred, y[te], center), 'constant': macro(const, y[te], center)}

results = {}
for K in (4, 6, 8):
    groups = ward_like(C, K)
    sizes = sorted(len(g) for g in groups)
    print(f'\nK={K}  cluster sizes {sizes}'); sys.stdout.flush()
    per = []
    for gi, g in enumerate(groups):
        cs = [compounds[i] for i in g]
        r = run(cs)
        if r is None: continue
        # size-matched random controls
        rand = []
        for _ in range(20):
            rc = rng.choice(compounds, len(cs), replace=False).tolist()
            rr = run(rc)
            if rr: rand.append(rr['ridge'])
        r['random_matched_mean'] = float(np.mean(rand)) if rand else None
        r['random_matched_sd'] = float(np.std(rand)) if rand else None
        r['gap'] = r['ridge'] - r['random_matched_mean'] if rand else None
        r['cluster'] = gi
        per.append(r)
        print('  cluster %d: %2d compounds, %3d rows | ridge %+.4f  random(same size) %+.4f  gap %+.4f'
              % (gi, r['n_test_compounds'], r['n_test_rows'], r['ridge'],
                 r['random_matched_mean'], r['gap'])); sys.stdout.flush()
    results[f'K={K}'] = per
    w = np.array([p['n_test_rows'] for p in per], float)
    cl = np.array([p['ridge'] for p in per]); rd = np.array([p['random_matched_mean'] for p in per])
    print('  row-weighted: cluster %+.4f   random %+.4f   gap %+.4f'
          % (np.average(cl, weights=w), np.average(rd, weights=w),
             np.average(cl, weights=w) - np.average(rd, weights=w)))

json.dump(results, open(OUT / 'results.json', 'w'), indent=1)
print('\nwrote', OUT / 'results.json')
