#!/usr/bin/env python3
"""The system penalty, finally measured without a time gap.

The first round put rat hepatocytes at 24 hours against rat liver at 4 days, because the
repeat-dose arm starts at day 4, and concluded that system and time could not be separated.
That was wrong: TG-GATEs also ran a SINGLE-dose rat liver arm sampled at 3, 6, 9 and 24
hours. The 24 hour cell matches the in vitro grid exactly, so species and time can both be
held while only the system changes.

Every row carries a compound-shuffled arm, because a large part of what crosses any of these
boundaries is the generic response that a random other compound would also have produced.
Rows are also reported inside a fixed ceiling band, since the ratio observed/ceiling tracks
the ceiling and medians taken over different ceiling distributions are not comparable.
"""
import argparse, itertools, json, math, sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from run_system_vs_species import load, pearson, direction   # noqa: E402


def one(A, B, da, ta, db, tb, label, draws, rng, min_rel=0.10):
    ca = {k[0]: A['index'][k] for k in A['index'] if k[1] == da and k[2] == ta}
    cb = {k[0]: B['index'][k] for k in B['index'] if k[1] == db and k[2] == tb}
    shared = sorted(set(ca) & set(cb))
    if len(shared) < 20:
        print(f'{label}: only {len(shared)} shared compounds'); return None
    Av = A['values'][:, [ca[c] for c in shared], :]
    Bv = B['values'][:, [cb[c] for c in shared], :]
    na, nb = min(2, Av.shape[2]), min(2, Bv.shape[2])
    out = {'label': label, 'a': f'{da}/{ta}', 'b': f'{db}/{tb}', 'n': len(shared), 'rows': []}
    for i, c in enumerate(shared):
        a, b = Av[:, i, :], Bv[:, i, :]
        rel_a, rel_b = pearson(a[:, 0], a[:, 1]), pearson(b[:, 0], b[:, 1])
        cross = [pearson(a[:, x], b[:, y]) for x, y in itertools.product(range(na), range(nb))]
        cross = [v for v in cross if np.isfinite(v)]
        obs = float(np.mean(cross)) if cross else float('nan')
        others = [j for j in range(len(shared)) if j != i]
        pick = rng.choice(others, size=min(draws, len(others)), replace=False)
        sh = []
        for j in pick:
            cj = [pearson(a[:, x], Bv[:, j, y]) for x, y in itertools.product(range(na), range(nb))]
            cj = [v for v in cj if np.isfinite(v)]
            if cj:
                sh.append(float(np.mean(cj)))
        ok = rel_a >= min_rel and rel_b >= min_rel
        bound = math.sqrt(rel_a * rel_b) if ok else None
        out['rows'].append({
            'compound': c, 'obs': obs, 'bound': bound,
            'frac': obs / bound if bound and np.isfinite(obs) else None,
            'shuffled': float(np.median(sh)) if sh else None,
            'specific': (obs - float(np.median(sh))) if sh and np.isfinite(obs) else None,
            'specific_frac': ((obs - float(np.median(sh))) / bound)
                             if sh and bound and np.isfinite(obs) else None})
    return out


def report(o, lo=0.60, hi=0.78):
    if not o:
        return
    rs = [r for r in o['rows'] if r.get('frac') is not None]
    f = np.array([r['frac'] for r in rs]) * 100
    sp = np.array([r['specific_frac'] for r in rs if r.get('specific_frac') is not None]) * 100
    bd = np.array([r['bound'] for r in rs])
    ob = np.array([r['obs'] for r in rs])
    shu = np.array([r['shuffled'] for r in rs if r.get('shuffled') is not None])
    band = (bd >= lo) & (bd < hi)
    bandtxt = (f'{np.median(f[band]):6.1f}  (n={int(band.sum())})'
               if band.sum() >= 12 else f'{"too few":>13s}')
    print(f"  {o['label']:46s} n={o['n']:3d}  obs {np.median(ob):+.4f}  "
          f"shuf {np.median(shu):+.4f}  ceiling {np.median(bd):.3f}")
    print(f"  {'':46s} % of ceiling {np.median(f):6.1f}   compound-specific {np.median(sp):6.1f}"
          f"   at matched ceiling {bandtxt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-hep', required=True)
    ap.add_argument('--rat-hep', required=True)
    ap.add_argument('--rat-single', required=True)
    ap.add_argument('--rat-repeat', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--draws', type=int, default=25)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    axis = json.load(open(a.axis))['axis']
    H = load(a.human_hep, 'homo_sapiens', axis)
    R = load(a.rat_hep, 'rattus_norvegicus', axis)
    S = load(a.rat_single, 'rattus_norvegicus', axis)
    V = load(a.rat_repeat, 'rattus_norvegicus', axis)
    print(f'axis verified for four files: {len(axis):,} ortholog pairs')
    rng = np.random.default_rng(a.seed)
    jobs = [
        (R, S, 'High', '24 hr', 'High', '24 hr', 'SYSTEM, TIME MATCHED: rat hep 24h vs rat liver 24h'),
        (R, V, 'High', '24 hr', 'High', '4 day', 'SYSTEM, time mismatched: rat hep 24h vs liver 4d'),
        (H, S, 'High', '24 hr', 'High', '24 hr', 'SPECIES+SYSTEM, time matched: human hep 24h vs rat liver 24h'),
        (H, R, 'High', '24 hr', 'High', '24 hr', 'SPECIES only: human hep 24h vs rat hep 24h'),
        (S, S, 'High', '24 hr', 'High', '9 hr', 'TIME in vivo: 24h vs 9h'),
        (S, S, 'High', '24 hr', 'High', '6 hr', 'TIME in vivo: 24h vs 6h'),
        (S, S, 'High', '24 hr', 'High', '3 hr', 'TIME in vivo: 24h vs 3h'),
        (S, V, 'High', '24 hr', 'High', '4 day', 'TIME in vivo: single 24h vs repeat 4d'),
    ]
    out = {}
    for A, B, da, ta, db, tb, lbl in jobs:
        o = one(A, B, da, ta, db, tb, lbl, a.draws, rng)
        report(o)
        if o:
            out[lbl] = o
    json.dump(out, open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
