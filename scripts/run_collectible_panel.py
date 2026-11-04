#!/usr/bin/env python3
"""What exchange rate is reachable with only the species a company can actually collect?

The candidate panel contains pig, cow, sheep, rabbit, rat and mouse. Those are the six lowest
single-donor correlations with human in this panel; the three highest are primates, which are
not on the collection list. The analysis measures the agreement achievable by combining these six species.

Everything is reported as correlation with the human response rather than as recovery under
an identity map, because the identity map charges each donor for its response amplitude as
well as its agreement and drove two donors below zero for that reason alone. r^2 is the
recovery a single fitted scaling coefficient would reach, which is the honest ceiling for a
weighted-average predictor.

The condition-specific column removes each species' mean across the nine dose-by-time cells
first, so it answers the harder question: does the panel predict WHICH condition, or only
that a response occurred.
"""
import itertools, json, sys
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from run_reconstruction_specificity import load   # noqa: E402

COLLECTIBLE = ['Sus scrofa', 'Bos taurus', 'Ovis aries',
               'Oryctolagus cuniculus', 'Mus musculus', 'Rattus norvegicus']
PRIMATE = ['Pan troglodytes', 'Macaca', 'Papio anubis']
SHORT = {'Sus scrofa': 'pig', 'Bos taurus': 'cow', 'Ovis aries': 'sheep',
         'Oryctolagus cuniculus': 'rabbit', 'Mus musculus': 'mouse',
         'Rattus norvegicus': 'rat', 'Pan troglodytes': 'chimp',
         'Macaca': 'macaque', 'Papio anubis': 'baboon'}


def corr(a, b, m):
    x, y = a[m], b[m]
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / d if d > 0 else np.nan


def score(G, keep, ti, idx, nT, centred=False):
    """mean correlation with human across conditions, for an equally weighted subset."""
    H = G
    if centred:
        with np.errstate(invalid='ignore'):
            H = G - np.nanmean(G, axis=1, keepdims=True)
    rs = []
    for t in range(nT):
        pred = np.nanmean(np.stack([H[i, t] for i in idx]), axis=0)
        y = H[ti, t]
        m = keep & np.isfinite(y) & np.isfinite(pred)
        rs.append(corr(y, pred, m))
    return float(np.nanmean(rs))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/dev/stdout'
    data_dir = sys.argv[2] if len(sys.argv) > 2 else 'artifacts/gse249010_v1/processed'
    species, tasks, G = load(data_dir)
    keep = np.isfinite(G).all(axis=(0, 1))
    ti = species.index('Homo sapiens')
    nT = len(tasks)
    ix = {s: species.index(s) for s in species}
    coll = [ix[s] for s in COLLECTIBLE if s in ix]
    prim = [ix[s] for s in PRIMATE if s in ix]
    print(f'{len(tasks)} conditions, {int(keep.sum()):,} genes')
    print(f'collectible: {[SHORT[species[i]] for i in coll]}')
    print(f'primates (not collected): {[SHORT[species[i]] for i in prim]}\n')

    rows = []
    for k in range(1, len(coll) + 1):
        for sub in itertools.combinations(coll, k):
            rows.append((sorted(SHORT[species[i]] for i in sub),
                         score(G, keep, ti, sub, nT),
                         score(G, keep, ti, sub, nT, centred=True)))
    rows.sort(key=lambda r: -r[1])
    print(f'all {len(rows)} subsets of the collectible six, best first')
    print(f'{"panel":42s} {"r with human":>12s} {"r^2":>7s} {"condition-specific r":>21s}')
    for names, r, rc in rows[:12]:
        print(f'  {"+".join(names):40s} {r:+12.4f} {r*r:7.4f} {rc:+21.4f}')
    print('  ...')
    for names, r, rc in rows[-4:]:
        print(f'  {"+".join(names):40s} {r:+12.4f} {r*r:7.4f} {rc:+21.4f}')

    print('\nreference points')
    for lbl, idx in (('all six collectible', coll),
                     ('chimpanzee alone', [ix['Pan troglodytes']]),
                     ('macaque alone', [ix['Macaca']]),
                     ('best collectible + macaque', None),
                     ('all nine donors', [i for i in range(len(species)) if i != ti])):
        if idx is None:
            best = max(rows, key=lambda r: r[1])[0]
            idx = [ix[s] for s in species if SHORT.get(s) in best] + [ix['Macaca']]
            lbl = f'{"+".join(best)}+macaque'
        r = score(G, keep, ti, idx, nT)
        rc = score(G, keep, ti, idx, nT, centred=True)
        print(f'  {lbl:40s} {r:+12.4f} {r*r:7.4f} {rc:+21.4f}')

    best = rows[0]
    allsix = score(G, keep, ti, coll, nT)
    chimp = score(G, keep, ti, [ix['Pan troglodytes']], nT)
    print(f'\npanel comparison:')
    print(f'  best collectible panel   r = {best[1]:.4f}  ({"+".join(best[0])})')
    print(f'  all six together         r = {allsix:.4f}')
    print(f'  chimpanzee alone         r = {chimp:.4f}')
    print(f'  so the collectible six reach {100*best[1]/chimp:.1f}% of a chimpanzee, '
          f'on correlation')
    json.dump({'conditions': tasks, 'subsets': [{'panel': n, 'r': r, 'r2': r*r,
                                                 'r_condition_specific': rc}
                                                for n, r, rc in rows],
               'all_six': allsix, 'chimp': chimp},
              open(out_path, 'w') if out_path != '/dev/stdout' else sys.stdout, indent=1)


if __name__ == '__main__':
    main()
