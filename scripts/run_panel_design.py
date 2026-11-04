#!/usr/bin/env python3
"""Which set of animal species carries the most information about human?

The practical decision upstream of any clinic is not "is this model good" but "which
species do I put in the panel". That is a subset-selection question and it is
enumerable: nine donors give 511 non-empty subsets, four give 15.

The measure is how well the human response can be reconstructed from the panel, with
the combination FITTED ON OTHER CONTRASTS and evaluated on a held-out one. Fitting is
the realistic setting -- nobody uses an animal readout uncalibrated -- and
leave-one-contrast-out keeps it honest. This project has shown elsewhere that a single
fitted scalar closes most of the gap for a single donor, so an uncalibrated comparison
would understate every panel and would understate the weak donors most.

Recovery is 1 - SSE/SST with SST about zero, so zero means "predicting no response".

Three things are reported, and the third is the one that would be usable:
  * the best panel of each size, and what the conventional rodent pair achieves;
  * the marginal value of adding each remaining species to a given base panel;
  * how close a panel chosen WITHOUT any human data -- by sequence identity and by
    spreading donors across the identity range -- comes to the oracle best panel.
Only the third is advice. The first two need the human answer to compute and so can
only be read backwards, which is a description of this dataset and not a method.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

NAME = {'Pan troglodytes': 'pan_troglodytes', 'Macaca': 'macaca_mulatta',
        'Macaca fascicularis': 'macaca_fascicularis', 'Papio anubis': 'papio_anubis',
        'Bos taurus': 'bos_taurus', 'Ovis aries': 'ovis_aries', 'Sus scrofa': 'sus_scrofa',
        'Oryctolagus cuniculus': 'oryctolagus_cuniculus',
        'Canis lupus familiaris': 'canis_lupus_familiaris',
        'Mus musculus': 'mus_musculus', 'Rattus norvegicus': 'rattus_norvegicus'}
SHORT = {'Pan troglodytes': 'chimp', 'Macaca': 'macaque', 'Macaca fascicularis': 'cyno',
         'Papio anubis': 'baboon', 'Bos taurus': 'cow', 'Ovis aries': 'sheep',
         'Sus scrofa': 'pig', 'Oryctolagus cuniculus': 'rabbit',
         'Canis lupus familiaris': 'dog', 'Mus musculus': 'mouse',
         'Rattus norvegicus': 'rat'}


def load(processed):
    d = Path(processed)
    rows = list(csv.DictReader((d / 'response_metadata.tsv').open(), delimiter='\t'))
    data = np.load(d / 'responses.npy')
    species = sorted({r['species'] for r in rows})
    tasks = sorted({r['task_id'] for r in rows})
    tasks = [t for t in tasks
             if all(any(r['species'] == s and r['task_id'] == t for r in rows) for s in species)]
    G = np.full((len(species), len(tasks), data.shape[1]), np.nan)
    for si, s in enumerate(species):
        for ti, t in enumerate(tasks):
            ix = [i for i, r in enumerate(rows) if r['species'] == s and r['task_id'] == t]
            if ix:
                with np.errstate(invalid='ignore'):
                    G[si, ti] = np.nanmean(data[ix], axis=0)
    return species, tasks, G


def cv_recovery(G, target, subset, keep, tasks):
    """Leave-one-contrast-out: fit the donor combination on the others, score on it."""
    out = []
    n_t = G.shape[1]
    for held in range(n_t):
        tr = [t for t in range(n_t) if t != held]
        X = np.concatenate([np.stack([G[j, t][keep] for j in subset], axis=1) for t in tr])
        y = np.concatenate([G[target, t][keep] for t in tr])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if ok.sum() < 200:
            out.append(np.nan); continue
        w, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        Xh = np.stack([G[j, held][keep] for j in subset], axis=1)
        yh = G[target, held][keep]
        m = np.isfinite(Xh).all(axis=1) & np.isfinite(yh)
        pred = Xh[m] @ w
        sse = float(((yh[m] - pred) ** 2).sum()); sst = float((yh[m] ** 2).sum())
        out.append(1 - sse / sst if sst > 0 else np.nan)
    return float(np.nanmean(out)), [float(x) for x in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--identity', required=True)
    ap.add_argument('--target', default='Homo sapiens')
    ap.add_argument('--conventional', default='Mus musculus,Rattus norvegicus')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    species, tasks, G = load(args.processed)
    keep = np.isfinite(G).all(axis=(0, 1))
    tgt = species.index(args.target)
    donors = [i for i in range(len(species)) if i != tgt]
    idn = json.loads(Path(args.identity).read_text())['per_species']
    ident = {}
    for i in donors:
        k = NAME.get(species[i], species[i])
        if k in idn:
            ident[i] = idn[k]['mean_identity']
    print(f'{args.label}: target {args.target}, {len(donors)} donors, '
          f'{len(tasks)} contrasts, {int(keep.sum()):,} genes')
    print('  identity to target: ' +
          ', '.join(f'{SHORT.get(species[i], species[i])} {ident[i]:.1f}%'
                    for i in sorted(donors, key=lambda j: -ident.get(j, 0)) if i in ident))

    res = {}
    for r in range(1, len(donors) + 1):
        for sub in itertools.combinations(donors, r):
            m, per = cv_recovery(G, tgt, list(sub), keep, tasks)
            res[sub] = {'mean': m, 'per_contrast': per}
    print(f'  evaluated {len(res)} subsets, cross-validated over contrasts')

    full = tuple(donors)
    out = {'label': args.label, 'target': args.target,
           'donors': [species[i] for i in donors], 'n_contrasts': len(tasks),
           'n_genes': int(keep.sum()),
           'identity': {species[i]: ident.get(i) for i in donors},
           'full_panel_recovery': res[full]['mean'], 'best_by_size': {}, 'subsets': {}}
    for sub, v in res.items():
        out['subsets'][','.join(SHORT.get(species[i], species[i]) for i in sub)] = v['mean']

    print(f'\n  best panel of each size (cross-validated recovery of {args.target}):')
    for r in range(1, len(donors) + 1):
        cands = {s: v['mean'] for s, v in res.items() if len(s) == r and np.isfinite(v['mean'])}
        if not cands:
            continue
        best = max(cands, key=cands.get)
        worst = min(cands, key=cands.get)
        out['best_by_size'][str(r)] = {
            'best': [species[i] for i in best], 'best_recovery': cands[best],
            'worst': [species[i] for i in worst], 'worst_recovery': cands[worst],
            'median_recovery': float(np.median(list(cands.values())))}
        print(f'    k={r}  best {cands[best]:+.4f}  '
              f'{", ".join(SHORT.get(species[i], species[i]) for i in best):34s} '
              f'(median over panels of this size {np.median(list(cands.values())):+.4f}, '
              f'worst {cands[worst]:+.4f})')

    conv = tuple(sorted(species.index(s.strip()) for s in args.conventional.split(',')
                        if s.strip() in species))
    if conv and all(i in donors for i in conv):
        base = res[conv]['mean']
        label = ' + '.join(SHORT.get(species[i], species[i]) for i in conv)
        print(f'\n  the conventional panel, {label}: {base:+.4f}')
        print(f'  what adding one more species buys:')
        gains = []
        for i in donors:
            if i in conv:
                continue
            sub = tuple(sorted(conv + (i,)))
            g = res[sub]['mean'] - base
            gains.append((g, species[i], res[sub]['mean']))
        gains.sort(reverse=True)
        for g, sp, tot in gains:
            print(f'    + {SHORT.get(sp, sp):10s} {g:+.4f}   -> {tot:+.4f}')
        out['conventional_panel'] = {'species': [species[i] for i in conv],
                                     'recovery': base,
                                     'marginal_gain': [{'species': sp, 'gain': g, 'total': tot}
                                                       for g, sp, tot in gains]}
        best3 = max(((v['mean'], s) for s, v in res.items() if len(s) == len(conv) + 1),
                    key=lambda x: x[0])
        print(f'  best panel of that size overall: '
              f'{", ".join(SHORT.get(species[i], species[i]) for i in best3[1])} '
              f'at {best3[0]:+.4f}')

    # a rule that uses no data from the target species
    print(f'\n  panels chosen WITHOUT the target\'s data, against the oracle:')
    rules = {}
    by_id = sorted((i for i in donors if i in ident), key=lambda i: -ident[i])
    rules['top identity'] = lambda r: tuple(sorted(by_id[:r]))

    def spread(r):
        lo, hi = ident[by_id[-1]], ident[by_id[0]]
        picks, used = [], set()
        for q in np.linspace(hi, lo, r):
            j = min((i for i in by_id if i not in used), key=lambda i: abs(ident[i] - q))
            picks.append(j); used.add(j)
        return tuple(sorted(picks))
    rules['spread across identity'] = spread
    for rn, fn in rules.items():
        row = []
        for r in range(1, len(by_id) + 1):
            sub = fn(r)
            if sub not in res:
                continue
            orc = out['best_by_size'].get(str(r), {}).get('best_recovery')
            row.append((r, res[sub]['mean'], orc,
                        ', '.join(SHORT.get(species[i], species[i]) for i in sub)))
        out[f'rule: {rn}'] = [{'k': k, 'recovery': v, 'oracle': o, 'panel': p}
                              for k, v, o, p in row]
        print(f'    {rn}:')
        for k, v, o, p in row:
            frac = (v / o * 100) if o and o > 0 else float('nan')
            print(f'      k={k}  {v:+.4f}  ({frac:5.1f}% of the oracle {o:+.4f})  {p}')

    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
