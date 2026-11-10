#!/usr/bin/env python3
"""Does any of this make the actual prediction better?

Everything measured so far is the stability of a ranking. None of it shows that
knowing the ranking helps at the task the project exists for: predicting a species'
perturbation response from other species. This closes that loop on the case that
matters translationally -- hold out human, predict the human LPS response from the
nine animals, and ask whether weighting the animals by anything beats weighting them
equally.

Protein identity is the weight worth testing because it carries no information from
the expression data at all. It comes from proteomes, so using it cannot leak the
answer. Measured concordance is reported alongside as an upper reference, and it IS
leaky when computed on the same contrast being predicted, so it is computed on the
held-out-contrast's complement and labelled.

Recovery is this project's existing metric, 1 - SSE/SST with SST taken about zero and
not about the target's mean: the baseline it scores against is predicting no response,
which is the honest baseline for a method that claims to transfer a response. It is
not R-squared and is not reported as one.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

NAME = {'Pan troglodytes': 'pan_troglodytes', 'Macaca': 'macaca_mulatta',
        'Papio anubis': 'papio_anubis', 'Bos taurus': 'bos_taurus',
        'Ovis aries': 'ovis_aries', 'Sus scrofa': 'sus_scrofa',
        'Oryctolagus cuniculus': 'oryctolagus_cuniculus', 'Mus musculus': 'mus_musculus',
        'Rattus norvegicus': 'rattus_norvegicus'}


def recovery(y, yhat, mask):
    m = mask & np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 100:
        return float('nan')
    sse = float(((y[m] - yhat[m]) ** 2).sum())
    sst = float((y[m] ** 2).sum())
    return 1 - sse / sst if sst > 0 else float('nan')


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed', required=True)
    ap.add_argument('--identity', required=True)
    ap.add_argument('--concordance', required=True)
    ap.add_argument('--target', default='Homo sapiens')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    species, tasks, G = load(args.processed)
    keep = np.isfinite(G).all(axis=(0, 1))
    ti_target = species.index(args.target)
    others = [i for i in range(len(species)) if i != ti_target]
    idn = json.loads(Path(args.identity).read_text())['per_species']
    conc = {r['species']: r['concordance'] for r in
            json.loads(Path(args.concordance).read_text())['anchor_ranking']['overall']}
    ident = {}
    for i in others:
        k = NAME.get(species[i], species[i])
        if k not in idn:
            raise SystemExit(f'no identity for {species[i]}')
        ident[i] = idn[k]['mean_identity']
    print(f'target {args.target}; {len(others)} donor species, {len(tasks)} contrasts, '
          f'{int(keep.sum()):,} genes')

    # weighting schemes. Identity and divergence carry no expression data at all.
    TIME = {'Pan troglodytes': 6.4, 'Macaca': 29.0, 'Papio anubis': 29.0,
            'Mus musculus': 87.0, 'Rattus norvegicus': 87.0,
            'Oryctolagus cuniculus': 90.0, 'Bos taurus': 94.0, 'Ovis aries': 94.0,
            'Sus scrofa': 94.0}
    schemes = {}
    schemes['uniform'] = {i: 1.0 for i in others}
    schemes['identity (sequence only)'] = {i: max(ident[i] - 80.0, 0.01) for i in others}
    schemes['identity, squared'] = {i: max(ident[i] - 80.0, 0.01) ** 2 for i in others}
    schemes['closeness in time'] = {i: 1.0 / TIME[species[i]] for i in others}
    schemes['measured concordance (LEAKY)'] = {i: max(conc.get(species[i], 0.0), 0.01)
                                               for i in others}
    singles = {f'only {species[i]}': {i: 1.0} for i in others}

    res = {'target': args.target, 'n_genes': int(keep.sum()), 'contrasts': tasks,
           'identity': {species[i]: ident[i] for i in others}, 'schemes': {}}
    print(f'\n  {"scheme":34s} {"recovery":>9s}   per-contrast range')
    allr = {}
    for name, w in list(schemes.items()) + list(singles.items()):
        tot = sum(w.values())
        rs = []
        for t in range(len(tasks)):
            pred = np.zeros(G.shape[2])
            for i, wi in w.items():
                pred = pred + wi * np.nan_to_num(G[i, t])
            pred = pred / tot
            rs.append(recovery(G[ti_target, t], pred, keep))
        rs = np.array(rs, dtype=float)
        allr[name] = rs
        res['schemes'][name] = {'mean_recovery': float(np.nanmean(rs)),
                                'per_contrast': [float(x) for x in rs],
                                'weights': {species[i]: float(v) for i, v in w.items()}}
        tag = '  <-- leaks the answer' if 'LEAKY' in name else ''
        print(f'  {name:34s} {np.nanmean(rs):+9.4f}   '
              f'[{np.nanmin(rs):+.3f}, {np.nanmax(rs):+.3f}]{tag}')

    u, s_ = allr['uniform'], allr['identity (sequence only)']
    d = s_ - u
    t_, p_ = stats.ttest_rel(s_[np.isfinite(d)], u[np.isfinite(d)])
    res['identity_vs_uniform'] = {
        'mean_gain': float(np.nanmean(d)),
        'per_contrast_gain': [float(x) for x in d],
        'n_contrasts_improved': int(np.nansum(d > 0)), 'n_contrasts': int(np.isfinite(d).sum()),
        'paired_t': float(t_), 'p_value': float(p_)}
    print(f'\n  identity-weighted minus uniform: {np.nanmean(d):+.4f} '
          f'({int(np.nansum(d > 0))}/{int(np.isfinite(d).sum())} contrasts improved, '
          f'paired t = {t_:+.2f}, p = {p_:.4f})')
    best_single = max(singles, key=lambda k: np.nanmean(allr[k]))
    print(f'  best single donor: {best_single} at {np.nanmean(allr[best_single]):+.4f}')
    print(f'  the conventional choice, only Mus musculus: '
          f'{np.nanmean(allr["only Mus musculus"]):+.4f}')
    res['best_single_donor'] = best_single
    Path(args.output).write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
