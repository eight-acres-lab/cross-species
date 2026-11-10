#!/usr/bin/env python3
"""Does the human reconstruction predict the CONDITION, or only the average response?

Holding out human and predicting its LPS response from nine animals gives recovery +0.479
for chimpanzee alone. That number is only useful if it reflects the condition -- this dose,
this timepoint -- rather than the generic shape of an endotoxin response that any mammal
would produce under any condition.

Three tests, each a different way of taking the generic part away:

  mismatched   predict human at condition t from the donor's response at a DIFFERENT
               condition t'. If this scores as well as the matched prediction, the matched
               number was never about the condition.
  self-mean    predict human at condition t from HUMAN's own mean over the other conditions.
               No animal at all. This is the baseline a donor species has to beat, and
               nothing in the project has ever reported it.
  residual     subtract each species' across-condition mean from both sides, then predict.
               What is left is the part specific to the condition.

Recovery keeps the project's definition, 1 - SSE/sum(y^2) about zero rather than about the
target's mean, so zero means "predicted no response". Under the residual test the target is
itself a residual, so the denominator changes with it; that row is comparable within its own
block and not against the raw block.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed', required=True)
    ap.add_argument('--target', default='Homo sapiens')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    species, tasks, G = load(a.processed)
    keep = np.isfinite(G).all(axis=(0, 1))
    ti = species.index(a.target)
    others = [i for i in range(len(species)) if i != ti]
    nT = len(tasks)
    print(f'target {a.target}; {len(others)} donors; {nT} conditions; '
          f'{int(keep.sum()):,} genes finite everywhere')
    print(f'conditions: {tasks}')

    donors = {'uniform (all 9)': others}
    for i in others:
        donors[f'only {species[i]}'] = [i]

    def pred_matched(ds, t):
        return np.mean([np.nan_to_num(G[i, t]) for i in ds], axis=0)

    res = {'target': a.target, 'conditions': tasks, 'n_genes': int(keep.sum()), 'blocks': {}}

    print(f'\n{"":34s} {"matched":>9s} {"mismatched":>11s} {"gap":>8s}')
    print('  how much of the matched score survives using a DIFFERENT condition')
    block = {}
    for name, ds in donors.items():
        m_scores, x_scores = [], []
        for t in range(nT):
            m_scores.append(recovery(G[ti, t], pred_matched(ds, t), keep))
            for tp in range(nT):
                if tp != t:
                    x_scores.append(recovery(G[ti, t], pred_matched(ds, tp), keep))
        m = float(np.nanmean(m_scores))
        x = float(np.nanmedian(x_scores))
        block[name] = {'matched': m, 'mismatched_median': x, 'gap': m - x}
        print(f'  {name:32s} {m:+9.4f} {x:+11.4f} {m - x:+8.4f}')
    res['blocks']['raw'] = block

    print(f'\n  the baseline no one has reported: human predicting itself, no animal at all')
    self_scores = []
    for t in range(nT):
        rest = [tp for tp in range(nT) if tp != t]
        with np.errstate(invalid='ignore'):
            pred = np.nanmean(np.stack([G[ti, tp] for tp in rest]), axis=0)
        self_scores.append(recovery(G[ti, t], np.nan_to_num(pred), keep))
    self_mean = float(np.nanmean(self_scores))
    print(f'  {"human mean over other conditions":32s} {self_mean:+9.4f}   '
          f'[{np.nanmin(self_scores):+.3f}, {np.nanmax(self_scores):+.3f}]')
    res['self_mean_recovery'] = self_mean
    res['self_mean_per_condition'] = [float(x) for x in self_scores]

    print(f'\n  after removing each species mean across conditions (condition-specific part only)')
    with np.errstate(invalid='ignore'):
        Gc = G - np.nanmean(G, axis=1, keepdims=True)
    blockr = {}
    for name, ds in donors.items():
        rs = []
        for t in range(nT):
            pred = np.mean([np.nan_to_num(Gc[i, t]) for i in ds], axis=0)
            rs.append(recovery(Gc[ti, t], pred, keep))
        v = float(np.nanmean(rs))
        blockr[name] = {'residual_recovery': v}
        print(f'  {name:32s} {v:+9.4f}')
    res['blocks']['residual'] = blockr

    json.dump(res, open(a.output, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
