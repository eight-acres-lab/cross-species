#!/usr/bin/env python3
"""Does the ranking of which species resemble which transfer across contexts?

The gene-level finding -- that which genes carry a cross-species shared response is a
property of the contrast, not of the gene -- does not by itself say anything about the
question a toxicologist actually asks, which is "which animal is the best model here".
Those are different objects. The gene ranking lives in thousands of dimensions; the
species-concordance ranking lives in the C(n,2) species pairs, tens of dimensions. A
low-dimensional summary can be stable where its high-dimensional substrate is not, so
this has to be measured rather than inferred, and it could come out either way.

The measurement. For one contrast t, the concordance of species i and j is the
correlation across genes of their two response vectors -- the standard cross-species
concordance in this literature. For a window of contrasts it is the Fisher-z mean over
the window. Comparing two disjoint windows then asks whether the same species look
alike in both.

Two readouts, because they answer different questions:
  * the full pair vector, which asks whether the whole similarity structure holds;
  * the ranking of species by resemblance to one anchor, which is the preclinical
    species-selection question in its literal form.

Everything is scored against its own attenuation bound, measured from replicates, so a
low number cannot be a noise artefact. Species are never resampled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def fisher_mean(rs):
    rs = np.asarray([r for r in rs if np.isfinite(r)], dtype=float)
    if rs.size == 0:
        return float('nan')
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


def pair_vector(G, cols, keep):
    """Concordance of every species pair, averaged over the chosen contrasts."""
    n = G.shape[0]
    out = []
    for i, j in itertools.combinations(range(n), 2):
        rs = []
        for t in cols:
            a, b = G[i, t], G[j, t]
            m = keep & np.isfinite(a) & np.isfinite(b)
            if m.sum() > 100 and np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0:
                rs.append(stats.spearmanr(a[m], b[m]).statistic)
        out.append(fisher_mean(rs))
    return np.asarray(out, dtype=float)


def anchor_vector(G, cols, keep, anchor):
    """Concordance of every other species with one anchor species."""
    n = G.shape[0]
    out = []
    for j in range(n):
        if j == anchor:
            continue
        rs = []
        for t in cols:
            a, b = G[anchor, t], G[j, t]
            m = keep & np.isfinite(a) & np.isfinite(b)
            if m.sum() > 100 and np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0:
                rs.append(stats.spearmanr(a[m], b[m]).statistic)
        out.append(fisher_mean(rs))
    return np.asarray(out, dtype=float)


def agree(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return float('nan')
    return float(stats.spearmanr(x[m], y[m]).statistic)


def rel_k(r1, k):
    return float('nan') if not np.isfinite(r1) or r1 <= 0 else k * r1 / (1 + (k - 1) * r1)


def load(processed, responses='responses.npy', symbol_map=None, key='symbol'):
    d = Path(processed)
    rows = list(csv.DictReader((d / 'response_metadata.tsv').open(), delimiter='\t'))
    data = np.load(d / responses)
    species = sorted({r['species'] for r in rows})
    tasks = sorted({r['task_id'] for r in rows})
    tasks = [t for t in tasks
             if all(any(r['species'] == s and r['task_id'] == t for r in rows) for s in species)]
    G = np.full((len(species), len(tasks), data.shape[1]), np.nan)
    reps = defaultdict(list)
    for si, s in enumerate(species):
        for ti, t in enumerate(tasks):
            ix = [i for i, r in enumerate(rows) if r['species'] == s and r['task_id'] == t]
            ix.sort(key=lambda i: rows[i].get('subject') or rows[i].get('sample_id') or '')
            if ix:
                with np.errstate(invalid='ignore'):
                    G[si, ti] = np.nanmean(data[ix], axis=0)
                reps[(si, ti)] = ix
    return species, tasks, G, data, reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, choices=['selenite', 'blood', 'hepatocyte'])
    ap.add_argument('--processed', required=True)
    ap.add_argument('--anchor', default=None,
                    help='species to rank the others against; the preclinical question')
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    species, tasks, G, data, reps = load(args.processed)
    keep = np.isfinite(G).all(axis=(0, 1))
    print(f'{args.panel}: {len(species)} species x {len(tasks)} contrasts, '
          f'{int(keep.sum()):,} complete genes')
    print(f'  species: {species}')

    # replicate grids, for the attenuation bound
    singles = []
    if (Path(args.processed) / 'per_array.npz').exists():
        pa = np.load(Path(args.processed) / 'per_array.npz')
        meta = list(csv.DictReader((Path(args.processed) / 'response_metadata.tsv').open(),
                                   delimiter='\t'))
        tmin = {m['task_id']: m['timepoint_min'] for m in meta}
        n_rep = min(pa[k].shape[0] for k in pa.files)
        for r in range(n_rep):
            singles.append(np.stack([np.stack([pa[f'{s}|{tmin[t]}'][r] for t in tasks])
                                     for s in species]))
    else:
        n_rep = min((len(v) for v in reps.values()), default=0)
        n_rep = min(n_rep, 4)
        for r in range(n_rep):
            g = np.full_like(G, np.nan)
            for (si, ti), ix in reps.items():
                if r < len(ix):
                    g[si, ti] = data[ix[r]]
            singles.append(g)
    print(f'  {n_rep} replicates per cell for the bound')

    anchor = species.index(args.anchor) if args.anchor and args.anchor in species else None
    if args.anchor and anchor is None:
        raise SystemExit(f'anchor {args.anchor!r} not among {species}')

    W = min(args.window, len(tasks) // 2)
    wins = list(itertools.combinations(range(len(tasks)), W))
    vec, avec, relv, rela = {}, {}, {}, {}
    for w in wins:
        vec[w] = pair_vector(G, list(w), keep)
        sv = [pair_vector(s, list(w), keep) for s in singles]
        rr = [agree(sv[i], sv[j]) for i, j in itertools.combinations(range(len(sv)), 2)]
        relv[w] = rel_k(fisher_mean(rr), len(sv))
        if anchor is not None:
            avec[w] = anchor_vector(G, list(w), keep, anchor)
            sa = [anchor_vector(s, list(w), keep, anchor) for s in singles]
            ra = [agree(sa[i], sa[j]) for i, j in itertools.combinations(range(len(sa)), 2)]
            rela[w] = rel_k(fisher_mean(ra), len(sa))

    pairs = []
    for a, b in itertools.combinations(wins, 2):
        if set(a) & set(b):
            continue
        row = {'window_a': [tasks[i] for i in a], 'window_b': [tasks[i] for i in b],
               'pair_spearman': agree(vec[a], vec[b]),
               'pair_bound': float(np.sqrt(max(relv[a], 0) * max(relv[b], 0)))}
        row['pair_percent_of_bound'] = (100 * row['pair_spearman'] / row['pair_bound']
                                        if row['pair_bound'] > 0 else None)
        if anchor is not None:
            row['anchor_spearman'] = agree(avec[a], avec[b])
            row['anchor_bound'] = float(np.sqrt(max(rela[a], 0) * max(rela[b], 0)))
            row['anchor_percent_of_bound'] = (100 * row['anchor_spearman'] / row['anchor_bound']
                                              if row['anchor_bound'] > 0 else None)
        pairs.append(row)

    def summarise(field, bfield):
        v = [p[field] for p in pairs if p.get(field) is not None and np.isfinite(p[field])]
        b = [p[bfield] for p in pairs if p.get(bfield) is not None and np.isfinite(p[bfield])]
        return (float(np.mean(v)) if v else float('nan'),
                float(np.min(v)) if v else float('nan'),
                float(np.max(v)) if v else float('nan'),
                float(np.mean(b)) if b else float('nan'), len(v))

    out = {'panel': args.panel, 'species': species, 'contrasts': tasks,
           'n_genes': int(keep.sum()), 'window': W, 'n_disjoint_pairs': len(pairs),
           'n_replicates': n_rep, 'anchor': args.anchor, 'pairs': pairs}
    m, lo, hi, bd, n = summarise('pair_percent_of_bound', 'pair_bound')
    rm, *_ = summarise('pair_spearman', 'pair_bound')
    out['pair_structure'] = {'mean_percent_of_bound': m, 'min': lo, 'max': hi,
                             'mean_bound': bd, 'n': n, 'mean_spearman': rm,
                             'n_species_pairs': len(list(itertools.combinations(species, 2)))}
    print(f"\nwhole similarity structure ({out['pair_structure']['n_species_pairs']} species pairs)")
    print(f'  mean agreement {rm:+.4f}  bound {bd:.4f}  '
          f'{m:.1f}% of bound  (range {lo:.1f}-{hi:.1f}%, n={n})')
    if anchor is not None:
        m2, lo2, hi2, bd2, n2 = summarise('anchor_percent_of_bound', 'anchor_bound')
        rm2, *_ = summarise('anchor_spearman', 'anchor_bound')
        out['anchor_ranking'] = {'mean_percent_of_bound': m2, 'min': lo2, 'max': hi2,
                                 'mean_bound': bd2, 'n': n2, 'mean_spearman': rm2}
        print(f'\nranking of species by resemblance to {args.anchor}')
        print(f'  mean agreement {rm2:+.4f}  bound {bd2:.4f}  '
              f'{m2:.1f}% of bound  (range {lo2:.1f}-{hi2:.1f}%, n={n2})')
        full = anchor_vector(G, list(range(len(tasks))), keep, anchor)
        order = [s for s in species if s != args.anchor]
        rank = sorted(zip(order, full), key=lambda x: -x[1])
        out['anchor_ranking']['overall'] = [{'species': s, 'concordance': float(v)}
                                            for s, v in rank]
        print(f'  pooled over all contrasts, resemblance to {args.anchor}:')
        for s, v in rank:
            print(f'    {s:28s} {v:+.3f}')
        per = {}
        for ti, t in enumerate(tasks):
            av = anchor_vector(G, [ti], keep, anchor)
            per[t] = [order[k] for k in np.argsort(-av)]
        out['anchor_ranking']['per_contrast_order'] = per
        tops = {t: v[0] for t, v in per.items()}
        print(f'  most {args.anchor}-like species, per contrast:')
        for t, s in tops.items():
            print(f'    {t:10s} {s}')
        out['anchor_ranking']['top_per_contrast'] = tops
        out['anchor_ranking']['n_distinct_winners'] = len(set(tops.values()))
        print(f'  distinct winners across {len(tasks)} contrasts: {len(set(tops.values()))}')

    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()
