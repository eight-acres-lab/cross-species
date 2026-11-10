#!/usr/bin/env python3
"""Per-compound rat-human concordance in hepatocytes, against the rat's own in vivo ALT.

Everything here is label free: the predictor is cross-species agreement measured on the
one-to-one ortholog axis, and the outcome is serum ALT measured in the same rats whose
livers were arrayed. No human DILI annotation is involved.

The statistic is agreement expressed as a fraction of its own attenuation bound,
sqrt(rel_human * rel_rat), because a frank toxicant produces a large reproducible
response and would score high on raw correlation for reasons that have nothing to do
with whether the species agree.

This docstring used to claim that dividing by the bound 'removes exactly that'. It does
not, and the claim was false as written. Across the 129 usable compounds the ratio still
tracks the ceiling at Spearman +0.68 and the response amplitude at +0.59; inside the single
High/24h cell, splitting compounds into amplitude thirds gives 7.0%, 15.6% and 23.8% of
ceiling. A median over compounds is therefore a mixed statistic, and comparisons between
rows whose ceiling distributions differ have to be made inside a fixed ceiling band.

A compound with no response has a reliability near zero, which makes the ratio explode.
Such compounds are reported separately rather than averaged in.
"""
import argparse, json, math, re
import numpy as np

ENS_VERSIONED = re.compile(r'^(ENS[A-Z]*G\d+)\.\d+$')


def bare(gene_id):
    m = ENS_VERSIONED.match(gene_id.strip())
    return m.group(1) if m else gene_id.strip()


SALT = re.compile(r',?\s*\b(hydrochloride|hydrobromide|sulfate|sulphate|sodium|potassium|'
                  r'calcium|maleate|mesylate|citrate|tartrate|acetate|phosphate|nitrate|'
                  r'succinate|fumarate|besylate|tosylate|monohydrate|dihydrate|trihydrate)\b',
                  re.I)
PAREN = re.compile(r'\([^)]*\)')
NONWORD = re.compile(r'[^a-z0-9]')


def normalise(name):
    s = name.strip().strip('"').lower()
    s = PAREN.sub(' ', s)
    s = s.split(';')[0]
    prev = None
    while prev != s:
        prev = s
        s = SALT.sub(' ', s)
    return NONWORD.sub('', s.replace('&', ' and '))


def corr(a, b, kind='pearson', min_n=200):
    """min_n defaults to the gene-vector case; comparisons over compounds are far
    shorter and must pass their own floor, or every such statistic silently returns nan."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float('nan'), int(m.sum())
    x, y = a[m], b[m]
    if kind == 'spearman':
        x = np.argsort(np.argsort(x)).astype(float)
        y = np.argsort(np.argsort(y)).astype(float)
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return (float(x @ y) / d if d > 0 else float('nan')), int(m.sum())


def rel_k(r1, k):
    """Spearman-Brown: reliability of a mean of k replicates."""
    if not (r1 > 0):
        return float('nan')
    return k * r1 / (1 + (k - 1) * r1)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {'values': z['values'], 'genes': z['genes'],
            'key': [(normalise(str(c)), str(d), str(t))
                    for c, d, t in zip(z['compounds'], z['doses'], z['times'])],
            'raw_compounds': [str(c) for c in z['compounds']]}


def cell_lookup(ds):
    return {k: i for i, k in enumerate(ds['key'])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', required=True)
    ap.add_argument('--rat', required=True)
    ap.add_argument('--alt', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--dose', default='High')
    ap.add_argument('--time', default='24 hr')
    ap.add_argument('--min-rel', type=float, default=0.10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    H, R = load(a.human), load(a.rat)
    # The two files cannot have equal gene labels: each is tagged with its own species'
    # Ensembl ids. What has to hold is that position i in both files is the same ortholog
    # pair, so both label columns are checked against the axis file they were built from.
    axis = json.load(open(a.axis))['axis']
    want_h = [bare(p['homo_sapiens']) for p in axis]
    want_r = [bare(p['rattus_norvegicus']) for p in axis]
    for lbl, got, want in (('human', list(H['genes']), want_h), ('rat', list(R['genes']), want_r)):
        if [str(x) for x in got] != want:
            bad = next((i for i, (g, w) in enumerate(zip(got, want)) if str(g) != w), len(got))
            raise SystemExit(f'{lbl} file is not in axis order: position {bad} holds '
                             f'{got[bad] if bad < len(got) else "<end>"!r}, axis says {want[bad]!r}')
    print(f'axis alignment verified: {len(want_h):,} ortholog pairs, both files in order')
    hi, ri = cell_lookup(H), cell_lookup(R)

    alt_raw = json.load(open(a.alt))
    alt = {normalise(k): v['alt_fold_max'] for k, v in alt_raw.items()}

    # A normaliser that strips counterions can collide two genuinely different compounds
    # ("diethyl maleate" loses "maleate"); a silent merge would average unrelated data.
    for lbl, ds in (('human', H), ('rat', R)):
        seen = {}
        for norm, raw in zip([k[0] for k in ds['key']], ds['raw_compounds']):
            seen.setdefault(norm, set()).add(raw)
        bad = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        if bad:
            raise SystemExit(f'{lbl}: normalisation merged distinct compound names: {bad}')
        print(f'{lbl}: {len(seen):,} compound names, no normalisation collisions')

    hc = {k[0] for k in H['key'] if k[1] == a.dose and k[2] == a.time}
    rc = {k[0] for k in R['key'] if k[1] == a.dose and k[2] == a.time}
    shared = sorted(hc & rc)
    print(f'compounds at {a.dose}/{a.time}: human {len(hc)}, rat {len(rc)}, shared {len(shared)}')
    print(f'of those with a rat in vivo ALT fold change: {sum(1 for c in shared if c in alt)}')

    rows, dropped = [], []
    for c in shared:
        h = H['values'][:, hi[(c, a.dose, a.time)], :2]
        r = R['values'][:, ri[(c, a.dose, a.time)], :2]
        r1h, nh = corr(h[:, 0], h[:, 1])
        r1r, nr = corr(r[:, 0], r[:, 1])
        hm, rm = np.nanmean(h, axis=1), np.nanmean(r, axis=1)
        obs, n = corr(hm, rm)
        obs_s, _ = corr(hm, rm, 'spearman')
        relh, relr = rel_k(r1h, 2), rel_k(r1r, 2)
        rec = {'compound': c, 'n_genes': n, 'r1_human': r1h, 'r1_rat': r1r,
               'rel_human': relh, 'rel_rat': relr, 'obs_pearson': obs,
               'obs_spearman': obs_s,
               'rms_human': float(np.sqrt(np.nanmean(hm ** 2))),
               'rms_rat': float(np.sqrt(np.nanmean(rm ** 2))),
               'alt_fold': alt.get(c)}
        if not (r1h >= a.min_rel and r1r >= a.min_rel):
            rec['why_dropped'] = (f'replicate correlation below {a.min_rel} '
                                  f'(human {r1h:.3f}, rat {r1r:.3f}): no measurable response, '
                                  'so the bound is not estimable')
            dropped.append(rec)
            continue
        bound = math.sqrt(relh * relr)
        rec['bound'] = bound
        rec['frac_of_bound'] = obs / bound
        rows.append(rec)

    print(f'\nusable: {len(rows)}   dropped for unmeasurable response: {len(dropped)}')
    if rows:
        f = np.array([x['frac_of_bound'] for x in rows])
        o = np.array([x['obs_pearson'] for x in rows])
        b = np.array([x['bound'] for x in rows])
        print(f'raw agreement      median {np.median(o):+.3f}  [{o.min():+.3f}, {o.max():+.3f}]')
        print(f'attenuation bound  median {np.median(b):+.3f}  [{b.min():+.3f}, {b.max():+.3f}]')
        print(f'% of bound         median {np.median(f)*100:5.1f}%  '
              f'[{f.min()*100:.1f}%, {f.max()*100:.1f}%]')
        print(f'  above the bound (impossible, diagnoses shared structure): '
              f'{(f > 1).sum()}/{len(f)}')

        have = [x for x in rows if x['alt_fold'] is not None and x['alt_fold'] > 0]
        print(f'\n--- relation to rat in vivo ALT (n={len(have)}) ---')
        if len(have) >= 12:
            fr = np.array([x['frac_of_bound'] for x in have])
            lal = np.log2(np.array([x['alt_fold'] for x in have]))
            mag = np.log2(np.array([max(x['rms_human'], 1e-6) for x in have]))
            rawo = np.array([x['obs_pearson'] for x in have])
            for lbl, v in (('raw agreement', rawo), ('% of bound', fr),
                           ('human response magnitude', mag),
                           ('rat response magnitude',
                            np.log2(np.array([max(x['rms_rat'], 1e-6) for x in have]))),
                           ('attenuation bound', np.array([x['bound'] for x in have]))):
                rs, nn = corr(v, lal, 'spearman', min_n=12)
                print(f'  spearman({lbl:26s}, log2 ALT fold) = {rs:+.3f}  (n={nn})')
            # response magnitude is the obvious confound: a frank toxicant moves more
            # genes, which raises reliability and therefore raw agreement. Rank-residualise
            # both sides on magnitude before asking whether agreement carries any signal.
            def rank(v):
                return np.argsort(np.argsort(v)).astype(float)
            def resid(y, x):
                x = rank(x) - rank(x).mean()
                y = rank(y) - rank(y).mean()
                return y - x * (float(x @ y) / float(x @ x))
            for lbl, v in (('raw agreement', rawo), ('% of bound', fr)):
                rs, nn = corr(resid(v, mag), resid(lal, mag), 'pearson', min_n=12)
                print(f'  partial, magnitude removed({lbl:14s}, log2 ALT) = {rs:+.3f}  (n={nn})')
            pos = fr[lal >= 1]
            neg = fr[lal < 1]
            if len(pos) >= 3 and len(neg) >= 3:
                print(f'  compounds with ALT >=2x: n={len(pos)} median % of bound '
                      f'{np.median(pos)*100:.1f}%')
                print(f'  compounds with ALT < 2x: n={len(neg)} median % of bound '
                      f'{np.median(neg)*100:.1f}%')
        else:
            print('  too few compounds with both measurements to say anything')

        print('\n--- the 12 compounds that transfer best ---')
        for x in sorted(rows, key=lambda z: -z['frac_of_bound'])[:12]:
            al = f"{x['alt_fold']:.2f}x" if x['alt_fold'] else '   -  '
            print(f"  {x['frac_of_bound']*100:5.1f}% of bound  obs {x['obs_pearson']:+.3f}  "
                  f"bound {x['bound']:.3f}  ALT {al:>7s}  n={x['n_genes']:5d}  {x['compound'][:26]}")
        print('--- the 12 that transfer worst ---')
        for x in sorted(rows, key=lambda z: z['frac_of_bound'])[:12]:
            al = f"{x['alt_fold']:.2f}x" if x['alt_fold'] else '   -  '
            print(f"  {x['frac_of_bound']*100:5.1f}% of bound  obs {x['obs_pearson']:+.3f}  "
                  f"bound {x['bound']:.3f}  ALT {al:>7s}  n={x['n_genes']:5d}  {x['compound'][:26]}")

    json.dump({'dose': a.dose, 'time': a.time, 'min_rel': a.min_rel,
               'usable': rows, 'dropped': dropped},
              open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
