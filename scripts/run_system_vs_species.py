#!/usr/bin/env python3
"""Compare human/rat hepatocytes and rat hepatocytes/liver on a shared gene axis.

These descriptive contrasts match the High dose label, not actual concentrations or
intracellular exposure. They do not isolate a causal species or system effect. See
audit_dose_and_multiplicity.py for the concentration-matched sensitivity analysis.

Three statistics are reported for each comparison. Only gene-level Pearson fractions
are used for the manuscript's classical attenuation interpretation. Normalising the
direction and aggregated scores does not itself establish a calibrated ceiling.

  gene level   Pearson across genes
  direction    fraction of responding genes that move the same way
  aggregated   Pearson across GO biological process terms

Cross-side values average the four replicate pairings. The Pearson denominator uses
within-side replicate correlations. A ceiling interpretation additionally requires
independent errors, equal replicate precision within each side and one common measured
gene population. This archived implementation masks each correlation separately;
its ratios are descriptive scores, not verified fractions of an attainable bound.
See audit_replicate_normalisation.py for exact independent-error counterexamples.
"""
import argparse, collections, itertools, json, math, re
import numpy as np

NORMALISATION_NOTE = (
    "NOTE: replicate-normalised scores are descriptive, not verified noise ceilings. "
    "Pairwise gene masks and unequal replicate precision can invalidate that interpretation."
)

ENS_VERSIONED = re.compile(r'^(ENS[A-Z]*G\d+)\.\d+$')
SALT = re.compile(r',?\s*\b(hydrochloride|hydrobromide|sulfate|sulphate|sodium|potassium|'
                  r'calcium|maleate|mesylate|citrate|tartrate|acetate|phosphate|nitrate|'
                  r'succinate|fumarate|besylate|tosylate|monohydrate|dihydrate|trihydrate)\b',
                  re.I)
PAREN = re.compile(r'\([^)]*\)')
NONWORD = re.compile(r'[^a-z0-9]')


def bare(g):
    m = ENS_VERSIONED.match(g.strip())
    return m.group(1) if m else g.strip()


def normalise(name):
    s = name.strip().strip('"').lower()
    s = PAREN.sub(' ', s).split(';')[0]
    prev = None
    while prev != s:
        prev = s
        s = SALT.sub(' ', s)
    return NONWORD.sub('', s.replace('&', ' and '))


def pearson(x, y, min_n=200):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return float('nan')
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 0 else float('nan')


def direction(x, y, thr=0.5, min_n=50):
    """Agreement in sign, among genes that clear the threshold on at least one side.
    Chance is 0.5, so the reported value is rescaled to 0 at chance."""
    m = np.isfinite(x) & np.isfinite(y) & ((np.abs(x) >= thr) | (np.abs(y) >= thr))
    if m.sum() < min_n:
        return float('nan')
    agree = (np.sign(x[m]) == np.sign(y[m])).mean()
    return float(2 * agree - 1)


def aggregate(vec, term_rows):
    out = np.full(len(term_rows), np.nan)
    for i, idx in enumerate(term_rows):
        v = vec[idx]
        v = v[np.isfinite(v)]
        if v.size >= 5:
            out[i] = v.mean()
    return out


def load(path, species_key, axis):
    z = np.load(path, allow_pickle=False)
    want = [bare(p[species_key]) for p in axis]
    got = [str(g) for g in z['genes']]
    if got != want:
        bad = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)
        raise SystemExit(f'{path} is not in axis order at position {bad}: {got[bad]} vs {want[bad]}')
    return {'values': z['values'],
            'index': {(normalise(str(c)), str(d), str(t)): i
                      for i, (c, d, t) in enumerate(zip(z['compounds'], z['doses'], z['times']))}}


def compare(A, B, dose_a, time_a, dose_b, time_b, label, term_rows, min_rel):
    ca = {k[0] for k in A['index'] if k[1] == dose_a and k[2] == time_a}
    cb = {k[0] for k in B['index'] if k[1] == dose_b and k[2] == time_b}
    shared = sorted(ca & cb)
    stats = {'label': label, 'a': f'{dose_a}/{time_a}', 'b': f'{dose_b}/{time_b}',
             'n_compounds_shared': len(shared), 'per_compound': []}
    for c in shared:
        a = A['values'][:, A['index'][(c, dose_a, time_a)], :]
        b = B['values'][:, B['index'][(c, dose_b, time_b)], :]
        na = min(2, a.shape[1])
        nb = min(2, b.shape[1])
        if na < 2 or nb < 2:
            continue
        rec = {'compound': c}
        for name, fn in (('gene', pearson), ('direction', direction)):
            rel_a = fn(a[:, 0], a[:, 1])
            rel_b = fn(b[:, 0], b[:, 1])
            cross = [fn(a[:, i], b[:, j]) for i, j in itertools.product(range(na), range(nb))]
            cross = [x for x in cross if np.isfinite(x)]
            obs = float(np.mean(cross)) if cross else float('nan')
            ok = rel_a >= min_rel and rel_b >= min_rel
            rec[name] = {'rel_a': rel_a, 'rel_b': rel_b, 'obs': obs,
                         'bound': math.sqrt(rel_a * rel_b) if ok else None,
                         'frac': obs / math.sqrt(rel_a * rel_b) if ok and np.isfinite(obs) else None}
        if term_rows:
            fa = [aggregate(a[:, i], term_rows) for i in range(na)]
            fb = [aggregate(b[:, j], term_rows) for j in range(nb)]
            rel_a = pearson(fa[0], fa[1], min_n=30)
            rel_b = pearson(fb[0], fb[1], min_n=30)
            cross = [pearson(fa[i], fb[j], min_n=30) for i, j in itertools.product(range(na), range(nb))]
            cross = [x for x in cross if np.isfinite(x)]
            obs = float(np.mean(cross)) if cross else float('nan')
            ok = rel_a >= min_rel and rel_b >= min_rel
            rec['aggregated'] = {'rel_a': rel_a, 'rel_b': rel_b, 'obs': obs,
                                 'bound': math.sqrt(rel_a * rel_b) if ok else None,
                                 'frac': obs / math.sqrt(rel_a * rel_b) if ok and np.isfinite(obs) else None}
        stats['per_compound'].append(rec)
    return stats


def summarise(stats, keys):
    print(f"\n=== {stats['label']}  ({stats['a']} vs {stats['b']}) ===")
    print(f"  compounds in common: {stats['n_compounds_shared']}, "
          f"with two replicates each side: {len(stats['per_compound'])}")
    for k in keys:
        fr = [r[k]['frac'] for r in stats['per_compound'] if r.get(k, {}).get('frac') is not None]
        ob = [r[k]['obs'] for r in stats['per_compound'] if r.get(k, {}).get('frac') is not None]
        bd = [r[k]['bound'] for r in stats['per_compound'] if r.get(k, {}).get('frac') is not None]
        if not fr:
            print(f'  {k:11s} no compound passed the reliability floor'); continue
        fr, ob, bd = np.array(fr), np.array(ob), np.array(bd)
        q = np.percentile(fr, [25, 50, 75]) * 100
        print(f'  {k:11s} n={len(fr):3d}  observed {np.median(ob):+.3f}  ceiling {np.median(bd):.3f}  '
              f'| % of ceiling  q1 {q[0]:5.1f}  median {q[1]:5.1f}  q3 {q[2]:5.1f}   '
              f'above ceiling {(fr > 1).sum()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-hep', required=True)
    ap.add_argument('--rat-hep', required=True)
    ap.add_argument('--rat-vivo', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--go', help='gene -> GO BP term table, two columns, tab separated')
    ap.add_argument('--min-rel', type=float, default=0.10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    print(NORMALISATION_NOTE)

    axis = json.load(open(a.axis))['axis']
    H = load(a.human_hep, 'homo_sapiens', axis)
    R = load(a.rat_hep, 'rattus_norvegicus', axis)
    V = load(a.rat_vivo, 'rattus_norvegicus', axis)
    print(f'axis verified for all three files: {len(axis):,} ortholog pairs')

    term_rows = []
    if a.go:
        human_ids = [bare(p['homo_sapiens']) for p in axis]
        pos = {g: i for i, g in enumerate(human_ids)}
        by_term = collections.defaultdict(list)
        for line in open(a.go, encoding='utf-8', errors='replace'):
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            i = pos.get(bare(parts[0]))
            if i is not None:
                by_term[parts[1]].append(i)
        term_rows = [np.array(sorted(set(v))) for v in by_term.values() if 5 <= len(set(v)) <= 500]
        print(f'GO terms usable on the axis: {len(term_rows):,} '
              f'(median {int(np.median([len(t) for t in term_rows]))} genes)')

    keys = ['gene', 'direction'] + (['aggregated'] if term_rows else [])
    out = {}
    out['species_changed'] = compare(H, R, 'High', '24 hr', 'High', '24 hr',
                                     'species changed, system held: human hepatocyte vs rat hepatocyte',
                                     term_rows, a.min_rel)
    summarise(out['species_changed'], keys)
    for vt in ('4 day', '29 day'):
        k = f'system_changed_{vt.replace(" ", "")}'
        out[k] = compare(R, V, 'High', '24 hr', 'High', vt,
                         f'system changed, species held: rat hepatocyte vs rat liver in vivo ({vt})',
                         term_rows, a.min_rel)
        summarise(out[k], keys)
        k2 = f'both_changed_{vt.replace(" ", "")}'
        out[k2] = compare(H, V, 'High', '24 hr', 'High', vt,
                          f'both changed: human hepatocyte vs rat liver in vivo ({vt})',
                          term_rows, a.min_rel)
        summarise(out[k2], keys)

    json.dump(out, open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
