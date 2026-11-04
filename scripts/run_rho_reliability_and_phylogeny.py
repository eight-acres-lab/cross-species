#!/usr/bin/env python3
"""Does the species-pair redundancy matrix carry anything a phylogeny does not already give?

The business case rests on rho being a measurable property of a context. An earlier check
found the matrix measured in one context predicts another at Spearman +0.55 and +0.60, but
restricting to species pairs in different taxonomic orders left six and four pairs, at which
nothing could be concluded. This widens that test.

Two things make it possible. First, the blood panel subtracts a no-LPS control matched to
each (subject, time), verified by reading the preparation script, so its three timepoints are
control-disjoint and can serve as three contexts over the full ten-species set -- 45 pairs, 35
of them across taxonomic orders. Its dose arms at one timepoint share a control and are never
used as separate contexts.

Second, rho has a measurement error of its own, and until now it has been compared against
nothing. Splitting subjects within one context and measuring rho twice gives that reliability,
so cross-context agreement can be reported as a fraction of what rho's own precision permits,
in the same currency as everything else in this project.

Phylogeny enters two ways: as a hard restriction to pairs in different orders, and as a graded
taxonomic distance partialled out of both sides.

Two corrections after review. The restriction now starts at a taxonomic distance of 4, which
is where different orders begin; an earlier threshold of 3 re-admitted six same-order pairs,
four of them primate-primate and the highest-rho points in the panel. And significance is
assessed by relabelling species rather than by shuffling pair values, because the entries of a
species-by-species matrix are not exchangeable and shuffling them yields a null that is too
narrow.
"""
import argparse, csv, itertools, json, math
from pathlib import Path
import numpy as np

# genus, family, order, superorder -- enough to grade a pair without claiming divergence times
TAX = {
    'Homo sapiens':          ('Homo', 'Hominidae', 'Primates', 'Euarchontoglires'),
    'Pan troglodytes':       ('Pan', 'Hominidae', 'Primates', 'Euarchontoglires'),
    'Macaca':                ('Macaca', 'Cercopithecidae', 'Primates', 'Euarchontoglires'),
    'Macaca fascicularis':   ('Macaca', 'Cercopithecidae', 'Primates', 'Euarchontoglires'),
    'Papio anubis':          ('Papio', 'Cercopithecidae', 'Primates', 'Euarchontoglires'),
    'Mus musculus':          ('Mus', 'Muridae', 'Rodentia', 'Euarchontoglires'),
    'Mus spretus':           ('Mus', 'Muridae', 'Rodentia', 'Euarchontoglires'),
    'Rattus norvegicus':     ('Rattus', 'Muridae', 'Rodentia', 'Euarchontoglires'),
    'Oryctolagus cuniculus': ('Oryctolagus', 'Leporidae', 'Lagomorpha', 'Euarchontoglires'),
    'Bos taurus':            ('Bos', 'Bovidae', 'Artiodactyla', 'Laurasiatheria'),
    'Bubalus bubalis':       ('Bubalus', 'Bovidae', 'Artiodactyla', 'Laurasiatheria'),
    'Capra hircus':          ('Capra', 'Bovidae', 'Artiodactyla', 'Laurasiatheria'),
    'Ovis aries':            ('Ovis', 'Bovidae', 'Artiodactyla', 'Laurasiatheria'),
    'Sus scrofa':            ('Sus', 'Suidae', 'Artiodactyla', 'Laurasiatheria'),
    'Equus caballus':        ('Equus', 'Equidae', 'Perissodactyla', 'Laurasiatheria'),
    'Canis lupus familiaris':('Canis', 'Canidae', 'Carnivora', 'Laurasiatheria'),
}
ALIAS = {'Macaca fascicularis': 'Macaca', 'Macaca mulatta': 'Macaca'}
ENSRE = None


def taxdist(a, b):
    """1 same genus, 2 same family, 3 same order, 4 same superorder, 5 beyond."""
    ta, tb = TAX.get(a), TAX.get(b)
    if ta is None or tb is None:
        return None
    for i in range(4):
        if ta[i] == tb[i]:
            return i + 1
    return 5


SYM2ENS = {}


def load_bridge(path):
    import re
    rd = csv.DictReader(open(path, encoding='utf-8', errors='replace'), delimiter='\t')
    sym_col = next(c for c in rd.fieldnames if 'symbol' in c.lower())
    ens_col = next(c for c in rd.fieldnames if 'ensembl' in c.lower())
    for r in rd:
        sym, ens = (r.get(sym_col) or '').strip(), (r.get(ens_col) or '').strip()
        if not sym or not ens or ens in ('---', 'NA'):
            continue
        ids = {x.split('.')[0] for x in re.split(r'[\s:,;|]+', ens) if x and x not in ('---', 'NA')}
        if len(ids) != 1:
            continue
        g = next(iter(ids))
        for one in re.split(r'\s*///\s*', sym):
            one = one.strip().upper()
            if one and one not in SYM2ENS:
                SYM2ENS[one] = g


def canon(g):
    g = str(g).strip()
    return g.split('.')[0] if g.upper().startswith('ENSG') else SYM2ENS.get(g.upper(), '')


def corr(a, b, min_n=200):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y) / d if d > 0 else np.nan


def centered_ranks(values):
    """Average tied ranks, centred at zero; independent of observation order."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = (starts + (counts - 1) / 2.0)[inverse]
    return ranks - ranks.mean()


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    rx, ry = centered_ranks(x[m]), centered_ranks(y[m])
    d = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / d if d > 0 else np.nan


def partial_spearman(x, y, z):
    """Residualise average-tie ranks of x and y on ranked z, then correlate."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return np.nan
    rx, ry, rz = (centered_ranks(v[m]) for v in (x, y, z))
    if float(rz @ rz) <= 0:
        return spearman(x[m], y[m])
    bx = rx - rz * (float(rz @ rx) / float(rz @ rz))
    by = ry - rz * (float(rz @ ry) / float(rz @ rz))
    d = math.sqrt(float(bx @ bx) * float(by @ by))
    return float(bx @ by) / d if d > 0 else np.nan


def load_blood(d):
    rows = list(csv.DictReader((Path(d) / 'response_metadata.tsv').open(), delimiter='\t'))
    data = np.load(Path(d) / 'responses.npy')
    genes = np.array([canon(g) for g in np.load(Path(d) / 'genes.npy', allow_pickle=True)])
    return rows, data, genes


def species_vectors(rows, data, keyfn, subjects=None):
    """species -> mean response vector over the rows selected by keyfn (and subject filter)."""
    acc = {}
    for i, r in enumerate(rows):
        if not keyfn(r):
            continue
        if subjects is not None and r['subject'] not in subjects:
            continue
        s = ALIAS.get(r['species'], r['species'])
        acc.setdefault(s, []).append(data[i])
    with np.errstate(invalid='ignore'):
        return {s: np.nanmean(np.vstack(v), axis=0) for s, v in acc.items() if v}


def rho_matrix(vecs, gidx=None):
    out = {}
    for a, b in itertools.combinations(sorted(vecs), 2):
        va, vb = vecs[a], vecs[b]
        if gidx is not None:
            va, vb = va[gidx], vb[gidx]
        r = corr(va, vb)
        if np.isfinite(r):
            out[(a, b)] = r
    return out


def compare(m1, m2, label, rng, n_perm=20000):
    pairs = sorted(set(m1) & set(m2))
    if len(pairs) < 5:
        print(f'  {label}: only {len(pairs)} shared pairs, skipped')
        return None
    x = np.array([m1[p] for p in pairs])
    y = np.array([m2[p] for p in pairs])
    td = np.array([taxdist(*p) for p in pairs], dtype=float)
    # td is 1 same genus, 2 same family, 3 same ORDER, 4 same superorder, 5 beyond.
    # An earlier version used td >= 3, which re-admitted the six same-order pairs --
    # four of them primate-primate, the highest-rho points in the panel -- while calling
    # them cross-order. Different order starts at 4.
    cross = td >= 4
    res = {'label': label, 'n_pairs': len(pairs), 'n_cross_order': int(cross.sum())}

    # The 45 entries of a species-by-species matrix are not exchangeable: each species
    # appears in n-1 of them, so shuffling pair values destroys a dependence the null
    # should preserve and returns p-values that are too small. The null here is Mantel's:
    # relabel the species of one matrix and read the same cells, which keeps each matrix's
    # internal structure intact. Restriction and the taxonomic covariate stay on the true
    # labels, because they define the hypothesis rather than the null.
    sp = sorted({s for pr in pairs for s in pr})
    pos = {s: i for i, s in enumerate(sp)}
    n_sp = len(sp)
    pi = np.array([pos[p[0]] for p in pairs])
    pj = np.array([pos[p[1]] for p in pairs])
    M2 = np.full((n_sp, n_sp), np.nan)
    for p_, v in zip(pairs, y):
        M2[pos[p_[0]], pos[p_[1]]] = M2[pos[p_[1]], pos[p_[0]]] = v

    def mantel(stat_fn, obs, mask=None, n=n_perm):
        if not np.isfinite(obs):
            return np.nan
        hits = 0
        for _ in range(n):
            q = rng.permutation(n_sp)
            yp = M2[q[pi], q[pj]]
            v = stat_fn(yp)
            if np.isfinite(v) and v >= obs:
                hits += 1
        # (1 + hits) / (1 + n): a never-zero Monte Carlo estimate. The minimum
        # records simulation resolution; it is not an upper confidence bound.
        return (1.0 + hits) / (1.0 + n)

    res['spearman_all'] = spearman(x, y)
    res['p_all'] = mantel(lambda yp: spearman(x, yp), res['spearman_all'])
    if cross.sum() >= 5:
        res['spearman_cross'] = spearman(x[cross], y[cross])
        res['p_cross'] = mantel(lambda yp: spearman(x[cross], yp[cross]), res['spearman_cross'])
    else:
        res['spearman_cross'] = np.nan; res['p_cross'] = np.nan
    res['partial_taxdist'] = partial_spearman(x, y, td)
    res['p_partial'] = mantel(lambda yp: partial_spearman(x, yp, td),
                              res['partial_taxdist'], n=min(n_perm, 4000))
    res['n_species'] = n_sp
    res['max_distinct_permutations'] = math.factorial(n_sp) if n_sp <= 12 else None
    res['spearman_taxdist_vs_x'] = spearman(td, x)
    print(f'  {label:44s} pairs {len(pairs):3d} (cross-order {int(cross.sum()):3d})  '
          f'all {res["spearman_all"]:+.3f} p={res["p_all"]:.4f}  '
          f'cross {res["spearman_cross"]:+.3f} p={res["p_cross"]:.4f}  '
          f'partial {res["partial_taxdist"]:+.3f} p={res["p_partial"]:.4f}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blood', required=True)
    ap.add_argument('--macrophage')
    ap.add_argument('--hepatocyte')
    ap.add_argument('--bridge', required=True)
    ap.add_argument('--seed', type=int, default=20261022)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    load_bridge(a.bridge)
    rng = np.random.default_rng(a.seed)
    rows, data, genes = load_blood(a.blood)
    times = sorted({r['time_h'] for r in rows}, key=int)
    print(f'blood: {len(rows)} rows, {len(set(r["species"] for r in rows))} species, times {times}')
    out = {'blood_times': times, 'comparisons': [], 'reliability': []}

    # ---- rho per timepoint, and its own reliability from a subject split ----
    per_time, per_time_halves = {}, {}
    for t in times:
        kf = (lambda tt: (lambda r: r['time_h'] == tt))(t)
        per_time[t] = rho_matrix(species_vectors(rows, data, kf))
        subs = {}
        for r in rows:
            if r['time_h'] == t:
                subs.setdefault(ALIAS.get(r['species'], r['species']), set()).add(r['subject'])
        h1, h2 = set(), set()
        for s, v in subs.items():
            v = sorted(v)
            h1 |= set(v[:len(v) // 2]); h2 |= set(v[len(v) // 2:])
        per_time_halves[t] = (rho_matrix(species_vectors(rows, data, kf, h1)),
                              rho_matrix(species_vectors(rows, data, kf, h2)))

    print('\n=== how precisely is rho measured? subject split inside one timepoint ===')
    rel = []
    for t in times:
        r = compare(per_time_halves[t][0], per_time_halves[t][1],
                    f'blood {t}h: half of subjects vs the other half', rng)
        if r:
            r['kind'] = 'reliability'; out['reliability'].append(r); rel.append(r['spearman_all'])
    rel_mean = float(np.nanmean(rel)) if rel else np.nan
    sb = 2 * rel_mean / (1 + rel_mean) if np.isfinite(rel_mean) and rel_mean > -1 else np.nan
    print(f'  mean split-half reliability of rho = {rel_mean:+.3f}; '
          f'Spearman-Brown for the full subject set = {sb:+.3f}')
    out['rho_reliability_halfset'] = rel_mean
    out['rho_reliability_fullset'] = sb

    print('\n=== does rho measured at one timepoint predict another? (10 species, 45 pairs) ===')
    for t1, t2 in itertools.combinations(times, 2):
        r = compare(per_time[t1], per_time[t2], f'blood {t1}h vs {t2}h', rng)
        if r:
            r['kind'] = 'cross_time'; out['comparisons'].append(r)

    # ---- across panels ----
    for name, path, key in (('macrophage', a.macrophage, 'task_id'),
                            ('hepatocyte', a.hepatocyte, 'task_id')):
        if not path:
            continue
        rows2 = list(csv.DictReader((Path(path) / 'response_metadata.tsv').open(), delimiter='\t'))
        d2 = np.load(Path(path) / 'responses.npy')
        g2 = np.array([canon(g) for g in np.load(Path(path) / 'genes.npy', allow_pickle=True)])
        shared = sorted((set(genes) & set(g2)) - {''})
        if len(shared) < 500:
            print(f'\n{name}: only {len(shared)} shared genes, skipped'); continue
        p1 = {}
        for j, x in enumerate(genes):
            if x and x not in p1:
                p1[x] = j
        p2 = {}
        for j, x in enumerate(g2):
            if x and x not in p2:
                p2[x] = j
        i1 = np.array([p1[x] for x in shared]); i2 = np.array([p2[x] for x in shared])
        print(f'\n=== blood vs {name} ({len(shared):,} shared genes) ===')
        # Two estimators of the same quantity, reported together because they disagree and
        # the earlier run used the second one. "pooled" correlates the condition-averaged
        # vectors; "per-condition" averages the correlation computed inside each condition,
        # which keeps each condition's own structure instead of letting the across-condition
        # consensus dominate.
        mb = rho_matrix(species_vectors(rows, data, lambda r: True), i1)
        mo = rho_matrix(species_vectors(rows2, d2, lambda r: True), i2)
        r = compare(mb, mo, f'blood vs {name} [pooled]', rng)
        if r:
            r['kind'] = f'cross_panel_{name}_pooled'; out['comparisons'].append(r)

        def per_condition(rws, dat, gidx):
            conds = sorted({x['task_id'] for x in rws})
            mats = []
            for c in conds:
                v = species_vectors(rws, dat, (lambda cc: (lambda x: x['task_id'] == cc))(c))
                if len(v) >= 2:
                    mats.append(rho_matrix(v, gidx))
            allp = set().union(*[set(m) for m in mats]) if mats else set()
            agg = {}
            for pr in allp:
                vals = [m[pr] for m in mats if pr in m and np.isfinite(m[pr])]
                if vals:
                    agg[pr] = float(np.mean(vals))
            return agg
        mb2 = per_condition(rows, data, i1)
        mo2 = per_condition(rows2, d2, i2)
        r = compare(mb2, mo2, f'blood vs {name} [per-condition]', rng)
        if r:
            r['kind'] = f'cross_panel_{name}_percond'; out['comparisons'].append(r)

    print('\n=== cross-context agreement as a fraction of rho\'s own reliability ===')
    for c in out['comparisons']:
        for k, lbl in (('spearman_all', 'all pairs'), ('spearman_cross', 'cross-order only')):
            v = c.get(k)
            if np.isfinite(v) and np.isfinite(sb) and sb > 0:
                print(f'  {c["label"]:38s} {lbl:17s} {v:+.3f}  = {100*v/sb:5.1f}% of ceiling')
    json.dump(out, open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
