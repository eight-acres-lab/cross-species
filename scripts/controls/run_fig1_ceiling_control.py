"""Apply this project's own ceiling normalisation to its own Figure 1.

Figure 1 reports a RAW correlation between each donor species' response and the human
response. The project's central method divides such a correlation by the agreement the
data's own replicate structure permits, precisely so that poor transfer and poor
measurement stay apart. Figure 1 does not do this. Since the raw ranking tracks response
amplitude at Spearman +0.93, the question is whether it survives the normalisation.
"""
import csv, itertools, json, numpy as np

D = 'artifacts/gse249010_v1/processed/'
H = 'Homo sapiens'
rng = np.random.default_rng(20261023)

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan

def spearman(a, b):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pear(r(a), r(b))

rows = list(csv.DictReader(open(D + 'response_metadata.tsv'), delimiter='\t'))
X = np.load(D + 'responses.npy')
sp = sorted({r['species'] for r in rows})
tk = sorted({r['task_id'] for r in rows})
tk = [t for t in tk if all(any(r['species'] == s and r['task_id'] == t for r in rows) for s in sp)]

idx = {(s, t): [i for i, r in enumerate(rows) if r['species'] == s and r['task_id'] == t]
       for s in sp for t in tk}
subj = {(s, t): [rows[i]['subject'] for i in idx[(s, t)]] for s in sp for t in tk}
mean = {s: {t: np.nanmean(X[idx[(s, t)]], axis=0) for t in tk} for s in sp}

def splithalf(s, t, draws=200):
    """Reliability of one species' response in one condition: split its subjects."""
    ii = idx[(s, t)]
    su = sorted(set(subj[(s, t)]))
    if len(su) < 4:
        return np.nan
    vals = []
    half = len(su) // 2
    combos = list(itertools.combinations(su, half))
    if len(combos) > draws:
        combos = [tuple(rng.choice(su, half, replace=False)) for _ in range(draws)]
    for A in combos:
        A = set(A); Bs = [u for u in su if u not in A]
        a = np.nanmean(X[[i for i in ii if rows[i]['subject'] in A]], axis=0)
        b = np.nanmean(X[[i for i in ii if rows[i]['subject'] in Bs]], axis=0)
        vals.append(pear(a, b))
    r1 = float(np.nanmean(vals))
    # Spearman-Brown back up to the full subject set
    return 2 * r1 / (1 + r1) if r1 > -1 else np.nan

rel = {s: float(np.nanmean([splithalf(s, t) for t in tk])) for s in sp}
nsub = {s: len(set(r['subject'] for r in rows if r['species'] == s)) for s in sp}

donors = [s for s in sp if s != H]
raw = {s: float(np.nanmean([pear(mean[s][t], mean[H][t]) for t in tk])) for s in donors}
norm = {s: raw[s] / np.sqrt(rel[s] * rel[H]) if rel[s] > 0 and rel[H] > 0 else np.nan
        for s in donors}

SH = {'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
      'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
      'Mus musculus':'mouse','Rattus norvegicus':'rat','Homo sapiens':'human'}
meta = list(csv.DictReader(open(D + 'sample_metadata.tsv'), delimiter='\t'))
GRP = {m['organism']: m['group'][0] for m in meta}

print('human own reliability (ceiling base): %.4f  (n=%d subjects)' % (rel[H], nsub[H]))
print()
print('%-9s %s %4s %9s %9s %11s' % ('species','g','n','FIG1 raw','own rel','ceiling-norm'))
for s in sorted(donors, key=lambda x: -raw[x]):
    print('%-9s  %s %4d %9.4f %9.4f %11.4f' % (SH[s], GRP[s], nsub[s], raw[s], rel[s], norm[s]))
print()
print('ranking by the normalised value:')
for i, s in enumerate(sorted(donors, key=lambda x: -norm[x]), 1):
    print('  %d. %-9s %.4f   (raw rank %d)' % (
        i, SH[s], norm[s], sorted(donors, key=lambda x: -raw[x]).index(s) + 1))
print()
o = sorted(donors, key=lambda s: -raw[s])
print('Spearman(raw ranking, normalised ranking) = %+.3f' % spearman([raw[s] for s in o], [norm[s] for s in o]))
print('Spearman(raw, own reliability)            = %+.3f' % spearman([raw[s] for s in o], [rel[s] for s in o]))
print('Spearman(normalised, own reliability)     = %+.3f' % spearman([norm[s] for s in o], [rel[s] for s in o]))
print()
prim = [s for s in donors if s in ('Pan troglodytes','Macaca','Papio anubis')]
oth = [s for s in donors if s not in prim]
print('primate mean   raw %.4f -> normalised %.4f' % (np.mean([raw[s] for s in prim]), np.mean([norm[s] for s in prim])))
print('non-primate    raw %.4f -> normalised %.4f' % (np.mean([raw[s] for s in oth]), np.mean([norm[s] for s in oth])))

json.dump({'raw': raw, 'own_reliability': rel, 'ceiling_normalised': norm,
           'n_subjects': nsub, 'group': GRP},
          open('artifacts/biology/concordance/fig1_ceiling_control.json', 'w'), indent=1)
print('\nwrote artifacts/biology/concordance/fig1_ceiling_control.json')
