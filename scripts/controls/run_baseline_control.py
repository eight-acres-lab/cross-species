"""Is the Figure 1 species ranking a response ranking, or a resting-state ranking?

Gregory 2024's own headline is that the sensitive/resilient separation sits largely at
BASELINE (3,012 genes at >=2-fold/FDR<0.05) rather than in the LPS response. If the donor
ranking tracks resting-state similarity to human, it is a resting-state ranking and the
product claim changes. The response values are already baseline-subtracted per subject and
timepoint, so this asks a different question: does a species' resting similarity to human
predict how well its RESPONSE predicts the human response?
"""
import csv, gzip, json, numpy as np

D = 'artifacts/gse249010_v1/processed/'
RAW = 'artifacts/gse249010_v1/data/GSE249010_processed_data_TPM.txt.gz'
H = 'Homo sapiens'

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pear(r(a), r(b))

# ---------- response side: reproduce Figure 1 ----------
rows = list(csv.DictReader(open(D + 'response_metadata.tsv'), delimiter='\t'))
X = np.load(D + 'responses.npy')
genes = [str(g) for g in np.load(D + 'genes.npy', allow_pickle=True)]
sp = sorted({r['species'] for r in rows})
tk = sorted({r['task_id'] for r in rows})
tk = [t for t in tk if all(any(r['species'] == s and r['task_id'] == t for r in rows) for s in sp)]
resp = {s: {t: np.nanmean(X[[i for i, r in enumerate(rows) if r['species'] == s and r['task_id'] == t]], axis=0)
            for t in tk} for s in sp}
donors = [s for s in sp if s != H]
fig1 = {s: float(np.nanmean([pear(resp[s][t], resp[H][t]) for t in tk])) for s in donors}

# ---------- baseline side: the zero-dose samples ----------
meta = list(csv.DictReader(open(D + 'sample_metadata.tsv'), delimiter='\t'))
nc = [m for m in meta if m['treatment'] == 'NC']
want = {m['sample_id'] for m in nc}
print(f'zero-dose samples: {len(nc)}  species: {len(set(m["organism"] for m in nc))}')

with gzip.open(RAW, 'rt') as fh:
    hdr = fh.readline().rstrip('\n').split('\t')
    cols = hdr[1:]
    keep = [i for i, c in enumerate(cols) if c in want]
    kept_ids = [cols[i] for i in keep]
    gg, M = [], []
    for line in fh:
        f = line.rstrip('\n').split('\t')
        gg.append(f[0])
        vals = f[1:]
        M.append([0.0 if vals[i] in ('NA', '', 'null') else float(vals[i]) for i in keep])
M = np.asarray(M, dtype=np.float32)
assert gg == genes, 'gene axis mismatch between raw matrix and processed genes.npy'
B = np.log2(np.maximum(M, 0.0) + 1.0)          # same transform as the pipeline
print(f'baseline matrix: {B.shape[0]} genes x {B.shape[1]} zero-dose samples')

byid = {m['sample_id']: m for m in nc}
base = {}
for s in sp:
    idx = [j for j, sid in enumerate(kept_ids) if byid[sid]['organism'] == s]
    base[s] = B[:, idx].mean(axis=1)

# species-specific deviation: remove what every mammal shares
G = np.vstack([base[s] for s in sp])
dev = {s: base[s] - G.mean(axis=0) for s in sp}

raw_sim = {s: pear(base[s], base[H]) for s in donors}
dev_sim = {s: pear(dev[s], dev[H]) for s in donors}

# ---------- diagnostics ----------
amp = {s: float(np.nanmean([np.nanstd(resp[s][t]) for t in tk])) for s in sp}   # response size
lowfrac = {s: float(np.mean(base[s] < 0.1)) for s in sp}                        # compressed genes

SH = {'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
      'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
      'Mus musculus':'mouse','Rattus norvegicus':'rat','Homo sapiens':'human'}
GRP = {m['organism']: m['group'][0] for m in meta}

order = sorted(donors, key=lambda s: -fig1[s])
print()
print('%-9s %s %8s %10s %10s %9s %9s' % ('species','grp','FIG1 r','base r','base dev r','resp sd','low%'))
for s in order:
    print('%-9s  %s  %8.4f %10.4f %10.4f %9.4f %8.1f%%' %
          (SH[s], GRP[s], fig1[s], raw_sim[s], dev_sim[s], amp[s], 100*lowfrac[s]))
print('%-9s  %s  %8s %10s %10s %9.4f %8.1f%%' % (SH[H], GRP[H], '-', '-', '-', amp[H], 100*lowfrac[H]))

f = [fig1[s] for s in order]
print()
print('Spearman across the 9 donors:')
print('  FIG1 vs resting similarity (raw)            %+.3f' % spearman(f, [raw_sim[s] for s in order]))
print('  FIG1 vs resting similarity (species dev)    %+.3f' % spearman(f, [dev_sim[s] for s in order]))
print('  FIG1 vs response amplitude                  %+.3f' % spearman(f, [amp[s] for s in order]))
print('  FIG1 vs fraction of near-zero-baseline genes %+.3f' % spearman(f, [lowfrac[s] for s in order]))

json.dump({'fig1': fig1, 'baseline_raw': raw_sim, 'baseline_dev': dev_sim,
           'resp_amplitude': amp, 'low_baseline_fraction': lowfrac, 'group': GRP},
          open('artifacts/biology/concordance/baseline_control.json', 'w'), indent=1)
print('\nwrote artifacts/biology/concordance/baseline_control.json')
