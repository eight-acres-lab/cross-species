"""Does the NA-to-zero convention bias the species ordering?

prepare_gse249010_pilot.py:53 converts the literal tokens NA, '' and null to TPM 0.0, so a
gene that was never measured becomes a gene measured at zero and the two are indistinguishable
downstream. That is harmless if the missingness rate is the same for every species. It is not
harmless if it tracks species, because the species ordering is the headline result and the
same thing that drives missingness -- how well a genome is annotated and how well its reads
map onto the shared human axis -- also tracks relatedness to human.

Step 1 here: measure the missingness rate per species and test whether it predicts the
ordering. Step 2 rebuilds the responses with NA masked rather than zeroed and recomputes
Figure 1.
"""
import csv, gzip, json, sys
import numpy as np

D = 'artifacts/gse249010_v1/processed/'
RAW = 'artifacts/gse249010_v1/data/GSE249010_processed_data_TPM.txt.gz'
MISSING = {'NA', '', 'null'}

meta = list(csv.DictReader(open(D + 'sample_metadata.tsv'), delimiter='\t'))
by_id = {m['sample_id']: m for m in meta}
genes_axis = [str(g) for g in np.load(D + 'genes.npy', allow_pickle=True)]

print('reading raw matrix, recording which cells are the literal token'); sys.stdout.flush()
with gzip.open(RAW, 'rt') as fh:
    cols = fh.readline().rstrip('\n').split('\t')[1:]
    gg, V, M = [], [], []
    for line in fh:
        f = line.rstrip('\n').split('\t')
        gg.append(f[0]); vals = f[1:]
        M.append([v in MISSING for v in vals])
        V.append([0.0 if v in MISSING else float(v) for v in vals])
assert gg == genes_axis, 'gene axis mismatch'
V = np.asarray(V, dtype=np.float32)          # NA already coerced to 0, as the pipeline does
M = np.asarray(M, dtype=bool)                # True where the token was missing
print(f'matrix {V.shape}, missing cells {M.mean():.2%}'); sys.stdout.flush()

SH = {'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
      'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
      'Mus musculus':'mouse','Rattus norvegicus':'rat','Homo sapiens':'human'}
species = sorted({m['organism'] for m in meta})
colsp = np.array([by_id[c]['organism'] for c in cols])

FIG1 = {'Pan troglodytes':0.6756,'Macaca':0.5336,'Papio anubis':0.5066,'Bos taurus':0.3450,
        'Oryctolagus cuniculus':0.3344,'Ovis aries':0.2780,'Sus scrofa':0.2655,
        'Mus musculus':0.2283,'Rattus norvegicus':0.2061}

rate = {s: float(M[:, colsp == s].mean()) for s in species}
print()
print('%-10s %10s %10s'%('species','NA rate','FIG1'))
for s in sorted(species, key=lambda z: -rate[z]):
    print('%-10s %9.2f%% %10s'%(SH[s], 100*rate[s], f'{FIG1[s]:.4f}' if s in FIG1 else 'target'))

def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan
def spear(a, b):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pear(r(a), r(b))

don = [s for s in species if s in FIG1]
print()
print('Spearman(NA rate, FIG1 value) over the nine donors = %+.3f'
      % spear([rate[s] for s in don], [FIG1[s] for s in don]))
print('  (a strong negative means the species that score low are the species with more missing data)')

# genes that are missing in EVERY sample of a species -- the ones the "expressed in all ten"
# filter removes, but for a reason that is not biology
allmiss = {s: int(M[:, colsp == s].all(axis=1).sum()) for s in species}
print()
print('%-10s %s'%('species','genes missing in every sample of that species'))
for s in sorted(species, key=lambda z: -allmiss[z]):
    print('%-10s %6d  (%.1f%% of axis)'%(SH[s], allmiss[s], 100*allmiss[s]/len(gg)))

json.dump({'na_rate': rate, 'genes_all_missing': allmiss,
           'spearman_rate_vs_fig1': spear([rate[s] for s in don], [FIG1[s] for s in don])},
          open('artifacts/gse249010_v1/na_audit.json', 'w'), indent=1)
from pathlib import Path
cache = Path('artifacts/gse249010_v1/na_cache')
cache.mkdir(parents=True, exist_ok=True)
np.save(cache / 'na_mask.npy', M)
np.save(cache / 'raw_vals.npy', V)
print('\nwrote artifacts/gse249010_v1/na_audit.json and cached the mask')
