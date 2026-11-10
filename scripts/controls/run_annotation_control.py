"""Does the donor ranking survive restriction to genes measured well in every species?

The open alternative explanation for Figure 1 is annotation quality: species whose genomes
are annotated less well map fewer genes onto the shared human ENSG axis, those genes sit at
or near zero TPM, their log fold-changes are compressed toward zero, and the species scores
low for a reason that is about reference genomes rather than biology. Because annotation
quality correlates with how much a species has been studied, which correlates with
relatedness to human, this would manufacture exactly the observed primate gradient.

The test: drop every gene that is not detectably expressed at rest in ALL ten species, so
that no species can be penalised for genes it failed to map, and recompute the ranking.
"""
import csv, gzip, json, numpy as np

D='artifacts/gse249010_v1/processed/'
RAW='artifacts/gse249010_v1/data/GSE249010_processed_data_TPM.txt.gz'
H='Homo sapiens'
rng=np.random.default_rng(20261023)

def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    a=a-a.mean(); b=b-b.mean()
    d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else np.nan
def spearman(a,b):
    r=lambda v: np.argsort(np.argsort(np.asarray(v,float))).astype(float)
    return pear(r(a),r(b))

rows=list(csv.DictReader(open(D+'response_metadata.tsv'),delimiter='\t'))
X=np.load(D+'responses.npy')
genes=[str(g) for g in np.load(D+'genes.npy',allow_pickle=True)]
sp=sorted({r['species'] for r in rows})
tk=sorted({r['task_id'] for r in rows})
tk=[t for t in tk if all(any(r['species']==s and r['task_id']==t for r in rows) for s in sp)]
idx={(s,t):[i for i,r in enumerate(rows) if r['species']==s and r['task_id']==t] for s in sp for t in tk}

meta=list(csv.DictReader(open(D+'sample_metadata.tsv'),delimiter='\t'))
nc=[m for m in meta if m['treatment']=='NC']; want={m['sample_id'] for m in nc}
byid={m['sample_id']:m for m in nc}
with gzip.open(RAW,'rt') as fh:
    hdr=fh.readline().rstrip('\n').split('\t'); cols=hdr[1:]
    keep=[i for i,c in enumerate(cols) if c in want]; kept=[cols[i] for i in keep]
    M=[]
    for line in fh:
        f=line.rstrip('\n').split('\t'); v=f[1:]
        M.append([0.0 if v[i] in ('NA','','null') else float(v[i]) for i in keep])
B=np.log2(np.maximum(np.asarray(M,dtype=np.float32),0.0)+1.0)
base={s:B[:,[j for j,sid in enumerate(kept) if byid[sid]['organism']==s]].mean(axis=1) for s in sp}

SH={'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
    'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
    'Mus musculus':'mouse','Rattus norvegicus':'rat','Homo sapiens':'human'}
donors=[s for s in sp if s!=H]
SIX=['Sus scrofa','Bos taurus','Ovis aries','Oryctolagus cuniculus','Rattus norvegicus','Mus musculus']

def fig1(mask, subj_pick=None):
    """mean-of-condition Pearson against human, optionally on a subject resample."""
    def prof(s,t):
        ii=idx[(s,t)]
        if subj_pick is not None:
            su=subj_pick[s]; ii=[i for i in ii if rows[i]['subject'] in su] or ii
        return np.nanmean(X[ii],axis=0)[mask]
    return {s: float(np.nanmean([pear(prof(s,t), prof(H,t)) for t in tk])) for s in donors}

allg=np.ones(len(genes),bool)
res={'all genes (25,638)': fig1(allg)}
for thr in (0.5,1.0,2.0):
    mask=np.ones(len(genes),bool)
    for s in sp: mask &= base[s] > thr
    res[f'expressed in all 10 species (log2TPM>{thr}): {int(mask.sum()):,}']=fig1(mask)

print('Figure 1 recomputed on progressively stricter common gene sets')
print()
hdr=list(res)
print('%-9s'%'species' + ''.join('%26s'%h.split(':')[0][:25] for h in hdr))
print('%-9s'%'' + ''.join('%26s'%('n='+h.split(':')[1].strip() if ':' in h else '') for h in hdr))
order=sorted(donors,key=lambda s:-res[hdr[0]][s])
for s in order:
    print('%-9s'%SH[s] + ''.join('%26.4f'%res[h][s] for h in hdr))
print()
b=[res[hdr[0]][s] for s in order]
for h in hdr[1:]:
    print('  Spearman(all genes, %-44s) = %+.3f' % (h.split(':')[0], spearman(b,[res[h][s] for s in order])))
print()
for h in hdr:
    pr=np.mean([res[h][s] for s in donors if s in ('Pan troglodytes','Macaca','Papio anubis')])
    npr=np.mean([res[h][s] for s in donors if s not in ('Pan troglodytes','Macaca','Papio anubis')])
    print('  %-50s primate %.4f  non-primate %.4f  ratio %.2fx' % (h.split(':')[0], pr, npr, pr/npr))

# ---------- bootstrap r-bar and rho over subjects ----------
print()
print('bootstrap of the two curve parameters (1,000 subject-level resamples, six collectible species)')
subs={s:sorted({r['subject'] for r in rows if r['species']==s}) for s in sp}
def params(pick):
    def prof(s,t):
        ii=[i for i in idx[(s,t)] if rows[i]['subject'] in pick[s]] or idx[(s,t)]
        return np.nanmean(X[ii],axis=0)
    rb=float(np.mean([np.nanmean([pear(prof(s,t),prof(H,t)) for t in tk]) for s in SIX]))
    pw=[np.nanmean([pear(prof(a,t),prof(b_,t)) for t in tk])
        for i,a in enumerate(SIX) for b_ in SIX[i+1:]]
    return rb, float(np.mean(pw))
pt=params({s:set(subs[s]) for s in sp})
boot=[]
for _ in range(1000):
    pick={s:set(rng.choice(subs[s],len(subs[s]),replace=True)) for s in sp}
    try: boot.append(params(pick))
    except Exception: pass
boot=np.array(boot)
for nm,i,printed in (('r̄',0,0.2762),('ρ',1,0.2654)):
    lo,hi=np.percentile(boot[:,i],[2.5,97.5])
    print('  %-3s point %.4f  (printed %.4f)   95%% CI [%.4f, %.4f]  width %.4f'%(nm,pt[i],printed,lo,hi,hi-lo))
asym=boot[:,0]/np.sqrt(boot[:,1])
lo,hi=np.percentile(asym,[2.5,97.5])
mac=0.533630996294816
print('  asymptote r̄/√ρ  point %.4f   95%% CI [%.4f, %.4f]'%(pt[0]/np.sqrt(pt[1]),lo,hi))
print('  P(asymptote > one macaque %.4f) = %.1f%%'%(mac,100*float((asym>mac).mean())))

json.dump({'by_geneset':{k:v for k,v in res.items()},
           'rbar_ci':[float(x) for x in np.percentile(boot[:,0],[2.5,97.5])],
           'rho_ci':[float(x) for x in np.percentile(boot[:,1],[2.5,97.5])],
           'asymptote_ci':[float(lo),float(hi)],
           'p_asymptote_above_macaque':float((asym>mac).mean())},
          open('artifacts/biology/concordance/annotation_control.json','w'),indent=1)
print('\nwrote artifacts/biology/concordance/annotation_control.json')
