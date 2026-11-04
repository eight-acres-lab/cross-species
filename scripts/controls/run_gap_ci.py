"""Is the asymptote-minus-macaque gap resolvable on any gene set?

Jackknife the GAP directly (leave one subject out), rather than the two quantities
separately: they share the human target and the same subjects, so a paired interval on the
difference is the right object and is much tighter than either margin alone.
"""
import csv, gzip, json, numpy as np
D='artifacts/gse249010_v1/processed/'
RAW='artifacts/gse249010_v1/data/GSE249010_processed_data_TPM.txt.gz'
H='Homo sapiens'; MAC='Macaca'
SIX=['Sus scrofa','Bos taurus','Ovis aries','Oryctolagus cuniculus','Rattus norvegicus','Mus musculus']
def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else np.nan
rows=list(csv.DictReader(open(D+'response_metadata.tsv'),delimiter='\t'))
X=np.load(D+'responses.npy'); genes=np.load(D+'genes.npy',allow_pickle=True)
sp=sorted({r['species'] for r in rows}); tk=sorted({r['task_id'] for r in rows})
tk=[t for t in tk if all(any(r['species']==s and r['task_id']==t for r in rows) for s in sp)]
idx={(s,t):[i for i,r in enumerate(rows) if r['species']==s and r['task_id']==t] for s in sp for t in tk}
subs={s:sorted({r['subject'] for r in rows if r['species']==s}) for s in sp}
meta=list(csv.DictReader(open(D+'sample_metadata.tsv'),delimiter='\t'))
nc=[m for m in meta if m['treatment']=='NC']; want={m['sample_id'] for m in nc}; byid={m['sample_id']:m for m in nc}
with gzip.open(RAW,'rt') as fh:
    cols=fh.readline().rstrip('\n').split('\t')[1:]
    keep=[i for i,c in enumerate(cols) if c in want]; kept=[cols[i] for i in keep]
    M=[[0.0 if v[i] in ('NA','','null') else float(v[i]) for i in keep]
       for v in (l.rstrip('\n').split('\t')[1:] for l in fh)]
B=np.log2(np.maximum(np.asarray(M,dtype=np.float32),0.0)+1.0)
base={s:B[:,[j for j,x in enumerate(kept) if byid[x]['organism']==s]].mean(axis=1) for s in sp}

def gap(mask, drop=None):
    def prof(s,t):
        ii=[i for i in idx[(s,t)] if rows[i]['subject']!=drop] or idx[(s,t)]
        return np.nanmean(X[ii],axis=0)[mask]
    rb=float(np.mean([np.nanmean([pear(prof(s,t),prof(H,t)) for t in tk]) for s in SIX]))
    rho=float(np.mean([np.nanmean([pear(prof(a,t),prof(b,t)) for t in tk])
                       for i,a in enumerate(SIX) for b in SIX[i+1:]]))
    mac=float(np.nanmean([pear(prof(MAC,t),prof(H,t)) for t in tk]))
    return rb/np.sqrt(rho)-mac

sets={'all genes':np.ones(len(genes),bool)}
for thr in (0.5,1.0,2.0):
    m=np.ones(len(genes),bool)
    for s in sp: m &= base[s]>thr
    sets[f'all 10 expressed (>{thr})']=m
pool=[u for s in SIX+[H,MAC] for u in subs[s]]; n=len(pool)
print(f'paired jackknife of (asymptote - macaque) over {n} subjects\n')
print('%-26s %10s %9s %20s %s'%('gene set','gap','SE','95% CI','verdict'))
out={}
for nm,m in sets.items():
    pt=gap(m); reps=np.array([gap(m,u) for u in pool])
    se=float(np.sqrt((n-1)/n*((reps-reps.mean())**2).sum()))
    lo,hi=pt-1.96*se,pt+1.96*se
    v='不可判定' if lo<0<hi else ('渐近线确实低于猕猴' if hi<0 else '渐近线确实高于猕猴')
    out[nm]=dict(gap=pt,se=se,lo=lo,hi=hi)
    print('%-26s %+10.4f %9.4f   [%+.4f, %+.4f]  %s'%(nm,pt,se,lo,hi,v))
json.dump(out,open('artifacts/biology/concordance/gap_ci.json','w'),indent=1)
print('\nwrote artifacts/biology/concordance/gap_ci.json')
