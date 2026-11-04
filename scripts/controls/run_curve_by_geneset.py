"""Carry the gene-set robustness test from Figure 1 to the curve that prices the panel.

Restricting to genes expressed in all ten species raised every donor correlation by about
+0.15. The saturation curve's parameters -- r-bar, rho, and the asymptote they imply -- were
never recomputed on those gene sets, and the whole pricing argument rests on a 0.002 gap
between that asymptote and one macaque. If the asymptote moves above the macaque, "no
collectible panel reaches a macaque" becomes "a handful of species reach one", which is the
opposite conclusion.
"""
import csv, gzip, json, numpy as np
D='artifacts/gse249010_v1/processed/'
RAW='artifacts/gse249010_v1/data/GSE249010_processed_data_TPM.txt.gz'
H='Homo sapiens'
SIX=['Sus scrofa','Bos taurus','Ovis aries','Oryctolagus cuniculus','Rattus norvegicus','Mus musculus']

def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else np.nan

rows=list(csv.DictReader(open(D+'response_metadata.tsv'),delimiter='\t'))
X=np.load(D+'responses.npy')
genes=[str(g) for g in np.load(D+'genes.npy',allow_pickle=True)]
sp=sorted({r['species'] for r in rows}); tk=sorted({r['task_id'] for r in rows})
tk=[t for t in tk if all(any(r['species']==s and r['task_id']==t for r in rows) for s in sp)]
idx={(s,t):[i for i,r in enumerate(rows) if r['species']==s and r['task_id']==t] for s in sp for t in tk}
mean={s:{t:np.nanmean(X[idx[(s,t)]],axis=0) for t in tk} for s in sp}

# baselines, to build the same gene sets Figure 1 used
meta=list(csv.DictReader(open(D+'sample_metadata.tsv'),delimiter='\t'))
nc=[m for m in meta if m['treatment']=='NC']; want={m['sample_id'] for m in nc}
byid={m['sample_id']:m for m in nc}
with gzip.open(RAW,'rt') as fh:
    cols=fh.readline().rstrip('\n').split('\t')[1:]
    keep=[i for i,c in enumerate(cols) if c in want]; kept=[cols[i] for i in keep]
    M=[]
    for line in fh:
        v=line.rstrip('\n').split('\t')[1:]
        M.append([0.0 if v[i] in ('NA','','null') else float(v[i]) for i in keep])
B=np.log2(np.maximum(np.asarray(M,dtype=np.float32),0.0)+1.0)
base={s:B[:,[j for j,x in enumerate(kept) if byid[x]['organism']==s]].mean(axis=1) for s in sp}

MAC='Macaca'
def curve(mask):
    rb=float(np.mean([np.nanmean([pear(mean[s][t][mask],mean[H][t][mask]) for t in tk]) for s in SIX]))
    pw=[np.nanmean([pear(mean[a][t][mask],mean[b][t][mask]) for t in tk])
        for i,a in enumerate(SIX) for b in SIX[i+1:]]
    rho=float(np.mean(pw))
    mac=float(np.nanmean([pear(mean[MAC][t][mask],mean[H][t][mask]) for t in tk]))
    asym=rb/np.sqrt(rho)
    # k such that R(k) = macaque
    den=rb**2-mac**2*rho
    k=mac**2*(1-rho)/den if den>0 else float('inf')
    # measured six-species panel
    six=float(np.nanmean([pear(np.mean([mean[s][t][mask] for s in SIX],axis=0),mean[H][t][mask]) for t in tk]))
    return rb,rho,asym,mac,k,six

sets={'all genes':np.ones(len(genes),bool)}
for thr in (0.5,1.0,2.0):
    m=np.ones(len(genes),bool)
    for s in sp: m &= base[s]>thr
    sets[f'expressed in all 10 (>{thr})']=m

print('%-28s %7s %7s %9s %9s %12s %10s %8s'%('gene set','n genes','r̄','ρ','asymptote','one macaque','gap','k'))
res={}
for nm,m in sets.items():
    rb,rho,asym,mac,k,six=curve(m)
    res[nm]=dict(n=int(m.sum()),rbar=rb,rho=rho,asymptote=asym,macaque=mac,gap=asym-mac,
                 k=None if not np.isfinite(k) else float(k),six_panel=six)
    ks='unreachable' if not np.isfinite(k) or k<0 else f'{k:.0f}'
    print('%-28s %7d %7.4f %7.4f %9.4f %9.4f %+12.4f %10s'%(nm,m.sum(),rb,rho,asym,mac,asym-mac,ks))
print()
print('%-28s %12s %12s'%('gene set','6-sp panel','% of asymptote'))
for nm,v in res.items():
    print('%-28s %12.4f %11.1f%%'%(nm,v['six_panel'],100*v['six_panel']/v['asymptote']))
json.dump(res,open('artifacts/biology/concordance/curve_by_geneset.json','w'),indent=1)
print('\nwrote artifacts/biology/concordance/curve_by_geneset.json')
