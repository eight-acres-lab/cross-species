import csv, json, numpy as np
D='artifacts/gse249010_v1/processed/'; H='Homo sapiens'
def pear(a,b):
    m=np.isfinite(a)&np.isfinite(b); a,b=a[m],b[m]
    a=a-a.mean(); b=b-b.mean(); d=np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d>0 else np.nan
rows=list(csv.DictReader(open(D+'response_metadata.tsv'),delimiter='\t'))
X=np.load(D+'responses.npy')
sp=sorted({r['species'] for r in rows}); tk=sorted({r['task_id'] for r in rows})
tk=[t for t in tk if all(any(r['species']==s and r['task_id']==t for r in rows) for s in sp)]
idx={(s,t):[i for i,r in enumerate(rows) if r['species']==s and r['task_id']==t] for s in sp for t in tk}
subs={s:sorted({r['subject'] for r in rows if r['species']==s}) for s in sp}
def val(s,drop=None):
    def prof(x,t):
        ii=[i for i in idx[(x,t)] if rows[i]['subject']!=drop] or idx[(x,t)]
        return np.nanmean(X[ii],axis=0)
    return float(np.nanmean([pear(prof(s,t),prof(H,t)) for t in tk]))
old=json.load(open('artifacts/biology/concordance/fig1_ci.json'))
out={}
print('%-22s %8s %9s %9s'%('species','value','SE pooled','SE strat'))
for s in [x for x in sp if x!=H]:
    pt=val(s); v=0.0
    for g in (s,H):
        ng=len(subs[g]); r=np.array([val(s,u) for u in subs[g]])
        v += (ng-1)/ng*((r-r.mean())**2).sum()
    se=float(np.sqrt(v))
    out[s]=dict(point=pt,se=se,lo=pt-1.96*se,hi=pt+1.96*se,n_subjects=len(subs[s]))
    print('%-22s %8.4f %9.4f %9.4f'%(s,pt,old[s]['se'],se))
json.dump(out,open('artifacts/biology/concordance/fig1_ci.json','w'),indent=1)
print('\nSE 范围: %.4f – %.4f  (合并版是 %.4f – %.4f)'%(
    min(v['se'] for v in out.values()),max(v['se'] for v in out.values()),
    min(v['se'] for v in old.values()),max(v['se'] for v in old.values())))
