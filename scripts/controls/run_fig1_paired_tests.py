"""Stratified jackknife: the pooled version mixes strata of unequal size.

Deleting one subject perturbs only its own species, so pooling subjects from two species
and the target into a single delete-one sequence is not an exchangeable jackknife. The
standard estimator sums a within-stratum variance over strata. If the two agree, the
interval is insensitive to the choice; if they do not, the stratified one is correct.
"""
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
SH={'Pan troglodytes':'chimp','Macaca':'macaque','Papio anubis':'baboon','Bos taurus':'cow',
    'Oryctolagus cuniculus':'rabbit','Ovis aries':'sheep','Sus scrofa':'pig',
    'Mus musculus':'mouse','Rattus norvegicus':'rat'}
def se_pooled(fn,strata):
    pool=[u for g in strata for u in subs[g]]; n=len(pool)
    r=np.array([fn(u) for u in pool])
    return float(np.sqrt((n-1)/n*((r-r.mean())**2).sum()))
def se_strat(fn,strata):
    v=0.0
    for g in strata:
        ng=len(subs[g]); r=np.array([fn(u) for u in subs[g]])
        v += (ng-1)/ng*((r-r.mean())**2).sum()
    return float(np.sqrt(v))
donors=[s for s in sp if s!=H]
order=sorted(donors,key=lambda s:-val(s))
print('%-20s %9s %9s %9s %18s %18s'%('comparison','diff','SE pooled','SE strat','pooled CI','stratified CI'))
out={}
for i,a in enumerate(order):
    for b in order[i+1:]:
        f=lambda u,a=a,b=b: val(a,u)-val(b,u)
        pt=val(a)-val(b)
        sp_=se_pooled(f,[a,b,H]); ss=se_strat(f,[a,b,H])
        sigp = (pt-1.96*sp_>0) or (pt+1.96*sp_<0)
        sigs = (pt-1.96*ss>0) or (pt+1.96*ss<0)
        out[f'{SH[a]} vs {SH[b]}']=dict(diff=pt,se_pooled=sp_,se_strat=ss,sig_strat=bool(sigs))
        if sigp!=sigs or not sigs:
            print('%-20s %+9.4f %9.4f %9.4f  [%+.3f,%+.3f] %s [%+.3f,%+.3f] %s'%(
                f'{SH[a]} vs {SH[b]}',pt,sp_,ss,pt-1.96*sp_,pt+1.96*sp_,'*' if sigp else ' ',
                pt-1.96*ss,pt+1.96*ss,'*' if sigs else ' '))
ns=[k for k,v in out.items() if not v['sig_strat']]
print(f'\n分层刀切下分不开的: {len(ns)} 对 — {", ".join(ns)}')
print('（合并刀切下是 7 对）')
json.dump(out,open('artifacts/biology/concordance/fig1_paired.json','w'),indent=1)
