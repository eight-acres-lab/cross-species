"""Leave-one-subject-out interval for the two curve parameters.

A naive subject bootstrap is biased downward here: drawing with replacement reduces the
number of DISTINCT subjects per species, which adds noise to each species' mean profile and
attenuates every correlation the statistic is built from. The point estimate then falls
outside its own interval, which is what we observed. The jackknife keeps every draw at
n-1 distinct subjects, so the attenuation is small and constant.
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
SIX=['Sus scrofa','Bos taurus','Ovis aries','Oryctolagus cuniculus','Rattus norvegicus','Mus musculus']
subs={s:sorted({r['subject'] for r in rows if r['species']==s}) for s in sp}

def params(drop=None):
    def prof(s,t):
        ii=[i for i in idx[(s,t)] if rows[i]['subject']!=drop] or idx[(s,t)]
        return np.nanmean(X[ii],axis=0)
    rb=float(np.mean([np.nanmean([pear(prof(s,t),prof(H,t)) for t in tk]) for s in SIX]))
    pw=[np.nanmean([pear(prof(a,t),prof(b,t)) for t in tk]) for i,a in enumerate(SIX) for b in SIX[i+1:]]
    return rb,float(np.mean(pw))

full=params()
allsub=[u for s in SIX+[H] for u in subs[s]]
J=np.array([params(u) for u in allsub])
n=len(allsub)
print(f'jackknife over {n} subjects (six collectible species + human)')
mac=0.533630996294816
out={}
for nm,i,pt in (('r̄',0,full[0]),('ρ',1,full[1])):
    ps=J[:,i]; se=np.sqrt((n-1)/n*((ps-ps.mean())**2).sum())
    out[nm]=(pt,se)
    print(f'  {nm:3s} point {pt:.4f}   jackknife SE {se:.4f}   95% CI [{pt-1.96*se:.4f}, {pt+1.96*se:.4f}]')
# asymptote by the delta method on the jackknife replicates
a=J[:,0]/np.sqrt(J[:,1]); pa=full[0]/np.sqrt(full[1])
se=np.sqrt((n-1)/n*((a-a.mean())**2).sum())
print(f'  asymptote point {pa:.4f}   jackknife SE {se:.4f}   95% CI [{pa-1.96*se:.4f}, {pa+1.96*se:.4f}]')
print(f'  one macaque = {mac:.4f}   gap = {pa-mac:+.4f}   = {(pa-mac)/se:+.2f} SE')
print()
print(f'  → 间距 {pa-mac:+.4f} 是 {abs((pa-mac)/se):.2f} 个标准误，符号不可判定' if abs(pa-mac)<1.96*se
      else '  → 间距显著')
json.dump({'rbar':{'point':full[0],'se':out['r̄'][1]},'rho':{'point':full[1],'se':out['ρ'][1]},
           'asymptote':{'point':pa,'se':se},'macaque':mac,'n_subjects':n},
          open('artifacts/biology/concordance/curve_param_ci.json','w'),indent=1)
print('wrote artifacts/biology/concordance/curve_param_ci.json')
