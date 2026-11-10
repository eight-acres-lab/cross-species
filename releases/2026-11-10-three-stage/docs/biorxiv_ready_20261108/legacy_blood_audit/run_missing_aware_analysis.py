"""Frozen blood matrix and pooling reanalysis, preserving biological sampling units."""
from pathlib import Path
import csv,hashlib,itertools,json,os,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
import numpy as np
from scipy.stats import rankdata,spearmanr,pearsonr

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/biorxiv_ready_20261108/legacy_blood')
PREP=ROOT/'prepared_v1';OUT=ROOT/'analysis_v1';B=500;SEED=20261108
TIMES=[2,6,24];DOSES=[10,100,1000]
HUMAN='Homo sapiens'
TAX={'Homo sapiens':('Homo','Hominidae','Primates','Euarchontoglires'),'Pan troglodytes':('Pan','Hominidae','Primates','Euarchontoglires'),'Macaca':('Macaca','Cercopithecidae','Primates','Euarchontoglires'),'Papio anubis':('Papio','Cercopithecidae','Primates','Euarchontoglires'),'Mus musculus':('Mus','Muridae','Rodentia','Euarchontoglires'),'Rattus norvegicus':('Rattus','Muridae','Rodentia','Euarchontoglires'),'Oryctolagus cuniculus':('Oryctolagus','Leporidae','Lagomorpha','Euarchontoglires'),'Bos taurus':('Bos','Bovidae','Artiodactyla','Laurasiatheria'),'Ovis aries':('Ovis','Bovidae','Artiodactyla','Laurasiatheria'),'Sus scrofa':('Sus','Suidae','Artiodactyla','Laurasiatheria')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(p,x):Path(p).write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);x=x-x.mean();y=y-y.mean();d=np.sqrt(x@x*(y@y));return float(x@y/d) if d>0 else np.nan
def spearman(x,y):return corr(rankdata(x),rankdata(y))
def taxdist(a,b):
 for i,(x,y) in enumerate(zip(TAX[a],TAX[b])):
  if x==y:return i+1
 return 5
def matrix(v):
 centered=v-v.mean(axis=1,keepdims=True);norm=np.linalg.norm(centered,axis=1)
 assert np.all(norm>0)
 return centered@centered.T/(norm[:,None]*norm[None,:])
def compare(a,b,pi,pj,tax):
 x,y=a[pi,pj],b[pi,pj];cross=tax>=4;rx,ry,rz=rankdata(x),rankdata(y),rankdata(tax)
 design=np.column_stack([np.ones(len(rz)),rz])
 return [spearman(x,y),spearman(x[cross],y[cross]),corr(rx-design@np.linalg.lstsq(design,rx,rcond=None)[0],ry-design@np.linalg.lstsq(design,ry,rcond=None)[0])]
def subset_means(indices,meta,responses,species,tasks,selected=None):
 result=[]
 for s in species:
  rows=[]
  for t in tasks:
   ix=[i for i in indices if meta[i]['species']==s and meta[i]['task_id']==t and (selected is None or meta[i]['subject'] in selected[s])]
   if not ix:return None
   rows.append(responses[ix].mean(axis=0))
  result.append(rows)
 return np.asarray(result)
def time_matrices(groupmeans,tasks):
 return {t:matrix(groupmeans[:,[i for i,key in enumerate(tasks) if key.endswith(f'|{t}h')]].mean(axis=1)) for t in TIMES}
def historical_time_matrices(indices,meta,responses,species,selected=None):
 out={}
 for t in TIMES:
  v=[]
  for s in species:
   ix=[i for i in indices if meta[i]['species']==s and meta[i]['time_h']==t and (selected is None or meta[i]['subject'] in selected[s])]
   assert ix;v.append(responses[ix].mean(axis=0))
  out[t]=matrix(np.asarray(v))
 return out
def pooling(groupmeans,species,tasks):
 h=species.index(HUMAN);animal=[i for i,s in enumerate(species) if s!=HUMAN];six=[i for i in animal if TAX[species[i]][2]!='Primates']
 y=groupmeans[h];ynorm=np.mean(y*y,axis=1)
 records=[]
 for size in range(1,len(animal)+1):
  for selected in itertools.combinations(animal,size):
   p=groupmeans[list(selected)].mean(axis=0);loss=np.mean((p-y)**2,axis=1);pear=np.array([corr(a,b) for a,b in zip(p,y)])
   records.append({'species':[species[i] for i in selected],'size':size,'six_nonprimate_subset':set(selected)<=set(six),'macro_pearson':float(pear.mean()),'macro_mse':float(loss.mean()),'macro_recovery':float(np.mean(1-loss/ynorm)),'per_condition_pearson':pear,'per_condition_mse':loss,'per_condition_recovery':1-loss/ynorm})
 summaries={}
 for label,eligible in [('all_nine',records),('six_nonprimates',[r for r in records if r['six_nonprimate_subset']])]:
  curve=[]
  for size in sorted({r['size'] for r in eligible}):
   rows=[r for r in eligible if r['size']==size]
   curve.append({'size':size,'subsets':len(rows),'mean_pearson':np.mean([r['macro_pearson'] for r in rows]),'min_pearson':min(r['macro_pearson'] for r in rows),'max_pearson':max(r['macro_pearson'] for r in rows),'mean_mse':np.mean([r['macro_mse'] for r in rows]),'min_mse':min(r['macro_mse'] for r in rows),'max_mse':max(r['macro_mse'] for r in rows),'mean_recovery':np.mean([r['macro_recovery'] for r in rows]),'max_pearson_oracle_species':max(rows,key=lambda r:r['macro_pearson'])['species'],'min_mse_oracle_species':min(rows,key=lambda r:r['macro_mse'])['species']})
  summaries[label]=curve
 return records,summaries

def main():
 assert not OUT.exists();started=time.monotonic()
 audit=json.loads((PREP/'audit.json').read_text())
 for name,expected in audit['files'].items():assert sha(PREP/name)==expected
 with np.load(PREP/'observations.npz',allow_pickle=False) as z:
  complete=z['common_complete_mask'];primary_response=z['responses'][:,complete].copy();legacy_expr=np.log2(1+np.nan_to_num(z['tpm'],nan=0.));genes=z['gene_ids'][complete].copy()
 meta=json.loads((PREP/'response_metadata.json').read_text());indices=[i for i,r in enumerate(meta) if r['primary_shared_grid']]
 legacy_response=np.asarray([legacy_expr[r['treated_index']]-legacy_expr[r['control_index']] for r in meta]);del legacy_expr
 species=sorted({r['species'] for r in meta});tasks=sorted({meta[i]['task_id'] for i in indices},key=lambda k:(int(k.split('|')[-1][:-1]),int(k.split('|')[1][:-4])))
 subjects={s:sorted({meta[i]['subject'] for i in indices if meta[i]['species']==s}) for s in species}
 halves=[{s:values[:len(values)//2] for s,values in subjects.items()},{s:values[len(values)//2:] for s,values in subjects.items()}]
 pi,pj=np.triu_indices(len(species),k=1);tax=np.asarray([taxdist(species[i],species[j]) for i,j in zip(pi,pj)])
 groupmeans=subset_means(indices,meta,primary_response,species,tasks);assert groupmeans is not None
 rho=time_matrices(groupmeans,tasks)
 half_matrices=[time_matrices(subset_means(indices,meta,primary_response,species,tasks,selected),tasks) for selected in halves]
 fixed={t:compare(half_matrices[0][t],half_matrices[1][t],pi,pj,tax) for t in TIMES}
 fixed_mean=float(np.mean([fixed[t][0] for t in TIMES]))
 cross_time={f'{a}h_vs_{b}h':compare(rho[a],rho[b],pi,pj,tax) for a,b in itertools.combinations(TIMES,2)}
 rng=np.random.default_rng(SEED);draws=[];split_records=[]
 for draw in range(B):
  for attempt in range(1,10001):
   selected=[{},{}]
   for s,units in subjects.items():
    order=rng.permutation(units).tolist();k=len(units)//2;selected[0][s]=order[:k];selected[1][s]=order[k:]
   gm=[subset_means(indices,meta,primary_response,species,tasks,side) for side in selected]
   if all(v is not None for v in gm):break
  else:raise RuntimeError('No valid unit partition')
  matrices=[time_matrices(v,tasks) for v in gm]
  draws.append([compare(matrices[0][t],matrices[1][t],pi,pj,tax) for t in TIMES]);split_records.append({'draw':draw,'attempts':attempt,'halves':selected})
  if draw%100==0:print(json.dumps({'unit_split_draw_completed':draw}),flush=True)
 draws=np.asarray(draws)
 legacy_matrices=historical_time_matrices(list(range(len(meta))),meta,legacy_response,species)
 old_halves=[historical_time_matrices(list(range(len(meta))),meta,legacy_response,species,side) for side in halves]
 legacy_fixed={t:compare(old_halves[0][t],old_halves[1][t],pi,pj,tax) for t in TIMES}
 legacy_shared=time_matrices(subset_means(indices,meta,legacy_response,species,tasks),tasks)
 complete_rowweighted=historical_time_matrices(indices,meta,primary_response,species)
 records,curves=pooling(groupmeans,species,tasks)
 _,legacy_curves=pooling(subset_means(indices,meta,legacy_response,species,tasks),species,tasks)
 # Independent correlation and pooling spot checks, using explicit row operations.
 checks=[]
 for t in TIMES:
  x,y=half_matrices[0][t][pi,pj],half_matrices[1][t][pi,pj]
  gap=abs(float(spearmanr(x,y).statistic)-fixed[t][0]);assert gap<1e-12;checks.append(gap)
 for record in [records[0],records[8],records[13],records[100],records[-1]]:
  ids=[species.index(s) for s in record['species']]
  ps=[];ms=[]
  for j,task in enumerate(tasks):
   p=sum(groupmeans[i,j] for i in ids)/len(ids);h=groupmeans[species.index(HUMAN),j]
   ps.append(float(pearsonr(p,h).statistic));ms.append(float(sum((p-h)**2)/len(p)))
  gap=max(abs(np.mean(ps)-record['macro_pearson']),abs(np.mean(ms)-record['macro_mse']));assert gap<1e-11;checks.append(gap)
 OUT.mkdir()
 np.savez_compressed(OUT/'arrays.npz',species=np.asarray(species),conditions=np.asarray(tasks),genes=genes,group_means=groupmeans,**{f'rho_{t}h':rho[t] for t in TIMES},split_metrics=draws)
 save(OUT/'split_assignments.json',split_records);save(OUT/'subsets.json',records)
 report={'status':'complete','protocol_sha256':sha(ROOT/'PROTOCOL_v1.zh.md'),'code_sha256':sha(__file__),'prepared_audit_sha256':sha(PREP/'audit.json'),'source_axis_genes':25638,'fixed_complete_genes':len(genes),'species':species,'condition_ids':tasks,'perturbation_identities':1,'biological_sampling_units':44,'primary_response_rows':len(indices),'metric_order':['spearman_all45','spearman_cross_order35','partial_rank_taxonomic_distance'],'aggregation':'Within species-condition average available biological units; within time average three dose means equally. Same units recur across conditions.','fixed_sorted_unit_split':fixed,'fixed_sorted_mean_split_half':fixed_mean,'fixed_sorted_Spearman_Brown_extrapolation':2*fixed_mean/(1+fixed_mean),'random_split_sensitivity':{'seed':SEED,'draws':B,'same_unit_assignment_across_all_conditions':True,'metrics_by_time':{t:{'quantiles_2.5_50_97.5':np.quantile(draws[:,j,:],[.025,.5,.975],axis=0),'minimum':draws[:,j,:].min(axis=0),'maximum':draws[:,j,:].max(axis=0)} for j,t in enumerate(TIMES)},'mean_over_time_all45_quantiles':np.quantile(draws[:,:,0].mean(axis=1),[.025,.5,.975]),'rejected_partition_proposals':sum(r['attempts']-1 for r in split_records),'interpretation':'Variation of500 balanced partitions of the same44 units; not an independent species/stimulus population confidence interval.'},'cross_time':cross_time,'curves':curves,'single_source':[r for r in records if r['size']==1],'history_sensitivity':{'zero_filled_full_axis_all388_rowweighted_fixed_split':legacy_fixed,'legacy_mean_split_half':float(np.mean([legacy_fixed[t][0] for t in TIMES])),'zero_filled_full_axis_primary387_conditionweighted_cross_time':{f'{a}h_vs_{b}h':compare(legacy_shared[a],legacy_shared[b],pi,pj,tax) for a,b in itertools.combinations(TIMES,2)},'complete_axis_primary387_rowweighted_cross_time':{f'{a}h_vs_{b}h':compare(complete_rowweighted[a],complete_rowweighted[b],pi,pj,tax) for a,b in itertools.combinations(TIMES,2)},'zero_filled_full_axis_primary387_conditionweighted_curves':legacy_curves},'independent_spot_checks':{'checks':len(checks),'maximum_absolute_difference':max(checks),'passed':True},'runtime_seconds':time.monotonic()-started,'limits':['Author gene labels are retained; full cross-species one-to-one mapping unverified.','Complete-case axis may enrich conserved/easily mapped/detected genes; difference from zero-filled full axis changes gene population and missing treatment.','Rodent pools are biological units; mouse25/rat4 animals per pool not treated as independent samples.','Best/worst subsets are retrospective oracle descriptions.','Only oneLPS identity; nine dose/time conditions and repeated splits do not add independent stimuli.','No phylogeny-calibrated pvalue, general species ranking, clinical prediction or noise-ceiling claim.']}
 report['files']={p.name:sha(p) for p in OUT.iterdir() if p.is_file()};save(OUT/'report.json',report)
 print(json.dumps(plain({'fixed_split':fixed,'random_mean_split_quantiles':report['random_split_sensitivity']['mean_over_time_all45_quantiles'],'cross_time':cross_time,'legacy_mean':report['history_sensitivity']['legacy_mean_split_half'],'runtime_seconds':report['runtime_seconds']}),indent=2))
if __name__=='__main__':main()
