"""Independent raw-data and arithmetic verification; imports no production helpers."""
from pathlib import Path
import csv,gzip,hashlib,json,itertools,time,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
import numpy as np
from scipy.stats import spearmanr,rankdata
ROOT=Path('/hy-tmp/cross-species-feedback-20261108/biorxiv_ready_20261108/legacy_blood')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 started=time.monotonic();checks=0;values=0;maxgap=0.
 def check(flag,message):
  nonlocal checks
  assert bool(flag),message;checks+=1
 def close(a,b,message,tol=1e-10):
  nonlocal values,maxgap
  a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
  check(a.shape==b.shape,message+' shape');check(np.array_equal(np.isfinite(a),np.isfinite(b)),message+' finite')
  mask=np.isfinite(a);gap=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0
  check(gap<=tol,message+f' gap{gap}');values+=int(mask.sum());maxgap=max(maxgap,gap)
 prepared=ROOT/'prepared_v1';out=ROOT/'analysis_v1';audit=json.loads((prepared/'audit.json').read_text());report=json.loads((out/'report.json').read_text())
 for name,expected in audit['source_hashes'].items():check(sha(ROOT/'input'/name)==expected,name)
 for name,expected in audit['files'].items():check(sha(prepared/name)==expected,name)
 for name,expected in report['files'].items():check(sha(out/name)==expected,name)
 check(sha(ROOT/'PROTOCOL_v1.zh.md')==report['protocol_sha256'],'protocol')
 rawgenes=[];rawvalues=[]
 with gzip.open(ROOT/'input/GSE249010_processed_data_TPM.txt.gz','rt') as f:
  reader=csv.reader(f,delimiter='\t');samples=next(reader)[1:]
  for row in reader:
   rawgenes.append(row[0]);rawvalues.append([np.nan if x in {'NA','','null'} else float(x) for x in row[1:]])
 raw=np.asarray(rawvalues,dtype=float).T;meta=json.loads((prepared/'response_metadata.json').read_text());samplemeta=json.loads((prepared/'sample_metadata.json').read_text())
 primary=[i for i,r in enumerate(meta) if r['dose_ng_ml'] in {10,100,1000} and r['time_h'] in {2,6,24}]
 used=sorted(set([meta[i][key] for i in primary for key in ['treated_index','control_index']]))
 mask=np.isfinite(raw[used]).all(axis=0);log=np.log2(1+raw)
 response=np.asarray([log[r['treated_index']]-log[r['control_index']] for r in meta])
 with np.load(prepared/'observations.npz',allow_pickle=False) as z:
  close(z['tpm'],raw,'allraw',0);close(z['responses'],response,'allresponses',0)
  check(np.array_equal(z['common_complete_mask'],mask),'fixedaxis');check(np.array_equal(z['sample_ids'],samples),'sampleaxis');check(np.array_equal(z['gene_ids'],rawgenes),'geneaxis')
 check(mask.sum()==12276 and len(primary)==387,'fixedcohort')
 for r in meta:
  tr,ct=samplemeta[r['treated_index']],samplemeta[r['control_index']]
  check(tr['species']==ct['species']==r['species'],'speciespair')
  check(tr['subject']==ct['subject']==r['subject'],'subjectpair')
  check(tr['time']==ct['time']==str(r['time_h']),'timepair')
  check(ct['dose']=='0' and float(tr['dose'])==r['dose_ng_ml'],'dosepair')
 response=response[:,mask];sp=report['species'];tasks=report['condition_ids'];times=[2,6,24]
 def mean_matrix(selected=None):
  weights=np.zeros((len(sp)*len(tasks),len(meta)))
  for i,s in enumerate(sp):
   for j,t in enumerate(tasks):
    rows=[k for k in primary if meta[k]['species']==s and meta[k]['task_id']==t and (selected is None or meta[k]['subject'] in selected[s])]
    check(len(rows)>0,'eachspeciesconditionobserved');weights[i*len(tasks)+j,rows]=1/len(rows)
  return (weights@response).reshape(len(sp),len(tasks),-1)
 means=mean_matrix()
 with np.load(out/'arrays.npz',allow_pickle=False) as z:
  close(z['group_means'],means,'groupmeans');saved_split=z['split_metrics'].copy()
  for t in times:
   ti=[j for j,key in enumerate(tasks) if key.endswith(f'|{t}h')]
   actual=np.corrcoef(means[:,ti].mean(axis=1));close(z[f'rho_{t}h'],actual,'rho')
 rows=json.loads((out/'subsets.json').read_text());h=sp.index('Homo sapiens')
 for row in rows:
  p=means[[sp.index(s) for s in row['species']]].sum(axis=0)/row['size'];y=means[h]
  mse=np.mean((p-y)**2,axis=1);pear=[float(np.corrcoef(a,b)[0,1]) for a,b in zip(p,y)];rec=1-mse/np.mean(y*y,axis=1)
  close(row['per_condition_mse'],mse,'subsetmse');close(row['per_condition_pearson'],pear,'subsetpearson');close(row['per_condition_recovery'],rec,'subsetrecovery')
  close([row['macro_mse'],row['macro_pearson'],row['macro_recovery']],[np.mean(mse),np.mean(pear),np.mean(rec)],'subsetmacro')
 check(len(rows)==511,'all511subsets')
 assignments=json.loads((out/'split_assignments.json').read_text());check(len(assignments)==500,'all500splits')
 unitsets={s:{meta[i]['subject'] for i in primary if meta[i]['species']==s} for s in sp};pi,pj=np.triu_indices(len(sp),1)
 for draw,record in enumerate(assignments):
  for s in sp:
   a,b=map(set,[record['halves'][0][s],record['halves'][1][s]])
   check(not (a&b) and a|b==unitsets[s],'unitpartition');check(len(a)==len(unitsets[s])//2,'balancedhalf')
  # Audit all assignments but independent numerical split reconstruction on50 fixed draws.
  if draw%10==0:
   gm=[mean_matrix(side) for side in record['halves']]
   for j,t in enumerate(times):
    ids=[k for k,key in enumerate(tasks) if key.endswith(f'|{t}h')]
    matrices=[np.corrcoef(v[:,ids].mean(axis=1)) for v in gm]
    point=float(spearmanr(matrices[0][pi,pj],matrices[1][pi,pj]).statistic)
    close(point,saved_split[draw,j,0],'splitindependent')
 close(np.quantile(saved_split[:,:,0].mean(axis=1),[.025,.5,.975]),report['random_split_sensitivity']['mean_over_time_all45_quantiles'],'splitquantiles')
 result={'status':'pass','checks_passed':checks,'checks_failed':0,'numeric_values_compared':values,'max_abs_difference':maxgap,'full_raw_and_response_reconstruction':True,'all511_subsets_independently_scored':True,'all500_unit_partitions_verified':True,'independent_numerical_split_draws':50,'scope':'Data identity, missing mask, exact control correspondence, condition weighting, pooling and descriptive split sensitivity. Does not validate biological independence or orthology generation.','auditor_sha256':sha(__file__),'prepared_audit_sha256':sha(prepared/'audit.json'),'analysis_report_sha256':sha(out/'report.json'),'elapsed_seconds':time.monotonic()-started}
 (ROOT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
