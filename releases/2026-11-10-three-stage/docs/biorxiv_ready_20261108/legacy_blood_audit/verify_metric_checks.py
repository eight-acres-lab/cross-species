"""Independent verification of frozen retrieval and held-record references."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,math,os,time,itertools
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
import numpy as np
ROOT=Path('/hy-tmp/cross-species-feedback-20261108')
MET=ROOT/'biorxiv_ready_20261108/metric_checks_v1'
PAIR=ROOT/'crossspecies_direct_03'
EXT=ROOT/'external_transfer_04'
ARMS=['prior','correct_fit_correct_query','correct_fit_wrong_query','wrong_fit_correct_query','wrong_fit_wrong_query']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
 started=time.monotonic();checks=0;numeric=0;maxgap=0.
 def check(ok,label):
  nonlocal checks
  assert bool(ok),label;checks+=1
 def close(a,b,label,tol=1e-10):
  nonlocal numeric,maxgap
  x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
  check(x.shape==y.shape,label+'shape');check(np.array_equal(np.isfinite(x),np.isfinite(y)),label+'finite')
  m=np.isfinite(x);gap=float(np.max(np.abs(x[m]-y[m]))) if m.any() else 0.
  check(gap<=tol,label+str(gap));numeric+=int(m.sum());maxgap=max(maxgap,gap)
 metricmanifest=json.loads((MET/'manifest.json').read_text())
 for name,expected in metricmanifest['files'].items():check(sha(MET/name)==expected,'metricsfile:'+name)
 bundle=load(PAIR/'bundle_v1/bundle.npz');meta=json.loads((PAIR/'bundle_v1/metadata.json').read_text())
 retrieved=json.loads((MET/'retrieval_report.json').read_text())['reports'];retrieval_audits=[]
 units={'M':Decimal('1'),'mM':Decimal('0.001'),'uM':Decimal('0.000001'),'μM':Decimal('0.000001'),'nM':Decimal('0.000000001')}
 for rr in retrieved:
  regime=rr['regime'];run=PAIR/'runs'/(regime+'_v1');fit=json.loads((run/'fit_manifest.json').read_text())
  check(sha(run/'fit_manifest.json')==rr['frozen_fit_manifest_sha256'],'fitmanifest')
  key=[]
  for row in meta:
   v=[Decimal(str(row['time_h'])),Decimal(str(row['rat_dose_value']))*units[row['rat_dose_unit']]]
   if regime=='paired_exposure':v.append(Decimal(str(row['human_dose_value']))*units[row['human_dose_unit']])
   key.append(tuple(v))
  output=[]
  stored=json.loads((MET/(regime+'_retrieval_rows.json')).read_text())
  expected={(r['fold'],r['seed'],r['arm'],r['row']):r for r in stored}
  check(len(expected)==len(stored),'unique_metric_rows')
  for fold in fit['outer_folds']:
   path=run/f'fold_{fold}_base.npz';check(sha(path)==fit['output_files'][path.name],'basehash');base=load(path)
   test=base['test_indices'];center=base['target_mean'];targets=bundle['target'][test].astype(float)-center
   groups={}
   for i,index in enumerate(test):groups.setdefault(key[int(index)],[]).append(i)
   for seed in fit['permutation_seeds']:
    path=run/f'fold_{fold}_seed_{seed}.npz';check(sha(path)==fit['output_files'][path.name],'seedhash');seeddata=load(path)
    preds={'prior':base['prior_prediction'],'correct_fit_correct_query':base['correct_prediction'],**{arm:seeddata[arm] for arm in ARMS[2:]}}
    for indices in groups.values():
     n=len(indices);check(n>=2,'candidate_min2');check(len(set(bundle['compounds'][test[indices]]))==n,'compound_unique_candidates')
     candidate=targets[indices]
     for arm,pred in preds.items():
      query=pred[indices].astype(float)-center
      # np.corrcoef independently centers/scales both profile collections.
      correlation=np.corrcoef(np.vstack([query,candidate]))[:n,n:]
      qnorm=np.linalg.norm(query-query.mean(axis=1,keepdims=True),axis=1)
      ynorm=np.linalg.norm(candidate-candidate.mean(axis=1,keepdims=True),axis=1)
      for pos,local in enumerate(indices):
       row=int(test[local]);e=expected[(fold,seed,arm,row)]
       valid=bool(qnorm[pos]>1e-12 and np.all(ynorm>1e-12));check(valid==e['defined'],'defined')
       true=correlation[pos,pos]
       if valid:
        higher=sum(v>true+1e-12 for v in correlation[pos]);tied=sum(abs(v-true)<=1e-12 for v in correlation[pos]);check(tied>=1,'tie_group_nonempty')
        top=1/tied if higher==0 else 0.;mrr=math.fsum(1/rank for rank in range(higher+1,higher+tied+1))/tied
       else:top=mrr=float('nan')
       chance=1/n;chance_mrr=math.fsum(1/k for k in range(1,n+1))/n
       close([top,mrr,chance,chance_mrr],[e['top1'],e['mrr'],e['chance_top1'],e['chance_mrr']],'retrievalrow')
       check(e['compound']==str(bundle['compounds'][row]) and e['candidates']==n,'query_identity_and_candidate_size')
       output.append({'row':row,'compound':e['compound'],'seed':seed,'arm':arm,'defined':valid,'top1':top,'mrr':mrr,'chance_top1':chance,'chance_mrr':chance_mrr})
  common=set(r['row'] for r in output)
  for r in output:
   if not r['defined']:common.discard(r['row'])
  check(len(common)==rr['common_defined_conditions'],'common_defined_count')
  for arm in ARMS:
   for summary in rr['summary'][arm]:
    rows=[r for r in output if r['arm']==arm and r['seed']==summary['seed'] and r['row'] in common]
    groups={}
    for row in rows:groups.setdefault(row['compound'],[]).append(row)
    check(len(rows)==summary['conditions'] and len(groups)==summary['compounds'],'aggregate_counts')
    for metric in ['top1','mrr','chance_top1','chance_mrr']:
     value=math.fsum(math.fsum(row[metric] for row in rows)/len(rows) for rows in groups.values())/len(groups)
     close(value,summary[metric],'compound_macro_retrieval')
  retrieval_audits.append({'regime':regime,'all_query_arm_seed_rows':len(output),'common_defined_conditions':len(common),'all_folds_all_seeds_all_five_arms_recomputed':True})
 inp=EXT/'inputs_v1';run=EXT/'run_v1';im=json.loads((inp/'manifest.json').read_text());fm=json.loads((run/'fit_manifest.json').read_text())
 for filename in ['external_arms.npz','external_truth.npz','task_metadata.json']:check(sha(inp/filename)==im['files'][filename],'externalinputhash')
 check(sha(run/'external_predictions.npz')==fm['output_files']['external_predictions.npz'],'externalpredhash')
 arm=load(inp/'external_arms.npz');truth=load(inp/'external_truth.npz');pred=load(run/'external_predictions.npz')
 tasks=json.loads((inp/'task_metadata.json').read_text());report=json.loads((MET/'measurement_reference_report.json').read_text());stored=json.loads((MET/'measurement_reference_rows.json').read_text())
 bykey={(r['task_id'],r['held_treated_sample'],r['held_control_sample']):r for r in stored}
 pos={int(index):i for i,index in enumerate(arm['original_arm_indices'])};rules=['zero','human_mean','rat_copy','scalar_nointercept','chemical_context','source_plus_context','fused_wrong_source_120','fused_wrong_source_201'];identities=[]
 for taskindex,task in enumerate(tasks):
  group=task['sides']['human'];ti=[pos[i] for i in group['treated_arm_rows']];ci=[pos[i] for i in group['control_arm_rows']]
  check(len(ti)==len(ci)==3 and not(set(ti)&set(ci)),'3plus3disjointarms')
  treated=arm['arm_logcpm'][ti];control=arm['arm_logcpm'][ci];mu=treated.mean(0)-control.mean(0);close(mu,truth['target'][taskindex],'fulltarget')
  targetrows=[];recordmetrics=[]
  for i,j in itertools.product(range(3),repeat=2):
   target=treated[i]-control[j];ref=(treated.sum(0)-treated[i])/2-(control.sum(0)-control[j])/2
   close(ref,(3*mu-target)/2,'complement_reference_identity')
   key=(task['perturbation_code'],str(arm['sample_ids'][ti[i]]),str(arm['sample_ids'][ci[j]]));e=bykey[key]
   check(e['reference_treated_samples']==[str(arm['sample_ids'][ti[k]]) for k in range(3) if k!=i],'treated_reference_disjoint')
   check(e['reference_control_samples']==[str(arm['sample_ids'][ci[k]]) for k in range(3) if k!=j],'control_reference_disjoint')
   metrics={}
   for name,p in {'same_condition_human_reference':ref,**{name:pred[name][taskindex] for name in rules}}.items():
    error=math.fsum(float(d*d) for d in p-target)/len(target)
    pear=float(np.corrcoef(p,target)[0,1]) if np.std(p)>0 and np.std(target)>0 else float('nan')
    close([error,pear],[e['scores'][name]['mse'],e['scores'][name]['pearson']],'heldrecord_scores')
    metrics[name]={'mse':error,'pearson':pear}
   targetrows.append(target);recordmetrics.append(metrics)
  variance=float(np.mean((np.asarray(targetrows)-mu)**2));s=report['tasks'][taskindex]
  close(variance,s['held_response_variation'],'heldvariation')
  for name in ['same_condition_human_reference',*rules]:
   avg=math.fsum(row[name]['mse'] for row in recordmetrics)/9
   pears=[row[name]['pearson'] for row in recordmetrics if np.isfinite(row[name]['pearson'])];avgpear=math.fsum(pears)/len(pears) if pears else float('nan')
   close([avg,avgpear],[s['rules'][name]['mean_mse'],s['rules'][name]['mean_defined_pearson']],'heldmacro')
   if name=='same_condition_human_reference':close(avg,2.25*variance,'referenceMSE2.25V')
   else:close(avg,np.mean((pred[name][taskindex]-mu)**2)+variance,'fixedmodelmeanplusvariation')
  identities.append({'task':task['perturbation_code'],'human_reference_mean_mse':s['rules']['same_condition_human_reference']['mean_mse'],'zero_mean_mse':s['rules']['zero']['mean_mse'],'V':variance,'reference_MSE_equals_2.25V':True,'fixed_model_MSE_equals_mean_endpoint_MSE_plus_V':True})
 check(len(bykey)==27,'all27heldrecords')
 result={'status':'pass','checks_passed':checks,'checks_failed':0,'numeric_values_compared':numeric,'max_abs_difference':maxgap,'retrieval':retrieval_audits,'measurement_reference':identities,'all27_held_record_combinations_verified':True,'production_functions_imported':False,'metric_manifest_sha256':sha(MET/'manifest.json'),'auditor_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-started,'interpretation':['The human reference is a complementary2+2 measurement-record reference with unknown donor identity; it is not an independent biological positive control or noise ceiling.','All9partitions of each3+3task overlap and satisfy an exact complement algebra, rather than creating9 independent experiments.','High chance in exact-exposure retrieval must be reported next to accuracy; outcomes only enter scoring.','This check proves implementation and source correspondence, not generalization or mechanistic specificity.']}
 (MET/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
