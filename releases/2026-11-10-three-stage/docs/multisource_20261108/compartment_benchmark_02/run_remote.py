"""Fixed exploratory SRSF source-compartment comparisons; all arrays stay remote."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,re,time
import numpy as np
import torch

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/multisource/compartment_benchmark_02')
DATA=Path('/hy-tmp/cross-species-feedback-20261108/multisource/genetic_expansion_02')
OUT=ROOT/'v1'
TARGETS=('SRSF1','SRSF5','SRSF7')
SOURCES=('whole_cell','cytoplasmic')
RULES=('zero','human_history_mean','whole_raw','cyto_raw','equal_raw','whole_scaled','cyto_scaled','equal_scaled','historical_raw_source_selection')
PATTERN={'human':re.compile(r'^ENSG[0-9]+(?:\.[0-9]+)?$'),'mouse':re.compile(r'^ENSMUSG[0-9]+(?:\.[0-9]+)?$')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exclusion_reason(gene):
 if '_PAR_' in gene:return 'PAR_suffix'
 if gene.startswith('ENSGR'):return 'ENSGR'
 if gene.isdigit():return 'numeric_identifier'
 return 'other_noncanonical_identifier'

def strict_native(genes,values,side,sample_ids,exclusions):
 genes=list(map(str,genes));values=np.asarray(values,float)
 assert values.ndim==2 and values.shape==(len(genes),len(sample_ids))
 assert np.isfinite(values).all() and (values>=0).all()
 accepted=np.array([bool(PATTERN[side].fullmatch(g)) for g in genes])
 ids=[g.split('.')[0] for g,ok in zip(genes,accepted) if ok]
 assert len(ids)==len(set(ids)), 'Strict version-stripped native ID conflict'
 native=values[accepted];totals=native.sum(axis=0)
 assert (totals>0).all()
 log_values=np.log2(1+native*(1e6/totals)[None,:])
 # Per-library positive scaling must cancel from the declared composition transform.
 scaled=native*np.linspace(.13,7,len(sample_ids))[None,:]
 check=np.log2(1+scaled*(1e6/scaled.sum(axis=0))[None,:])
 invariance=float(np.max(np.abs(check-log_values)))
 assert invariance<1e-10
 rejects=Counter(exclusion_reason(g) for g,ok in zip(genes,accepted) if not ok)
 for sample in sample_ids:
  for row,(gene,ok) in enumerate(zip(genes,accepted)):
   if not ok:exclusions.append((sample,row,gene,exclusion_reason(gene)))
 audit={'side':side,'original_gene_rows':len(genes),'strict_native_gene_rows':len(ids),'excluded_by_identifier_reason':dict(rejects),'excluded_examples':[g for g,ok in zip(genes,accepted) if not ok][:12],'strict_id_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'per_library_scale_invariance_max_abs_difference':invariance}
 return ids,log_values,totals,audit

def cpu_metrics(p,y):
 pc=p-p.mean(axis=-1,keepdims=True);yc=y-y.mean(axis=-1,keepdims=True)
 den=np.sqrt(np.square(pc).sum(axis=-1)*np.square(yc).sum(axis=-1))
 pearson=np.divide((pc*yc).sum(axis=-1),den,out=np.full(den.shape,np.nan),where=den>0)
 return {'mse':np.mean((p-y)**2,axis=-1),'pearson':pearson,'predicted_rms':np.sqrt(np.mean(p*p,axis=-1)),'target_rms':np.sqrt(np.mean(y*y,axis=-1))}

def main():
 assert not OUT.exists(),'Use an empty output directory'
 total_start=time.monotonic()
 frozen=json.loads((ROOT/'input_freeze.json').read_text())
 for row in frozen['files']:assert sha(ROOT/'input'/row['remote_filename'])==row['sha256']
 selection=json.loads((ROOT/'input/compartment_selection.json').read_text());entries=selection['entries']
 bridge=json.loads((ROOT/'input/compartment_sample_bridge.json').read_text())
 assert len({r['perturb_id'] for r in entries})==len(entries)==16
 for entry in entries:
  assert entry['sample_identity_and_role_verified'] and entry['condition']=='KD'
  assert len(entry['controls'])==len(entry['perturbs'])==2 and not set(entry['controls'])&set(entry['perturbs'])
  for role,key in [('control','controls'),('treated','perturbs')]:
   observed={r['srx'] for r in bridge if r['perturb_id']==entry['perturb_id'] and r['role']==role}
   assert observed==set(entry[key]),(entry['perturb_id'],role)
 samples={};sample_audit=[];input_hashes={};exclusions=[]
 human_samples=sorted({s for e in entries if e['species']=='Homo sapiens' for s in e['controls']+e['perturbs']})
 for srx in human_samples:
  audit_path=DATA/'compartment_human'/(srx+'_audit.json');record=json.loads(audit_path.read_text())
  path=Path(record['audit']['native_file']);assert sha(path)==record['audit']['native_sha256']
  assert sha(record['raw_file'])==record['sha256']
  input_hashes[str(audit_path)]=sha(audit_path);input_hashes[str(path)]=record['audit']['native_sha256'];input_hashes[record['raw_file']]=record['sha256']
  with np.load(path,allow_pickle=False) as a:
   quantities=a['quantities'].tolist();counts=a['values'][:,quantities.index('expected_count')].copy();genes=a['gene_ids'].copy()
  ids,logs,totals,check=strict_native(genes,counts[:,None],'human',[srx],exclusions)
  samples[srx]={'side':'human','genes':ids,'values':logs[:,0]}
  sample_audit.append({'sample':srx,'gsm':record['gsm'],'model':record['cell_model'],'reported_strict_estimated_count_mass':float(totals[0]),'quantity':'RSEM expected_count composition on all strict ENSG rows','is_sequencing_depth':False,**check})
 p19dir=DATA/'balanced_splicing/author_p19/audit'
 manifest=json.loads((p19dir/'native/manifest.json').read_text());input_hashes[str(p19dir/'native/manifest.json')]=sha(p19dir/'native/manifest.json')
 for compartment in SOURCES:
  record=next(r for r in manifest['files'] if r['compartment']==compartment and r['kind']=='estimated_counts')
  for field,hashfield in [('native_npz','npz_sha256'),('source_path','source_sha256'),('columns_path','columns_sha256'),('axis_path','axis_sha256')]:
   assert sha(record[field])==record[hashfield];input_hashes[record[field]]=record[hashfield]
  with np.load(record['native_npz'],allow_pickle=False) as a:
   counts=a['author_values'].copy();genes=a['native_gene_ids'].copy();srxs=a['experiment_ids'].tolist();gsms=a['gsm'].tolist()
   assert a['observed_mask'].all()
  ids,logs,totals,check=strict_native(genes,counts,'mouse',srxs,exclusions)
  for i,srx in enumerate(srxs):
   assert srx not in samples
   samples[srx]={'side':'mouse','genes':ids,'values':logs[:,i]}
   sample_audit.append({'sample':srx,'gsm':gsms[i],'model':'P19','measurement_compartment':compartment,'reported_strict_estimated_count_mass':float(totals[i]),'quantity':'Author-reported estimated-count composition on all strict ENSMUSG rows; raw versus library-scaled export unresolved','is_sequencing_depth':False,**check})
 # Conservative high-confidence relation uniqueness across every eligible mapping row.
 h2m=defaultdict(set);m2h=defaultdict(set);returned=0;eligible_rows=0
 with (ROOT/'input/target_orthology.tsv').open() as f:
  for row in csv.reader(f,delimiter='\t'):
   assert len(row)==6;returned+=1
   h,hs,m,ms,kind,confidence=row
   if kind!='ortholog_one2one' or confidence!='1' or not re.fullmatch(r'ENSG[0-9]+',h) or not re.fullmatch(r'ENSMUSG[0-9]+',m):continue
   h2m[h].add(m);m2h[m].add(h);eligible_rows+=1
 maps={h:next(iter(mm)) for h,mm in h2m.items() if len(mm)==1 and len(m2h[next(iter(mm))])==1}
 human_support=set.intersection(*(set(v['genes']) for v in samples.values() if v['side']=='human'))
 mouse_support=set.intersection(*(set(v['genes']) for v in samples.values() if v['side']=='mouse'))
 axis=sorted((h,m) for h,m in maps.items() if h in human_support and m in mouse_support)
 assert len(axis)>1000 and len({r[0] for r in axis})==len(axis)==len({r[1] for r in axis})
 selected_samples=sorted({s for e in entries for s in e['controls']+e['perturbs']})
 aligned={}
 for srx in selected_samples:
  row=samples[srx];ix={g:i for i,g in enumerate(row['genes'])};col=0 if row['side']=='human' else 1
  aligned[srx]=row['values'][[ix[pair[col]] for pair in axis]]
 response={e['perturb_id']:np.mean([aligned[s] for s in e['perturbs']],axis=0)-np.mean([aligned[s] for s in e['controls']],axis=0) for e in entries}
 def human_entry(model,target,role='primary'):
  result=[e for e in entries if e['species']=='Homo sapiens' and e['cell_model']==model and e['target']==target and e['selection_role']==role]
  assert len(result)==1;return result[0]
 def mouse_entries(target):
  result=[]
  for compartment in SOURCES:
   matched=[e for e in entries if e['species']=='Mus musculus' and e['target']==target and e['measurement_compartment']==compartment]
   assert len(matched)==1;result.append(matched[0])
  return result
 cases=[]
 for model in ('HepG2','K562'):
  for held in TARGETS:cases.append({'group':'primary','model':model,'target':held,'human':human_entry(model,held),'training_targets':[t for t in TARGETS if t!=held]})
 cases.append({'group':'supplement_SRSF3','model':'HepG2','target':'SRSF3','human':human_entry('HepG2','SRSF3','supplement_SRSF3_HepG2_only'),'training_targets':list(TARGETS)})
 cases.append({'group':'alternative_HepG2_SRSF5_batch','model':'HepG2','target':'SRSF5','human':human_entry('HepG2','SRSF5','sensitivity_alternative_HepG2_SRSF5_batch'),'training_targets':['SRSF1','SRSF7']})
 for held in ('SRSF1','SRSF7'):cases.append({'group':'K562_without_lowKD_SRSF5','model':'K562','target':held,'human':human_entry('K562',held),'training_targets':[t for t in ('SRSF1','SRSF7') if t!=held]})
 all_predictions=[];all_targets=[];records=[]
 for case in cases:
  training_h=[human_entry(case['model'],t) for t in case['training_targets']]
  train_h=np.asarray([response[e['perturb_id']] for e in training_h])
  training_s=[mouse_entries(t) for t in case['training_targets']]
  train_s=np.asarray([[response[e['perturb_id']] for e in pair] for pair in training_s])
  denom=(train_s**2).sum(axis=(0,2));num=(train_s*train_h[:,None]).sum(axis=(0,2));beta=np.zeros(2)
  valid=denom>1e-12;beta[valid]=np.maximum(num[valid]/denom[valid],0.)
  train_raw_losses=((train_s-train_h[:,None])**2).mean(axis=(0,2));chosen=int(np.argmin(train_raw_losses))
  held_s_entries=mouse_entries(case['target']);held_s=np.asarray([response[e['perturb_id']] for e in held_s_entries]);human=response[case['human']['perturb_id']]
  preds={'zero':np.zeros(len(axis)),'human_history_mean':train_h.mean(axis=0),'whole_raw':held_s[0],'cyto_raw':held_s[1],'equal_raw':held_s.mean(axis=0),'whole_scaled':beta[0]*held_s[0],'cyto_scaled':beta[1]*held_s[1],'equal_scaled':(beta[:,None]*held_s).mean(axis=0),'historical_raw_source_selection':held_s[chosen]}
  assert case['target'] not in case['training_targets']
  if case['target']=='SRSF5':assert all(e['target']!='SRSF5' for e in training_h)
  flags=[]
  if case['model']=='K562' and case['target']=='SRSF5':flags.append('Low reported shRNA knockdown23.1/20.5%; retained in primary results')
  if case['group']=='alternative_HepG2_SRSF5_batch':flags.append('Alternative human SRSF5 experimental batch; all SRSF5 records excluded from fitting')
  if case['group']=='K562_without_lowKD_SRSF5':flags.append('Predeclared low-efficacy sensitivity; only two remaining related target identities')
  record={'case_id':case['group']+'|'+case['model']+'|'+case['target'],'analysis_group':case['group'],'human_model':case['model'],'held_target':case['target'],'human_query_contrast':case['human']['perturb_id'],'source_query_contrasts':[e['perturb_id'] for e in held_s_entries],'fit_targets':case['training_targets'],'fit_human_contrasts':[e['perturb_id'] for e in training_h],'fit_source_contrasts':[[e['perturb_id'] for e in pair] for pair in training_s],'source_order':list(SOURCES),'beta':beta.tolist(),'fit_numerator':num.tolist(),'fit_denominator':denom.tolist(),'historical_raw_training_mse':train_raw_losses.tolist(),'historical_raw_selected_source':SOURCES[chosen],'quality_flags':flags,'query_human_used_for_fitting':False}
  records.append(record)
  for rule in RULES:all_predictions.append(preds[rule]);all_targets.append(human)
 predictions=np.asarray(all_predictions);targets=np.asarray(all_targets)
 cpu=cpu_metrics(predictions,targets)
 assert predictions.shape==(90,len(axis)) and np.isfinite(predictions).all()
 assert torch.cuda.is_available()
 torch.cuda.init();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();gpu_start=time.monotonic()
 p=torch.as_tensor(predictions,device='cuda',dtype=torch.float64);y=torch.as_tensor(targets,device='cuda',dtype=torch.float64)
 pc=p-p.mean(dim=-1,keepdim=True);yc=y-y.mean(dim=-1,keepdim=True);den=(pc.square().sum(dim=-1)*yc.square().sum(dim=-1)).sqrt()
 gpu={'mse':(p-y).square().mean(dim=-1),'pearson':torch.where(den>0,(pc*yc).sum(dim=-1)/den,torch.full_like(den,float('nan'))),'predicted_rms':p.square().mean(dim=-1).sqrt(),'target_rms':y.square().mean(dim=-1).sqrt()}
 result={k:v.cpu().numpy() for k,v in gpu.items()};torch.cuda.synchronize();gpu_seconds=time.monotonic()-gpu_start
 maxdiff=0.;checked=0;finite_checked=0
 for metric in cpu:
  assert np.array_equal(np.isfinite(cpu[metric]),np.isfinite(result[metric]))
  finite=np.isfinite(cpu[metric]);diff=float(np.max(np.abs(cpu[metric][finite]-result[metric][finite]))) if finite.any() else 0.
  assert diff<=1e-10,(metric,diff);maxdiff=max(maxdiff,diff);checked+=len(finite);finite_checked+=int(finite.sum())
 for ci,record in enumerate(records):
  metrics={}
  for ri,rule in enumerate(RULES):
   index=ci*len(RULES)+ri
   metrics[rule]={metric:float(values[index]) if np.isfinite(values[index]) else None for metric,values in result.items()}
  record['rules']=metrics
  zero=metrics['zero']['mse'];history=metrics['human_history_mean']['mse']
  record['relative_mse_reduction_over_zero']={rule:1-m['mse']/zero if zero>0 else None for rule,m in metrics.items()}
  record['relative_mse_reduction_over_human_history']={rule:1-m['mse']/history if history>0 else None for rule,m in metrics.items()}
 grouped={}
 for record in records:grouped.setdefault(record['analysis_group']+'|'+record['human_model'],[]).append(record)
 macro=[]
 for group,rr in grouped.items():
  scores={rule:float(np.mean([r['rules'][rule]['mse'] for r in rr])) for rule in RULES}
  macro.append({'group':group,'query_count':len(rr),'targets':[r['held_target'] for r in rr],'macro_mse':scores,'rule':'Equal average over listed query targets; groups kept separate'})
 OUT.mkdir()
 with (OUT/'gene_axis.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['human_gene_id','mouse_gene_id']);w.writerows(axis)
 with (OUT/'excluded_native_rows.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['sample','native_row_zero_based','original_gene_id','reason']);w.writerows(exclusions)
 np.savez_compressed(OUT/'sample_log_composition.npz',sample_ids=np.asarray(selected_samples),values=np.stack([aligned[s] for s in selected_samples]),human_gene_ids=np.asarray([a[0] for a in axis]),mouse_gene_ids=np.asarray([a[1] for a in axis]))
 np.savez_compressed(OUT/'responses.npz',contrast_ids=np.asarray([e['perturb_id'] for e in entries]),responses=np.stack([response[e['perturb_id']] for e in entries]),human_gene_ids=np.asarray([a[0] for a in axis]),mouse_gene_ids=np.asarray([a[1] for a in axis]))
 np.savez_compressed(OUT/'predictions.npz',predictions=predictions.reshape(len(records),len(RULES),len(axis)),human_targets=targets.reshape(len(records),len(RULES),len(axis))[:,0],case_ids=np.asarray([r['case_id'] for r in records]),rules=np.asarray(RULES),human_gene_ids=np.asarray([a[0] for a in axis]))
 report={'evidence_level':'E1_exploratory_source_measurement_compartment','protocol_sha256':sha(ROOT/'PROTOCOL_v1.zh.md'),'script_sha256':sha(__file__),'input_freeze_sha256':sha(ROOT/'input_freeze.json'),'input_metadata':frozen,'input_expression_files':input_hashes,'fixed_gene_axis':{'genes':len(axis),'mapping_rows':returned,'eligible_mapping_rows':eligible_rows,'unique_bidirectional_pairs_before_presence':len(maps),'human_ambiguous_partners':sum(len(v)>1 for v in h2m.values()),'mouse_ambiguous_partners':sum(len(v)>1 for v in m2h.values()),'strict_human_support':len(human_support),'strict_mouse_support':len(mouse_support),'response_dependent_filtering':False},'normalization':'log2(1+1e6*reported_estimated_count/sum_of_reported_estimated_count_over_all_strict_native_gene_IDs_per_file). P19 is author-gene-set estimated-count composition; mass is not sequencing depth.','selected_unique_samples':len(selected_samples),'normalization_sample_audit':sample_audit,'contrasts':entries,'cases':records,'macro_results':macro,'source_interpretation':'Two measurement compartments of the same mouse P19 cell model, with fraction pairing labels; not two animal species or independently sourced cell models.','quality_flags':{'K562_SRSF5':selection['SRSF5_K562_quality_flag'],'HepG2_SRSF5':selection['SRSF5_HepG2_choice']},'limitations':['Human shRNA endpoint time unresolved; mouse P19 esiRNA sampled36h.','Cell line/species/study/quantifier and perturbation method are confounded across human and mouse. HepG2/K562 and P19 are distinct tumour-derived cell contexts.','P19 whole/cyto fractions derive from the same sample design; A/B labels do not establish new independent donors. Shared GFP controls and overlapping LOTO training sets limit independence.','P19 raw versus scaled estimated-count export field unresolved; positive library-wide scaling cancels from this composition, but gene-specific quantifier and length effects do not.','Only three related primary SRSF targets. SRSF3 and alternative SRSF5 are supplementary, not a broad independent confirmation.','Source choice is evaluated descriptively on the observed tasks.'],'runtime':{'device_name':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'dtype':'float64','gpu_metric_wall_seconds':gpu_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metric_prediction_rows':len(predictions),'resamples':0,'total_wall_seconds':time.monotonic()-total_start},'cpu_gpu_validation':{'metric_value_or_finiteness_checks':checked,'finite_value_checks':finite_checked,'max_abs_difference':maxdiff,'tolerance':1e-10,'passed':True},'output_files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'genes':len(axis),'cases':len(records),'macro_results':macro,'runtime':report['runtime'],'cpu_gpu_validation':report['cpu_gpu_validation']},indent=2))

if __name__=='__main__':main()
