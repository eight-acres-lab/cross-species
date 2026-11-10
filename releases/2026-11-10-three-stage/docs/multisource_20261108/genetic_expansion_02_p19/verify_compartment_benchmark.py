"""Independent SRSF audit: re-read deposited gzip files, never import benchmark code."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,gzip,hashlib,json,re
import numpy as np

BASE=Path('/hy-tmp/cross-species-feedback-20261108/multisource')
CHECKS=0
MAXDIFF=0.0
ARRAY_VALUES=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(cond,msg):
 global CHECKS
 CHECKS+=1
 if not cond:raise AssertionError(msg)

def close(a,b,msg,tol=1e-10):
 global MAXDIFF,ARRAY_VALUES
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 check(a.shape==b.shape,msg+' shape')
 check(np.array_equal(np.isfinite(a),np.isfinite(b)),msg+' finite mask')
 mask=np.isfinite(a);diff=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.
 check(diff<=tol,msg+' numerical difference '+str(diff))
 MAXDIFF=max(MAXDIFF,diff);ARRAY_VALUES+=int(mask.sum())

def build_inputs(root):
 selection=json.loads((root/'compartment_selection.json').read_text())
 entries=selection['entries'];sources=json.loads((root/'human_sources.json').read_text())['files']
 wanted={s for e in entries if e['species']=='Homo sapiens' for s in e['controls']+e['perturbs']}
 check(len(wanted)==26,'26 unique human libraries')
 available={r['srx']:r for r in sources};check(set(available)==wanted,'raw source inventory matches selection')
 human={};input_details=[];raw_sha={}
 for srx in sorted(wanted):
  e=available[srx];p=Path(e['raw_file']);check(sha(p)==e['sha256'],'human raw SHA '+srx)
  raw_sha[str(p)]=e['sha256'];genes=[];values=[];excluded=Counter();all_ids=[]
  with gzip.open(p,'rt') as f:
   r=csv.DictReader(f,delimiter='\t');check('expected_count' in r.fieldnames,'RSEM expected_count column')
   for row in r:
    gene=row['gene_id'];all_ids.append(gene)
    if re.fullmatch(r'ENSG[0-9]+(?:\.[0-9]+)?',gene):
     genes.append(gene.split('.')[0]);values.append(float(row['expected_count']))
    else:
     excluded['PAR' if '_PAR_' in gene else 'ENSGR' if gene.startswith('ENSGR') else 'numeric' if gene.isdigit() else 'other']+=1
  check(len(all_ids)==len(set(all_ids)),'human raw ID uniqueness')
  check(len(genes)==len(set(genes)),'human version stripping uniqueness')
  v=np.asarray(values,dtype=np.float64);check(np.isfinite(v).all() and (v>=0).all() and v.sum()>0,'human count validity')
  human[srx]=(genes,np.log2(1+v/(v.sum()/1e6)))
  input_details.append({'srx':srx,'raw_rows':len(all_ids),'strict_genes':len(genes),'excluded':dict(excluded),'denominator':float(v.sum())})
 mouse={};mouse_info=[]
 p19=BASE/'genetic_expansion_02/balanced_splicing/author_p19/audit'
 samples=json.loads((root/'p19_samples.json').read_text());native_manifest=json.loads((p19/'native/manifest.json').read_text())
 for tag,comp in [('WholeCell','whole_cell'),('Cyto','cytoplasmic')]:
  p=p19/'raw'/('GSE69733_'+tag+'_Gene_Counts_by_Replicate.txt.gz')
  expected=next(x for x in native_manifest['files'] if x['compartment']==comp and x['kind']=='estimated_counts')
  check(sha(p)==expected['source_sha256'],'P19 raw SHA '+comp);raw_sha[str(p)]=expected['source_sha256']
  bycolumn={r['matrix_column']:r for r in samples if r['compartment']==comp}
  with gzip.open(p,'rt') as f:
   reader=csv.DictReader(f,delimiter='\t');columns=reader.fieldnames[1:-3];rows=list(reader)
  check(set(columns)==set(bycolumn),'P19 deposited columns exact sample role map')
  all_genes=[r['ensembl_gene_id'] for r in rows]
  keep=[i for i,g in enumerate(all_genes) if re.fullmatch(r'ENSMUSG[0-9]+(?:\.[0-9]+)?',g)]
  genes=[all_genes[i].split('.')[0] for i in keep];check(len(genes)==len(set(genes)),'P19 stable ID uniqueness')
  v=np.asarray([[float(rows[i][c]) for c in columns] for i in keep],dtype=np.float64)
  check(np.isfinite(v).all() and (v>=0).all() and (v.sum(axis=0)>0).all(),'P19 counts valid')
  transformed=np.log2(1+v/(v.sum(axis=0)/1e6)[None,:])
  for j,c in enumerate(columns):mouse[bycolumn[c]['experiment']]=(genes,transformed[:,j])
  mouse_info.append({'compartment':comp,'raw_genes':len(all_genes),'strict_genes':len(genes),'columns':columns,'denominators':v.sum(axis=0).tolist()})
 # Stable IDs and original orthology relationships only; symbols do not establish pairs.
 all_h=defaultdict(set);all_m=defaultdict(set);good=set();bad_headers=0
 with (root/'target_orthology.tsv').open() as f:
  for r in csv.reader(f,delimiter='\t'):
   check(len(r)==6,'six orthology fields')
   h,hs,m,ms,kind,conf=r
   if not re.fullmatch(r'ENSG[0-9]+',h) or not re.fullmatch(r'ENSMUSG[0-9]+',m):continue
   all_h[h].add(m);all_m[m].add(h)
   if kind=='ortholog_one2one' and conf=='1':good.add((h,m))
 candidates=sorted((h,m) for h,m in good if len(all_h[h])==1 and len(all_m[m])==1)
 human_common=set.intersection(*(set(g) for g,v in human.values()))
 mouse_common=set.intersection(*(set(g) for g,v in mouse.values()))
 axis=[(h,m) for h,m in candidates if h in human_common and m in mouse_common]
 check(axis and len({h for h,m in axis})==len(axis) and len({m for h,m in axis})==len(axis),'unique common ortholog axis')
 all_arrays={}
 for side,src,ids in [('human',human,[h for h,m in axis]),('mouse',mouse,[m for h,m in axis])]:
  for srx,(g,v) in src.items():
   index={x:i for i,x in enumerate(g)};all_arrays[srx]=v[[index[x] for x in ids]]
 responses={}
 for e in entries:
  control=e['controls'];treated=e['perturbs'];check(set(control).isdisjoint(treated),'disjoint contrast arms')
  check(len(control)==2 and len(treated)==2,'two selected replicates per arm')
  responses[e['perturb_id']]=np.stack([all_arrays[x] for x in treated]).mean(axis=0)-np.stack([all_arrays[x] for x in control]).mean(axis=0)
 primary={x for e in entries if e['selection_role']=='primary' for x in e['controls']+e['perturbs']}
 check(len(primary)==34,'34 primary unique samples')
 return {'selection':selection,'entries':entries,'responses':responses,'axis':axis,'human_filtering':input_details,'mouse_inputs':mouse_info,'raw_sha256':raw_sha,'sample_arrays':all_arrays}

def independent_cases(data):
 entries=data['entries'];resp=data['responses'];main=['SRSF1','SRSF5','SRSF7']
 H={(e['cell_model'],e['target']):e for e in entries if e['species']=='Homo sapiens' and e['selection_role']=='primary'}
 M={(e['measurement_compartment'],e['target']):e for e in entries if e['species']=='Mus musculus'}
 specs=[]
 for model in ['HepG2','K562']:
  for target in main:specs.append(('primary',model,target,[x for x in main if x!=target],H[(model,target)]))
 extra=next(e for e in entries if e['selection_role']=='supplement_SRSF3_HepG2_only' and e['species']=='Homo sapiens')
 specs.append(('supplement','HepG2','SRSF3',main,extra))
 alt=next(e for e in entries if e['selection_role']=='sensitivity_alternative_HepG2_SRSF5_batch')
 specs.append(('alternative_batch','HepG2','SRSF5',['SRSF1','SRSF7'],alt))
 for target in ['SRSF1','SRSF7']:specs.append(('quality','K562',target,[x for x in ['SRSF1','SRSF7'] if x!=target],H[('K562',target)]))
 results=[]
 for kind,model,target,train,endpoint in specs:
  check(target not in train,'held target excluded from training')
  y=resp[endpoint['perturb_id']];ys=np.stack([resp[H[(model,t)]['perturb_id']] for t in train])
  xs={c:np.stack([resp[M[(c,t)]['perturb_id']] for t in train]) for c in ['whole_cell','cytoplasmic']}
  x={c:resp[M[(c,target)]['perturb_id']] for c in xs}
  beta={c:max(0,float(np.sum(xs[c]*ys)/np.sum(xs[c]**2))) if np.sum(xs[c]**2)>1e-12 else 0. for c in xs}
  history={c:float(np.mean((xs[c]-ys)**2)) for c in xs}
  choice='whole_cell' if history['whole_cell']<=history['cytoplasmic'] else 'cytoplasmic'
  preds={'zero':np.zeros_like(y),'human_history_mean':ys.mean(axis=0),'whole_raw':x['whole_cell'],'cyto_raw':x['cytoplasmic'],'equal_raw':(x['whole_cell']+x['cytoplasmic'])/2,'whole_scaled':x['whole_cell']*beta['whole_cell'],'cyto_scaled':x['cytoplasmic']*beta['cytoplasmic'],'equal_scaled':(x['whole_cell']*beta['whole_cell']+x['cytoplasmic']*beta['cytoplasmic'])/2,'historical_raw_source_selection':x[choice]}
  metrics={}
  for name,pred in preds.items():
   sd=pred.std()*y.std();metrics[name]={'mse':float(np.mean((pred-y)**2)),'pearson':float(np.mean((pred-pred.mean())*(y-y.mean()))/sd) if sd else None,'prediction_rms':float(np.sqrt(np.mean(pred**2))),'target_rms':float(np.sqrt(np.mean(y**2)))}
  results.append({'kind':kind,'model':model,'target':target,'train_targets':train,'human_id':endpoint['perturb_id'],'betas':beta,'history_mse':history,'choice':choice,'truth':y,'predictions':preds,'metrics':metrics})
 return results

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',type=Path,required=True);ap.add_argument('--production-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 data=build_inputs(args.input_dir);cases=independent_cases(data)
 prod=args.production_dir;report=json.loads((prod/'report.json').read_text());parent=prod.parent
 check(report['script_sha256']==sha(parent/'run_remote.py'),'production script SHA')
 check(report['protocol_sha256']==sha(parent/'PROTOCOL_v1.zh.md'),'frozen protocol SHA')
 check(report['input_freeze_sha256']==sha(parent/'input_freeze.json'),'input freeze SHA')
 frozen=json.loads((parent/'input_freeze.json').read_text())
 for r in frozen['files']:check(sha(parent/'input'/r['remote_filename'])==r['sha256'],'frozen metadata '+r['remote_filename'])
 for p,digest in report['input_expression_files'].items():check(sha(Path(p))==digest,'reported expression input SHA '+p)
 for name,digest in report['output_files'].items():check(sha(prod/name)==digest,'reported output SHA '+name)
 check(sha(args.input_dir/'compartment_selection.json')==sha(parent/'input/compartment_selection.json'),'independent selection matches frozen source')
 check(sha(args.input_dir/'target_orthology.tsv')==sha(parent/'input/target_orthology.tsv'),'independent orthology matches frozen source')
 with (prod/'gene_axis.tsv').open() as f:axis=[(r['human_gene_id'],r['mouse_gene_id']) for r in csv.DictReader(f,delimiter='\t')]
 check(axis==data['axis'],'independent exact ortholog ID axis')
 check(report['fixed_gene_axis']['genes']==len(axis),'reported gene count')
 expected_samples={s for e in data['entries'] for s in e['controls']+e['perturbs']}
 check(len(expected_samples)==46 and report['selected_unique_samples']==46,'46 selected unique samples including supplements')
 with np.load(prod/'sample_log_composition.npz',allow_pickle=False) as a:
  check(set(a['sample_ids'].tolist())==expected_samples,'selected sample arrays cover exact manifest')
  for s,v in zip(a['sample_ids'],a['values']):close(v,data['sample_arrays'][str(s)],'independent sample composition '+str(s))
 with np.load(prod/'responses.npz',allow_pickle=False) as a:
  check(set(a['contrast_ids'].tolist())==set(data['responses']),'16 contrasts complete')
  for cid,v in zip(a['contrast_ids'],a['responses']):close(v,data['responses'][str(cid)],'independent contrast '+str(cid))
 labels={'primary':'primary','supplement':'supplement_SRSF3','alternative_batch':'alternative_HepG2_SRSF5_batch','quality':'K562_without_lowKD_SRSF5'}
 byid={labels[c['kind']]+'|'+c['model']+'|'+c['target']:c for c in cases}
 check(len(report['cases'])==10 and len(byid)==10,'10 independent cases')
 case_records={r['case_id']:r for r in report['cases']};check(set(case_records)==set(byid),'case identities')
 with np.load(prod/'predictions.npz',allow_pickle=False) as a:
  rules=a['rules'].tolist();check(len(rules)==9 and set(rules)==set(cases[0]['predictions']),'nine declared rules')
  check(set(a['case_ids'].tolist())==set(byid),'prediction case IDs')
  for cid,target,preds in zip(a['case_ids'],a['human_targets'],a['predictions']):
   c=byid[str(cid)];r=case_records[str(cid)]
   close(target,c['truth'],'held human truth '+str(cid))
   check(r['fit_targets']==c['train_targets'],'fit target identities '+str(cid))
   check(r['human_query_contrast']==c['human_id'],'query contrast identity')
   check(c['target'] not in r['fit_targets'] and r['query_human_used_for_fitting'] is False,'held target not used for fit')
   hmain={(e['cell_model'],e['target']):e for e in data['entries'] if e['species']=='Homo sapiens' and e['selection_role']=='primary'}
   mm={(e['measurement_compartment'],e['target']):e for e in data['entries'] if e['species']=='Mus musculus'}
   check(r['fit_human_contrasts']==[hmain[(c['model'],t)]['perturb_id'] for t in c['train_targets']],'exact fit human IDs')
   check(r['fit_source_contrasts']==[[mm[(comp,t)]['perturb_id'] for comp in r['source_order']] for t in c['train_targets']],'exact fit source IDs')
   close(r['beta'],[c['betas'][comp] for comp in r['source_order']],'train-only beta '+str(cid))
   close(r['historical_raw_training_mse'],[c['history_mse'][comp] for comp in r['source_order']],'train-only source choice losses')
   check(r['historical_raw_selected_source']==c['choice'],'source decision from history only')
   for rule,pred in zip(rules,preds):
    close(pred,c['predictions'][rule],'independent prediction '+str(cid)+' '+rule)
    for key,value in c['metrics'][rule].items():
     rk='predicted_rms' if key=='prediction_rms' else key;reported=r['rules'][rule][rk]
     if value is None:check(reported is None,'undefined correlation preserved')
     else:close(reported,value,'metric '+str(cid)+' '+rule+' '+rk)
    for denominator,field in [('zero','relative_mse_reduction_over_zero'),('human_history_mean','relative_mse_reduction_over_human_history')]:
     baseline=c['metrics'][denominator]['mse'];value=1-c['metrics'][rule]['mse']/baseline if baseline>0 else None
     if value is None:check(r[field][rule] is None,'undefined relative metric preserved')
     else:close(r[field][rule],value,'relative MSE '+str(cid)+' '+rule+' '+field)
  indexes={str(cid):i for i,cid in enumerate(a['case_ids'])}
  close(a['predictions'][indexes['primary|HepG2|SRSF5']],a['predictions'][indexes['alternative_HepG2_SRSF5_batch|HepG2|SRSF5']],'alternate SRSF5 changes truth only',tol=0.)
  check(not np.array_equal(a['human_targets'][indexes['primary|HepG2|SRSF5']],a['human_targets'][indexes['alternative_HepG2_SRSF5_batch|HepG2|SRSF5']]),'alternate SRSF5 has distinct target data')
 for row in report['macro_results']:
  chosen=[c for cid,c in byid.items() if cid.rsplit('|',1)[0]==row['group']]
  check(len(chosen)==row['query_count'],'macro query count '+row['group'])
  check([c['target'] for c in chosen]==row['targets'],'macro exact target identities')
  for rule,value in row['macro_mse'].items():close(value,np.mean([c['metrics'][rule]['mse'] for c in chosen]),'equal-target macro '+row['group']+' '+rule)
 human_audit={r['sample']:r for r in report['normalization_sample_audit'] if r['side']=='human'}
 reason={'PAR':'PAR_suffix','ENSGR':'ENSGR','numeric':'numeric_identifier','other':'other_noncanonical_identifier'}
 for r in data['human_filtering']:
  p=human_audit[r['srx']];check(p['strict_native_gene_rows']==r['strict_genes'] and p['original_gene_rows']==r['raw_rows'],'human strict-ID support')
  check(p['excluded_by_identifier_reason']=={reason[k]:v for k,v in r['excluded'].items()},'human exclusion categories')
  close(p['reported_strict_estimated_count_mass'],r['denominator'],'human reported composition denominator',tol=1e-6)
 check(all('SRSF5' not in c['train_targets'] and c['target']!='SRSF5' for c in cases if c['kind']=='quality'),'lowKD subgroup excludes all SRSF5 fit/evaluation')
 check(next(c for c in cases if c['kind']=='supplement')['train_targets']==['SRSF1','SRSF5','SRSF7'],'SRSF3 trained only on primary three')
 check(report['cpu_gpu_validation']['passed'] and report['cpu_gpu_validation']['max_abs_difference']<=1e-10,'reported CPU GPU numerical agreement')
 result={'status':'pass','checks_passed':CHECKS,'checks_failed':0,'numeric_values_compared':ARRAY_VALUES,'maximum_absolute_difference':MAXDIFF,'genes':len(axis),'primary_samples':34,'all_selected_samples':46,'human_original_gzip_files':26,'p19_original_gzip_files':2,'contrasts':len(data['responses']),'cases':10,'predictions':90,'production_code_imported':False,'all_array_work_remote':True,'production_report_sha256':sha(prod/'report.json'),'protocol_sha256':report['protocol_sha256'],'independent_script_sha256':sha(Path(__file__)),'human_ID_filtering':data['human_filtering'],'metadata_interpretation_issue':{'field':'K562 SRSF5 ENCORE KD% 23.1/20.5','status':'definition_unresolved_depletion_vs_remaining_expression','numerical_audit_failure':False,'required_reporting_correction':'Do not assert confirmed low knockdown efficacy or use it as a causal explanation. Treat lowKD branch names as frozen historical labels and report prespecified SRSF5-removal sensitivity.'},'strong_checks':{'original_gzip_composition_rebuilt':True,'shared_control_IDs_preserved':True,'fixed_ortholog_axis_rebuilt':True,'LOTO_fit_uses_only_other_targets':True,'alternative_SRSF5_predictions_identical_to_primary_held5':True,'supplement_SRSF3_fit_primary_three_only':True,'lowKD_sensitivity_excludes_SRSF5':True,'macro_MSE_equal_query_weight':True},'limits':['Numerical agreement is E1 reproducibility, not independent biological validation.','Three related primary targets, shared controls and overlapping training sets do not establish a broad task population.','Source compartments share one P19 cell model; no pure species/cell/compartment causal interpretation.','Historical source selection performance is exploratory; original replicate and perturbation-strength limits remain.','K562 SRSF5 23.1/20.5 are the official ENCORE KD% fields; calculation from original validation attachments has not been independently checked. Removing this target is a prespecified sensitivity, not proof that low knockdown causes a result.']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['human_ID_filtering','limits']},indent=2))
