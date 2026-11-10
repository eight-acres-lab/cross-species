"""Independent original-text reconstruction; no imports from production analysis."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np

P=Path('/hy-tmp/cross-species-feedback-20261108/multisource/e5919_processed_2025')
checks=[];diffs=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value,detail=None):
 checks.append({'name':name,'passed':bool(value),**({'detail':detail} if detail is not None else {})})
 if not value:raise AssertionError(name+' '+str(detail))
def close(name,observed,expected,tol=1e-10):
 if observed is None or expected is None:check(name,observed is None and expected is None);return
 a,b=np.asarray(observed),np.asarray(expected);check(name+'_shape',a.shape==b.shape)
 if np.issubdtype(a.dtype,np.number):
  check(name+'_nonfinite_pattern',np.array_equal(np.isfinite(a),np.isfinite(b)))
  finite=np.isfinite(a)&np.isfinite(b);d=float(np.max(np.abs(a[finite]-b[finite]))) if np.any(finite) else 0.;diffs.append(d);check(name,d<=tol,{'max_abs_difference':d})
 else:check(name,np.array_equal(a,b))
def load_table(path):
 with path.open() as f:
  rr=csv.reader(f,delimiter='\t');head=next(rr);rows=list(rr)
 genes=[r[0] for r in rows];matrix=np.asarray([[float(x) for x in r[1:]] for r in rows]);return head[1:],genes,matrix
def metrics(pred,truth):
 n=len(truth);px=pred-pred.mean();ty=truth-truth.mean();den=float(np.linalg.norm(px)*np.linalg.norm(ty))
 return {'mse':math.fsum((float(a)-float(b))**2 for a,b in zip(pred,truth))/n,'pearson':float(np.dot(px,ty)/den) if den>0 else None,'predicted_rms':math.sqrt(math.fsum(float(x)*float(x) for x in pred)/n),'human_rms':math.sqrt(math.fsum(float(x)*float(x) for x in truth)/n)}
def compare_metrics(label,actual,saved):
 for key in ['mse','pearson','predicted_rms','human_rms']:close(label+'/'+key,actual[key],saved[key])

def main(out):
 e0=json.loads((P/'e0_v1/audit.json').read_text());e1=json.loads((P/'e1_v1/report.json').read_text());post=json.loads((P/'posthoc_scalar_v1/report.json').read_text())
 for path,expected in e0['input_files'].items():check('raw_sha/'+Path(path).name,sha(path)==expected)
 for name,expected in e0['files'].items():check('E0_sha/'+name,sha(P/'e0_v1'/name)==expected)
 check('E1_E0_audit_sha',sha(P/'e0_v1/audit.json')==e1['e0_audit_sha256'])
 for name,expected in e1['input_files'].items():check('E1_input_sha/'+name,sha(P/'e0_v1'/name)==expected)
 for name,expected in e1['output_files'].items():check('E1_output_sha/'+name,sha(P/'e1_v1'/name)==expected)
 for name,expected in post['output_files'].items():check('post_output_sha/'+name,sha(P/'posthoc_scalar_v1'/name)==expected)
 for report,protocol,script in [(e1,'E1_PROTOCOL_v1.md','run_e1_remote.py'),(post,'POSTHOC_SCALAR_PROTOCOL_v1.md','run_posthoc_scalar_remote.py')]:
  check('protocol_sha/'+protocol,sha(P/protocol)==report['protocol_sha256']);check('script_sha/'+script,sha(P/script)==report['script_sha256'])
 check('post_source_report_sha',sha(P/'e1_v1/report.json')==post['source_report_sha256']);check('post_response_input_sha',sha(P/'e1_v1/responses.npz')==post['response_input_sha256'])
 sample_ids,genes,tpm=load_table(P/'homo_vs_mus_tpmData_Hagai.txt');count_samples,count_genes,counts=load_table(P/'homo_vs_mus_countData_Hagai.txt')
 check('original_axes',sample_ids==count_samples and genes==count_genes and len(genes)==15851 and len(sample_ids)==36)
 check('original_unique_axes',len(set(genes))==len(genes) and len(set(sample_ids))==36)
 check('original_nonnegative_finite',np.isfinite(tpm).all() and np.isfinite(counts).all() and (tpm>=0).all() and (counts>=0).all())
 with (P/'small_metaData_Hagai_homo_vs_mus.txt').open() as f:
  rr=csv.reader(f,delimiter='\t');header=next(rr);info=list(rr)
 check('author_metadata_header',header==['run_accession','study_name','run_accession','treatment','species','individual'])
 check('author_duplicate_label_fields_agree',all(r[0]==r[2] for r in info))
 info_by_id={r[0]:r for r in info};check('metadata_exact_axis',set(info_by_id)==set(sample_ids) and len(info)==36)
 with (P/'1-1_human_mouse_metadata.csv').open() as f:metadata=list(csv.DictReader(f))
 key=list(metadata[0])[0];gene_info={r[key]:r for r in metadata};check('metadata_gene_ids_unique',len(gene_info)==len(metadata))
 mousegenes=[gene_info[g]['Mouse gene stable ID'] for g in genes]
 check('full_one_to_one_axis',all(gene_info[g]['Mouse homology type']=='ortholog_one2one' for g in genes) and len(mousegenes)==len(set(mousegenes)))
 with (P/'e0_v1/gene_axis.tsv').open() as f:ga=list(csv.DictReader(f,delimiter='\t'))
 check('E0_gene_axis',[(r['human_gene_id'],r['mouse_gene_id']) for r in ga]==list(zip(genes,mousegenes)))
 with np.load(P/'e0_v1/author_processed_arrays.npz',allow_pickle=False) as a:
  close('E0_author_tpm',tpm,a['author_tpm'],0);close('E0_count_values',counts,a['estimated_counts'],0);close('E0_sample_ids',sample_ids,a['author_sample_ids']);close('E0_human_gene_ids',genes,a['human_gene_ids']);close('E0_mouse_gene_ids',mousegenes,a['mouse_gene_ids'])
 # Pair directly using author information, not saved contrast indices.
 bycondition={}
 for i,sid in enumerate(sample_ids):
  row=info_by_id[sid];key=(row[4],row[5],row[3]);check('unique_author_condition/'+sid,key not in bycondition);bycondition[key]=i
 expression=np.log2(tpm+1.0);stimuli=['IFNB','polyIC'];pairs={'IFNB':('IFNB','UNST'),'polyIC':('PIC4','LF4')};response={};pair_records=[]
 for species,n in [('Homo_sapiens',6),('Mus_musculus',3)]:
  check('individual_axis/'+species,sorted({r[5] for r in info if r[4]==species},key=int)==[str(i) for i in range(1,n+1)])
  block=[]
  for stimulus in stimuli:
   t,c=pairs[stimulus];rows=[]
   for donor in [str(i) for i in range(1,n+1)]:
    ti,ci=bycondition[(species,donor,t)],bycondition[(species,donor,c)];rows.append(expression[:,ti]-expression[:,ci]);pair_records.append({'species':species,'author_individual':donor,'family':stimulus,'treated_column':ti,'control_column':ci})
   block.append(rows)
  response[species]=np.asarray(block)
 human=response['Homo_sapiens'];mouse=response['Mus_musculus'];hm=human.mean(axis=1);mm=mouse.mean(axis=1)
 specs=json.loads((P/'e0_v1/contrast_specification.json').read_text());map_specs={(r['species'],r['author_individual'],r['family']):r for r in specs}
 for r in pair_records:
  s=map_specs[(r['species'],r['author_individual'],r['family'])];check('contrast_pair/'+s['contrast_id'],r['treated_column']==s['treated_column'] and r['control_column']==s['control_column']);check('time_matching/'+s['contrast_id'],s['time_matched_control']==(r['family']=='polyIC'))
 sample_meta=json.loads((P/'e0_v1/sample_metadata.json').read_text())
 for r in sample_meta:check('timing_label/'+r['author_sample_id'],r['deposited_timing_h']=={'IFNB':8,'UNST':0,'PIC4':4,'LF4':4}[r['condition']] and r['ena_run_accession'] is None)
 with np.load(P/'e1_v1/responses.npz',allow_pickle=False) as a:
  close('E1_raw_TPM_response_human',human,a['human']);close('E1_raw_TPM_response_mouse',mouse,a['mouse']);close('E1_stimulus_axis',stimuli,a['stimuli']);close('E1_genes',genes,a['human_gene_ids']);close('E1_mouse_genes',mousegenes,a['mouse_gene_ids'])
 # Recheck original metadata group means, including order-independent gene lookup.
 mean_checks=0;mean_max=0.
 for species,short in [('Homo_sapiens','homo_sapiens'),('Mus_musculus','mus_musculus')]:
  for condition,tag in [('IFNB','IFNB'),('UNST','UNST'),('PIC4','PIC'),('LF4','LF')]:
   columns=[i for i,s in enumerate(sample_ids) if info_by_id[s][4]==species and info_by_id[s][3]==condition];actual=tpm[:,columns].mean(axis=1);reported=np.asarray([float(gene_info[g][f'Hagai_{short}_{tag}_av_TPM']) for g in genes]);mean_max=max(mean_max,float(np.max(np.abs(actual-reported))));mean_checks+=len(genes);close('author_group_TPM_mean/'+species+'/'+condition,actual,reported)
 for si,stim in enumerate(stimuli):
  saved=next(r for r in e1['point_results'] if r['stimulus']==stim);rules={'same_stimulus_mouse':mm[si],'zero_response':np.zeros(len(genes)),'other_stimulus_mouse':mm[1-si]}
  for rule,pred in rules.items():
   actual=metrics(pred,hm[si]);compare_metrics('point/'+stim+'/'+rule,actual,saved['mean_human_target'][rule]);close('point_gain/'+stim+'/'+rule,1-actual['mse']/np.mean(hm[si]**2),saved['mean_target_relative_mse_reduction_over_zero'][rule])
   for d in saved['human_individual_targets']:compare_metrics('individual/'+stim+'/'+d['author_individual']+'/'+rule,metrics(pred,human[si,int(d['author_individual'])-1]),d['rules'][rule])
 # Verify complete frozen random-index schedules and all reported quantiles from saved draws.
 rng=np.random.default_rng(20261108);expected_h=rng.integers(0,6,size=(2000,6));expected_m=rng.integers(0,3,size=(2000,3))
 with np.load(P/'e1_v1/resampling.npz',allow_pickle=False) as b:
  check('human_index_schedule',np.array_equal(b['human_indices'],expected_h));check('mouse_index_schedule',np.array_equal(b['mouse_indices'],expected_m))
  for si,stim in enumerate(stimuli):
   for trial in range(20):
    h=human[si,expected_h[trial]].mean(axis=0);m=mouse[si,expected_m[trial]].mean(axis=0);other=mouse[1-si,expected_m[trial]].mean(axis=0)
    for rule,pred in [('same_stimulus_mouse',m),('zero_response',np.zeros(len(genes))),('other_stimulus_mouse',other)]:
     computed=metrics(pred,h)
     for k,v in computed.items():
      saved=float(b[f'{stim}__{rule}__{k}'][trial]);close(f'bootstrap20/{stim}/{trial}/{rule}/{k}',np.nan if v is None else v,saved)
   sensitivity=next(r for r in e1['resampling_sensitivity'] if r['stimulus']==stim)
   for rule,summary in sensitivity['rules'].items():
    for metric_name in ['mse','pearson','predicted_rms','human_rms']:
     v=b[f'{stim}__{rule}__{metric_name}'];actual=np.quantile(v,[0.025,0.5,0.975]) if np.isfinite(v).all() else None;close(f'sensitivity/{stim}/{rule}/{metric_name}',actual,summary[metric_name])
    zero=b[f'{stim}__zero_response__mse'];scores=b[f'{stim}__{rule}__mse'];gains=1-scores/zero;close(f'sensitivity_gain/{stim}/{rule}',np.quantile(gains,[0.025,0.5,0.975]),summary['relative_mse_reduction_over_zero_quantiles']);close(f'sensitivity_benefit_fraction/{stim}/{rule}',float((scores<zero).mean()),summary['fraction_resamples_lower_mse_than_zero'])
 # Independent fsum beta fits only the other stimulus; all predictions re-created from original text.
 with np.load(P/'posthoc_scalar_v1/predictions.npz',allow_pickle=False) as a:
  close('post_human_means',hm,a['human_mean']);close('post_mouse_means',mm,a['mouse_mean']);close('post_gene_axis',genes,a['human_gene_ids']);close('post_stimuli',stimuli,a['stimuli'])
  for si,stim in enumerate(stimuli):
   other=1-si;x,y=mm[other],hm[other];den=math.fsum(float(v)*float(v) for v in x);num=math.fsum(float(v)*float(w) for v,w in zip(x,y));beta=max(num/den,0.) if den>1e-12 else 0.
   saved=next(r for r in post['results'] if r['held_stimulus']==stim);close('post_beta/'+stim,beta,saved['beta']);close('post_beta_numerator/'+stim,num,saved['fit_source_dot_target']);close('post_beta_denominator/'+stim,den,saved['fit_source_squared_sum']);check('post_fit_stimulus/'+stim,saved['fit_stimulus']==stimuli[other])
   rules={'source_scaled':beta*mm[si],'source_raw':mm[si],'zero_response':np.zeros(len(genes)),'other_stimulus_human_mean':hm[other],'other_stimulus_mouse_scaled':beta*mm[other]};scores={}
   for rule,pred in rules.items():close('post_prediction/'+stim+'/'+rule,pred,a[stim+'__'+rule]);scores[rule]=metrics(pred,hm[si]);compare_metrics('post_metric/'+stim+'/'+rule,scores[rule],saved['rules'][rule])
   close('post_gain_zero/'+stim,1-scores['source_scaled']['mse']/scores['zero_response']['mse'],saved['scaled_relative_mse_reduction_over_zero']);close('post_gain_history/'+stim,1-scores['source_scaled']['mse']/scores['other_stimulus_human_mean']['mse'],saved['scaled_relative_mse_reduction_over_human_history'])
 result={'status':'pass','audit_date':'2026-11-08','independence':'Reconstructed original author text matrices and sample_info using own csv/numpy/math code. No import or invocation of production E0/E1/posthoc functions. Remote CPU only.','checks_passed':sum(r['passed'] for r in checks),'checks_failed':sum(not r['passed'] for r in checks),'max_absolute_difference':max(diffs),'tolerance':1e-10,'original_matrix_shape':list(tpm.shape),'responses_reconstructed':len(pair_records),'complete_genes':len(genes),'human_author_individuals':6,'mouse_author_individuals':3,'author_group_mean_comparisons':mean_checks,'author_group_mean_max_difference':mean_max,'bootstrap_draws_independently_recomputed':20,'full_saved_index_schedules_verified':2000,'all_saved_bootstrap_summary_statistics_verified':True,'posthoc_rules_verified':5,'renormalization_applied':False,'raw_accession_bridge_claimed':False,'all_declared_provenance_hashes_verified':True,'implementation_findings':[],'design_risks':[{'severity':'interpretation_limit','finding':'Both posthoc endpoints were seen and the same author individuals appear in both stimuli. Cross-stimulus fitting is retrospective rule sensitivity; a new-stimulus or donor-independent validation claim is unsupported.'},{'severity':'interpretation_limit','finding':'IFNB8-UNST0 includes time-in-culture differences. PIC4-LF4 is time-matched. Condition-level timing cannot substitute for the unresolved per-column raw-accession bridge.'},{'severity':'interpretation_limit','finding':'The supplied axis is an author common-gene subset. Original TPM preservation is appropriate; no full-native library-size claim is available.'},{'severity':'interpretation_limit','finding':'Two related immune stimuli and one mouse source cannot establish general source recommendation. Shared immune programs and shrinkage can contribute to the improvement; the five fixed controls are informative but do not exhaust every no-new-source predictor.'},{'severity':'terminology_note','finding':'The historical human mean is a specified baseline. Calling it a strong baseline is a design judgment rather than a demonstrated bound; higher-level E2/E3 claims need expanded baselines and independent tasks.'}],'overclaim_review':'Local E1 and posthoc writeups explicitly retain these principal limits; no substantive current conclusion exceeds their stated E1/posthoc scope.','input_report_sha256':{'e0':sha(P/'e0_v1/audit.json'),'e1':sha(P/'e1_v1/report.json'),'posthoc':sha(P/'posthoc_scalar_v1/report.json')},'checks':checks}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','design_risks']},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
