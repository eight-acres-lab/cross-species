"""Posthoc E1 scalar rule sensitivity; no changes to E1v1 data or estimates."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/multisource/e5919_processed_2025')
OUT=ROOT/'posthoc_scalar_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(p,y):
 pc=p-p.mean();yc=y-y.mean();den=float(np.sqrt(np.dot(pc,pc)*np.dot(yc,yc)))
 return {'mse':float(np.mean((p-y)**2)),'pearson':float(np.dot(pc,yc)/den) if den>0 else None,'predicted_rms':float(np.sqrt(np.mean(p*p))),'human_rms':float(np.sqrt(np.mean(y*y)))}

def main():
 assert not OUT.exists()
 before=time.monotonic()
 source_report=json.loads((ROOT/'e1_v1/report.json').read_text())
 response_path=ROOT/'e1_v1/responses.npz'
 assert sha(response_path)==source_report['output_files']['responses.npz']
 with np.load(response_path,allow_pickle=False) as saved:
  human=saved['human'].mean(axis=1);mouse=saved['mouse'].mean(axis=1);stimuli=saved['stimuli'].tolist();genes=saved['human_gene_ids'].copy()
 assert human.shape==mouse.shape==(2,15851) and stimuli==['IFNB','polyIC']
 records=[];predictions={};maximum=0.;checks=0
 for held in range(2):
  training=1-held;x,y=mouse[training],human[training]
  denominator=float(np.square(x).sum());numerator=float(np.dot(x,y));beta=max(numerator/denominator,0.) if denominator>1e-12 else 0.
  independent_den=math.fsum(float(v)*float(v) for v in x)
  independent_num=math.fsum(float(a)*float(b) for a,b in zip(x,y))
  independent_beta=max(independent_num/independent_den,0.) if independent_den>1e-12 else 0.
  maximum=max(maximum,abs(beta-independent_beta));checks+=1
  assert abs(beta-independent_beta)<=1e-10
  rules={'source_scaled':beta*mouse[held],'source_raw':mouse[held],'zero_response':np.zeros(len(genes)),'other_stimulus_human_mean':human[training],'other_stimulus_mouse_scaled':beta*mouse[training]}
  values={rule:metrics(p,human[held]) for rule,p in rules.items()}
  for rule,p in rules.items():
   independent_mse=math.fsum((float(a)-float(b))**2 for a,b in zip(p,human[held]))/len(genes)
   maximum=max(maximum,abs(independent_mse-values[rule]['mse']));checks+=1
   assert abs(independent_mse-values[rule]['mse'])<=1e-10
   predictions[stimuli[held]+'__'+rule]=p
  baseline=values['zero_response']['mse'];history=values['other_stimulus_human_mean']['mse']
  records.append({'held_stimulus':stimuli[held],'fit_stimulus':stimuli[training],'beta':beta,'fit_source_dot_target':numerator,'fit_source_squared_sum':denominator,'rules':values,'scaled_relative_mse_reduction_over_zero':1-values['source_scaled']['mse']/baseline,'scaled_relative_mse_reduction_over_human_history':1-values['source_scaled']['mse']/history,'fit_reads_held_human_target':False})
 OUT.mkdir()
 np.savez_compressed(OUT/'predictions.npz',human_mean=human,mouse_mean=mouse,human_gene_ids=genes,stimuli=np.asarray(stimuli),**predictions)
 report={'evidence_level':'posthoc_E1_retrospective_rule_sensitivity','reason_for_posthoc_branch':'Raw-copy E1v1 outcomes already inspected; fixed prior scalar rule reviewed after observing positive correlation with increased MSE.','protocol_sha256':sha(ROOT/'POSTHOC_SCALAR_PROTOCOL_v1.md'),'script_sha256':sha(__file__),'response_input_sha256':sha(response_path),'source_report_sha256':sha(ROOT/'e1_v1/report.json'),'genes':len(genes),'rule':'ScalarCalibrator(mode=scaled): nonnegative no-intercept least-squares scalar; denominator threshold1e-12.','existing_rule_reference_sha256':'2e7c3c7c278126dbe34a9e4ce419fba3dbb6dff1296c63469bb9759de434a404','results':records,'independent_cpu_validation':{'checks':checks,'max_abs_difference':maximum,'tolerance':1e-10,'passed':True,'method':'math.fsum scalar beta and MSE checked against NumPy vector formulas'},'runtime':{'device':'remote_CPU','wall_seconds':time.monotonic()-before,'additional_resamples':0},'limitations':['Both stimuli and their human results were seen before this posthoc question. This is not independent new-stimulus confirmation.','The two related immune stimuli reuse the same author individuals.','Source scaled is compared with the historical human mean as a strong no-new-animal-measurement reference.','One mouse source and two stimuli cannot support general source selection.','IFNB8h-UNST0h includes time-in-culture confounding; author sample-to-ERR identity remains unresolved.'],'output_files':{'predictions.npz':sha(OUT/'predictions.npz')}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'results':records,'independent_cpu_validation':report['independent_cpu_validation']},indent=2))

if __name__=='__main__':main()
