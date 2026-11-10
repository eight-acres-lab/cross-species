from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path('/hy-tmp/cross-species-feedback-20261108/multisource/compartment_benchmark_02')
OUT=ROOT/'independent_validation/target_transcript_QC.json'
selection=json.loads((ROOT/'input/compartment_selection.json').read_text())
report=json.loads((ROOT/'v1/report.json').read_text())
verification=json.loads((ROOT/'independent_validation/compartment_independent_verification.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'v1/report.json')==verification['production_report_sha256']
assert sha(ROOT/'v1/responses.npz')==report['output_files']['responses.npz']
assert sha(ROOT/'v1/sample_log_composition.npz')==report['output_files']['sample_log_composition.npz']
with (ROOT/'input/target_orthology.tsv').open() as f:mapping=list(csv.reader(f,delimiter='\t'))
with np.load(ROOT/'v1/responses.npz',allow_pickle=False) as a:
 ids_h=a['human_gene_ids'].tolist();ids_m=a['mouse_gene_ids'].tolist();responses={str(c):v for c,v in zip(a['contrast_ids'],a['responses'])}
with np.load(ROOT/'v1/sample_log_composition.npz',allow_pickle=False) as a:samples={str(s):v for s,v in zip(a['sample_ids'],a['values'])}
target_ids={}
for target in ['SRSF1','SRSF3','SRSF5','SRSF7']:
 rows=[r for r in mapping if r[1]==target]
 humans={r[0] for r in rows};eligible={(r[0],r[2],r[3]) for r in rows if r[4]=='ortholog_one2one' and r[5]=='1' and r[2]}
 if len(humans)!=1 or len(eligible)!=1:
  target_ids[target]={'status':'ambiguous_or_missing_symbol_mapping','human_ids':sorted(humans),'eligible_pairs':[list(x) for x in sorted(eligible)]};continue
 h,m,ms=next(iter(eligible));all_mouse_ids={r[2] for r in mapping if r[3]==ms and r[2]}
 if len(all_mouse_ids)!=1 or h not in ids_h or m not in ids_m or ids_h.index(h)!=ids_m.index(m):
  target_ids[target]={'status':'ambiguous_mouse_symbol_or_absent_common_axis','human_gene_id':h,'mouse_gene_id':m,'mouse_symbol':ms};continue
 target_ids[target]={'status':'exact_unique_frozen_symbol_and_ortholog_pair_on_axis','human_gene_id':h,'human_symbol':target,'mouse_gene_id':m,'mouse_symbol':ms,'axis_index':ids_h.index(h)}
rows=[]
for entry in selection['entries']:
 target=entry['target'];ident=target_ids[target]
 row={'perturb_id':entry['perturb_id'],'target':target,'species':entry['species'],'cell_model':entry['cell_model'],'measurement_compartment':entry['measurement_compartment'],'selection_role':entry['selection_role'],'gene_identity':ident,'treated_samples':entry['perturbs'],'control_samples':entry['controls']}
 if ident['status']=='exact_unique_frozen_symbol_and_ortholog_pair_on_axis':
  index=ident['axis_index'];treated=[float(samples[s][index]) for s in entry['perturbs']];control=[float(samples[s][index]) for s in entry['controls']]
  delta=float(np.mean(treated)-np.mean(control));assert abs(delta-float(responses[entry['perturb_id']][index]))<1e-12
  row.update(treated_log2_one_plus_composition=treated,control_log2_one_plus_composition=control,mean_treated=float(np.mean(treated)),mean_control=float(np.mean(control)),target_mRNA_log2_one_plus_composition_change=delta,direction='positive' if delta>0 else 'negative' if delta<0 else 'zero')
 rows.append(row)
result={'date':'2026-11-08','scope':'Posthoc target-transcript input QC only; no model, selection, branch or fit change','quantity':'Mean treated log2(1+reported-count-composition per million) minus mean control on the frozen common ortholog axis','count':len(rows),'target_identity_mapping':target_ids,'contrasts':rows,'interpretation_limits':['Relative mRNA composition change does not equal protein knockdown efficacy or replace the definition of the reported ENCORE KD% field.','Positive target transcript values are flags for source/design review; they do not by themselves establish reversed treatment labels.','Cytoplasmic and whole-cell abundance can differ through RNA processing and compartment distribution.','Pseudocount, estimated-count quantifiers, library composition and individual transcript isoforms limit interpretation as absolute transcript depletion.','All primary and supplementary branches remain included exactly as frozen.'],'normalization_or_prediction_changed':False,'production_responses_sha256':sha(ROOT/'v1/responses.npz'),'independent_verification_sha256':sha(ROOT/'independent_validation/compartment_independent_verification.json'),'mapping_sha256':sha(ROOT/'input/target_orthology.tsv'),'script_sha256':sha(__file__)}
OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'contrast_count':len(rows),'rows':[{'id':r['perturb_id'],'target':r['target'],'model':r['cell_model'],'compartment':r['measurement_compartment'],'delta':r.get('target_mRNA_log2_one_plus_composition_change'),'direction':r.get('direction')} for r in rows]},indent=2))
