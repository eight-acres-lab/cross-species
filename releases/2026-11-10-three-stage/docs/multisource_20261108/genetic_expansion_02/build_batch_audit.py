"""Audit a bounded genetic-response batch from cached primary sample metadata."""
from pathlib import Path
import csv,hashlib,json,re
P=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse_soft(path):
 records=[];r=None
 for line in path.read_text().splitlines():
  if line.startswith('^SAMPLE = '):r={'gsm':line.split(' = ',1)[1]};records.append(r)
  elif r is not None and line.startswith('!Sample_') and ' = ' in line:
   k,v=line[len('!Sample_'):].split(' = ',1);r.setdefault(k,[]).append(v)
 return records

def main():
 entries=json.loads((P/'download_entries.json').read_text());wanted={x for e in entries for x in e['expected_experiments']}
 lookup={};evidence={}
 for path in [P/'evidence/GSE69733.soft',P/'evidence/GSE171418.soft',*sorted((P/'evidence').glob('GSM*.soft'))]:
  evidence[str(path.relative_to(P))]=digest(path)
  for r in parse_soft(path):
   ids=re.findall(r'SRX\d+',' '.join(r.get('relation',[])))
   for srx in ids:
    if srx in wanted:
     if srx in lookup:raise ValueError('duplicate original sample '+srx)
     lookup[srx]=(r,str(path.relative_to(P)))
 assert set(lookup)==wanted
 samples=[];contrasts=[]
 for e in entries:
  target=e['target'].rstrip('-').upper();model='HepG2' if e['species']=='Homo sapiens' else 'P19_whole_cell' if 'P19' in e['cell_context'] else 'RAW264.7_uninfected'
  sampleids=[]
  for role,key in [('control','controls'),('treated','perturbs')]:
   for srx in e[key]:
    r,source=lookup[srx];chars=r.get('characteristics_ch1',[]);joined=' '.join(chars);title=r['title'][0]
    if model=='HepG2':
     assert 'line: HepG2' in chars
     expected='Non-specific target control' if role=='control' else target
     assert expected in r['source_name_ch1'][0]
     descriptions=r.get('description',[])
     bio=next(x.split(': ',1)[1] for x in descriptions if x.startswith('biological replicate number:'))
     tech=next(x.split(': ',1)[1] for x in descriptions if x.startswith('technical replicate number:'))
     if role=='treated':assert 'possible controls: ENCSR603TCV' in descriptions
     else:assert 'experiment encode accession: ENCSR603TCV' in descriptions
     method='shRNA';time=None;unit='ENCODE biological replicate label; common HepG2 cell-line donor; not independent donors'
    elif model=='P19_whole_cell':
     assert 'cell fraction: whole cell' in chars
     assert ('genotype/variation: KD '+('GFP' if role=='control' else target)) in chars
     bio=title.rsplit(' ',1)[-1];tech=None;method='esiRNA';time=36;unit='Series reports 2 biological replicates and 2 technical replicates per condition; A/B retained as biological labels; technical runs not counted as added biological units'
    else:
     assert 'treatment: no treatment' in chars and '_UN_' in title and 'cell line: RAW 264.7' in chars
     assert ('genotype: '+('SCR shRNA Control' if role=='control' else target+' KD shRNA Knockdown')) in chars
     bio=title.rsplit(' ',1)[-1];tech=None;method='shRNA';time=None;unit='Deposited triplicate sample labels; donor/culture independence and knockdown interval not established here'
    sampleids.append(r['gsm'])
    samples.append({'perturb_id':e['perturb_id'],'target':target,'model':model,'srx':srx,'gsm':r['gsm'],'role':role,'title':title,'species':r['organism_ch1'][0],'method':method,'time_after_knockdown_h':time,'cell_fraction':'whole_cell_or_total_RNA','infection_state':'uninfected' if model=='RAW264.7_uninfected' else 'no_infection_in_selected_sample_metadata','biological_replicate_label':bio,'technical_replicate_label':tech,'independent_unit_status':unit,'donor_pairing_verified':False,'source':source,'original_characteristics':chars,'treatment_protocol':r.get('treatment_protocol_ch1',[]),'data_processing':r.get('data_processing',[]),'geo_series':r.get('series_id',[])})
  contrasts.append({'perturb_id':e['perturb_id'],'target':target,'model':model,'sample_identity_and_role_match':True,'gsm_ids':sampleids,'controls':e['controls'],'perturbs':e['perturbs'],'design_status':'admitted_for_source_specific_descriptive_curation_after_expression_gates'})
 unique={r['srx'] for r in samples};assert len(unique)==21
 (P/'sample_bridge.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False)+'\n')
 fields=['perturb_id','target','model','srx','gsm','role','title','species','method','time_after_knockdown_h','biological_replicate_label','technical_replicate_label','independent_unit_status','source']
 with (P/'sample_bridge.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(samples)
 audit={'date':'2026-11-08','targets':['SRSF1','SRSF7'],'human_models':['HepG2'],'mouse_cell_models':['P19','RAW264.7'],'selected_mouse_measurements':['P19 whole-cell','RAW264.7 uninfected whole-cell'],'contrasts':contrasts,'unique_srx_samples':21,'sample_rows_with_reused_controls':len(samples),'design_status':'admitted_for_descriptive_curation_after_expression_gates','model_training_ready':False,'source_selection_generalization_ready':False,'source_evidence_sha256':evidence,'input_selection_sha256':digest(P/'download_entries.json'),'shared_control_groups':{m:sorted({s['srx'] for s in samples if s['model']==m and s['role']=='control'}) for m in ['HepG2','P19_whole_cell','RAW264.7_uninfected']},'limitations':['Only two distinct target identities, both splicing-factor family; source ranks remain retrospective.','Each cell model occurs in a different study and assay/protocol; cell-model effects and study effects are inseparable.','P19 uses36h esiRNA while HepG2/RAW use shRNA; timing and knockdown magnitude not equated.','Whole-cell and cytoplasmic P19 measurements come from the same cell source and cannot count as two cell models.','Shared controls induce dependency across targets; repeated source subsets add no independent targets.','PerturbAtlas missing-record semantics remain unresolved; no zero imputation.','Original RAW264.7 files are declared RPKM; they cannot be normalized as raw counts.','P19 original Cufflinks gene-count tables require numeric/column/replicate inspection before use.'],'next_gates':['Complete remote expression integrity audit and retain missing masks.','Define common gene axis and within-source quantity-appropriate normalization.','Freeze descriptive source/combination rules and report both targets without source cherry-picking.','Keep donor/experimental pairing unknown unless documented.']}
 (P/'batch_design_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'samples':len(unique),'contrasts':len(contrasts),'roles_exact':True}))
if __name__=='__main__':main()
