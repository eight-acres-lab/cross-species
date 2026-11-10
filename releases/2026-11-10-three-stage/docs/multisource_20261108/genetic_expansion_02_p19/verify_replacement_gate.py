"""Independent, metadata-only assay gate for the official SRSF5 replacement."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'docs/multisource_20261108/genetic_expansion_02/srsf5_kd_definition'
OUT=ROOT/'outputs/multisource_20261108/remote_expansion_02/replacement_srsf5_independent_verification.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((P/'ENCSR906RHU.json').read_text());new=json.loads((P/'ENCSR528YVC.json').read_text());control=json.loads((P/'ENCSR292PXV.json').read_text())
checks=[]
def verify(ok,claim):
 if not ok:raise AssertionError(claim)
 checks.append(claim)
verify(old['status']=='archived','Old experiment is archived')
verify(new['status']=='released','Official replacement is released')
verify(new['@id'] in old['superseded_by'],'Old official record points to new replacement')
verify(old['@id'] in new['supersedes'],'New official record reciprocally supersedes old')
for d in [old,new]:
 verify(d['target']['label']=='SRSF5','Exact SRSF5 target '+d['accession'])
 verify(d['biosample_ontology']['term_name']=='K562','Exact K562 context '+d['accession'])
verify(old['assay_term_name']=='shRNA knockdown followed by RNA-seq','Old assay is shRNA RNA-seq')
verify(new['assay_term_name']=='CRISPR genome editing followed by RNA-seq','New assay is CRISPR RNA-seq')
controls=[c['accession'] if isinstance(c,dict) else c.rstrip('/').split('/')[-1] for c in new['possible_controls']]
verify(controls==['ENCSR292PXV'],'Replacement possible_control is ENCSR292PXV')
verify(control['accession']=='ENCSR292PXV' and control['status']=='released','New control is released exact accession')
verify(control['biosample_ontology']['term_name']=='K562','New control is K562')
verify(control['assay_term_name']==new['assay_term_name'],'New control has matching CRISPR assay')
verify(control.get('target') is None,'New control has no targeted gene')
replicates=[]
for d,role in [(new,'treated'),(control,'control')]:
 bio=[]
 for r in d['replicates']:
  lib=r['library'];bs=lib['biosample'];bio.append(r['biological_replicate_number'])
  verify(bs['biosample_ontology']['term_name']=='K562','Replicate biosample is K562 '+bs['accession'])
  mods=bs.get('genetic_modifications',[])
  methods=sorted({m['method'] for m in mods if isinstance(m,dict) and 'method' in m})
  verify(methods==['CRISPR'],'Replicate genetic modification method is CRISPR '+bs['accession'])
  replicates.append({'experiment':d['accession'],'role':role,'biological_replicate_label':r['biological_replicate_number'],'technical_replicate_label':r['technical_replicate_number'],'library':lib['accession'],'biosample':bs['accession'],'modification_methods':methods,'modification_categories':sorted({m.get('category') for m in mods if isinstance(m,dict) and m.get('category')})})
 verify(sorted(bio)==[1,2],'Two declared biological-replicate labels '+d['accession'])
for row in json.loads((P/'replacement_api_sources.json').read_text()):
 verify(sha(P/(row['accession']+'.json'))==row['sha256'],'Official API snapshot SHA '+row['accession'])
old_source=json.loads((P/'audit.json').read_text())['files_sha256']['ENCSR906RHU.json']
verify(sha(P/'ENCSR906RHU.json')==old_source,'Old official API snapshot SHA')
result={'date':'2026-11-08','status':'hold_method_mismatch','gate':'same_method_SRSF5_shRNA_endpoint_replacement','metadata_checks_passed':len(checks),'metadata_checks_failed':0,'checks':checks,'official_replacement_chain_verified':True,'old_experiment':{'accession':old['accession'],'status':old['status'],'assay':old['assay_term_name'],'possible_controls':[c['accession'] for c in old['possible_controls']],'submitter_comment':old['submitter_comment']},'candidate_experiment':{'accession':new['accession'],'status':new['status'],'assay':new['assay_term_name'],'target':'SRSF5','cell_model':'K562','possible_controls':controls},'candidate_replicates':replicates,'hold_reason':'Official supersession changes perturbation method from shRNA knockdown to CRISPR genome editing. It cannot isolate a same-method knockdown-quality improvement under the existing replacement plan.','allowed_handling':'Keep official supersession evidence and frozen v1 results. Hold the proposed same-method replacement benchmark. A future cross-method comparison would require an explicitly separate design; none is created here.','raw_expression_downloaded_by_this_audit':False,'response_normalization_performed':False,'gene_axis_intersection_evaluated':False,'new_predictions_or_metrics_generated':False,'target_transcript_QC_for_replacement_performed':False,'old_benchmark_modified':False,'production_functions_imported':False,'source_urls':{'old':'https://www.encodeproject.org/ENCSR906RHU/?format=json','new':'https://www.encodeproject.org/ENCSR528YVC/?format=json','control':'https://www.encodeproject.org/ENCSR292PXV/?format=json'},'source_sha256':{name:sha(P/name) for name in ['ENCSR906RHU.json','ENCSR528YVC.json','ENCSR292PXV.json','replacement_api_sources.json']},'independent_script_sha256':sha(__file__)}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'metadata_checks_passed':len(checks),'old_assay':old['assay_term_name'],'new_assay':new['assay_term_name'],'new_control':controls,'replacement_metrics_generated':False},indent=2))
