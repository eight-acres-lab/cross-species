"""Verify ENCODE SRSF5 accession linkage and assay compatibility."""
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
verify(old['status']=='archived','ENCSR906RHU is archived')
verify(new['status']=='released','ENCSR528YVC is released')
verify(new['@id'] in old['superseded_by'],'ENCSR906RHU links to ENCSR528YVC through superseded_by')
verify(old['@id'] in new['supersedes'],'ENCSR528YVC links to ENCSR906RHU through supersedes')
for d in [old,new]:
 verify(d['target']['label']=='SRSF5','Exact SRSF5 target '+d['accession'])
 verify(d['biosample_ontology']['term_name']=='K562','Exact K562 context '+d['accession'])
verify(old['assay_term_name']=='shRNA knockdown followed by RNA-seq','ENCSR906RHU uses shRNA knockdown followed by RNA-seq')
verify(new['assay_term_name']=='CRISPR genome editing followed by RNA-seq','ENCSR528YVC uses CRISPR genome editing followed by RNA-seq')
controls=[c['accession'] if isinstance(c,dict) else c.rstrip('/').split('/')[-1] for c in new['possible_controls']]
verify(controls==['ENCSR292PXV'],'ENCSR528YVC lists ENCSR292PXV as its control')
verify(control['accession']=='ENCSR292PXV' and control['status']=='released','ENCSR292PXV is released')
verify(control['biosample_ontology']['term_name']=='K562','ENCSR292PXV uses K562 cells')
verify(control['assay_term_name']==new['assay_term_name'],'ENCSR292PXV uses the matching CRISPR assay')
verify(control.get('target') is None,'ENCSR292PXV has no targeted gene')
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
verify(sha(P/'ENCSR906RHU.json')==old_source,'ENCSR906RHU API snapshot SHA')
result={'date':'2026-11-08','status':'method_mismatch','gate':'SRSF5_shRNA_assay_compatibility','metadata_checks_passed':len(checks),'metadata_checks_failed':0,'checks':checks,'official_replacement_chain_verified':True,'shrna_experiment':{'accession':old['accession'],'status':old['status'],'assay':old['assay_term_name'],'possible_controls':[c['accession'] for c in old['possible_controls']],'submitter_comment':old['submitter_comment']},'crispr_experiment':{'accession':new['accession'],'status':new['status'],'assay':new['assay_term_name'],'target':'SRSF5','cell_model':'K562','possible_controls':controls},'crispr_replicates':replicates,'assay_compatibility':'ENCSR906RHU uses shRNA knockdown; ENCSR528YVC uses CRISPR genome editing. The accessions differ in perturbation method.','comparison_scope':'ENCODE accession linkage and assay metadata are verified. The CRISPR record is excluded from the shRNA endpoint comparison.','raw_expression_downloaded_by_this_audit':False,'response_normalization_performed':False,'gene_axis_intersection_evaluated':False,'new_predictions_or_metrics_generated':False,'target_transcript_QC_for_replacement_performed':False,'shrna_benchmark_modified':False,'production_functions_imported':False,'source_urls':{'shrna':'https://www.encodeproject.org/ENCSR906RHU/?format=json','crispr':'https://www.encodeproject.org/ENCSR528YVC/?format=json','control':'https://www.encodeproject.org/ENCSR292PXV/?format=json'},'source_sha256':{name:sha(P/name) for name in ['ENCSR906RHU.json','ENCSR528YVC.json','ENCSR292PXV.json','replacement_api_sources.json']},'independent_script_sha256':sha(__file__)}
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'metadata_checks_passed':len(checks),'shrna_assay':old['assay_term_name'],'crispr_assay':new['assay_term_name'],'crispr_control':controls,'replacement_metrics_generated':False},indent=2))
