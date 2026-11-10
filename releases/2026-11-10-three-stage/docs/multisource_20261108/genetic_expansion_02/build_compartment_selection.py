"""Frozen metadata selection for P19 measurement-compartment comparisons."""
from pathlib import Path
import csv,hashlib,json,re
from build_batch_audit import parse_soft
P=Path(__file__).resolve().parent
catalog=json.loads((P.parent/'genetic_extension/human_mouse_catalog.json').read_text());byid={r['perturb_id']:r for r in catalog}
# Select by released, directly recorded knockdown characterization before any expression outcome.
primary_human=['Perturb_8241','Perturb_8096','Perturb_8242','Perturb_9926','Perturb_9927','Perturb_9928']
primary_mouse=['Perturb_4693','Perturb_4697','Perturb_4699','Perturb_4686','Perturb_4690','Perturb_4692']
supplement=['Perturb_8163','Perturb_4695','Perturb_4688']
sensitivity=['Perturb_8214']
lookup={}
paths=[P/'evidence/GSE69733.soft',*sorted((P/'evidence').glob('GSM*.soft'))]
for path in paths:
 for r in parse_soft(path):
  for sr in re.findall(r'SRX\d+',' '.join(r.get('relation',[]))):
   if sr in lookup:raise ValueError('duplicate original SRX '+sr)
   lookup[sr]=(r,path)
entries=[];samples=[];resources=[];seen_files=set()
for pid in primary_human+primary_mouse+supplement+sensitivity:
 row=byid[pid];target=row['target'].rstrip('-').upper();human=row['scientific_name']=='Homo sapiens';cell='HepG2' if 'HepG2' in row['tissue_name'] else 'K562' if human else 'P19';fraction='whole_cell' if human or '(whole cell)' in row['tissue_name'] else 'cytoplasmic';role_set='primary' if pid in primary_human+primary_mouse else 'supplement_SRSF3_HepG2_only' if pid in supplement else 'sensitivity_alternative_HepG2_SRSF5_batch'
 entry={'perturb_id':pid,'target':target,'condition':'KD','species':row['scientific_name'],'cell_model':cell,'measurement_compartment':fraction,'selection_role':role_set,'study_accession':row['study_accession'],'controls':row['controls'].split(','),'perturbs':row['perturbs'].split(','),'sample_identity_and_role_verified':True,'assay_method':'shRNA' if human else 'esiRNA','training_ready':False}
 econtrols=[];etargetcontrols=[]
 for role,key in [('control','controls'),('treated','perturbs')]:
  for sr in entry[key]:
   r,path=lookup[sr];chars=r.get('characteristics_ch1',[]);desc=r.get('description',[]);bio=None;tech=None;encode=None
   if human:
    assert 'line: '+cell in chars
    assert ('Non-specific target control' if role=='control' else target) in r['source_name_ch1'][0]
    bio=next(x.split(': ',1)[1] for x in desc if x.startswith('biological replicate number:'))
    tech=next(x.split(': ',1)[1] for x in desc if x.startswith('technical replicate number:'))
    encode=next(x.split(': ',1)[1] for x in desc if x.startswith('experiment encode accession:'))
    if role=='control':econtrols.append(encode)
    else:etargetcontrols+=next(x.split(': ',1)[1].split(',') for x in desc if x.startswith('possible controls:'))
    urls=[v for k,vs in r.items() if k.startswith('supplementary_file') for v in vs if '_gene_quantifications_GRCh38.tsv.gz' in v]
    assert len(urls)==1
    url=urls[0].replace('ftp://','https://')
    if url not in seen_files:
     resources.append({'srx':sr,'gsm':r['gsm'],'species':row['scientific_name'],'cell_model':cell,'title':r['title'][0],'url':url,'assembly':'GRCh38','file_kind':'ENCODE_gene_quantifications','declared_quantity':'Gene quantification TSV; inspect header expected_count/TPM/FPKM before quantity choice','biological_replicate_label':bio,'technical_replicate_label':tech,'encode_experiment':encode,'download_location':'remote_only','source_metadata':str(path.relative_to(P))});seen_files.add(url)
   else:
    assert 'cell fraction: '+('whole cell' if fraction=='whole_cell' else 'cytoplasm') in chars
    assert 'genotype/variation: KD '+('GFP' if role=='control' else target) in chars
    bio=r['title'][0].rsplit(' ',1)[-1]
   samples.append({'perturb_id':pid,'selection_role':role_set,'target':target,'cell_model':cell,'measurement_compartment':fraction,'srx':sr,'gsm':r['gsm'],'role':role,'title':r['title'][0],'biological_replicate_label':bio,'technical_replicate_label':tech,'encode_experiment':encode,'method':'shRNA' if human else 'esiRNA','time_h':None if human else 36,'source':str(path.relative_to(P)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'p19_fraction_pair_label':None if human else ('GFP' if role=='control' else target)+'|'+bio,'donor_id':None})
 if human:assert set(etargetcontrols)==set(econtrols),(pid,etargetcontrols,econtrols)
 entries.append(entry)
geo=(P/'evidence/GSE69733.soft').read_text()
for line in geo.splitlines():
 if line.startswith('!Series_supplementary_file = ') and any(k in line for k in ['WholeCell_Gene_Counts_by_Replicate','Cyto_Gene_Counts_by_Replicate']):
  url=line.split(' = ',1)[1].replace('ftp://','https://');resources.append({'species':'Mus musculus','cell_model':'P19','measurement_compartment':'whole_cell' if 'WholeCell' in url else 'cytoplasmic','url':url,'file_kind':'author_Cufflinks_gene_counts_by_replicate','declared_quantity':'Author Cufflinks gene counts; inspect sample columns and numeric scale; preserve author version separately from portal','download_location':'remote_only','source_metadata':'evidence/GSE69733.soft'})
(P/'compartment_selection.json').write_text(json.dumps({'date':'2026-11-08','primary_targets':['SRSF1','SRSF5','SRSF7'],'primary_human_models':['HepG2','K562'],'mouse_cell_models':['P19'],'mouse_measurement_compartments':['whole_cell','cytoplasmic'],'entries':entries,'primary_contrasts':12,'primary_unique_samples':len({r['srx'] for r in samples if r['selection_role']=='primary'}),'all_unique_samples':len({r['srx'] for r in samples}),'technical_pairing':'Study describes whole-cell/cytoplasmic fractions from identical sample. A/B labels are aligned within target/control; physical aliquot/donor IDs unreported. Fraction-pair labels must not inflate independent sample count.','SRSF3_K562':'No released shRNA contrast in frozen catalog; official ENCORE reports shRNA NotSatisfied and a separate released CRISPR RNA-seq ENCSR914IEV. Excluded from same-mode primary rectangle.','SRSF5_HepG2_choice':'ENCSR781YNI/Perturb_8096 has direct reported KD67.0/74.9%; choose before expression outcome. ENCSR447UCG/Perturb_8214 retained separately as alternative batch.','SRSF5_K562_quality_flag':'ENCSR906RHU released shRNA reported KD23.1/20.5%; preserve as low efficacy and report SRSF1/SRSF7 subset sensitivity, do not assume equivalent perturbation strength.','claims_allowed':'Exploratory dependence on source measurement compartment in fixed observed tasks; human target context can be compared with study/batch limits.','claims_not_supported':['Two distinct mouse cell sources','General new-target source recommendation','Pure causal cell-compartment or cell-type effects','Independent-donor inference from replicate labels'],'model_training_ready':False},indent=2,ensure_ascii=False)+'\n')
(P/'compartment_sample_bridge.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False)+'\n')
with (P/'compartment_sample_bridge.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(samples[0]),delimiter='\t');w.writeheader();w.writerows(samples)
(P/'compartment_expression_sources.json').write_text(json.dumps(resources,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'entries':len(entries),'samples':len({r['srx'] for r in samples}),'primary_samples':len({r['srx'] for r in samples if r['selection_role']=='primary'}),'expression_resources':len(resources)}))
