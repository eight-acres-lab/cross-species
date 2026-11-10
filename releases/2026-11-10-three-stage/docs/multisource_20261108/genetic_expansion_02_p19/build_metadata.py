from pathlib import Path
import csv,json,re,hashlib,collections
ROOT=Path(__file__).resolve().parent
soft=ROOT.parent/'genetic_expansion_02/evidence/GSE69733.soft'
records=[];current=None
for line in soft.read_text().splitlines():
 if line.startswith('^SAMPLE = '):
  current={'gsm':line.split(' = ',1)[1]};records.append(current)
 elif line.startswith('^'):current=None
 elif current is not None and line.startswith('!Sample_') and ' = ' in line:
  k,v=line[len('!Sample_'):].split(' = ',1);current.setdefault(k,[]).append(v)
with (ROOT/'evidence/SRP059321_runs.tsv').open() as f:runs=list(csv.DictReader(f,delimiter='\t'))
by_exp=collections.defaultdict(list)
for r in runs:by_exp[r['experiment_accession']].append(r)
output=[]
for r in records:
 chars=dict(x.split(': ',1) for x in r['characteristics_ch1'])
 target=chars['genotype/variation'].removeprefix('KD ')
 frac={'whole cell':'whole_cell','cytoplasm':'cytoplasmic'}[chars['cell fraction']]
 rep=re.search(r'Replicate ([AB])$',r['title'][0]).group(1)
 exps=[re.search(r'term=(SRX\d+)',x).group(1) for x in r['relation'] if re.search(r'term=(SRX\d+)',x)]
 assert len(exps)==1 and len(by_exp[exps[0]])>=1
 eruns=by_exp[exps[0]]
 assert all(x['sample_title']==r['title'][0] and x['library_strategy']=='RNA-Seq' for x in eruns)
 output.append({'gsm':r['gsm'],'experiment':exps[0],'sample_title':r['title'][0],'cell_model':'P19_embryonal_carcinoma','compartment':frac,'target_label':target,'matrix_column':target+'_'+rep,'biological_replicate_label':rep,'biological_replicate_evidence':'Publication Methods and GEO series declare two biological replicates per condition','role':'mock_control' if target=='GFP' else 'treated','time_h':36,'perturbation_method':'esiRNA','fraction_pair_label':target+'_'+rep,'fraction_pair_basis':'GEO series states cytoplasmic and whole-cell RNA from identical sample; A/B labels link fractions. Physical aliquot identifiers unavailable.','treatment_control_pairing_verified':False,'donor_id':None,'run_accessions':sorted(x['run_accession'] for x in eruns),'sequencing_runs':len(eruns),'run_count_is_biological_sample_count':False,'selected_primary_three_target_panel':target in ['SRSF1','SRSF5','SRSF7','GFP'],'selected_four_target_panel':target in ['SRSF1','SRSF3','SRSF5','SRSF7','GFP'],'treatment_protocol':r['treatment_protocol_ch1'],'extract_protocol':r['extract_protocol_ch1'],'source_soft_sha256':hashlib.sha256(soft.read_bytes()).hexdigest()})
assert len(output)==32 and len({r['experiment'] for r in output})==32
for target in ['SRSF'+str(i) for i in range(1,8)]+['GFP']:
 for rep in 'AB':assert {r['compartment'] for r in output if r['target_label']==target and r['biological_replicate_label']==rep}=={'whole_cell','cytoplasmic'}
(ROOT/'samples.json').write_text(json.dumps(output,indent=2)+'\n')
fields=['gsm','experiment','sample_title','cell_model','compartment','target_label','matrix_column','biological_replicate_label','role','time_h','fraction_pair_label','sequencing_runs','selected_primary_three_target_panel','selected_four_target_panel']
with (ROOT/'samples.tsv').open('w') as f:w=csv.DictWriter(f,fields,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(output)
report={'samples':32,'compartments':2,'biological_replicates_per_condition':2,'sequencing_runs':len(runs),'runs_per_sample_distribution':dict(collections.Counter(r['sequencing_runs'] for r in output)),'fraction_pairs_by_condition_replicate':16,'selected_primary_samples':sum(x['selected_primary_three_target_panel'] for x in output),'selected_four_target_samples':sum(x['selected_four_target_panel'] for x in output),'study':'GSE69733','PMID':'26944680','paper_doi':'10.1101/gad.276477.115','classification':'P19 embryonal carcinoma cells; whole/cyto measurements of same cell model','sample_metadata_sha256':hashlib.sha256((ROOT/'samples.json').read_bytes()).hexdigest()}
(ROOT/'sample_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
