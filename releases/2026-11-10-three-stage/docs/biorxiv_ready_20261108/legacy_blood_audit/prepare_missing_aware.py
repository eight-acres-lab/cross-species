"""Rebuild public blood TPM observations without interpreting missing tokens as zero."""
from pathlib import Path
from collections import defaultdict, Counter
import csv,gzip,hashlib,json,re
import numpy as np

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/biorxiv_ready_20261108/legacy_blood')
OUT=ROOT/'prepared_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def soft_records(path):
 out={};current=None
 with gzip.open(path,'rt') as f:
  for line in f:
   line=line.rstrip('\r\n')
   if line.startswith('^SAMPLE = '):
    accession=line.split(' = ',1)[1];current={'geo_accession':accession};out[accession]=current
   elif current is not None:
    if line.startswith('!Sample_title = '):current['title']=line.split(' = ',1)[1]
    elif line.startswith('!Sample_organism_ch1 = '):current['species']=line.split(' = ',1)[1]
    elif line.startswith('!Sample_characteristics_ch1 = '):
     key,value=line.split(' = ',1)[1].split(':',1);current[key.strip()]=value.strip()
 return out

def main():
 assert not OUT.exists(),'Preserve frozen preparation'
 provenance=json.loads((ROOT/'input/provenance.json').read_text())
 matrix=ROOT/'input/GSE249010_processed_data_TPM.txt.gz';soft=ROOT/'input/GSE249010_family.soft.gz'
 for path in [matrix,soft]:assert sha(path)==provenance['downloaded_files'][path.name]['sha256']
 genes=[];values=[];missing_tokens=Counter()
 with gzip.open(matrix,'rt',newline='') as f:
  reader=csv.reader(f,delimiter='\t');header=next(reader);sample_ids=header[1:]
  assert len(sample_ids)==len(set(sample_ids))
  for row in reader:
   assert len(row)==len(header)
   genes.append(row[0]);v=[]
   for token in row[1:]:
    if token in {'NA','','null'}:missing_tokens[token]+=1;v.append(np.nan)
    else:
     number=float(token)
     assert np.isfinite(number) and number>=0
     v.append(number)
   values.append(v)
 assert len(genes)==len(set(genes))
 tpm=np.asarray(values,dtype=np.float64).T
 records=soft_records(soft);bybar={}
 for row in records.values():
  match=re.search(r'\[([^\]]+)\]',row.get('title',''))
  if match:
   barcode=match[1];assert barcode not in bybar;bybar[barcode]=row
 metadata=[];groups=defaultdict(list)
 for i,sample in enumerate(sample_ids):
  row=dict(bybar[sample]);row.update(sample_id=sample,column_index=i)
  row['species']='Macaca' if row['species'] in {'Macaca','Macaca mulatta','Macaca fascicularis'} else row['species']
  metadata.append(row);groups[(row['species'],row['subject'],row['time'])].append(i)
 pairs=[];excluded=[]
 for (species,subject,time),indices in sorted(groups.items()):
  controls=[i for i in indices if metadata[i].get('dose')=='0']
  if len(controls)!=1:
   excluded.append({'species':species,'subject':subject,'time':time,'reason':'control_count_not_one','controls':controls,'samples':[sample_ids[i] for i in indices]});continue
  for i in indices:
   row=metadata[i];dose=row.get('dose')
   if dose=='0':continue
   primary=dose in {'10','100','1000'} and time in {'2','6','24'}
   assert row.get('doseunit')=='ng/mL' and row.get('timeunit')=='Hour'
   pairs.append({'species':species,'subject':subject,'time_h':int(time),'dose_ng_ml':float(dose),'task_id':f'LPS|{dose}ngml|{time}h','treated_index':i,'control_index':controls[0],'treated_sample_id':sample_ids[i],'control_sample_id':sample_ids[controls[0]],'primary_shared_grid':primary})
 primary=[r for r in pairs if r['primary_shared_grid']]
 used=sorted({r[k] for r in primary for k in ['treated_index','control_index']})
 observed=np.isfinite(tpm);complete=observed[used].all(axis=0)
 expression=np.log2(1+tpm)
 responses=np.stack([expression[r['treated_index']]-expression[r['control_index']] for r in pairs])
 species=sorted({r['species'] for r in primary});tasks=sorted({r['task_id'] for r in primary})
 subjects={s:sorted({r['subject'] for r in primary if r['species']==s}) for s in species}
 assert len(species)==10 and len(tasks)==9 and sum(map(len,subjects.values()))==44
 assert all(any(r['species']==s and r['task_id']==t for r in primary) for s in species for t in tasks)
 unit_table=[{'species':s,'subject':u,'sampling_unit':'pool_of_25_animals' if s=='Mus musculus' else 'pool_of_4_animals' if s=='Rattus norvegicus' else 'individual_donor','independent_animal_count_not_used':25 if s=='Mus musculus' else 4 if s=='Rattus norvegicus' else 1} for s in species for u in subjects[s]]
 OUT.mkdir()
 np.savez_compressed(OUT/'observations.npz',tpm=tpm,log2_tpm=expression,responses=responses,observed_mask=observed,common_complete_mask=complete,gene_ids=np.asarray(genes),sample_ids=np.asarray(sample_ids))
 save(OUT/'sample_metadata.json',metadata);save(OUT/'response_metadata.json',pairs);save(OUT/'sampling_units.json',unit_table)
 audit={'status':'E0_missing_aware_inputs','source_hashes':{p.name:sha(p) for p in [matrix,soft]},'genes_total':len(genes),'sample_columns':len(sample_ids),'response_rows_all':len(pairs),'primary_response_rows':len(primary),'primary_measurement_columns':len(used),'primary_conditions':tasks,'sampling_units_by_species':{s:len(v) for s,v in subjects.items()},'sampling_units':sum(map(len,subjects.values())),'missing_token_counts':dict(missing_tokens),'missing_fraction':float((~observed).mean()),'common_complete_genes':int(complete.sum()),'common_complete_fraction':float(complete.mean()),'missing_by_species':{s:{'sample_columns':sum(r['species']==s for r in metadata),'missing_fraction':float((~observed[[r['column_index'] for r in metadata if r['species']==s]]).mean())} for s in species},'unpaired_control_groups_excluded':excluded,'gene_axis':'Author deposited common human-Ensembl-labelled axis; one-to-one orthology not independently verified.','normalization':'log2(author TPM+1), no renormalization and no clipping or filling of missing values','complete_case_rule':'Fixed gene axis finite across every measured sample needed for all387 primary response rows; depends on observation presence, not response magnitude','scope':'One LPS stimulus with3doses×3times and repeated biological units, including rodent pools. No model performance calculated.','files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'preparer_sha256':sha(__file__)}
 save(OUT/'audit.json',audit);print(json.dumps({k:audit[k] for k in ['sample_columns','response_rows_all','primary_response_rows','sampling_units','genes_total','common_complete_genes','missing_fraction']},indent=2))
if __name__=='__main__':main()
