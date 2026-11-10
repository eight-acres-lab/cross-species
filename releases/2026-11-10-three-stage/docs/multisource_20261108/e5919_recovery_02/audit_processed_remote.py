"""Audit the 2025 author reanalysis as its own processed E0 version."""
from pathlib import Path
import csv,hashlib,json,re
import numpy as np

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/multisource/e5919_processed_2025')
OUT=ROOT/'e0_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matrix(path):
 with path.open() as f:
  reader=csv.reader(f,delimiter='\t');header=next(reader)
  assert header[0]=='' and len(header)==37 and len(set(header[1:]))==36
  genes=[];values=[]
  for row in reader:
   assert len(row)==37 and row[0].startswith('ENSG')
   genes.append(row[0]);values.append([float(v) for v in row[1:]])
 a=np.array(values,dtype=float)
 assert len(set(genes))==len(genes) and np.isfinite(a).all() and (a>=0).all()
 return genes,header[1:],a

def main():
 report={'repo_commit':'2436f382cf80553c68442f5b5ff3d6492dca2971','artifact_status':'E0_author_processed_common_gene_axis','new_normalization_applied':False,'native_complete_gene_matrix_available':False,'raw_accession_bridge_recovered':False,'old_x18957_bridge_recovered':False,'performance_evaluated':False}
 source=json.loads((ROOT/'processed_relay_manifest.json').read_text())
 for row in source['files']:
  assert sha(row['remote_path'])==row['sha256']
 genes,samples,counts=matrix(ROOT/'homo_vs_mus_countData_Hagai.txt')
 genes2,samples2,tpm=matrix(ROOT/'homo_vs_mus_tpmData_Hagai.txt')
 assert genes==genes2 and samples==samples2
 report.update(matrix_shape=list(counts.shape),count_values_fractional=int(np.count_nonzero(counts!=np.round(counts))),count_values_zero=int(np.count_nonzero(counts==0)),tpm_values_fractional=int(np.count_nonzero(tpm!=np.round(tpm))),tpm_values_zero=int(np.count_nonzero(tpm==0)),counts_tpm_identical=bool(np.array_equal(counts,tpm)),counts_column_sums=counts.sum(axis=0).tolist(),tpm_column_sums=tpm.sum(axis=0).tolist())
 with (ROOT/'small_metaData_Hagai_homo_vs_mus.txt').open() as f:
  reader=csv.reader(f,delimiter='\t');header=next(reader)
  assert header==['run_accession','study_name','run_accession','treatment','species','individual']
  sample_rows=list(reader)
 assert len(sample_rows)==36 and all(len(row)==6 and row[0]==row[2] for row in sample_rows)
 lookup={row[0]:row for row in sample_rows}
 assert len(lookup)==36 and set(lookup)==set(samples)
 metadata=[];groups={}
 for i,sample in enumerate(samples):
  row=lookup[sample];condition,species,individual=row[3:]
  assert sample==f'Hagai_{condition}_{species}_{individual}'
  assert species in ('Homo_sapiens','Mus_musculus') and condition in ('IFNB','LF4','PIC4','UNST')
  groups.setdefault((species,individual),{})[condition]=i
  metadata.append({'column_index':i,'author_sample_id':sample,'sample_identifier_kind':'author_derived_label_not_ENA_accession','author_species':species,'author_individual':individual,'author_donor_key':species+'|'+individual,'condition':condition,'cell_type':'dermal_fibroblast','sdrf_donor_id':None,'ena_run_accession':None,'identity_basis':'Exact frozen author sample_info row; no numeric correspondence to SDRF assumed','deposited_timing_h':{'IFNB':8,'UNST':0,'PIC4':4,'LF4':4}[condition],'timing_basis':'Condition-level original SDRF protocol; per-run accession connection unresolved'})
 assert all(set(v)=={'IFNB','LF4','PIC4','UNST'} for v in groups.values())
 assert sum(key[0]=='Homo_sapiens' for key in groups)==6 and sum(key[0]=='Mus_musculus' for key in groups)==3
 with (ROOT/'1-1_human_mouse_metadata.csv').open() as f:
  rows=list(csv.DictReader(f))
 annotation={};ambiguous=set()
 for row in rows:
  human=row[''];identity=(row['Mouse gene stable ID'],row['Mouse homology type'])
  if human in annotation and annotation[human]!=identity:ambiguous.add(human)
  annotation[human]=identity
 assert not set(genes)&ambiguous
 assert set(genes)<=set(annotation)
 mouse=[annotation[g][0] for g in genes]
 assert all(m.startswith('ENSMUSG') for m in mouse) and len(set(mouse))==len(mouse)
 assert all(annotation[g][1]=='ortholog_one2one' for g in genes)
 # Check published group means against supplied per-sample TPM; does not assign any ENA identities.
 meta={r['']:r for r in rows}
 max_difference=0.0;checks=0
 for species,short in [('Homo_sapiens','homo_sapiens'),('Mus_musculus','mus_musculus')]:
  for condition,condshort in [('IFNB','IFNB'),('UNST','UNST'),('PIC4','PIC'),('LF4','LF')]:
   ix=[i for i,m in enumerate(metadata) if m['author_species']==species and m['condition']==condition]
   mean=tpm[:,ix].mean(axis=1)
   author=np.array([float(meta[g]['Hagai_'+short+'_'+condshort+'_av_TPM']) for g in genes])
   assert np.isfinite(author).all()
   delta=float(np.max(np.abs(mean-author)));max_difference=max(max_difference,delta);checks+=len(genes)
 report['tpm_group_mean_consistency']={'comparisons':checks,'max_abs_difference':max_difference,'allclose_rtol_1e-8_atol_1e-8':max_difference<1e-8,'identity_bridge_inferred_from_values':False}
 contrasts=[]
 for (species,individual),indices in sorted(groups.items()):
  for family,treated,control in [('IFNB','IFNB','UNST'),('polyIC','PIC4','LF4')]:
   contrasts.append({'contrast_id':species+'|author_individual_'+individual+'|'+family,'species':species,'author_individual':individual,'family':family,'treated_column':indices[treated],'control_column':indices[control],'author_pairing_basis':'Same explicit individual field in author sample_info','time_matched_control':family=='polyIC','response_computed':False})
 OUT.mkdir(exist_ok=False)
 np.savez_compressed(OUT/'author_processed_arrays.npz',estimated_counts=counts,author_tpm=tpm,human_gene_ids=np.array(genes),mouse_gene_ids=np.array(mouse),author_sample_ids=np.array(samples))
 (OUT/'sample_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 (OUT/'contrast_specification.json').write_text(json.dumps(contrasts,indent=2)+'\n')
 with (OUT/'gene_axis.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['human_gene_id','mouse_gene_id','author_orthology_type']);w.writerows((h,m,'ortholog_one2one') for h,m in zip(genes,mouse))
 report.update(donors={'human_author_labels':6,'mouse_author_labels':3},sample_columns=36,paired_responses=len(contrasts),stimulus_families=['IFNB','polyIC'],common_axis_genes=len(genes),orthology_source='Author metadata stable-ID pairs, annotated ortholog_one2one; no current Ensembl confidence attached',measurement_units={'estimated_counts':'Salmon estimated gene counts according to 2025 methods and author rounding comment; README uses ambiguous term TPM counts','author_tpm':'Salmon gene TPM according to dedicated TPM file, repository README and group-mean numerical agreement'},limitations=['Author common-gene subset is not the full native transcriptome; count-column totals cannot automatically be used as full library sizes.','The author metadata labels do not provide raw ENA run links. Do not rewrite them as original donor accessions or infer x18957 correspondence.','Two condition families share author donors. No independent perturbation population or source-selection generalization.','IFNB8h uses untreated0h control; time in culture remains confounded. polyIC4h uses4h lipofectamine control.','Only human and mouse sample-level arrays supplied here; four-species table in repo contains DE summaries, not per-donor count matrices.','Author one-to-one axis version should remain distinct from current-release axes.'],files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},input_files={row['remote_path']:row['sha256'] for row in source['files']},audit_script_sha256=sha(__file__))
 with np.load(OUT/'author_processed_arrays.npz',allow_pickle=False) as saved:
  assert np.array_equal(saved['estimated_counts'],counts) and np.array_equal(saved['author_tpm'],tpm)
 report['readback_verified']=True
 (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
