from pathlib import Path
import csv,gzip,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw';OUT=ROOT/'native';OUT.mkdir(exist_ok=False)
SAMPLES=json.loads((ROOT/'samples.json').read_text())
expected=json.loads((ROOT/'remote_download_manifest.json').read_text())+json.loads((ROOT/'remote_fpkm_download_manifest.json').read_text())
expected={Path(x['remote_path']).name:x for x in expected}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'artifact_role':'E0_author_estimated_count_and_condition_FPKM_versions','normalization_applied':False,'count_quantity':'author-reported Cuffdiff estimated gene fragment counts; raw versus library-scaled export field unresolved','raw_depth_inference_allowed':False,'integer_raw_count_inference_allowed':False,'counts_samples':32,'sample_metadata_sha256':sha(ROOT/'samples.json'),'files':[],'cross_compartment':{}}
axes={}
for tag,comp in [('WholeCell','whole_cell'),('Cyto','cytoplasmic')]:
 for kind,suffix in [('estimated_counts','Gene_Counts_by_Replicate'),('condition_fpkm','Gene_FPKMs')]:
  p=RAW/('GSE69733_'+tag+'_'+suffix+'.txt.gz');assert sha(p)==expected[p.name]['sha256']
  with gzip.open(p,'rt') as f:
   reader=csv.DictReader(f,delimiter='\t');header=reader.fieldnames
   assert header[0]=='ensembl_gene_id' and header[-3:]==['external_gene_name','description','gene_biotype']
   cols=header[1:-3];rows=list(reader)
  ids=[r['ensembl_gene_id'] for r in rows];assert len(ids)==len(set(ids)) and all(ids)
  values=np.asarray([[np.nan if r[c] in ['NA','NaN',''] else float(r[c]) for c in cols] for r in rows],dtype=np.float64)
  assert not np.isinf(values).any() and np.all(values[np.isfinite(values)]>=0)
  arrays={'author_values':values,'native_gene_ids':np.asarray(ids),'sample_columns' if kind=='estimated_counts' else 'condition_columns':np.asarray(cols),'gene_symbols':np.asarray([r['external_gene_name'] for r in rows]),'gene_biotype':np.asarray([r['gene_biotype'] for r in rows]),'observed_mask':np.isfinite(values)}
  columns_meta=[]
  if kind=='estimated_counts':
   bycol={r['matrix_column']:r for r in SAMPLES if r['compartment']==comp}
   assert len(bycol)==16 and set(bycol)==set(cols)
   columns_meta=[bycol[c] for c in cols]
   arrays.update(gsm=np.asarray([r['gsm'] for r in columns_meta]),experiment_ids=np.asarray([r['experiment'] for r in columns_meta]))
  else:
   assert cols==['SRSF'+str(i) for i in range(1,8)]+['GFP']
   columns_meta=[{'condition':c,'aggregation_level':'condition_summary','replicate_columns_available':False,'compartment':comp,'must_not_be_treated_as_individual_samples':True} for c in cols]
  out=OUT/(tag+'_'+kind+'.npz');np.savez_compressed(out,**arrays)
  with np.load(out,allow_pickle=False) as saved:
   for k,v in arrays.items():assert np.array_equal(saved[k],v,equal_nan=True) if v.dtype.kind=='f' else np.array_equal(saved[k],v)
  metap=OUT/(tag+'_'+kind+'_columns.json');metap.write_text(json.dumps(columns_meta,indent=2)+'\n')
  axis=OUT/(tag+'_'+kind+'_gene_axis.tsv')
  with axis.open('w') as f:
   w=csv.writer(f,delimiter='\t');w.writerow(['row_index_zero_based','ensembl_gene_id','gene_symbol','gene_biotype']);w.writerows((i,g,r['external_gene_name'],r['gene_biotype']) for i,(g,r) in enumerate(zip(ids,rows)))
  observed=np.isfinite(values);nonzero=values[observed & (values!=0)]
  stats={'compartment':comp,'kind':kind,'source_path':str(p),'source_sha256':sha(p),'native_npz':str(out),'npz_sha256':sha(out),'columns_path':str(metap),'columns_sha256':sha(metap),'axis_path':str(axis),'axis_sha256':sha(axis),'shape_genes_columns':list(values.shape),'unique_native_genes':len(ids),'duplicate_native_ids':0,'missing_entries':int((~observed).sum()),'negative_entries':0,'explicit_zero_entries':int((values==0).sum()),'all_zero_rows':int((values==0).all(axis=1).sum()),'fractional_nonzero_entries':int((np.abs(nonzero-np.rint(nonzero))>1e-8).sum()),'per_column_reported_mass':np.nansum(values,axis=0).tolist(),'is_raw_sequencing_depth':False,'raw_values_roundtrip_verified':True,'aggregation_level':'individual_biological_replicate_label' if kind=='estimated_counts' else 'condition_summary'}
  report['files'].append(stats);axes[(tag,kind)]=ids
report['cross_compartment']={'counts_axes_identical':axes[('WholeCell','estimated_counts')]==axes[('Cyto','estimated_counts')],'fpkm_axes_identical':axes[('WholeCell','condition_fpkm')]==axes[('Cyto','condition_fpkm')],'counts_fpkm_axes_identical_each_compartment':all(axes[(t,'estimated_counts')]==axes[(t,'condition_fpkm')] for t in ['WholeCell','Cyto']),'unique_count_genes_shared':len(set(axes[('WholeCell','estimated_counts')])&set(axes[('Cyto','estimated_counts')]))}
report['script_sha256']=sha(Path(__file__))
(OUT/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
