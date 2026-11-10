"""Independently re-read original gzip rows and compare every preserved native value."""
from pathlib import Path
import csv,gzip,hashlib,json,re
import numpy as np
P=Path(__file__).resolve().parent
relay=json.loads((P/'remote_human_relay_audit.json').read_text());reports=[];axis=None
for row in relay['files']:
 path=Path(row['raw_file']);a=row['audit'];native=Path(a['native_file'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']==row['local_stream_sha256']
 assert hashlib.sha256(native.read_bytes()).hexdigest()==a['native_sha256']
 with np.load(native,allow_pickle=False) as f:
  ids=f['gene_ids'].tolist();base=f['unversioned_gene_ids'].tolist();quantities=f['quantities'].tolist();values=f['values'];transcripts=f['transcript_ids'].tolist()
 n=0;maxdiff=0.;md5=hashlib.md5();decompressed_bytes=0
 with gzip.open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):md5.update(b);decompressed_bytes+=len(b)
 with gzip.open(path,'rt',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t')
  for r in reader:
   assert ids[n]==r['gene_id']
   assert transcripts[n]==r.get('transcript_id(s)',r.get('transcript_ids',''))
   vals=np.asarray([float(r[k]) for k in quantities]);assert np.isfinite(vals).all() and (vals>=0).all()
   maxdiff=max(maxdiff,float(np.abs(vals-values[n]).max()));n+=1
 assert n==len(ids) and maxdiff==0
 assert len(ids)==len(set(ids)) and len(base)==len(set(base))
 if axis is None:axis=ids
 else:assert ids==axis
 # Metadata name correction leaves original arrays unchanged.
 sidecar=P/(row['srx']+'_audit.json');s=json.loads(sidecar.read_text())
 if 'one_to_one_mapping_ready' in s['audit']:s['audit']['version_stripped_axis_unique']=s['audit'].pop('one_to_one_mapping_ready')
 s['identifier_note']='Identifier uniqueness passed; orthology not yet established.';sidecar.write_text(json.dumps(s,indent=2)+'\n')
 reports.append({'srx':row['srx'],'rows_verified':n,'numeric_cells_verified':n*len(quantities),'max_absolute_difference':maxdiff,'gzip_crc_verified':True,'uncompressed_md5':md5.hexdigest(),'uncompressed_bytes':decompressed_bytes})
classes={'standard_ENSG_gene_ids':sum(bool(re.fullmatch(r'ENSG\d+(?:\.\d+)?(?:_PAR_Y)?',x)) for x in axis),'numeric_unknown_ids':sum(x.isdecimal() for x in axis),'ENSGR_special_ids':sum(x.startswith('ENSGR') for x in axis),'gSpikein_ERCC_ids':sum(x.startswith('gSpikein_ERCC') for x in axis),'gSpikein_phiX_ids':sum(x.startswith('gSpikein_phiX') for x in axis)}
result={'status':'pass','scope':'Independent full original-table to native-NPZ verification; no prediction or normalization','files':reports,'sample_count':len(reports),'features_per_sample':len(axis),'common_order_identical':True,'native_feature_types':classes,'total_raw_compressed_bytes':sum(r['bytes'] for r in relay['files']),'all_native_numeric_cells_verified':sum(r['numeric_cells_verified'] for r in reports),'max_absolute_difference':max(r['max_absolute_difference'] for r in reports),'source_selection_sha256':relay['selection_sha256'],'native_quantities':quantities,'orthology_ready':False,'normalization_applied':False,'model_training_ready':False,'raw_and_arrays_location':str(P),'local_raw_files_written':False,'native_schema':{'gene_ids':'1d original feature IDs, preserving version','unversioned_gene_ids':'1d original IDs with trailing gene annotation version removed; other identifiers preserved','values':'features by quantities matrix; columns indexed by quantities array','quantities':'expected_count, TPM, FPKM, length, effective_length','transcript_ids':'author original transcript_id(s) string per gene'}}
assert sum(classes.values())==len(axis)
result['normalization_candidate_note']='Strict standard ENSG numeric-ID subset has60675 features. Numeric unknown, ENSGR-special and spike-in features require explicit handling and are not automatically orthologous.'
(P/'independent_native_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['files','native_schema']},indent=2))
