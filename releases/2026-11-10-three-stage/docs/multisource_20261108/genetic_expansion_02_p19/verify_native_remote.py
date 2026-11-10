from pathlib import Path
import csv,gzip,hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
m=json.loads((R/'native/manifest.json').read_text());out=[]
for e in m['files']:
 src=Path(e['source_path']);npz=Path(e['native_npz'])
 assert hashlib.sha256(src.read_bytes()).hexdigest()==e['source_sha256'] and hashlib.sha256(npz.read_bytes()).hexdigest()==e['npz_sha256']
 with np.load(npz,allow_pickle=False) as a:
  ids=a['native_gene_ids'];values=a['author_values'];cols=a['sample_columns'] if e['kind']=='estimated_counts' else a['condition_columns']
  assert np.isfinite(values).all() and (values>=0).all()
  n=0
  with gzip.open(src,'rt') as f:
   for row in csv.DictReader(f,delimiter='\t'):
    assert row['ensembl_gene_id']==ids[n]
    for j,col in enumerate(cols):assert float(row[col])==values[n,j]
    n+=1
  assert n==values.shape[0]
  out.append({'kind':e['kind'],'compartment':e['compartment'],'rows':n,'numeric_values_exactly_verified':int(values.size),'status':'pass'})
r={'status':'pass','checks':out,'all_original_values_preserved':True,'normalization_applied':False,'manifest_sha256':hashlib.sha256((R/'native/manifest.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'native_independent_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
