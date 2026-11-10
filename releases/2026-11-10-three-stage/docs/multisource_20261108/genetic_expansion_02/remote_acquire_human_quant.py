"""Bounded remote acquisition and native E0 curation of frozen ENCODE gene TSVs."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,re,time
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def curate(path,entry,out):
 ids=[];values=[];extra=[];columns=None
 with gzip.open(path,'rt',newline='') as f:
  reader=csv.DictReader(f,delimiter='\t');columns=reader.fieldnames
  required=['gene_id','expected_count','TPM']
  if not columns or any(k not in columns for k in required):raise ValueError('Expected RSEM gene_id/expected_count/TPM header, received '+str(columns))
  numeric=['expected_count','TPM']+[k for k in ['FPKM','length','effective_length'] if k in columns]
  for row in reader:
   if None in row or any(v is None for v in row.values()):raise ValueError('Malformed table row')
   if not row['gene_id']:raise ValueError('Empty gene identifier')
   vals=[float(row[k]) for k in numeric]
   if not all(math.isfinite(x) and x>=0 for x in vals):raise ValueError('Nonfinite/negative numeric cell')
   ids.append(row['gene_id']);values.append(vals);extra.append(row.get('transcript_id(s)',row.get('transcript_ids','')))
 if not ids:raise ValueError('Empty gene table')
 matrix=np.asarray(values,dtype=np.float64);gene_ids=np.asarray(ids);base=np.asarray([re.sub(r'\.\d+(?=_PAR_Y$|$)','',x) for x in ids])
 rawdup=len(ids)-len(set(ids));base_dup=len(base)-len(set(base.tolist()))
 native=out/'native';native.mkdir(exist_ok=True)
 dest=native/(entry['srx']+'.npz');np.savez_compressed(dest,gene_ids=gene_ids,unversioned_gene_ids=base,transcript_ids=np.asarray(extra),quantities=np.asarray(numeric),values=matrix)
 with np.load(dest,allow_pickle=False) as r:
  assert np.array_equal(r['gene_ids'],gene_ids) and np.array_equal(r['values'],matrix)
 vals={k:matrix[:,i] for i,k in enumerate(numeric)}
 return {'native_file':str(dest),'native_sha256':digest(dest),'gene_rows':len(ids),'header':columns,'quantity_columns':numeric,'numeric_quantity':'RSEM expected_count (estimated gene-level counts), TPM and optional FPKM; original values retained','expected_count_sum':float(vals['expected_count'].sum()),'TPM_sum':float(vals['TPM'].sum()),'expected_count_zeros':int((vals['expected_count']==0).sum()),'expected_count_fractional':int((vals['expected_count']!=np.floor(vals['expected_count'])).sum()),'raw_gene_id_duplicates':rawdup,'version_stripped_id_collisions':base_dup,'gene_id_order_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'gene_id_examples':ids[:8],'gzip_crc_verified':True,'no_normalization_applied':True,'no_zero_imputation':True,'native_e0_ready':rawdup==0,'version_stripped_axis_unique':rawdup==0 and base_dup==0,'readback_verified':True}

def acquire(entry,out,batch_end):
 raw=out/'raw';raw.mkdir(exist_ok=True);filename=entry['url'].rsplit('/',1)[1];path=raw/filename;partial=raw/(filename+'.part');start=time.monotonic()
 result={**entry,'started_unix':time.time(),'raw_file':str(path),'status':'started','max_bytes':100000000,'max_active_seconds':300}
 try:
  if not path.exists():
   if partial.exists():raise ValueError('Prior partial exists; choose a fresh directory to avoid ambiguous resume')
   with urlopen(Request(entry['url'],headers={'User-Agent':'Auditable-genetic-data-curation/1.0'}),timeout=20) as response:
    expected=int(response.headers.get('Content-Length') or 0)
    if expected>100000000:raise ValueError('100MB per file cap')
    result.update(http_status=response.status,content_length=expected or None,final_url=response.url)
    n=0
    with partial.open('wb') as f:
     while True:
      if time.monotonic()>batch_end or time.monotonic()-start>300:raise TimeoutError('Active download deadline')
      b=response.read1(65536)
      if not b:break
      n+=len(b)
      if n>100000000:raise ValueError('100MB cap')
      f.write(b)
    if expected and n!=expected:raise ValueError('Content-Length mismatch')
   # Fully parse and check gzip before naming downloaded resource complete.
   audit=curate(partial,entry,out);partial.replace(path)
  else:audit=curate(path,entry,out)
  result.update(status='complete_native_curated',bytes=path.stat().st_size,sha256=digest(path),audit=audit)
 except Exception as error:result.update(status='incomplete_or_failed',error=type(error).__name__+': '+str(error),partial_bytes=partial.stat().st_size if partial.exists() else 0)
 result['elapsed_seconds']=time.monotonic()-start;write(out/(entry['srx']+'_audit.json'),result)
 print(json.dumps({'srx':entry['srx'],'status':result['status'],'bytes':result.get('bytes'),'error':result.get('error')}),flush=True)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 selected=json.loads(args.selection.read_text());entries=selected['files'];assert len(entries)==26 and len({r['srx'] for r in entries})==26
 args.out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+1200;results=[]
 with ThreadPoolExecutor(max_workers=4) as executor:
  futures=[executor.submit(acquire,e,args.out,deadline) for e in entries]
  for future in as_completed(futures):results.append(future.result())
 results.sort(key=lambda r:r['srx']);success=[r for r in results if r['status']=='complete_native_curated'];sets=[]
 for r in success:
  with np.load(r['audit']['native_file'],allow_pickle=False) as f:sets.append(set(f['gene_ids'].tolist()))
 axes={'files_complete':len(success),'files_requested':len(entries),'complete_shared_native_ids':len(set.intersection(*sets)) if sets else 0,'native_gene_order_identical':len({r['audit']['gene_id_order_sha256'] for r in success})==1,'all_raw_gene_ids_unique':all(r['audit']['raw_gene_id_duplicates']==0 for r in success),'version_stripped_collisions_total':sum(r['audit']['version_stripped_id_collisions'] for r in success)}
 report={'selection_sha256':digest(args.selection),'script_sha256':digest(__file__),'location_policy':'Raw data and arrays only on remote /hy-tmp','batch_total_active_seconds_cap':1200,'workers':4,'files':results,'axes':axes,'native_e0_ready':len(success)==26 and axes['all_raw_gene_ids_unique'],'normalization_applied':False,'response_computed':False,'model_training_ready':False,'limitations':['RSEM expected_count is an estimated gene count; native TPM retained separately.','Current values are a separate ENCODE processing version; do not merge with portal or P19 gene-count units.','Preserve all primary/supplement/sensitivity sample selections and declared control relations.','No cross-species mapping, normalization or inference performed.']}
 write(args.out/'batch_audit.json',report);print(json.dumps(axes),flush=True)
 if len(success)!=26:raise SystemExit(2)
if __name__=='__main__':main()
