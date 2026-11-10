import json,os,time,hashlib,csv,io
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
p=Path('/hy-tmp/cross-species-feedback-20261108/multisource/e5919_processed_2025');p.mkdir(parents=True,exist_ok=True)
commit='2436f382cf80553c68442f5b5ff3d6492dca2971'
files=['EdgeR/count_matrix/homo_vs_mus_countData_Hagai.txt','TPM_heatmap/homo_vs_mus_tpmData_Hagai.txt']
def acquire(name):
 u='https://raw.githubusercontent.com/HagaiLab/transcriptional_evolution_of_antiviral_genes/'+commit+'/'+name
 start=time.monotonic();r={'url':u,'path':str(p/name.split('/')[-1])}
 try:
  with urlopen(Request(u,headers={'User-Agent':'E5919-recovery-audit/1.0'}),timeout=45) as h:
   r['http_status']=h.status;r['content_length']=h.headers.get('Content-Length');body=h.read(10000001)
  if len(body)>10000000:raise ValueError('10MB cap')
  q=Path(r['path']);q.write_bytes(body)
  rawlines=body.decode().splitlines()
  header=next(csv.reader(rawlines,delimiter='\t'))
  r.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),lines=len(rawlines),header=header,first_row_preview=rawlines[1][:500],status='downloaded')
 except Exception as e:r.update(status='failed',error=str(e))
 r['elapsed_seconds']=time.monotonic()-start
 return r
with ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(acquire,files))
def probe(u):
 start=time.monotonic();r={'url':u,'requested_bytes':1048576,'purpose':'bounded reachability prefix; no FASTQ file saved'}
 try:
  with urlopen(Request(u,headers={'User-Agent':'E5919-recovery-audit/1.0','Range':'bytes=0-1048575'}),timeout=35) as h:
   r.update(http_status=h.status,content_range=h.headers.get('Content-Range'),content_length=h.headers.get('Content-Length'))
   b=h.read(1048576)
  r.update(received_bytes=len(b),gzip_magic=b[:2]==b'\x1f\x8b',sha256=hashlib.sha256(b).hexdigest(),status='prefix_received')
 except Exception as e:r.update(status='failed',error=str(e))
 r['elapsed_seconds']=time.monotonic()-start
 if r.get('received_bytes'):r['effective_prefix_MiB_s']=r['received_bytes']/1048576/r['elapsed_seconds']
 return r
probe_urls=['https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR206/002/ERR2060212/ERR2060212_1.fastq.gz', 'https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR206/008/ERR2060258/ERR2060258_1.fastq.gz']
with ThreadPoolExecutor(max_workers=2) as pool:probes=list(pool.map(probe,probe_urls))
resource={}
for name,path in [('cpu_quota','/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),('cpu_period','/sys/fs/cgroup/cpu/cpu.cfs_period_us'),('memory_limit','/sys/fs/cgroup/memory/memory.limit_in_bytes')]:
 try:resource[name]=Path(path).read_text().strip()
 except Exception:pass
report={'repo_commit':commit,'processed_files':out,'fastq_prefix_probes':probes,'resources':resource,'fastq_downloaded':False,'indexes_downloaded':False}
(p/'acquisition_probe.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
