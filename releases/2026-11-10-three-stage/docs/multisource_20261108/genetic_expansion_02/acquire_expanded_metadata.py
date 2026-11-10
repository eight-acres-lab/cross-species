from pathlib import Path
from urllib.request import urlopen,Request
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,xml.etree.ElementTree as E
p=Path(__file__).resolve().parent;root=p/'evidence';catalog=json.loads((p.parent/'genetic_extension/human_mouse_catalog.json').read_text())
pids=['Perturb_8241','Perturb_8242','Perturb_8096','Perturb_8214','Perturb_8163','Perturb_9926','Perturb_9927','Perturb_9928']
rows=[r for r in catalog if r['perturb_id'] in pids];ids=sorted({i for r in rows for s in ['controls','perturbs'] for i in r[s].split(',')})
def get(i):
 records=[];f=root/(i+'.xml');u='https://www.ebi.ac.uk/ena/browser/api/xml/'+i
 try:
  if not f.exists():f.write_bytes(urlopen(Request(u,headers={'User-Agent':'ResearchMetadataAudit/1.0'}),timeout=40).read())
  t=E.fromstring(f.read_bytes());gsm=t.find('.//EXPERIMENT').attrib['alias'];records.append({'accession':i,'url':u,'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  u='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+gsm+'&targ=self&form=text&view=full';f=root/(gsm+'.soft')
  if not f.exists():f.write_bytes(urlopen(Request(u,headers={'User-Agent':'ResearchMetadataAudit/1.0'}),timeout=40).read())
  records.append({'accession':gsm,'url':u,'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 except Exception as e:records.append({'accession':i,'url':u,'error':str(e)})
 return records
with ThreadPoolExecutor(max_workers=4) as pool: rr=list(pool.map(get,ids))
r=sum(rr,[]);(p/'expanded_human_metadata_sources.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'experiments':len(ids),'source_records':len(r),'failures':[r for r in r if 'error' in r]}))
