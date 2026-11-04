#!/usr/bin/env python3
"""Fetch sample metadata for GSE249010 through ENA's public API."""
import csv, json, re, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

OUT=Path('artifacts/gse249010_v1/data'); OUT.mkdir(parents=True,exist_ok=True)
q=urllib.parse.quote('study_accession=PRJNA1047007',safe='')
u='https://www.ebi.ac.uk/ena/portal/api/search?result=sample&query='+q+'&fields=sample_accession,sample_alias,description&format=tsv&limit=10000'
raw=urllib.request.urlopen(u,timeout=60).read().decode()
records=[]
seen=set()
for row in csv.DictReader(raw.splitlines(),delimiter='\t'):
    if row['sample_accession'] not in seen:
        seen.add(row['sample_accession'])
        records.append({'sample_accession':row['sample_accession'],'sample_alias':row.get('sample_alias',''),'description':row.get('description','')})

def one(row):
    acc=row['sample_accession']
    url=f'https://www.ebi.ac.uk/ena/browser/api/xml/{acc}'
    try:
        xml=urllib.request.urlopen(url,timeout=60).read()
        root=ET.fromstring(xml)
        attrs={a.findtext('TAG'):a.findtext('VALUE') for a in root.findall('.//SAMPLE_ATTRIBUTE') if a.findtext('TAG')}
        desc=root.findtext('.//TITLE') or row.get('description','')
        m=re.search(r'\[([^\]]+)\]',desc)
        barcode=m.group(1) if m else ''
        return {'sample_accession':acc,'sample_alias':row['sample_alias'],'description':desc,'barcode':barcode,**attrs}
    except Exception as e:
        return {'sample_accession':acc,'sample_alias':row['sample_alias'],'description':'','error':repr(e)}

out=[]
with ThreadPoolExecutor(max_workers=16) as ex:
    futures=[ex.submit(one,r) for r in records]
    for i,f in enumerate(as_completed(futures),1):
        out.append(f.result())
        if i%50==0: print(i)
out.sort(key=lambda x:x.get('sample_alias',''))
json.dump(out,open(OUT/'sample_metadata.json','w'),indent=2)
fields=sorted(set().union(*(x.keys() for x in out)))
with open(OUT/'sample_metadata.tsv','w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(out)
print('samples',len(out),'errors',sum('error' in x for x in out),'barcodes',sum(bool(x.get('barcode')) for x in out))
