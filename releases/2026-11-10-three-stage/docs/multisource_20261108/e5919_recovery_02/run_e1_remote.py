"""Frozen E1 author-TPM comparison; paired-author-individual sensitivity on CUDA."""
from pathlib import Path
import hashlib,json,time,platform
import numpy as np
import torch

ROOT=Path('/hy-tmp/cross-species-feedback-20261108/multisource/e5919_processed_2025')
E0=ROOT/'e0_v1'
OUT=ROOT/'e1_v1'
FAMILIES=('IFNB','polyIC')
RULES=('same_stimulus_mouse','zero_response','other_stimulus_mouse')
SEED=20261108
B=2000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cpu_metrics(p,h):
 p=np.asarray(p,float);h=np.asarray(h,float)
 pc=p-p.mean(axis=-1,keepdims=True);hc=h-h.mean(axis=-1,keepdims=True)
 denominator=np.sqrt(np.sum(pc**2,axis=-1)*np.sum(hc**2,axis=-1))
 correlation=np.divide(np.sum(pc*hc,axis=-1),denominator,out=np.full(np.shape(denominator),np.nan),where=denominator>0)
 return {'mse':np.mean((p-h)**2,axis=-1),'pearson':correlation,'predicted_rms':np.sqrt(np.mean(p**2,axis=-1)),'human_rms':np.sqrt(np.mean(h**2,axis=-1))}

def gpu_metrics(p,h):
 pc=p-p.mean(dim=-1,keepdim=True);hc=h-h.mean(dim=-1,keepdim=True)
 denominator=torch.sqrt((pc.square().sum(dim=-1))*(hc.square().sum(dim=-1)))
 correlation=torch.where(denominator>0,(pc*hc).sum(dim=-1)/denominator,torch.full_like(denominator,float('nan')))
 return {'mse':(p-h).square().mean(dim=-1),'pearson':correlation,'predicted_rms':p.square().mean(dim=-1).sqrt(),'human_rms':h.square().mean(dim=-1).sqrt()}

def scalar_metrics(p,h):
 return {key:float(value) if np.isfinite(value) else None for key,value in cpu_metrics(p,h).items()}

def quantiles(values):
 good=np.asarray(values)[np.isfinite(values)]
 return [float(x) for x in np.quantile(good,[.025,.5,.975])] if len(good) else None

def main():
 assert not OUT.exists(),'Use a new output version instead of replacing frozen results'
 audit=json.loads((E0/'audit.json').read_text())
 for name,expected in audit['files'].items():assert sha(E0/name)==expected,name
 protocol=ROOT/'E1_PROTOCOL_v1.md'
 assert protocol.exists()
 with np.load(E0/'author_processed_arrays.npz',allow_pickle=False) as source:
  tpm=source['author_tpm'].copy();genes=source['human_gene_ids'].copy();mouse_genes=source['mouse_gene_ids'].copy();samples=source['author_sample_ids'].copy()
 assert tpm.shape==(15851,36) and np.isfinite(tpm).all() and (tpm>=0).all()
 logtpm=np.log2(1+tpm)
 spec=json.loads((E0/'contrast_specification.json').read_text())
 metadata=json.loads((E0/'sample_metadata.json').read_text())
 donors={species:sorted({r['author_individual'] for r in spec if r['species']==species},key=int) for species in ('Homo_sapiens','Mus_musculus')}
 assert len(donors['Homo_sapiens'])==6 and len(donors['Mus_musculus'])==3
 responses={}
 for species in donors:
  rows=[]
  for family in FAMILIES:
   family_rows=[]
   for individual in donors[species]:
    selected=[r for r in spec if r['species']==species and r['family']==family and r['author_individual']==individual]
    assert len(selected)==1
    r=selected[0];t,c=r['treated_column'],r['control_column']
    assert metadata[t]['author_individual']==metadata[c]['author_individual']==individual
    assert metadata[t]['author_species']==metadata[c]['author_species']==species
    assert (metadata[t]['condition'],metadata[c]['condition'])==(('IFNB','UNST') if family=='IFNB' else ('PIC4','LF4'))
    family_rows.append(logtpm[:,t]-logtpm[:,c])
   rows.append(family_rows)
  responses[species]=np.asarray(rows)
 h=responses['Homo_sapiens'];m=responses['Mus_musculus']
 results=[]
 for s,family in enumerate(FAMILIES):
  predictions={'same_stimulus_mouse':m[s].mean(axis=0),'zero_response':np.zeros(len(genes)),'other_stimulus_mouse':m[1-s].mean(axis=0)}
  row={'stimulus':family,'mean_human_target':{rule:scalar_metrics(p,h[s].mean(axis=0)) for rule,p in predictions.items()},'human_individual_targets':[]}
  for i,individual in enumerate(donors['Homo_sapiens']):
   row['human_individual_targets'].append({'author_individual':individual,'rules':{rule:scalar_metrics(p,h[s,i]) for rule,p in predictions.items()}})
  base=row['mean_human_target']['zero_response']['mse']
  row['mean_target_relative_mse_reduction_over_zero']={rule:1-vals['mse']/base for rule,vals in row['mean_human_target'].items()}
  results.append(row)
 rng=np.random.default_rng(SEED)
 hi=rng.integers(0,6,size=(B,6));mi=rng.integers(0,3,size=(B,3))
 hw=np.eye(6)[hi].mean(axis=1);mw=np.eye(3)[mi].mean(axis=1)
 assert torch.cuda.is_available(),'CUDA requested by protocol; do not silently fall back'
 device=torch.device('cuda:0');torch.cuda.init();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(device)
 torch.cuda.synchronize(device);start=time.monotonic()
 ht=torch.as_tensor(h,device=device,dtype=torch.float64);mt=torch.as_tensor(m,device=device,dtype=torch.float64)
 hwt=torch.as_tensor(hw,device=device,dtype=torch.float64);mwt=torch.as_tensor(mw,device=device,dtype=torch.float64)
 hb=torch.matmul(hwt.unsqueeze(0),ht);mb=torch.matmul(mwt.unsqueeze(0),mt)
 bootstrap={}
 for s,family in enumerate(FAMILIES):
  rules={'same_stimulus_mouse':mb[s],'zero_response':torch.zeros_like(hb[s]),'other_stimulus_mouse':mb[1-s]}
  for rule,p in rules.items():
   for metric,value in gpu_metrics(p,hb[s]).items():bootstrap[family+'__'+rule+'__'+metric]=value.cpu().numpy()
 torch.cuda.synchronize(device);elapsed=time.monotonic()-start
 runtime={'device':str(device),'device_name':torch.cuda.get_device_name(device),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'dtype':'float64','resamples':B,'seed':SEED,'gpu_resampling_wall_seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'python_version':platform.python_version()}
 # Separate NumPy CPU pathway: reindex individual response rows directly.
 checks=0;maxdiff=0.
 for s,family in enumerate(FAMILIES):
  hc=np.stack([h[s,indices].mean(axis=0) for indices in hi[:20]])
  same=np.stack([m[s,indices].mean(axis=0) for indices in mi[:20]])
  other=np.stack([m[1-s,indices].mean(axis=0) for indices in mi[:20]])
  for rule,p in zip(RULES,(same,np.zeros_like(hc),other)):
   for name,cpu in cpu_metrics(p,hc).items():
    gpu=bootstrap[family+'__'+rule+'__'+name][:20]
    assert np.array_equal(np.isfinite(cpu),np.isfinite(gpu))
    finite=np.isfinite(cpu);difference=float(np.max(np.abs(cpu[finite]-gpu[finite]))) if finite.any() else 0.
    maxdiff=max(maxdiff,difference);checks+=len(cpu)
    assert difference<=1e-10,(family,rule,name,difference)
 sensitivity=[]
 for family in FAMILIES:
  baseline=bootstrap[family+'__zero_response__mse']
  row={'stimulus':family,'interpretation':'Descriptive sensitivity to resampling the observed author individuals; not a multi-stimulus population confidence interval or p-value','rules':{}}
  for rule in RULES:
   metrics={name:quantiles(bootstrap[family+'__'+rule+'__'+name]) for name in ('mse','pearson','predicted_rms','human_rms')}
   gain=1-bootstrap[family+'__'+rule+'__mse']/baseline
   metrics['relative_mse_reduction_over_zero_quantiles']=quantiles(gain)
   metrics['fraction_resamples_lower_mse_than_zero']=float(np.mean(gain>0))
   row['rules'][rule]=metrics
  sensitivity.append(row)
 OUT.mkdir()
 np.savez_compressed(OUT/'responses.npz',human=h,mouse=m,human_gene_ids=genes,mouse_gene_ids=mouse_genes,stimuli=np.asarray(FAMILIES),human_author_individuals=np.asarray(donors['Homo_sapiens']),mouse_author_individuals=np.asarray(donors['Mus_musculus']))
 np.savez_compressed(OUT/'resampling.npz',human_indices=hi,mouse_indices=mi,**bootstrap)
 report={'evidence_level':'E1_descriptive','protocol_sha256':sha(protocol),'script_sha256':sha(__file__),'e0_audit_sha256':sha(E0/'audit.json'),'input_files':audit['files'],'units':'paired difference in log2(1+author Salmon TPM), on author common ortholog subset; no column re-normalization','genes':len(genes),'human_author_individuals':6,'mouse_author_individuals':3,'stimulus_families':list(FAMILIES),'point_results':results,'resampling_sensitivity':sensitivity,'runtime':runtime,'cpu_gpu_validation':{'resamples_checked':20,'scalar_metric_comparisons':checks,'max_abs_difference':maxdiff,'tolerance':1e-10,'passed':True,'same_indices_shared_across_stimuli':True},'limitations':audit['limitations']+['No model training, source selection, fitted scale, or outcome-dependent feature selection.','Human and mouse author individuals are paired within species, not between species.','Sensitivity resamples6human and3mouse author individuals; independent stimulus family count remains2.'],'output_files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'point_results':[{'stimulus':r['stimulus'],'mean_human_target':r['mean_human_target'],'relative_mse_reduction':r['mean_target_relative_mse_reduction_over_zero']} for r in results],'runtime':runtime,'cpu_gpu_validation':report['cpu_gpu_validation']},indent=2))

if __name__=='__main__':main()
