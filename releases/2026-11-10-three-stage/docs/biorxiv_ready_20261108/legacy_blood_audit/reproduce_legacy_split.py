"""Evaluate timepoint-specific sorted-unit splits with zero-filled float32 values."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path('/hy-tmp/cross-species-feedback-20261108/biorxiv_ready_20261108/legacy_blood')
sys.path.insert(0,str(ROOT/'code'))
import run_rho_reliability_and_phylogeny as old
def main():
 meta=json.loads((ROOT/'prepared_v1/response_metadata.json').read_text())
 with np.load(ROOT/'prepared_v1/observations.npz',allow_pickle=False) as z:
  x=np.log2(np.nan_to_num(z['tpm'].astype(np.float32),nan=0)+np.float32(1))
 responses=np.stack([x[r['treated_index']]-x[r['control_index']] for r in meta])
 rows=[{'species':r['species'],'subject':r['subject'],'time_h':str(r['time_h'])} for r in meta]
 values=[]
 for time in ['2','6','24']:
  units={}
  for row in rows:
   if row['time_h']==time:units.setdefault(row['species'],set()).add(row['subject'])
  halves=[set(),set()]
  for species,ids in units.items():
   ids=sorted(ids);halves[0]|=set(ids[:len(ids)//2]);halves[1]|=set(ids[len(ids)//2:])
  matrices=[old.rho_matrix(old.species_vectors(rows,responses,lambda row:row['time_h']==time,side)) for side in halves]
  pairs=sorted(set(matrices[0])&set(matrices[1]));a=np.array([matrices[0][key] for key in pairs]);b=np.array([matrices[1][key] for key in pairs])
  values.append({'time_h':time,'spearman':old.spearman(a,b),'units':{s:len(v) for s,v in units.items()},'halves':[sorted(side) for side in halves]})
 report={'scope':'Sensitivity analysis with NA values filled with zero, float32 log2 transformation, all 388 responses, and available units sorted separately at each timepoint for split assignment.','per_time':values,'mean_split_half':float(np.mean([row['spearman'] for row in values])),'difference_from_primary_split_design':'The sensitivity analysis assigns units separately at each timepoint; the primary analysis uses a common assignment across times and conditions. Missing unit-time combinations can change half assignments.','legacy_script_sha256':hashlib.sha256((ROOT/'code/run_rho_reliability_and_phylogeny.py').read_bytes()).hexdigest()}
 assert abs(report['mean_split_half']-.9)<1e-12
 (ROOT/'analysis_v1/legacy_exact_compatibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'legacy_mean_split_half':report['mean_split_half']}))
if __name__=='__main__':main()
