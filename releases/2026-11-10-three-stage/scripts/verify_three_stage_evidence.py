#!/usr/bin/env python3
"""Independently compare manuscript source data with frozen score records."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'docs/manuscript_rewrite_20261109/figures/three_stage'
def read(p):return json.loads((ROOT/p).read_text())
def rows(p):
 with (ROOT/p).open(newline='') as f:return list(csv.DictReader(f))
def near(a,b):
 if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):raise AssertionError((a,b))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args();checks=0
 roots={'exp2':'outputs/gse325993_state_20261110/state_primary_v1/audit','exp3':'outputs/gse325993_screen_validation_20261110/validation_v1/audit'}
 scores={k:rows(v+'/line_seed_scores.csv') for k,v in roots.items()}
 for r in rows(str((HERE/'source_data/Fig7_macaque_line_means.csv').relative_to(ROOT))):
  v=[float(x['mean_response_mse']) for x in scores[r['screen']] if x['variant']==r['group'] and x['target_line']==r['target_line']];assert len(v)==3;near(r['seed_mean_mse'],sum(v)/3);checks+=1
 for screen,path in roots.items():
  sums=read(path+'/summary.json')['variants']
  for arm in ['H','Hextra_cells','Monkey56','Monkey56A0','Monkey82','Monkey82A0','Three_quarter_H_response']:
   v=[float(x['mean_response_mse']) for x in scores[screen] if x['variant']==arm];assert len(v)==6;near(sum(v)/6,sums[arm]['macro']['mean_response_mse']);checks+=1
 for r in rows(str((HERE/'source_data/Fig7_m56_human_paired.csv').relative_to(ROOT))):
  v={x['variant']:float(x['mean_response_mse']) for x in scores['exp3'] if x['target_line']==r['line'] and x['seed']==r['seed']};near(r['mse_extra_human_minus_m56'],v['Hextra_cells']-v['Monkey56']);checks+=1
 rabbit=rows('outputs/nonrodent_singlecell_20261110/rabbit_primary_v1/audit/donor_scores.csv');bydonor=defaultdict(dict)
 for r in rabbit:bydonor[r['donor']][r['variant']]=float(r['mean_response_mse'])
 assert len(bydonor)==10
 for r in rows(str((HERE/'source_data/Fig7_rabbit_paired.csv').relative_to(ROOT))):
  v=bydonor[r['donor']];near(r['mse_H_minus_group'],v['H']-v[r['group']]);checks+=1
 rs=read('outputs/nonrodent_singlecell_20261110/rabbit_primary_v1/audit/summary.json')['variants']
 for arm in ['H','H+Hextra','H+Rabbit','H+RabbitA0','Hbase_mean_response','Zero_response']:
  near(sum(x[arm] for x in bydonor.values())/10,rs[arm]['mean_response_mse']['mean']);checks+=1
 assert all(x['H+Rabbit']>x['H'] and x['H+Rabbit']>x['H+RabbitA0'] and x['H+Hextra']<x['H'] for x in bydonor.values())
 m=read('docs/nonrodent_singlecell_20261110/prepared/manifest.json');budgets=rows(str((HERE/'source_data/Fig6_budgets.csv').relative_to(ROOT)))
 assert int(budgets[0]['human_added_cells'])==int(budgets[0]['animal_added_cells'])==256
 for r,f in zip(budgets[1:],m['fold_budgets']):
  assert int(r['human_added_cells'])==int(r['animal_added_cells'])==f['human_extra']==f['rabbit_source']==f['rabbit_null'];checks+=1
 prefix='outputs/remote_gpu_20261108/'
 native=read(prefix+'state_native_formal_64/report.json')['test'];mis=read(prefix+'state_mispaired_posthoc/report.json')['mismatched_test'];policy=read(prefix+'state_gaussian_policy_formal_64/report.json')['test'];sc=read(prefix+'scdfm_nft_formal/report.json')['test'];wi=read(prefix+'scdfm_witness_formal/evaluation_float64.json')['evaluation']
 lookup={'a':{'Released':native['released_checkpoint'],'Paired':native['native_energy_sft'],'Mispaired':mis},'b':{'Zero update':policy['zero_update_gaussian_reference'],'Pathwise':policy['pathwise'],'REINFORCE':policy['reinforce']},'c':{'Released':sc['released_checkpoint'],'Supervised':sc['native_fm'],'Reward update':sc['nft'],'Direct gradient':sc['terminal_gradient']},'d':{'Released':wi['released_checkpoint'],'Witness':wi['witness'],'Matched supervision':wi['native_fm']}}
 for r in rows(str((HERE/'source_data/Fig5_adaptation.csv').relative_to(ROOT))):
  near(r['value'],lookup[r['panel']][r['method']]['by_condition'][r['condition']][r['metric']]);checks+=1
 sec=ROOT/'docs/manuscript_rewrite_20261109/sections'
 if sec.exists():
  main='\n'.join(p.read_text() for p in sec.glob('*.md') if p.name not in ['supplement.md','theory_and_hypotheses.md','results_sources.md'])
  assert all(x not in main for x in ['20261110, 20261111','20261110/11/12','research planning','literature and provenance checks','## Ethics'])
  assert main.count('\\[')==2 and (sec/'supplement.md').read_text().count('\\[')==1
  reg=read('docs/manuscript_rewrite_20261109/figures/registry.json');assert len(reg)==7 and [x['number'] for x in reg]==list(range(1,8))
 result={'status':'passed','numeric_comparisons':checks,'biological_target_units':{'iPSC':2,'LPS':10},'scope':'Frozen scores, figure source data and current text constraints; no retraining or new biological replication.'}
 if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
