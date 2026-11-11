#!/usr/bin/env python3
"""Verify scDFM population metrics and chemical-condition bootstrap intervals."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_species.official_state import sha256_file


def metrics(prediction, target, bandwidth):
    n,m=len(prediction),len(target)
    xx,yy,xy=cdist(prediction,prediction),cdist(target,target),cdist(prediction,target)
    energy=2*xy.mean()-xx.sum()/(n*(n-1))-yy.sum()/(m*(m-1))
    kxx,kyy,kxy=[np.exp(-distance**2/(2*bandwidth**2)) for distance in [xx,yy,xy]]
    mmd=(kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean()
    return {'energy_u':float(energy),'rbf_mmd2_u':float(mmd),
            'mean_mse':float(np.mean((prediction.mean(0)-target.mean(0))**2)),
            'variance_mae':float(np.mean(np.abs(prediction.var(0,ddof=1)-target.var(0,ddof=1))))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--processed-dir',type=Path,required=True)
    args=parser.parse_args()
    original_path=args.run_dir/'report.json'
    original=json.loads(original_path.read_text())
    spec=json.loads((args.run_dir/'specification.json').read_text())
    if original['specification_sha256']!=sha256_file(args.run_dir/'specification.json'):
        raise ValueError('Formal specification differs from report seal')
    matrix=sparse.load_npz(args.processed_dir/'expression_lognorm_hvg5000.npz')
    columns=np.asarray(spec['gene_columns'])
    checks={}
    baseline_targets=None
    for method,result in original['test'].items():
        path=args.run_dir/(method+'_test_predictions.npz')
        with np.load(path,allow_pickle=False) as archive:
            values={key:archive[key].copy() for key in archive.files}
        prediction,raw,target=values['predictions'],values['raw_predictions'],values['targets']
        if (prediction.shape!=raw.shape or prediction.ndim!=4 or target.shape!=(prediction.shape[0],prediction.shape[2],prediction.shape[3])
                or not all(np.isfinite(value).all() for value in [prediction,raw,target])):
            raise ValueError('Invalid saved population arrays: '+method)
        if not np.array_equal(prediction,np.maximum(raw,0)):
            raise ValueError('Raw/clipped output readout differs: '+method)
        if baseline_targets is None:baseline_targets=target.copy()
        if not np.array_equal(target,baseline_targets):raise ValueError('Methods use different test targets')
        if values['gene_names'].astype(str).tolist()!=spec['gene_names']:
            raise ValueError('Saved gene order differs from specification')
        conditions=values['condition_order'].astype(str).tolist()
        if conditions!=spec['roles']['test'] or conditions!=result['condition_order']:
            raise ValueError('Saved condition order differs')
        recomputed={}
        maximum_difference=0.
        for index,condition in enumerate(conditions):
            record=spec['evaluation_populations'][condition]
            expected=matrix[np.asarray(record['target_rows'])][:,columns].toarray()
            if not np.array_equal(expected,target[index]):raise ValueError('Saved truth differs from original real cells')
            draws=[metrics(draw,target[index],spec['rbf_bandwidth_fit_rows_only']) for draw in prediction[index]]
            recomputed[condition]={key:float(np.mean([draw[key] for draw in draws])) for key in draws[0]}
            for key,value in recomputed[condition].items():
                difference=abs(value-result['by_condition'][condition][key]);maximum_difference=max(maximum_difference,difference)
                if difference>1e-8:raise ValueError('Saved per-condition metric differs: '+method+'/'+condition+'/'+key)
        for key in result['macro']:
            value=float(np.mean([row[key] for row in recomputed.values()]))
            if abs(value-result['macro'][key])>1e-8:raise ValueError('Saved macro differs: '+method+'/'+key)
        checks[method]={'prediction_sha256':sha256_file(path),'conditions':len(conditions),'rollouts':prediction.shape[1],
                        'cells_per_population':prediction.shape[2],'genes':prediction.shape[3],
                        'maximum_recomputed_metric_difference':maximum_difference,'source_truth_and_raw_readout_verified':True}
    audited=deepcopy(original)
    comparisons=[(contrast,'released_checkpoint',method) for method,contrast in audited['versus_released'].items()]
    comparisons += [(audited['nft_vs_native_fm'],'native_fm','nft'),(audited['nft_vs_terminal_gradient'],'terminal_gradient','nft')]
    for contrast,baseline,method in comparisons:
        gains=np.asarray(contrast['per_condition_gain'])
        expected=np.array([original['test'][baseline]['by_condition'][condition]['energy_u']-original['test'][method]['by_condition'][condition]['energy_u']
                           for condition in contrast['conditions']])
        if not np.allclose(gains,expected,rtol=0,atol=1e-12):raise ValueError('Paired contrast direction/values differ')
        draws=np.random.default_rng(spec['seed']).integers(0,len(gains),size=(2000,len(gains)))
        interval=np.quantile(gains[draws].mean(1),[.025,.975]).tolist()
        inherited=contrast.pop('fixed_prediction_cytokine_bootstrap_95_interval')
        if not np.allclose(interval,inherited,rtol=0,atol=1e-12):raise ValueError('Bootstrap interval fails independent replay')
        contrast['fixed_prediction_chemical_condition_bootstrap_95_interval']=inherited
        contrast['bootstrap_unit']='chemical_condition'
        contrast['bootstrap_replicates']=2000
        contrast['warning']='Descriptive fixed-prediction interval over seven chemical conditions; shared controls, shared drug components, one A549 context and one optimization seed. No biological-replicate or population-coverage inference.'
    audit={'original_report_sha256':sha256_file(original_path),'specification_sha256':sha256_file(args.run_dir/'specification.json'),
           'checks':checks,'numerical_values_changed':False,
           'label_correction':'Original report reused State interval helper labels. This copy changes cytokine-specific field names and warnings to chemical-condition terminology; loss and interval numbers are preserved.',
           'audit_script_sha256':sha256_file(__file__)}
    audited['output_audit']=audit
    for name,value in [('output_audit.json',audit),('report_audited.json',audited)]:
        path=args.run_dir/name
        if path.exists():raise FileExistsError('Preserve prior audit: '+str(path))
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
