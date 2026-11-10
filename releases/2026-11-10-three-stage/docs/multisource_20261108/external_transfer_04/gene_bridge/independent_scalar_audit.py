"""Audit the fixed-beta3x3 posthoc correspondence scores without refitting."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path('/hy-tmp/cross-species-feedback-20261108/external_transfer_04')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT/'posthoc_scalar_correspondence_v1'
    report = json.loads((out/'report.json').read_text())
    original = json.loads((ROOT/'run_v1/report.json').read_text())
    provenance = report['provenance']
    checks, numeric, maxdiff = 0, 0, 0.

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    def compare(actual, expected, label):
        nonlocal numeric, maxdiff
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        require(a.shape == b.shape, label+' shape')
        require(np.isfinite(a).all() and np.isfinite(b).all(), label+' finite')
        difference = float(np.max(np.abs(a-b))) if a.size else 0.
        maxdiff = max(maxdiff, difference)
        numeric += a.size
        require(difference < 1e-12, label+' values')

    for name, expected in provenance['original_run_files_sha256'].items():
        require(sha(ROOT/'run_v1'/name) == expected, 'original frozen file '+name)
    paths = {'input_manifest_sha256':ROOT/'inputs_v1/manifest.json',
             'train_npz_sha256':ROOT/'inputs_v1/train.npz',
             'external_inputs_sha256':ROOT/'inputs_v1/external_inputs.npz',
             'external_truth_sha256':ROOT/'inputs_v1/external_truth.npz',
             'output_npz_sha256':out/'fixed_scalar_correspondence.npz',
             'protocol_sha256':ROOT/'posthoc_scalar_correspondence_PROTOCOL_v1.zh.md',
             'script_sha256':ROOT/'posthoc_scalar_correspondence_run.py'}
    for key,path in paths.items():
        require(sha(path)==provenance[key], 'posthoc provenance '+key)
    with np.load(ROOT/'inputs_v1/external_inputs.npz',allow_pickle=False) as x:
        source=x['source'].astype(float)
        tasks=x['task_ids'].tolist()
        human_ids=x['human_gene_ids'];rat_ids=x['rat_gene_ids']
    with np.load(ROOT/'inputs_v1/external_truth.npz',allow_pickle=False) as y:
        truth=y['target'].astype(float)
        require(y['task_ids'].tolist()==tasks, 'truth task identities')
    with np.load(ROOT/'run_v1/final_fit.npz',allow_pickle=False) as state:
        beta=float(state['scalar_nointercept_beta'])
        history=state['human_mean'].astype(float)
    predictions=source*beta
    matrix=np.asarray([[math.fsum((float(p)-float(y))**2 for p,y in zip(predictions[j],truth[i]))/truth.shape[1]
                        for j in range(3)] for i in range(3)])
    zero=np.asarray([math.fsum(float(y)**2 for y in row)/len(row) for row in truth])
    mean_loss=np.asarray([math.fsum((float(p)-float(y))**2 for p,y in zip(history,row))/len(row) for row in truth])
    compare(report['beta'],beta,'frozen beta')
    require(report['new_model_fitting'] is False and report['new_hyperparameter_selection'] is False,'no new model declaration')
    require(report['task_order']==tasks,'report task order')
    compare(report['mse_query_by_donor'],matrix,'all9 MSE')
    for i,row in enumerate(report['rows']):
        wrong=[j for j in range(3) if j!=i]
        meanwrong=sum(matrix[i,j] for j in wrong)/2
        require(row['human_query']==tasks[i] and row['correct_donor']==tasks[i],'query identity')
        for j,entry in enumerate(row['all_donors']):
            require(entry['rat_donor']==tasks[j] and entry['correct_identity']==(i==j),'donor identity')
            compare(entry['mse'],matrix[i,j],'saved pair MSE')
        compare(row['correct_mse'],matrix[i,i],'correct score')
        compare(row['wrong_loss_average'],meanwrong,'mean of wrong losses')
        compare(row['wrong_average_minus_correct'],meanwrong-matrix[i,i],'correct advantage')
        compare(row['zero_mse'],zero[i],'zero baseline')
        compare(row['frozen_human_history_mse'],mean_loss[i],'history baseline')
        require(row['correct_mse_lower_than_both_wrong']==bool(matrix[i,i]<min(matrix[i,j] for j in wrong)),'individual wrong comparison flag')
        require(row['correct_mse_lower_than_wrong_average']==bool(matrix[i,i]<meanwrong),'average wrong comparison flag')
        compare(original['external_tasks'][i]['rules']['scalar_nointercept']['mse'],matrix[i,i],'original diagonal score unchanged')
    compare(report['macro_correct_mse'],np.trace(matrix)/3,'macro correct')
    compare(report['macro_wrong_loss_average'],sum(matrix[i,j] for i in range(3) for j in range(3) if i!=j)/6,'macro wrong')
    with np.load(out/'fixed_scalar_correspondence.npz',allow_pickle=False) as saved:
        require(saved['task_ids'].tolist()==tasks,'array task identity')
        require(np.array_equal(saved['human_gene_ids'],human_ids) and np.array_equal(saved['rat_gene_ids'],rat_ids),'array gene identity')
        compare(saved['beta'],beta,'array beta')
        compare(saved['fixed_scalar_predictions_by_donor'],predictions,'predictions use unchanged beta')
        compare(saved['mse_query_by_donor'],matrix,'array all9 MSE')
        compare(saved['zero_mse'],zero,'array zero')
        compare(saved['human_history_mse'],mean_loss,'array history')
    result={'status':'pass','checks_passed':checks,'checks_failed':0,'numeric_values_compared':numeric,
            'max_abs_difference':maxdiff,'fixed_beta':beta,'fixed_genes':len(human_ids),
            'correct_lower_than_wrong_average_tasks':int(sum(matrix[i,i]<np.mean([matrix[i,j] for j in range(3) if j!=i]) for i in range(3))),
            'correct_lower_than_each_wrong_tasks':int(sum(matrix[i,i]<min(matrix[i,j] for j in range(3) if j!=i) for i in range(3))),
            'original_run_hashes_unchanged':True,'no_model_fitting_performed':True,
            'method':'Read fixed original beta and external source/truth; recompute9MSE with independent math.fsum; separately score both wrong donors before averaging; check zero/history, original diagonal, all arrays and hashes.',
            'report_sha256':sha(out/'report.json'),'auditor_sha256':sha(__file__),
            'limitations':['Posthoc three-task cross-condition mismatch; it changes dose/time with identity and is not a pure identity-specificity or prospective validation.',
                           'KI696 correct donor is not the lowest-error donor; no donor or coefficient was selected from these scores.']}
    target=ROOT/'verification/posthoc_scalar_independent_verification.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
