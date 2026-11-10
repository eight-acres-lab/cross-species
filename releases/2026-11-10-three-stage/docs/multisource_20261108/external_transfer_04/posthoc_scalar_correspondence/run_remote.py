"""Posthoc fixed-scalar correspondence diagnostic; no fitting or model selection."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

ROOT = Path('/hy-tmp/cross-species-feedback-20261108/external_transfer_04')
INPUTS = ROOT / 'inputs_v1'
ORIGINAL = ROOT / 'run_v1'
OUT = ROOT / 'posthoc_scalar_correspondence_v1'
PROTOCOL = ROOT / 'posthoc_scalar_correspondence_PROTOCOL_v1.zh.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as x:
        return {name: x[name].copy() for name in x.files}


def main():
    started = time.monotonic()
    if OUT.exists():
        raise ValueError('Use a new output version; original/posthoc outputs must not be overwritten')
    source_manifest = json.loads((INPUTS / 'manifest.json').read_text())
    fit_manifest = json.loads((ORIGINAL / 'fit_manifest.json').read_text())
    main_report = json.loads((ORIGINAL / 'report.json').read_text())
    before = {name: sha(ORIGINAL / name) for name in ['report.json', 'fit_manifest.json', 'final_fit.npz', 'external_predictions.npz']}
    if before['fit_manifest.json'] != main_report['fit_manifest_sha256']:
        raise ValueError('Frozen main-report fit reference changed')
    for name in ['final_fit.npz', 'external_predictions.npz']:
        if before[name] != fit_manifest['output_files'][name]:
            raise ValueError('Frozen original model or prediction changed')
    for name in ['train.npz', 'external_inputs.npz', 'external_truth.npz']:
        if sha(INPUTS / name) != source_manifest['files'][name]:
            raise ValueError('Input checksum differs: ' + name)
    train, source, target = read(INPUTS / 'train.npz'), read(INPUTS / 'external_inputs.npz'), read(INPUTS / 'external_truth.npz')
    state, original_predictions = read(ORIGINAL / 'final_fit.npz'), read(ORIGINAL / 'external_predictions.npz')
    x, y = np.asarray(source['source'], dtype=np.float64), np.asarray(target['target'], dtype=np.float64)
    tx, ty = np.asarray(train['source'], dtype=np.float64), np.asarray(train['target'], dtype=np.float64)
    if x.shape != y.shape or x.shape != (3, 7307):
        raise ValueError('Expected the frozen three-task7307-gene comparison')
    for other in [target['task_ids'], original_predictions['task_ids']]:
        np.testing.assert_array_equal(source['task_ids'], other)
    for name in ['human_gene_ids', 'rat_gene_ids']:
        np.testing.assert_array_equal(source[name], train[name])
        np.testing.assert_array_equal(source[name], state[name])
        np.testing.assert_array_equal(source[name], original_predictions[name])
    if not all(np.isfinite(a).all() for a in [x, y, tx, ty]):
        raise ValueError('Nonfinite input')
    beta = float(state['scalar_nointercept_beta'])
    numerator = math.fsum(float(a) * float(b) for a, b in zip(tx.flat, ty.flat))
    denominator = math.fsum(float(a) * float(a) for a in tx.flat)
    check_beta = max(0., numerator / denominator) if denominator > 1e-12 else 0.
    errors = [abs(beta - check_beta)]
    if errors[0] > 1e-10:
        raise ValueError('Frozen beta disagrees with independent training-only formula')
    per_source_predictions = beta * x
    np.testing.assert_array_equal(per_source_predictions, original_predictions['scalar_nointercept'])
    mse = np.mean((per_source_predictions[None, :, :] - y[:, None, :]) ** 2, axis=2)
    zero = np.mean(y ** 2, axis=1)
    history = np.mean((state['human_mean'][None] - y) ** 2, axis=1)
    for query in range(3):
        for donor in range(3):
            independent = math.fsum((beta * float(a) - float(b)) ** 2 for a, b in zip(x[donor], y[query])) / x.shape[1]
            errors.append(abs(independent - mse[query, donor]))
        independent_zero = math.fsum(float(value) ** 2 for value in y[query]) / x.shape[1]
        independent_history = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(state['human_mean'], y[query])) / x.shape[1]
        errors += [abs(independent_zero - zero[query]), abs(independent_history - history[query])]
        prior_row = main_report['external_tasks'][query]
        if prior_row['task_id'] != str(source['task_ids'][query]):
            raise ValueError('Main report task order differs')
        errors += [abs(mse[query, query] - prior_row['rules']['scalar_nointercept']['mse']),
                   abs(zero[query] - prior_row['rules']['zero']['mse']),
                   abs(history[query] - prior_row['rules']['human_mean']['mse'])]
    if max(errors) > 1e-10:
        raise ValueError('Independent score or frozen diagonal mismatch')
    task_meta = json.loads((INPUTS / 'task_metadata.json').read_text())
    rows = []
    for query, task in enumerate(source['task_ids']):
        wrong = [donor for donor in range(3) if donor != query]
        wrong_average = float(np.mean(mse[query, wrong]))
        rows.append({'human_query': str(task), 'correct_donor': str(task),
                     'all_donors': [{'rat_donor': str(source['task_ids'][donor]), 'correct_identity': donor == query,
                                    'mse': float(mse[query, donor])} for donor in range(3)],
                     'correct_mse': float(mse[query, query]), 'wrong_loss_average': wrong_average,
                     'correct_mse_lower_than_both_wrong': bool(np.all(mse[query, query] < mse[query, wrong])),
                     'correct_mse_lower_than_wrong_average': bool(mse[query, query] < wrong_average),
                     'wrong_average_minus_correct': wrong_average - float(mse[query, query]),
                     'zero_mse': float(zero[query]), 'frozen_human_history_mse': float(history[query])})
    OUT.mkdir()
    np.savez(OUT / 'fixed_scalar_correspondence.npz', task_ids=source['task_ids'],
             beta=np.asarray(beta), fixed_scalar_predictions_by_donor=per_source_predictions,
             mse_query_by_donor=mse, zero_mse=zero, human_history_mse=history,
             human_gene_ids=source['human_gene_ids'], rat_gene_ids=source['rat_gene_ids'])
    after = {name: sha(ORIGINAL / name) for name in before}
    if before != after:
        raise ValueError('Original frozen run changed during diagnostic')
    report = {'scope': 'Posthoc descriptive fixed-scalar drug-correspondence diagnostic after inspection of the external main report.',
              'new_model_fitting': False, 'new_hyperparameter_selection': False, 'original_run_unchanged': True,
              'beta': beta, 'independent_training_formula_beta': check_beta, 'fixed_genes': x.shape[1],
              'task_order': source['task_ids'].tolist(), 'rows': rows, 'mse_query_by_donor': mse.tolist(),
              'macro_correct_mse': float(np.trace(mse) / 3),
              'macro_wrong_loss_average': float(np.mean([row['wrong_loss_average'] for row in rows])),
              'task_metadata_as_saved': task_meta,
              'aggregation': 'Score each wrong donor separately for the same human query, then average its two losses. No averaging of wrong prediction vectors.',
              'limitations': [
                  'Wrong-source changes drug, concentration and sometimes time; cannot isolate identity effects at fixed exposure.',
                  'Only three related stress tasks; two NRF2 activators share pathway background. No task-population p-value or confidence interval.',
                  'This posthoc diagnostic cannot select a better donor or beta, replace the original six-rule result, or establish a new deployment gain.',
                  'TG-RMA and external logCPM remain different processed measurement versions.',
              ],
              'validation': {'independent_formula_and_frozen_score_checks': len(errors), 'max_abs_difference': max(errors),
                             'tolerance': 1e-10, 'passed': True},
              'provenance': {'protocol_sha256': sha(PROTOCOL), 'script_sha256': sha(__file__),
                             'input_manifest_sha256': sha(INPUTS / 'manifest.json'), 'original_run_files_sha256': before,
                             'train_npz_sha256': sha(INPUTS / 'train.npz'), 'external_inputs_sha256': sha(INPUTS / 'external_inputs.npz'),
                             'external_truth_sha256': sha(INPUTS / 'external_truth.npz'),
                             'output_npz_sha256': sha(OUT / 'fixed_scalar_correspondence.npz')},
              'runtime': {'device': 'remote_CPU', 'wall_seconds': time.monotonic() - started}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'beta': beta, 'rows': rows, 'validation': report['validation']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
