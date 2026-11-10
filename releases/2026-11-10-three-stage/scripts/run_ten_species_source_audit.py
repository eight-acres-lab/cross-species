#!/usr/bin/env python3
"""Descriptive multisource audit of one purified-stimulus fibroblast experiment.

All source subsets and adding-source edges are retained. No source-selection
learner or generalization confidence interval can be identified from one stimulus.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_species.multisource_data import load_curated_panel, file_hash


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def equal_subset_weights(n_sources):
    subsets = [subset for size in range(1, n_sources + 1)
               for subset in combinations(range(n_sources), size)]
    weights = np.zeros((len(subsets), n_sources))
    for i, subset in enumerate(subsets): weights[i, list(subset)] = 1 / len(subset)
    return subsets, weights


def mse_from_gram(sources, human, weights):
    gram = sources @ sources.T / sources.shape[1]
    cross = sources @ human / sources.shape[1]
    return np.einsum('mi,ij,mj->m', weights, gram, weights) - 2 * weights @ cross + np.mean(human**2)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-12 else None


def donor_draws(n_draws, n_sources, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n_draws, n_sources + 1, 3), dtype=np.int64)


def resampled_losses(human_donors, source_donors, weights, draws, device='cpu', chunk_size=64):
    """Joint target/source donor resampling, computed via per-draw gene Gram matrices."""
    if human_donors.shape[0] != 3 or source_donors.shape[1] != 3:
        raise ValueError('This frozen audit requires three donors per species')
    donors = np.concatenate([human_donors[None], source_donors], axis=0)
    if not np.isfinite(donors).all(): raise ValueError('Complete axis contains missing values')
    d = torch.as_tensor(donors, dtype=torch.float64, device=device)
    w = torch.as_tensor(weights, dtype=torch.float64, device=device)
    species = torch.arange(len(donors), device=device)[None, :, None]
    result = []
    for start in range(0, len(draws), chunk_size):
        index = torch.as_tensor(draws[start:start+chunk_size], dtype=torch.long, device=device)
        sample = d[species, index].mean(dim=2)
        target, source = sample[:, 0], sample[:, 1:]
        gram = source @ source.transpose(1, 2) / source.shape[-1]
        cross = (source * target[:, None]).mean(dim=2)
        loss = torch.einsum('mi,bij,mj->bm', w, gram, w) - 2 * cross @ w.T + target.square().mean(dim=1)[:, None]
        result.append(loss.cpu().numpy())
    return np.concatenate(result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261108)
    args = ap.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()): ap.error('Use a fresh output directory')
    if args.draws < 2: ap.error('Need at least two sensitivity draws')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_curated_panel(args.panel)
    with np.load(args.panel/'donor_responses.npz', allow_pickle=False) as data:
        genes, mask = data['gene_ids'], data['global_eligible_mask']
        response, species, donors = data['response'][:, mask], data['species'], data['donors']
    if not np.array_equal(genes, panel.feature_ids) or not np.array_equal(mask, panel.global_eligible_mask):
        raise ValueError('Donor and arm feature axes differ')
    sources = [s for s in panel.species_order.tolist() if s != 'human']
    human = response[species == 'human']
    source_donors = np.stack([response[species == s] for s in sources])
    if human.shape[0] != 3 or source_donors.shape[1] != 3: raise ValueError('Three donors per species required')
    subsets, weights = equal_subset_weights(len(sources))
    draws = donor_draws(args.draws, len(sources), args.seed)
    spec = {'study': 'GSE105160', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'One synthetic polyIC 24h stimulus, nine animal sources, three donors per species; descriptive response transfer and measured-donor sensitivity only.',
            'target': 'Human in vitro primary dermal fibroblast treatment-minus-own-donor-mock response.',
            'sources': sources, 'subset_count': len(subsets), 'gene_count': int(mask.sum()),
            'axis_rule': 'Complete frozen ten-species one-to-one axis; identical across every subset and target donor.',
            'primary': 'MSE against human donor-mean response; individual-human-donor MSE also retained.',
            'budgets': 'Every acquired source uses all three measured donors and their culture replicates. Added sources increase measurement budget; this audit has no equal-total-budget causal contrast.',
            'premeasurement_rule': 'Select source with greatest Pearson control-state similarity using all source/human control measurements; controls themselves require measurement. No treated human values select this rule.',
            'no_learned_selector': 'One perturbation cannot train or independently validate a new-perturbation source-choice rule.',
            'oracle': 'All performance-ranked source/subset lists are post-outcome descriptive references, not deployable selection.',
            'donor_resampling': {'draws': args.draws, 'seed': args.seed, 'fixed_weights': True,
                                 'interpretation': 'Three observed donors per species resampled with replacement; sensitivity ranges, not generalization or species-population confidence intervals.'},
            'source_order_matches_taxonomy': 'Phylogenetic dependencies are not treated as independent species replicates.',
            'panel_manifest_sha256': file_hash(args.panel/'manifest.json'),
            'donor_responses_sha256': file_hash(args.panel/'donor_responses.npz'),
            'code_sha256': file_hash(__file__), 'device': args.device}
    save(args.out_dir/'specification.json', spec)
    mean_h, mean_s = human.mean(axis=0), source_donors.mean(axis=1)
    predictions = weights @ mean_s
    losses = mse_from_gram(mean_s, mean_h, weights)
    individual_losses = ((predictions[:, None, :] - human[None, :, :])**2).mean(axis=2)
    baseline_by_species = {}
    for s in panel.species_order:
        rows = [i for i, r in enumerate(panel.arm_metadata) if r['species'] == s and r['role'] == 'vehicle']
        by_donor = {}
        for i in rows: by_donor.setdefault(panel.arm_metadata[i]['biological_id'], []).append(i)
        baseline_by_species[str(s)] = np.stack([panel.arm_logcpm[ii][:, mask].mean(axis=0) for ii in by_donor.values()]).mean(axis=0)
    baseline_similarity = [pearson(baseline_by_species[s], baseline_by_species['human']) for s in sources]
    valid = [i for i, value in enumerate(baseline_similarity) if value is not None]
    selected = max(valid, key=lambda i: baseline_similarity[i]) if valid else None
    records = []
    for i, subset in enumerate(subsets):
        records.append({'sources': [sources[s] for s in subset], 'source_count': len(subset),
                        'mean_target_mse': float(losses[i]), 'individual_target_mse': individual_losses[i].tolist(),
                        'mean_individual_target_mse': float(individual_losses[i].mean()),
                        'pearson_mean_target': pearson(predictions[i], mean_h),
                        'response_rms': float(np.sqrt(np.mean(predictions[i]**2)))})
    edge_records = []
    index = {subset: i for i, subset in enumerate(subsets)}
    for subset in subsets:
        for added in range(len(sources)):
            if added in subset: continue
            expanded = tuple(sorted(subset + (added,)))
            before, after = index[subset], index[expanded]
            edge_records.append({'before_index': before, 'after_index': after, 'added_source': sources[added],
                                 'gain': float(losses[before] - losses[after])})
    cpu = resampled_losses(human, source_donors, weights, draws[:4], 'cpu', 4)
    start = time.monotonic()
    sensitivity = resampled_losses(human, source_donors, weights, draws, args.device)
    if args.device == 'cuda': torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    error = float(np.max(np.abs(cpu - sensitivity[:4])))
    if error > 1e-10: raise ValueError('CPU/GPU resampling disagreement')
    interval = np.quantile(sensitivity, [.025, .5, .975], axis=0)
    for i, row in enumerate(records): row['observed_donor_resampling_quantiles'] = interval[:, i].tolist()
    human_reference = (human.sum(axis=0)[None] - human) / 2
    report = {'specification_sha256': file_hash(args.out_dir/'specification.json'),
              'single_source_records': records[:len(sources)], 'all_subset_records': records,
              'human_response_rms': float(np.sqrt(np.mean(mean_h**2))),
              'zero_mean_target_mse': float(np.mean(mean_h**2)),
              'zero_individual_target_mse': float(np.mean(human**2)),
              'same_perturbation_human_LODO_reference_mse': float(np.mean((human_reference-human)**2)),
              'same_perturbation_reference_warning': 'Uses human outcomes for this perturbation in other donors; not equal information to animal-only prediction.',
              'random_source_expected_mean_mse': float(np.mean(losses[:len(sources)])),
              'all_sources_equal_mean_mse': float(losses[-1]),
              'control_state_selection': {'similarities': dict(zip(sources, baseline_similarity)), 'selected_source': None if selected is None else sources[selected],
                                          'selected_mean_target_mse': None if selected is None else float(losses[selected])},
              'post_outcome_oracle': {'best_source': sources[int(np.argmin(losses[:len(sources)]))],
                                      'best_subset_index': int(np.argmin(losses)), 'deployable': False},
              'added_source_edges': edge_records, 'negative_added_edges': sum(row['gain'] < 0 for row in edge_records),
              'device_verification': {'device': str(torch.cuda.get_device_name()) if args.device=='cuda' else 'cpu',
                                      'dtype': 'float64', 'same_draw_cpu_max_error': error, 'elapsed_seconds': elapsed,
                                      'peak_gpu_allocated_bytes': int(torch.cuda.max_memory_allocated()) if args.device=='cuda' else None},
              'inference_limits': ['One stimulus, no trained selection or held-perturbation validation.',
                                  'Species and mapping/assay effects remain coupled; native-versus-human mapping sensitivity requires separate evaluation.',
                                  'Squirrel-monkey sample/reference taxon discrepancy retained in source provenance.',
                                  'Negative edges refer to equal averaging with added source measurements; no intrinsic harm or optimal acquisition claim.']}
    save(args.out_dir/'report.json', report)
    np.savez_compressed(args.out_dir/'predictions_and_sensitivity.npz', human_donor_responses=human, source_donor_responses=source_donors,
                        source_ids=np.asarray(sources), gene_ids=genes[mask], subset_weights=weights, predictions=predictions,
                        donor_draws=draws, resampled_losses=sensitivity)
    print(json.dumps({'sources': len(sources), 'subsets': len(subsets), 'genes': int(mask.sum()),
                      'single_source_mse': {sources[i]: float(losses[i]) for i in range(len(sources))},
                      'zero_mse': report['zero_mean_target_mse'], 'all_sources_mse': float(losses[-1]),
                      'control_rule': report['control_state_selection'], 'gpu': report['device_verification']}, indent=2), flush=True)


if __name__ == '__main__': main()
