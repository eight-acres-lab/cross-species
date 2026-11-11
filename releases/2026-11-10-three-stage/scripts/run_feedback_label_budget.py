#!/usr/bin/env python3
"""Fixed compound-label budgets for chemical priors and paired rat measurements.

This is a retrospective within-model-family learning curve. All budgets and
subsampling seeds are recorded before fitting. Complete-budget predictions are
reused from the frozen seven-point nested benchmark exactly once.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __name__ == "__main__":
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--threads", type=int, default=2)
    _known, _ = _pre.parse_known_args()
    for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[_key] = str(_known.threads)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_feedback_nested_benchmark as nested
from cross_species.feedback_data import load_feedback_data
from cross_species.feedback_evaluation import compound_macro_loss

ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
BUDGETS = (16, 32, 64)
SUBSAMPLE_SEEDS = (17, 29, 43)
METHODS = ("chemical_context", "source_plus_chemical_context")


def training_compound_order(groups, subsample_seed, outer_fold, split_seed):
    """Identifiers and fixed seeds determine nested budget prefixes."""
    identifiers = np.asarray(sorted(set(np.asarray(groups).astype(str).tolist())))
    rng = np.random.default_rng(np.random.SeedSequence([subsample_seed, outer_fold, split_seed]))
    return identifiers[rng.permutation(len(identifiers))]


def budget_rows(groups, outer_train, budget, subsample_seed, outer_fold, split_seed):
    order = training_compound_order(np.asarray(groups)[outer_train], subsample_seed, outer_fold, split_seed)
    if not isinstance(budget, (int, np.integer)) or budget < 3 or budget > len(order):
        raise ValueError("Each finite budget must lie between 3 and outer-training compound count")
    selected = order[:budget]
    rows = np.asarray(outer_train)[np.isin(np.asarray(groups)[outer_train], selected)]
    return rows, selected


def chemical_rows(chemical, rows):
    return {name: chemical[name][rows] for name in ["fingerprints", "descriptors"]}


def prepare_cohort(data, chemical, molecular_provenance, absolute_concentrations=False):
    exposure = None
    provenance = dict(molecular_provenance)
    if absolute_concentrations:
        data, exposure = nested.absolute_concentration_data(data)
        chemical = chemical_rows(chemical, exposure["original_row_indices"])
        provenance["actual_model_compounds_after_exposure_filter"] = sorted(set(data.compounds.tolist()))
    return data, chemical, provenance, exposure


def fit_budget_fold(source, context, human, groups, chemical, train, test, *, seed, alphas=ALPHAS):
    """No outer human labels enter subset fitting, transforms, or alpha selection."""
    selected, trials, inner = nested.select_alphas(source[train], context[train], human[train],
        groups[train], alphas, 3, seed, chemical=chemical_rows(chemical, train))
    inputs = nested.ChemicalKernelInputs(source[train], context[train], **chemical_rows(chemical, train))
    kt = inputs.train_kernels()
    kv = inputs.cross(source[test], context[test], **chemical_rows(chemical, test))
    predictions, state = {}, inputs.saved_transforms()
    for name in METHODS:
        ridge = nested.SpectralRidge(kt[name], human[train])
        predictions[name] = ridge.predict(kv[name], selected[name])
        state[name + "_dual_coef"] = ridge.coefficients(selected[name])
    state["target_mean"] = human[train].mean(axis=0)
    state["inner_fold"] = inner
    return predictions, {"selected_alphas": selected, "inner_trials": trials}, state


def aggregate_seed_metrics(seed_metrics, *, n_boot, seed):
    """Average seed losses, never predictions, then compounds equally."""
    if not seed_metrics:
        raise ValueError("At least one fitted seed is required")
    group_ids = seed_metrics[0][METHODS[0]]["group_ids"]
    for metrics in seed_metrics:
        for name in METHODS:
            if metrics[name]["group_ids"] != group_ids:
                raise ValueError("All seed/method compound axes must match")
    grouped = {name: np.stack([metrics[name]["group_losses"] for metrics in seed_metrics]) for name in METHODS}
    means = {name: values.mean(axis=0) for name, values in grouped.items()}
    deltas = means[METHODS[0]] - means[METHODS[1]]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    interval = np.percentile(deltas[samples].mean(axis=1), [2.5, 97.5])
    seed_macro = {name: values.mean(axis=1) for name, values in grouped.items()}
    seed_gains = seed_macro[METHODS[0]] - seed_macro[METHODS[1]]
    return {"fitted_seed_count": len(seed_metrics), "group_ids": group_ids,
            "mean_seed_group_losses": means, "group_gains": deltas,
            "mean_seed_macro_mse": {name: float(values.mean()) for name, values in seed_macro.items()},
            "seed_macro_mse": seed_macro, "seed_macro_gains": seed_gains,
            "seed_macro_mse_sample_sd": {name: float(values.std(ddof=1)) if len(values) > 1 else None
                                           for name, values in seed_macro.items()},
            "seed_gain_sample_sd": float(seed_gains.std(ddof=1)) if len(seed_gains) > 1 else None,
            "gain_from_source": float(deltas.mean()), "ci95": interval.tolist(),
            "n_compounds": len(deltas), "n_boot": n_boot, "bootstrap_seed": seed,
            "inference_scope": "Descriptive compound bootstrap of mean seed-specific losses from fixed OOF predictions. Seeds and fold-trained models share data; no refitting or population-coverage guarantee. Three subsamples do not form independent biological replications."}


def run_benchmark(data, chemical, molecular_provenance, out_dir, *, full_run_dir,
                  seed=20261107, budgets=BUDGETS, subsample_seeds=SUBSAMPLE_SEEDS,
                  alphas=ALPHAS, n_boot=5000, absolute_concentrations=False):
    out_dir, full_run_dir = Path(out_dir), Path(full_run_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    data, chemical, molecular_provenance, exposure = prepare_cohort(
        data, chemical, molecular_provenance, absolute_concentrations)
    source, human = np.asarray(data.source_delta, dtype=float), np.asarray(data.target_delta, dtype=float)
    columns, context = nested.context_columns(data), nested.model_context(data)
    groups = np.asarray(data.compounds).astype(str)
    folds = nested.group_folds(groups, 5, seed)
    original = json.loads((full_run_dir / "report.json").read_text())
    if (original["input_hashes"] != data.input_hashes
            or original["molecular_features"]["sha256"] != molecular_provenance["sha256"]
            or original["alpha_grid"] != list(alphas)
            or original["context_inputs"] != data.context_feature_ids[columns].tolist()):
        raise ValueError("Full-budget source inputs or regularization grid differ")
    with np.load(full_run_dir / "oof_predictions.npz", allow_pickle=False) as payload:
        if not all(np.array_equal(payload[name], value) for name, value in
                   [("sample_ids", data.sample_ids), ("compounds", groups), ("outer_fold", folds),
                    ("human_gene_ids", data.feature_ids), ("truth", human.astype(np.float32))]):
            raise ValueError("Full-budget cohort, folds, or gene axis differ")
        full_predictions = {name: payload[name].copy() for name in METHODS}
    # Prepare every subset before fitting or evaluating any budget.
    plans = {}
    for budget in budgets:
        plans[str(budget)] = {}
        for subsample_seed in subsample_seeds:
            plans[str(budget)][str(subsample_seed)] = []
            for fold in range(5):
                outer_train = np.flatnonzero(folds != fold)
                rows, compounds = budget_rows(groups, outer_train, budget, subsample_seed, fold, seed)
                plans[str(budget)][str(subsample_seed)].append({
                    "outer_fold": fold, "training_compounds": compounds.tolist(),
                    "training_compound_count": len(compounds), "training_condition_count": len(rows),
                    "training_rows": rows})
    all_conditions = [int(np.sum(folds != fold)) for fold in range(5)]
    all_compounds = [len(set(groups[folds != fold].tolist())) for fold in range(5)]
    code_files = [Path(__file__), Path(nested.__file__), ROOT / "src/cross_species/feedback_data.py",
                  ROOT / "src/cross_species/feedback_evaluation.py"]
    specification = {
        "scope": "Retrospective human-label-budget learning curve with nested compound cross-validation.",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": len(source), "compounds": len(set(groups.tolist())), "target_genes": human.shape[1],
        "outer_folds": 5, "inner_folds": 3, "split_seed": seed,
        "finite_budgets": list(budgets), "subsample_seeds": list(subsample_seeds), "alpha_grid": list(alphas),
        "subset_rule": "Sort outer-training compound IDs; permute once with SeedSequence([subsample_seed, outer_fold, split_seed]); finite budgets take nested prefixes. Retain every condition of selected compounds.",
        "paired_budget": "Each arm receives exactly the same human training labels, compounds, conditions, gene axis, inner folds and alpha grid. Rat training labels use the same selected pairs; measured test rat responses are available only to the source arm.",
        "kernel": original["kernel"], "input_hashes": data.input_hashes,
        "molecular_features": molecular_provenance, "data_warnings": list(data.warnings),
        "context_inputs": data.context_feature_ids[columns].tolist(),
        "full_budget_reuse": {"run_dir": str(full_run_dir.resolve()), "fitted_seed_count": 1,
                               "report_sha256": nested.file_hash(full_run_dir / "report.json"),
                               "oof_sha256": nested.file_hash(full_run_dir / "oof_predictions.npz"),
                               "training_compounds_by_fold": all_compounds,
                               "training_conditions_by_fold": all_conditions,
                               "states": {str(fold): str((full_run_dir / ("fold_%d_transforms_models.npz" % fold)).resolve())
                                          for fold in range(5)}},
        "all_subset_plans": plans,
        "aggregation": "Score each seed's OOF predictions separately; average the three seed-specific losses for each compound, then compounds equally. Do not ensemble predictions before loss. All-budget reuse is counted once.",
        "reporting": "All planned budgets and seeds are reported. Outer outcomes select no budget, seed, or fallback. Seed variation describes training-composition sensitivity.",
        "limitations": ["Budgets count paired training compounds and measured condition rows.",
                        "Three subsamples overlap, and models share outer folds; they are not independent replication.",
                        "A smaller-budget source model matching a larger-budget prior at a point estimate would not establish economic savings or statistical equivalence.",
                        "Evaluation uses joint RMA normalization and condition means from one animal species and one liver system."],
        "code_hashes": {str(path.relative_to(ROOT)): nested.file_hash(path) for path in code_files},
    }
    if exposure is not None:
        specification["absolute_concentrations"] = exposure
    out_dir.mkdir(parents=True, exist_ok=True)
    nested.save_json(out_dir / "specification.json", specification)
    (out_dir / "code_snapshot").mkdir()
    for path in code_files:
        shutil.copy2(path, out_dir / "code_snapshot" / path.name)
    with (out_dir / "training_subsets.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["budget", "subsample_seed", "outer_fold", "compound"])
        for budget in budgets:
            for subsample_seed in subsample_seeds:
                for plan in plans[str(budget)][str(subsample_seed)]:
                    writer.writerows([budget, subsample_seed, plan["outer_fold"], compound]
                                     for compound in plan["training_compounds"])
    started, results = time.monotonic(), {}
    for budget in budgets:
        per_seed = []
        budget_dir = out_dir / ("budget_%d" % budget)
        budget_dir.mkdir()
        for subsample_seed in subsample_seeds:
            trial_dir = budget_dir / ("seed_%d" % subsample_seed)
            trial_dir.mkdir()
            predictions = {name: np.empty_like(human, dtype=np.float32) for name in METHODS}
            records = []
            for plan in plans[str(budget)][str(subsample_seed)]:
                fold, train = plan["outer_fold"], np.asarray(plan["training_rows"])
                test = np.flatnonzero(folds == fold)
                print("budget %d seed %d outer fold %d/5: %d labeled conditions" %
                      (budget, subsample_seed, fold + 1, len(train)), flush=True)
                predicted, record, state = fit_budget_fold(source, context, human, groups, chemical,
                                                           train, test, seed=seed + 1009 * (fold + 1), alphas=alphas)
                for name, value in predicted.items():
                    predictions[name][test] = value
                np.savez_compressed(trial_dir / ("fold_%d_state.npz" % fold), **state,
                                    train_sample_ids=data.sample_ids[train], test_sample_ids=data.sample_ids[test],
                                    human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids,
                                    context_feature_ids=data.context_feature_ids[columns])
                record.update(plan)
                records.append(record)
            np.savez_compressed(trial_dir / "oof_predictions.npz", sample_ids=data.sample_ids,
                                compounds=groups, outer_fold=folds, human_gene_ids=data.feature_ids,
                                truth=human.astype(np.float32), **predictions)
            # Per-seed scores are retained, but no later fit reads them.
            metrics = {name: compound_macro_loss(value, human, groups) for name, value in predictions.items()}
            seed_report = {"budget_compounds": budget, "subsample_seed": subsample_seed,
                           "fold_records": records, "metrics": metrics}
            nested.save_json(trial_dir / "report.json", seed_report)
            per_seed.append(seed_report)
        aggregate = aggregate_seed_metrics([value["metrics"] for value in per_seed], n_boot=n_boot, seed=seed)
        results[str(budget)] = {"budget_compounds": budget, "subsample_seeds": list(subsample_seeds),
                                "per_seed": per_seed, "aggregate": aggregate}
        nested.save_json(budget_dir / "report.json", results[str(budget)])
    # Read the all-budget outputs once; never retrain or duplicate them across seeds.
    full_metrics = {name: compound_macro_loss(value, human, groups) for name, value in full_predictions.items()}
    full_aggregate = aggregate_seed_metrics([full_metrics], n_boot=n_boot, seed=seed)
    results["all"] = {"budget_compounds_by_outer_fold": all_compounds,
                      "training_conditions_by_outer_fold": all_conditions,
                      "reused": True, "subsample_seeds": [], "metrics": full_metrics,
                      "aggregate": full_aggregate}
    report = {**specification, "results": results, "elapsed_seconds": time.monotonic() - started}
    nested.save_json(out_dir / "report.json", report)
    with (out_dir / "budget_summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["human_training_compounds", "fitted_subsamples", "chemical_prior_mse", "source_plus_prior_mse",
                         "gain_from_source", "gain_ci_lower", "gain_ci_upper", "gain_subsample_sd"])
        for budget, value in results.items():
            summary = value["aggregate"]
            writer.writerow([budget, summary["fitted_seed_count"],
                             *[summary["mean_seed_macro_mse"][name] for name in METHODS],
                             summary["gain_from_source"], *summary["ci95"], summary["seed_gain_sample_sd"]])
    with (out_dir / "compound_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["budget", "compound", "mean_seed_prior_mse", "mean_seed_source_mse", "gain"])
        for budget, value in results.items():
            summary = value["aggregate"]
            for i, compound in enumerate(summary["group_ids"]):
                writer.writerow([budget, compound, *[summary["mean_seed_group_losses"][name][i] for name in METHODS],
                                 summary["group_gains"][i]])
    print(json.dumps(nested.plain({budget: value["aggregate"] for budget, value in results.items()}), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/tg_gates_rat_human_v1/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feedback_cache"))
    parser.add_argument("--compound-features", type=Path, default=Path("artifacts/feedback_compound_features_20261107/features.npz"))
    parser.add_argument("--full-run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/feedback_label_budget_20261107"))
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--absolute-concentrations", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.n_boot < 1:
        parser.error("threads and n-boot must be positive")
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=args.seed)
    data, chemical, provenance = nested.align_compound_features(data, args.compound_features)
    reference = args.full_run_dir or Path("outputs/feedback_nested_chemical_20261107_absolute_v1" if args.absolute_concentrations
                                         else "outputs/feedback_nested_chemical_20261107_full_v2")
    run_benchmark(data, chemical, provenance, args.out_dir, full_run_dir=reference,
                  seed=args.seed, n_boot=args.n_boot, absolute_concentrations=args.absolute_concentrations)


if __name__ == "__main__":
    main()
