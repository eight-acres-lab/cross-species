#!/usr/bin/env python3
"""Frozen retrospective tests of source information beyond training common direction.

The two common features control projection amplitude and total response magnitude.
They do not constitute a complete shared-biology or causal decomposition.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--threads", type=int, default=2)
    _preargs, _ = _parser.parse_known_args()
    for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[_key] = str(_preargs.threads)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_feedback_nested_benchmark import (
    CONTEXT_COLUMNS, DEFAULT_ALPHAS, SpectralRidge, file_hash, group_folds, save_json,
)
from cross_species.feedback_data import load_feedback_data
from cross_species.feedback_evaluation import compound_macro_loss, paired_compound_bootstrap

MODEL_BLOCKS = {
    "context_only": ("context",),
    "context_plus_common": ("context", "common"),
    "context_plus_common_residual": ("context", "common", "residual"),
    "full_source_plus_context": ("context", "source"),
}
REFERENCES = ("zero_response", "training_human_context_mean")
TIE_ATOL = 1e-10


class TrainingMeans:
    """Row-weighted stratum means with an explicit all-training mean fallback."""

    def __init__(self, values, strata):
        values, strata = np.asarray(values, dtype=float), np.asarray(strata).astype(str)
        if values.ndim != 2 or len(values) != len(strata) or not len(values) or not np.isfinite(values).all():
            raise ValueError("Finite training matrix and one stratum per row required")
        self.global_mean = values.mean(axis=0)
        self.keys = np.asarray(sorted(set(strata.tolist())))
        self.means = np.stack([values[strata == key].mean(axis=0) for key in self.keys])
        self.counts = np.asarray([np.sum(strata == key) for key in self.keys])
        self.lookup = {str(key): i for i, key in enumerate(self.keys)}

    def transform(self, strata):
        strata = np.asarray(strata).astype(str)
        fallback = np.asarray([key not in self.lookup for key in strata])
        means = np.stack([self.means[self.lookup[key]] if key in self.lookup else self.global_mean for key in strata])
        return means, fallback

    def state(self, prefix):
        return {prefix + "_strata": self.keys, prefix + "_means": self.means,
                prefix + "_counts": self.counts, prefix + "_global_mean": self.global_mean}


class TrainingCommon(TrainingMeans):
    """Projection amplitude and total RMS; residuals use unstandardized deltas."""

    def transform(self, source, strata):
        source = np.asarray(source, dtype=float)
        directions, fallback = super().transform(strata)
        if source.shape != directions.shape or not np.isfinite(source).all():
            raise ValueError("Source shape/finite values must match training gene axis")
        squared_norm = np.square(directions).sum(axis=1)
        valid = squared_norm > 1e-24
        amplitude = np.zeros(len(source))
        amplitude[valid] = (source[valid] * directions[valid]).sum(axis=1) / squared_norm[valid]
        common = np.column_stack([amplitude, np.sqrt(np.square(source).mean(axis=1))])
        residual = source - amplitude[:, None] * directions
        return common, residual, {"fallback": fallback, "zero_direction": ~valid}


class BlockKernels:
    """Each block has its own train-only z-scores and feature-count normalization."""

    def __init__(self, blocks):
        self.mean, self.scale, self.train = {}, {}, {}
        for key, value in blocks.items():
            value = np.asarray(value, dtype=float)
            if value.ndim != 2 or not value.shape[1] or not np.isfinite(value).all():
                raise ValueError("Finite nonempty feature blocks required")
            self.mean[key], self.scale[key] = value.mean(axis=0), value.std(axis=0)
            self.scale[key][self.scale[key] < 1e-8] = 1.
            self.train[key] = (value - self.mean[key]) / self.scale[key]

    def cross(self, blocks):
        kernels = {}
        for key in self.train:
            z = (np.asarray(blocks[key], dtype=float) - self.mean[key]) / self.scale[key]
            kernels[key] = z @ self.train[key].T / z.shape[1]
        return {name: sum(kernels[key] for key in keys) / len(keys) for name, keys in MODEL_BLOCKS.items()}

    def state(self):
        return {f"{key}_{kind}": value for kind, store in [("mean", self.mean), ("scale", self.scale)] for key, value in store.items()}


def feature_blocks(common_fit, source, context, strata):
    common, residual, diagnostics = common_fit.transform(source, strata)
    return {"context": context, "common": common, "residual": residual, "source": source}, diagnostics


def select_alphas(source, context, target, groups, strata, alphas, inner_folds, seed):
    folds = group_folds(groups, inner_folds, seed)
    losses = {name: np.empty((len(alphas), len(source))) for name in MODEL_BLOCKS}
    for fold in range(inner_folds):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        common = TrainingCommon(source[train], strata[train])
        train_blocks, _ = feature_blocks(common, source[train], context[train], strata[train])
        valid_blocks, _ = feature_blocks(common, source[valid], context[valid], strata[valid])
        kernels = BlockKernels(train_blocks)
        ktrain, kvalid = kernels.cross(train_blocks), kernels.cross(valid_blocks)
        for name in MODEL_BLOCKS:
            ridge = SpectralRidge(ktrain[name], target[train])
            losses[name][:, valid] = ridge.row_losses(kvalid[name], target[valid], alphas)
    selected, trials = {}, {}
    unique = sorted(set(groups.tolist()))
    for name in MODEL_BLOCKS:
        macro = [np.mean([row[groups == group].mean() for group in unique]) for row in losses[name]]
        selected[name] = float(alphas[int(np.argmin(macro))])
        trials[name] = [{"alpha": float(a), "compound_macro_mse": float(m)} for a, m in zip(alphas, macro)]
    return selected, trials, folds


def fit_outer_fold(source, context, target, groups, strata, train, test,
                   alphas=DEFAULT_ALPHAS, inner_folds=3, seed=0):
    selected, trials, inner = select_alphas(source[train], context[train], target[train],
                                           groups[train], strata[train], alphas, inner_folds, seed)
    common = TrainingCommon(source[train], strata[train])
    train_blocks, train_diagnostics = feature_blocks(common, source[train], context[train], strata[train])
    test_blocks, test_diagnostics = feature_blocks(common, source[test], context[test], strata[test])
    kernels = BlockKernels(train_blocks)
    ktrain, ktest = kernels.cross(train_blocks), kernels.cross(test_blocks)
    predictions, states = {}, {}
    for name in MODEL_BLOCKS:
        ridge = SpectralRidge(ktrain[name], target[train])
        predictions[name] = ridge.predict(ktest[name], selected[name])
        states[name + "_dual_coef"] = ridge.coefficients(selected[name])
    human_fit = TrainingMeans(target[train], strata[train])
    human_context, human_fallback = human_fit.transform(strata[test])
    predictions["zero_response"] = np.zeros_like(human_context)
    predictions["training_human_context_mean"] = human_context
    state = {**common.state("source_common"), **human_fit.state("human_context"), **kernels.state(), **states,
             "train_indices": train, "test_indices": test, "inner_fold": inner,
             "target_mean": target[train].mean(axis=0),
             "source_common_fallback_test": test_diagnostics["fallback"],
             "source_common_zero_direction_test": test_diagnostics["zero_direction"],
             "human_context_fallback_test": human_fallback}
    diagnostics = {"source_fallback_test": int(test_diagnostics["fallback"].sum()),
                   "source_zero_direction_test": int(test_diagnostics["zero_direction"].sum()),
                   "source_zero_direction_train": int(train_diagnostics["zero_direction"].sum()),
                   "human_context_fallback_test": int(human_fallback.sum())}
    return predictions, {"selected_alphas": selected, "inner_trials": trials, "diagnostics": diagnostics}, state


def tied_ranking(scores, correct, atol=TIE_ATOL):
    """Expected metrics under uniform random ordering inside the true score tie."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 <= correct < len(scores):
        raise ValueError("Finite scores and in-range true candidate required")
    true = scores[correct]
    greater = int(np.sum(scores > true + atol))
    tied = int(np.sum(np.abs(scores - true) <= atol))
    positions = np.arange(greater + 1, greater + tied + 1, dtype=float)
    return {"top1": 1. / tied if greater == 0 else 0., "rank": float(positions.mean()),
            "mrr": float((1. / positions).mean()), "true_tie_size": tied}


def identification(predictions, truth, human_context_mean, groups, strata, folds):
    """Only frozen outer predictions enter this terminal all-gene retrieval scorer."""
    groups, strata, folds = np.asarray(groups).astype(str), np.asarray(strata).astype(str), np.asarray(folds)
    truth = np.asarray(truth, dtype=float)
    human_context_mean = np.asarray(human_context_mean, dtype=float)
    if truth.shape != human_context_mean.shape or not np.isfinite(truth).all():
        raise ValueError("Finite equal-shaped truth and training-context mean required")
    seen = set()
    for group, stratum in zip(groups, strata):
        if (group, stratum) in seen:
            raise ValueError("Identification requires one candidate per compound/time/dose; duplicate compound-stratum")
        seen.add((group, stratum))
    row_metrics = {name: [] for name in predictions}
    for i in range(len(truth)):
        pool = np.flatnonzero((folds == folds[i]) & (strata == strata[i]))
        k = len(pool)
        chance = {"top1": 1. / k, "rank": (k + 1) / 2., "mrr": float((1. / np.arange(1, k + 1)).mean())}
        candidate = truth[pool] - human_context_mean[pool]
        candidate -= candidate.mean(axis=1, keepdims=True)
        cnorm = np.sqrt(np.square(candidate).sum(axis=1))
        for name, prediction in predictions.items():
            row = {"row": i, "compound": groups[i], "fold": int(folds[i]), "stratum": strata[i],
                   "candidates": k, "chance": chance, "defined": False, "reason": None}
            query = np.asarray(prediction[i], dtype=float) - human_context_mean[i]
            query -= query.mean()
            qnorm = float(np.sqrt(np.square(query).sum()))
            if k < 2:
                row["reason"] = "fewer_than_two_candidates"
            elif not np.isfinite(cnorm).all() or np.any(cnorm <= 1e-12):
                row["reason"] = "undefined_candidate_pearson"
            elif not np.isfinite(query).all() or qnorm <= 1e-12:
                row["reason"] = "undefined_query_pearson"
            else:
                scores = np.clip(candidate @ query / (cnorm * qnorm), -1., 1.)
                correct = int(np.flatnonzero(pool == i)[0])
                row.update(tied_ranking(scores, correct))
                row["defined"] = True
            row_metrics[name].append(row)
    common = np.asarray([all(row_metrics[name][i]["defined"] for name in MODEL_BLOCKS) for i in range(len(truth))])
    summary = {}
    for name, rows in row_metrics.items():
        natural = np.asarray([row["defined"] for row in rows])
        summary[name] = {"all_defined_queries": summarize_identification(rows, natural),
                         "common_four_model_queries": summarize_identification(rows, common & natural),
                         "undefined_reasons": {reason: sum(row["reason"] == reason for row in rows)
                                               for reason in sorted({row["reason"] for row in rows if row["reason"]})}}
    return summary, row_metrics, common


def summarize_identification(rows, mask):
    groups = sorted({row["compound"] for row in rows})
    selected = [row for row, keep in zip(rows, mask) if keep]
    by_compound = []
    for group in groups:
        subset = [row for row in selected if row["compound"] == group]
        if subset:
            record = {"compound": group, "queries": len(subset)}
            for metric in ["top1", "rank", "mrr"]:
                record[metric] = float(np.mean([row[metric] for row in subset]))
                record["chance_" + metric] = float(np.mean([row["chance"][metric] for row in subset]))
            by_compound.append(record)
    result = {"defined_queries": len(selected), "total_queries": len(rows),
              "defined_compounds": len(by_compound), "total_compounds": len(groups),
              "candidate_count_range": [min(row["candidates"] for row in selected), max(row["candidates"] for row in selected)] if selected else None,
              "compound_metrics": by_compound}
    for metric in ["top1", "rank", "mrr"]:
        result[metric] = float(np.mean([row[metric] for row in by_compound])) if by_compound else None
        result["chance_" + metric] = float(np.mean([row["chance_" + metric] for row in by_compound])) if by_compound else None
    return result


def run_benchmark(data, out_dir, *, outer_folds=5, inner_folds=3, alphas=DEFAULT_ALPHAS, seed=20261107, n_boot=5000):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty output directory; frozen runs cannot be overwritten")
    source, target = np.asarray(data.source_delta, dtype=float), np.asarray(data.target_delta, dtype=float)
    context = np.asarray(data.context[:, CONTEXT_COLUMNS], dtype=float)
    groups = np.asarray(data.compounds).astype(str)
    strata = np.asarray([format(float(row["time_h"]), ".12g") + "|" + row["dose_level"].casefold() for row in data.metadata])
    if source.shape != target.shape or source.ndim != 2 or len(context) != len(source) or not all(np.isfinite(x).all() for x in [source, target, context]):
        raise ValueError("Finite aligned full-gene source/target/context required")
    if len(set(zip(groups.tolist(), strata.tolist()))) != len(source):
        raise ValueError("Duplicate compound-stratum condition; do not create duplicate retrieval candidates")
    if not len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Fixed positive alpha grid required")
    folds = group_folds(groups, outer_folds, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), ROOT / "scripts/run_feedback_nested_benchmark.py",
                  ROOT / "src/cross_species/feedback_data.py", ROOT / "src/cross_species/feedback_evaluation.py"]
    specification = {
        "scope": "Retrospective compound-nested specificity controls. Prior public-data outcomes were inspected; not preregistered or prospective.",
        "frozen_before_model_fitting": True, "seed": seed, "outer_folds": outer_folds, "inner_folds": inner_folds,
        "alpha_grid": list(alphas), "rows": len(source), "compounds": len(set(groups.tolist())), "target_genes": target.shape[1],
        "models": {name: list(blocks) for name, blocks in MODEL_BLOCKS.items()},
        "common_direction": "Source row-mean by training time/nominal-dose stratum, re-estimated inside every inner and outer training fold. Unseen strata use the all-training source mean. Norm squared <=1e-24 gives amplitude zero and leaves the source residual unchanged.",
        "common_features": ["dot(source, training_context_source_mean)/squared_norm(training_context_source_mean)", "sqrt(mean(source**2))"],
        "residual": "source minus projection_amplitude times training_context_source_mean, in original source delta units",
        "kernel": "Each block independently train-only column-z-scored and divided by feature count; each model averages its block kernels with equal weights. Human targets retain original units and a training gene intercept.",
        "selection": "Same human labels, full gene axis and alpha grid; each model minimizes pooled inner compound-macro MSE. Fitting ridge MSE weights rows; evaluation weights compounds equally.",
        "primary": "context_plus_common MSE minus context_plus_common_residual MSE; positive means gain beyond the specified amplitude/direction controls.",
        "identification_rules": {"similarity": "Pearson across all genes after subtracting outer-training human mean for the same time/nominal-dose stratum; unseen strata use global training human mean.",
                           "candidate_pool": "Unique compounds in the same outer-test fold and time/nominal-dose stratum, including the correct compound. Duplicate compound-stratum rows are rejected before fitting.",
                           "chance": "Per query: top1=1/K, rank=(K+1)/2, MRR=harmonic(K)/K; then rows within compound and compounds equally.",
                           "ties": "Absolute score tolerance 1e-10; uniform ordering in the true-score tie. Top1 fractional if tied for first; rank is mean tied rank; MRR is mean reciprocal of tied positions.",
                           "undefined": "Fewer than 2 candidates, candidate centred norm <=1e-12, or query centred norm <=1e-12 remains missing with reasons. Undefined candidate keeps its whole pool undefined, without shrinking K.",
                           "comparability": "Report each model's coverage and a common-defined query set for the four learned models. Zero response and training human context mean are separate references."},
        "context_inputs": data.context_feature_ids[list(CONTEXT_COLUMNS)].tolist(),
        "input_hashes": data.input_hashes, "data_warnings": list(data.warnings),
        "code_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in code_files},
        "interval_scope": "Paired compound bootstrap of fixed OOF predictions; no model refitting or donor uncertainty. Overlapping training folds induce dependence; percentile coverage is not established.",
        "limitations": ["Common features control projection amplitude and total magnitude, including residual energy; they do not exhaust shared biology or generic stress.",
                        "Residual information is an operational prediction control, not a causal or biological-specificity decomposition.",
                        "One animal species and one broad liver in-vitro system; nominal dose levels can differ in absolute concentration.",
                        "Joint historical RMA, prior target inspection and replicate-averaged profiles limit prospective inference.",
                        "Retrieval is terminal evaluation on known candidate human profiles, not a deployable label-free identification system.",
                        "A negative result does not establish that measured source responses contain only common biology."],
    }
    save_json(out_dir / "specification.json", specification)
    snapshot = out_dir / "code_snapshot"
    snapshot.mkdir()
    for p in code_files:
        shutil.copy2(p, snapshot / p.name)
    with (out_dir / "fold_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "compound", "outer_fold", "stratum"])
        writer.writerows(zip(data.sample_ids, groups, folds, strata))
    predictions = {name: np.empty_like(target, dtype=np.float32) for name in [*MODEL_BLOCKS, *REFERENCES]}
    fold_records, started = [], time.monotonic()
    for fold in range(outer_folds):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        print(f"specificity fold {fold + 1}/{outer_folds}: {len(train)} train / {len(test)} test", flush=True)
        values, selection, state = fit_outer_fold(source, context, target, groups, strata, train, test,
                                                alphas, inner_folds, seed + 1009 * (fold + 1))
        for name in predictions:
            predictions[name][test] = values[name]
        np.savez_compressed(out_dir / f"fold_{fold}_transforms_models.npz", **state,
                            human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids)
        record = {"fold": fold, "training_compounds": len(set(groups[train].tolist())),
                  "test_compounds": len(set(groups[test].tolist())), **selection}
        fold_records.append(record)
        save_json(out_dir / f"fold_{fold}_selection.json", record)
        print(f"specificity fold {fold + 1} predictions frozen", flush=True)
    # Performance is calculated only after every outer prediction has been frozen.
    np.savez_compressed(out_dir / "oof_predictions.npz", sample_ids=data.sample_ids, compounds=groups,
                        outer_fold=folds, strata=strata, human_gene_ids=data.feature_ids,
                        source_gene_ids=data.source_feature_ids, truth=target.astype(np.float32), **predictions)
    metrics = {name: compound_macro_loss(value, target, groups) for name, value in predictions.items()}
    gains = {}
    for index, (label, baseline, method) in enumerate([
        ("residual_beyond_common", "context_plus_common", "context_plus_common_residual"),
        ("common_beyond_context", "context_only", "context_plus_common"),
        ("full_source_beyond_common", "context_plus_common", "full_source_plus_context"),
        ("full_source_beyond_context", "context_only", "full_source_plus_context"),
    ]):
        gain = paired_compound_bootstrap(predictions[baseline], predictions[method], target, groups,
                                         n_boot=n_boot, seed=seed + index)
        gain["inference_scope"] = specification["interval_scope"]
        gains[label] = gain
    retrieval, rows, common_queries = identification(predictions, target,
        predictions["training_human_context_mean"].astype(float), groups, strata, folds)
    save_json(out_dir / "identification_rows.json", rows)
    for record in fold_records:
        ix = folds == record["fold"]
        record["metrics"] = {name: compound_macro_loss(value[ix], target[ix], groups[ix]) for name, value in predictions.items()}
        record["residual_beyond_common_delta"] = record["metrics"]["context_plus_common"]["macro_loss"] - record["metrics"]["context_plus_common_residual"]["macro_loss"]
    report = {**specification, "metrics": metrics, "gains": gains, "identification": retrieval,
              "identification_common_four_model_queries": int(common_queries.sum()),
              "fold_results": fold_records, "elapsed_seconds": time.monotonic() - started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__}}
    save_json(out_dir / "report.json", report)
    with (out_dir / "compound_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["compound", *[name + "_mse" for name in predictions], "residual_beyond_common_delta"])
        for i, group in enumerate(metrics["context_only"]["group_ids"]):
            writer.writerow([group, *[metrics[name]["group_losses"][i] for name in predictions], gains["residual_beyond_common"]["group_deltas"][i]])
    print(f"Completed specificity evaluation: {len(source)} conditions, {len(set(groups.tolist()))} compounds, {target.shape[1]} genes", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/tg_gates_rat_human_v1/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feedback_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/feedback_specificity_20261107"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()
    if args.threads < 1 or args.n_boot < 1:
        parser.error("threads and n-boot must be positive")
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=20261107)
    run_benchmark(data, args.out_dir, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
