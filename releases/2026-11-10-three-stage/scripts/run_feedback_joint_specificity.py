#!/usr/bin/env python3
"""Joint molecular/context and source-amplitude controls for measured residual input.

Only the two predeclared joint models are fitted. Their prediction difference is an
operational kernel comparison and does not identify a causal biological component.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if __name__ == "__main__":
    _preparser = argparse.ArgumentParser(add_help=False)
    _preparser.add_argument("--threads", type=int, default=2)
    _preargs, _ = _preparser.parse_known_args()
    for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[_key] = str(_preargs.threads)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_feedback_nested_benchmark import (
    CenteredKernel, ChemicalKernelInputs, SpectralRidge,
    absolute_concentration_data, align_compound_features, context_columns,
    file_hash, group_folds, model_context, save_json,
)
from run_feedback_specificity_benchmark import TrainingCommon
from cross_species.feedback_data import load_feedback_data
from cross_species.feedback_evaluation import compound_macro_loss, paired_compound_bootstrap

ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
RESIDUAL_WEIGHTS = (0., .25, .5, .75)
BASELINE = "chemical_context_plus_common"
AUGMENTED = "chemical_context_plus_common_residual"
METHODS = (BASELINE, AUGMENTED)


class CenteredLinearBlock:
    def __init__(self, train):
        train = np.asarray(train, dtype=float)
        if train.ndim != 2 or not len(train) or not train.shape[1] or not np.isfinite(train).all():
            raise ValueError("Finite nonempty linear training block required")
        self.mean, self.scale = train.mean(axis=0), train.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.
        self.train = (train - self.mean) / self.scale
        self.center = CenteredKernel(self.train @ self.train.T / self.train.shape[1])

    def cross(self, value):
        value = (np.asarray(value, dtype=float) - self.mean) / self.scale
        return self.center.cross(value @ self.train.T / self.train.shape[1])

    def state(self, prefix):
        return {prefix + "_mean": self.mean, prefix + "_scale": self.scale,
                prefix + "_kernel_column_mean": self.center.column_mean,
                prefix + "_kernel_grand_mean": np.asarray(self.center.grand_mean),
                prefix + "_kernel_trace_scale": np.asarray(self.center.scale)}


class JointKernelInputs:
    def __init__(self, source_train, context_train, strata_train, fingerprints, descriptors):
        self.common_fit = TrainingCommon(source_train, strata_train)
        common, residual, _ = self.common_fit.transform(source_train, strata_train)
        self.prior = ChemicalKernelInputs(source_train, context_train, fingerprints, descriptors)
        self.common = CenteredLinearBlock(common)
        self.residual = CenteredLinearBlock(residual)

    @staticmethod
    def combine(prior, common, residual):
        return {"base": (prior + common) / 2., "residual": residual}

    def train_kernels(self):
        return self.combine(self.prior.prior_center.train, self.common.center.train, self.residual.center.train)

    def cross(self, source, context, strata, fingerprints, descriptors):
        common, residual, diagnostics = self.common_fit.transform(source, strata)
        prior = self.prior.prior_cross(context, fingerprints=fingerprints, descriptors=descriptors)
        return self.combine(prior, self.common.cross(common), self.residual.cross(residual)), diagnostics

    def state(self):
        return {**{"molecular_" + k: v for k, v in self.prior.saved_transforms().items()},
                **self.common_fit.state("source_common"),
                **self.common.state("common"), **self.residual.state("residual")}


def chemical_rows(chemical, rows):
    return {name: chemical[name][rows] for name in ["fingerprints", "descriptors"]}


def weighted_kernel(blocks, residual_weight):
    return (1. - residual_weight) * blocks["base"] + residual_weight * blocks["residual"]


def select_alphas(source, context, target, groups, strata, chemical, alphas, inner_folds, seed):
    folds = group_folds(groups, inner_folds, seed)
    losses = {weight: np.empty((len(alphas), len(source))) for weight in RESIDUAL_WEIGHTS}
    for fold in range(inner_folds):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inputs = JointKernelInputs(source[train], context[train], strata[train], **chemical_rows(chemical, train))
        kt = inputs.train_kernels()
        kv, _ = inputs.cross(source[valid], context[valid], strata[valid], **chemical_rows(chemical, valid))
        for weight in RESIDUAL_WEIGHTS:
            ridge = SpectralRidge(weighted_kernel(kt, weight), target[train])
            losses[weight][:, valid] = ridge.row_losses(weighted_kernel(kv, weight), target[valid], alphas)
    selected, trials = {}, {}
    unique = sorted(set(groups.tolist()))
    candidates = []
    for weight in RESIDUAL_WEIGHTS:
        macro = [np.mean([row[groups == group].mean() for group in unique]) for row in losses[weight]]
        candidates.extend({"alpha": float(a), "residual_weight": weight, "compound_macro_mse": float(v)} for a, v in zip(alphas, macro))
    trials[BASELINE] = [trial for trial in candidates if trial["residual_weight"] == 0.]
    trials[AUGMENTED] = candidates
    for name in METHODS:
        # Fixed candidate order breaks exact loss ties toward smaller w, then alpha.
        best = min(trials[name], key=lambda trial: trial["compound_macro_mse"])
        selected[name] = {"alpha": best["alpha"], "residual_weight": best["residual_weight"]}
    return selected, trials, folds


def fit_outer_fold(source, context, target, groups, strata, chemical, train, test,
                   alphas=ALPHAS, inner_folds=3, seed=0):
    selected, trials, inner = select_alphas(source[train], context[train], target[train], groups[train],
        strata[train], chemical_rows(chemical, train), alphas, inner_folds, seed)
    inputs = JointKernelInputs(source[train], context[train], strata[train], **chemical_rows(chemical, train))
    kt = inputs.train_kernels()
    kv, diagnostic = inputs.cross(source[test], context[test], strata[test], **chemical_rows(chemical, test))
    predictions, states = {}, {}
    for name in METHODS:
        alpha, weight = selected[name]["alpha"], selected[name]["residual_weight"]
        ridge = SpectralRidge(weighted_kernel(kt, weight), target[train])
        predictions[name] = ridge.predict(weighted_kernel(kv, weight), alpha)
        states[name + "_dual_coef"] = ridge.coefficients(alpha)
    state = {**inputs.state(), **states, "train_indices": train, "test_indices": test,
             "inner_fold": inner, "human_training_mean": target[train].mean(axis=0)}
    selection = {"selected_parameters": selected, "inner_trials": trials,
                 "selected_upper_grid_boundary": {name: selected[name]["alpha"] == max(alphas) for name in METHODS},
                 "test_source_common_fallback_rows": int(diagnostic["fallback"].sum()),
                 "test_source_zero_direction_rows": int(diagnostic["zero_direction"].sum())}
    return predictions, selection, state


def validate_feature_manifest(feature_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    actual = file_hash(feature_path)
    if manifest.get("features_sha256") != actual:
        raise ValueError("Chemical feature hash does not match its verification manifest")
    return {"features_file": str(Path(feature_path).resolve()), "features_sha256": actual,
            "manifest_file": str(Path(manifest_path).resolve()), "manifest_sha256": file_hash(manifest_path),
            "verification": manifest.get("verification"), "source": manifest.get("source"),
            "manifest_accepted_compounds": len(manifest.get("accepted", []))}


def run_benchmark(data, chemical, molecular_provenance, out_dir, *, alphas=ALPHAS,
                  outer_folds=5, inner_folds=3, seed=20261107, n_boot=5000,
                  absolute_concentrations=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty output directory; existing frozen runs cannot be overwritten")
    exposure_provenance = None
    if absolute_concentrations:
        data, exposure_provenance = absolute_concentration_data(data)
        chemical = chemical_rows(chemical, exposure_provenance["original_row_indices"])
        molecular_provenance = {**molecular_provenance,
            "actual_model_compounds_after_exposure_filter": sorted(set(data.compounds.tolist()))}
    source, target = np.asarray(data.source_delta, dtype=float), np.asarray(data.target_delta, dtype=float)
    context = model_context(data)
    groups = np.asarray(data.compounds).astype(str)
    strata = np.asarray([format(float(row["time_h"]), ".12g") + "|" + row["dose_level"].casefold() for row in data.metadata])
    if source.ndim != 2 or source.shape != target.shape or len(source) != len(context) or not all(np.isfinite(v).all() for v in [source, target, context]):
        raise ValueError("Finite aligned full-gene source, target and context required")
    if not len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Fixed positive regularization grid required")
    if not all(len(chemical[name]) == len(source) for name in ["fingerprints", "descriptors"]):
        raise ValueError("Chemical features must be aligned to the modelling rows")
    folds = group_folds(groups, outer_folds, seed)
    code_files = [Path(__file__), ROOT / "scripts/run_feedback_specificity_benchmark.py",
        ROOT / "scripts/run_feedback_nested_benchmark.py", ROOT / "src/cross_species/feedback_data.py",
        ROOT / "src/cross_species/feedback_evaluation.py"]
    specification = {
        "scope": "Retrospective joint-control analysis specified after inspection of separate chemical-prior and common-direction results. Not preregistered or prospective.",
        "frozen_before_model_fitting": True, "rows": len(source), "compounds": len(set(groups.tolist())),
        "target_genes": target.shape[1], "outer_folds": outer_folds, "inner_folds": inner_folds, "seed": seed,
        "alpha_grid": list(alphas), "bootstrap_draws": n_boot, "bootstrap_seed": seed,
        "models": {BASELINE: {"kernel": "Kbase=.5*Kprior+.5*Kcommon", "residual_weight": 0.},
                   AUGMENTED: {"kernel": "(1-w)*Kbase+w*Kresidual", "residual_weight_grid": list(RESIDUAL_WEIGHTS)}},
        "molecular_prior": "Frozen ChemicalKernelInputs Kprior: half PubChem881 Tanimoto plus half descriptor RBF form Kmol; standardized context RBF forms Kctx; raw prior=(Kmol+Kctx+Kmol*Kctx)/3. RBF exp(-squared_distance/(2*dimension)); descriptor missing values use training medians. Prior is training-centered and divided by training mean diagonal.",
        "common_control": "Training source mean direction by time/nominal-dose stratum, with global training source mean fallback; features are projection amplitude a and source RMS. Norm squared <=1e-24 yields a=0 and unchanged residual.",
        "source_residual": "source-a*training_source_context_mean in original source units; re-estimated inside every inner and outer training fold.",
        "linear_block_normalization": "Common and residual independently training-column z-scored (scales<1e-8 set to1), linear Gram divided by block feature count, then training-centered and divided by training mean diagonal. Degenerate trace<=1e-12 uses scale1 and retains a zero block. Query kernels use the same training centering/scale.",
        "selection": "Both models use identical human labels, compound folds, all supplied orthologues and seven-point alpha grid. Baseline selects alpha; augmented jointly selects w/alpha by pooled inner compound-macro MSE. Exact ties choose smaller w then alpha. w=0 contains the exact baseline procedure. Outer outcomes never trigger fallback or selection. Ridge fitting weights rows; evaluation weights compounds equally.",
        "primary": "Baseline joint chemical/context+common MSE minus augmented joint chemical/context+common+residual MSE, in full human gene space.",
        "interval_scope": "5000 paired compound bootstrap draws conditional on fixed OOF predictions by default; no refitting or donor uncertainty. Overlapping fold training induces dependence; percentile population coverage is not established.",
        "context_inputs": data.context_feature_ids[context_columns(data)].tolist(),
        "input_hashes": data.input_hashes, "molecular_provenance": molecular_provenance,
        "data_warnings": list(data.warnings),
        "code_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in code_files},
        "limitations": ["Joint conditioning controls the specified molecular/context kernel, projection amplitude and total source magnitude; it does not exhaust chemical knowledge or shared biology.",
                        "The augmented family preserves the prior/common relative weights and includes w=0, while introducing inner selection of residual weight. Delta MSE is an operational predictive contrast, not a causal biological fraction. Containing the baseline does not guarantee better outer-fold performance.",
                        "Feature-verified cohort excludes unavailable or unresolved chemical identities; compare both models within this cohort.",
                        "Joint historical RMA, prior public-label inspection, averaged profiles and nominal dose mismatches remain applicable.",
                        "Only one animal species and one broad liver in-vitro system are evaluated; no new wet-laboratory data or clinical outcomes are included."],
    }
    if exposure_provenance is not None:
        specification["absolute_concentrations"] = exposure_provenance
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "specification.json", specification)
    snapshot = out_dir / "code_snapshot"
    snapshot.mkdir()
    for path in code_files:
        shutil.copy2(path, snapshot / path.name)
    with (out_dir / "fold_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "compound", "outer_fold", "stratum"])
        writer.writerows(zip(data.sample_ids, groups, folds, strata))
    predictions = {name: np.empty_like(target, dtype=np.float32) for name in METHODS}
    fold_records, started = [], time.monotonic()
    for fold in range(outer_folds):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        print(f"joint specificity fold {fold+1}/{outer_folds}: {len(train)} train / {len(test)} test", flush=True)
        values, selection, state = fit_outer_fold(source, context, target, groups, strata, chemical,
            train, test, alphas=alphas, inner_folds=inner_folds, seed=seed+1009*(fold+1))
        for name in METHODS:
            predictions[name][test] = values[name]
        np.savez_compressed(out_dir / f"fold_{fold}_transforms_models.npz", **state,
                            human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids)
        record = {"fold": fold, "training_compounds": len(set(groups[train].tolist())),
                  "test_compounds": len(set(groups[test].tolist())), **selection}
        fold_records.append(record)
        save_json(out_dir / f"fold_{fold}_selection.json", record)
        print(f"joint specificity fold {fold+1} predictions frozen", flush=True)
    # All terminal outcomes are scored only after the full OOF predictions freeze.
    np.savez_compressed(out_dir / "oof_predictions.npz", sample_ids=data.sample_ids, compounds=groups,
        outer_fold=folds, human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids,
        strata=strata, context=context, context_feature_ids=np.asarray(specification["context_inputs"]),
        truth=target.astype(np.float32), **predictions)
    metrics = {name: compound_macro_loss(predictions[name], target, groups) for name in METHODS}
    gain = paired_compound_bootstrap(predictions[BASELINE], predictions[AUGMENTED], target, groups,
                                     n_boot=n_boot, seed=seed)
    gain["inference_scope"] = specification["interval_scope"]
    for record in fold_records:
        ix = folds == record["fold"]
        record["metrics"] = {name: compound_macro_loss(predictions[name][ix], target[ix], groups[ix]) for name in METHODS}
        record["joint_residual_gain"] = record["metrics"][BASELINE]["macro_loss"] - record["metrics"][AUGMENTED]["macro_loss"]
    report = {**specification, "metrics": metrics, "joint_residual_gain": gain, "fold_results": fold_records,
              "selected_upper_boundary_fold_counts": {name: sum(r["selected_upper_grid_boundary"][name] for r in fold_records) for name in METHODS},
              "elapsed_seconds": time.monotonic()-started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                "thread_limits": {key: os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}}
    save_json(out_dir / "report.json", report)
    with (out_dir / "compound_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["compound", BASELINE+"_mse", AUGMENTED+"_mse", "joint_residual_gain"])
        for i, group in enumerate(metrics[BASELINE]["group_ids"]):
            writer.writerow([group, metrics[BASELINE]["group_losses"][i], metrics[AUGMENTED]["group_losses"][i], gain["group_deltas"][i]])
    output_files = [p for p in out_dir.rglob("*") if p.is_file()]
    save_json(out_dir / "output_hashes.json", {str(p.relative_to(out_dir)): file_hash(p) for p in output_files})
    print(f"Completed joint specificity: {len(source)} rows, {len(set(groups.tolist()))} compounds, {target.shape[1]} genes", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/tg_gates_rat_human_v1/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feedback_cache"))
    parser.add_argument("--compound-features", type=Path, default=Path("artifacts/feedback_compound_features_20261107/features.npz"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("artifacts/feedback_compound_features_20261107/manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/feedback_joint_specificity_20261107"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--absolute-concentrations", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    verified = validate_feature_manifest(args.compound_features, args.feature_manifest)
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=20261107)
    data, chemical, cohort = align_compound_features(data, args.compound_features)
    run_benchmark(data, chemical, {**cohort, **verified}, args.out_dir,
                  absolute_concentrations=args.absolute_concentrations)


if __name__ == "__main__":
    main()
