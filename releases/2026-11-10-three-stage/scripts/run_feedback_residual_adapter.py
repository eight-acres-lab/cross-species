#!/usr/bin/env python3
"""Cross-fitted measurement-error residual ridge around a chemical/context prior.

This is a classical cross-fitted residual calibration benchmark. Reference human
and rat nuisance predictions exclude each reference fold from fitting and alpha
selection. At deployment, only an observed rat response updates the human prior.
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
from cross_species.feedback_evaluation import compound_macro_loss, paired_compound_bootstrap

ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
METHODS = ("human_prior", "bias_only", "residual_gamma1", "residual_gamma05")


def chemical_rows(chemical, rows):
    return {key: chemical[key][rows] for key in ["fingerprints", "descriptors"]}


def prepare_cohort(data, chemical, molecular_provenance, absolute_concentrations=False):
    exposure = None
    provenance = dict(molecular_provenance)
    if absolute_concentrations:
        data, exposure = nested.absolute_concentration_data(data)
        chemical = chemical_rows(chemical, exposure["original_row_indices"])
        provenance["actual_model_compounds_after_exposure_filter"] = sorted(set(data.compounds.tolist()))
    return data, chemical, provenance, exposure


def select_prior_alphas(source, context, human, groups, chemical, *, inner_folds, seed, alphas=ALPHAS):
    """Select nuisance hyperparameters on supplied training compounds only."""
    folds = nested.group_folds(groups, inner_folds, seed)
    losses = {name: np.empty((len(alphas), len(source))) for name in ["human", "rat"]}
    for fold in range(inner_folds):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inputs = nested.ChemicalKernelInputs(source[train], context[train], **chemical_rows(chemical, train))
        kt = inputs.prior_center.train
        kv = inputs.prior_cross(context[valid], **chemical_rows(chemical, valid))
        for name, target in [("human", human), ("rat", source)]:
            ridge = nested.SpectralRidge(kt, target[train])
            losses[name][:, valid] = ridge.row_losses(kv, target[valid], alphas)
    unique = sorted(set(groups.tolist()))
    selected, trials = {}, {}
    for name in losses:
        macro = [float(np.mean([row[groups == group].mean() for group in unique])) for row in losses[name]]
        selected[name] = float(alphas[int(np.argmin(macro))])
        trials[name] = [{"alpha": float(alpha), "inner_compound_macro_mse": value}
                        for alpha, value in zip(alphas, macro)]
    return selected, trials, folds


class ChemicalPriorPair:
    """Two output maps sharing a human-label-free molecular/context kernel."""

    @classmethod
    def fit(cls, source, context, human, chemical, selected_alphas):
        result = cls()
        result.inputs = nested.ChemicalKernelInputs(source, context, **chemical)
        kernel = result.inputs.prior_center.train
        human_ridge = nested.SpectralRidge(kernel, human)
        rat_ridge = nested.SpectralRidge(kernel, source)
        result.human_coef = human_ridge.coefficients(selected_alphas["human"])
        result.rat_coef = rat_ridge.coefficients(selected_alphas["rat"])
        result.human_mean = human_ridge.target_mean.copy()
        result.rat_mean = rat_ridge.target_mean.copy()
        return result

    def predict(self, context, fingerprints, descriptors):
        kernel = self.inputs.prior_cross(context, fingerprints=fingerprints, descriptors=descriptors)
        return {"human": kernel @ self.human_coef + self.human_mean,
                "rat": kernel @ self.rat_coef + self.rat_mean}

    def state(self):
        state = self.inputs.saved_transforms()
        state.update(train_standardized_context=self.inputs.context_train,
                     human_coef=self.human_coef, rat_coef=self.rat_coef,
                     human_mean=self.human_mean, rat_mean=self.rat_mean)
        return {name: np.asarray(value).copy() for name, value in state.items()}

    @classmethod
    def from_state(cls, state):
        """Restore a prior pair without targets or any transform fitting."""
        result = cls()
        inputs = nested.ChemicalKernelInputs.__new__(nested.ChemicalKernelInputs)
        for name in ["source_mean", "source_scale", "context_mean", "context_scale", "descriptor_median", "descriptor_mean", "descriptor_scale"]:
            setattr(inputs, name, state[name].copy())
        inputs.context_train = state["train_standardized_context"].copy()
        inputs.fingerprints_train = state["train_fingerprints"].copy()
        inputs.descriptors_train = state["train_standardized_descriptors"].copy()
        for name in ["prior", "source"]:
            center = nested.CenteredKernel.__new__(nested.CenteredKernel)
            center.column_mean = state[name + "_kernel_train_column_mean"].copy()
            center.grand_mean = float(state[name + "_kernel_train_grand_mean"])
            center.scale = float(state[name + "_kernel_train_trace_scale"])
            setattr(inputs, name + "_center", center)
        result.inputs = inputs
        for name in ["human_coef", "rat_coef", "human_mean", "rat_mean"]:
            setattr(result, name, state[name].copy())
        return result


class ResidualChain:
    """Frozen deployment API: predict(source, context, fingerprints, descriptors)."""

    @classmethod
    def fit(cls, prior_pair, source_innovation, human_residual):
        result = cls()
        result.prior = prior_pair
        result.innovation_mean = source_innovation.mean(axis=0)
        result.innovation_scale = source_innovation.std(axis=0)
        result.innovation_scale[result.innovation_scale < 1e-8] = 1.
        result.train_z = (source_innovation - result.innovation_mean) / result.innovation_scale
        kernel = result.train_z @ result.train_z.T / result.train_z.shape[1]
        ridge = nested.SpectralRidge(kernel, human_residual)
        result.residual_coef = ridge.coefficients(1.)
        result.bias = ridge.target_mean.copy()
        return result

    def predict_prior(self, context, fingerprints, descriptors):
        return self.prior.predict(context, fingerprints, descriptors)["human"]

    def predict(self, source, context, fingerprints, descriptors, gamma=1.):
        """Gamma zero exactly equals the bias-only baseline; no human target input."""
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        prior = self.prior.predict(context, fingerprints, descriptors)
        innovation = np.asarray(source, dtype=float) - prior["rat"]
        z = (innovation - self.innovation_mean) / self.innovation_scale
        conditional = (z @ self.train_z.T / self.train_z.shape[1]) @ self.residual_coef
        return prior["human"] + self.bias + gamma * conditional

    def predict_methods(self, source, context, fingerprints, descriptors):
        prior = self.prior.predict(context, fingerprints, descriptors)
        z = (np.asarray(source, dtype=float) - prior["rat"] - self.innovation_mean) / self.innovation_scale
        conditional = (z @ self.train_z.T / self.train_z.shape[1]) @ self.residual_coef
        bias = prior["human"] + self.bias
        return {"human_prior": prior["human"], "bias_only": bias,
                "residual_gamma1": bias + conditional, "residual_gamma05": bias + .5 * conditional}

    def state(self):
        state = {"prior_" + name: value for name, value in self.prior.state().items()}
        state.update({name: np.asarray(getattr(self, name)).copy() for name in
                      ["innovation_mean", "innovation_scale", "train_z", "residual_coef", "bias"]})
        return state

    @classmethod
    def from_state(cls, state):
        result = cls()
        result.prior = ChemicalPriorPair.from_state({name[6:]: value for name, value in state.items() if name.startswith("prior_")})
        for name in ["innovation_mean", "innovation_scale", "train_z", "residual_coef", "bias"]:
            setattr(result, name, state[name].copy())
        return result

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as payload:
            return cls.from_state({name: payload[name].copy() for name in payload.files})


def fit_reference_fold(source, context, human, groups, chemical, train, held,
                       *, seed, alphas=ALPHAS):
    """The reference held labels never enter nuisance fitting or alpha selection."""
    selected, trials, inner = select_prior_alphas(source[train], context[train], human[train],
                                                 groups[train], chemical_rows(chemical, train),
                                                 inner_folds=2, seed=seed, alphas=alphas)
    pair = ChemicalPriorPair.fit(source[train], context[train], human[train],
                                 chemical_rows(chemical, train), selected)
    predictions = pair.predict(context[held], **chemical_rows(chemical, held))
    return pair, predictions, {"selected_alphas": selected, "inner_trials": trials,
                                "selection_inner_fold": inner}


def fit_outer_chain(source, context, human, groups, chemical, train, test, *, seed,
                    alphas=ALPHAS, reference_folds=3, state_callback=None):
    """Fit on outer-train labels; test human values are not accessed."""
    st, ct, ht, gt = source[train], context[train], human[train], groups[train]
    chemistry = chemical_rows(chemical, train)
    reference = nested.group_folds(gt, reference_folds, seed + 701)
    source_residual, human_residual = np.empty_like(st), np.empty_like(ht)
    nuisance_predictions = {"human": np.empty_like(ht), "rat": np.empty_like(st)}
    records = []
    for fold in range(reference_folds):
        fit, held = np.flatnonzero(reference != fold), np.flatnonzero(reference == fold)
        pair, prediction, record = fit_reference_fold(st, ct, ht, gt, chemistry, fit, held,
                                                      seed=seed + 1709 * (fold + 1), alphas=alphas)
        source_residual[held] = st[held] - prediction["rat"]
        human_residual[held] = ht[held] - prediction["human"]
        for name in nuisance_predictions:
            nuisance_predictions[name][held] = prediction[name]
        record.update(reference_fold=fold, fit_local_indices=fit, held_local_indices=held)
        records.append(record)
        if state_callback is not None:
            state_callback(fold, pair.state(), record)
    selected, trials, inner = select_prior_alphas(st, ct, ht, gt, chemistry,
                                                 inner_folds=3, seed=seed, alphas=alphas)
    deployed_prior = ChemicalPriorPair.fit(st, ct, ht, chemistry, selected)
    chain = ResidualChain.fit(deployed_prior, source_residual, human_residual)
    predictions = chain.predict_methods(source[test], context[test], **chemical_rows(chemical, test))
    info = {"final_prior_selected_alphas": selected, "final_prior_inner_trials": trials,
            "final_prior_inner_fold": inner, "reference_fold": reference,
            "reference_records": records, "nuisance_predictions": nuisance_predictions,
            "source_innovation": source_residual, "human_residual": human_residual}
    return chain, predictions, info


def run_benchmark(data, chemical, molecular_provenance, out_dir, *, direct_run_dir,
                  seed=20261107, alphas=ALPHAS, n_boot=5000, absolute_concentrations=False):
    out_dir, direct_run_dir = Path(out_dir), Path(direct_run_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    data, chemical, molecular_provenance, exposure = prepare_cohort(
        data, chemical, molecular_provenance, absolute_concentrations)
    source, human = np.asarray(data.source_delta, dtype=float), np.asarray(data.target_delta, dtype=float)
    columns, context = nested.context_columns(data), nested.model_context(data)
    groups = np.asarray(data.compounds).astype(str)
    folds = nested.group_folds(groups, 5, seed)
    if source.shape != human.shape or not all(np.isfinite(value).all() for value in [source, human, context]):
        raise ValueError("Finite aligned full-axis source/human/context required")
    direct_report = json.loads((direct_run_dir / "report.json").read_text())
    if (direct_report["input_hashes"] != data.input_hashes
            or direct_report["molecular_features"]["sha256"] != molecular_provenance["sha256"]
            or direct_report["alpha_grid"] != list(alphas)
            or direct_report["context_inputs"] != data.context_feature_ids[columns].tolist()):
        raise ValueError("Direct comparison inputs/grid differ")
    with np.load(direct_run_dir / "oof_predictions.npz", allow_pickle=False) as payload:
        if not all(np.array_equal(payload[name], value) for name, value in
                   [("sample_ids", data.sample_ids), ("compounds", groups), ("outer_fold", folds),
                    ("human_gene_ids", data.feature_ids), ("truth", human.astype(np.float32))]):
            raise ValueError("Direct comparison cohort/folds/gene axis differ")
        direct = payload["source_plus_chemical_context"].copy()
        previous_prior = payload["chemical_context"].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__), Path(nested.__file__), ROOT / "src/cross_species/feedback_data.py",
                  ROOT / "src/cross_species/feedback_evaluation.py"]
    specification = {
        "scope": "Retrospective classical cross-fitted residual-ridge measurement update; specified after prior diagnostics, not a new theory, preregistration, prospective or independent validation.",
        "rows": len(source), "compounds": len(set(groups.tolist())), "target_genes": human.shape[1],
        "seed": seed, "outer_folds": 5, "reference_folds": 3, "reference_selection_folds": 2,
        "final_prior_selection_folds": 3, "prior_alpha_grid": list(alphas), "adapter_alpha": 1.,
        "prior": "Frozen ChemicalKernelInputs Kprior; separate human and rat outputs. Each reference held compound is excluded from both nuisance fitting and its alpha selection. All preprocessing is local to each nuisance training fold.",
        "adapter": "Reference OOF E_S=S-predS and E_H=H-predH; train-only column-standardized E_S, linear kernel divided by all source genes, ridge alpha1 with mean(E_H) intercept b.",
        "deployment": "Final human/rat priors each select alpha using three inner folds of all outer-training compounds. Test innovation is observed rat response minus final rat prior; no test human response enters prediction.",
        "gamma": "Prediction=human_prior+b+gamma*f(E_S). gamma1 is primary; gamma0 equals bias-only exactly; gamma0.5 is fixed sensitivity. Gamma never changes after outer scores.",
        "primary_comparison": "bias_only compound-macro full-gene MSE minus residual_gamma1 MSE; positive means source-dependent gain beyond the training residual mean.",
        "secondary_comparisons": ["gamma1 vs human_prior", "gamma1 vs same-input direct source/prior fusion", "fixed gamma0.5 vs bias_only"],
        "direct_reference": str(direct_run_dir.resolve()), "direct_report_sha256": nested.file_hash(direct_run_dir / "report.json"),
        "input_hashes": data.input_hashes, "molecular_features": molecular_provenance,
        "context_inputs": data.context_feature_ids[columns].tolist(),
        "data_warnings": list(data.warnings), "code_hashes": {str(path.relative_to(ROOT)): nested.file_hash(path) for path in code_files},
        "interval_scope": "Compound bootstrap of frozen OOF predictions; overlapping outer/reference training induces dependence, without model refitting or donor uncertainty. Percentile population coverage is not established.",
        "limitations": ["OOF nuisance priors use two thirds of outer training compounds, whereas deployment priors use all outer training compounds; innovation/error distributions may shift.",
                        "One animal species, one broad liver system, historical joint RMA and averaged profiles remain applicable.",
                        "The method is classical residual ridge. A runnable update chain does not establish a new optimization method or guarantee positive transfer."],
    }
    if exposure is not None:
        specification["absolute_concentrations"] = exposure
    nested.save_json(out_dir / "specification.json", specification)
    (out_dir / "code_snapshot").mkdir()
    for path in code_files:
        shutil.copy2(path, out_dir / "code_snapshot" / path.name)
    with (out_dir / "fold_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "compound", "outer_fold"])
        writer.writerows(zip(data.sample_ids, groups, folds))
    predictions = {name: np.empty_like(human, dtype=np.float32) for name in METHODS}
    records, prior_reconstruction, started = [], [], time.monotonic()
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fold_dir = out_dir / ("outer_%d" % fold)
        fold_dir.mkdir()
        print("residual adapter outer fold %d/5: %d train/%d test conditions" % (fold + 1, len(train), len(test)), flush=True)

        def save_reference(reference_fold, state, info):
            np.savez_compressed(fold_dir / ("reference_%d_prior_state.npz" % reference_fold), **state,
                                fit_sample_ids=data.sample_ids[train[info["fit_local_indices"]]],
                                held_sample_ids=data.sample_ids[train[info["held_local_indices"]]],
                                human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids)
            nested.save_json(fold_dir / ("reference_%d_selection.json" % reference_fold), info)

        chain, prediction, info = fit_outer_chain(source, context, human, groups, chemical,
                                                  train, test, seed=seed + 1009 * (fold + 1),
                                                  alphas=alphas, state_callback=save_reference)
        for name, value in prediction.items():
            predictions[name][test] = value
        discrepancy = float(np.max(np.abs(predictions["human_prior"][test] - previous_prior[test])))
        prior_reconstruction.append({"fold": fold, "maximum_absolute_error": discrepancy})
        if not np.allclose(predictions["human_prior"][test], previous_prior[test], rtol=1e-6, atol=2e-7):
            raise ValueError("Final human-only prior differs from the fixed direct-comparison prior")
        np.savez_compressed(fold_dir / "deployment_state.npz", **chain.state(),
                            human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids,
                            context_feature_ids=data.context_feature_ids[columns],
                            train_sample_ids=data.sample_ids[train], test_sample_ids=data.sample_ids[test])
        np.savez_compressed(fold_dir / "reference_oof.npz", sample_ids=data.sample_ids[train],
                            compounds=groups[train], reference_fold=info["reference_fold"],
                            human_prior_prediction=info["nuisance_predictions"]["human"].astype(np.float32),
                            rat_prior_prediction=info["nuisance_predictions"]["rat"].astype(np.float32),
                            source_innovation=info["source_innovation"].astype(np.float32),
                            human_residual=info["human_residual"].astype(np.float32))
        hyper = {name: value for name, value in info.items() if name not in
                 {"nuisance_predictions", "source_innovation", "human_residual"}}
        hyper.update(outer_fold=fold, train_indices=train, test_indices=test)
        nested.save_json(fold_dir / "selection.json", hyper)
        records.append(hyper)
        print("outer fold %d predictions frozen; final prior alphas %s" % (fold + 1, info["final_prior_selected_alphas"]), flush=True)
    predictions["direct_source_prior_fusion"] = direct
    np.savez_compressed(out_dir / "oof_predictions.npz", sample_ids=data.sample_ids, compounds=groups,
                        outer_fold=folds, human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids,
                        truth=human.astype(np.float32), **predictions)
    metrics = {name: compound_macro_loss(value, human, groups) for name, value in predictions.items()}
    comparisons = {}
    for index, (name, baseline, method) in enumerate([
        ("residual_vs_bias", "bias_only", "residual_gamma1"),
        ("residual_vs_prior", "human_prior", "residual_gamma1"),
        ("residual_vs_direct_fusion", "direct_source_prior_fusion", "residual_gamma1"),
        ("gamma05_vs_bias", "bias_only", "residual_gamma05"),
    ]):
        gain = paired_compound_bootstrap(predictions[baseline], predictions[method], human, groups,
                                         n_boot=n_boot, seed=seed + index)
        gain["inference_scope"] = specification["interval_scope"]
        comparisons[name] = gain
    report = {**specification, "metrics": metrics, "comparisons": comparisons,
              "fold_records": records, "elapsed_seconds": time.monotonic() - started,
              "final_prior_reproduces_previous_nested_prior_with_float_tolerance": True,
              "prior_reconstruction_checks": prior_reconstruction}
    nested.save_json(out_dir / "report.json", report)
    with (out_dir / "compound_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["compound", *[name + "_mse" for name in predictions]])
        for i, group in enumerate(metrics["human_prior"]["group_ids"]):
            writer.writerow([group, *[metrics[name]["group_losses"][i] for name in predictions]])
    print(json.dumps(nested.plain({"metrics": {name: value["macro_loss"] for name, value in metrics.items()},
                                  "comparisons": comparisons, "elapsed_seconds": report["elapsed_seconds"]}), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/tg_gates_rat_human_v1/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feedback_cache"))
    parser.add_argument("--compound-features", type=Path, default=Path("artifacts/feedback_compound_features_20261107/features.npz"))
    parser.add_argument("--direct-run-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/feedback_residual_adapter_20261107"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--absolute-concentrations", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.n_boot < 1:
        parser.error("threads and n-boot must be positive")
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=args.seed)
    data, chemical, provenance = nested.align_compound_features(data, args.compound_features)
    reference = args.direct_run_dir or Path("outputs/feedback_nested_chemical_20261107_absolute_v1" if args.absolute_concentrations
                                            else "outputs/feedback_nested_chemical_20261107_full_v2")
    run_benchmark(data, chemical, provenance, args.out_dir, direct_run_dir=reference,
                  seed=args.seed, n_boot=args.n_boot, absolute_concentrations=args.absolute_concentrations)


if __name__ == "__main__":
    main()
