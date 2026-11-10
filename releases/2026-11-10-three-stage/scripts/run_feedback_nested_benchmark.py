#!/usr/bin/env python3
"""Retrospective compound-nested comparison of rat measurements and context.

All available eligible treatment compounds enter metadata-only outer folds. Human
responses are scored on the complete fixed ortholog axis. Outer outcomes never
select genes, transforms, regularization, folds, or a reported model family.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

# Set BLAS limits before importing NumPy in a standalone invocation.
if __name__ == "__main__":
    _preparser = argparse.ArgumentParser(add_help=False)
    _preparser.add_argument("--threads", type=int, default=2)
    _preargs, _ = _preparser.parse_known_args()
    for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[_key] = str(_preargs.threads)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.feedback_data import load_feedback_data
from cross_species.feedback_evaluation import compound_macro_loss, paired_compound_bootstrap

DEFAULT_ALPHAS = (.001, .01, .1, 1., 10.)
ABSOLUTE_ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
CONTEXT_COLUMNS = (0, 3, 4, 5, 6)
ABSOLUTE_CONTEXT_FEATURES = ("time_h", "log10_rat_M", "log10_human_M", "dose_Low", "dose_Middle", "dose_High", "is_lps")
METHODS = ("context_only", "source_plus_context", "source_copy", "calibrated_source")
CHEMICAL_METHODS = ("chemical_context", "source_plus_chemical_context", "source_copy", "calibrated_source")


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def group_folds(groups, n_folds, seed):
    """Assign entire compounds using identifiers and a seed, without responses."""
    groups = np.asarray(groups).astype(str)
    unique = np.asarray(sorted(set(groups.tolist())))
    if len(unique) < n_folds or n_folds < 2:
        raise ValueError("At least n_folds distinct compounds and n_folds >= 2 are required")
    order = np.random.default_rng(seed).permutation(len(unique))
    assignments = {unique[position]: rank % n_folds for rank, position in enumerate(order)}
    return np.asarray([assignments[group] for group in groups], dtype=np.int64)


def _scaler(train):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-8] = 1.
    return mean, scale


class KernelInputs:
    """Train-only block standardization; feature-count-normalized linear kernels."""

    def __init__(self, source_train, context_train):
        source_train = np.asarray(source_train, dtype=float)
        context_train = np.asarray(context_train, dtype=float)
        if (source_train.ndim != 2 or context_train.ndim != 2
                or len(source_train) != len(context_train) or not len(source_train)
                or not source_train.shape[1] or not context_train.shape[1]
                or not np.isfinite(source_train).all() or not np.isfinite(context_train).all()):
            raise ValueError("Finite nonempty source/context training matrices are required")
        self.source_mean, self.source_scale = _scaler(source_train)
        self.context_mean, self.context_scale = _scaler(context_train)
        self.source_train = (source_train - self.source_mean) / self.source_scale
        self.context_train = (context_train - self.context_mean) / self.context_scale

    def cross(self, source, context):
        source = (np.asarray(source, dtype=float) - self.source_mean) / self.source_scale
        context = (np.asarray(context, dtype=float) - self.context_mean) / self.context_scale
        kc = context @ self.context_train.T / self.context_train.shape[1]
        ks = source @ self.source_train.T / self.source_train.shape[1]
        return {"context_only": kc, "source_plus_context": .5 * (ks + kc)}

    def train_kernels(self):
        kc = self.context_train @ self.context_train.T / self.context_train.shape[1]
        ks = self.source_train @ self.source_train.T / self.source_train.shape[1]
        return {"context_only": kc, "source_plus_context": .5 * (ks + kc)}

    def saved_transforms(self):
        return {"source_mean": self.source_mean, "source_scale": self.source_scale,
                "context_mean": self.context_mean, "context_scale": self.context_scale}


class CenteredKernel:
    """Center and normalize using only the training Gram matrix."""

    def __init__(self, train_kernel):
        self.column_mean = train_kernel.mean(axis=0)
        self.grand_mean = float(self.column_mean.mean())
        centered = train_kernel - self.column_mean[None, :] - self.column_mean[:, None] + self.grand_mean
        scale = float(np.trace(centered) / len(centered))
        self.scale = scale if scale > 1e-12 else 1.
        self.train = centered / self.scale

    def cross(self, kernel):
        return (kernel - kernel.mean(axis=1, keepdims=True)
                - self.column_mean[None, :] + self.grand_mean) / self.scale


def tanimoto(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    intersection = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def rbf(left, right):
    distance = (np.square(left).sum(axis=1)[:, None]
                + np.square(right).sum(axis=1)[None, :] - 2 * left @ right.T)
    return np.exp(-np.maximum(distance, 0.) / (2 * left.shape[1]))


class ChemicalKernelInputs(KernelInputs):
    """Frozen molecular/context interaction prior, optionally plus rat measurements."""

    def __init__(self, source_train, context_train, fingerprints, descriptors):
        super().__init__(source_train, context_train)
        self.fingerprints_train = np.asarray(fingerprints, dtype=float).copy()
        descriptors = np.asarray(descriptors, dtype=float)
        self.descriptor_median = np.asarray([
            float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.
            for column in descriptors.T])
        imputed = np.where(np.isnan(descriptors), self.descriptor_median, descriptors)
        self.descriptor_mean, self.descriptor_scale = _scaler(imputed)
        self.descriptors_train = (imputed - self.descriptor_mean) / self.descriptor_scale
        raw_prior = self._prior(self.fingerprints_train, self.descriptors_train, self.context_train)
        raw_source = self.source_train @ self.source_train.T / self.source_train.shape[1]
        self.prior_center = CenteredKernel(raw_prior)
        self.source_center = CenteredKernel(raw_source)

    def _prior(self, fingerprints, descriptors, context):
        molecular = .5 * tanimoto(fingerprints, self.fingerprints_train)
        molecular += .5 * rbf(descriptors, self.descriptors_train)
        contextual = rbf(context, self.context_train)
        return (molecular + contextual + molecular * contextual) / 3.

    def cross(self, source, context, fingerprints=None, descriptors=None):
        if fingerprints is None or descriptors is None:
            raise ValueError("Molecular fingerprints and descriptors are required")
        source = (np.asarray(source, dtype=float) - self.source_mean) / self.source_scale
        prior = self.prior_cross(context, fingerprints=fingerprints, descriptors=descriptors)
        measured = self.source_center.cross(source @ self.source_train.T / self.source_train.shape[1])
        return {"chemical_context": prior, "source_plus_chemical_context": .5 * (prior + measured)}

    def prior_cross(self, context, *, fingerprints, descriptors):
        """Query the molecular/context prior without an animal test measurement."""
        context = (np.asarray(context, dtype=float) - self.context_mean) / self.context_scale
        imputed = np.where(np.isnan(descriptors), self.descriptor_median, descriptors)
        descriptors = (imputed - self.descriptor_mean) / self.descriptor_scale
        return self.prior_center.cross(self._prior(fingerprints, descriptors, context))

    def train_kernels(self):
        return {"chemical_context": self.prior_center.train,
                "source_plus_chemical_context": .5 * (self.prior_center.train + self.source_center.train)}

    def saved_transforms(self):
        result = super().saved_transforms()
        result.update({"descriptor_median": self.descriptor_median,
                       "descriptor_mean": self.descriptor_mean, "descriptor_scale": self.descriptor_scale,
                       "train_fingerprints": self.fingerprints_train,
                       "train_standardized_descriptors": self.descriptors_train})
        for name, center in [("prior", self.prior_center), ("source", self.source_center)]:
            result.update({name + "_kernel_train_column_mean": center.column_mean,
                           name + "_kernel_train_grand_mean": np.asarray(center.grand_mean),
                           name + "_kernel_train_trace_scale": np.asarray(center.scale)})
        return result


def _chemical_rows(chemical, rows):
    return {name: chemical[name][rows] for name in ["fingerprints", "descriptors"]}


def context_columns(data):
    """Use every verified absolute-exposure feature when that context is present."""
    names = data.context_feature_ids.tolist()
    present = [name in names for name in ["log10_rat_M", "log10_human_M"]]
    if any(present):
        if not all(present) or not all(name in names for name in ABSOLUTE_CONTEXT_FEATURES):
            raise ValueError("Incomplete absolute-concentration context")
        return [names.index(name) for name in ABSOLUTE_CONTEXT_FEATURES]
    return list(CONTEXT_COLUMNS)


def model_context(data):
    return np.asarray(data.context[:, context_columns(data)], dtype=float)


def absolute_concentration_data(data):
    """Filter only verified molar metadata; derive planned-dose context without outcomes.

    Units are case sensitive. No molecular-weight conversion, inferred unit,
    approximate amount matching, chemical alias, or dose rounding is introduced.
    """
    unit_to_M = {"M": 1., "mM": 1e-3, "uM": 1e-6, "μM": 1e-6, "nM": 1e-9}
    accepted, records, excluded = [], [], []
    for row_index, row in enumerate(data.metadata):
        reasons, converted = [], {}
        for species in ["rat", "human"]:
            unit = row.get(species + "_dose_unit", "")
            provenance = row.get(species + "_dose_unit_source", "unknown")
            if unit not in unit_to_M:
                reasons.append(species + "_unsupported_or_unknown_molar_unit")
            if provenance not in {"attributes", "metadata"}:
                reasons.append(species + "_unit_not_uniquely_verified")
            try:
                amount = float(row[species + "_dose_value"])
            except (ValueError, TypeError, KeyError):
                amount = float("nan")
            if not np.isfinite(amount) or amount <= 0:
                reasons.append(species + "_dose_not_finite_positive")
            if unit in unit_to_M and np.isfinite(amount) and amount > 0:
                concentration = amount * unit_to_M[unit]
                if not np.isfinite(concentration) or concentration <= 0:
                    reasons.append(species + "_converted_concentration_invalid")
                else:
                    converted[species] = concentration
        if reasons:
            excluded.append({"sample_id": str(data.sample_ids[row_index]), "compound": row["compound"],
                             "reasons": reasons, "rat_dose_value": row.get("rat_dose_value"),
                             "rat_dose_unit": row.get("rat_dose_unit"), "human_dose_value": row.get("human_dose_value"),
                             "human_dose_unit": row.get("human_dose_unit")})
            continue
        accepted.append(row_index)
        records.append({"sample_id": str(data.sample_ids[row_index]), "compound": row["compound"],
                        "time_h": float(row["time_h"]), "dose_level": row["dose_level"],
                        "rat_dose_value": row["rat_dose_value"], "rat_dose_unit": row["rat_dose_unit"],
                        "rat_unit_source": row["rat_dose_unit_source"],
                        "human_dose_value": row["human_dose_value"], "human_dose_unit": row["human_dose_unit"],
                        "human_unit_source": row["human_dose_unit_source"],
                        "rat_concentration_M": converted["rat"], "human_concentration_M": converted["human"],
                        "log10_rat_M": float(np.log10(converted["rat"])),
                        "log10_human_M": float(np.log10(converted["human"])),
                        "concentration_match_status": row.get("concentration_match_status", "unknown")})
    selected = np.asarray(accepted, dtype=int)
    if not len(selected):
        raise ValueError("No uniquely verified positive bilateral molar treatment concentrations")
    subset = copy.copy(data)
    for field in ["source_delta", "target_delta", "context", "sample_ids", "compounds", "roles"]:
        if hasattr(data, field):
            setattr(subset, field, getattr(data, field)[selected].copy())
    subset.context = np.asarray(subset.context, dtype=float)
    subset.context[:, 0] = [row["time_h"] for row in records]
    subset.context[:, 1] = [row["log10_rat_M"] for row in records]
    subset.context[:, 2] = [row["log10_human_M"] for row in records]
    subset.context_feature_ids = np.asarray(ABSOLUTE_CONTEXT_FEATURES)
    subset.metadata = [{**data.metadata[i], "rat_concentration_M": record["rat_concentration_M"],
                        "human_concentration_M": record["human_concentration_M"],
                        "log10_rat_M": record["log10_rat_M"], "log10_human_M": record["log10_human_M"]}
                       for i, record in zip(selected, records)]
    subset.warnings = tuple(
        ("Before absolute-concentration filtering: " + warning)
        if ("Dose units are missing" in warning or "Nominal concentration comparison is unknown" in warning)
        else warning for warning in data.warnings) + (
            "Absolute-concentration retained cohort has zero unknown/ambiguous units; both arms receive planned nominal molar concentrations.",)
    provenance = {
        "rule": "Both planned treatment doses must be finite and positive with uniquely resolved attributes/metadata units among M,mM,uM,μM,nM. Case-sensitive fixed conversion to M followed by log10; no inferred units, mass-to-molar conversion, aliases, or rounding.",
        "unit_to_M": unit_to_M, "input_conditions": len(data.metadata), "retained_conditions": len(selected),
        "input_compounds": len(set(data.compounds.tolist())), "retained_compounds": len(set(subset.compounds.tolist())),
        "excluded_compounds": sorted(set(data.compounds.tolist()) - set(subset.compounds.tolist())),
        "matching_status_counts": dict(collections.Counter(row["concentration_match_status"] for row in records)),
        "retained_unknown_or_ambiguous_unit_conditions": 0,
        "retained_unit_source_counts": {species: dict(collections.Counter(record[species + "_unit_source"] for record in records))
                                         for species in ["rat", "human"]},
        "original_row_indices": selected, "retained_metadata": records, "excluded_metadata": excluded,
        "information_available_to_both_arms": "Time, planned rat concentration, planned human concentration, nominal dose indicators and molecular features. Source-plus-prior alone additionally receives measured rat expression response.",
        "interpretation": "Nominal administered molar concentrations; intracellular exposure equivalence is not established. Different verified source/target concentrations are retained.",
    }
    return subset, provenance


def align_compound_features(data, path):
    """Restrict both arms by verified identity; no target-response-based filtering."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"compounds", "fingerprints", "descriptors", "descriptor_names", "verified_mask"}
        if set(payload.files) != required:
            raise ValueError("Compound feature NPZ fields differ from the required contract")
        features = {name: payload[name].copy() for name in required}
    normalize = lambda value: " ".join(str(value).replace("_", " ").casefold().split())
    names = [normalize(name) for name in features["compounds"]]
    fp, descriptors, verified = features["fingerprints"], features["descriptors"], features["verified_mask"]
    if (len(set(names)) != len(names) or fp.shape != (len(names), 881)
            or descriptors.ndim != 2 or descriptors.shape[0] != len(names)
            or descriptors.shape[1] != len(features["descriptor_names"])
            or not descriptors.shape[1] or verified.shape != (len(names),) or verified.dtype.kind != "b"
            or not np.isin(fp, [0, 1]).all() or np.isinf(descriptors).any()):
        raise ValueError("Invalid verified molecular feature shapes or values")
    if np.any(verified & (fp.sum(axis=1) == 0)):
        raise ValueError("Verified fingerprints cannot be empty")
    positions = {name: i for i, name in enumerate(names) if verified[i]}
    keep = np.asarray([normalize(compound) in positions for compound in data.compounds])
    if not keep.any():
        raise ValueError("No verified compounds match the response data")
    selected = np.flatnonzero(keep)
    feature_rows = np.asarray([positions[normalize(data.compounds[i])] for i in selected])
    subset = copy.copy(data)
    for field in ["source_delta", "target_delta", "context", "sample_ids", "compounds", "roles"]:
        if hasattr(data, field):
            setattr(subset, field, getattr(data, field)[selected].copy())
    subset.metadata = [dict(data.metadata[i]) for i in selected]
    chemical = {"fingerprints": fp[feature_rows], "descriptors": descriptors[feature_rows]}
    provenance = {"file": str(Path(path).resolve()), "sha256": file_hash(path),
                  "descriptor_names": features["descriptor_names"].tolist(),
                  "join": "Case-fold, underscores to spaces, whitespace collapse; no chemical alias inference.",
                  "included_compounds": sorted(set(subset.compounds.tolist())),
                  "excluded_compounds": sorted(set(data.compounds[~keep].tolist())),
                  "selection_basis": "Verified reagent identity and feature availability only; no response values.",
                  "original_row_indices": selected}
    return subset, chemical, provenance


class SpectralRidge:
    """Multi-output dual ridge, with an unpenalized training-target intercept."""

    def __init__(self, train_kernel, target_train):
        target_train = np.asarray(target_train, dtype=float)
        self.target_mean = target_train.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh((train_kernel + train_kernel.T) * .5)
        # Remove numerical null directions. This tolerance is fixed before evaluation.
        keep = eigenvalues > max(1e-12, float(eigenvalues.max(initial=0.)) * 1e-12)
        self.values = eigenvalues[keep]
        self.vectors = eigenvectors[:, keep]
        self.projected_targets = self.vectors.T @ (target_train - self.target_mean)

    def predict(self, cross_kernel, alpha):
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        weights = (cross_kernel @ self.vectors) / (self.values + alpha)
        return weights @ self.projected_targets + self.target_mean

    def row_losses(self, cross_kernel, target_valid, alphas):
        """Exact gene-MSE moments avoid constructing a gene matrix for every alpha."""
        centered = np.asarray(target_valid, dtype=float) - self.target_mean
        projected_kernel = cross_kernel @ self.vectors
        gram_targets = self.projected_targets @ self.projected_targets.T
        cross_targets = centered @ self.projected_targets.T
        true_norm = np.square(centered).sum(axis=1)
        output = []
        for alpha in alphas:
            weights = projected_kernel / (self.values + alpha)
            predicted_norm = ((weights @ gram_targets) * weights).sum(axis=1)
            cross_term = (weights * cross_targets).sum(axis=1)
            output.append(np.maximum((predicted_norm - 2 * cross_term + true_norm)
                                     / centered.shape[1], 0.))
        return np.asarray(output)

    def coefficients(self, alpha):
        return self.vectors @ (self.projected_targets / (self.values + alpha)[:, None])


def select_alphas(source, context, target, groups, alphas, inner_folds, seed, chemical=None):
    """Receives outer-training rows only; nested validation selects each alpha."""
    assignments = group_folds(groups, inner_folds, seed)
    model_names = (CHEMICAL_METHODS if chemical is not None else METHODS)[:2]
    losses = {name: np.empty((len(alphas), len(source))) for name in model_names}
    for fold in range(inner_folds):
        train, valid = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
        inputs = (KernelInputs(source[train], context[train]) if chemical is None else
                  ChemicalKernelInputs(source[train], context[train], **_chemical_rows(chemical, train)))
        train_kernels = inputs.train_kernels()
        valid_kernels = inputs.cross(source[valid], context[valid],
                                     **({} if chemical is None else _chemical_rows(chemical, valid)))
        for name in model_names:
            ridge = SpectralRidge(train_kernels[name], target[train])
            losses[name][:, valid] = ridge.row_losses(valid_kernels[name], target[valid], alphas)
    trials, selected = {}, {}
    unique = sorted(set(np.asarray(groups).tolist()))
    for name in model_names:
        macro = np.asarray([np.mean([row[np.asarray(groups) == group].mean() for group in unique])
                            for row in losses[name]])
        selected[name] = float(alphas[int(np.argmin(macro))])
        trials[name] = [{"alpha": float(alpha), "inner_compound_macro_mse": float(loss)}
                        for alpha, loss in zip(alphas, macro)]
    return selected, trials, assignments


def wrong_source_permutation(groups, strata, sample_ids, seed):
    """A fixed within-context bijection; same-compound donors remain ineligible."""
    groups, strata, sample_ids = map(np.asarray, [groups, strata, sample_ids])
    donor = np.arange(len(groups))
    eligible = np.zeros(len(groups), dtype=bool)
    rng = np.random.default_rng(seed)
    for stratum in sorted(set(strata.tolist())):
        rows = np.flatnonzero(strata == stratum)
        rows = np.asarray(sorted(rows, key=lambda row: (str(groups[row]), str(sample_ids[row]))))
        if len(rows) < 2:
            continue
        # Choose the greatest number of cross-compound donors using identifiers only.
        best, best_count = None, -1
        for offset in rng.permutation(np.arange(1, len(rows))):
            candidate = np.roll(rows, int(offset))
            count = int(np.sum(groups[rows] != groups[candidate]))
            if count > best_count:
                best, best_count = candidate, count
            if count == len(rows):
                break
        donor[rows] = best
        eligible[rows] = groups[rows] != groups[best]
    return donor, eligible


def fit_outer_fold(source, context, target, groups, sample_ids, strata, train, test,
                   alphas=DEFAULT_ALPHAS, inner_folds=3, seed=0, chemical=None):
    """Freeze models from train labels, then predict test using source/context only."""
    source, context, target = map(np.asarray, [source, context, target])
    selected, trials, inner = select_alphas(source[train], context[train], target[train],
                                           groups[train], alphas, inner_folds, seed,
                                           chemical=None if chemical is None else _chemical_rows(chemical, train))
    model_names = (CHEMICAL_METHODS if chemical is not None else METHODS)[:2]
    inputs = (KernelInputs(source[train], context[train]) if chemical is None else
              ChemicalKernelInputs(source[train], context[train], **_chemical_rows(chemical, train)))
    train_kernels = inputs.train_kernels()
    test_kernels = inputs.cross(source[test], context[test],
                                **({} if chemical is None else _chemical_rows(chemical, test)))
    predictions, states = {}, {}
    for name in model_names:
        ridge = SpectralRidge(train_kernels[name], target[train])
        predictions[name] = ridge.predict(test_kernels[name], selected[name])
        states[name + "_dual_coef"] = ridge.coefficients(selected[name])
    target_mean, source_mean = target[train].mean(axis=0), source[train].mean(axis=0)
    centered_source = source[train] - source_mean
    denominator = np.square(centered_source).sum()
    scalar = float(np.sum(centered_source * (target[train] - target_mean)) / denominator) if denominator > 0 else 0.
    predictions["source_copy"] = source[test].astype(float).copy()
    predictions["calibrated_source"] = target_mean + scalar * (source[test] - source_mean)
    donor, eligible = wrong_source_permutation(groups[test], strata[test], sample_ids[test], seed + 301)
    wrong_kernel = inputs.cross(source[test][donor], context[test],
                                 **({} if chemical is None else _chemical_rows(chemical, test)))[model_names[1]]
    predictions["wrong_" + model_names[1]] = wrong_kernel @ states[model_names[1] + "_dual_coef"] + target_mean
    state = {**inputs.saved_transforms(), **states, "target_mean": target_mean,
             "calibrated_source_mean": source_mean, "calibrated_source_scalar": np.asarray(scalar),
             "train_indices": train, "test_indices": test, "inner_fold": inner,
             "wrong_donor_indices": test[donor], "wrong_source_eligible": eligible}
    return predictions, {"selected_alphas": selected, "inner_trials": trials}, state


def row_pearson(prediction, truth):
    """Correlation across genes per condition; constants remain undefined (NaN)."""
    prediction, truth = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Pearson inputs must have equal matrix shapes")
    x, y = prediction - prediction.mean(axis=1, keepdims=True), truth - truth.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.square(x).sum(axis=1) * np.square(y).sum(axis=1))
    valid = (np.ptp(prediction, axis=1) > 0) & (np.ptp(truth, axis=1) > 0) & (denominator > 0)
    values = np.full(len(prediction), np.nan)
    values[valid] = np.clip((x[valid] * y[valid]).sum(axis=1) / denominator[valid], -1., 1.)
    return values


def correlation_summary(values, groups):
    group_ids = sorted(set(np.asarray(groups).tolist()))
    by_group = []
    for group in group_ids:
        selected = values[np.asarray(groups) == group]
        by_group.append(float(selected[np.isfinite(selected)].mean()) if np.isfinite(selected).any() else None)
    defined = [value for value in by_group if value is not None]
    return {"compound_macro_pearson": float(np.mean(defined)) if defined else None,
            "defined_conditions": int(np.isfinite(values).sum()), "total_conditions": len(values),
            "defined_compounds": len(defined), "total_compounds": len(group_ids),
            "group_ids": group_ids, "group_pearson": by_group}


def run_benchmark(data, out_dir, *, outer_folds=5, inner_folds=3,
                  alphas=None, seed=20261107, n_boot=5000, compound_features=None,
                  absolute_concentrations=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    chemical, molecular_provenance, exposure_provenance = None, None, None
    if compound_features is not None:
        data, chemical, molecular_provenance = align_compound_features(data, compound_features)
    if absolute_concentrations:
        data, exposure_provenance = absolute_concentration_data(data)
        if chemical is not None:
            chemical = _chemical_rows(chemical, exposure_provenance["original_row_indices"])
            molecular_provenance["actual_model_compounds_after_exposure_filter"] = sorted(set(data.compounds.tolist()))
    if alphas is None:
        alphas = ABSOLUTE_ALPHAS if absolute_concentrations else DEFAULT_ALPHAS
    columns = context_columns(data)
    methods = CHEMICAL_METHODS if chemical is not None else METHODS
    prior_name, source_name = methods[:2]
    wrong_source_name = "wrong_" + source_name
    source, target = np.asarray(data.source_delta, dtype=float), np.asarray(data.target_delta, dtype=float)
    context = np.asarray(data.context[:, columns], dtype=float)
    if (source.shape != target.shape or len(source) != len(context)
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or not np.isfinite(context).all()):
        raise ValueError("Finite source/target arrays must share the fixed full ortholog axis")
    if any(alpha <= 0 or not np.isfinite(alpha) for alpha in alphas) or not alphas:
        raise ValueError("A fixed positive alpha grid is required")
    folds = group_folds(data.compounds, outer_folds, seed)
    strata = np.asarray([str(row["time_h"]) + "|" + row["dose_level"] for row in data.metadata])
    code_files = [Path(__file__), ROOT / "src/cross_species/feedback_data.py",
                  ROOT / "src/cross_species/feedback_evaluation.py"]
    specification = {
        "scope": "Retrospective nested compound CV; prior project outcomes were examined. Not preregistered, prospective, or externally validated.",
        "outer_folds": outer_folds, "inner_folds": inner_folds, "seed": seed, "alpha_grid": list(alphas),
        "rows": len(source), "compounds": len(set(data.compounds.tolist())), "target_genes": target.shape[1],
        "gene_axis": "All supplied one-to-one orthologs; identical full human gene axis in all folds and methods; no target-variance gene selection.",
        "kernel": "Train-only column z-scores; Ks=XsXsT/p, Kc=XcXcT/q; context=Kc; source+context=.5Ks+.5Kc. Human targets retain original gene-space units with a training-gene intercept.",
        "calibrated_source": "Training human gene mean + one global beta times (source delta - training source gene mean); beta minimizes training gene-space MSE.",
        "primary": "Compound-macro full-gene delta-response MSE; delta=%s MSE minus %s MSE." % (prior_name, source_name),
        "secondary": "Per-condition full response Pearson and training-human-mean residual Pearson; undefined constant-vector correlations remain missing.",
        "inner_selection": "Each model has the same alpha grid, outer-training labels, target axis, and group folds; minimum pooled inner compound-macro MSE. No outer-label selection.",
        "training_objective": "Row-weighted ridge MSE; evaluation weights compounds equally.",
        "wrong_source": "Source response permuted within outer-test time/dose strata using IDs and seed only; same-compound donors ineligible. Matched/incorrect-source comparison uses identical eligible rows.",
        "context_inputs": data.context_feature_ids[columns].tolist(),
        "input_hashes": data.input_hashes, "data_warnings": list(data.warnings),
        "code_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in code_files},
        "interval_scope": "Compound resampling of fixed OOF predictions. Overlapping fold training creates dependence; no refitting, donor uncertainty, or reliable population coverage guarantee.",
        "limitations": ["One animal species and one broad liver in-vitro system; species selection is not evaluated.",
                        "Context has no chemical structure or mechanism descriptors; benefit is relative to this specified context baseline.",
                        "Historical joint RMA and condition-level replicate means remain source-data limitations.",
                        "Nominal dose categories may use different absolute concentrations across species; inspect concentration_match_status."],
    }
    if chemical is not None:
        specification["molecular_features"] = molecular_provenance
        specification["kernel"] = (
            "Descriptor NaNs use training-column medians (all-missing training columns use zero); descriptors/context then training z-scores. "
            "Kmol=.5 Tanimoto(PubChem881)+.5 RBF(descriptors); Kctx=RBF(context); RBF=exp(-squared distance/(2*dimension)). "
            "Kprior=(Kmol+Kctx+Kmol*Kctx)/3, with a fixed molecule-context interaction. "
            "Kprior and count-normalized train-z-scored rat linear Ks are separately training-centered and divided by training mean diagonal. "
            "Chemical-only prior uses Kprior; source+same-prior=.5 Kprior+.5 Ks. All weights/bandwidth rules fixed before outer evaluation.")
        specification["limitations"][1] = (
            "PubChem fingerprints/physchem form the specified molecular prior; no target annotation or pretrained chemical model is included. "
            "Verified complete-compound cohort differs from the full cohort and must be compared within cohort.")
    if exposure_provenance is not None:
        specification["absolute_concentrations"] = exposure_provenance
        specification["wrong_source"] = (
            "Auxiliary nominal-stratum source mismatch: permute within outer-test time/nominal-dose strata. "
            "Absolute planned concentrations can differ between donor and query; this is not a pure identity comparison at identical full exposure context. "
            "Desired query context remains fixed. Main paired model comparison uses the identical full exposure context in both arms.")
        specification["limitations"][3] = (
            "Both planned source and human molar concentrations are supplied to both arms; known different concentrations are retained. "
            "The incremental input is a measured rat transcriptome. Nominal concentrations do not establish equal intracellular exposure.")
    save_json(out_dir / "specification.json", specification)
    if exposure_provenance is not None:
        with (out_dir / "absolute_concentration_metadata.tsv").open("w", newline="") as handle:
            fields = list(exposure_provenance["retained_metadata"][0])
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(exposure_provenance["retained_metadata"])
    snapshot = out_dir / "code_snapshot"
    snapshot.mkdir()
    for path in code_files:
        shutil.copy2(path, snapshot / path.name)
    with (out_dir / "fold_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "compound", "outer_fold", "context_stratum", "concentration_match_status"])
        writer.writerows([data.sample_ids[i], data.compounds[i], folds[i], strata[i],
                          data.metadata[i].get("concentration_match_status", "unknown")] for i in range(len(source)))
    predictions = {name: np.empty_like(target, dtype=np.float32) for name in [*methods, wrong_source_name]}
    training_mean = np.empty_like(target, dtype=np.float32)
    donor_ids, wrong_eligible = np.empty(len(source), dtype=data.sample_ids.dtype), np.zeros(len(source), dtype=bool)
    fold_results, started = [], time.monotonic()
    for fold in range(outer_folds):
        fold_start = time.monotonic()
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        print("outer fold %d/%d: %d train/%d test conditions" % (fold + 1, outer_folds, len(train), len(test)), flush=True)
        predicted, selection, state = fit_outer_fold(source, context, target, data.compounds,
                                                    data.sample_ids, strata, train, test,
                                                    alphas, inner_folds, seed + 1009 * (fold + 1), chemical=chemical)
        for name, values in predicted.items():
            predictions[name][test] = values
        training_mean[test] = state["target_mean"]
        wrong_eligible[test] = state["wrong_source_eligible"]
        donor_ids[test] = data.sample_ids[state["wrong_donor_indices"]]
        np.savez_compressed(out_dir / ("fold_%d_transforms_models.npz" % fold),
                            **state, human_gene_ids=data.feature_ids,
                            source_gene_ids=data.source_feature_ids,
                            train_sample_ids=data.sample_ids[train], test_sample_ids=data.sample_ids[test],
                            context_feature_ids=data.context_feature_ids[columns])
        np.savez_compressed(out_dir / ("fold_%d_predictions.npz" % fold),
                            sample_ids=data.sample_ids[test], compounds=data.compounds[test],
                            truth=target[test].astype(np.float32), **{name: value.astype(np.float32) for name, value in predicted.items()})
        record = {"fold": fold, "train_compounds": len(set(data.compounds[train].tolist())),
                  "test_compounds": len(set(data.compounds[test].tolist())), **selection,
                  "seconds": time.monotonic() - fold_start}
        fold_results.append(record)
        save_json(out_dir / ("fold_%d_selection.json" % fold), record)
        print("outer fold %d frozen; selected %s; %.1f seconds" % (fold + 1, selection["selected_alphas"], record["seconds"]), flush=True)
        # No outer performance is printed while folds are being fitted.
    metrics, correlations = {}, {}
    for name in methods:
        metrics[name] = compound_macro_loss(predictions[name], target, data.compounds)
        response = row_pearson(predictions[name], target)
        residual = row_pearson(predictions[name] - training_mean, target - training_mean)
        correlations[name] = {"response": correlation_summary(response, data.compounds),
                              "train_mean_residual": correlation_summary(residual, data.compounds)}
        np.savez_compressed(out_dir / (name + "_correlations.npz"), response_pearson=response,
                            train_mean_residual_pearson=residual)
    primary = paired_compound_bootstrap(predictions[prior_name], predictions[source_name],
                                        target, data.compounds, n_boot=n_boot, seed=seed)
    primary["inference_scope"] = specification["interval_scope"]
    wrong_comparison = None
    if len(set(data.compounds[wrong_eligible].tolist())) >= 2:
        wrong_comparison = paired_compound_bootstrap(predictions[wrong_source_name][wrong_eligible],
                                                    predictions[source_name][wrong_eligible],
                                                    target[wrong_eligible], data.compounds[wrong_eligible],
                                                    n_boot=n_boot, seed=seed + 1)
        wrong_comparison["inference_scope"] = specification["interval_scope"]
    report = {**specification, "metrics": metrics, "correlations": correlations,
              "primary_gain_from_source": primary, "matched_vs_wrong_source_gain": wrong_comparison,
              "wrong_source_eligible_conditions": int(wrong_eligible.sum()),
              "fold_results": fold_results, "elapsed_seconds": time.monotonic() - started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "thread_limits": {key: os.environ.get(key) for key in
                                                ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}}
    np.savez_compressed(out_dir / "oof_predictions.npz", sample_ids=data.sample_ids,
                        compounds=data.compounds, outer_fold=folds,
                        human_gene_ids=data.feature_ids, source_gene_ids=data.source_feature_ids,
                        truth=target.astype(np.float32), fold_training_human_mean=training_mean,
                        wrong_source_donor_ids=donor_ids, wrong_source_eligible=wrong_eligible,
                        **predictions)
    save_json(out_dir / "report.json", report)
    with (out_dir / "compound_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["compound", *[name + "_mse" for name in methods], "source_gain"])
        group_order = metrics[prior_name]["group_ids"]
        for i, group in enumerate(group_order):
            values = [metrics[name]["group_losses"][i] for name in methods]
            writer.writerow([group, *values, values[0] - values[1]])
    print(json.dumps(plain({"compounds": specification["compounds"], "genes": target.shape[1],
                           "primary_gain": primary, "elapsed_seconds": report["elapsed_seconds"]}), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/tg_gates_rat_human_v1/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/feedback_cache"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--matched-concentration", action="store_true")
    parser.add_argument("--compound-features", type=Path)
    parser.add_argument("--absolute-concentrations", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=args.seed,
                              require_matched_concentration=args.matched_concentration)
    run_benchmark(data, args.out_dir, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                  alphas=args.alphas, seed=args.seed, n_boot=args.n_boot, compound_features=args.compound_features,
                  absolute_concentrations=args.absolute_concentrations)


if __name__ == "__main__":
    main()
