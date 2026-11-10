"""Partition-local animal pairing controls for frozen compound folds.

Fit functions never inspect the outer query human response. The original
chemical prior and fixed half-prior/half-animal kernel are retained.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_feedback_nested_benchmark as legacy

DEFAULT_ALPHAS = (.001, .01, .1, 1., 10., 100., 1000.)
ARMS = ("prior", "correct_fit_correct_query", "correct_fit_wrong_query",
        "wrong_fit_correct_query", "wrong_fit_wrong_query")
UNITS = {"M": Decimal("1"), "mM": Decimal(".001"), "uM": Decimal(".000001"),
         "μM": Decimal(".000001"), "µM": Decimal(".000001"), "nM": Decimal(".000000001")}


def _decimal(value, label, *, positive=False):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("Invalid decimal " + label) from error
    if not number.is_finite() or (positive and number <= 0):
        raise ValueError("Nonfinite or nonpositive " + label)
    return number


def exact_strata(metadata, regime):
    """Exact declared units, without floating log-dose keys or dose rounding."""
    if regime not in {"source_exposure", "paired_exposure"}:
        raise ValueError("Unknown exposure regime")
    strata = []
    for row in metadata:
        values = [_decimal(row["time_h"], "time_h")]
        for species in (("rat",) if regime == "source_exposure" else ("rat", "human")):
            unit = str(row[species + "_dose_unit"]).strip()
            if unit not in UNITS:
                raise ValueError("Unknown declared molar unit: " + unit)
            values.append(_decimal(row[species + "_dose_value"], species + " dose", positive=True) * UNITS[unit])
        strata.append("|".join(format(value.normalize(), "f") for value in values))
    return np.asarray(strata)


def stable_seed(seed, regime, outer, inner, role):
    payload = json.dumps([int(seed), str(regime), int(outer), None if inner is None else int(inner), str(role)],
                         separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _derangements(n):
    a, b = 1, 0
    for k in range(2, n + 1):
        a, b = b, (k - 1) * (a + b)
    return a if n == 0 else b


def eligible_partition(indices, strata, compounds):
    """Keep whole strata admitting a different-compound bijection.

    Non-derangeable strata are excluded in full, identically for every arm.
    In the frozen cohort each compound appears at most once in each stratum.
    """
    indices = np.asarray(indices, dtype=int)
    strata, compounds = np.asarray(strata), np.asarray(compounds)
    if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
        raise ValueError("Partition row indices must be unique")
    if np.any(indices < 0) or np.any(indices >= len(strata)) or len(strata) != len(compounds):
        raise ValueError("Invalid partition index axis")
    eligible, records = [], []
    for key in sorted(set(strata[indices].tolist())):
        rows = indices[strata[indices] == key]
        counts = Counter(compounds[rows].tolist())
        admitted = len(rows) >= 2 and max(counts.values()) <= len(rows) // 2
        records.append({"stratum": str(key), "indices": rows.tolist(), "size": len(rows),
                        "distinct_compounds": len(counts), "eligible": admitted,
                        "all_compounds_unique": len(counts) == len(rows),
                        "possible_derangements": str(_derangements(len(rows))) if len(counts) == len(rows) else None,
                        "reason": "admitted" if admitted else "no_full_different_compound_bijection"})
        if admitted:
            eligible.extend(rows.tolist())
    admitted_set = set(eligible)
    kept = np.asarray([index for index in indices if index in admitted_set], dtype=int)
    return kept, {"input_indices": indices.tolist(), "kept_indices": kept.tolist(),
                  "excluded_indices": [int(index) for index in indices if index not in admitted_set],
                  "strata": records,
                  "kept_compounds": len(set(compounds[kept].tolist())) if len(kept) else 0,
                  "rows_in_size_two_strata": sum(row["size"] for row in records if row["eligible"] and row["size"] == 2)}


def derangement_map(indices, strata, compounds, seed):
    """Random full permutations, rejection sampled; no cyclic-shift fallback."""
    indices = np.asarray(indices, dtype=int)
    strata, compounds = np.asarray(strata), np.asarray(compounds)
    kept, eligibility = eligible_partition(indices, strata, compounds)
    if not np.array_equal(kept, indices):
        raise ValueError("Permutation partition includes ineligible rows")
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(indices), dtype=int)
    attempts = []
    for record in eligibility["strata"]:
        local = np.flatnonzero(strata[indices] == record["stratum"])
        for attempt in range(1, 1001):
            candidate = rng.permutation(local)
            if np.all(compounds[indices[local]] != compounds[indices[candidate]]):
                permutation[local] = candidate
                attempts.append({"stratum": record["stratum"], "attempts": attempt, "size": len(local)})
                break
        else:
            raise RuntimeError("No accepted full derangement in1000 attempts: " + record["stratum"])
    if (not np.array_equal(np.sort(permutation), np.arange(len(indices)))
            or not np.array_equal(strata[indices], strata[indices[permutation]])
            or np.any(compounds[indices] == compounds[indices[permutation]])):
        raise AssertionError("Invalid partition-local source derangement")
    return permutation, {"seed": int(seed), "local_permutation": permutation.tolist(),
                         "query_indices": indices.tolist(), "donor_indices": indices[permutation].tolist(),
                         "stratum_attempts": attempts}


def mapping_diversity(permutations, indices, strata):
    indices, strata = np.asarray(indices), np.asarray(strata)
    arrays = [np.asarray(p, dtype=int) for p in permutations]
    return {"draws": len(arrays), "unique_full_mappings": len({tuple(p.tolist()) for p in arrays}),
            "strata": [{"stratum": str(key), "size": int(np.sum(strata[indices] == key)),
                         "unique_realized_mappings": len({tuple(indices[p[strata[indices] == key]].tolist()) for p in arrays})}
                        for key in sorted(set(strata[indices].tolist()))],
            "scope": "Repeated mappings and random seeds do not add independent biological tasks."}


@dataclass
class KernelPartition:
    prior_train: torch.Tensor
    source_train: torch.Tensor
    prior_query: torch.Tensor
    source_query: torch.Tensor
    transforms: dict

    def kernels(self, fit_permutation=None, query_permutation=None, use_source=True):
        if not use_source:
            return self.prior_train, self.prior_query
        p = torch.arange(len(self.source_train), device=self.source_train.device) if fit_permutation is None else torch.as_tensor(fit_permutation, device=self.source_train.device, dtype=torch.long)
        q = torch.arange(len(self.source_query), device=self.source_query.device) if query_permutation is None else torch.as_tensor(query_permutation, device=self.source_query.device, dtype=torch.long)
        return (.5 * (self.prior_train + self.source_train[p][:, p]),
                .5 * (self.prior_query + self.source_query[q][:, p]))


def prepare_partition(source, context, fingerprints, descriptors, train, query, device="cpu"):
    train, query = np.asarray(train, dtype=int), np.asarray(query, dtype=int)
    if not len(train) or not len(query):
        raise ValueError("Empty fit/query partition")
    inputs = legacy.ChemicalKernelInputs(source[train], context[train], fingerprints[train], descriptors[train])
    prior_query = inputs.prior_cross(context[query], fingerprints=fingerprints[query], descriptors=descriptors[query])
    standardized = (np.asarray(source[query]) - inputs.source_mean) / inputs.source_scale
    source_query = inputs.source_center.cross(standardized @ inputs.source_train.T / inputs.source_train.shape[1])
    tensor = lambda value: torch.as_tensor(value, device=device, dtype=torch.float64)
    return KernelPartition(tensor(inputs.prior_center.train), tensor(inputs.source_center.train),
                           tensor(prior_query), tensor(source_query), inputs.saved_transforms())


class TorchRidge:
    """Float64 dual ridge with the unchanged legacy numerical-null rule."""

    def __init__(self, train_kernel, target_train):
        self.target_train = torch.as_tensor(target_train, device=train_kernel.device, dtype=torch.float64)
        self.target_mean = self.target_train.mean(dim=0)
        values, vectors = torch.linalg.eigh(.5 * (train_kernel + train_kernel.T))
        threshold = max(1e-12, float(torch.clamp(values.max(), min=0).item()) * 1e-12)
        keep = values > threshold
        self.values, self.vectors = values[keep], vectors[:, keep]
        self.eigenvalues, self.eigenvectors = self.values, self.vectors
        self.projected_targets = self.vectors.T @ (self.target_train - self.target_mean)

    def predict(self, cross_kernel, alpha):
        if alpha <= 0:
            raise ValueError("Positive ridge alpha required")
        return ((cross_kernel @ self.vectors) / (self.values + alpha)) @ self.projected_targets + self.target_mean

    def row_losses(self, cross_kernel, target_valid, alphas):
        centered = torch.as_tensor(target_valid, device=self.target_mean.device, dtype=torch.float64) - self.target_mean
        projected = cross_kernel @ self.vectors
        gram = self.projected_targets @ self.projected_targets.T
        cross = centered @ self.projected_targets.T
        truth_norm = centered.square().sum(dim=1)
        output = []
        for alpha in alphas:
            if alpha <= 0:
                raise ValueError("Positive ridge alpha required")
            weights = projected / (self.values + alpha)
            loss = ((weights @ gram) * weights).sum(dim=1) - 2 * (weights * cross).sum(dim=1) + truth_norm
            output.append(torch.clamp(loss / centered.shape[1], min=0))
        return torch.stack(output).cpu().numpy()

    def coefficients(self, alpha):
        return self.vectors @ (self.projected_targets / (self.values + alpha)[:, None])


def _numpy(value):
    return value.detach().cpu().numpy().copy()


def validate_bundle(bundle, metadata):
    source, target = np.asarray(bundle["source"]), np.asarray(bundle["target"])
    n = len(source)
    if source.ndim != 2 or source.shape != target.shape or not n or not np.isfinite(source).all():
        raise ValueError("Invalid paired source/target axes")
    for name in ("sample_ids", "compounds", "outer_fold"):
        if np.asarray(bundle[name]).shape != (n,):
            raise ValueError("Invalid row axis: " + name)
    if len(set(bundle["sample_ids"].tolist())) != n or len(metadata) != n:
        raise ValueError("Duplicate sample identity or metadata size mismatch")
    for index, row in enumerate(metadata):
        if str(row.get("sample_id", row.get("pair_id"))) != str(bundle["sample_ids"][index]) or row["compound"] != bundle["compounds"][index]:
            raise ValueError("Bundle and metadata sample identity differ")
    folds = np.asarray(bundle["outer_fold"])
    unique = sorted(set(folds.tolist()))
    if unique != list(range(len(unique))) or len(unique) < 2:
        raise ValueError("Outer fold labels must be contiguous and nontrivial")
    inner = np.asarray(bundle["inner_fold_by_outer"])
    if inner.shape != (len(unique), n):
        raise ValueError("Invalid frozen inner-fold array")
    for outer in unique:
        if np.any(inner[outer, folds == outer] != -1) or np.any(inner[outer, folds != outer] < 0):
            raise ValueError("Inner fold mask disagrees with outer holdout")
        if len(set(inner[outer, folds != outer].tolist())) < 2:
            raise ValueError("At least two inner folds required")
    for compound in set(bundle["compounds"].tolist()):
        rows = np.asarray(bundle["compounds"]) == compound
        if len(set(folds[rows].tolist())) != 1:
            raise ValueError("A compound crosses outer folds")
        for outer in unique:
            if len(set(inner[outer, rows].tolist())) != 1:
                raise ValueError("A compound crosses inner folds")
    # Outer held human values are deliberately not examined by this validator.
    for name in ("context", "fingerprints", "descriptors"):
        if np.asarray(bundle[name]).ndim != 2 or len(bundle[name]) != n:
            raise ValueError("Invalid covariate shape: " + name)
    return unique


def _select(losses, compounds, alphas):
    scored = np.isfinite(losses).all(axis=0)
    if not scored.any():
        raise ValueError("No jointly eligible inner validation rows")
    groups = np.asarray(compounds)
    unique = sorted(set(groups[scored].tolist()))
    macro = np.asarray([np.mean([row[scored & (groups == group)].mean() for group in unique]) for row in losses])
    position = int(np.argmin(macro))
    return float(alphas[position]), {"trials": [{"alpha": float(a), "compound_macro_mse": float(value)} for a, value in zip(alphas, macro)],
                                    "validation_indices": np.flatnonzero(scored).tolist(),
                                    "validation_compounds": len(unique)}


def legacy_permutation_check(bundle, partition, train, query, fit_permutation, query_permutation, alpha, *, tolerance=1e-8):
    """Actually rebuild permuted inputs and solve with the independent legacy CPU path."""
    source, context = bundle["source"], bundle["context"]
    p, q = np.asarray(fit_permutation), np.asarray(query_permutation)
    inputs = legacy.ChemicalKernelInputs(source[train[p]], context[train], bundle["fingerprints"][train], bundle["descriptors"][train])
    train_kernel = inputs.train_kernels()["source_plus_chemical_context"]
    query_kernel = inputs.cross(source[query[q]], context[query], fingerprints=bundle["fingerprints"][query], descriptors=bundle["descriptors"][query])["source_plus_chemical_context"]
    actual_train, actual_query = partition.kernels(p, q)
    kernel_delta = max(float(np.max(np.abs(_numpy(actual_train) - train_kernel))),
                       float(np.max(np.abs(_numpy(actual_query) - query_kernel))))
    # This helper must only receive a query whose targets are permitted fitting/validation labels.
    cpu = legacy.SpectralRidge(train_kernel, bundle["target"][train])
    gpu = TorchRidge(actual_train, bundle["target"][train])
    prediction_delta = float(np.max(np.abs(cpu.predict(query_kernel, alpha) - _numpy(gpu.predict(actual_query, alpha)))))
    loss_delta = float(np.max(np.abs(cpu.row_losses(query_kernel, bundle["target"][query], [alpha]) - gpu.row_losses(actual_query, bundle["target"][query], [alpha]))))
    if max(kernel_delta, prediction_delta, loss_delta) > tolerance:
        raise AssertionError("Legacy permutation equivalence failed")
    return {"kernel_max_abs_difference": kernel_delta, "prediction_max_abs_difference": prediction_delta,
            "row_loss_max_abs_difference": loss_delta, "tolerance": tolerance, "passed": True}


def fit_outer(bundle, metadata, outer, regime, seeds, device="cpu", alphas=DEFAULT_ALPHAS):
    """Freeze one outer fold's five arms without reading its human query values."""
    validate_bundle(bundle, metadata)
    if not seeds or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("At least one unique permutation seed required")
    if any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Positive finite alpha grid required")
    source = np.asarray(bundle["source"], dtype=np.float64)
    context = np.asarray(bundle["context"], dtype=np.float64)
    compounds = np.asarray(bundle["compounds"])
    strata = exact_strata(metadata, regime)
    frozen_outer, frozen_inner = np.asarray(bundle["outer_fold"]), np.asarray(bundle["inner_fold_by_outer"])[outer]
    base_train, base_test = np.flatnonzero(frozen_outer != outer), np.flatnonzero(frozen_outer == outer)
    train, train_audit = eligible_partition(base_train, strata, compounds)
    test, test_audit = eligible_partition(base_test, strata, compounds)
    if not len(train) or not len(test):
        raise ValueError("Outer partition has no eligible fit or query rows")
    inner_parts = []
    prior_losses = np.full((len(alphas), len(source)), np.nan)
    correct_losses = np.full_like(prior_losses, np.nan)
    inner_audit = []
    for inner in sorted(set(frozen_inner[base_train].tolist())):
        ti, ta = eligible_partition(np.flatnonzero((frozen_outer != outer) & (frozen_inner != inner)), strata, compounds)
        vi, va = eligible_partition(np.flatnonzero((frozen_outer != outer) & (frozen_inner == inner)), strata, compounds)
        if not len(ti) or not len(vi):
            raise ValueError("Inner partition has no eligible fit/validation rows")
        part = prepare_partition(source, context, bundle["fingerprints"], bundle["descriptors"], ti, vi, device)
        yi, yv = bundle["target"][ti], bundle["target"][vi]
        for use_source, losses in ((False, prior_losses), (True, correct_losses)):
            kt, kv = part.kernels(use_source=use_source)
            losses[:, vi] = TorchRidge(kt, yi).row_losses(kv, yv, alphas)
        inner_parts.append((inner, ti, vi, part))
        inner_audit.append({"fold": int(inner), "fit": ta, "validation": va})
    prior_alpha, prior_trials = _select(prior_losses, compounds, alphas)
    correct_alpha, correct_trials = _select(correct_losses, compounds, alphas)
    outer_part = prepare_partition(source, context, bundle["fingerprints"], bundle["descriptors"], train, test, device)
    yt = bundle["target"][train]
    kp, qp = outer_part.kernels(use_source=False)
    kc, qc = outer_part.kernels()
    prior_model, correct_model = TorchRidge(kp, yt), TorchRidge(kc, yt)
    prior_prediction = _numpy(prior_model.predict(qp, prior_alpha))
    correct_prediction = _numpy(correct_model.predict(qc, correct_alpha))
    predictions = {"prior": np.broadcast_to(prior_prediction, (len(seeds), *prior_prediction.shape)),
                   "correct_fit_correct_query": np.broadcast_to(correct_prediction, (len(seeds), *correct_prediction.shape))}
    for name in ARMS[2:]:
        predictions[name] = np.empty((len(seeds), len(test), source.shape[1]), dtype=np.float64)
    state = {"target_mean": _numpy(correct_model.target_mean), "prior_coefficients": _numpy(prior_model.coefficients(prior_alpha)),
             "correct_coefficients": _numpy(correct_model.coefficients(correct_alpha)), "transforms": outer_part.transforms,
             "wrong_coefficients": []}
    selections, mapping_records, checks = [], [], []
    outer_fit_maps, outer_query_maps = [], []
    inner_maps = {int(i): {"fit": [], "query": []} for i, *_ in inner_parts}
    for seed_position, seed in enumerate(seeds):
        null_losses = np.full_like(prior_losses, np.nan)
        seed_maps = {"seed": int(seed), "inner": []}
        for inner, ti, vi, part in inner_parts:
            p, pa = derangement_map(ti, strata, compounds, stable_seed(seed, regime, outer, inner, "inner_fit"))
            q, qa = derangement_map(vi, strata, compounds, stable_seed(seed, regime, outer, inner, "inner_query"))
            kt, kv = part.kernels(p, q)
            null_losses[:, vi] = TorchRidge(kt, bundle["target"][ti]).row_losses(kv, bundle["target"][vi], alphas)
            seed_maps["inner"].append({"fold": int(inner), "fit": pa, "query": qa})
            inner_maps[int(inner)]["fit"].append(p); inner_maps[int(inner)]["query"].append(q)
            if seed_position == 0 and len(checks) == 0:
                checks.append(legacy_permutation_check(bundle, part, ti, vi, p, q, 1.))
        alpha, trials = _select(null_losses, compounds, alphas)
        p, pa = derangement_map(train, strata, compounds, stable_seed(seed, regime, outer, None, "outer_fit"))
        q, qa = derangement_map(test, strata, compounds, stable_seed(seed, regime, outer, None, "outer_query"))
        seed_maps.update(outer_fit=pa, outer_query=qa)
        wrong_train, wrong_correct_query = outer_part.kernels(p, None)
        _, wrong_wrong_query = outer_part.kernels(p, q)
        _, correct_wrong_query = outer_part.kernels(None, q)
        wrong_model = TorchRidge(wrong_train, yt)
        predictions[ARMS[2]][seed_position] = _numpy(correct_model.predict(correct_wrong_query, correct_alpha))
        predictions[ARMS[3]][seed_position] = _numpy(wrong_model.predict(wrong_correct_query, alpha))
        predictions[ARMS[4]][seed_position] = _numpy(wrong_model.predict(wrong_wrong_query, alpha))
        state["wrong_coefficients"].append(_numpy(wrong_model.coefficients(alpha)))
        selections.append({"seed": int(seed), "wrong_alpha": alpha, "wrong_inner_selection": trials})
        mapping_records.append(seed_maps); outer_fit_maps.append(p); outer_query_maps.append(q)
    diversity = {"outer_fit": mapping_diversity(outer_fit_maps, train, strata),
                 "outer_query": mapping_diversity(outer_query_maps, test, strata), "inner": []}
    for inner, ti, vi, _ in inner_parts:
        diversity["inner"].append({"fold": int(inner), "fit": mapping_diversity(inner_maps[int(inner)]["fit"], ti, strata),
                                  "query": mapping_diversity(inner_maps[int(inner)]["query"], vi, strata)})
    return {"train_indices": train, "test_indices": test, "predictions": predictions,
            "selection": {"prior_alpha": prior_alpha, "correct_alpha": correct_alpha,
                          "prior_inner_selection": prior_trials, "correct_inner_selection": correct_trials, "per_seed": selections},
            "audit": {"outer": int(outer), "regime": regime, "outer_fit": train_audit, "outer_query": test_audit,
                      "inner": inner_audit, "donor_maps": mapping_records, "mapping_diversity": diversity,
                      "legacy_reference_checks": checks, "outer_human_used_in_fit": False,
                      "frozen_outer_fold": frozen_outer.tolist(), "frozen_inner_fold": frozen_inner.tolist()},
            "states": state}


def score_predictions(predictions, target, compounds, training_mean=None):
    """Terminal scoring of already frozen arrays; no fitting occurs here."""
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(compounds)
    if not np.isfinite(target).all():
        raise ValueError("Terminal human truth must be finite")
    unique = sorted(set(groups.tolist()))
    result = {}
    if training_mean is not None:
        training_mean = np.asarray(training_mean, dtype=np.float64)
        if training_mean.shape not in {(target.shape[1],), target.shape} or not np.isfinite(training_mean).all():
            raise ValueError("Invalid frozen training-human-mean reference")

    def correlations(values, truth):
        centered_p = values - values.mean(axis=2, keepdims=True)
        centered_y = truth - truth.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(centered_p ** 2, axis=2) * np.sum(centered_y ** 2, axis=1)[None])
        valid = (np.ptp(values, axis=2) > 0) & (np.ptp(truth, axis=1)[None] > 0) & (denominator > 0)
        output = np.full(denominator.shape, np.nan)
        np.divide(np.sum(centered_p * centered_y[None], axis=2), denominator, out=output, where=valid)
        return np.clip(output, -1., 1.)

    for arm, values in predictions.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != target.shape or not np.isfinite(values).all():
            raise ValueError("Invalid frozen predictions: " + arm)
        row_losses = np.mean((values - target[None]) ** 2, axis=2)
        group_losses = np.stack([row_losses[:, groups == group].mean(axis=1) for group in unique], axis=1)
        result[arm] = {"row_losses": row_losses, "compound_losses": group_losses,
                       "compound_ids": unique, "macro_mse": group_losses.mean(axis=1),
                       "response_pearson": correlations(values, target)}
        if training_mean is not None:
            result[arm]["train_mean_residual_pearson"] = correlations(values - training_mean, target - training_mean)
    return result
