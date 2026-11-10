"""Fixed, low-capacity source calibration for small multi-study panels.

Arrays have task x source x gene and task x gene axes. Source identity is selected
using historical labels only. Genes are features, not independent task replicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from cross_species.multisource_models import simplex_weights


def finite(value, ndim, name):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.ndim != ndim or any(size == 0 for size in value.shape) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite, nonempty {ndim}-D array")
    return value


def subsets(n):
    if not 1 <= n <= 8:
        raise ValueError("Exact subset enumeration supports one to eight sources")
    return [tuple(c) for k in range(1, n + 1) for c in combinations(range(n), k)]


@dataclass
class ScalarCalibrator:
    beta: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    mode: str
    fallback_sources: tuple

    @classmethod
    def fit(cls, source, target, *, mode="scaled"):
        x, y = finite(source, 3, "source"), finite(target, 2, "target")
        if y.shape != (x.shape[0], x.shape[2]) or mode not in {"scaled", "affine_gene_mean"}:
            raise ValueError("Source/target task and gene axes or mode are invalid")
        source_mean = x.mean(0) if mode == "affine_gene_mean" else np.zeros(x.shape[1:])
        target_mean = y.mean(0) if mode == "affine_gene_mean" else np.zeros(y.shape[1])
        xc, yc = x - source_mean, y - target_mean
        denominator = np.square(xc).sum(axis=(0, 2))
        numerator = np.einsum("tsg,tg->s", xc, yc)
        beta = np.zeros(x.shape[1], dtype=np.float64)
        valid = denominator > 1e-12
        beta[valid] = np.maximum(numerator[valid] / denominator[valid], 0.)
        return cls(beta.copy(), source_mean.copy(), target_mean.copy(), mode,
                   tuple(np.flatnonzero(~valid).tolist()))

    def predict(self, source):
        x = finite(source, 3, "source")
        if x.shape[1:] != self.source_mean.shape:
            raise ValueError("Source order/count or gene axis differs from training")
        return self.target_mean[None, None] + self.beta[None, :, None] * (x - self.source_mean[None])

    def state(self):
        return {"mode": self.mode, "beta": self.beta.copy(),
                "source_mean": self.source_mean.copy(), "target_mean": self.target_mean.copy(),
                "fallback_sources": list(self.fallback_sources)}


def historical_calibration_predictions(source, target, mode):
    x, y = finite(source, 3, "source"), finite(target, 2, "target")
    if len(x) < 2:
        raise ValueError("At least two historical tasks are needed for source selection")
    result = np.empty_like(x)
    records = []
    for held in range(len(x)):
        training = np.flatnonzero(np.arange(len(x)) != held)
        model = ScalarCalibrator.fit(x[training], y[training], mode=mode)
        result[held] = model.predict(x[held:held + 1])[0]
        records.append({"held_reference": held, "fit_references": training.tolist(),
                        "beta": model.beta.tolist(), "fallback_sources": list(model.fallback_sources)})
    return result, records


@dataclass
class HistoricalPanelModel:
    source_ids: tuple
    target_mean: np.ndarray
    calibrators: dict
    weights: dict
    selected_source: dict
    selection_losses: dict
    oof_records: dict
    target_scale: float
    ridge_multiplier: float
    training_tasks: int

    @classmethod
    def fit(cls, source, target, source_ids, *, ridge_multiplier=1.):
        x, y = finite(source, 3, "source"), finite(target, 2, "target")
        names = tuple(map(str, source_ids))
        if len(x) < 2 or y.shape != (len(x), x.shape[2]) or len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError("Need >=2 historical tasks, unique sources and matched feature axes")
        if not np.isfinite(ridge_multiplier) or ridge_multiplier <= 0:
            raise ValueError("A fixed positive shrinkage multiplier is required")
        scale = max(float(np.square(y).mean()), 1e-8)
        calibration = {mode: ScalarCalibrator.fit(x, y, mode=mode)
                       for mode in ["scaled", "affine_gene_mean"]}
        candidates, provenance = {"raw": x}, {}
        for mode in calibration:
            candidates[mode], provenance[mode] = historical_calibration_predictions(x, y, mode)
        losses = {mode: np.square(values - y[:, None]).mean(axis=(0, 2))
                  for mode, values in candidates.items()}
        chosen = {mode: int(np.argmin(value)) for mode, value in losses.items()}
        weights = {}
        for subset in subsets(len(names)):
            for mode, reference_prediction in candidates.items():
                full = np.zeros(len(names))
                full[list(subset)] = simplex_weights(reference_prediction[:, subset], y,
                                                     ridge=ridge_multiplier * scale)
                weights[(mode, subset)] = full
        return cls(names, y.mean(0).copy(), calibration, weights, chosen,
                   {key: value.copy() for key, value in losses.items()}, provenance,
                   scale, float(ridge_multiplier), len(x))

    def linear_estimators(self):
        """Coefficient/offset form for fixed-estimator measured-arm sensitivity."""
        count, genes = len(self.source_ids), len(self.target_mean)
        result = {"zero": (np.zeros(count), np.zeros(genes)),
                  "human_history_mean": (np.zeros(count), self.target_mean.copy())}
        for mode in ["raw", "scaled", "affine_gene_mean"]:
            if mode == "raw":
                beta, offsets = np.ones(count), np.zeros((count, genes))
            else:
                cal = self.calibrators[mode]
                beta = cal.beta
                offsets = cal.target_mean[None] - beta[:, None] * cal.source_mean
            for subset in subsets(count):
                suffix = "+".join(self.source_ids[i] for i in subset)
                equal = np.zeros(count); equal[list(subset)] = 1 / len(subset)
                for label, weights in [("equal", equal), ("shrink", self.weights[(mode, subset)])]:
                    result[f"{mode}/{label}/{suffix}"] = (weights * beta, weights @ offsets)
            chosen = np.zeros(count); chosen[self.selected_source[mode]] = 1.
            result[f"{mode}/history_selected"] = (chosen * beta, chosen @ offsets)
        return result

    def predict(self, source):
        x = finite(source, 3, "source")
        if x.shape[1] != len(self.source_ids) or x.shape[2] != len(self.target_mean):
            raise ValueError("Source identity or target gene axis mismatch")
        return {name: np.einsum("tsg,s->tg", x, coefficients) + offset
                for name, (coefficients, offset) in self.linear_estimators().items()}

    def choose_from_controls(self, source_controls, target_controls):
        """An explicitly control-informed rule, with no treated target input.

        All candidate source and target controls must already be available. The
        selection is not free source acquisition or a species-distance estimate.
        """
        source, target = finite(source_controls, 3, "source_controls"), finite(target_controls, 2, "target_controls")
        if source.shape[1:] != (len(self.source_ids), len(self.target_mean)) or target.shape != (len(source), source.shape[2]):
            raise ValueError("Control axes do not match fitted source/gene identities")
        x = source - source.mean(-1, keepdims=True)
        y = target - target.mean(-1, keepdims=True)
        denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)[:, None]
        similarity = np.full(denominator.shape, np.nan)
        np.divide(np.einsum("tsg,tg->ts", x, y), denominator, out=similarity, where=denominator > 0)
        indices = np.full(len(source), -1, dtype=int)
        for task, scores in enumerate(similarity):
            eligible = np.flatnonzero(np.isfinite(scores))
            if len(eligible):
                indices[task] = eligible[np.argmax(scores[eligible])]
        return indices, similarity

    def state(self):
        return {"source_ids": list(self.source_ids), "training_tasks": self.training_tasks,
                "ridge_multiplier": self.ridge_multiplier, "target_scale": self.target_scale,
                "calibrators": {key: value.state() for key, value in self.calibrators.items()},
                "weights": {mode + "/" + "+".join(self.source_ids[i] for i in subset): value.copy()
                            for (mode, subset), value in self.weights.items()},
                "selected_source_before_measurement": {key: self.source_ids[index] for key, index in self.selected_source.items()},
                "historical_selection_losses": self.selection_losses,
                "reference_crossfit": self.oof_records,
                "selection_warning": "Raw selection uses historical raw-copy errors. Scaled/affine selection and weights use historical leave-one-task-out predictions. Few historical tasks do not establish stable context-dependent selection."}


def evaluate_predictions(predictions, truth):
    y = finite(truth, 2, "truth")
    result = {}
    for name, value in predictions.items():
        prediction = finite(value, 2, name)
        if prediction.shape != y.shape:
            raise ValueError("Predictions and targets differ")
        xc, yc = prediction - prediction.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
        denom = np.sqrt(np.square(xc).sum(1) * np.square(yc).sum(1))
        correlation = np.full(len(y), np.nan)
        np.divide((xc * yc).sum(1), denom, out=correlation, where=denom > 0)
        errors = np.square(prediction - y).mean(1)
        result[name] = {"mse": errors, "macro_mse": float(errors.mean()),
                        "pearson": correlation, "pearson_defined": np.isfinite(correlation)}
    return result


def error_geometry(predictions, truth, source_ids):
    """Target-using terminal diagnostic; never supplied to prediction or selection."""
    x, y = finite(predictions, 3, "source_predictions"), finite(truth, 2, "truth")
    if y.shape != (len(x), x.shape[2]) or len(source_ids) != x.shape[1]:
        raise ValueError("Incompatible diagnostic axes")
    error = x - y[:, None]
    gram = np.einsum("tsg,tug->tsu", error, error) / x.shape[-1]
    rms = np.sqrt(np.square(x).mean(-1))
    return {"source_ids": list(source_ids), "error_inner_products": gram,
            "prediction_rms": rms, "human_rms": np.sqrt(np.square(y).mean(-1)),
            "interpretation": "Describes held-target error alignment after scoring; not an experiment-before-measurement rule or biological variance decomposition."}
