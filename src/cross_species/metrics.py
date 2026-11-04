"""Metrics that do not depend on a heavy statistics stack."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def _finite_pair(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = _finite_pair(x, y)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = _finite_pair(x, y)
    if len(a) < 2:
        return float("nan")
    return pearson(_rankdata(a), _rankdata(b))


def roc_auc(y_true: Iterable[float], scores: Iterable[float]) -> float:
    y, s = _finite_pair(y_true, scores)
    labels = y > 0
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true: Iterable[float], scores: Iterable[float]) -> float:
    y, s = _finite_pair(y_true, scores)
    labels = y > 0
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ordered = labels[order]
    cumulative = np.cumsum(ordered)
    positions = np.arange(1, len(ordered) + 1)
    return float((cumulative[ordered] / positions[ordered]).sum() / positives)


def task_metrics(y_true: Mapping[str, np.ndarray], y_pred: Mapping[str, np.ndarray], top_k: int = 500) -> dict:
    """Macro-average task metrics and flattened metrics."""

    per_task = []
    flat_true = []
    flat_pred = []
    for task_id in sorted(y_true):
        actual = np.asarray(y_true[task_id], dtype=float)
        predicted = np.asarray(y_pred[task_id], dtype=float)
        if actual.shape != predicted.shape:
            raise ValueError(f"shape mismatch for task {task_id}: {actual.shape} vs {predicted.shape}")
        top = np.argsort(-np.abs(actual))[: min(top_k, len(actual))]
        per_task.append(
            {
                "task_id": task_id,
                "pearson_top_k": pearson(actual[top], predicted[top]),
                "spearman_top_k": spearman(actual[top], predicted[top]),
                "direction_auroc": roc_auc(actual, predicted),
                "direction_auprc": average_precision(actual, predicted),
            }
        )
        flat_true.append(actual)
        flat_pred.append(predicted)
    if not per_task:
        return {"n_tasks": 0}
    top_p = [r["pearson_top_k"] for r in per_task]
    top_s = [r["spearman_top_k"] for r in per_task]
    auroc = [r["direction_auroc"] for r in per_task]
    auprc = [r["direction_auprc"] for r in per_task]
    flat_true_arr = np.concatenate(flat_true)
    flat_pred_arr = np.concatenate(flat_pred)
    return {
        "n_tasks": len(per_task),
        "pearson_top_k_macro": float(np.nanmean(top_p)),
        "spearman_top_k_macro": float(np.nanmean(top_s)),
        "direction_auroc_macro": float(np.nanmean(auroc)),
        "direction_auprc_macro": float(np.nanmean(auprc)),
        "pearson_flat": pearson(flat_true_arr, flat_pred_arr),
        "spearman_flat": spearman(flat_true_arr, flat_pred_arr),
        "per_task": per_task,
    }


def bootstrap_task_metrics(
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    *,
    top_k: int = 500,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Resample complete tasks and summarize metric uncertainty.

    Cells or genes are never resampled here.  With very few tasks the interval
    is intentionally coarse; the result records ``n_tasks`` so callers cannot
    mistake it for a cell-level confidence interval.
    """
    task_ids = sorted(set(y_true) & set(y_pred))
    if not task_ids:
        return {"n_tasks": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    base = task_metrics(y_true, y_pred, top_k=top_k)
    per_task = base["per_task"]
    metric_names = (
        "pearson_top_k_macro",
        "spearman_top_k_macro",
        "direction_auroc_macro",
        "direction_auprc_macro",
        "pearson_flat",
        "spearman_flat",
    )
    per_task_names = {
        "pearson_top_k_macro": "pearson_top_k",
        "spearman_top_k_macro": "spearman_top_k",
        "direction_auroc_macro": "direction_auroc",
        "direction_auprc_macro": "direction_auprc",
    }
    draws = {name: [] for name in metric_names}
    # Macro metrics can be bootstrapped from their already computed task-level
    # values.  This avoids re-ranking every gene thousands of times.
    task_metric_arrays = {
        name: np.asarray([row[field] for row in per_task], dtype=float)
        for name, field in per_task_names.items()
    }
    for _ in range(max(1, n_boot)):
        sampled = rng.integers(0, len(task_ids), size=len(task_ids))
        for name, values in task_metric_arrays.items():
            draws[name].append(float(np.nanmean(values[sampled])))
        flat_true = np.concatenate([np.asarray(y_true[task_ids[i]], dtype=float) for i in sampled])
        flat_pred = np.concatenate([np.asarray(y_pred[task_ids[i]], dtype=float) for i in sampled])
        draws["pearson_flat"].append(pearson(flat_true, flat_pred))
        # Ties are uncommon for response vectors; ordinal ranks make this
        # bootstrap fast while the primary exact Spearman remains in results.
        draws["spearman_flat"].append(
            pearson(np.argsort(np.argsort(flat_true)), np.argsort(np.argsort(flat_pred)))
        )
    summary = {"n_tasks": len(task_ids), "n_boot": max(1, n_boot), "seed": seed}
    for name, values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        summary[name] = {
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "lower_2.5": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
            "median": float(np.median(finite)) if len(finite) else float("nan"),
            "upper_97.5": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        }
    return summary
