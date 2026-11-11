"""Compound-macro evaluation for cross-species predictions (NumPy only).

These functions consume frozen predictions. Bootstrap intervals condition on
the fitted models and resample compounds. The resampling unit is the compound.
"""
import numpy as np


def _finite_matrix(value, name):
    value = np.array(value, dtype=float, copy=True)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("%s must be a nonempty two-dimensional array" % name)
    if not np.isfinite(value).all():
        raise ValueError("%s must contain only finite values" % name)
    return value


def _group_rows(groups, n_rows):
    groups = np.asarray(groups, dtype=object)
    if groups.ndim != 1 or len(groups) != n_rows:
        raise ValueError("groups must contain one identifier per row")
    identifiers, positions = [], {}
    for row, value in enumerate(groups):
        value = value.item() if isinstance(value, np.generic) else value
        if value is None or (isinstance(value, (int, float)) and not np.isfinite(value)):
            raise ValueError("group identifiers cannot be missing or non-finite")
        try:
            hash(value)
        except TypeError as error:
            raise ValueError("group identifiers must be hashable") from error
        if value not in positions:
            identifiers.append(value)
            positions[value] = []
        positions[value].append(row)
    return identifiers, [np.asarray(positions[g], dtype=int) for g in identifiers]


def compound_macro_loss(pred_gene, truth_gene, groups):
    """Average gene MSE within rows, rows within compounds, then compounds equally."""
    predictions = _finite_matrix(pred_gene, "pred_gene")
    truth = _finite_matrix(truth_gene, "truth_gene")
    if predictions.shape != truth.shape:
        raise ValueError("prediction and truth shapes must be identical")
    identifiers, positions = _group_rows(groups, len(predictions))
    row_losses = np.square(predictions - truth).mean(axis=1)
    group_losses = np.asarray([row_losses[idx].mean() for idx in positions])
    return {"macro_loss": float(group_losses.mean()), "group_ids": identifiers,
            "group_losses": group_losses, "row_losses": row_losses,
            "group_sizes": np.asarray([len(idx) for idx in positions]),
            "n_groups": len(identifiers)}


def paired_compound_bootstrap(baseline_pred, method_pred, truth_gene, groups,
                               n_boot=2000, seed=0):
    """Positive delta means improvement. Resample the same groups for both methods.

    The small_n flag uses fewer than ten compounds as a reporting prompt, not as a
    power or validity threshold. Even larger sets may be underpowered or dependent.
    """
    baseline = compound_macro_loss(baseline_pred, truth_gene, groups)
    method = compound_macro_loss(method_pred, truth_gene, groups)
    if baseline["n_groups"] < 2:
        raise ValueError("paired compound bootstrap requires at least two groups")
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    deltas = baseline["group_losses"] - method["group_losses"]
    rng = np.random.default_rng(seed)
    # Chunk draws to keep memory bounded for larger test panels.
    bootstrap = np.empty(n_boot)
    for start in range(0, n_boot, 256):
        size = min(256, n_boot - start)
        idx = rng.integers(0, len(deltas), size=(size, len(deltas)))
        bootstrap[start:start + size] = deltas[idx].mean(axis=1)
    interval = np.percentile(bootstrap, [2.5, 97.5])
    return {"delta": float(deltas.mean()), "ci95": interval.tolist(),
            "baseline_macro_loss": baseline["macro_loss"],
            "method_macro_loss": method["macro_loss"],
            "group_ids": baseline["group_ids"], "group_deltas": deltas,
            "n_groups": len(deltas), "n_boot": int(n_boot), "seed": int(seed),
            "small_n": len(deltas) < 10, "small_n_threshold": 10,
            "inference_scope": "Paired compound resampling conditional on fitted models; "
                               "no refitting, donor-noise or training-set uncertainty."}


def _average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2.0)[inverse]


def _spearman_quality(scores, losses):
    x, y = _average_ranks(scores), _average_ranks(-losses)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denominator) if denominator > 0 else None


def _defined_mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _macro_metrics(row_metrics, positions):
    """Undefined correlations/pair comparisons remain undefined, with coverage counts."""
    grouped, macro, defined = {}, {}, {}
    for name, values in row_metrics.items():
        means = [_defined_mean([values[i] for i in idx]) for idx in positions]
        grouped[name] = means
        macro[name] = _defined_mean(means)
        defined[name] = sum(value is not None for value in means)
    return grouped, macro, defined


def candidate_reward_audit(scores, true_losses, groups, names=None, best_fixed_index=None):
    """Audit higher-is-better rewards against lower-is-better observed human losses.

    Ranking is computed within each row and macro-averaged over compounds. True-loss
    ties are excluded from pair accuracy; reward ties score one half. Spearman uses
    average ranks and is None for a constant reward or loss vector.

    Top-score ties select the first candidate in the caller's fixed order. The oracle
    uses human truth for diagnosis only. best_fixed_index must be chosen beforehand
    on calibration data; this function deliberately never selects it from test losses.
    Returned indices let the caller select/decode its original candidate predictions.
    """
    scores = _finite_matrix(scores, "scores")
    losses = _finite_matrix(true_losses, "true_losses")
    if scores.shape != losses.shape or scores.shape[1] < 2:
        raise ValueError("scores/losses must have equal (N,K) shapes with K >= 2")
    if (losses < 0).any():
        raise ValueError("true_losses must be nonnegative")
    n_rows, n_candidates = scores.shape
    identifiers, positions = _group_rows(groups, n_rows)
    names = (["candidate_%d" % i for i in range(n_candidates)] if names is None
             else list(names))
    if len(names) != n_candidates or any(not isinstance(n, str) for n in names):
        raise ValueError("names must contain one string per candidate")
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    if best_fixed_index is not None and (
            isinstance(best_fixed_index, bool)
            or not isinstance(best_fixed_index, (int, np.integer))
            or not 0 <= best_fixed_index < n_candidates):
        raise ValueError("best_fixed_index must be an in-range integer chosen on calibration")

    selected = scores.argmax(axis=1)
    oracle = losses.argmin(axis=1)
    rows = np.arange(n_rows)
    selected_loss, oracle_loss = losses[rows, selected], losses[rows, oracle]
    pair_accuracy, rank_correlation, comparable_pairs = [], [], []
    left, right = np.triu_indices(n_candidates, k=1)
    for reward, loss in zip(scores, losses):
        quality_difference = loss[right] - loss[left]
        reward_difference = reward[left] - reward[right]
        comparable = quality_difference != 0
        count = int(comparable.sum())
        comparable_pairs.append(count)
        if count:
            agreement = np.sign(quality_difference[comparable]) * np.sign(
                reward_difference[comparable])
            pair_accuracy.append(float(np.mean((agreement + 1) / 2)))
        else:
            pair_accuracy.append(None)
        rank_correlation.append(_spearman_quality(reward, loss))

    row_metrics = {"pair_accuracy": pair_accuracy, "spearman_quality": rank_correlation,
                   "selected_loss": selected_loss.tolist(), "oracle_loss": oracle_loss.tolist(),
                   "regret": (selected_loss - oracle_loss).tolist()}
    if best_fixed_index is not None:
        fixed = losses[:, best_fixed_index]
        row_metrics.update(best_fixed_loss=fixed.tolist(),
                           improvement_over_best_fixed=(fixed - selected_loss).tolist())
    grouped, macro, defined = _macro_metrics(row_metrics, positions)
    candidate_group_losses = np.asarray([losses[idx].mean(axis=0) for idx in positions])
    return {"names": names, "selected_indices": selected, "oracle_indices": oracle,
            "selection_tie_counts": (scores == scores.max(axis=1, keepdims=True)).sum(axis=1),
            "comparable_pairs_per_row": np.asarray(comparable_pairs),
            "row_metrics": row_metrics, "group_ids": identifiers,
            "group_sizes": np.asarray([len(idx) for idx in positions]),
            "group_metrics": grouped, "macro_metrics": macro,
            "defined_groups_per_metric": defined, "n_groups": len(identifiers),
            "candidate_macro_losses": candidate_group_losses.mean(axis=0),
            "best_fixed_index": None if best_fixed_index is None else int(best_fixed_index),
            "best_fixed_name": None if best_fixed_index is None else names[best_fixed_index],
            "tie_rule": "Top-score ties select the first candidate in the fixed input order; "
                        "pairwise reward ties score 0.5 and true-loss ties are excluded."}
