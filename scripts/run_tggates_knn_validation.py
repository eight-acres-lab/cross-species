#!/usr/bin/env python3
"""CPU-only rat-to-human Open TG-GATEs validation.

The model predicts a held-out human expression vector by retrieving similar
rat treatment conditions from training compounds. All choices are selected by
compound-grouped validation inside the published training split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as handle:
        handle.readline()
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            ids.append(fields[0])
            rows.append(np.asarray(fields[1:], dtype=np.float32))
    return ids, np.stack(rows)


def row_zscore(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)


def row_pearson(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=1, keepdims=True)
    truth = truth - truth.mean(axis=1, keepdims=True)
    denom = np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1), 1e-8)
    return np.sum(pred * truth, axis=1) / denom


def row_cosine(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1), 1e-8)
    return np.sum(pred * truth, axis=1) / denom


def row_nrmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
    scale = np.maximum(truth.std(axis=1), 1e-8)
    return rmse / scale


def similarity_matrix(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    metadata_train: list[dict[str, str]],
    metadata_valid: list[dict[str, str]],
    center_mode: str,
    restriction: str,
) -> np.ndarray:
    if center_mode == "global_centered":
        train_center = x_train.mean(axis=0, keepdims=True)
        x_train = x_train - train_center
        x_valid = x_valid - train_center
    z_train = row_zscore(x_train)
    z_valid = row_zscore(x_valid)
    scores = z_valid @ z_train.T / z_train.shape[1]
    if restriction == "all":
        return scores
    mask = np.ones(scores.shape, dtype=bool)
    for i, valid_meta in enumerate(metadata_valid):
        for j, train_meta in enumerate(metadata_train):
            if valid_meta["time_h"] != train_meta["time_h"]:
                mask[i, j] = False
            if restriction == "same_dose_time" and valid_meta["dose_level"] != train_meta["dose_level"]:
                mask[i, j] = False
    return np.where(mask, scores, -1e9)


def retrieve(
    scores: np.ndarray,
    y_train: np.ndarray,
    k: int,
    aggregation: str,
    temperature: float,
) -> np.ndarray:
    prediction = np.empty((scores.shape[0], y_train.shape[1]), dtype=np.float32)
    for i in range(scores.shape[0]):
        # Masked conditions must not enter a small-subset retrieval merely
        # because fewer than k same-dose/time neighbours are available.
        eligible = np.flatnonzero(np.isfinite(scores[i]) & (scores[i] > -1e8))
        if not len(eligible):
            raise ValueError('no eligible neighbours; caller must choose an explicit fallback')
        count = min(k, len(eligible))
        indices = eligible[np.argpartition(scores[i, eligible], -count)[-count:]]
        if aggregation == "mean":
            prediction[i] = y_train[indices].mean(axis=0)
        else:
            weights = np.exp((scores[i, indices] - scores[i, indices].max()) / temperature)
            prediction[i] = np.average(y_train[indices], axis=0, weights=weights)
    return prediction


def evaluate(prediction: np.ndarray, truth: np.ndarray, center: np.ndarray) -> dict[str, float]:
    centered_prediction = prediction - center
    centered_truth = truth - center
    scores = row_pearson(centered_prediction, centered_truth)
    cosine_scores = row_cosine(prediction, truth)
    nrmse_scores = row_nrmse(prediction, truth)
    flat_pred = centered_prediction.ravel()
    flat_truth = centered_truth.ravel()
    flat_pred -= flat_pred.mean()
    flat_truth -= flat_truth.mean()
    overall = float(np.dot(flat_pred, flat_truth) / max(np.linalg.norm(flat_pred) * np.linalg.norm(flat_truth), 1e-8))
    flat_raw_pred = prediction.ravel()
    flat_raw_truth = truth.ravel()
    raw_cosine = float(np.dot(flat_raw_pred, flat_raw_truth) / max(np.linalg.norm(flat_raw_pred) * np.linalg.norm(flat_raw_truth), 1e-8))
    overall_nrmse = float(np.sqrt(np.mean((prediction - truth) ** 2)) / max(float(truth.std()), 1e-8))
    return {
        "overall_pearson": overall,
        "macro_sample_pearson": float(scores.mean()),
        "median_sample_pearson": float(np.median(scores)),
        "q25_sample_pearson": float(np.quantile(scores, 0.25)),
        "min_sample_pearson": float(scores.min()),
        "overall_raw_cosine": raw_cosine,
        "macro_raw_cosine": float(cosine_scores.mean()),
        "median_raw_cosine": float(np.median(cosine_scores)),
        "overall_nrmse": overall_nrmse,
        "macro_nrmse": float(nrmse_scores.mean()),
        "median_nrmse": float(np.median(nrmse_scores)),
    }


def compound_folds(compounds: list[str], n_folds: int = 3) -> list[np.ndarray]:
    unique = sorted(set(compounds))
    folds = [[] for _ in range(n_folds)]
    for i, compound in enumerate(unique):
        folds[i % n_folds].append(compound)
    return [np.array(fold) for fold in folds]


def matched_change(values: np.ndarray, pair_ids: list[str], rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each compound/time control from its matched condition."""
    control_index: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        if row["dose_level"].lower() == "control":
            control_index[(row["compound"], row["time_h"])] = i
    changed = np.full_like(values, np.nan, dtype=np.float32)
    valid = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        control = control_index.get((row["compound"], row["time_h"]))
        if control is not None:
            changed[i] = values[i] - values[control]
            valid[i] = True
    if not np.all(valid):
        raise ValueError("every pair must have a matched compound/time control for change mode")
    return changed, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-mode", choices=["absolute", "change"], default="change", help="primary label is matched-control delta; absolute is retained for historical replication")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_ids, x = load_matrix(data_dir / "pairs_rat_X.tsv")
    y_ids, y = load_matrix(data_dir / "pairs_human_Y.tsv")
    if pair_ids != y_ids:
        raise ValueError("rat and human pair IDs are not aligned")
    metadata = {row["pair_id"]: row for row in csv.DictReader((data_dir / "pairs_metadata.tsv").open(), delimiter="\t")}
    splits = {row["pair_id"]: row["split"] for row in csv.DictReader((data_dir / "pairs_splits.tsv").open(), delimiter="\t")}
    rows = [metadata[pair_id] for pair_id in pair_ids]
    if args.response_mode == "change":
        x, _ = matched_change(x, pair_ids, rows)
        y, _ = matched_change(y, pair_ids, rows)
    train_indices = np.array([splits[pair_id] == "train" for pair_id in pair_ids])
    if args.response_mode == "change":
        # Controls are needed to construct matched deltas, then removed from
        # both retrieval and evaluation so zero control vectors cannot inflate
        # or distort the response-label score.
        treatment = np.array([row["dose_level"].lower() != "control" for row in rows])
        train_indices &= treatment
        test_indices = (~np.array([splits[pair_id] == "train" for pair_id in pair_ids])) & treatment
        n_excluded_controls = int(np.sum(~treatment))
    else:
        test_indices = ~train_indices
        n_excluded_controls = 0
    train_rows = np.where(train_indices)[0]
    test_rows = np.where(test_indices)[0]
    train_compounds = [rows[i]["compound"] for i in train_rows]

    candidates = []
    for center_mode in ["absolute", "global_centered"]:
        for restriction in ["all", "same_time", "same_dose_time"]:
            for aggregation in ["mean", "exp"]:
                for k in [3, 5, 10, 20]:
                    candidates.append({
                        "center_mode": center_mode,
                        "restriction": restriction,
                        "aggregation": aggregation,
                        "k": k,
                        "temperature": 0.1,
                    })

    folds = compound_folds(train_compounds, 3)
    validation_results = []
    for candidate in candidates:
        fold_metrics = []
        for held_out_compounds in folds:
            valid_mask = np.array([compound in held_out_compounds for compound in train_compounds])
            inner_train = train_rows[~valid_mask]
            valid = train_rows[valid_mask]
            meta_inner = [rows[i] for i in inner_train]
            meta_valid = [rows[i] for i in valid]
            scores = similarity_matrix(x[inner_train], x[valid], meta_inner, meta_valid, candidate["center_mode"], candidate["restriction"])
            prediction = retrieve(scores, y[inner_train], candidate["k"], candidate["aggregation"], candidate["temperature"])
            center = y[inner_train].mean(axis=0)
            fold_metrics.append(evaluate(prediction, y[valid], center))
        aggregate = {key: float(np.mean([m[key] for m in fold_metrics])) for key in fold_metrics[0]}
        validation_results.append({**candidate, **aggregate, "folds": fold_metrics})

    selected = max(validation_results, key=lambda row: (row["macro_sample_pearson"], row["overall_pearson"]))
    test_scores = similarity_matrix(x[train_rows], x[test_rows], [rows[i] for i in train_rows], [rows[i] for i in test_rows], selected["center_mode"], selected["restriction"])
    test_prediction = retrieve(test_scores, y[train_rows], selected["k"], selected["aggregation"], selected["temperature"])
    train_center = y[train_rows].mean(axis=0)
    test_metrics = evaluate(test_prediction, y[test_rows], train_center)

    baseline = {
        "overall_pearson": 0.696,
        "median_sample_pearson": 0.779,
        "source": "Eastman & Pande 2019 reproduction summary in Zenodo record",
        "applies_to": "absolute expression-state label only",
    }
    result = {
        "dataset": "Open TG-GATEs processed rat-human in-vitro liver pairs",
        "response_mode": args.response_mode,
        "n_pairs": len(pair_ids),
        "n_train_pairs": int(train_indices.sum()),
        "n_test_pairs": int(test_indices.sum()),
        "n_excluded_control_rows": n_excluded_controls,
        "n_train_compounds": len(set(train_compounds)),
        "n_test_compounds": len({rows[i]['compound'] for i in test_rows}),
        "feature_counts": {"rat": int(x.shape[1]), "human": int(y.shape[1])},
        "selection": selected,
        "test_metrics": test_metrics,
        "constant_baseline_metrics": evaluate(np.broadcast_to(train_center, y[test_rows].shape), y[test_rows], train_center),
        "historical_reference": baseline,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "validation_grid.json").write_text(json.dumps(validation_results, indent=2), encoding="utf-8")
    with (output_dir / "test_predictions_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "compound", "dose_level", "time_h", "pearson_centered", "raw_cosine", "nrmse"])
        test_scores_rows = row_pearson(test_prediction - train_center, y[test_rows] - train_center)
        test_cosine_rows = row_cosine(test_prediction, y[test_rows])
        test_nrmse_rows = row_nrmse(test_prediction, y[test_rows])
        for pair_id, row_idx, score, cosine, nrmse in zip([pair_ids[i] for i in test_rows], test_rows, test_scores_rows, test_cosine_rows, test_nrmse_rows):
            row = rows[row_idx]
            writer.writerow([pair_id, row["compound"], row["dose_level"], row["time_h"], float(score), float(cosine), float(nrmse)])
    print(json.dumps({"selected": selected, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
