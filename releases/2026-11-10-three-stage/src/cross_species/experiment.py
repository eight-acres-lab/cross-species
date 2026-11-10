"""Run leave-one-task-out transfer experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .data import TaskTable
from .metrics import task_metrics
from .models import DirectTransfer, MetaRidge, PhyloWeightedMean, ProteinKernelTransfer, SpeciesMean


def _predict_all(
    table: TaskTable,
    train_species: Sequence[str],
    target_species: str,
    distances: Mapping[tuple[str, str], float],
    gene_embeddings: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    models = {
        "direct": DirectTransfer(distances),
        "species_mean": SpeciesMean(),
        "phylo_weighted_mean": PhyloWeightedMean(distances),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    predictions["meta_ridge"] = {}
    if gene_embeddings is not None:
        if gene_embeddings.shape[0] != len(table.genes):
            raise ValueError("gene_embeddings rows must match response features")
        predictions["protein_kernel"] = {}
        protein_kernel = ProteinKernelTransfer(gene_embeddings)
    else:
        protein_kernel = None
    ridge = MetaRidge()
    for task_id in table.task_ids:
        task = table.tasks[task_id]
        if target_species not in task:
            continue
        for name, model in models.items():
            predictions[name][task_id] = model.predict(task, train_species, target_species)
        predictions["meta_ridge"][task_id] = ridge.fit_predict(table, task_id, train_species, target_species)
        if protein_kernel is not None:
            predictions["protein_kernel"][task_id] = protein_kernel.predict(task, train_species, target_species)
    return predictions


def run_experiment(
    table: TaskTable,
    train_species: Sequence[str],
    target_species: str,
    distances: Mapping[tuple[str, str], float],
    *,
    top_k: int = 500,
    gene_embeddings: np.ndarray | None = None,
) -> dict:
    """Evaluate M0-M3 and return predictions plus metrics."""

    predictions = _predict_all(table, train_species, target_species, distances, gene_embeddings=gene_embeddings)
    actual = {task_id: table.tasks[task_id][target_species] for task_id in table.task_ids if target_species in table.tasks[task_id]}
    output = {
        "train_species": list(train_species),
        "target_species": target_species,
        "n_target_tasks": len(actual),
        "models": {},
    }
    for name, pred in predictions.items():
        output["models"][name] = {"metrics": task_metrics(actual, pred, top_k=top_k), "predictions": pred}
    baseline = output["models"]["species_mean"]["metrics"].get("pearson_top_k_macro", float("nan"))
    for details in output["models"].values():
        score = details["metrics"].get("pearson_top_k_macro", float("nan"))
        details["metrics"]["gain_vs_species_mean"] = float(score - baseline) if np.isfinite(score) and np.isfinite(baseline) else float("nan")
    return output
