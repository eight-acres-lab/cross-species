"""Small response-transfer models for the design-stage pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .data import TaskTable


def _available(task: Mapping[str, np.ndarray], species: Sequence[str]) -> list[str]:
    return [s for s in species if s in task]


@dataclass
class DirectTransfer:
    distances: Mapping[tuple[str, str], float]

    def predict(self, task: Mapping[str, np.ndarray], train_species: Sequence[str], target_species: str) -> np.ndarray:
        available = _available(task, train_species)
        if not available:
            raise ValueError("no training species is available for this task")
        source = min(available, key=lambda s: self.distances.get((s, target_species), self.distances.get((target_species, s), 1e9)))
        return np.asarray(task[source], dtype=float).copy()


@dataclass
class SpeciesMean:
    def predict(self, task: Mapping[str, np.ndarray], train_species: Sequence[str], target_species: str) -> np.ndarray:
        available = _available(task, train_species)
        if not available:
            raise ValueError("no training species is available for this task")
        return np.mean(np.vstack([task[s] for s in available]), axis=0)


@dataclass
class PhyloWeightedMean:
    distances: Mapping[tuple[str, str], float]
    temperature: float = 1.0

    def predict(self, task: Mapping[str, np.ndarray], train_species: Sequence[str], target_species: str) -> np.ndarray:
        available = _available(task, train_species)
        if not available:
            raise ValueError("no training species is available for this task")
        d = np.asarray([self.distances.get((s, target_species), self.distances.get((target_species, s), 1e9)) for s in available], dtype=float)
        weights = np.exp(-d / max(self.temperature, 1e-8))
        weights /= weights.sum()
        return np.average(np.vstack([task[s] for s in available]), axis=0, weights=weights)


@dataclass
class MetaRidge:
    """Learn target response from the mean response of other species.

    Each training task contributes one pseudo-example per available species:
    the held-out species is y and the mean of the remaining species is X. This
    keeps the feature dimension fixed for T1/T2/T3 and prevents a target-specific
    species slot from leaking into the human prediction.
    """

    alpha: float = 10.0
    min_examples: int = 3

    def fit_predict(
        self,
        table: TaskTable,
        task_id: str,
        train_species: Sequence[str],
        target_species: str,
    ) -> np.ndarray:
        target_task = table.tasks[task_id]
        available_target = _available(target_task, train_species)
        if not available_target:
            raise ValueError(f"task {task_id} has no source species")
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for other_task_id, task in table.tasks.items():
            if other_task_id == task_id:
                continue
            available = _available(task, train_species)
            if len(available) < 2:
                continue
            for pseudo_target in available:
                sources = [s for s in available if s != pseudo_target]
                x_rows.append(np.mean(np.vstack([task[s] for s in sources]), axis=0))
                y_rows.append(np.asarray(task[pseudo_target], dtype=float))
        if len(x_rows) < self.min_examples:
            return np.mean(np.vstack([target_task[s] for s in available_target]), axis=0)
        x = np.vstack(x_rows)
        y = np.vstack(y_rows)
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        x_center = x - x_mean
        y_center = y - y_mean
        # Per-feature scaling keeps high-expression genes from dominating the
        # multi-output ridge. Constant columns remain at unit scale.
        scale = x_center.std(axis=0)
        scale[scale == 0] = 1.0
        x_scaled = x_center / scale
        gram = x_scaled.T @ x_scaled + self.alpha * np.eye(x_scaled.shape[1])
        coef = np.linalg.solve(gram, x_scaled.T @ y_center)
        source_mean = np.mean(np.vstack([target_task[s] for s in available_target]), axis=0)
        return y_mean + ((source_mean - x_mean) / scale) @ coef


@dataclass
class ProteinKernelTransfer:
    """Optional low-parameter transfer using fixed protein embeddings.

    ``embeddings`` is one row per response feature after feature alignment.
    The kernel smooths a source response over nearby protein representations;
    ``identity_weight`` keeps the original per-gene signal dominant. This is a
    deliberately small proxy for the protein-aware M3 idea, suitable before
    wiring SATURN or UCE into the pipeline.
    """

    embeddings: np.ndarray
    temperature: float = 0.2
    identity_weight: float = 0.7

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("embeddings must have shape [n_genes, embedding_dim]")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        similarity = normalized @ normalized.T
        weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / max(self.temperature, 1e-8))
        weights /= weights.sum(axis=1, keepdims=True)
        self._kernel = weights

    def predict(self, task: Mapping[str, np.ndarray], train_species: Sequence[str], target_species: str) -> np.ndarray:
        available = _available(task, train_species)
        if not available:
            raise ValueError("no training species is available for this task")
        source = np.mean(np.vstack([task[s] for s in available]), axis=0)
        smoothed = self._kernel @ source
        w = min(max(self.identity_weight, 0.0), 1.0)
        return w * source + (1.0 - w) * smoothed
