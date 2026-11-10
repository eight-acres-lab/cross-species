"""Synthetic task generator used to test the evaluation machinery."""

from __future__ import annotations

import numpy as np

from .data import TaskTable


def make_synthetic_table(seed: int = 7, n_tasks: int = 18, n_genes: int = 80) -> TaskTable:
    rng = np.random.default_rng(seed)
    genes = [f"gene_{i:04d}" for i in range(n_genes)]
    species = ["mouse", "rat", "rabbit", "pig", "human"]
    clade_offset = {"mouse": -0.30, "rat": -0.18, "rabbit": 0.02, "pig": 0.14, "human": 0.22}
    tasks = {}
    for task_i in range(n_tasks):
        latent = rng.normal(0, 1, n_genes)
        effect = rng.normal(0, 0.18, n_genes)
        task = {}
        for sp in species:
            # A shared signal plus a species-specific offset and measurement noise.
            task[sp] = latent + clade_offset[sp] * effect + rng.normal(0, 0.08, n_genes)
        tasks[f"stimulus_{task_i:02d}"] = task
    return TaskTable(genes=genes, tasks=tasks)
