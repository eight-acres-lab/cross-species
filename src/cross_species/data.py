"""Input and aggregation helpers.

The pilot format is intentionally plain CSV so the evaluation code can be
audited before introducing AnnData or a foundation-model dependency.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass
class TaskTable:
    """Responses indexed as ``task_id -> species -> gene vector``."""

    genes: List[str]
    tasks: Dict[str, Dict[str, np.ndarray]]

    @property
    def species(self) -> List[str]:
        return sorted({s for task in self.tasks.values() for s in task})

    @property
    def task_ids(self) -> List[str]:
        return sorted(self.tasks)

    def subset(self, task_ids: Iterable[str]) -> "TaskTable":
        wanted = set(task_ids)
        return TaskTable(self.genes, {k: v for k, v in self.tasks.items() if k in wanted})


def _read_csv(path: str | Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_task_table(
    metadata_path: str | Path,
    response_path: str | Path,
    *,
    task_columns: Sequence[str] | None = None,
    task_column: str | None = "task_id",
) -> TaskTable:
    """Load response rows and average technical/biological replicate rows.

    ``metadata.csv`` must contain ``sample_id`` and ``species``.  It should
    contain ``task_id``; otherwise ``task_columns`` are joined with ``|``.
    ``responses.csv`` contains ``sample_id`` followed by numeric gene columns.
    The function never sees or infers a target species, which keeps the loader
    neutral for leave-one-species-out evaluation.
    """

    metadata = _read_csv(metadata_path)
    response_rows = _read_csv(response_path)
    if not metadata or not response_rows:
        raise ValueError("metadata and responses must contain at least one row")
    meta_by_id = {row["sample_id"]: row for row in metadata}
    response_ids = [row["sample_id"] for row in response_rows]
    missing = [sample_id for sample_id in response_ids if sample_id not in meta_by_id]
    if missing:
        raise ValueError(f"response samples missing from metadata: {missing[:3]}")

    genes = [name for name in response_rows[0] if name != "sample_id"]
    if not genes:
        raise ValueError("responses.csv must have at least one gene column")
    groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for row in response_rows:
        sample_id = row["sample_id"]
        meta = meta_by_id[sample_id]
        if task_column and meta.get(task_column):
            task_id = meta[task_column]
        else:
            if not task_columns:
                raise ValueError("provide task_id or task_columns")
            task_id = "|".join(meta.get(column, "") for column in task_columns)
        species = meta["species"]
        values = np.asarray([float(row[g]) for g in genes], dtype=float)
        groups[(task_id, species)].append(values)

    tasks: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for (task_id, species), rows in groups.items():
        tasks[task_id][species] = np.nanmean(np.vstack(rows), axis=0)
    return TaskTable(genes=genes, tasks=dict(tasks))


def write_task_table(table: TaskTable, metadata_path: str | Path, response_path: str | Path) -> None:
    """Write a TaskTable in the same simple CSV format used by the runner."""

    metadata_rows = []
    response_rows = []
    row_id = 0
    for task_id in table.task_ids:
        for species, vector in sorted(table.tasks[task_id].items()):
            sample_id = f"row-{row_id}"
            row_id += 1
            metadata_rows.append({"sample_id": sample_id, "task_id": task_id, "species": species})
            response_rows.append({"sample_id": sample_id, **{g: f"{x:.10g}" for g, x in zip(table.genes, vector)}})
    with open(metadata_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "task_id", "species"])
        writer.writeheader()
        writer.writerows(metadata_rows)
    with open(response_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *table.genes])
        writer.writeheader()
        writer.writerows(response_rows)
