from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cross_species.experiment import run_experiment
from cross_species.metrics import average_precision, bootstrap_task_metrics, pearson, roc_auc, spearman
from cross_species.synthetic import make_synthetic_table


DISTANCES = {
    (a, b): 0.1
    for a in ["mouse", "rat", "rabbit", "pig", "human"]
    for b in ["mouse", "rat", "rabbit", "pig", "human"]
    if a != b
}


class PilotTest(unittest.TestCase):
    def test_metrics_perfect_prediction(self):
        y = np.array([-2.0, -0.5, 0.2, 1.3])
        self.assertAlmostEqual(pearson(y, y), 1.0)
        self.assertAlmostEqual(spearman(y, y), 1.0)
        self.assertAlmostEqual(roc_auc(y, y), 1.0)
        self.assertAlmostEqual(average_precision(y, y), 1.0)

    def test_human_is_evaluation_only(self):
        table = make_synthetic_table(n_tasks=8, n_genes=12)
        result = run_experiment(table, ["mouse", "rat"], "human", DISTANCES, top_k=5)
        self.assertEqual(result["target_species"], "human")
        self.assertEqual(result["n_target_tasks"], 8)
        for details in result["models"].values():
            self.assertNotIn("human", details.get("train_species", []))
            self.assertEqual(details["metrics"]["n_tasks"], 8)

    def test_single_species_falls_back_without_crashing(self):
        table = make_synthetic_table(n_tasks=5, n_genes=8)
        result = run_experiment(table, ["mouse"], "human", DISTANCES, top_k=4)
        self.assertEqual(result["models"]["meta_ridge"]["metrics"]["n_tasks"], 5)

    def test_protein_kernel_proxy_accepts_gene_embeddings(self):
        table = make_synthetic_table(n_tasks=5, n_genes=8)
        embeddings = np.eye(8)
        result = run_experiment(table, ["mouse", "rat"], "human", DISTANCES, top_k=4, gene_embeddings=embeddings)
        self.assertIn("protein_kernel", result["models"])
        self.assertEqual(result["models"]["protein_kernel"]["metrics"]["n_tasks"], 5)

    def test_task_bootstrap_records_task_count(self):
        table = make_synthetic_table(n_tasks=4, n_genes=8)
        actual = {task: table.tasks[task]["human"] for task in table.task_ids}
        predicted = {task: table.tasks[task]["mouse"] for task in table.task_ids}
        result = bootstrap_task_metrics(actual, predicted, top_k=4, n_boot=20, seed=7)
        self.assertEqual(result["n_tasks"], 4)
        self.assertEqual(result["n_boot"], 20)
        self.assertLessEqual(result["pearson_top_k_macro"]["lower_2.5"], result["pearson_top_k_macro"]["upper_97.5"])


if __name__ == "__main__":
    unittest.main()
