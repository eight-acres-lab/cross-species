#!/usr/bin/env python3
"""Recompute paper summaries, bootstrap intervals and supplementary table values.

Inputs are the deposited per-compound, per-line, per-seed and per-donor scores.
Printed manuscript values are recorded in figures/paper_values.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_gse325993_state as exp2
import audit_gse325993_screen as exp3
import audit_nonrodent_lps_state as rabbit

EXP2 = "outputs/gse325993_state_20261110/state_primary_v1"
EXP3 = "outputs/gse325993_screen_validation_20261110/validation_v1"
RABBIT = "outputs/nonrodent_singlecell_20261110/rabbit_primary_v1"
HUMAN = "outputs/remote_gpu_20261108/"
INTEGER_COLUMNS = {
    "fold", "seed", "selected_step", "selected_step_from_exp2", "n_control",
    "n_target", "n_genes", "query_cells", "target_cells", "genes",
}
IDENTIFIER_COLUMNS = {
    "compound", "target_line", "train_line", "donor", "variant",
    "prediction_sha256", "model_weight_sha256",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    return json.loads((ROOT / relative).read_text())


def read_rows(relative, delimiter=","):
    with (ROOT / relative).open(newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter=delimiter))
    rows = []
    for source in source_rows:
        row = {}
        for key, value in source.items():
            if value == "":
                row[key] = None
            elif key in IDENTIFIER_COLUMNS:
                row[key] = value
            elif key in INTEGER_COLUMNS:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"{label}: calculated {actual}, deposited {expected}")


def numeric_pairs(actual, expected, path="summary"):
    if isinstance(actual, dict):
        for key, value in actual.items():
            require(key in expected, f"Missing deposited field: {path}/{key}")
            yield from numeric_pairs(value, expected[key], f"{path}/{key}")
    elif isinstance(actual, list):
        require(len(actual) == len(expected), f"Changed list length: {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            yield from numeric_pairs(left, right, f"{path}/{index}")
    elif isinstance(actual, bool):
        require(actual == expected, f"Changed Boolean field: {path}")
    elif isinstance(actual, (int, float)):
        yield path, float(actual), float(expected)
    elif actual is None:
        require(expected is None, f"Changed undefined field: {path}")


def compare_summary(actual, expected):
    comparisons, maximum = 0, 0.0
    for label, left, right in numeric_pairs(actual, expected):
        close(left, right, label)
        comparisons += 1
        maximum = max(maximum, abs(left - right))
    return {"numeric_comparisons": comparisons, "max_absolute_difference": maximum}


def animal_summaries():
    rows2 = read_rows(EXP2 + "/audit/line_seed_scores.csv")
    rows3 = read_rows(EXP3 + "/audit/line_seed_scores.csv")
    rows_rabbit = read_rows(RABBIT + "/audit/donor_scores.csv")
    require((len(rows2), len(rows3), len(rows_rabbit)) == (96, 96, 110),
            "Expected 96 exp2, 96 exp3 and 110 rabbit score rows")
    s2, _, _ = exp2.summarize(
        rows2, read(EXP2 + "/ALL_FITS_SELECTED.json")["fits"],
        read("docs/gse325993_state_20261110/CONTRACT_v1.json"))
    s3, _, _ = exp3.summarize(
        rows3, read("docs/gse325993_screen_validation_20261110/CONTRACT_v1.json"))
    sr, _ = rabbit.summarize(rows_rabbit, read(RABBIT + "/ALL_FITS_SELECTED.json")["fits"])
    results = {}
    for name, summary, base, rows in (
        ("exp2", s2, EXP2, rows2), ("exp3", s3, EXP3, rows3),
        ("rabbit", sr, RABBIT, rows_rabbit),
    ):
        results[name] = {
            "input_score_rows": len(rows), "variants": len(summary["variants"]),
            **compare_summary(summary, read(base + "/audit/summary.json")),
        }
    results["rabbit"]["bootstrap_replicates"] = rabbit.BOOTSTRAP_REPLICATES
    return (s2, s3, sr), results


def compound_summaries():
    results = {}
    for directory in (
        "feedback_nested_chemical_20261107_absolute_v1",
        "feedback_joint_specificity_20261107_absolute",
        "feedback_residual_adapter_20261107_absolute",
    ):
        base = "outputs/" + directory
        rows = read_rows(base + "/compound_metrics.tsv", delimiter="\t")
        report = read(base + "/report.json")
        require(len(rows) == 126, f"Expected 126 compound rows: {directory}")
        means, comparisons = {}, 0
        for method, reference in report["metrics"].items():
            require([row["compound"] for row in rows] == reference["group_ids"],
                    f"Compound order changed: {directory}/{method}")
            values = np.asarray([row[method + "_mse"] for row in rows])
            require(len(values) == len(reference["group_losses"]), "Compound count changed")
            for actual, expected in zip(values, reference["group_losses"]):
                close(actual, expected, directory + "/" + method)
            means[method] = float(values.mean())
            close(means[method], reference["macro_loss"], directory + "/" + method + "/macro")
            comparisons += len(values) + 1
        result = {"compounds": len(rows), "model_means": means,
                  "numeric_comparisons": comparisons}
        if "primary_gain_from_source" in report:
            reference = report["primary_gain_from_source"]
            delta = np.asarray([row["chemical_context_mse"] - row["source_plus_chemical_context_mse"]
                                for row in rows])
            rng = np.random.default_rng(reference["seed"])
            bootstrap = np.empty(reference["n_boot"])
            for start in range(0, len(bootstrap), 256):
                size = min(256, len(bootstrap) - start)
                indices = rng.integers(0, len(delta), size=(size, len(delta)))
                bootstrap[start:start + size] = delta[indices].mean(axis=1)
            interval = np.percentile(bootstrap, [2.5, 97.5])
            close(delta.mean(), reference["delta"], "TG-GATEs paired gain")
            for actual, expected in zip(interval, reference["ci95"]):
                close(actual, expected, "TG-GATEs bootstrap interval")
            result.update({
                "gain_pct": float(100 * delta.mean() / means["chemical_context"]),
                "positive_compounds": int((delta > 0).sum()),
                "bootstrap_replicates": len(bootstrap), "delta_ci95": interval.tolist(),
                "bootstrap_numeric_comparisons": 3,
            })
        results[directory] = result
    return results


def external_summary(report):
    means = {}
    for rule, reference in report["macro_mse"].items():
        means[rule] = float(np.mean([row["rules"][rule]["mse"] for row in report["external_tasks"]]))
        close(means[rule], reference, "External transfer/" + rule)
    return {
        "drug_rows": len(report["external_tasks"]), "means": means,
        "numeric_comparisons": len(means),
        "fusion_gain_pct": 100 * (means["chemical_context"] - means["source_plus_context"]) / means["chemical_context"],
        "scalar_gain_vs_zero_pct": 100 * (means["zero"] - means["scalar_nointercept"]) / means["zero"],
    }


def human_summaries():
    groups = {
        "state_native": read(HUMAN + "state_native_formal_64/report.json")["test"],
        "state_mismatched": {"mismatched": read(HUMAN + "state_mispaired_posthoc/report.json")["mismatched_test"]},
        "state_policy": read(HUMAN + "state_gaussian_policy_formal_64/report.json")["test"],
        "scdfm_population": read(HUMAN + "scdfm_nft_formal/report.json")["test"],
        "scdfm_witness": read(HUMAN + "scdfm_witness_formal/evaluation_float64.json")["evaluation"],
    }
    recalculated, comparisons = {}, 0
    for name, methods in groups.items():
        recalculated[name] = {}
        for method, record in methods.items():
            recalculated[name][method] = {}
            for metric, reference in record["macro"].items():
                values = [row.get(metric) for row in record["by_condition"].values()]
                values = [value for value in values if value is not None]
                if values:
                    value = float(np.mean(values))
                    close(value, reference, f"Human-only/{name}/{method}/{metric}")
                    recalculated[name][method][metric] = value
                    comparisons += 1
    return recalculated, comparisons


def printed_value_matches(actual, expected, label):
    decimals = len(expected.partition(".")[2])
    require(round(float(actual), decimals) == round(float(expected), decimals),
            f"{label}: calculated {actual}, manuscript {expected}")


def manuscript_values(reference, animal, compounds, external, human):
    s2, s3, sr = animal
    abstract = {
        "exp2_M56_gain_pct": s2["paired_comparisons"]["Monkey56:gain_over_H"]["percent_of_comparator"],
        "exp3_M56_gain_pct": s3["paired_comparisons"]["Monkey56:gain_over_H"]["percent_of_comparator"],
        "exp3_M82_gain_pct": s3["paired_comparisons"]["Monkey82:gain_over_H"]["percent_of_comparator"],
        "exp3_human_gain_pct": s3["paired_comparisons"]["extra_human_gain"]["percent_of_comparator"],
        "rabbit_gain_pct": sr["paired_comparisons"]["delta_A"]["percent_of_left_mean"],
        "rabbit_human_gain_pct": sr["paired_comparisons"]["delta_H"]["percent_of_left_mean"],
        "TG_GATEs_gain_pct": compounds["feedback_nested_chemical_20261107_absolute_v1"]["gain_pct"],
    }
    require(set(abstract) == set(reference["abstract_percentages"]), "Abstract value keys changed")
    for key, value in abstract.items():
        printed_value_matches(value, reference["abstract_percentages"][key], "Abstract/" + key)

    def macro(summary, variant):
        return summary["variants"][variant]["macro"]["mean_response_mse"]

    rules = ["zero", "human_mean", "rat_copy", "scalar_nointercept", "chemical_context", "source_plus_context"]
    tables = {
        "1": [[row["rules"][rule]["mse"] for row in external["external_tasks"]]
              + [float(np.mean([row["rules"][rule]["mse"] for row in external["external_tasks"]]))]
              for rule in rules],
        "2": [[human["state_native"]["released_checkpoint"]["native_energy"]],
              [human["state_native"]["native_energy_sft"]["native_energy"]],
              [human["state_mismatched"]["mismatched"]["native_energy"]]],
        "3": [[human["scdfm_population"][method][metric] for metric in ("energy_u", "mean_mse")]
              for method in ("released_checkpoint", "native_fm", "nft", "terminal_gradient")],
        "4": [[human["scdfm_witness"][method][metric] for metric in ("energy_v", "mean_mse")]
              for method in ("released_checkpoint", "witness", "native_fm")],
    }
    variants = ["H", "Hextra_cells", "Monkey56", "Monkey56A0", "Monkey82", "Monkey82A0"]
    tables["5"] = [
        [macro(s2, v), 100 * (macro(s2, "H") - macro(s2, v)) / macro(s2, "H"),
         macro(s3, v), 100 * (macro(s3, "H") - macro(s3, v)) / macro(s3, "H")]
        for v in variants
    ]
    tables["6"] = []
    for screen, line in (("exp2", "63Ab2.2-02"), ("exp2", "29B5-39s"),
                         ("exp3", "63Ab2.2-02"), ("exp3", "29B5-39s")):
        summary = s2 if screen == "exp2" else s3
        tables["6"].append([
            next(row["mean_response_mse"] for row in summary["variants"][v]["per_line"]
                 if row["target_line"] == line)
            for v in ("H", "Hextra_cells", "Monkey56", "Monkey82")
        ])
    controls = ["Zero_response", "Hbase_mean_response", "Half_H_response", "Three_quarter_H_response"]
    controls += [v + "|cal_shrink" for v in variants]
    tables["7"] = [[macro(s2, v), macro(s3, v)] for v in controls]
    human_reference = sr["variants"]["H"]["mean_response_mse"]["mean"]
    tables["8"] = [
        [sr["variants"][v]["mean_response_mse"]["mean"],
         100 * (human_reference - sr["variants"][v]["mean_response_mse"]["mean"]) / human_reference]
        for v in ("H", "H+Hextra", "H+Rabbit", "H+RabbitA0", "Hbase_mean_response", "Zero_response")
    ]
    require(set(tables) == set(reference["supplementary_tables"]), "Supplementary table keys changed")
    table_checks = []
    for number, calculated in tables.items():
        expected = reference["supplementary_tables"][number]["rows"]
        require(len(calculated) == len(expected), f"Table {number}: row count changed")
        count = 0
        for index, (values, row) in enumerate(zip(calculated, expected)):
            require(len(values) == len(row["values"]), f"Table {number}: column count changed")
            for actual, printed in zip(values, row["values"]):
                printed_value_matches(actual, printed, f"Table {number}/row {index + 1}")
                count += 1
        table_checks.append({"table": int(number), "numeric_cells_checked": count})
    return {"abstract_percentages": abstract, "abstract_percentage_checks": len(abstract),
            "supplementary_tables": table_checks,
            "supplementary_numeric_cells_checked": sum(row["numeric_cells_checked"] for row in table_checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the verification report to this JSON file")
    args = parser.parse_args()
    animal, animal_checks = animal_summaries()
    compounds = compound_summaries()
    external = read("outputs/external_transfer_04/report.json")
    human, human_checks = human_summaries()
    result = {
        "status": "passed", "input_level": "deposited score records",
        "animal_summary_recomputations": animal_checks,
        "compound_recomputations": compounds,
        "external_transfer": external_summary(external),
        "human_only_macro_means_recomputed": human_checks,
        "manuscript_comparison": manuscript_values(
            read("figures/paper_values.json"), animal, compounds, external, human),
        "numpy_version": np.__version__,
    }
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        destination = args.out.resolve()
        protected = {ROOT / entry["path"] for entry in read("DATA_MANIFEST.json")["files"]}
        protected.add(ROOT / "DATA_MANIFEST.json")
        report_path = ROOT / "figures/generated/paper_results.json"
        require(destination not in protected or destination == report_path,
                "Choose an output path outside the deposited input files")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
