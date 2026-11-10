#!/usr/bin/env python3
"""Check the interpretation of archived replicate-normalised Pearson scores.

The exact finite distribution below enumerates independent Rademacher signal and
error variables. It is an analytic counterexample, not simulated biological data.
The empirical summary only reads archived correlations; it does not rerun TG-GATEs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from run_system_vs_species import pearson

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "artifacts/biology/dili/system_vs_species.json",
    "artifacts/biology/reviewer_followup.json",
)


def independent_profiles(reliabilities):
    """A common unit-variance signal plus four mutually independent errors.

    Every joint state is equally likely; all population moments are therefore
    exactly represented. Repetition only clears the production Pearson size floor.
    """
    if len(reliabilities) != 4 or any(not 0 < r <= 1 for r in reliabilities):
        raise ValueError("four reliabilities in (0, 1] are required")
    basis = np.tile(np.array(list(itertools.product((-1., 1.), repeat=5))), (16, 1))
    return np.column_stack([
        basis[:, 0] + math.sqrt(1 / r - 1) * basis[:, i + 1]
        for i, r in enumerate(reliabilities)
    ])


def score(profiles):
    within = [pearson(profiles[:, a], profiles[:, b]) for a, b in ((0, 1), (2, 3))]
    across = [pearson(profiles[:, a], profiles[:, b]) for a in (0, 1) for b in (2, 3)]
    denominator = math.sqrt(within[0] * within[1])
    observed = float(np.mean(across))
    return {"within_side_correlations": within, "cross_side_correlations": across,
            "observed_mean": observed, "denominator": denominator,
            "normalised_score": observed / denominator}


def analytic_score(reliabilities):
    a1, a2, b1, b2 = reliabilities
    numerator = (math.sqrt(a1) + math.sqrt(a2)) * (math.sqrt(b1) + math.sqrt(b2)) / 4
    return numerator / (a1 * a2 * b1 * b2) ** .25


def counterexamples():
    results = []
    for label, rs in (
        ("equal reliability within each side", [.64, .64, .81, .81]),
        ("unequal reliability, independent errors", [.81, .09, .81, .09]),
        ("unequal reliability with denominator in [0.60, 0.78)", [.99, .49, .99, .49]),
    ):
        results.append({"case": label, "individual_reliabilities": rs,
                        "latent_correlation": 1.0, "analytic_score": analytic_score(rs),
                        **score(independent_profiles(rs))})
    return results


def mask_audit():
    profiles = independent_profiles([.64] * 4)
    for column in range(4):
        profiles[column * 16:(column + 1) * 16, column] = np.nan
    pairs = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
    return {
        "type": "constructed missingness example, not empirical coverage",
        "pairwise_gene_counts": [int(np.isfinite(profiles[:, [a, b]]).all(axis=1).sum())
                                 for a, b in pairs],
        "common_gene_count": int(np.isfinite(profiles).all(axis=1).sum()),
        "interpretation": "The production Pearson function masks each pair separately. "
                          "Equal nominal axes do not guarantee one common measured gene set.",
    }


def empirical_summary():
    species = json.loads((ROOT / SOURCES[0]).read_text())["species_changed"]["per_compound"]
    metadata = json.loads((ROOT / SOURCES[1]).read_text())["dose"]["per_compound"]
    values = {r["compound"]: r["gene"] for r in species}
    result = {}
    for name, require_matched in (("high_label_band", False), ("equal_concentration_band", True)):
        rows = [{"compound": r["compound"], "observed": values[r["compound"]]["obs"],
                 "normalised_score": values[r["compound"]]["frac"]}
                for r in metadata if r["in_ceiling_band"] and
                (not require_matched or r["concentration_matched"])]
        result[name] = {"n": len(rows), "median_observed": median(r["observed"] for r in rows),
                        "median_score": median(r["normalised_score"] for r in rows), "rows": rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "audit_date": "2026-11-06",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        "status": "Interpretation audit; original expression matrices have not been rerun",
        "score_formula": "mean of four cross-side correlations / sqrt(within_A * within_B)",
        "ceiling_requirements": ["independent additive errors, independent of latent responses",
                                 "equal replicate reliability within each side",
                                 "common measured gene population for all six correlations",
                                 "matched averaging level"],
        "counterexamples": counterexamples(),
        "missing_gene_masks": mask_audit(),
        "archived_empirical_summary": empirical_summary(),
        "conclusion": "These archived scores are descriptive normalisations, not verified "
                      "fractions of an attainable upper bound. Ratios above one do not "
                      "identify shared-control contamination.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
