#!/usr/bin/env python3
"""Recompute reviewer sensitivity checks from the deposited metadata and results.

This does not rerun expression preprocessing, LINCS, or injury prediction. It audits
nominal treatment concentrations, adjusts the 18 Table 3 tests as one post-hoc family,
and records the donor-pair counts retained by the hepatocyte preprocessing metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from scipy.stats import binomtest

from run_system_vs_species import normalise

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "doses": "artifacts/tg_gates_rat_human_v1/metadata/open_tggates_cel_file_attribute.csv",
    "species": "artifacts/biology/dili/system_vs_species.json",
    "time": "artifacts/biology/dili/mismatch_controls.json",
    "raw": "artifacts/biology/concordance/rho_cross_panel_both_estimators.json",
    "adjusted": "artifacts/biology/concordance/rho_double_centred.json",
    "donors": "artifacts/biology/gse330151/response_metadata.tsv",
}
BAND = (0.60, 0.78)


def holm(pvalues):
    """Holm's step-down adjustment, returned in the original order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("p-values must be finite and between zero and one")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def compound_key(name):
    # The expression artifacts spell the metadata's TNFα as tnfalpha.
    return normalise(name.replace("α", "alpha"))


def concentration(value, unit):
    """Canonicalise concentrations without converting mass to molar units."""
    unit = unit.strip().replace("µ", "μ")
    conversions = {
        "M": (1e6, "μM"), "mM": (1e3, "μM"),
        "μM": (1, "μM"), "nM": (1e-3, "μM"),
        "mg/mL": (1e3, "μg/mL"), "μg/mL": (1, "μg/mL"),
        "ng/mL": (1e-3, "μg/mL"),
    }
    if unit not in conversions:
        return None
    factor, canonical = conversions[unit]
    number = float(value) * factor
    if not math.isfinite(number) or number <= 0:
        return None
    return number, canonical


def dose_match(left, right):
    """Missing or conflicting concentrations are never classified as matched."""
    if len(left) != 1 or len(right) != 1 or None in left or None in right:
        return False
    a, b = next(iter(left)), next(iter(right))
    return a[1] == b[1] and math.isclose(a[0], b[0], rel_tol=1e-9, abs_tol=1e-12)


def usable(row):
    value = row["gene"]["frac"]
    return value is not None and math.isfinite(value)


def in_band(row):
    bound = row["gene"]["bound"]
    return usable(row) and bound is not None and BAND[0] <= bound < BAND[1]


def paired_summary(rows):
    differences = [r["time_minus_species_pp"] for r in rows]
    positives = sum(d > 0 for d in differences)
    nonzero = sum(d != 0 for d in differences)
    return {
        "n": len(rows), "n_nonzero": nonzero, "time_greater": positives,
        "median_difference_pp": median(differences) if differences else None,
        "p_two_sided_sign_nominal": float(binomtest(positives, nonzero).pvalue) if nonzero else None,
        "rows": rows,
    }


def dose_audit():
    doses = defaultdict(set)
    original = defaultdict(set)
    with (ROOT / INPUTS["doses"]).open(encoding="cp932") as handle:
        for row in csv.DictReader(handle):
            if row["EXP_TEST_TYPE"] != "in vitro" or row["DOSE_LEVEL"] != "High":
                continue
            key = (compound_key(row["COMPOUND_NAME"]), row["SPECIES"], row["SACRIFICE_PERIOD"])
            doses[key].add(concentration(row["DOSE"], row["DOSE_UNIT"]))
            original[key].add((float(row["DOSE"]), row["DOSE_UNIT"]))

    species = json.loads((ROOT / INPUTS["species"]).read_text())["species_changed"]["per_compound"]
    time = json.loads((ROOT / INPUTS["time"]).read_text())
    matched = set()
    records = []
    for row in species:
        compound = row["compound"]
        human_key, rat_key = (compound, "Human", "24 hr"), (compound, "Rat", "24 hr")
        same = dose_match(doses[human_key], doses[rat_key])
        if same:
            matched.add(compound)
        records.append({
            "compound": compound,
            "human_doses": sorted(original[human_key]), "rat_doses": sorted(original[rat_key]),
            "concentration_matched": same, "in_ceiling_band": in_band(row),
            "fraction": row["gene"]["frac"], "ceiling": row["gene"]["bound"],
        })
    band_rows = [r for r in records if r["in_ceiling_band"]]
    matched_rows = [r for r in band_rows if r["concentration_matched"]]
    paired = {}
    for side in ("rat", "human"):
        for hour in (8, 2):
            contrast = f"{side} hepatocyte | time 24 hr vs {hour} hr"
            lookup = {r["compound"]: r for r in time[contrast]["per_compound"]}
            paired[contrast] = {}
            for name, predicate in (("finite", usable), ("both_in_band", in_band)):
                rows = []
                for row in species:
                    compound = row["compound"]
                    other = lookup.get(compound)
                    if compound not in matched or other is None or not predicate(row) or not predicate(other):
                        continue
                    if not dose_match(doses[(compound, side.title(), "24 hr")],
                                      doses[(compound, side.title(), f"{hour} hr")]):
                        continue
                    rows.append({
                        "compound": compound,
                        "time_minus_species_pp": 100 * (other["gene"]["frac"] - row["gene"]["frac"]),
                    })
                paired[contrast][name] = paired_summary(rows)
    return {
        "selection": "High in vitro arms; identical unique nominal concentrations at 24 h across species; paired time contrasts additionally require the same concentration at both times",
        "concentration_note": "Nominal concentration is not intracellular exposure; mass and molar units are not interconverted",
        "name_alias": {"TNFα": "tnfalpha"}, "ceiling_band_left_closed_right_open": list(BAND),
        "original_band_n": len(band_rows), "matched_band_n": len(matched_rows),
        "mismatched_band_n": len(band_rows) - len(matched_rows),
        "matched_band_median_percent": 100 * median(r["fraction"] for r in matched_rows),
        "per_compound": records, "paired_time_sensitivities": paired,
    }


def multiplicity_audit():
    raw = json.loads((ROOT / INPUTS["raw"]).read_text())["comparisons"][:3]
    adjusted = json.loads((ROOT / INPUTS["adjusted"]).read_text())["double_centred"]
    records = []
    for name, rows in (("raw", raw), ("row-adjusted", adjusted)):
        for row in rows:
            for field in ("p_all", "p_cross", "p_partial"):
                records.append({"matrix": name, "comparison": row["label"], "test": field,
                                "p_nominal": row[field]})
    for row, value in zip(records, holm([r["p_nominal"] for r in records])):
        row["p_holm_table3_18"] = value
    raw_six = [r.copy() for r in records if r["matrix"] == "raw" and r["test"] != "p_all"]
    for row, value in zip(raw_six, holm([r["p_nominal"] for r in raw_six])):
        row["p_holm_raw_taxonomy_6"] = value
    return {
        "method": "Holm step-down, using unrounded deposited p-values",
        "main_family": "All 18 Table 3 tests: 3 time pairs × 3 summaries × raw/row-adjusted matrices; selected post hoc",
        "scope": "Within-table sensitivity, not study-wide confirmatory error control",
        "table3": records, "colleague_six_test_sensitivity": raw_six,
    }


def donor_audit():
    with (ROOT / INPUTS["donors"]).open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        "source": INPUTS["donors"], "n_response_rows": len(rows),
        "donor_pairs_per_response": sorted({int(r["n_donor_pairs"]) for r in rows}),
        "interpretation": "Each deposited response averages three paired donor log-ratio profiles. These are replicated source data; donor-level matrix reliability was not estimated in the cross-tissue analysis.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/biology/reviewer_followup.json")
    args = parser.parse_args()
    result = {
        "audit_date": "2026-11-06",
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in INPUTS.values()},
        "dose": dose_audit(), "multiplicity": multiplicity_audit(), "hepatocyte_donors": donor_audit(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    dose = result["dose"]
    print(f"Concentration-matched: {dose['matched_band_n']}/{dose['original_band_n']}; median {dose['matched_band_median_percent']:.6f}%")
    print(f"Wrote dose, Holm and donor audits to {args.out}")


if __name__ == "__main__":
    main()
