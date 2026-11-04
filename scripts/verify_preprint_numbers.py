#!/usr/bin/env python3
"""Check every load-bearing number in the preprint against the artifact it came from.

One row per claim: where it appears, the value as written, the artifact, and how to reach
the value inside it. The script loads each artifact, recomputes, and compares. Exit status
is non-zero on any mismatch, so this can gate a commit.

Two failure modes this exists to catch, both of which have happened here:

  - a number copied from a run log rather than from the stored artifact, where a later run
    overwrote the artifact and the document silently diverged;
  - a number corrected in one document and not in the others that quote it.

Usage:
    .venv/bin/python scripts/verify_preprint_numbers.py
    .venv/bin/python scripts/verify_preprint_numbers.py --show-all
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAND = (0.60, 0.78)


def load(rel: str):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return json.loads(p.read_text())


def band_rows(entry, key="per_compound", value="frac"):
    """Compounds whose ceiling falls inside the band, as {compound: value}."""
    out = {}
    for r in entry[key]:
        g = r.get("gene") or r
        v, b = g.get(value), g.get("bound")
        if v is None or b is None or not math.isfinite(v) or not math.isfinite(b):
            continue
        if BAND[0] <= b < BAND[1]:
            out[r["compound"]] = v
    return out


def median_pct(entry, key="per_compound", value="frac"):
    v = list(band_rows(entry, key, value).values())
    return 100.0 * st.median(v) if v else float("nan")


def sign_test(diffs):
    n = len(diffs)
    k = sum(1 for d in diffs if d > 0)
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return k, n, min(1.0, p)


def paired(a_entry, b_entry, a_key="per_compound", b_key="per_compound"):
    a, b = band_rows(a_entry, a_key), band_rows(b_entry, b_key)
    shared = sorted(set(a) & set(b))
    return sign_test([b[c] - a[c] for c in shared])


# --------------------------------------------------------------------------------------
# One entry per number quoted in the preprint.
#   claim : where it appears, so a failure is actionable
#   paper : the value as written
#   get   : callable returning the value recomputed from artifacts
# --------------------------------------------------------------------------------------
CHECKS: list[dict] = [

    # ---- 2.1 the reliability the paper is built on --------------------------------
    dict(claim="2.1 / Abstract: rho split-half 0.900",
         paper=0.900, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                     ["rho_reliability_halfset"]),
    dict(claim="2.1 / Abstract: Spearman-Brown to the full subject set 0.947",
         paper=0.947, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                     ["rho_reliability_fullset"]),

    # ---- 2.2 Table 1, every row, read inside the band ------------------------------
    dict(claim="Table 1: species, human vs rat hepatocyte 15.2%",
         paper=15.20, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/system_vs_species.json")["species_changed"])),
    dict(claim="Table 1: rat liver in vivo 24h vs 9h 52.6%",
         paper=52.61, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["TIME in vivo: 24h vs 9h"], "rows")),
    dict(claim="Table 1: rat liver single 24h vs repeat 4d 51.6%",
         paper=51.57, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["TIME in vivo: single 24h vs repeat 4d"], "rows")),
    dict(claim="Table 1: rat liver in vivo 24h vs 6h 42.2%",
         paper=42.17, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["TIME in vivo: 24h vs 6h"], "rows")),
    dict(claim="Table 1: rat liver in vivo 24h vs 3h 25.6%",
         paper=25.65, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["TIME in vivo: 24h vs 3h"], "rows")),
    dict(claim="Table 1: system, dish vs animal time matched 8.6%",
         paper=8.56, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["SYSTEM, TIME MATCHED: rat hep 24h vs rat liver 24h"], "rows")),
    dict(claim="Table 1: species and system together 3.5%",
         paper=3.53, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/matched_time_system.json")
                                ["SPECIES+SYSTEM, time matched: human hep 24h vs rat liver 24h"], "rows")),
    dict(claim="Table 1: human hepatocyte 24h vs 8h 44.8%",
         paper=44.81, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/mismatch_controls.json")
                                ["human hepatocyte | time 24 hr vs 8 hr"])),
    dict(claim="Table 1: rat hepatocyte 24h vs 8h 32.0%",
         paper=31.99, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/mismatch_controls.json")
                                ["rat hepatocyte | time 24 hr vs 8 hr"])),
    dict(claim="Table 1: human hepatocyte 24h vs 2h 12.4%",
         paper=12.37, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/mismatch_controls.json")
                                ["human hepatocyte | time 24 hr vs 2 hr"])),
    dict(claim="Table 1: rat hepatocyte 24h vs 2h 9.0%",
         paper=9.01, tol=0.05,
         get=lambda: median_pct(load("artifacts/biology/dili/mismatch_controls.json")
                                ["rat hepatocyte | time 24 hr vs 2 hr"])),

    # ---- 2.2 the paired tests that carry the ordering ------------------------------
    dict(claim="2.2: rat threefold vs species, in band, 16 of 19",
         paper=16, tol=0,
         get=lambda: paired(load("artifacts/biology/dili/system_vs_species.json")["species_changed"],
                            load("artifacts/biology/dili/mismatch_controls.json")
                            ["rat hepatocyte | time 24 hr vs 8 hr"])[0]),
    dict(claim="2.2: human threefold vs species, in band, 18 of 18",
         paper=18, tol=0,
         get=lambda: paired(load("artifacts/biology/dili/system_vs_species.json")["species_changed"],
                            load("artifacts/biology/dili/mismatch_controls.json")
                            ["human hepatocyte | time 24 hr vs 8 hr"])[0]),
    dict(claim="2.2: system vs species does not separate, 8 of 16",
         paper=8, tol=0,
         get=lambda: paired(load("artifacts/biology/dili/system_vs_species.json")["species_changed"],
                            load("artifacts/biology/dili/matched_time_system.json")
                            ["SYSTEM, TIME MATCHED: rat hep 24h vs rat liver 24h"],
                            b_key="rows")[0]),

    # ---- 2.3 the curve ------------------------------------------------------------
    dict(claim="2.3: r-bar 0.276",
         paper=0.2762, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/core_numbers.json")["collectible six"]["rbar"]),
    dict(claim="2.3: rho 0.265",
         paper=0.2654, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/core_numbers.json")["collectible six"]["rho"]),
    dict(claim="2.3: asymptote 0.536",
         paper=0.5361, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/core_numbers.json")["collectible six"]["asymptote"]),
    dict(claim="2.3: asymptote jackknife SE 0.021",
         paper=0.0208, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/curve_param_ci.json")["asymptote"]["se"]),
    dict(claim="2.3: two species reach 77% of the asymptote",
         paper=77.2, tol=0.1,
         get=lambda: 100.0 * load("artifacts/biology/concordance/core_numbers.json")
                     ["collectible six"]["rows"][1]["measured_best"]
                     / load("artifacts/biology/concordance/core_numbers.json")["collectible six"]["asymptote"]),
    dict(claim="2.3: six species reach 83%",
         paper=83.2, tol=0.1,
         get=lambda: 100.0 * load("artifacts/biology/concordance/core_numbers.json")
                     ["collectible six"]["rows"][5]["measured_best"]
                     / load("artifacts/biology/concordance/core_numbers.json")["collectible six"]["asymptote"]),

    # ---- 2.4 the phylogeny controls and the decomposition --------------------------
    dict(claim="2.4: 2h vs 6h cross-order +0.515",
         paper=0.515, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                      ["comparisons"] if c["label"] == "blood 2h vs 6h"][0]["spearman_cross"]),
    dict(claim="2.4: 2h vs 24h cross-order +0.084",
         paper=0.084, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                      ["comparisons"] if c["label"] == "blood 2h vs 24h"][0]["spearman_cross"]),
    dict(claim="2.4: 6h vs 24h cross-order +0.707",
         paper=0.707, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                      ["comparisons"] if c["label"] == "blood 6h vs 24h"][0]["spearman_cross"]),
    dict(claim="2.4: cross-order pairs 35",
         paper=35, tol=0,
         get=lambda: load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                     ["comparisons"][0]["n_cross_order"]),
    dict(claim="2.4: additive share at 2h 44.5%",
         paper=44.5, tol=0.1,
         get=lambda: 100.0 * load("artifacts/biology/concordance/rho_double_centred.json")
                     ["additive_share"]["2"]),
    dict(claim="2.4: additive share at 24h 56.2%",
         paper=56.2, tol=0.1,
         get=lambda: 100.0 * load("artifacts/biology/concordance/rho_double_centred.json")
                     ["additive_share"]["24"]),
    dict(claim="2.4: double-centred 2h vs 24h cross-order +0.417",
         paper=0.417, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_double_centred.json")
                      ["double_centred"] if "2h vs 24h" in c["label"]][0]["spearman_cross"]),
    dict(claim="2.4: main effects 2h vs 24h Spearman 0.600",
         paper=0.600, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_main_effects.json")
                      ["cross_time"] if c["pair"] == "2h vs 24h"][0]["main_effect_spearman"]),
    dict(claim="2.4: blood vs macrophage per-condition cross-order +0.371",
         paper=0.371, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                      ["comparisons"] if "macrophage [per-condition]" in c["label"]][0]["spearman_cross"]),
    dict(claim="2.4: blood vs macrophage pooled cross-order -0.086",
         paper=-0.086, tol=5e-4,
         get=lambda: [c for c in load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")
                      ["comparisons"] if "macrophage [pooled]" in c["label"]][0]["spearman_cross"]),

    # ---- 2.5 what the ordering survives --------------------------------------------
    dict(claim="2.5 / Figure 1: chimpanzee 0.676",
         paper=0.6756, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/reconstruct_metric_audit.json")
                     ["Pan troglodytes"]["r"]),
    dict(claim="2.5 / Figure 1: rat 0.206",
         paper=0.2061, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/reconstruct_metric_audit.json")
                     ["Rattus norvegicus"]["r"]),
    dict(claim="2.5: 33 of 36 species pairs separate",
         paper=33, tol=0,
         get=lambda: sum(1 for v in load("artifacts/biology/concordance/fig1_paired.json").values()
                         if v.get("sig_strat"))),
    dict(claim="2.5: gap on the unfiltered axis +0.002",
         paper=0.0025, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/gap_ci.json")["all genes"]["gap"]),
    dict(claim="2.5: gap at the loosest filter -0.026, interval excludes zero",
         paper=-0.0257, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/gap_ci.json")["all 10 expressed (>0.5)"]["gap"]),

    # ---- 5 Limitations: the human self-prediction baseline -------------------------
    dict(claim="Abstract / Limitations: human self-prediction 0.816 in correlation units",
         paper=0.816, tol=2e-3, needs_blood=True,
         get=lambda: human_self()),
    dict(claim="Limitations: nine-species equal weight 0.642",
         paper=0.642, tol=5e-4,
         get=lambda: load("artifacts/biology/concordance/reconstruct_metric_audit.json")
                     ["uniform (all 9)"]["r"]),

    # ---- 2.7 the organ-injury null ------------------------------------------------
    # These four were written into the manuscript without a check and three of them had
    # drifted: the ALT association was quoted as +0.017 here, +0.015 in the internal
    # results and +0.02 in the README, against +0.010 on disk; the in vivo p bound was
    # quoted at 0.044 against a true maximum of 0.0445; and the hepatocyte comparison
    # count named no family.
    dict(claim="2.7: in vivo ALT prediction, worst permutation p <= 0.045",
         paper=0.045, tol=1e-3,
         get=lambda: max(alt_run(e, th)["results"]["rat liver in vivo (same animals)"]["p_auc"]
                         for e in ALT_ENDPOINTS for th in ALT_THRESHOLDS)),
    dict(claim="2.7: hepatocyte arm-by-setting comparisons deposited = 45",
         paper=45, tol=0,
         get=lambda: float(len(hepatocyte_arms()))),
    dict(claim="2.7: exactly one hepatocyte comparison clears its own null",
         paper=1, tol=0,
         get=lambda: float(sum(1 for auc, p95 in hepatocyte_arms() if auc > p95))),
    dict(claim="2.7 / Limitations: fraction of ceiling against log2 ALT, Spearman +0.010",
         paper=0.0104, tol=1e-3,
         get=lambda: alt_association()),

    # ---- 2.3 / rule 2: percentages of the asymptote -------------------------------
    # Design rule 2 quoted 77% and 83% side by side without saying that the first is the
    # best of fifteen two-species panels and the second the only six-species one. The
    # average two-species panel reaches 65%, which is the number a chooser without prior
    # knowledge gets, so all three are checked here.
    dict(claim="Abstract / rule 2: best two species reach 77% of the asymptote",
         paper=77.2, tol=0.3,
         get=lambda: panel_pct(2, "best")),
    dict(claim="rule 2: the average two-species panel reaches 65%",
         paper=65.2, tol=0.3,
         get=lambda: panel_pct(2, "mean")),
    dict(claim="Abstract / 2.3 / rule 2: all six reach 83% of the asymptote",
         paper=83.2, tol=0.3,
         get=lambda: panel_pct(6, "best")),

    # ---- 2.4: the double-centred matrix, partialled on taxonomic distance ---------
    # These three answer the objection that double-centring passes only because it
    # removed the part carrying relatedness. They were computed and deposited but not
    # reported; the raw matrix's partials were.
    dict(claim="2.4: double-centred partial, 2h vs 6h, +0.849",
         paper=0.8491, tol=5e-4, get=lambda: dc_partial(0)),
    dict(claim="2.4: double-centred partial, 2h vs 24h, +0.643",
         paper=0.6434, tol=5e-4, get=lambda: dc_partial(1)),
    dict(claim="2.4: double-centred partial, 6h vs 24h, +0.851",
         paper=0.8512, tol=5e-4, get=lambda: dc_partial(2)),
    dict(claim="2.4: all three double-centred partials at Mantel p = 2.5e-4",
         paper=0.00025, tol=1e-5,
         get=lambda: max(r["p_partial"] for r in
                         load("artifacts/biology/concordance/rho_double_centred.json")["double_centred"])),

    # ---- 2.4 / Figure 3: the two cross-panel estimators --------------------------
    # Figure 3 and the text use the same two-estimator artifact. The 2026-11-04
    # correction averages tied ranks for taxonomy controls; these all/cross-order
    # estimates are unchanged and guard against restoring an older panel run.
    dict(claim="2.4: macrophage per-condition, all pairs +0.503",
         paper=0.503, tol=1e-3, get=lambda: xpanel("percond", "spearman_all")),
    dict(claim="2.4: macrophage per-condition, cross-order +0.371",
         paper=0.3714, tol=1e-3, get=lambda: xpanel("percond", "spearman_cross")),
    dict(claim="2.4: macrophage pooled, all pairs +0.333",
         paper=0.3333, tol=1e-3, get=lambda: xpanel("pooled", "spearman_all")),
    dict(claim="2.4: macrophage pooled, cross-order -0.086 (opposite sign)",
         paper=-0.0857, tol=1e-3, get=lambda: xpanel("pooled", "spearman_cross")),

    # ---- Methods: numbers the rewritten section states ----------------------------
    dict(claim="Methods: the curve's parameters are jackknifed over 31 subjects",
         paper=31, tol=0,
         get=lambda: float(load("artifacts/biology/concordance/curve_param_ci.json")["n_subjects"])),
    dict(claim="Methods / 2.5: the loosest expression filter retains 6,477 genes",
         paper=6477, tol=0, get=lambda: geneset_n("0.5")),
    dict(claim="Methods / 2.5: the tightest expression filter retains 2,973 genes",
         paper=2973, tol=0, get=lambda: geneset_n("2.0")),

    # ---- Table 1: the reliability partials, new to the manuscript ----------------
    dict(claim="Table 1: 2 h reliability, partial on taxonomy 0.738",
         paper=0.738, tol=5e-4, get=lambda: rel_row(0, "partial_taxdist")),
    dict(claim="Table 1: 6 h reliability, partial on taxonomy 0.945",
         paper=0.945, tol=5e-4, get=lambda: rel_row(1, "partial_taxdist")),
    dict(claim="Table 1: 24 h reliability, partial on taxonomy 0.922",
         paper=0.922, tol=5e-4, get=lambda: rel_row(2, "partial_taxdist")),

    # ---- Table 3: the all-pairs columns, new to the manuscript -------------------
    dict(claim="Table 3: raw 2 h vs 6 h, all pairs +0.684",
         paper=0.684, tol=5e-4, get=lambda: xtime("RAW", 0, "spearman_all")),
    dict(claim="Table 3: raw 2 h vs 24 h, all pairs +0.438",
         paper=0.438, tol=5e-4, get=lambda: xtime("RAW", 1, "spearman_all")),
    dict(claim="Table 3: raw 6 h vs 24 h, all pairs +0.848",
         paper=0.848, tol=5e-4, get=lambda: xtime("RAW", 2, "spearman_all")),
    dict(claim="Table 3: double-centred 2 h vs 6 h, all pairs +0.878",
         paper=0.878, tol=5e-4, get=lambda: xtime("DC", 0, "spearman_all")),
    dict(claim="Table 3: double-centred 2 h vs 24 h, all pairs +0.717",
         paper=0.717, tol=5e-4, get=lambda: xtime("DC", 1, "spearman_all")),
    dict(claim="Table 3: double-centred 6 h vs 24 h, all pairs +0.879",
         paper=0.879, tol=5e-4, get=lambda: xtime("DC", 2, "spearman_all")),

    # ---- 2.7: the bootstrap intervals, deposited but never reported --------------
    dict(claim="2.7: ALT at 3-fold, in vivo interval lower bound 0.425 (spans 0.5)",
         paper=0.425, tol=1e-3, get=lambda: vivo_ci("ALTIUL", "3.0", 0)),
    dict(claim="2.7: ALT at 3-fold, in vivo interval upper bound 0.903",
         paper=0.903, tol=1e-3, get=lambda: vivo_ci("ALTIUL", "3.0", 1)),
    dict(claim="2.7: AST at 3-fold, in vivo interval 0.755 to 0.993",
         paper=0.755, tol=1e-3, get=lambda: vivo_ci("ASTIUL", "3.0", 0)),

    # ---- Discussion: the external check on a third party's published model --------
    dict(claim="Discussion: their baseline reproduces at Pearson delta 0.4515",
         paper=0.4515, tol=5e-4,
         get=lambda: rh("split_0", "pdex_static/pearson_delta_mean")),
    dict(claim="Discussion: their baseline reproduces at PR-AUC 0.4034",
         paper=0.4034, tol=5e-4, get=lambda: rh("split_0", "state/pr_auc_mean")),
    dict(claim="Discussion: random panels give mean Pearson delta 0.5600",
         paper=0.5600, tol=1e-3, get=lambda: rh_mean("randpanel", "state/pearson_delta_mean")),
    dict(claim="Discussion: clustered panels give mean Pearson delta 0.5127",
         paper=0.5127, tol=1e-3, get=lambda: rh_mean("cluster", "state/pearson_delta_mean")),
    dict(claim="Discussion: random panels give mean Spearman 0.5987",
         paper=0.5987, tol=1e-3, get=lambda: rh_mean("randpanel", "state/spearman_lfc_sig_mean")),
    dict(claim="Discussion: clustered panels give mean Spearman 0.5158",
         paper=0.5158, tol=1e-3, get=lambda: rh_mean("cluster", "state/spearman_lfc_sig_mean")),
    dict(claim="Discussion: the four random panels span 0.020 in Pearson delta",
         paper=0.020, tol=2e-3, get=lambda: rh_span("randpanel", "state/pearson_delta_mean")),
    dict(claim="Discussion: clustered and random panels are disjoint on Pearson delta",
         paper=1, tol=0, get=lambda: rh_disjoint("state/pearson_delta_mean")),
    dict(claim="Discussion: clustered and random panels are disjoint on Spearman",
         paper=1, tol=0, get=lambda: rh_disjoint("state/spearman_lfc_sig_mean")),
    dict(claim="Discussion: PR-AUC is the metric that does NOT separate",
         paper=0, tol=0, get=lambda: rh_disjoint("state/pr_auc_mean")),

    # ---- the third dataset: the detector, and the claim it cost us ---------------
    dict(claim="2.2: LINCS, 948 plates", paper=948, tol=0, get=lambda: lincs("plates")),
    dict(claim="2.2: LINCS, median 20 control wells per plate",
         paper=20, tol=0, get=lambda: lincs("ctrl_per_plate_median")),
    dict(claim="2.2: LINCS, median 350 treated wells per plate",
         paper=350, tol=0, get=lambda: lincs("treated_per_plate_median")),
    dict(claim="2.2: LINCS, 1,785 compounds", paper=1785, tol=0, get=lambda: lincs("n_compounds")),
    dict(claim="2.2: LINCS, 1,912 units with a ceiling",
         paper=1912, tol=0, get=lambda: lincs("ceiling", "n_usable")),
    dict(claim="2.2: LINCS, 471 of them exceed the bound",
         paper=471, tol=0, get=lambda: lincs("ceiling", "n_above_one")),
    dict(claim="2.2: LINCS, the worst reaches 2.68",
         paper=2.679, tol=5e-3, get=lambda: lincs("ceiling", "max")),
    dict(claim="Discussion: cluster-holdout ridge spans 0.168 to 0.178",
         paper=0.1679, tol=1e-3, get=lambda: split_span("cl", "ridge", min)),
    dict(claim="Discussion: cluster-holdout ridge upper end 0.178",
         paper=0.1783, tol=1e-3, get=lambda: split_span("cl", "ridge", max)),
    dict(claim="Discussion: the metadata-only baseline spans 0.045 to 0.050",
         paper=0.0453, tol=1e-3, get=lambda: split_span("cl", "meta", min)),
    dict(claim="Discussion: the metadata baseline upper end 0.050",
         paper=0.0502, tol=1e-3, get=lambda: split_span("cl", "meta", max)),
    dict(claim="Discussion: the cluster holdout costs 0.016 to 0.017",
         paper=0.0166, tol=1e-3, get=lambda: degradation(max)),
    dict(claim="Discussion: ridge never falls below the baseline under either split",
         paper=1, tol=0, get=lambda: never_flips()),

    # ---- the donor ceiling of a third party's panel ------------------------------
    dict(claim="Discussion: the donor consensus reaches 0.4315 on their test set",
         paper=0.4315, tol=1e-3, get=lambda: donor_ceiling_on_test()),
    dict(claim="Discussion: their reported score in the same unit is 0.4515",
         paper=0.4515, tol=5e-4,
         get=lambda: rh("split_0", "pdex_static/pearson_delta_mean")),
    dict(claim="Discussion: the ceiling spans 0.032 to 0.577 across their cell types",
         paper=0.0321, tol=1e-3,
         get=lambda: load("artifacts/rhaister_benchmark/donor_ceiling.json")["overall"]["ceiling_min"]),
    dict(claim="Discussion: 1,619 units carry a cell count",
         paper=1619, tol=0, get=lambda: float(depth()["n_units"])),
    dict(claim="Discussion: cell counts span 2 to 3,074",
         paper=3074, tol=0, get=lambda: depth()["cells_range"][1]),
    dict(claim="Discussion: pooled, cells and ceiling correlate +0.785",
         paper=0.785, tol=2e-3,
         get=lambda: depth()["spearman_cells_vs_ceiling_overall"]),
    dict(claim="Discussion: within a cell type the median correlation is +0.08",
         paper=0.079, tol=2e-3, get=lambda: depth()["within_median"]),
    dict(claim="Discussion: the ceiling has levelled off by twelve donors",
         paper=0.0096, tol=1e-3, get=lambda: max_last_step()),
    dict(claim="Discussion: the highest cell-type ceiling is 0.577",
         paper=0.5767, tol=1e-3,
         get=lambda: load("artifacts/rhaister_benchmark/donor_ceiling.json")["overall"]["ceiling_max"]),
]

# The 62 test perturbations of their split_0, read from splits/parse/split_0/split.toml on the
# compute host. Held here so the comparison is against exactly the rows they scored, 62 x 12 =
# 744, which is the n_samples their own run reports.
RHAISTER_TEST = (
    'C5a TWEAK LIF IL-17C FGF-beta BAFF FLT3L IFN-lambda2 4-1BBL IL-17E IL-2 IL-19 PRL IL-16 '
    'IL-7 GM-CSF IL-17A IL-13 IL-36-alpha PSPN EPO IL-9 IL-31 IL-11 LT-alpha1-beta2 TL1A '
    'IL-32-beta IFN-omega IL-8 Noggin IFN-epsilon IL-17F Leptin IL-33 IL-35 OSM IL-34 HGF '
    'IL-1Ra LIGHT CD27L SCF IL-1-alpha TGF-beta1 CD30L IL-24 IFN-gamma IFN-alpha1 IFN-lambda1 '
    'GITRL VEGF ADSF IL-15 EGF IL-3 IL-17B IL-6 IL-4 IL-10 CT-1 IFN-lambda3 TRAIL').split()


def depth():
    return load("artifacts/rhaister_benchmark/depth_vs_ceiling.json")


def max_last_step() -> float:
    """Largest absolute final step of the donor-count curve, across cell types.

    The ceiling is measured against the consensus of the other donors, so a bigger panel
    would give a better consensus and a higher ceiling. This asks whether that has stopped
    mattering by the twelve donors in hand: if the step from eleven to twelve is within
    noise, the number is a property of the data rather than of the collection.
    """
    cts = load("artifacts/rhaister_benchmark/ceiling_saturation.json")["cell_types"]
    return max(abs(v["last_step"]) for v in cts.values())


def donor_ceiling_on_test() -> float:
    """Leave-one-donor-out ceiling over exactly the perturbations their split_0 scored."""
    pc = load("artifacts/rhaister_benchmark/donor_ceiling.json")[
        "by_cell_type"]["B_Intermediate_Memory"]["per_cytokine"]
    hit = [pc[c]["mean"] for c in RHAISTER_TEST if c in pc]
    if len(hit) != len(RHAISTER_TEST):
        raise ValueError(f"expected {len(RHAISTER_TEST)} test perturbations, found {len(hit)}")
    return st.mean(hit)


def _lincs():
    return load("artifacts/lincs_scale/scale_test.json")


def lincs(*path) -> float:
    d = _lincs()
    for k in path:
        d = d[k]
    return float(d)


def split_span(split: str, model: str, agg) -> float:
    """Across the three cluster granularities, the smallest or largest value of one series."""
    s = _lincs()["splits"]
    return agg(s[k]["row_weighted"][split][model] for k in s)


def degradation(agg) -> float:
    """How much the cluster holdout costs the model, across granularities."""
    s = _lincs()["splits"]
    return agg(s[k]["row_weighted"]["rnd"]["ridge"] - s[k]["row_weighted"]["cl"]["ridge"]
               for k in s)


def never_flips() -> float:
    """1 if the model beats the no-input baseline under BOTH splits at every granularity.

    This is the claim the scale test cost us: the earlier reading was that the two splits give
    opposite verdicts, which does not hold here.
    """
    s = _lincs()["splits"]
    return float(all(s[k]["row_weighted"][sp]["ridge"] > s[k]["row_weighted"][sp]["meta"]
                     for k in s for sp in ("cl", "rnd")))


def _rhaister():
    return load("artifacts/rhaister_benchmark/all.json")


def rh(split: str, field: str) -> float:
    return _rhaister()[split][field]


def _group(prefix: str, field: str):
    d = _rhaister()
    return [v[field] for k, v in d.items() if k.startswith(prefix)]


def rh_mean(prefix: str, field: str) -> float:
    return st.mean(_group(prefix, field))


def rh_span(prefix: str, field: str) -> float:
    g = _group(prefix, field)
    return max(g) - min(g)


def rh_disjoint(field: str) -> float:
    """1 if every clustered panel falls below every random one on this metric, else 0."""
    c, r = _group("cluster", field), _group("randpanel", field)
    return float(max(c) < min(r))


def vivo_ci(endpoint: str, threshold: str, end: int) -> float:
    """An end of the bootstrap interval on the in vivo arm of one endpoint-by-threshold run."""
    return alt_run(endpoint, threshold)["results"]["rat liver in vivo (same animals)"]["auc_ci95"][end]


def rel_row(i: int, field: str) -> float:
    """A split-half reliability row of Table 1, in timepoint order."""
    return load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")["reliability"][i][field]


def xtime(which: str, i: int, field: str) -> float:
    """A cross-time row of Table 3, preserving each row's permutation-run source."""
    if which == "RAW":
        return load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")["comparisons"][i][field]
    return load("artifacts/biology/concordance/rho_double_centred.json")["double_centred"][i][field]


def geneset_n(threshold: str) -> float:
    """Genes surviving a resting-expression filter of Section 2.5.

    annotation_control.json carries the count inside the gene-set label, as
    "expressed in all 10 species (log2TPM>0.5): 6,477", so the label is the artifact and
    is parsed rather than re-derived. The threshold also confirms the Methods wording.
    """
    by = load("artifacts/biology/concordance/annotation_control.json")["by_geneset"]
    hit = [k for k in by if f"log2TPM>{threshold})" in k]
    if len(hit) != 1:
        raise ValueError(f"expected one gene set at log2TPM>{threshold}, found {len(hit)}")
    return float(hit[0].rsplit(":", 1)[1].strip().replace(",", ""))


def xpanel(estimator: str, field: str) -> float:
    """A cross-panel statistic from the artifact Figure 3 is now drawn from."""
    rows = load("artifacts/biology/concordance/rho_cross_panel_both_estimators.json")["comparisons"]
    hit = [r for r in rows if estimator in r.get("kind", "")]
    if len(hit) != 1:
        raise ValueError(f"expected one {estimator} row, found {len(hit)}")
    return hit[0][field]


def dc_partial(i: int) -> float:
    """Tie-aware double-centred partials, in the same order as Table 3.

    Raw rows and their p-values come from rho_cross_panel_both_estimators.json;
    double-centred rows and their p-values come from rho_double_centred.json.
    """
    return load("artifacts/biology/concordance/rho_double_centred.json")["double_centred"][i]["partial_taxdist"]


def panel_pct(k: int, how: str) -> float:
    """The best or mean panel of size k, as a percentage of the pool's analytic asymptote."""
    asym = load("artifacts/biology/concordance/curve_param_ci.json")["asymptote"]["point"]
    vals = [s["r"] for s in load("artifacts/biology/concordance/collectible_panel.json")["subsets"]
            if len(s["panel"]) == k]
    agg = max(vals) if how == "best" else st.mean(vals)
    return 100.0 * agg / asym

ALT_ENDPOINTS = ("ALTIUL", "ASTIUL", "COMPOSITE")
ALT_THRESHOLDS = ("1.5", "2.0", "3.0")


def alt_run(endpoint: str, threshold: str):
    return load(f"artifacts/biology/dili/alt_prediction/pred_{endpoint}_{threshold}.json")


def hepatocyte_arms() -> list[tuple[float, float]]:
    """Every deposited hepatocyte arm-by-setting comparison, as (AUC, its null's p95).

    Three arms at gene level -- rat, human, and the two averaged -- plus rat and human
    aggregated to ontology terms, over nine endpoint-and-threshold settings. The response
    magnitude and |log2| > 1 sub-variants are separate reductions of the same arms and are
    not part of this family; the in vivo arm is the positive control, not a hepatocyte.
    """
    out = []
    for e in ALT_ENDPOINTS:
        for th in ALT_THRESHOLDS:
            for name, v in alt_run(e, th)["results"].items():
                if "vivo" in name or "magnitude" in name or "log2" in name:
                    continue
                out.append((v["auc"], v["auc_null_p95"]))
    return out


def alt_association() -> float:
    """Spearman between a compound's human-vs-rat fraction of ceiling and its log2 ALT."""
    from scipy.stats import spearmanr

    alt = alt_run("ALTIUL", "1.5")
    endpoint = dict(zip(alt["compounds"], alt["alt_log2"]))
    xs, ys = [], []
    for row in load("artifacts/biology/dili/system_vs_species.json")["species_changed"]["per_compound"]:
        frac = row.get("gene", {}).get("frac")
        y = endpoint.get(row["compound"])
        if frac is None or y is None or not (math.isfinite(frac) and math.isfinite(y)):
            continue
        xs.append(frac)
        ys.append(y)
    return float(spearmanr(xs, ys).statistic)


def human_self() -> float:
    """Held-out human condition predicted from that human's average over its others."""
    import csv
    D = ROOT / "artifacts/gse249010_v1/processed"
    rows = list(csv.DictReader((D / "response_metadata.tsv").open(), delimiter="\t"))
    X = np.load(D / "responses.npy", mmap_mode="r")
    H = "Homo sapiens"
    sp = sorted({r["species"] for r in rows})
    tk = sorted({r["task_id"] for r in rows})
    tk = [t for t in tk if all(any(r["species"] == s and r["task_id"] == t for r in rows) for s in sp)]
    mean = {t: np.nanmean(np.asarray(X[[i for i, r in enumerate(rows)
                                        if r["species"] == H and r["task_id"] == t]]), axis=0)
            for t in tk}

    def pear(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    return float(np.mean([pear(np.mean([mean[u] for u in tk if u != t], axis=0), mean[t]) for t in tk]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-all", action="store_true", help="print passing checks too")
    a = ap.parse_args()

    failed, skipped = [], []
    for c in CHECKS:
        try:
            got = c["get"]()
        except FileNotFoundError as e:
            skipped.append((c["claim"], f"missing artifact {e}"))
            continue
        except Exception as e:
            failed.append((c["claim"], c["paper"], None, f"{type(e).__name__}: {e}"))
            continue
        tol = c.get("tol", 5e-4)
        ok = abs(got - c["paper"]) <= tol
        if not ok:
            failed.append((c["claim"], c["paper"], got, f"differs by {abs(got - c['paper']):.4g}"))
        elif a.show_all:
            print(f"  OK    {c['claim']}\n          paper {c['paper']}  artifact {got:.4f}")

    print(f"\n{len(CHECKS)} numbers checked against their artifacts")
    print(f"{len(CHECKS) - len(failed) - len(skipped)} agree, {len(failed)} disagree, {len(skipped)} skipped")

    for claim, why in skipped:
        print(f"\n  SKIP  {claim}\n          {why}")
    for claim, paper, got, why in failed:
        shown = "raised" if got is None else f"{got:.4f}"
        print(f"\n  FAIL  {claim}\n          paper says {paper}, artifact gives {shown} ({why})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
