# Data catalogue

This is the data deposit for Section 5 of **Reliability and limits of cross-species
transcriptomic comparisons**. All paths are relative to the repository root.
`DATA_MANIFEST.json` records file sizes and SHA-256 checksums.

## Source studies

| Study | Public source | Included here |
|---|---|---|
| Ten-species blood | [GEO GSE249010](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE249010) | Processed response matrix, genes, sample/response metadata, SOFT metadata, and provenance in `artifacts/gse249010_v1/` |
| Five-species hepatocytes | [GEO GSE330151](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330151) | Response metadata and preprocessing provenance in `artifacts/biology/gse330151/`; summary estimates in `artifacts/biology/concordance/hepatocyte.json` and `artifacts/biology/concordance/panel_hepatocyte.json` |
| TG-GATEs | BioStudies [S-DIXA-AN-004](https://www.ebi.ac.uk/biostudies/studies/S-DIXA-AN-004), [005](https://www.ebi.ac.uk/biostudies/studies/S-DIXA-AN-005), [007](https://www.ebi.ac.uk/biostudies/studies/S-DIXA-AN-007) | Compound-level derived results, serum endpoints, orthologue mappings, and selected sample metadata |
| Seven yeasts | [ArrayExpress E-MTAB-7022](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-7022) and related accessions recorded in the preprocessing provenance | Response metadata and provenance in `artifacts/biology/selenite/`; summary estimates in `artifacts/biology/concordance/selenite.json` |
| FAANG macrophages | [PMC7386301, Supplementary Table S3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386301/); [ENA PRJEB22535](https://www.ebi.ac.uk/ena/browser/view/PRJEB22535) and related projects | Processed matrix, genes, response metadata and provenance in `artifacts/faang_bmdm/processed/` |
| LINCS L1000 Phase 2 | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) | `artifacts/lincs_scale/scale_test.json` |
| Rhaister / Parse PBMC | Authors' [model/code release](https://huggingface.co/tahoebio/Rhaister) and [Parse data release](https://huggingface.co/datasets/tahoebio/parse-de-rhaister) | Four benchmark JSON files and metric-only logs in `artifacts/rhaister_benchmark/` |

GSE3284, GSE3026, GSE36177 and GSE48119 are literature comparators discussed in the
manuscript, not inputs to an analysis in this study. They are not duplicated here.
The authors' model weights and the large upstream expression downloads are not redistributed.

## Manuscript results

Unless another directory is specified, the following filenames are under
`artifacts/biology/concordance/`.

| Manuscript item | Deposited files | Analysis / plotting code |
|---|---|---|
| Figure 1; Figure 4a; Figure 5a/b | Illustrative inputs constructed in code; not empirical measurements | `scripts/make_preprint_figures.py` |
| Figure 2 | Deposited response metadata for blood, hepatocytes, macrophages and yeasts; TG-GATEs compound-level results; LINCS and Parse scope annotations. Exact sources and counts: `dataset_overview.json` | `collect_preprint_datasets.py`, `make_preprint_figures.py` |
| Table 1 | `rho_cross_panel_both_estimators.json` | `run_rho_reliability_and_phylogeny.py` |
| Figure 3a | `reconstruct_metric_audit.json`, `fig1_ci.json` | `make_preprint_figures.py`, `controls/run_fig1_intervals.py` |
| Figure 3b-d; Section 2.3 | `core_numbers.json`, `collectible_panel.json`, `panel_blood.json`, `panel_hepatocyte.json`, `scaling_curve.json` | `make_preprint_figures.py`, `run_collectible_panel.py`, `run_panel_design.py` |
| Figure 4b-d; Section 2.4 | `rho_cross_panel_both_estimators.json` | `run_rho_reliability_and_phylogeny.py`, `plot_rho_phylogeny.py` |
| Table 3; Section 2.4 row adjustment and Holm sensitivity | `rho_cross_panel_both_estimators.json`, `rho_double_centred.json`, `rho_main_effects.json`, `artifacts/biology/reviewer_followup.json` | `controls/run_double_centred.py`, `controls/run_main_effects.py`, `audit_dose_and_multiplicity.py` |
| Figure 5c/d | `reconstruct_specificity.json` | `run_reconstruction_specificity.py`, `make_preprint_figures.py` |
| Table 2 | `artifacts/biology/dili/matched_time_system.json`, `mismatch_controls.json`, `system_vs_species.json` | `run_matched_time_system.py`, `run_mismatch_controls.py`, `run_system_vs_species.py` |
| Table 2 concentration sensitivity; GSE330151 donor audit | `artifacts/biology/reviewer_followup.json`; source sample attributes and donor-count metadata | `audit_dose_and_multiplicity.py` |
| Section 2.5 controls | `annotation_control.json`, `baseline_control.json`, `fig1_ci.json`, `fig1_paired.json`, `fig1_ceiling_control.json`, `curve_by_geneset.json`, `curve_param_ci.json`, `gap_ci.json` | Corresponding scripts in `scripts/controls/` |
| Missing-value controls | `artifacts/gse249010_v1/na_audit.json`, `na_masked_fig1.json` | `controls/run_na_audit.py`, `controls/run_na_masked_fig1.py` |
| Section 2.7 | All nine `artifacts/biology/dili/alt_prediction/pred_{endpoint}_{threshold}.json` files | `run_alt_prediction.py`; endpoints ALTIUL, ASTIUL, COMPOSITE; thresholds 1.5, 2.0, 3.0 |
| Serum endpoints | `artifacts/biology/dili/rat_vivo_alt.json`, `rat_vivo_markers.json` | `build_vivo_markers.py`, `run_dili_concordance.py` |
| LINCS holdout comparison; historical ratio excluded from inference | `artifacts/lincs_scale/scale_test.json` | `run_lincs_scale_test.py` |
| Rhaister panel-composition comparison | `artifacts/rhaister_benchmark/all.json`, `logs/*.out` | `make_rhaister_cluster_splits.py` constructs splits; the external Rhaister code evaluates them; `collect_rhaister_metrics.py` collects the metric lines |
| Rhaister donor-consensus baseline and sensitivity | `donor_ceiling.json`, `ceiling_saturation.json`, `depth_vs_ceiling.json` under `artifacts/rhaister_benchmark/` | Corresponding `run_rhaister_*.py` scripts |

Script filenames in the table are relative to `scripts/`. The identifier bridge is
`artifacts/reference/ensg_symbol_bridge.tsv`. The five figures use deposited results,
metadata or explicitly illustrative inputs; plotting does not rerun statistical analyses.

`core_numbers.json` and `reconstruct_metric_audit.json` are retained source-data summaries.
Their exact original producing scripts are not available in this snapshot. The figures and
registered numerical checks can be reproduced from these summaries; this is distinct from
an end-to-end reconstruction of those two artifacts from primary data.

## Additional protocol outputs

`artifacts/tg_gates_rat_human_v1/cluster_holdout/` contains `results.json`, `nonlinear.json`,
`metadata_control.json` and `bootstrap.json`, from the four corresponding control scripts.
These are the additional evaluation-protocol outputs identified in the original availability
statement; they are not additional manuscript figures. Earlier claims about split-dependent
verdict reversal should not be inferred from this deposit: use the current Discussion.

The Rhaister log excerpts retain only `RESULT key=value` lines needed to reconstruct
`all.json`; local paths and terminal diagnostics are omitted. A metadata provenance file
replaces local absolute paths with source filenames. These export transformations are noted
in the manifest and do not alter numerical results.

The reproducible normalisation audit is `artifacts/biology/replicate_normalisation_audit.json`,
built by `scripts/audit_replicate_normalisation.py`. It contains exact mathematical
counterexamples and summaries of archived scores, not a reanalysis of raw expression data.
