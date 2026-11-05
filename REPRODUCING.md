# Reproducing the publication snapshot

Run from the repository root. Use Python 3.12 or newer; the dependency versions used
for the release checks are recorded in `requirements.txt`.

## Checks and figures using only deposited data

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/verify_preprint_numbers.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/make_preprint_figures.py
```

The first command after installation verifies the original release checksums. Run it before
editing or rebuilding outputs. The numerical verifier should report 140 agreements, zero
disagreements and zero skips. It checks a registry of claims, not arbitrary prose changes.

The figure generator writes `docs/paper/tex/figures/Fig1` through `Fig5` as vector PDFs, editable SVGs and
600 dpi RGB PNGs. Its illustrative matrices are identified in the manuscript legends. Font rendering and PDF
metadata can differ across software versions; numerical source data are the reproducibility
target.

Figure 2 also exports `Fig2_source_data.json`, including metadata counts, their counting
definitions and source hashes. The blood grid includes only conditions shared by all ten
species (387 response rows); one of the 388 deposited rows lies outside that grid. TG-GATEs
counts are distinct compounds per comparison and must not be summed as independent cohorts.

## Manuscript build

Install Pandoc and either XeLaTeX or Tectonic, then run:

```bash
python scripts/build_preprint.py
```

This reads `docs/paper/preprint.md`, generates the editable LaTeX, and synchronises the two
PDF copies. The TeX Gyre fonts used by the template are supplied by TeX distributions. For
direct LaTeX editing, see `docs/paper/tex/README.md`.

## Recompute reliability from included matrices

The blood response matrix is 388 rows by 25,638 genes; the matching gene axis and response
metadata are included. Macrophage input and the ENSG-to-symbol bridge are also included.
The calculation uses 20,000 species-label permutations for all-pair/cross-order tests and
4,000 for partial correlations, so it takes longer than the quick checks.

```bash
python scripts/run_rho_reliability_and_phylogeny.py \
  --blood artifacts/gse249010_v1/processed \
  --macrophage artifacts/faang_bmdm/processed \
  --bridge artifacts/reference/ensg_symbol_bridge.tsv \
  --out artifacts/biology/concordance/rho_cross_panel_both_estimators.json
python scripts/controls/run_double_centred.py
python scripts/controls/run_main_effects.py
python scripts/run_collectible_panel.py artifacts/biology/concordance/collectible_panel.json
```

These commands overwrite deposited outputs. Use a separate checkout when comparing reruns
with the release. Defaults preserve the deposited seeds. Figures and Tables 1/3 use average
ranks for tied taxonomic distances. The tests check tie handling and table/source agreement.

## Analyses requiring external inputs

Install `requirements-analysis.txt` for the additional input readers. Source links are in
`DATA_AVAILABILITY.md`; the release does not claim a one-command raw-data rebuild.

| Analysis | Additional inputs | Entry points |
|---|---|---|
| Blood preprocessing and expression/NA controls | `GSE249010_processed_data_TPM.txt.gz`, placed in `artifacts/gse249010_v1/data/`; GEO SOFT is included | `prepare_gse249010_pilot.py`; `controls/run_annotation_control.py`, `run_baseline_control.py`, `run_curve_by_geneset.py`, `run_gap_ci.py`, `run_na_audit.py`, `run_na_masked_fig1.py` |
| Five-species hepatocytes | GSE330151 per-species count matrices and series metadata; five-way orthologue axis | `fetch_five_species_orthologs.py`, `prepare_gse330151.py` |
| Yeast | E-MTAB-7022 and related raw arrays/SDRF files, yeast orthologue axis and Candida identifier map | `build_yeast_ortholog_axis.py`, `prepare_selenite.py` |
| FAANG preprocessing | PMC7386301 Supplementary Table S3 spreadsheet | `prepare_faang_bmdm.py`; processed output is already included |
| TG-GATEs Table 2 and endpoint prediction | Expression matrices, probe and experimental annotations; aligned human/rat gene-axis JSON; the four prepared NPZ axes (human hepatocyte, rat hepatocyte, single-dose liver, repeat-dose liver) | `prepare_tggates_hep.py`, `run_system_vs_species.py`, `run_mismatch_controls.py`, `run_matched_time_system.py`, `run_alt_prediction.py` |
| Additional TG-GATEs holdout runs | `pairs_rat_X.tsv` and `pairs_human_Y.tsv` in `artifacts/tg_gates_rat_human_v1/data/`; pair metadata are included | Four holdout scripts in `scripts/controls/` |
| LINCS | GSE70138 Level 3 GCTX plus the matching `inst_info` and `gene_info` compressed tables, dated 2017-03-06 | `run_lincs_scale_test.py --data-dir INPUT_DIRECTORY` |
| Rhaister | Authors' Parse `cell_eval/all_celltypes_delta.parquet`, `pdex/all_celltypes_pdex.parquet`, model environment and `splits/parse/split_0` | Commands below |

Rebuilding the yeast orthologue axis also requires the DIAMOND command-line executable
and the species protein FASTA files named by `build_yeast_ortholog_axis.py`, supplied through
its `--pep-dir` argument. These are not installed by the Python requirements files.

The five-species hepatocyte, yeast and TG-GATEs expression matrices and their full frozen
preprocessing inputs are not present in this snapshot. Their retained metadata/provenance
describe the original transformations, and their derived results remain available for
inspection. Re-fetching live orthologue services can change the axis; do not silently compare
such a rebuilt analysis with the deposited one. The TG-GATEs orthologue TSV files included
here document mappings but do not replace the aligned JSON axis required by the NPZ workflow.

The missing-value audit now caches its temporary arrays in
`artifacts/gse249010_v1/na_cache/`; run `run_na_audit.py` before `run_na_masked_fig1.py`.
These caches and external downloads are ignored by Git.

## External Rhaister runs

Use the [authors' model instructions](https://huggingface.co/tahoebio/Rhaister/blob/main/README.md)
and [Parse dataset](https://huggingface.co/datasets/tahoebio/parse-de-rhaister).
The original external model revision was not pinned in the deposited run metadata; a later
upstream version is not guaranteed to reproduce the historical model metrics exactly.

```bash
python scripts/make_rhaister_cluster_splits.py \
  --data-dir INPUT_PARSE_DIRECTORY --split-dir RHAISTER_SPLITS_PARSE_DIRECTORY
python scripts/run_rhaister_donor_ceiling.py INPUT_DELTA_PARQUET
python scripts/run_rhaister_ceiling_saturation.py INPUT_DELTA_PARQUET
python scripts/run_rhaister_depth_vs_ceiling.py INPUT_PDEX_PARQUET
```

The split generator creates the cluster and random comparison panels. Evaluate each with
the external Rhaister package, using its `python -m rhaister.train ... --split parse/...`
interface. To reconstruct the deposited aggregate from the included metric excerpts:

```bash
python scripts/collect_rhaister_metrics.py artifacts/rhaister_benchmark/logs /tmp/rhaister_metrics.json
```

Two source-data summaries, `core_numbers.json` and `reconstruct_metric_audit.json`, lack an
exact original generating script in the available snapshot. They are retained unchanged;
figure regeneration and the registered checks use them directly. This limitation is separate
from the externally downloadable inputs above.
