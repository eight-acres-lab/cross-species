# How animal data can improve models of human perturbation responses

Analysis code, processed results and figure source data accompanying the manuscript.

The analyses evaluate measured animal responses as prediction inputs and animal observations as training supervision. Human-only adaptation controls and matched additional-human comparisons measure each source's contribution to prediction. The macaque experiment evaluates trained models in a second screen of the same study; the rabbit experiment tests transfer across cell sources and stimulation protocols.

## Start here

From the repository root:

```bash
cd releases/2026-11-10-three-stage
python3 scripts/verify_deposit.py
python3 scripts/verify_three_stage_evidence.py
```

Both checks use the Python standard library. `verify_deposit.py` validates every entry in [DATA_MANIFEST.json](DATA_MANIFEST.json). `verify_three_stage_evidence.py` verifies 186 numerical comparisons, including aggregation by cell line, seed and donor.

## Recompute results and generate figures

Use Python 3.12 with the Arial font and the system Cairo library:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/verify_paper_results.py --out figures/generated/paper_results.json
python scripts/reproduce_paper_figures.py --compare
```

On macOS with Cairo installed through Homebrew, set `DYLD_FALLBACK_LIBRARY_PATH` to its library directory if CairoSVG cannot locate the library.

`verify_paper_results.py` recalculates the deposited line–seed, donor and compound summaries, including 10,000 donor-bootstrap samples and 5,000 compound-bootstrap samples. It checks seven abstract percentages and 113 numeric cells across Supplementary Tables 1–8 against [paper_values.json](figures/paper_values.json).

`reproduce_paper_figures.py` draws the quantitative panels from the result records and renders the two study schematics from editable artwork and layout code. It exports PDF, SVG, PNG and source-data files to `figures/generated/`. The `--compare` option renders each generated PDF and its manuscript counterpart at 144 dpi and compares every pixel.

**[Figure gallery](figures/README.md)** · **[Six-figure PDF](figures/generated/paper_figures.pdf)** · **[Result verification](figures/generated/paper_results.json)** · **[Figure verification](figures/generated/figure_verification.json)**

| Directory | Contents |
| --- | --- |
| [scripts/](scripts/) | Data preparation, model fitting, frozen prediction and analysis checks |
| [src/cross_species/](src/cross_species/) | Shared prediction, model-adaptation and evaluation code |
| [docs/](docs/) | Frozen experiment configurations, source metadata, gene axes and figure source data |
| [outputs/](outputs/) | Analysis reports, scores, selections and fitted-run metadata |
| [artifacts/](artifacts/) | Supporting annotations, source mappings and compound-feature provenance |
| [reproduction/](reproduction/) | Source-data acquisition records and prepared-input checksums |
| [figures/](figures/) | Figures 1–6, editable artwork, captions, source data and verification reports |

## Find the results behind the figures

The table maps Figures 1–6 to their source data and analysis code.

| Manuscript figure | Data and implementation |
| --- | --- |
| Figure 1: hepatocyte prediction design | [Nested benchmark](scripts/run_feedback_nested_benchmark.py), [correspondence benchmark](scripts/run_crossspecies_pairing_benchmark.py) and [frozen comparison configuration](outputs/feedback_nested_chemical_20261107_absolute_v1/specification.json) |
| Figure 2: predictive information in rat responses | [Figure source data](figures/generated/source_data/Fig2_information.json), [primary results](outputs/feedback_nested_chemical_20261107_absolute_v1/), [response-profile controls](outputs/feedback_joint_specificity_20261107_absolute/) and [reference-budget comparisons](outputs/feedback_label_budget_20261107_absolute/) |
| Figure 3: correspondence and study transfer | [Figure source data](figures/generated/source_data/Fig3_correspondence_transport.json), [paired-response results](outputs/crossspecies_direct_03/) and [external-transfer results](outputs/external_transfer_04/) |
| Figure 4: human-only adaptation | [Figure source data](figures/generated/source_data/Fig4_adaptation.csv) and [State/scDFM result records](outputs/remote_gpu_20261108/) |
| Figure 5: animal-training design | [Cohort counts](figures/generated/source_data/Fig5_cohorts.csv), [matched cell budgets](figures/generated/source_data/Fig5_budgets.csv) and [gene axes](figures/generated/source_data/Fig5_gene_axes.csv) |
| Figure 6: animal-training results | [Macaque line means](figures/generated/source_data/Fig6_macaque_line_means.csv), [M56 versus additional-human comparisons](figures/generated/source_data/Fig6_m56_human_paired.csv) and [paired rabbit comparisons](figures/generated/source_data/Fig6_rabbit_paired.csv) |

## Supplementary Information

| Supplementary result | Result records |
| --- | --- |
| Table 1: external hepatocyte prediction | [External-transfer results](outputs/external_transfer_04/) and [source-swap analysis](docs/multisource_20261108/external_transfer_04/posthoc_scalar_correspondence/) |
| Table 2: State supervision and update controls | [Supervised continuation](outputs/remote_gpu_20261108/state_native_formal_64/), [mismatched-target control](outputs/remote_gpu_20261108/state_mispaired_posthoc/) and [matched update comparison](outputs/remote_gpu_20261108/state_gaussian_policy_formal_64/) |
| Table 3: scDFM update methods | [Population-update results](outputs/remote_gpu_20261108/scdfm_nft_formal/) |
| Table 4: scDFM witness sensitivity | [Witness and matched-supervision results](outputs/remote_gpu_20261108/scdfm_witness_formal/) |
| Tables 5–7: macaque screen results and calibration | [Initial-screen scores](outputs/gse325993_state_20261110/state_primary_v1/audit/) and [frozen second-screen scores](outputs/gse325993_screen_validation_20261110/validation_v1/audit/) |
| Table 8: rabbit augmentation | [Donor-level scores and summaries](outputs/nonrodent_singlecell_20261110/rabbit_primary_v1/audit/) |
| Discussion 1: source combinations | [Blood-panel analysis](docs/biorxiv_ready_20261108/legacy_blood_audit/), [fibroblast comparisons](outputs/multisource_20261108/benchmarks/GSE105160_v1/), [mapping sensitivity](outputs/multisource_20261108/benchmarks/GSE105160_mapping_sensitivity/), [hepatocyte source combinations](outputs/multisource_20261108/benchmarks/GSE330151_v1/) and [RNA-binding-protein compartment comparisons](docs/multisource_20261108/compartment_benchmark_02/) |
| Discussion 2: residual-adapter sensitivity | [Residual-adapter results](outputs/feedback_residual_adapter_20261107_absolute/) |

## Data sources and frozen configurations

| Experiment | Source data | Configuration and preparation records |
| --- | --- | --- |
| Rat–human hepatocytes | [Open TG-GATEs processed resource](https://doi.org/10.5281/zenodo.20204514) | [Acquisition catalog](reproduction/data_acquisition.json), [compound-feature provenance](artifacts/feedback_compound_features_20261107/manifest.json), [nested comparison specification](outputs/feedback_nested_chemical_20261107_absolute_v1/specification.json) |
| External hepatocytes | [GSE330151](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330151) | [Task definition](docs/multisource_20261108/external_transfer_04/hepatocyte_audit/gse330151_task_manifest.json) and [gene bridge](docs/multisource_20261108/external_transfer_04/gene_bridge/) |
| Macaque–human POU5F1 response | [GSE325993](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325993) | [Exp2 configuration](docs/gse325993_state_20261110/CONTRACT_v1.json), [exp2 input records](docs/gse325993_state_20261110/input/), [exp3 configuration](docs/gse325993_screen_validation_20261110/CONTRACT_v1.json) and [exp3 input records](docs/gse325993_screen_validation_20261110/input/) |
| Rabbit–human LPS response | [Hagai et al.](https://doi.org/10.1038/s41586-018-0657-2), [E-MTAB-6754](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6754) and [Lawlor et al.](https://doi.org/10.3389/fimmu.2021.636720) | [Frozen configuration](docs/nonrodent_singlecell_20261110/EVALUATION_CONTRACT_v3.json), [source metadata](docs/nonrodent_singlecell_20261110/) and [prepared-input definitions](docs/nonrodent_singlecell_20261110/prepared/) |

Source accessions, sample selections and checksums accompany the results in `docs/`, `artifacts/` and `outputs/`. Dataset and model reuse follows the source providers' terms.

## License

Original software and software documentation are available under the [MIT License](LICENSE). Third-party components, datasets, model weights and artwork retain their respective terms; see the [third-party notices](THIRD_PARTY_NOTICES.md).

## Running the analyses

To rerun an analysis, obtain its expression data, prepared arrays and State or scDFM model files using the acquisition records and input manifests. These records specify source locations and checksums. Expression matrices and model checkpoints are downloaded separately.

Numerical dependencies are listed in [requirements-core.txt](requirements-core.txt). The training environment uses Python 3.11.10, PyTorch 2.7.1 and CUDA 11.8, plus the dependencies of the upstream State or scDFM model. The [package configuration](pyproject.toml) defines the shared `cross_species` Python package.

Run these commands from this release directory to list each analysis's inputs:

```bash
python scripts/run_feedback_nested_benchmark.py --help
python scripts/run_crossspecies_pairing_benchmark.py --help
python scripts/run_crossstudy_kernel_transfer.py --help
python scripts/run_gse325993_state.py train-predict --help
python scripts/predict_gse325993_screen.py predict --help
python scripts/run_nonrodent_lps_state.py --help
```

The macaque runners require Linux and verify input paths and hashes against their configurations. Exp3 prediction loads the exp2 model records and selected weights. Write new runs to a separate output directory for comparison with the deposited scores.

For preparation and scoring, see the `prepare_*`, `audit_*` and `verify_*` scripts in [scripts/](scripts/). The [shared State interface](src/cross_species/state_assay_adapter.py) and [named-gene model adapter](src/cross_species/named_gene_state.py) implement the animal-training readout described in Methods.
