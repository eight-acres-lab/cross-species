# How animal data can improve models of human perturbation responses

This directory contains the code and processed analysis data for the manuscript by **Zihui Zhang, Wei Wang, Ziyan Zhou, Zhi Wang and Chen Peng**. It is the directory cited in the manuscript's Data Availability and Code Availability statements.

The analyses examine two uses of animal data: providing a measured animal response at prediction time, and adding animal observations during training. Human-only adaptation controls and matched additional-human comparisons help attribute the observed changes in prediction error. The macaque experiment includes frozen evaluation in a second screen of the same study; the rabbit experiment evaluates a larger change in biological and experimental context.

## Start here

Run commands from this directory:

```bash
cd releases/2026-11-10-three-stage
python3 scripts/verify_deposit.py
python3 scripts/verify_three_stage_evidence.py
```

Both checks use the Python standard library. `verify_deposit.py` validates every entry in [DATA_MANIFEST.json](DATA_MANIFEST.json). `verify_three_stage_evidence.py` makes 186 numerical comparisons between deposited figure source data and frozen score records, including per-line, per-seed and per-donor aggregation.

| Directory | Contents |
| --- | --- |
| [scripts/](scripts/) | Data preparation, model fitting, frozen prediction and analysis checks |
| [src/cross_species/](src/cross_species/) | Shared prediction, model-adaptation and evaluation code |
| [docs/](docs/) | Frozen experiment configurations, source metadata, gene axes and figure source data |
| [outputs/](outputs/) | Analysis reports, scores, selections and fitted-run metadata |
| [artifacts/](artifacts/) | Supporting annotations, source mappings and compound-feature provenance |
| [reproduction/](reproduction/) | Source-data acquisition records and prepared-input checksums |

## Find the results behind the figures

Figure numbers in this table follow the submitted manuscript. Deposited filenames retain their original identifiers so that existing references and verification scripts continue to resolve.

| Manuscript figure | Data and implementation |
| --- | --- |
| Figure 1: hepatocyte prediction design | [Nested benchmark](scripts/run_feedback_nested_benchmark.py), [correspondence benchmark](scripts/run_crossspecies_pairing_benchmark.py) and [frozen comparison configuration](outputs/feedback_nested_chemical_20261107_absolute_v1/specification.json) |
| Figure 2: predictive information in rat responses | [Figure source data](docs/biorxiv_ready_20261108/figures/results/Fig5_information_source_data.json), [primary results](outputs/feedback_nested_chemical_20261107_absolute_v1/), [response-profile controls](outputs/feedback_joint_specificity_20261107_absolute/) and [reference-budget comparisons](outputs/feedback_label_budget_20261107_absolute/) |
| Figure 3: correspondence and study transfer | [Correspondence source data](docs/biorxiv_ready_20261108/figures/results/Fig6_correspondence_source_data.json), [study-transfer source data](docs/biorxiv_ready_20261108/figures/results/Fig7_transport_source_data.json), [paired-response results](outputs/crossspecies_direct_03/) and [external-transfer results](outputs/external_transfer_04/) |
| Figure 4: human-only adaptation | [Source data](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig5_adaptation.csv) and [State/scDFM result records](outputs/remote_gpu_20261108/) |
| Figure 5: animal-training design | [Cohort counts](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig6_cohorts.csv), [matched cell budgets](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig6_budgets.csv) and [gene axes](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig6_gene_axes.csv) |
| Figure 6: animal-training results | [Macaque line means](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig7_macaque_line_means.csv), [M56 versus additional-human comparisons](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig7_m56_human_paired.csv) and [paired rabbit comparisons](docs/manuscript_rewrite_20261109/figures/three_stage/source_data/Fig7_rabbit_paired.csv) |

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

Additional source accessions, sample selections and checksums are recorded alongside the corresponding results in `docs/`, `artifacts/` and `outputs/`. Source datasets and pretrained models retain their original terms and attribution requirements.

## Running the analyses

The deposited results and source-data checks can be inspected directly after cloning. A full rerun requires the original expression data, prepared arrays and the upstream State or scDFM model files appropriate to each experiment. The acquisition records and prepared-input manifests specify their identities and checksums; large expression matrices and model checkpoints are obtained separately.

The numerical dependency versions are recorded in [requirements-core.txt](requirements-core.txt). The executed model-training environment used Python 3.11.10, PyTorch 2.7.1 and CUDA 11.8. Upstream model repositories supply additional model-specific dependencies. The [package configuration](pyproject.toml) defines the shared `cross_species` Python package.

These entry points describe the inputs for each analysis:

```bash
python scripts/run_feedback_nested_benchmark.py --help
python scripts/run_crossspecies_pairing_benchmark.py --help
python scripts/run_crossstudy_kernel_transfer.py --help
python scripts/run_gse325993_state.py train-predict --help
python scripts/predict_gse325993_screen.py predict --help
python scripts/run_nonrodent_lps_state.py --help
```

The macaque runners enforce the paths and artifact hashes in their frozen configurations, including the original exp2 model records used for exp3 prediction. Replay requires the contract-bound Linux environment and the recorded inputs. Use a separate output directory for new runs so the deposited scores remain available for comparison.

For preparation and scoring, see the `prepare_*`, `audit_*` and `verify_*` scripts in [scripts/](scripts/). The [shared State interface](src/cross_species/state_assay_adapter.py) and [named-gene model adapter](src/cross_species/named_gene_state.py) implement the animal-training readout described in Methods.
