# How animal data can improve models of human perturbation responses

Code and processed analysis data accompanying the manuscript by **Zihui Zhang, Wei Wang, Ziyan Zhou, Zhi Wang and Chen Peng**.

The study tests animal measurements as prediction inputs and as training supervision for models of human cellular responses. Matched comparisons measure prediction gains, performance relative to additional human observations, and transfer across experimental settings.

## Paper data and code

**[Open the paper's data and code directory](releases/2026-11-10-three-stage/).**

Data and code URL cited in the manuscript:

<https://github.com/eight-acres-lab/cross-species/tree/main/releases/2026-11-10-three-stage>

The [release guide](releases/2026-11-10-three-stage/README.md) maps scripts, configurations, results and source data to the manuscript figures and supplementary tables.

**[View Figures 1–6](releases/2026-11-10-three-stage/figures/README.md)** · **[Download all six figures as PDF](releases/2026-11-10-three-stage/figures/generated/paper_figures.pdf)**

| Analysis | Contents |
| --- | --- |
| Measured animal responses as prediction inputs | Rat-to-human hepatocyte prediction, correspondence controls and transfer to another study |
| Human-only adaptation controls | State and scDFM comparisons of supervision, target correspondence and update methods |
| Animal data as training supervision | Macaque-to-human State adaptation, frozen evaluation in a second screen, and rabbit-to-human inflammatory-response prediction |
| Supplementary source comparisons | Blood, fibroblast, hepatocyte and RNA-binding-protein response panels |

## Verify the deposited files

```bash
git clone https://github.com/eight-acres-lab/cross-species.git
cd cross-species/releases/2026-11-10-three-stage
python3 scripts/verify_deposit.py
python3 scripts/verify_three_stage_evidence.py
```

These commands use the Python standard library to verify file hashes and 186 numerical comparisons between figure source data and score records.

To recompute the paper's summaries and bootstrap intervals and generate its figures, install the [figure dependencies](releases/2026-11-10-three-stage/requirements-figures.txt), then run from the release directory:

```bash
python scripts/verify_paper_results.py --out figures/generated/paper_results.json
python scripts/reproduce_paper_figures.py --compare
```

The result check covers the seven percentages in the abstract and all 113 numeric cells in Supplementary Tables 1–8. The figure command exports PDF, SVG, PNG and source-data files and compares the rendered PDFs with the manuscript figures.

The repository-level [file inventory](DATA_MANIFEST.json) records file sizes and SHA-256 hashes. The paper directory also has its own [inventory](releases/2026-11-10-three-stage/DATA_MANIFEST.json).

## Correspondence

Zhi Wang: [witwang@pointeight.ai](mailto:witwang@pointeight.ai)

Chen Peng: [pengchenea@cau.edu.cn](mailto:pengchenea@cau.edu.cn)
