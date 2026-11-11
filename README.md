# How animal data can improve models of human perturbation responses

Code and processed analysis data accompanying the manuscript by **Zihui Zhang, Wei Wang, Ziyan Zhou, Zhi Wang and Chen Peng**.

The study evaluates animal measurements as prediction inputs and as training supervision for models of human cellular responses. Matched comparisons test the contribution of the animal source, its benefit relative to additional human observations, and its performance across experimental settings.

## Paper data and code

**[Open the paper's data and code directory](releases/2026-11-10-three-stage/).**

The manuscript's Data Availability and Code Availability statements point to:

<https://github.com/eight-acres-lab/cross-species/tree/main/releases/2026-11-10-three-stage>

That directory contains the analysis scripts, frozen experiment configurations, processed results and figure source data for this paper. Its [README](releases/2026-11-10-three-stage/README.md) maps the files to the manuscript figures and Supplementary Information and describes the inputs needed to rerun the analyses.

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

These checks use the Python standard library. They verify the deposited file hashes and compare the animal-training and human-only adaptation source data with the frozen score records. The numerical check covers 186 comparisons.

The repository-level [file inventory](DATA_MANIFEST.json) records file sizes and SHA-256 hashes. The paper directory also has its own [inventory](releases/2026-11-10-three-stage/DATA_MANIFEST.json).

## Correspondence

Zhi Wang: [witwang@pointeight.ai](mailto:witwang@pointeight.ai)

Chen Peng: [pengchenea@cau.edu.cn](mailto:pengchenea@cau.edu.cn)
