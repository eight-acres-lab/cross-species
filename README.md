# Reliability and limits of cross-species transcriptomic comparisons

Data and code accompanying the manuscript. This repository contains the publication
snapshot, without the development repository's commit history or unrelated experiments.

- [Read the manuscript](docs/paper/preprint.pdf) or its [Markdown source](docs/paper/preprint.md).
- [Data catalogue](DATA_AVAILABILITY.md): source studies, deposited results, and figure/table inputs.
- [Reproduction guide](REPRODUCING.md): verification, plotting, and analysis prerequisites.
- [SHA-256 inventory](DATA_MANIFEST.json): every file in this release.

## Quick start

The pinned dependencies require Python 3.12 or later. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/verify_preprint_numbers.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/make_preprint_figures.py
```

The deposited inputs support all 101 registered numerical checks and regeneration of
the four manuscript figures. The verifier checks the registered claims against their
source artifacts; it does not parse every number in the prose. Figure outputs are
vector PDF, editable SVG and 600 dpi RGB PNG in `docs/paper/tex/figures/`.

## Contents

| Directory | Contents |
|---|---|
| `artifacts/biology/concordance/` | Species comparisons, saturation, reliability, specificity, and controls |
| `artifacts/biology/dili/` | Time/species/system comparisons and nine endpoint prediction runs |
| `artifacts/gse249010_v1/` | Processed blood matrix, gene axis, metadata, and missing-value audits |
| `artifacts/faang_bmdm/processed/` | Processed macrophage matrix, gene axis, metadata, and provenance |
| `artifacts/lincs_scale/`, `artifacts/rhaister_benchmark/` | External benchmark results |
| `scripts/`, `src/`, `tests/` | Analysis, plotting, validation, and regression tests |
| `docs/paper/` | Manuscript, editable LaTeX, and figure files |

The release includes the corrected handling of tied taxonomic ranks. Figure 3 and
Tables 1/3 use the same current statistical sources. Teaching matrices are labelled
as illustrative and are separate from empirical results.

Large raw downloads, external model weights, and some intermediate matrices are obtained
from their original providers; the exact limitations are listed in the reproduction guide.
Primary data remain subject to their original providers' terms. Cite the source studies
listed in the manuscript when using their data.
