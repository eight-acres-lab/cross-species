# Reliability and limits of cross-species transcriptomic comparisons

Data and analysis code accompanying the manuscript. The current release contains
numerical results, processed inputs, source metadata and reproduction instructions.

**Interpretation correction (2026-11-06):** historical injury-prediction p-values require
recomputation with the corrected permutation procedure. The archived LINCS full-mean/half-sample
ratio is excluded from ceiling inference. See [the correction notes](REPRODUCING.md#interpretation-corrections-2026-11-06)
before interpreting those outputs; numerical checksum and consistency checks do not validate inference.
The [follow-up audit](artifacts/biology/reviewer_followup.json) records concentration-matched
sensitivities, Holm correction for Table 3, and the three donor pairs per GSE330151 group.

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
python scripts/audit_dose_and_multiplicity.py --out /tmp/reviewer_followup.json
python scripts/make_preprint_figures.py --out outputs/figures
```

The deposited inputs support 140 numerical consistency checks and regeneration of
the five manuscript figures. The verifier includes archived values whose inferential use
has been withdrawn. It checks stored expectations against their
source artifacts; it does not parse every number in the prose. Figure outputs are
vector PDF, editable SVG and 600 dpi RGB PNG in the local, Git-ignored
`outputs/figures/` directory. Rendered figures and manuscript documents are not
part of the current release.

## Contents

| Directory | Contents |
|---|---|
| `artifacts/biology/concordance/` | Species comparisons, saturation, reliability, specificity, and controls |
| `artifacts/biology/dili/` | Time/species/system comparisons and nine endpoint prediction runs |
| `artifacts/gse249010_v1/` | Processed blood matrix, gene axis, metadata, and missing-value audits |
| `artifacts/faang_bmdm/processed/` | Processed macrophage matrix, gene axis, metadata, and provenance |
| `artifacts/lincs_scale/`, `artifacts/rhaister_benchmark/` | External benchmark results |
| `scripts/`, `src/`, `tests/` | Analysis, plotting, validation, and regression tests |

The release includes the corrected handling of tied taxonomic ranks. Figure 4 and
Tables 1/3 use the same current statistical sources. Teaching matrices are labelled
as illustrative and are separate from empirical results.

Large raw downloads, external model weights, and some intermediate matrices are obtained
from their original providers; the exact limitations are listed in the reproduction guide.
Primary data remain subject to their original providers' terms. Cite the source studies
linked in `DATA_AVAILABILITY.md` when using their data.
