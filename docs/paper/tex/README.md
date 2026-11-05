# Editable manuscript and vector figures

The canonical text is `../preprint.md`. From the repository root:

```bash
python scripts/make_preprint_figures.py
python scripts/build_preprint.py
```

The builder requires Pandoc and either XeLaTeX or Tectonic. It writes `preprint.tex` and
synchronises this directory's PDF with `../preprint.pdf`.

This directory is also self-contained for direct LaTeX editing:

```bash
xelatex preprint.tex
xelatex preprint.tex
```

Use XeLaTeX, not pdfLaTeX. Check for missing-glyph and overfull-box warnings. Direct edits
to the generated TeX are overwritten by the Markdown build, so move lasting text changes
back to `../preprint.md`.

`figures/Fig1` through `Fig4` are 160 mm wide, in vector PDF, editable SVG and 300 dpi PNG.
Figure 1 is a standalone workflow; Figures 2–4 each contain four result/method panels.
All diagrams and plots are scripted with NumPy/Matplotlib. Empirical panels read deposited
JSON; teaching matrices and workflow miniatures are explicitly illustrative. Arial is used
when installed, with Liberation Sans and DejaVu Sans fallbacks. See the root data catalogue
for exact sources.
