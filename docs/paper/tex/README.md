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

`figures/Fig1` through `Fig4` are 160 mm wide, each with four panels, in vector PDF and
300 dpi PNG. All diagrams and plots are scripted with NumPy/Matplotlib. Empirical panels
read deposited JSON; teaching matrices are separately constructed and labelled illustrative.
See the root data catalogue for their exact sources.
