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

`figures/Fig1` through `Fig5` are 160 mm wide, in vector PDF, editable SVG and 600 dpi RGB PNG.
Figure 1 is a standalone workflow; Figure 2 summarises dataset scope and coverage;
Figures 3–5 each contain four result/method panels. `Fig2_source_data.json` in the
figure directory contains metadata counts, counting units and input hashes.
All diagrams and plots are scripted with NumPy/Matplotlib. Empirical panels read deposited
JSON; teaching matrices and workflow miniatures are identified as illustrative in the legends. Arial is used
when installed, with Liberation Sans and DejaVu Sans fallbacks. See the root data catalogue
for exact sources.

Figure text is pure black: primary labels are 8 pt, compact labels and ticks are 7 pt,
and panel letters are 8 pt bold upright lowercase. Letter columns and row spacing are
consistent; panel descriptions and interpretation are in the manuscript legends.
The [NC author instructions](https://mts-ncomms.nature.com/cgi-bin/main.plex?form_type=display_auth_instructions)
recommend 8 pt at final size and editable vector figures. Use the PDFs for submission;
SVG copies are editing aids and are not an accepted NC upload format.
