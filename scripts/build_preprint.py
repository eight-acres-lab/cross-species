#!/usr/bin/env python3
"""Build the editable LaTeX and both identical PDFs from preprint.md.

Requires Pandoc and either XeLaTeX or Tectonic. The generated tex/ folder remains
self-contained. This build uses the vector figures produced by make_preprint_figures.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper"
TEX = PAPER / "tex"


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def raw(text):
    return {"t": "RawBlock", "c": ["latex", text]}


def plain(inlines):
    parts = []
    for item in inlines:
        if item["t"] == "Str":
            parts.append(item["c"])
        elif item["t"] in ("Space", "SoftBreak", "LineBreak"):
            parts.append(" ")
        elif item["t"] in ("Strong", "Emph"):
            parts.append(plain(item["c"]))
    return "".join(parts)


def latex(blocks, version):
    doc = {"pandoc-api-version": version, "meta": {}, "blocks": blocks}
    return run(["pandoc", "-f", "json", "-t", "latex", "--wrap=none"],
               input=json.dumps(doc), capture_output=True).stdout.strip()


def without_prefix(inlines):
    # Figure/table caption labels are their own Strong inline in the source.
    assert inlines[0]["t"] == "Strong"
    return inlines[2:] if inlines[1]["t"] == "Space" else inlines[1:]


def make_table(block, caption, number, version):
    c = block["c"]
    header = c[3][1][0][1]
    rows = [row[1] for body in c[4] for row in body[3]]
    widths = {1: [0.16, 0.23, 0.27, 0.34], 2: [0.53, 0.16, 0.23, 0.08],
              3: [0.15, 0.16, 0.23, 0.23, 0.23], 4: [0.045, 0.32, 0.265, 0.37]}[number]
    n = len(widths)
    columns = []
    for i, w in enumerate(widths):
        align = "centering" if number < 4 and i > 0 else "raggedright"
        columns.append(">{\\" + align + r"\arraybackslash}p{(\linewidth - "
                       + str(2*(n-1)) + r"\tabcolsep) * \real{" + str(w) + "}}")
    spec = "@{}" + "\n".join(columns) + "@{}"

    def rowtex(cells, heading=False):
        result = []
        for i, cell in enumerate(cells):
            txt = latex(cell[4], version)
            if number == 3 and i > 1 and not heading:
                txt = txt.replace(" (p", r"\newline (p")
            result.append(r"\textbf{" + txt + "}" if heading else txt)
        return " & ".join(result) + r" \\"

    body = "\n\\addlinespace[3pt]\n".join(rowtex(row) for row in rows)
    return raw(r"\begin{table}[!htbp]" + "\n\\centering\n\\small\n"
               + r"\caption{" + caption + r"}\label{tab:" + str(number) + "}\n"
               + r"\begin{tabular}{" + spec + "}\n\\toprule\n"
               + rowtex(header, True) + "\n\\midrule\n" + body
               + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def prepare(doc):
    blocks, version = doc["blocks"], doc["pandoc-api-version"]
    title = blocks[0]["c"][2]
    date = blocks[1]["c"][0]["c"]
    doc["meta"] = {"title": {"t": "MetaInlines", "c": title},
                   "date": {"t": "MetaInlines", "c": date}}
    result = []
    section_labels = {}
    figures = tables = 0
    references = False
    abstract = False
    i = 2
    while i < len(blocks):
        b = copy.deepcopy(blocks[i])
        if b["t"] == "HorizontalRule":
            i += 1
            continue
        if b["t"] == "Header":
            heading = plain(b["c"][2])
            if abstract:
                result.append(raw("\\endgroup\n\\clearpage"))
                abstract = False
            if heading == "Abstract":
                result.append(raw("\\pdfbookmark[1]{Abstract}{abstract}\n"
                                  "\\section*{Abstract}\n\\begingroup\n"
                                  "\\fontsize{10.5}{12.8}\\selectfont"))
                abstract = True
            elif heading == "References":
                result.append(raw("\\clearpage\n\\phantomsection\n"
                                  "\\addcontentsline{toc}{section}{References}\n"
                                  "\\begingroup\\small\\setstretch{1.05}\n"
                                  "\\begin{thebibliography}{99}\n\\raggedright\n"
                                  "\\interlinepenalty=10000\n\\setlength{\\itemsep}{4pt}"))
                references = True
            else:
                if references:
                    result.append(raw("\\end{thebibliography}\n\\endgroup\n\\clearpage"))
                    references = False
                m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", heading)
                if m:
                    section_labels[m[1]] = b["c"][1][0]
                    b["c"][2] = b["c"][2][2:]
                else:
                    b["c"][1][1].append("unnumbered")
                    result.append(raw("\\phantomsection"))
                b["c"][0] -= 1
                result.append(b)
            i += 1
            continue
        if references:
            text = latex([b], version)
            match = re.match(r"^\{\[\}(\d+)\{\]\}\s*", text)
            if not match:
                raise ValueError(f"Unrecognised bibliography entry: {text[:80]}")
            result.append(raw(r"\bibitem{ref" + match[1] + "}" + text[match.end():]))
            i += 1
            continue
        if b["t"] == "Para" and b["c"][0]["t"] == "Image":
            figures += 1
            image_target = b["c"][0]["c"][2][0]
            expected_target = f"tex/figures/Fig{figures}.png"
            if image_target != expected_target:
                raise ValueError(f"Figure {figures} uses {image_target!r}; expected "
                                 f"{expected_target!r} and its matching vector PDF. "
                                 "Update the figure source and build mapping together.")
            cap = blocks[i+1]
            assert plain(cap["c"]).startswith(f"Figure {figures}.")
            caption = latex([{"t": "Para", "c": without_prefix(cap["c"])}], version)
            result.append(raw(r"\begin{figure}[!htbp]" + "\n\\centering\n"
                              + r"\includegraphics[width=\linewidth]{figures/Fig" + str(figures)
                              + ".pdf}\n\\caption{" + caption + r"}\label{fig:"
                              + str(figures) + "}\n\\end{figure}"))
            i += 2
            continue
        if b["t"] == "Para" and re.match(r"^Table \d+\.", plain(b["c"])):
            tables += 1
            assert plain(b["c"]).startswith(f"Table {tables}.")
            cap = latex([{"t": "Para", "c": without_prefix(b["c"])}], version)
            assert blocks[i+1]["t"] == "Table"
            result.append(make_table(blocks[i+1], cap, tables, version))
            i += 2
            continue
        if b["t"] == "Table":
            raise ValueError("Every manuscript table must have a numbered caption.")
        result.append(b)
        i += 1
    assert figures == tables == 4, (figures, tables)
    doc["blocks"] = result
    return doc, section_labels


def polish(text, labels):
    # Typeset combining overbars and exponent strings as a single mathematical object.
    text = text.replace("r̄", r"\ensuremath{\bar{r}}").replace("ρ̄", r"\ensuremath{\bar{\rho}}")
    supermap = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = re.sub(r"10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)",
                  lambda m: r"\ensuremath{10^{" + m[1].translate(supermap) + "}}", text)
    # Code paths can break at underscores and slashes; no long monospace margin overruns.
    text = re.sub(r"\\texttt\{([^{}]+)\}",
                  lambda m: r"\protect\path{" + m[1].replace(r"\_", "_") + "}", text)
    for number, label in sorted(labels.items(), key=lambda x: -len(x[0])):
        text = re.sub(r"\bSection\s+" + re.escape(number) + r"(?!\d|\.\d)",
                      lambda _: r"Section~\ref{" + label + "}", text)
    text = re.sub(r"\b(Figure|Table)\s+([1-4])(?!\d)", lambda m: m[1] + r"~\ref{"
                  + ("fig:" if m[1] == "Figure" else "tab:") + m[2] + "}", text)

    bibliography_numbers = {int(n) for n in re.findall(r"\\bibitem\{ref(\d+)\}", text)}

    def citation(match):
        numbers = []
        for part in re.split(r",\s*", match[1]):
            interval = re.split(r"[–-]+", part.strip())
            lo, hi = int(interval[0]), int(interval[-1])
            if not (1 <= lo <= hi):
                raise ValueError(f"Invalid citation range: {part!r}")
            missing = set(range(lo, hi+1)) - bibliography_numbers
            if missing:
                raise ValueError(f"Citation has no bibliography entry: {sorted(missing)}")
            numbers.extend(range(lo, hi+1))
        return r"\cite{" + ",".join(f"ref{n}" for n in numbers) + "}"

    return re.sub(r"\{\[\}(\d+(?:(?:,\s*|--?|–)\d+)*)\{\]\}", citation, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-only", action="store_true")
    args = parser.parse_args()
    doc = json.loads(run(["pandoc", str(PAPER / "preprint.md"), "-f", "markdown", "-t", "json"],
                         capture_output=True).stdout)
    doc, labels = prepare(doc)
    tex = run(["pandoc", "-f", "json", "-t", "latex", "--standalone", "--number-sections",
               "--metadata-file", str(PAPER / "preprint_pdf.yaml"),
               "--template", str(PAPER / "preprint_template.tex"), "--wrap=auto"],
              input=json.dumps(doc), capture_output=True).stdout
    tex = "% Generated from ../preprint.md by scripts/build_preprint.py.\n" + polish(tex, labels)
    (TEX / "preprint.tex").write_text(tex)
    if args.tex_only:
        return
    if shutil.which("xelatex"):
        for _ in range(2):
            run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "preprint.tex"], cwd=TEX)
    elif shutil.which("tectonic"):
        run(["tectonic", "--keep-logs", "preprint.tex"], cwd=TEX)
    else:
        raise SystemExit("Install XeLaTeX or Tectonic; editable preprint.tex has been generated.")
    log = (TEX / "preprint.log").read_text(errors="replace")
    errors = [line for line in log.splitlines() if any(s in line for s in
              ("Missing character:", "Overfull ", "undefined references", "undefined citations"))]
    if errors:
        raise SystemExit("Fix layout/build warnings before publishing:\n" + "\n".join(errors))
    shutil.copyfile(TEX / "preprint.pdf", PAPER / "preprint.pdf")
    print("Updated editable LaTeX and both identical preprint PDFs.")


if __name__ == "__main__":
    main()
