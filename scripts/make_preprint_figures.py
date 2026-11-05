#!/usr/bin/env python3
"""Draw a workflow, a dataset overview and three result composites at 160 mm width.

Empirical marks read deposited JSON estimates, without rerunning or changing analyses.
Teaching matrices and curves are explicitly illustrative and are never substituted for
measurements. PDF/SVG elements, including matrix cells and type, remain editable vectors.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image

from collect_preprint_datasets import collect_dataset_overview
from plot_rho_phylogeny import collect
from run_rho_reliability_and_phylogeny import centered_ranks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/biology/concordance'
MM = 1 / 25.4
WIDTH = 160
BLUE, ORANGE, PURPLE = '#2585BD', '#E78A40', '#8561AD'
TEAL, GREY, INK = '#239E92', '#87929E', '#000000'
LIGHT, LINE = '#F4F7FA', '#D7DFE6'
MIN_TEXT, BODY_TEXT, PANEL_TEXT = 7, 8, 8
PANEL_LEFT, PANEL_RIGHT, ROW_GAP = 3, 83, 12
RESPONSE = LinearSegmentedColormap.from_list('response', [BLUE, '#FFFFFF', '#ED8866'])
NAMES = {
    'Pan troglodytes': 'Chimpanzee', 'Macaca': 'Macaque',
    'Papio anubis': 'Baboon', 'Bos taurus': 'Cow', 'Ovis aries': 'Sheep',
    'Sus scrofa': 'Pig', 'Mus musculus': 'Mouse', 'Rattus norvegicus': 'Rat',
    'Oryctolagus cuniculus': 'Rabbit', 'Homo sapiens': 'Human',
}


def load(name):
    return json.loads((DATA / f'{name}.json').read_text())


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': BODY_TEXT, 'text.color': INK, 'axes.labelcolor': INK,
        'axes.labelsize': BODY_TEXT, 'axes.titlesize': BODY_TEXT,
        'xtick.labelsize': MIN_TEXT, 'ytick.labelsize': MIN_TEXT,
        'xtick.color': INK, 'ytick.color': INK,
        'legend.fontsize': MIN_TEXT, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'cross-species-figures',
        'savefig.facecolor': 'white', 'axes.axisbelow': True,
        'mathtext.default': 'regular',
    })


def canvas(height):
    """A figure and diagram canvas using physical millimetres from the top left."""
    fig = plt.figure(figsize=(WIDTH * MM, height * MM))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set(xlim=(0, WIDTH), ylim=(height, 0))
    bg.axis('off')
    return fig, bg


def axes(fig, x, top, w, h):
    fw, fh = fig.get_size_inches() / MM
    return fig.add_axes([x/fw, 1-(top+h)/fh, w/fw, h/fh])


def label(ax, x, y, text, size=BODY_TEXT, *, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, text, fontsize=max(size, MIN_TEXT), weight=weight, ha=ha, va=va,
                   color=INK, linespacing=1.25, **kw)


def panel(bg, x, y, letter):
    """Bare, upright lowercase panel letters; descriptions belong in the caption."""
    # A shared baseline also aligns letters with different ascenders (a/c versus b/d).
    label(bg, x, y+.8, letter, PANEL_TEXT, weight='bold', va='baseline')


def stage(bg, x, y, number, title, color):
    bg.add_patch(Circle((x+2, y), 2.3, fc='white', ec=color, lw=1.2))
    label(bg, x+2, y, str(number), BODY_TEXT, weight='bold', ha='center')
    label(bg, x+6, y, title, BODY_TEXT, weight='bold')


def operator(bg, x, y, symbol):
    """Draw simple arithmetic signs as strokes, without oversized text glyphs."""
    offsets = [0] if symbol == '-' else [-.5, .5]
    for dy in offsets:
        bg.plot([x-1.3, x+1.3], [y+dy, y+dy], color=INK, lw=.8)


def region(bg, x, y, w, h, color='white', edge=LINE, lw=0.65):
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle='round,pad=0,rounding_size=1.3',
                      fc=color, ec=edge, lw=lw)
    bg.add_patch(p)
    return p


def arrow(bg, start, end, *, color=INK, lw=0.85, style='-', connection='arc3'):
    p = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=7,
                        lw=lw, color=color, linestyle=style,
                        connectionstyle=connection, shrinkA=0, shrinkB=0)
    bg.add_patch(p)
    return p


def cells(bg, x, y, values, cell=2.5, *, vmin=-1, vmax=1, cmap=RESPONSE,
          edge='white', linewidth=0.45):
    """Draw a matrix directly in the diagram, with no rasterized image objects."""
    values = np.asarray(values)
    norm = Normalize(vmin, vmax)
    for i, j in np.ndindex(values.shape):
        if np.isfinite(values[i, j]):
            bg.add_patch(Rectangle((x+j*cell, y+i*cell), cell, cell,
                                  facecolor=cmap(norm(values[i, j])),
                                  edgecolor=edge, linewidth=linewidth))


def colour_key(fig, x, top, w, *, vmin, vmax, title, ticks=None):
    ax = axes(fig, x, top, w, 1.7)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=RESPONSE),
                       cax=ax, orientation='horizontal', ticks=ticks or [vmin, 0, vmax])
    cbar.solids.set_rasterized(False)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=MIN_TEXT, length=1.5, pad=1)
    cbar.ax.set_title(title, fontsize=MIN_TEXT, pad=3)


def toy_responses():
    # Four imaginary species and six genes; all derived schematics use these values.
    return np.array([[1.4, 0.8, -1.1, -0.6, 0.2, -0.5],
                     [0.9, 1.1, -0.7, -1.2, 0.4, -0.2],
                     [0.4, -0.8, -0.3, 1.2, 0.7, -0.4],
                     [1.0, 0.2, -0.9, 0.3, -0.8, 0.4]])


def toy_contexts():
    x = toy_responses()
    z = x + np.array([[0.1, 0.4, -0.1, 0.2, -0.1, 0.3],
                      [0.0, -0.2, 0.5, -0.1, 0.3, -0.4],
                      [-0.2, 0.1, 0.6, -0.3, 0.4, -0.1],
                      [0.3, -0.2, 0.1, 0.4, -0.1, -0.3]])
    return np.corrcoef(x), np.corrcoef(z)


def decomposition():
    x = np.array([[1.7, 1.2, 0.8, 1.5], [-1.2, -0.4, -0.8, -1.6],
                  [0.8, -0.4, 0.5, -0.1], [-0.3, 0.9, -0.9, 0.3],
                  [0.2, 0.7, 0.9, 0.2]])
    shared = np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1)
    return x, shared, x-shared


def save(fig, out, n):
    # Fixed page boxes preserve the documented print size; never use bbox_inches='tight'.
    fig.savefig(out/f'Fig{n}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    svg = out/f'Fig{n}.svg'
    fig.savefig(svg, metadata={'Date': None})
    # Matplotlib leaves trailing spaces in multiline SVG path data. Normalize the
    # serializer output so the editable files also pass the repository whitespace check.
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    preview = BytesIO()
    fig.savefig(preview, format='png', dpi=600)
    preview.seek(0)
    with Image.open(preview) as raster:
        raster.convert('RGB').save(out/f'Fig{n}.png', dpi=(600, 600))
    plt.close(fig)
    print(f'Fig{n}: editable PDF/SVG and 600 dpi PNG')


def figure1(out):
    """A single connected workflow, with no measured-result panels."""
    fig, bg = canvas(174)
    stage(bg, 3, 5, 1, 'Construct response profiles', BLUE)
    # Input contexts have their own roles; they are not merged into one biological dataset.
    region(bg, 3, 11, 41, 34, 'white', '#B6D9ED')
    label(bg, 7, 16, 'Experimental contexts', 8, weight='bold')
    for y, color, title, desc in [
        (23, BLUE, 'Blood panel', 'Species × dose × time'),
        (35, ORANGE, 'TG-GATEs', 'Human / rat; culture / liver'),
    ]:
        bg.add_patch(Circle((8, y), 1.25, fc=color, ec='none'))
        label(bg, 11, y, title, 7, weight='bold')
        label(bg, 7, y+5, desc, 6.3)
    arrow(bg, (45.5, 28), (51.5, 28))
    # Treated and control matrices use the same gene axis before differencing.
    perturb = np.array([[2.1, 1.8, .4, .8, 1.2, 1.0],
                        [2.3, 1.4, .6, 1.1, 1.8, .6],
                        [1.8, 2.0, .2, 1.0, 1.4, .9]])
    control = np.array([[1., 1.1, 1.3, 1.4, 1.0, 1.2],
                        [1.2, 1.0, 1.1, 1.7, 1.3, 1.0],
                        [.9, 1.2, 1.4, 1.6, 1.1, 1.3]])
    seq = LinearSegmentedColormap.from_list('expression', ['#F0F6FB', BLUE])
    cells(bg, 54, 24, perturb, 2.1, vmin=0, vmax=2.5, cmap=seq)
    cells(bg, 74, 24, control, 2.1, vmin=0, vmax=2.5, cmap=seq)
    label(bg, 60.3, 19.5, 'Perturbed', 6.7, ha='center')
    label(bg, 80.3, 19.5, 'Control', 6.7, ha='center')
    operator(bg, 70.3, 27, '-')
    label(bg, 70.5, 36, 'Gene alignment', 7, ha='center')
    arrow(bg, (89, 28), (96, 28))
    # Actual analytical matrices are continuous responses; these values only teach the layout.
    cells(bg, 100, 22, perturb-control, 3, vmin=-1.5, vmax=1.5)
    label(bg, 109, 16, 'Response matrix', 7, weight='bold', ha='center')
    label(bg, 109, 35, 'Genes', 6.5, ha='center')
    label(bg, 122, 26, 'Subject ×\ncondition', 7)
    colour_key(fig, 131, 38.5, 22, vmin=-1.5, vmax=1.5, title='Response (a.u.)')
    # A fork: species redundancy and profile-level ceilings are distinct constructions.
    bg.plot([109, 109, 40], [43.5, 49, 49], color=BLUE, lw=.8)
    bg.plot([109, 121], [49, 49], color=ORANGE, lw=.8)
    arrow(bg, (40, 49), (40, 55), color=BLUE)
    arrow(bg, (121, 49), (121, 55), color=ORANGE)
    label(bg, 43, 52, 'Blood panel', 6.2)
    label(bg, 124, 52, 'TG-GATEs', 6.2)
    region(bg, 3, 56, 75, 59, 'white', '#B6D9ED')
    region(bg, 82, 56, 75, 59, 'white', '#EFCDA9')
    stage(bg, 6, 61, 2, 'Quantify species redundancy', BLUE)
    stage(bg, 85, 61, 3, 'Estimate replicate ceiling', ORANGE)
    label(bg, 40.5, 67, 'Average subjects within each condition', 7, ha='center')
    x = toy_responses()
    cells(bg, 12, 74, x, 2.5, vmin=-1.5, vmax=1.5)
    for i, name in enumerate('ABCD'):
        label(bg, 10, 75.25+2.5*i, name, 6, ha='right')
    label(bg, 19.5, 87, 'Aligned genes', 6.2, ha='center')
    label(bg, 19.5, 70.5, 'Species means', 6.3, ha='center')
    arrow(bg, (30, 79), (44, 79), color=BLUE)
    label(bg, 37, 73, 'Pearson', 6, ha='center')
    label(bg, 37, 85, 'across genes', 6, ha='center')
    cells(bg, 49, 74, np.corrcoef(x), 2.5)
    label(bg, 54, 70.5, 'Redundancy', 7, ha='center')
    label(bg, 54, 87, 'Species pairs', 6.2, ha='center')
    bg.plot([7, 74], [92, 92], color=LINE, lw=.65)
    # Two compact reliability objects, not a second empirical heatmap.
    p, q = toy_contexts()
    for mat, xx in [(p, 8), (q, 18)]:
        a = mat.copy(); a[np.tril_indices(4)] = np.nan
        cells(bg, xx, 97, a, 1.6)
    arrow(bg, (27, 101), (32, 101), color=BLUE)
    label(bg, 35, 99, 'Subject splits', 7)
    label(bg, 35, 105, 'Pair-rank agreement', 7)
    # The ceiling uses replicate response correlations within sides, not rho split halves.
    label(bg, 119.5, 67, 'Replicate response profiles', 7, ha='center')
    strips = [(90, 76, 'A1', x[0]), (90, 87, 'A2', x[1]),
              (133, 76, 'B1', x[2]), (133, 87, 'B2', x[3])]
    for xx, yy, name, values in strips:
        cells(bg, xx, yy, values[None, :], 2, vmin=-1.5, vmax=1.5)
        label(bg, xx+6, yy-3 if yy == 76 else yy+5.5, name, 6.2, ha='center')
    arrow(bg, (96, 79), (96, 85), color=BLUE)
    arrow(bg, (139, 79), (139, 85), color=ORANGE)
    label(bg, 89, 83, 'rel A', 6.5, ha='right')
    label(bg, 146, 83, 'rel B', 6.5)
    arrow(bg, (104, 77), (130, 77), color=GREY)
    arrow(bg, (104, 88), (130, 88), color=GREY)
    arrow(bg, (104, 77), (130, 88), color='#B7C1CA', lw=.55)
    arrow(bg, (104, 88), (130, 77), color='#B7C1CA', lw=.55)
    label(bg, 117, 82.5, 'All cross-side\npairings', 5.8, ha='center',
          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
    label(bg, 119.5, 99, 'Ceiling = √(rel A × rel B)', 8, ha='center')
    label(bg, 119.5, 107, 'Fraction = agreement / ceiling', 7, ha='center')
    # Both constructions constrain interpretation, with different validation requirements.
    arrow(bg, (40, 115.5), (40, 121), color=BLUE)
    arrow(bg, (120, 115.5), (120, 121), color=ORANGE)
    region(bg, 3, 122, 154, 9, 'white', '#D4C9E4')
    label(bg, 80, 126.5,
          'Reliability  |  Control independence  |  Context validation', 8, ha='center')
    stage(bg, 3, 138, 4, 'Compare transfer limits', PURPLE)
    # Four output objects teach the analysis; all miniatures are explicitly schematic.
    for i, (color, title, pointer) in enumerate([
        (BLUE, 'Panel design', 'Donor value  ·  Fig. 3'),
        (PURPLE, 'Context reuse', 'Time / tissue  ·  Fig. 4'),
        (ORANGE, 'Response specificity', 'Shared / residual  ·  Fig. 5'),
        (TEAL, 'Boundary comparison', 'Fraction of ceiling  ·  Table 2'),
    ]):
        xx = 3 + i*39
        bg.plot([xx+1, xx+37], [144, 144], color=color, lw=1.6)
        label(bg, xx+19, 163, title, 6.6, weight='bold', ha='center')
        label(bg, xx+19, 168, pointer, 6.0, ha='center')
    # Equal-weight saturation under an illustrative exchangeable-donor model.
    k = np.arange(1, 18)
    expected = k*.3 / np.sqrt(k+k*(k-1)*.3)
    bg.plot([13, 13, 32], [148, 158, 158], color=GREY, lw=.6)
    bg.plot(13+19*(k-1)/16, 158-10*expected/.6, color=BLUE, lw=1.1)
    # Context reuse: match the same upper triangles, rather than implying a learned mapping.
    for xx, mat in [(49, p), (64, q)]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 149, upper, 1.9)
    arrow(bg, (58, 153), (63, 153), color=PURPLE)
    # Specificity is a subtraction within species, using exactly the illustrated decomposition.
    response, shared, residual = decomposition()
    cells(bg, 85, 150, response[:3], 1.2, vmin=-2, vmax=2)
    operator(bg, 93, 152, '-')
    cells(bg, 96, 150, shared[:3], 1.2, vmin=-2, vmax=2)
    operator(bg, 104, 152, '=')
    cells(bg, 107, 150, residual[:3], 1.2, vmin=-2, vmax=2)
    # A ceiling is an upper reference, with observed agreement expressed relative to it.
    bg.plot([130, 151], [156, 156], color=GREY, lw=.7)
    bg.plot([130, 144], [152, 152], color=TEAL, lw=1.5)
    bg.plot([151, 151], [149, 158], color=INK, lw=.65, ls=(0, (2, 2)))
    label(bg, 151, 147, 'Ceiling', 5.5, ha='center')
    save(fig, out, 1)


def figure2(out):
    """Dataset scope and the actual input coverage of the two core analyses."""
    data = collect_dataset_overview()
    (out/'Fig2_source_data.json').write_text(json.dumps(data, indent=2)+'\n')
    fig, bg = canvas(158)
    panel(bg, PANEL_LEFT, 5, 'a')
    label(bg, 7, 12, 'Dataset', 8)
    label(bg, 56, 12, 'Species (n)', 8, ha='center')
    label(bg, 99, 12, 'Biological system', 8, ha='center')
    label(bg, 141, 12, 'Study role', 8, ha='center')
    role_colours = {'Core analyses': BLUE, 'Replication': TEAL, 'External check': PURPLE}
    a = axes(fig, 39, 17, 34, 56)
    for i, row in enumerate(data['datasets']):
        yy = 21+8*i
        color = role_colours[row['role']]
        label(bg, 7, yy, row['name'], 8)
        label(bg, 99, yy, row['system'], 7, ha='center')
        label(bg, 141, yy, row['role'], 7, ha='center')
        a.barh(i, row['n_species'], height=.45, color=color, linewidth=0)
        a.text(row['n_species']+.35, i, str(row['n_species']), va='center', fontsize=8)
    a.set(xlim=(0, 12), ylim=(6.5, -.5))
    a.set_xticks([0, 5, 10]); a.set_yticks([])
    a.spines['left'].set_visible(False)

    # Both lower panels start on one baseline after the same 12 mm content gutter.
    lower = 79 + ROW_GAP
    panel(bg, PANEL_LEFT, lower, 'b')
    panel(bg, PANEL_RIGHT, lower, 'c')
    blood = data['blood']
    b = axes(fig, 26, 104, 42, 36)
    seq = ListedColormap(['#C9E2EE'])
    values = np.asarray(blood['counts'])
    b.pcolormesh(values, cmap=seq, vmin=0, vmax=50, edgecolors='white', linewidth=1.5)
    b.set(xlim=(0, len(blood['times_h'])), ylim=(len(blood['doses_ng_ml']), 0),
          xlabel='Time (h)', ylabel='LPS dose (ng/ml)')
    b.set_xticks(np.arange(len(blood['times_h']))+.5, [f'{v:g}' for v in blood['times_h']])
    b.set_yticks(np.arange(len(blood['doses_ng_ml']))+.5, [f'{v:,.0f}' for v in blood['doses_ng_ml']])
    b.tick_params(length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    for i, j in np.ndindex(values.shape):
        b.text(j+.5, i+.5, str(values[i,j]), fontsize=8, ha='center', va='center')

    c = axes(fig, 114, 104, 41, 36)
    rows = data['tggates']['comparisons']
    y = np.arange(len(rows))[::-1]
    c.barh(y, [r['n_shared'] for r in rows], color='#DCE4EA', height=.5, label='Shared')
    c.barh(y, [r['n_in_band'] for r in rows], color=ORANGE, height=.5, label='In band')
    for yy, row in zip(y, rows):
        c.text(row['n_shared']+4, yy, str(row['n_shared']), fontsize=7, va='center')
        c.text(row['n_in_band']/2, yy, str(row['n_in_band']), fontsize=7, va='center', ha='center')
    c.set(xlim=(0, 180), ylim=(-.6, len(rows)-.4), xlabel='Compounds (n)')
    c.set_yticks(y, [r['label'] for r in rows], fontsize=7)
    c.set_xticks([0, 50, 100, 150]); c.tick_params(axis='y', length=0, pad=3)
    c.spines['left'].set_visible(False)
    fig.legend(*c.get_legend_handles_labels(), loc='center', ncol=2, frameon=False,
               bbox_to_anchor=(129/160, 1-97/158), fontsize=7, handlelength=1.1,
               columnspacing=1.2, handletextpad=.4)
    save(fig, out, 2)


def figure3(out):
    """Empirical donor ordering and panel choices, then explicitly marked extrapolation."""
    core = load('core_numbers'); c = core['collectible six']
    fig, bg = canvas(146)
    # Row 1's labels end at 71 mm; row 2 begins after the common 12 mm gutter.
    second_row = 71 + ROW_GAP
    panel(bg, PANEL_LEFT, 5, 'a')
    panel(bg, PANEL_RIGHT, 5, 'b')
    panel(bg, PANEL_LEFT, second_row, 'c')
    panel(bg, PANEL_RIGHT, second_row, 'd')
    audit, ci = load('reconstruct_metric_audit'), load('fig1_ci')
    rows = sorted(((k, v) for k, v in audit.items() if k != 'uniform (all 9)'),
                  key=lambda kv: -kv[1]['r'])
    a = axes(fig, 25, 12, 42, 50)
    y = np.arange(len(rows))[::-1]
    for yy, (name, v) in zip(y, rows):
        err = 1.96*ci[name]['se']
        a.errorbar(v['r'], yy, xerr=err, fmt='o', color=BLUE,
                   ecolor=INK, markersize=3.7, elinewidth=.8, capsize=2,
                   markeredgecolor='white', markeredgewidth=.35)
        a.text(.96, yy, f"{v['r']:.3f}", fontsize=7, va='center', ha='right')
    a.set(ylim=(-.6, 8.6), xlim=(0, .82), xlabel='Mean Pearson correlation')
    a.set_yticks(y, [NAMES[k] for k, _ in rows]); a.set_xticks([0, .2, .4, .6])
    a.tick_params(axis='y', length=0, pad=4)
    a.spines['left'].set_visible(False)
    label(bg, 74, 8, 'r', 7, ha='right', style='italic')

    def panel_spread(ax, key):
        d = core[key]; kk = np.array([r['k'] for r in d])
        ax.fill_between(kk, [r['worst'] for r in d], [r['best'] for r in d],
                        color=BLUE, alpha=.08, linewidth=0)
        configs = [('best', ORANGE, 'o', '-', 'Best'),
                   ('mean', BLUE, 's', '-', 'Mean'),
                   ('worst', GREY, '^', '--', 'Worst')]
        for field, col, mark, ls, name in configs:
            ax.plot(kk, [r[field] for r in d], color=col, marker=mark,
                    ls=ls, ms=3.3, lw=1.1, markeredgecolor='white',
                    markeredgewidth=.3, label=name)
        ax.set(xlim=(.75, max(kk)+.25), ylim=(.17, .72),
               xlabel='Number of donor species', ylabel='Correlation with human')
        ax.set_xticks(kk); ax.set_yticks([.2, .4, .6])
        return ax

    b = panel_spread(axes(fig, 96, 12, 59, 50), 'spread_collectible six')
    panel_spread(axes(fig, 17, 91, 59, 43), 'spread_all nine')
    # Same encodings in both subset panels. The envelope is the full enumeration, not a CI.
    b.legend(loc='upper right', ncol=3, frameon=False, columnspacing=.6,
             handletextpad=.3, handlelength=1.1, fontsize=7)

    d = axes(fig, 96, 91, 59, 43)
    k = np.geomspace(1, 400, 600)
    expected = k*c['rbar']/np.sqrt(k+k*(k-1)*c['rho'])
    d.axvspan(6, 400, color='#F2F4F7', zorder=0)
    d.axvline(6, color=GREY, lw=.65, ls=(0, (2, 2)))
    inside = k <= 6
    d.plot(k[inside], expected[inside], color=BLUE, lw=1.3)
    # Include the exact endpoint so solid and dashed curves meet without a gap.
    ext = np.geomspace(6, 400, 400)
    d.plot(ext, ext*c['rbar']/np.sqrt(ext+ext*(ext-1)*c['rho']),
           color=BLUE, lw=1.2, ls=(0, (3, 2)))
    d.plot([r['k'] for r in core['spread_collectible six']],
           [r['mean'] for r in core['spread_collectible six']],
           's', color=BLUE, ms=3.4, mec='white', mew=.35)
    d.axhline(c['asymptote'], color=BLUE, lw=.65, ls=(0, (4, 2)))
    d.axhline(core['macaque'], color=PURPLE, lw=.7, ls=(0, (1, 2)))
    d.axhline(core['chimp'], color=GREY, lw=.7, ls=(0, (1, 2)))
    d.plot(core['k_to_match_macaque'], core['macaque'], 'D', color=PURPLE, ms=4,
           mec='white', mew=.3)
    d.legend(handles=[
        Line2D([], [], color=GREY, lw=.7, ls=':', label='Chimpanzee'),
        Line2D([], [], color=BLUE, lw=.7, ls='--', label='Asymptote'),
        Line2D([], [], color=PURPLE, lw=.7, ls=':', label='Macaque'),
    ], loc='upper left', frameon=True, facecolor='white', edgecolor='white',
       framealpha=1, fontsize=7, handlelength=1.8, labelspacing=.3)

    d.set(xscale='log', xlim=(.9, 430), ylim=(.18, .74),
          xlabel='Number of species (log scale)', ylabel='Correlation with human')
    d.set_xticks([1, 6, 20, 100, 400], ['1', '6', '20', '100', '400'])
    d.set_yticks([.2, .4, .6]); d.tick_params(axis='x', which='minor', bottom=False)
    save(fig, out, 3)


def coefficient_heatmap(fig, bg, x, top, rows, names):
    fields = [('spearman_all', 'p_all'), ('spearman_cross', 'p_cross'),
              ('partial_taxdist', 'p_partial')]
    ax = axes(fig, x, top, 43, 27)
    values = np.array([[r[field] for field, _ in fields] for r in rows])
    mesh = ax.pcolormesh(values, cmap=RESPONSE, vmin=-1, vmax=1,
                        edgecolors='white', linewidth=1.5, shading='flat')
    ax.set(xlim=(0, 3), ylim=(3, 0))
    ax.set_xticks(np.arange(3)+.5, ['All\npairs', 'Cross-\norder', 'Taxonomy\nadjusted'], fontsize=7)
    ax.set_yticks(np.arange(3)+.5, names, fontsize=7)
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i, row in enumerate(rows):
        for j, (field, pfield) in enumerate(fields):
            val = row[field]; star = '*' if row[pfield] < .05 else ''
            # Dark text on this bright, moderately saturated diverging palette is legible.
            ax.text(j+.5, i+.5, f'{val:.3f}{star}', ha='center', va='center', fontsize=8)
    return mesh


def figure4(out):
    fig, bg = canvas(158)
    # Three content rows end at 43, 98 and 155 mm, separated by 12 mm gutters.
    middle_row, last_row = 43 + ROW_GAP, 98 + ROW_GAP
    panel(bg, PANEL_LEFT, 5, 'a')
    p, q = toy_contexts()
    idx = np.triu_indices(4, 1)
    for mat, xx, name in [(p, 10, 'Context 1'), (q, 36, 'Context 2')]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 18, upper, 3.9)
        label(bg, xx+7.8, 9.5, name, 7, ha='center')
        for j, letter in enumerate('ABCD'):
            label(bg, xx+1.95+j*3.9, 15, letter, 7, ha='center')
            label(bg, xx-1, 19.95+j*3.9, letter, 7, ha='right')
        bg.add_patch(Rectangle((xx+2*3.9, 18), 3.9, 3.9, fill=False, ec=PURPLE, lw=1.1))
    arrow(bg, (54, 24), (62, 24))
    for yy, mat in [(19, p), (27, q)]:
        cells(bg, 65, yy, mat[idx][None, :], 4)
    for j, name in enumerate(['AB', 'AC', 'AD', 'BC', 'BD', 'CD']):
        label(bg, 67+j*4, 15.7, name, 5.5, ha='center')
    arrow(bg, (92, 24), (101, 24))
    sc = axes(fig, 113, 11, 31, 21)
    # Use the same average-tie convention as the empirical analysis for this teaching example.
    ranks = [centered_ranks(m[idx])+(len(idx[0])+1)/2 for m in (p, q)]
    sc.plot([.5, 6.5], [.5, 6.5], color=LINE, lw=.75)
    sc.scatter(*ranks, s=15, color=BLUE, edgecolors='white', linewidths=.35, zorder=3)
    j = 1  # Pair A-C, also outlined in both matrices.
    sc.scatter(ranks[0][j], ranks[1][j], s=26, facecolors='none', edgecolors=PURPLE, lw=.9)
    sc.set(xlim=(.5, 6.5), ylim=(.5, 6.5), xlabel='Rank in context 1', ylabel='Rank in context 2')
    sc.set_xticks([1, 3, 6]); sc.set_yticks([1, 3, 6])
    sc.tick_params(labelsize=7)
    sc.xaxis.label.set_size(7); sc.yaxis.label.set_size(7)

    data = collect(load('rho_cross_panel_both_estimators'))
    rel = [r for _, r, isrel in data if isrel]
    times = [r for _, r, _ in data if r.get('kind') == 'cross_time']
    tissues = [r for _, r, _ in data if r.get('kind', '').startswith('cross_panel_macrophage')]
    assert len(rel) == 3 and len(times) == 3 and len(tissues) == 2
    panel(bg, PANEL_LEFT, middle_row, 'b')
    panel(bg, PANEL_RIGHT, middle_row, 'c')
    coefficient_heatmap(fig, bg, 22, 62, rel, ['2 h', '6 h', '24 h'])
    coefficient_heatmap(fig, bg, 108, 62, times, ['2 vs 6 h', '2 vs 24 h', '6 vs 24 h'])
    colour_key(fig, 71, 73, 14, vmin=-1, vmax=1, title='Spearman r')
    panel(bg, PANEL_LEFT, last_row, 'd')
    d = axes(fig, 47, 120, 96, 26)
    cfg = [('spearman_all', 'p_all', GREY, 'o', 'All pairs'),
           ('spearman_cross', 'p_cross', BLUE, 's', 'Cross-order'),
           ('partial_taxdist', 'p_partial', PURPLE, 'D', 'Taxonomy adjusted')]
    for yy, row in zip([1, 0], tissues):
        for j, (field, pfield, color, marker, _) in enumerate(cfg):
            v = row[field]; pos = yy + (1-j)*.23
            d.plot(v, pos, marker=marker, ms=3.4, color=color, mec='white', mew=.3)
            d.text(v+.027, pos, f'{v:+.3f}'+('*' if row[pfield] < .05 else ''),
                   fontsize=7, va='center')
    d.set(xlim=(-.45, .80), ylim=(-.5, 1.5), xlabel='Matrix agreement (Spearman)')
    d.axvline(0, color=GREY, lw=.65)
    d.set_yticks([1, 0], ['Pool profiles,\nthen correlate', 'Correlate per condition,\nthen average'], fontsize=7)
    d.set_xticks([-.4, 0, .4, .8]); d.tick_params(axis='y', length=0)
    d.spines['left'].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=3.4, label=l)
                        for _, _, c, m, l in cfg], loc='center',
               bbox_to_anchor=(111/160, 1-111/158), ncol=3, frameon=False,
               columnspacing=.8, handletextpad=.3, fontsize=7)
    save(fig, out, 4)


def figure5(out):
    fig, bg = canvas(140)
    # Two equal-width columns; both panel-letter columns share the same x positions.
    second_row = 51 + ROW_GAP
    panel(bg, PANEL_LEFT, 5, 'a')
    panel(bg, PANEL_RIGHT, 5, 'b')
    for i, (v, title) in enumerate(zip(decomposition(), ['Response', 'Mean', 'Residual'])):
        xx = 11+i*23
        cells(bg, xx, 20.5, v, 3, vmin=-2, vmax=2)
        label(bg, xx+6, 11, title, 8, ha='center')
        for j in range(4):
            label(bg, xx+1.5+j*3, 17.2, str(j+1), 7, ha='center')
        if i == 0:
            for j in range(5):label(bg, xx-1.5, 22+3*j, f'g{j+1}', 7, ha='right')
    operator(bg, 28.5, 28, '-')
    operator(bg, 51.5, 28, '=')
    colour_key(fig, 27, 45, 26, vmin=-2, vmax=2, title='Response (a.u.)')
    # The pairing map shows the control operation without pretending it is observed data.
    for i, j in np.ndindex(4, 4):
        bg.add_patch(Rectangle((114+j*4.5, 20.5+i*4.5), 4.5, 4.5,
                              fc=BLUE if i == j else '#E6EBF0', ec='white', lw=.8))
    for j in range(4):
        label(bg, 116.25+j*4.5, 17.2, str(j+1), 7, ha='center')
        label(bg, 112.5, 22.75+j*4.5, str(j+1), 7, ha='right')
    label(bg, 123, 11, 'Human condition', 8, ha='center')
    label(bg, 104, 29.5, 'Donor condition', 8, ha='center', rotation=90)
    for xx, col, name in [(94, BLUE, 'Matched'), (122, '#D9E1E8', 'Mismatched')]:
        bg.add_patch(Rectangle((xx, 45-1.1), 2.2, 2.2, fc=col, ec='none'))
        label(bg, xx+4, 45, name, 7)
    panel(bg, PANEL_LEFT, second_row, 'c')
    panel(bg, PANEL_RIGHT, second_row, 'd')
    data = load('reconstruct_specificity')['blocks']; raw, res = data['raw'], data['residual']
    order = sorted(raw, key=lambda k: -raw[k]['matched']); yy = np.arange(len(order))[::-1]
    c = axes(fig, 29, 75, 47, 52)
    d = axes(fig, 100, 75, 55, 52)
    total = np.array([raw[k]['matched'] for k in order])
    mis = np.array([raw[k]['mismatched_median'] for k in order])
    c.hlines(yy, mis, total, color='#BAC6D1', lw=1)
    c.plot(mis, yy, 's', color=GREY, ms=3.2, label='Mismatched')
    c.plot(total, yy, 'o', color=BLUE, ms=3.7, label='Matched', mec='white', mew=.3)
    names = ['All nine (mean)' if k == 'uniform (all 9)' else NAMES[k.removeprefix('only ')]
             for k in order]
    c.set_yticks(yy, names, fontsize=7)
    c.set(xlim=(-.12, .54), ylim=(-.6, 9.6), xlabel='Raw recovery')
    c.set_xticks([-.1, .1, .3, .5])
    vals = [res[k]['residual_recovery'] for k in order]
    for y, v in zip(yy, vals):
        col = ORANGE if v > 0 else GREY
        d.hlines(y, 0, v, color=col, lw=1.2)
        d.plot(v, y, 'o' if v > 0 else 's', color=col, ms=3.6, mec='white', mew=.3)
        d.text(v+(.017 if v > 0 else -.017), y, f'{v:+.3f}', ha='left' if v > 0 else 'right',
               va='center', fontsize=7)
    d.set(xlim=(-.51, .29), ylim=(-.6, 9.6), xlabel='Residual recovery')
    d.set_xticks([-.4, -.2, 0, .2]); d.set_yticks(yy, [])
    for ax in [c, d]:
        ax.axvline(0, color=INK, lw=.6)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
    fig.legend(*c.get_legend_handles_labels(), loc='center',
               bbox_to_anchor=(48/160, 1-68/140), ncol=2, frameon=False,
               handletextpad=.35, columnspacing=1, fontsize=7)
    save(fig, out, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/paper/tex/figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    for draw in (figure1, figure2, figure3, figure4, figure5): draw(args.out)


if __name__ == '__main__':
    main()
