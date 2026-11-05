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
BLUE, ORANGE, PURPLE = '#39A9E6', '#FFA657', '#B18BE3'
TEAL, GREY, INK = '#46C7B4', '#A6AFBD', '#000000'
LIGHT, LINE = '#F4F7FA', '#D7DFE6'
MIN_TEXT, BODY_TEXT, PANEL_TEXT = 7, 8, 8
PANEL_LEFT, PANEL_RIGHT, ROW_GAP = 3, 83, 12
RESPONSE = LinearSegmentedColormap.from_list('response', ['#45AFE1', '#FFFFFF', '#FF9B75'])
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


def toy_replicates():
    # A separate teaching example: both within-side correlations exceed the
    # analysis's 0.10 reliability floor. Species in toy_responses are not replicates.
    return np.array([[1.4, 0.8, -1.1, -0.6, 0.2, -0.5],
                     [0.9, 1.1, -0.7, -1.2, 0.4, -0.2],
                     [1.0, 0.5, -0.8, 0.1, 0.7, -0.4],
                     [0.7, 0.7, -1.0, -0.2, 0.5, 0.1]])


def decomposition():
    x = np.array([[1.7, 1.2, 0.8, 1.5], [-1.2, -0.4, -0.8, -1.6],
                  [0.8, -0.4, 0.5, -0.1], [-0.3, 0.9, -0.9, 0.3],
                  [0.2, 0.7, 0.9, 0.2]])
    shared = np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1)
    return x, shared, x-shared


def save(fig, out, n):
    # One typographic hierarchy for all data axes, including schematic data axes.
    # Diagram labels and legends remain regular; axes are 8 pt, ticks are 7 pt.
    for ax in fig.axes[1:]:
        for title in (ax.xaxis.label, ax.yaxis.label):
            title.set_fontsize(BODY_TEXT)
            title.set_fontweight('bold')
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontsize(MIN_TEXT)
            tick.set_fontweight('bold')
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
    region(bg, 3, 11, 41, 36, '#F1F9FE', '#B6E1F5')
    label(bg, 7, 16, 'Experimental contexts', 8, weight='bold')
    for y, color, title, desc in [
        (23, BLUE, 'Blood panel', 'Species × dose × time'),
        (35, ORANGE, 'TG-GATEs', 'Human / rat; culture / liver'),
    ]:
        bg.add_patch(Circle((8, y), 1.25, fc=color, ec='none'))
        label(bg, 11, y, title, 7, weight='bold')
        label(bg, 7, y+5, desc, 6.3)
    arrow(bg, (45.5, 29), (51.5, 29))
    # Stack treated and control profiles on the same horizontal gene axis.
    perturb = np.array([[2.1, 1.8, .4, .8, 1.2, 1.0],
                        [2.3, 1.4, .6, 1.1, 1.8, .6],
                        [1.8, 2.0, .2, 1.0, 1.4, .9]])
    control = np.array([[1., 1.1, 1.3, 1.4, 1.0, 1.2],
                        [1.2, 1.0, 1.1, 1.7, 1.3, 1.0],
                        [.9, 1.2, 1.4, 1.6, 1.1, 1.3]])
    seq = LinearSegmentedColormap.from_list('expression', ['#F1FAFE', BLUE])
    cells(bg, 66, 15, perturb, 3.6, vmin=0, vmax=2.5, cmap=seq)
    cells(bg, 66, 32, control, 3.6, vmin=0, vmax=2.5, cmap=seq)
    label(bg, 63, 20.4, 'Perturbed', 7, ha='right')
    label(bg, 63, 37.4, 'Control', 7, ha='right')
    operator(bg, 76.8, 29, '-')
    label(bg, 76.8, 47, 'Gene alignment', 7, ha='center')
    arrow(bg, (92, 29), (103, 29))
    # Actual analytical matrices are continuous responses; these values only teach the layout.
    cells(bg, 117, 19, perturb-control, 4.6, vmin=-1.5, vmax=1.5)
    label(bg, 130.8, 12, 'Response matrix', 8, weight='bold', ha='center')
    label(bg, 130.8, 36, 'Genes', 7, ha='center')
    label(bg, 109.5, 25.9, 'Subject ×\ncondition', 7, ha='center', rotation=90)
    colour_key(fig, 119, 44, 29, vmin=-1.5, vmax=1.5, title='Response (a.u.)')
    # A fork: species redundancy and profile-level ceilings are distinct constructions.
    bg.plot([146.5, 154, 154, 40], [26, 26, 51, 51], color=LINE, lw=.8)
    arrow(bg, (40, 51), (40, 55), color=BLUE)
    arrow(bg, (121, 51), (121, 55), color=ORANGE)
    label(bg, 43, 54, 'Blood panel', 7)
    label(bg, 124, 54, 'TG-GATEs', 7)
    # Every numbered heading sits 6 mm above its content area. The branch
    # boxes group calculations; they do not enclose a different heading level.
    stage(bg, 3, 57, 2, 'Quantify species redundancy', BLUE)
    stage(bg, 82, 57, 3, 'Estimate replicate ceiling', ORANGE)
    region(bg, 3, 63, 75, 52, 'white', '#B6E1F5')
    region(bg, 82, 63, 75, 52, 'white', '#FFD7B1')
    x = toy_responses()
    cells(bg, 12, 73, x, 3.3, vmin=-1.5, vmax=1.5)
    for i, name in enumerate('ABCD'):
        label(bg, 10, 74.65+3.3*i, name, 7, ha='right')
    label(bg, 21.9, 68, 'Species means', 7, ha='center')
    arrow(bg, (35, 79.6), (49, 79.6), color=BLUE)
    label(bg, 42, 74, 'Pearson', 7, ha='center')
    cells(bg, 53, 73, np.corrcoef(x), 3.3)
    label(bg, 59.6, 68, 'Redundancy', 7, ha='center')
    bg.plot([8, 73], [91, 91], color=LINE, lw=.65)
    # Two compact reliability objects, not a second empirical heatmap.
    p, q = toy_contexts()
    for mat, xx, name in [(p, 15, 'Split 1'), (q, 48, 'Split 2')]:
        a = mat.copy(); a[np.tril_indices(4)] = np.nan
        cells(bg, xx, 97, a, 2.6)
        label(bg, xx+5.2, 94, name, 7, ha='center')
    arrow(bg, (30, 102), (43, 102), color=BLUE)
    label(bg, 40.5, 111.5, 'Subject splits test pair-rank agreement', 7, ha='center')
    # The ceiling uses replicate response correlations within sides, not rho split halves.
    reps = toy_replicates()
    strips = [(89, 73, 'A1', reps[0]), (89, 95, 'A2', reps[1]),
              (134, 73, 'B1', reps[2]), (134, 95, 'B2', reps[3])]
    for xx, yy, name, values in strips:
        cells(bg, xx, yy, values[None, :], 2.6, vmin=-1.5, vmax=1.5)
        label(bg, xx+7.8, yy-4 if yy == 73 else yy+6, name, 7, ha='center')
    arrow(bg, (96.8, 77.5), (96.8, 92.5), color=BLUE)
    arrow(bg, (141.8, 77.5), (141.8, 92.5), color=ORANGE)
    label(bg, 94, 85, 'rel A', 7, ha='right')
    label(bg, 144.5, 85, 'rel B', 7)
    for start, end in [(74.3, 74.3), (96.3, 96.3), (74.3, 96.3), (96.3, 74.3)]:
        arrow(bg, (107, start), (131.5, end), color=GREY, lw=.7)
    label(bg, 119.3, 85, 'Agreement', 7, ha='center',
          bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.2})
    label(bg, 119.5, 109, 'Fraction = agreement / √(rel A × rel B)', 7, ha='center')
    # Both constructions constrain interpretation, with different validation requirements.
    arrow(bg, (40, 115.5), (40, 121), color=BLUE)
    arrow(bg, (120, 115.5), (120, 121), color=ORANGE)
    region(bg, 3, 122, 154, 9, '#F9F5FD', '#DFD1F1')
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
        label(bg, xx+19, 165, title, 7, weight='bold', ha='center')
        label(bg, xx+19, 170, pointer, 7, ha='center')
    # Equal-weight saturation under an illustrative exchangeable-donor model.
    k = np.arange(1, 18)
    expected = k*.3 / np.sqrt(k+k*(k-1)*.3)
    bg.plot([9, 9, 36], [147, 160, 160], color=GREY, lw=.6)
    bg.plot(9+27*(k-1)/16, 160-13*expected/.6, color=BLUE, lw=1.2)
    # Context reuse: match the same upper triangles, rather than implying a learned mapping.
    for xx, mat in [(44, p), (65, q)]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 149, upper, 2.5)
    arrow(bg, (55, 154), (62, 154), color=PURPLE)
    # Specificity is a subtraction within species, using exactly the illustrated decomposition.
    response, shared, residual = decomposition()
    cells(bg, 83, 149, response, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    operator(bg, 93, 153.75, '-')
    cells(bg, 95.5, 149, shared, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    operator(bg, 105.5, 153.75, '=')
    cells(bg, 108, 149, residual, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    # A ceiling is an upper reference, with observed agreement expressed relative to it.
    bg.plot([125, 152], [159, 159], color=GREY, lw=.7)
    bg.plot([125, 145], [153, 153], color=TEAL, lw=1.5)
    bg.plot([152, 152], [149, 160], color=INK, lw=.65, ls=(0, (2, 2)))
    label(bg, 149, 147, 'Ceiling', 7, ha='center')
    save(fig, out, 1)


def figure2(out):
    """Dataset scope and the actual input coverage of the two core analyses."""
    data = collect_dataset_overview()
    (out/'Fig2_source_data.json').write_text(json.dumps(data, indent=2)+'\n')
    fig, bg = canvas(158)
    panel(bg, PANEL_LEFT, 5, 'a')
    label(bg, 7, 12, 'Dataset', 8, weight='bold')
    label(bg, 56, 12, 'Species (n)', 8, ha='center', weight='bold')
    label(bg, 99, 12, 'Biological system', 8, ha='center', weight='bold')
    label(bg, 141, 12, 'Study role', 8, ha='center', weight='bold')
    bg.plot([5, 157], [16, 16], color=INK, lw=.65)
    role_colours = {'Core analyses': BLUE, 'Additional panels': TEAL, 'External check': PURPLE}
    role_fills = {'Core analyses': '#EAF6FE', 'Additional panels': '#EAFBF5',
                  'External check': '#F5EDFC'}
    for role, fill in role_fills.items():
        indices = [i for i, r in enumerate(data['datasets']) if r['role'] == role]
        top = 17+8*min(indices)
        bg.add_patch(Rectangle((5, top), 152, 8*len(indices), fc=fill, ec='white', lw=1.5))
        label(bg, 141, top+4*len(indices), role, 7, ha='center')
    a = axes(fig, 39, 17, 34, 56)
    a.set_facecolor('none')
    for i, row in enumerate(data['datasets']):
        yy = 21+8*i
        color = role_colours[row['role']]
        label(bg, 7, yy, row['name'], 8)
        label(bg, 99, yy, row['system'], 7, ha='center')
        a.barh(i, row['n_species'], height=.45, color=color, linewidth=0)
        a.text(row['n_species']+.35, i, str(row['n_species']), va='center', fontsize=8)
    a.set(xlim=(0, 12), ylim=(6.5, -.5))
    a.set_xticks([0, 5, 10]); a.set_yticks([])
    a.spines['left'].set_visible(False)

    # More area for the two coverage plots, with shared top and bottom baselines.
    lower = 87
    panel(bg, PANEL_LEFT, lower, 'b')
    panel(bg, PANEL_RIGHT, lower, 'c')
    blood = data['blood']
    b = axes(fig, 23, 96, 50, 46)
    seq = ListedColormap(['#A4DCF5'])
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

    c = axes(fig, 112, 96, 43, 46)
    rows = data['tggates']['comparisons']
    y = np.arange(len(rows))[::-1]
    c.barh(y, [r['n_shared'] for r in rows], color='#E3E8F0', height=.58, label='Shared')
    c.barh(y, [r['n_in_band'] for r in rows], color=ORANGE, height=.58, label='In band')
    for yy, row in zip(y, rows):
        c.text(row['n_shared']+4, yy, str(row['n_shared']), fontsize=7, va='center')
        c.text(row['n_in_band']/2, yy, str(row['n_in_band']), fontsize=7, va='center', ha='center')
    c.set(xlim=(0, 180), ylim=(-.6, len(rows)-.4), xlabel='Compounds (n)')
    c.set_yticks(y, [r['label'] for r in rows], fontsize=7)
    c.set_xticks([0, 50, 100, 150]); c.tick_params(axis='y', length=0, pad=3)
    c.spines['left'].set_visible(False)
    fig.legend(*c.get_legend_handles_labels(), loc='center', ncol=2, frameon=False,
               bbox_to_anchor=(129/160, 1-89/158), fontsize=7, handlelength=1.1,
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
    # The asymptote is only 0.0025 above macaque: a second horizontal line
    # cannot be distinguished at print size. Report it in the caption instead.
    d.axhline(core['macaque'], color=PURPLE, lw=.7, ls=(0, (1, 2)))
    d.axhline(core['chimp'], color=GREY, lw=.7, ls=(0, (1, 2)))
    d.plot(core['k_to_match_macaque'], core['macaque'], 'D', color=PURPLE, ms=4,
           mec='white', mew=.3)
    d.legend(handles=[
        Line2D([], [], color=GREY, lw=.7, ls=':', label='Chimpanzee'),
        Line2D([], [], color=BLUE, lw=1.2, ls='--', label='Pooling model'),
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
    ax = axes(fig, x, top, 49, 34)
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
    fig, bg = canvas(162)
    middle_row, last_row = 52, 116
    panel(bg, PANEL_LEFT, 5, 'a')
    p, q = toy_contexts()
    idx = np.triu_indices(4, 1)
    for mat, xx, name in [(p, 9, 'Context 1'), (q, 38, 'Context 2')]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 21, upper, 4.7)
        label(bg, xx+9.4, 12, name, 7, ha='center')
        for j, letter in enumerate('ABCD'):
            label(bg, xx+2.35+j*4.7, 17.5, letter, 7, ha='center', weight='bold')
            label(bg, xx-1, 23.35+j*4.7, letter, 7, ha='right', weight='bold')
        bg.add_patch(Rectangle((xx+2*4.7, 21), 4.7, 4.7, fill=False, ec=PURPLE, lw=1.1))
    arrow(bg, (60, 29), (67, 29))
    for yy, mat in [(23, p), (33, q)]:
        cells(bg, 70, yy, mat[idx][None, :], 4.2)
    for j, name in enumerate(['AB', 'AC', 'AD', 'BC', 'BD', 'CD']):
        label(bg, 72.1+j*4.2, 19, name, 7, ha='center', weight='bold')
    arrow(bg, (98, 29), (105, 29))
    sc = axes(fig, 121, 14, 33, 25)
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
    coefficient_heatmap(fig, bg, 22, 60, rel, ['2 h', '6 h', '24 h'])
    coefficient_heatmap(fig, bg, 106, 60, times, ['2 vs 6 h', '2 vs 24 h', '6 vs 24 h'])
    colour_key(fig, 24, 45.5, 27, vmin=-1, vmax=1, title='Correlation')
    panel(bg, PANEL_LEFT, last_row, 'd')
    d = axes(fig, 43, 126, 98, 23)
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
    d.set_yticks([1, 0], ['Pool profiles,\nthen correlate', 'Correlate by condition,\nthen average'], fontsize=7)
    d.set_xticks([-.4, 0, .4, .8]); d.tick_params(axis='y', length=0)
    d.spines['left'].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=3.4, label=l)
                        for _, _, c, m, l in cfg], loc='center',
               bbox_to_anchor=(80/160, 1-116/162), ncol=3, frameon=False,
               columnspacing=.8, handletextpad=.3, fontsize=7)
    save(fig, out, 4)


def figure5(out):
    fig, bg = canvas(145)
    # The decomposition uses two thirds of the top row; the result plots share a baseline.
    second_row = 56 + ROW_GAP
    panel(bg, PANEL_LEFT, 5, 'a')
    panel(bg, 109, 5, 'b')
    for i, (v, title) in enumerate(zip(decomposition(), ['Response', 'Mean', 'Residual'])):
        xx = 12+i*32
        cells(bg, xx, 20.5, v, 4.3, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.4)
        label(bg, xx+8.6, 11, title, 8, ha='center')
        for j in range(4):
            label(bg, xx+2.15+j*4.3, 17.2, str(j+1), 7, ha='center', weight='bold')
        if i == 0:
            for j in range(5):
                label(bg, xx-1.5, 22.65+4.3*j, f'g{j+1}', 7, ha='right', weight='bold')
    operator(bg, 36.6, 31.25, '-')
    operator(bg, 68.6, 31.25, '=')
    colour_key(fig, 39, 51, 27, vmin=-2, vmax=2, title='Response (a.u.)')
    # The pairing map shows the control operation without pretending it is observed data.
    for i, j in np.ndindex(4, 4):
        bg.add_patch(Rectangle((129+j*6, 20.5+i*6), 6, 6,
                              fc=BLUE if i == j else '#E3E8F0', ec='white', lw=.8))
    for j in range(4):
        label(bg, 132+j*6, 17.2, str(j+1), 7, ha='center', weight='bold')
        label(bg, 127.5, 23.5+j*6, str(j+1), 7, ha='right', weight='bold')
    label(bg, 141, 11, 'Human condition', 8, ha='center', weight='bold')
    label(bg, 120, 32.5, 'Donor condition', 8, ha='center', rotation=90, weight='bold')
    for yy, col, name in [(49, BLUE, 'Matched'), (54, '#E3E8F0', 'Mismatched')]:
        bg.add_patch(Rectangle((129, yy-1.1), 2.2, 2.2, fc=col, ec='none'))
        label(bg, 133, yy, name, 7)
    panel(bg, PANEL_LEFT, second_row, 'c')
    panel(bg, PANEL_RIGHT, second_row, 'd')
    data = load('reconstruct_specificity')['blocks']; raw, res = data['raw'], data['residual']
    order = sorted(raw, key=lambda k: -raw[k]['matched']); yy = np.arange(len(order))[::-1]
    c = axes(fig, 29, 80, 47, 52)
    d = axes(fig, 100, 80, 55, 52)
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
               bbox_to_anchor=(48/160, 1-73/145), ncol=2, frameon=False,
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
