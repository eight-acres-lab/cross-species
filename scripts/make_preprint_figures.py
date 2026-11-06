#!/usr/bin/env python3
"""Draw a workflow, a dataset overview and three result composites at 160 mm width.

Empirical marks read deposited JSON estimates, without rerunning or changing analyses.
Teaching matrices and curves are explicitly illustrative and are never substituted for
measurements. PDF/SVG elements, including matrix cells and type, remain editable vectors.

Optional alignment QA: add the installed nature-figure skill's scripts directory to
PYTHONPATH and pass --audit-dir PATH. Normal reproduction needs only this repository's
dependencies. The audit helper is not needed to generate the figures.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re

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
INDIGO, OCHRE, PETROL = '#4C54A0', '#CA9424', '#02779B'
GREEN, GREY, INK = '#148843', '#7C8797', '#000000'
PERIWINKLE, CORAL = '#6F78B9', '#F37252'
# Categorical time colours are separate from the signed numerical colour scale.
BLOOD_TIME_COLOURS = {2: PERIWINKLE, 6: OCHRE, 24: GREEN}
LIGHT, LINE = '#F4F7FA', '#D7DFE6'
MIN_TEXT, BODY_TEXT, PANEL_TEXT = 7, 8, 8
PANEL_LEFT, PANEL_RIGHT, ROW_GAP = 3, 83, 12
ALIGNMENT_DIR = None
# Saturated endpoints keep zero white and retain the same linear, symmetric scale.
# Black text remains legible even at the most saturated end of the map.
RESPONSE = LinearSegmentedColormap.from_list('response', [PERIWINKLE, '#FFFFFF', CORAL])
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
    if 'rotation' in kw:
        kw.setdefault('rotation_mode', 'anchor')
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


def audit_alignment(fig, out, n):
    """Measure final geometry; QA stays optional for the public reproduction script."""
    from audit_panel_alignment import (audit_layout_manifest, exit_code,
                                       require_matplotlib_panel_alignment)
    # The full-page drawing canvas and colour bars are not empirical plot areas.
    # Panel a of Fig. 4 is a teaching schematic, while d spans both columns.
    definitions = {
        1: ([0], ['workflow'], [], []),
        2: ([1, 2, 3], ['a', 'b', 'c'], [['b', 'c']], []),
        3: ([1, 2, 3, 4], ['a', 'b', 'c', 'd'], [['a', 'b'], ['c', 'd']], [['b', 'd']]),
        4: ([1, 2, 3, 5], ['a', 'b', 'c', 'd'], [['b', 'c']], []),
        5: ([2, 3], ['c', 'd'], [['c', 'd']], []),
    }
    indices, ids, rows, columns = definitions[n]
    report = require_matplotlib_panel_alignment(
        fig, axes=[fig.axes[i] for i in indices], panel_ids=ids,
        row_groups=rows, column_groups=columns, strict=True,
        json_out=out/f'Fig{n}.alignment.json',
    )
    # Panel letters live on the physical canvas rather than on differently sized
    # data axes. Attach their measured anchors, then check those too.
    letters = {t.get_text(): t for t in fig.axes[0].texts
               if t.get_text() in ('a', 'b', 'c', 'd') and t.get_fontweight() == 'bold'}
    if n != 1:
        for p in report['layout']['panels']:
            text = letters[p['id']]
            p['panel_label'] = p['id']
            p['panel_label_anchor_pt'] = list(
                text.get_transform().transform(text.get_position()) * 72 / fig.dpi)
        report = audit_layout_manifest(report['layout'], require_panel_labels=True)
        if exit_code(report, strict=True):
            raise RuntimeError(f'Figure {n}: panel or label alignment failed: {report}')
    # Schematic/hero panels do not pretend to share empirical axis geometry.
    report['scope_notes'] = {
        1: 'One continuous schematic; colour key excluded. Steps 2/3 use equal boxes.',
        2: 'Inventory spans the full width; b/c share top/bottom edges.',
        3: 'a needs room for species names and estimates; b/c share width, height and y scale.',
        4: 'a is a teaching schematic; b/c share geometry and scale; d spans both columns.',
        5: 'a/b are schematic matrices with different roles; c/d align donor rows but use different denominators.',
    }[n]
    # Check all visible letter rows/columns, including drawing-only panels.
    letter_groups = {
        2: [('y', 'bc'), ('x', 'ab')],
        3: [('y', 'ab'), ('y', 'cd'), ('x', 'ac'), ('x', 'bd')],
        4: [('y', 'bc'), ('x', 'abd')],
        5: [('y', 'ab'), ('y', 'cd'), ('x', 'ac')],
    }
    report['canvas_letter_alignment'] = []
    for coordinate, group in letter_groups.get(n, []):
        index = 0 if coordinate == 'x' else 1
        values = [letters[k].get_transform().transform(letters[k].get_position())[index]
                  * 72 / fig.dpi for k in group]
        deviation = max(values)-min(values)
        assert deviation <= 1.5, (n, group, deviation)
        report['canvas_letter_alignment'].append({
            'letters': group, 'coordinate': coordinate, 'max_deviation_pt': deviation})
    (out/f'Fig{n}.alignment.json').write_text(json.dumps(report, indent=2)+'\n')


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
    if ALIGNMENT_DIR is not None:
        audit_alignment(fig, ALIGNMENT_DIR, n)
    # Fixed page boxes preserve the documented print size; never use bbox_inches='tight'.
    fig.savefig(out/f'Fig{n}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    svg = out/f'Fig{n}.svg'
    fig.savefig(svg, metadata={'Date': None})
    # Matplotlib leaves trailing spaces in multiline SVG path data. Normalize the
    # serializer output so the editable files also pass the repository whitespace check.
    svg_text = svg.read_text().replace(
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n', '')
    svg.write_text('\n'.join(line.rstrip() for line in svg_text.splitlines())+'\n')
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
    stage(bg, 3, 5, 1, 'Construct response profiles', INDIGO)
    # Input contexts have their own roles; they are not merged into one biological dataset.
    region(bg, 3, 11, 41, 36, '#F0F1F8', '#9C9FC7')
    label(bg, 7, 16, 'Experimental contexts', 8, weight='bold')
    for y, color, title, desc in [
        (23, INDIGO, 'Blood panel', 'Species × dose × time'),
        (35, OCHRE, 'TG-GATEs', 'Human / rat; culture / liver'),
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
    seq = LinearSegmentedColormap.from_list('expression', ['#F0F1F8', INDIGO])
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
    label(bg, 109.5, 25.9, 'Subject ×\ncondition', 7, ha='center', rotation=90,
          rotation_mode='anchor')
    colour_key(fig, 119, 44, 29, vmin=-1.5, vmax=1.5, title='Response (a.u.)')
    # A fork: species redundancy and profile-level ceilings are distinct constructions.
    bg.plot([146.5, 154, 154, 40], [26, 26, 51, 51], color=LINE, lw=.8)
    arrow(bg, (40, 51), (40, 55), color=INDIGO)
    arrow(bg, (121, 51), (121, 55), color=OCHRE)
    # Dataset-to-branch assignments are stated in the caption; keep the fork
    # clear of the numbered headings immediately below it.
    # Every numbered heading sits 6 mm above its content area. The branch
    # boxes group calculations; they do not enclose a different heading level.
    stage(bg, 3, 57, 2, 'Quantify species redundancy', INDIGO)
    stage(bg, 82, 57, 3, 'Estimate replicate ceiling', OCHRE)
    region(bg, 3, 63, 75, 52, 'white', '#9C9FC7')
    region(bg, 82, 63, 75, 52, 'white', '#D8BB7C')
    x = toy_responses()
    cells(bg, 12, 73, x, 3.3, vmin=-1.5, vmax=1.5)
    for i, name in enumerate('ABCD'):
        label(bg, 10, 74.65+3.3*i, name, 7, ha='right')
    label(bg, 21.9, 68, 'Species means', 7, ha='center')
    arrow(bg, (35, 79.6), (49, 79.6), color=INDIGO)
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
    arrow(bg, (30, 102), (43, 102), color=INDIGO)
    label(bg, 40.5, 111.5, 'Subject splits test pair-rank agreement', 7, ha='center')
    # The ceiling uses replicate response correlations within sides, not rho split halves.
    reps = toy_replicates()
    strips = [(89, 73, 'A1', reps[0]), (89, 95, 'A2', reps[1]),
              (134, 73, 'B1', reps[2]), (134, 95, 'B2', reps[3])]
    for xx, yy, name, values in strips:
        cells(bg, xx, yy, values[None, :], 2.6, vmin=-1.5, vmax=1.5)
        label(bg, xx+7.8, yy-4 if yy == 73 else yy+6, name, 7, ha='center')
    arrow(bg, (96.8, 77.5), (96.8, 92.5), color=INDIGO)
    arrow(bg, (141.8, 77.5), (141.8, 92.5), color=OCHRE)
    label(bg, 94, 85, 'rel A', 7, ha='right')
    label(bg, 144.5, 85, 'rel B', 7)
    for start, end in [(74.3, 74.3), (96.3, 96.3), (74.3, 96.3), (96.3, 74.3)]:
        arrow(bg, (107, start), (131.5, end), color=GREY, lw=.7)
    # Show all four cross-side pairings without masking their crossing.
    label(bg, 119.3, 102, 'Mean = agreement', 7, ha='center')
    label(bg, 119.5, 109, 'Fraction = agreement / √(rel A × rel B)', 7, ha='center')
    # Both constructions constrain interpretation, with different validation requirements.
    arrow(bg, (40, 115.5), (40, 121), color=INDIGO)
    arrow(bg, (120, 115.5), (120, 121), color=OCHRE)
    region(bg, 3, 122, 154, 9, '#EEF7F9', '#8FBECC')
    label(bg, 80, 126.5,
          'Reliability  |  Control independence  |  Context validation', 8, ha='center')
    stage(bg, 3, 138, 4, 'Compare transfer limits', PETROL)
    # Four output objects teach the analysis; all miniatures are explicitly schematic.
    for i, (color, title, pointer) in enumerate([
        (INDIGO, 'Panel design', 'Donor value  ·  Fig. 3'),
        (PETROL, 'Context reuse', 'Time / tissue  ·  Fig. 4'),
        (OCHRE, 'Response specificity', 'Shared / residual  ·  Fig. 5'),
        (GREEN, 'Boundary comparison', 'Fraction of ceiling  ·  Table 2'),
    ]):
        xx = 3 + i*39
        bg.plot([xx+1, xx+37], [144, 144], color=color, lw=1.6)
        label(bg, xx+19, 165, title, 7, weight='bold', ha='center')
        label(bg, xx+19, 170, pointer, 7, ha='center')
    # Equal-weight saturation under an illustrative exchangeable-donor model.
    k = np.arange(1, 18)
    expected = k*.3 / np.sqrt(k+k*(k-1)*.3)
    bg.plot([9, 9, 36], [147, 160, 160], color=GREY, lw=.6)
    bg.plot(9+27*(k-1)/16, 160-13*expected/.6, color=INDIGO, lw=1.2)
    # Context reuse: match the same upper triangles, rather than implying a learned mapping.
    for xx, mat in [(44, p), (65, q)]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 149, upper, 2.5)
    arrow(bg, (55, 154), (62, 154), color=PETROL)
    # Specificity is a subtraction within species, using exactly the illustrated decomposition.
    response, shared, residual = decomposition()
    cells(bg, 83, 149, response, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    operator(bg, 93, 153.75, '-')
    cells(bg, 95.5, 149, shared, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    operator(bg, 105.5, 153.75, '=')
    cells(bg, 108, 149, residual, 1.9, vmin=-2, vmax=2, edge='#DCE3EC', linewidth=.3)
    # A ceiling is an upper reference, with observed agreement expressed relative to it.
    bg.plot([125, 152], [159, 159], color=GREY, lw=.7)
    bg.plot([125, 145], [153, 153], color=GREEN, lw=1.5)
    bg.plot([152, 152], [149, 160], color=INK, lw=.65, ls=(0, (2, 2)))
    label(bg, 149, 147, 'Ceiling', 7, ha='center')
    save(fig, out, 1)


def figure2(out):
    """Dataset scope and the actual input coverage of the two core analyses."""
    data = collect_dataset_overview()
    (out/'Fig2_source_data.json').write_text(json.dumps(data, indent=2)+'\n')
    fig, bg = canvas(154)
    panel(bg, PANEL_LEFT, 5, 'a')
    label(bg, 7, 12, 'Dataset', 8, weight='bold')
    label(bg, 56, 12, 'Species (n)', 8, ha='center', weight='bold')
    label(bg, 99, 12, 'Biological system', 8, ha='center', weight='bold')
    label(bg, 141, 12, 'Study role', 8, ha='center', weight='bold')
    bg.plot([5, 157], [16, 16], color=INK, lw=.65)
    role_colours = {'Core analyses': INDIGO, 'Additional panels': GREEN, 'External check': PETROL}
    role_fills = {'Core analyses': '#EBECF6', 'Additional panels': '#E7F3E9',
                  'External check': '#E7F3F7'}
    for role, fill in role_fills.items():
        indices = [i for i, r in enumerate(data['datasets']) if r['role'] == role]
        top = 17+7*min(indices)
        bg.add_patch(Rectangle((5, top), 152, 7*len(indices), fc=fill, ec='white', lw=1.5))
        label(bg, 141, top+3.5*len(indices), role, 7, ha='center')
    a = axes(fig, 39, 17, 34, 49)
    a.set_facecolor('none')
    for i, row in enumerate(data['datasets']):
        yy = 20.5+7*i
        color = role_colours[row['role']]
        label(bg, 7, yy, row['name'], 8)
        label(bg, 99, yy, row['system'], 7, ha='center')
        a.barh(i, row['n_species'], height=.45, color=color, linewidth=0)
        a.text(row['n_species']+.35, i, str(row['n_species']), va='center', fontsize=8)
    a.set(xlim=(0, 12), ylim=(6.5, -.5))
    a.set_xticks([0, 5, 10]); a.set_yticks([])
    a.spines['left'].set_visible(False)
    # Stop the visible axis at the largest count; retain room for direct labels.
    a.spines['bottom'].set_bounds(0, 10)

    # More area for the two coverage plots, with shared top and bottom baselines.
    lower = 78
    panel(bg, PANEL_LEFT, lower, 'b')
    panel(bg, PANEL_RIGHT, lower, 'c')
    blood = data['blood']
    b = axes(fig, 23, 87, 50, 53)
    values = np.asarray(blood['counts'])
    # Hue identifies time, not count magnitude. The counts remain explicit text.
    seq = ListedColormap([BLOOD_TIME_COLOURS[t] for t in blood['times_h']])
    time_ids = np.broadcast_to(np.arange(len(blood['times_h'])), values.shape)
    b.pcolormesh(time_ids, cmap=seq, vmin=-.5, vmax=len(blood['times_h'])-.5,
                 edgecolors='white', linewidth=1.5)
    b.set(xlim=(0, len(blood['times_h'])), ylim=(len(blood['doses_ng_ml']), 0),
          xlabel='Time (h)', ylabel='LPS dose (ng/ml)')
    b.set_xticks(np.arange(len(blood['times_h']))+.5, [f'{v:g}' for v in blood['times_h']])
    b.set_yticks(np.arange(len(blood['doses_ng_ml']))+.5, [f'{v:,.0f}' for v in blood['doses_ng_ml']])
    b.tick_params(length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    for i, j in np.ndindex(values.shape):
        b.text(j+.5, i+.5, str(values[i,j]), fontsize=8, ha='center', va='center')

    c = axes(fig, 106, 87, 49, 53)
    rows = data['tggates']['comparisons']
    y = np.arange(len(rows))[::-1]
    c.barh(y, [r['n_shared'] for r in rows], color='#D1D9E3', height=.58, label='Shared')
    c.barh(y, [r['n_in_band'] for r in rows], color=OCHRE, height=.58, label='In band')
    for yy, row in zip(y, rows):
        c.text(row['n_shared']+4, yy, str(row['n_shared']), fontsize=7, va='center')
        c.text(row['n_in_band']/2, yy, str(row['n_in_band']), fontsize=7, va='center', ha='center')
    c.set(xlim=(0, 180), ylim=(-.6, len(rows)-.4), xlabel='Compounds (n)')
    # Wrapping the four time comparisons reserves width for the bars themselves.
    c.set_yticks(y, [r['label'].replace(', ', '\n') for r in rows], fontsize=7)
    c.set_xticks([0, 50, 100, 150]); c.tick_params(axis='y', length=0, pad=3)
    c.spines['left'].set_visible(False)
    fig.legend(*c.get_legend_handles_labels(), loc='center', ncol=2, frameon=False,
               bbox_to_anchor=(130/160, 1-80/154), fontsize=7, handlelength=1.1,
               columnspacing=1.2, handletextpad=.4)
    save(fig, out, 2)


def figure3(out):
    """Empirical donor ordering and panel choices, then explicitly marked extrapolation."""
    core = load('core_numbers'); c = core['collectible six']
    fig, bg = canvas(149)
    # Both rows have 48 mm plot areas; b/c have identical physical scales.
    second_row = 68 + ROW_GAP
    panel(bg, PANEL_LEFT, 5, 'a')
    panel(bg, PANEL_RIGHT, 5, 'b')
    panel(bg, PANEL_LEFT, second_row, 'c')
    panel(bg, PANEL_RIGHT, second_row, 'd')
    audit, ci = load('reconstruct_metric_audit'), load('fig1_ci')
    rows = sorted(((k, v) for k, v in audit.items() if k != 'uniform (all 9)'),
                  key=lambda kv: -kv[1]['r'])
    a = axes(fig, 25, 12, 42, 48)
    y = np.arange(len(rows))[::-1]
    for yy, (name, v) in zip(y, rows):
        err = 1.96*ci[name]['se']
        a.errorbar(v['r'], yy, xerr=err, fmt='o', color=INDIGO,
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
                        color=INDIGO, alpha=.12, linewidth=0)
        configs = [('best', OCHRE, 'o', '-', 'Best'),
                   ('mean', INDIGO, 's', '-', 'Mean'),
                   ('worst', GREY, '^', '--', 'Worst')]
        for field, col, mark, ls, name in configs:
            ax.plot(kk, [r[field] for r in d], color=col, marker=mark,
                    ls=ls, ms=3.3, lw=1.1, markeredgecolor='white',
                    markeredgewidth=.3, label=name)
        ax.set(xlim=(.75, max(kk)+.25), ylim=(.17, .72),
               xlabel='Number of donor species', ylabel='Correlation with human')
        ax.set_xticks(kk); ax.set_yticks([.2, .4, .6])
        return ax

    b = panel_spread(axes(fig, 96, 12, 59, 48), 'spread_collectible six')
    panel_spread(axes(fig, 17, 87, 59, 48), 'spread_all nine')
    # Same encodings in both subset panels. The envelope is the full enumeration, not a CI.
    fig.legend(*b.get_legend_handles_labels(), loc='center', ncol=3, frameon=False,
               bbox_to_anchor=(125/160, 1-5/149), columnspacing=.6,
               handletextpad=.3, handlelength=1.1, fontsize=7)

    d = axes(fig, 96, 87, 59, 48)
    k = np.geomspace(1, 400, 600)
    expected = k*c['rbar']/np.sqrt(k+k*(k-1)*c['rho'])
    d.axvspan(6, 400, color='#F2F4F7', zorder=0)
    d.axvline(6, color=GREY, lw=.65, ls=(0, (2, 2)))
    inside = k <= 6
    d.plot(k[inside], expected[inside], color=INDIGO, lw=1.3)
    # Include the exact endpoint so solid and dashed curves meet without a gap.
    ext = np.geomspace(6, 400, 400)
    d.plot(ext, ext*c['rbar']/np.sqrt(ext+ext*(ext-1)*c['rho']),
           color=INDIGO, lw=1.2, ls=(0, (3, 2)))
    d.plot([r['k'] for r in core['spread_collectible six']],
           [r['mean'] for r in core['spread_collectible six']],
           's', color=INDIGO, ms=3.4, mec='white', mew=.35)
    # The asymptote is only 0.0025 above macaque: a second horizontal line
    # cannot be distinguished at print size. Report it in the caption instead.
    d.axhline(core['macaque'], color=PETROL, lw=.7, ls=(0, (1, 2)))
    d.axhline(core['chimp'], color=GREY, lw=.7, ls=(0, (1, 2)))
    d.plot(core['k_to_match_macaque'], core['macaque'], 'D', color=PETROL, ms=4,
           mec='white', mew=.3)
    fig.legend(handles=[
        Line2D([], [], color=GREY, lw=.7, ls=':', label='Chimpanzee'),
        Line2D([], [], color=INDIGO, lw=1.2, ls='--', label='Pooling model'),
        Line2D([], [], color=PETROL, lw=.7, ls=':', label='Macaque'),
    ], loc='center', bbox_to_anchor=(123/160, 1-second_row/149),
       ncol=3, frameon=False, fontsize=7, handlelength=1.1,
       columnspacing=.6, handletextpad=.3)

    d.set(xscale='log', xlim=(.9, 430), ylim=(.18, .74),
          xlabel='Number of species (log scale)', ylabel='Correlation with human')
    d.set_xticks([1, 6, 20, 100, 400], ['1', '6', '20', '100', '400'])
    d.set_yticks([.2, .4, .6]); d.tick_params(axis='x', which='minor', bottom=False)
    save(fig, out, 3)


def coefficient_heatmap(fig, x, top, rows):
    fields = [('spearman_all', 'p_all'), ('spearman_cross', 'p_cross'),
              ('partial_taxdist', 'p_partial')]
    ax = axes(fig, x, top, 50, 40)
    values = np.array([[r[field] for field, _ in fields] for r in rows])
    mesh = ax.pcolormesh(values, cmap=RESPONSE, vmin=-1, vmax=1,
                        edgecolors='white', linewidth=1.5, shading='flat')
    ax.set(xlim=(0, 3), ylim=(3, 0))
    ax.set_xticks(np.arange(3)+.5, ['All\npairs', 'Cross-\norder', 'Taxonomy\nadjusted'], fontsize=7)
    time_groups = [tuple(int(t) for t in re.findall(r'(\d+)h', row['label']))
                   for row in rows]
    assert all(group and all(t in BLOOD_TIME_COLOURS for t in group)
               and len(group) <= 2 for group in time_groups), time_groups
    names = [' vs '.join(map(str, group))+' h' for group in time_groups]
    ax.set_yticks(np.arange(3)+.5, names, fontsize=7)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.tick_params(axis='y', length=0, pad=3)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i, row in enumerate(rows):
        for j, (field, pfield) in enumerate(fields):
            val = row[field]; star = '*' if row[pfield] < .05 else ''
            # Dark text on this bright, moderately saturated diverging palette is legible.
            ax.text(j+.5, i+.5, f'{val:.3f}{star}', ha='center', va='center', fontsize=8)
    return mesh


def figure4(out):
    fig, bg = canvas(175)
    middle_row, last_row = 58, 127
    panel(bg, PANEL_LEFT, 5, 'a')
    p, q = toy_contexts()
    idx = np.triu_indices(4, 1)
    for mat, xx, name in [(p, 9, 'Context 1'), (q, 38, 'Context 2')]:
        upper = mat.copy(); upper[np.tril_indices(4)] = np.nan
        cells(bg, xx, 18, upper, 4.7)
        label(bg, xx+9.4, 9, name, 7, ha='center')
        for j, letter in enumerate('ABCD'):
            label(bg, xx+2.35+j*4.7, 14.5, letter, 7, ha='center', weight='bold')
            label(bg, xx-1, 20.35+j*4.7, letter, 7, ha='right', weight='bold')
        bg.add_patch(Rectangle((xx+2*4.7, 18), 4.7, 4.7, fill=False, ec=PETROL, lw=1.1))
    arrow(bg, (60, 26), (67, 26))
    for yy, mat in [(20, p), (30, q)]:
        cells(bg, 70, yy, mat[idx][None, :], 4.2)
    for j, name in enumerate(['AB', 'AC', 'AD', 'BC', 'BD', 'CD']):
        label(bg, 72.1+j*4.2, 16, name, 7, ha='center', weight='bold')
    arrow(bg, (98, 26), (105, 26))
    sc = axes(fig, 121, 11, 33, 23)
    # Use the same average-tie convention as the empirical analysis for this teaching example.
    ranks = [centered_ranks(m[idx])+(len(idx[0])+1)/2 for m in (p, q)]
    sc.plot([.5, 6.5], [.5, 6.5], color=LINE, lw=.75)
    sc.scatter(*ranks, s=15, color=INDIGO, edgecolors='white', linewidths=.35, zorder=3)
    j = 1  # Pair A-C, also outlined in both matrices.
    sc.scatter(ranks[0][j], ranks[1][j], s=26, facecolors='none', edgecolors=PETROL, lw=.9)
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
    coefficient_heatmap(fig, 22, 66, rel)
    coefficient_heatmap(fig, 105, 66, times)
    colour_key(fig, 24, 42, 27, vmin=-1, vmax=1, title='Correlation')
    panel(bg, PANEL_LEFT, last_row, 'd')
    d = axes(fig, 43, 136, 98, 25)
    cfg = [('spearman_all', 'p_all', GREY, 'o', 'All pairs'),
           ('spearman_cross', 'p_cross', INDIGO, 's', 'Cross-order'),
           ('partial_taxdist', 'p_partial', PETROL, 'D', 'Taxonomy adjusted')]
    for yy, row in zip([1, 0], tissues):
        for j, (field, pfield, color, marker, _) in enumerate(cfg):
            v = row[field]; pos = yy + (1-j)*.23
            d.plot(v, pos, marker=marker, ms=3.4, color=color, mec='white', mew=.3)
            offset = .027 if v >= 0 else -.027
            d.text(v+offset, pos, f'{v:+.3f}'+('*' if row[pfield] < .05 else ''),
                   fontsize=7, va='center', ha='left' if v >= 0 else 'right')
    d.set(xlim=(-.45, .80), ylim=(-.5, 1.5), xlabel='Matrix agreement (Spearman)')
    d.axvline(0, color=GREY, lw=.65)
    d.set_yticks([1, 0], ['Pool profiles,\nthen correlate', 'Correlate by condition,\nthen average'], fontsize=7)
    d.set_xticks([-.4, 0, .4, .8]); d.tick_params(axis='y', length=0)
    d.spines['left'].set_visible(False)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls='', ms=3.4, label=l)
                        for _, _, c, m, l in cfg], loc='center',
               bbox_to_anchor=(80/160, 1-last_row/175), ncol=3, frameon=False,
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
                              fc=INDIGO if i == j else '#D1D9E3', ec='white', lw=.8))
    for j in range(4):
        label(bg, 132+j*6, 17.2, str(j+1), 7, ha='center', weight='bold')
        label(bg, 127.5, 23.5+j*6, str(j+1), 7, ha='right', weight='bold')
    label(bg, 141, 11, 'Human condition', 8, ha='center', weight='bold')
    label(bg, 120, 32.5, 'Donor condition', 8, ha='center', rotation=90,
          rotation_mode='anchor', weight='bold')
    for yy, col, name in [(49, INDIGO, 'Matched'), (54, '#D1D9E3', 'Mismatched')]:
        bg.add_patch(Rectangle((129, yy-1.1), 2.2, 2.2, fc=col, ec='none'))
        label(bg, 133, yy, name, 7)
    panel(bg, PANEL_LEFT, second_row, 'c')
    panel(bg, PANEL_RIGHT, second_row, 'd')
    data = load('reconstruct_specificity')['blocks']; raw, res = data['raw'], data['residual']
    order = sorted(raw, key=lambda k: -raw[k]['matched']); yy = np.arange(len(order))[::-1]
    c = axes(fig, 26, 80, 50, 52)
    d = axes(fig, 102, 80, 53, 52)
    # Matching light row bands help track the same donor across different targets.
    for ax in (c, d):
        for row in yy[::2]:
            ax.axhspan(row-.5, row+.5, color='#F4F7FA', lw=0, zorder=0)
    total = np.array([raw[k]['matched'] for k in order])
    mis = np.array([raw[k]['mismatched_median'] for k in order])
    c.hlines(yy, mis, total, color='#BAC6D1', lw=1)
    c.plot(mis, yy, 's', color=GREY, ms=3.2, label='Mismatched')
    c.plot(total, yy, 'o', color=INDIGO, ms=3.7, label='Matched', mec='white', mew=.3)
    names = ['All nine (mean)' if k == 'uniform (all 9)' else NAMES[k.removeprefix('only ')]
             for k in order]
    c.set_yticks(yy, names, fontsize=7)
    c.set(xlim=(-.12, .54), ylim=(-.6, 9.6), xlabel='Raw recovery')
    c.set_xticks([-.1, 0, .2, .4])
    vals = [res[k]['residual_recovery'] for k in order]
    for y, v in zip(yy, vals):
        col = OCHRE if v > 0 else GREY
        d.hlines(y, 0, v, color=col, lw=1.2)
        d.plot(v, y, 'o' if v > 0 else 's', color=col, ms=3.6, mec='white', mew=.3)
        d.text(v+(.017 if v > 0 else -.017), y, f'{v:+.3f}', ha='left' if v > 0 else 'right',
               va='center', fontsize=7)
    # Reserve room for complete signed values inside the row bands.
    d.set(xlim=(-.55, .33), ylim=(-.6, 9.6), xlabel='Residual recovery')
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
    global ALIGNMENT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'docs/paper/tex/figures')
    parser.add_argument('--audit-dir', type=Path,
                        help='optional nature-figure alignment reports (helper on PYTHONPATH)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.audit_dir is not None:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
        ALIGNMENT_DIR = args.audit_dir
    style()
    for draw in (figure1, figure2, figure3, figure4, figure5): draw(args.out)


if __name__ == '__main__':
    main()
