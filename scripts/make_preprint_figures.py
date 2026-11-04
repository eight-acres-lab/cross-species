#!/usr/bin/env python3
"""Draw four explanatory manuscript composites at their final, 160 mm width.

Measured results come from deposited JSON estimates, without rerunning analyses.
Small teaching matrices are explicitly labelled illustrative, computed consistently,
and never substituted for measurements. All heatmaps and diagrams are vector graphics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np

from plot_rho_phylogeny import collect

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/biology/concordance"
WIDTH = 160 / 25.4
BLUE, RED, PURPLE, GREY = "#1f6f8b", "#c0392b", "#7d3c98", "#858d96"
INK, PALE, GOLD = "#233344", "#f3f6f8", "#c69849"
RESPONSE = LinearSegmentedColormap.from_list("response", [BLUE, "#ffffff", RED])
NAMES = {
    "Pan troglodytes": "Chimpanzee", "Macaca": "Macaque",
    "Papio anubis": "Baboon", "Bos taurus": "Cow", "Ovis aries": "Sheep",
    "Sus scrofa": "Pig", "Mus musculus": "Mouse", "Rattus norvegicus": "Rat",
    "Oryctolagus cuniculus": "Rabbit", "Homo sapiens": "Human",
}



def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "axes.axisbelow": True,
    })


def save(fig, out, n):
    # Fixed page boxes keep the documented font sizes true at the manuscript width.
    fig.savefig(out / f"Fig{n}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"Fig{n}.png", dpi=300)
    plt.close(fig)
    print(f"Fig{n}: vector PDF and 300 dpi PNG")


def panel(ax, letter, title=None):
    ax.set_title(f"({letter})" + (f"  {title}" if title else ""),
                 loc="left", fontweight="bold", pad=9)


def axes(fig, x, y, w, h, *, diagram=False):
    """Place panels in physical inches, so print typography remains predictable."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x/fw, y/fh, w/fw, h/fh])
    if diagram:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    return ax


def box(ax, xy, width, height, text=None, *, color=PALE, edge="#d6dfe5", fontsize=9):
    p = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.025",
                       linewidth=0.7, edgecolor=edge, facecolor=color,
                       transform=ax.transAxes)
    ax.add_patch(p)
    if text:
        ax.text(xy[0]+width/2, xy[1]+height/2, text, ha="center", va="center",
                fontsize=fontsize, linespacing=1.4, transform=ax.transAxes)
    return p


def arrow(ax, start, end, *, color=INK, style="-", lw=1.1):
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "lw": lw, "linestyle": style, "mutation_scale": 10})


def matrix(ax, values, *, rowlabels=None, collabels=None, vmin=-1, vmax=1,
           cmap=RESPONSE, numbers=False, fontsize=8.5):
    """Use vector cells rather than an embedded raster image."""
    values = np.asarray(values)
    nr, nc = values.shape
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="white", linewidth=0.65, shading="flat")
    ax.set(xlim=(0, nc), ylim=(nr, 0), aspect="equal")
    ax.set_xticks(np.arange(nc)+0.5, collabels or []) if collabels else ax.set_xticks([])
    ax.set_yticks(np.arange(nr)+0.5, rowlabels or []) if rowlabels else ax.set_yticks([])
    ax.tick_params(length=0, pad=3, labelsize=fontsize)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if numbers:
        for i, j in np.ndindex(values.shape):
            v = values[i, j]
            if np.isfinite(v):
                # Choose ink from the actual cell colour, not the correlation sign.
                rgb = np.asarray(mesh.cmap(mesh.norm(v))[:3])
                linear = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
                luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
                black_contrast = (luminance+0.05)/0.05
                white_contrast = 1.05/(luminance+0.05)
                ink = 'black' if black_contrast >= white_contrast else 'white'
                label = f"{v:.2f}"
                ax.text(j+0.5, i+0.5, label, ha="center", va="center",
                        color=ink,
                        fontsize=fontsize)
    return mesh


def toy_responses():
    # Four imaginary species, six genes: every displayed rho is computed from these rows.
    return np.array([[1.4, 0.8, -1.1, -0.6, 0.2, -0.5],
                     [0.9, 1.1, -0.7, -1.2, 0.4, -0.2],
                     [0.4, -0.8, -0.3, 1.2, 0.7, -0.4],
                     [1.0, 0.2, -0.9, 0.3, -0.8, 0.4]])


def figure1(out):
    fig = plt.figure(figsize=(WIDTH, 7.15))
    a = axes(fig, 0.06, 5.12, 6.16, 1.75, diagram=True)
    panel(a, 'a', 'From perturbation responses to a species-by-species matrix')
    box(a, (0.005, 0.20), 0.23, 0.70, color='#edf4f6')
    a.text(0.12, 0.82, 'Blood + LPS', weight='bold', ha='center', fontsize=9)
    for row, color in enumerate([PURPLE, BLUE, GOLD]):
        a.scatter([0.065, 0.105, 0.145, 0.185], [0.67-row*0.09]*4, s=18,
                  color=color, edgecolors='white', linewidths=0.4)
    a.text(0.12, 0.34, '10 species\n3 doses x 3 timepoints', ha='center', va='center', fontsize=8.5)
    a.text(0.445, 0.92, 'Mean response per species', ha='center', weight='bold', fontsize=9)
    m = axes(fig, 2.10, 5.62, 1.45, 0.94)
    matrix(m, toy_responses(), rowlabels=['A', 'B', 'C', 'D'], vmin=-1.5, vmax=1.5)
    a.text(0.445, 0.20, 'Aligned genes', ha='center', fontsize=8.5)
    a.text(0.445, 0.06, 'Illustrative response values', ha='center', color=GREY, fontsize=8.5)
    arrow(a, (0.25, 0.54), (0.31, 0.54))
    a.text(0.275, 0.73, 'Subject\nmeans', ha='center', fontsize=8.5)
    arrow(a, (0.59, 0.54), (0.66, 0.54))
    a.text(0.63, 0.28, 'Pearson\nacross genes', ha='center', fontsize=8.5)
    a.text(0.82, 0.92, r'Redundancy matrix $\rho$', ha='center', weight='bold', fontsize=9)
    m = axes(fig, 4.47, 5.45, 1.31, 1.08)
    matrix(m, np.corrcoef(toy_responses()), rowlabels=['A', 'B', 'C', 'D'],
           collabels=['A', 'B', 'C', 'D'], vmin=-1, vmax=1)
    a.text(0.82, 0.06, 'Illustrative pair correlations', ha='center', color=GREY, fontsize=8.5)
    a.text(0.02, -0.03, 'Repeat by condition; average correlations across the nine conditions.', fontsize=9)
    # The two reliability constructions operate on different objects. Keep them separate.
    b = axes(fig, 0.06, 3.36, 2.98, 1.35, diagram=True)
    panel(b, 'b', 'Is the matrix reproducible?')
    b.text(0.5, 0.97, 'Within one blood timepoint', ha='center', fontsize=8.5)
    box(b, (0.01, 0.51), 0.37, 0.29, 'Subject half 1\n'+r'$\rho^{(1)}$', fontsize=8.5)
    box(b, (0.61, 0.51), 0.37, 0.29, 'Subject half 2\n'+r'$\rho^{(2)}$', fontsize=8.5)
    arrow(b, (0.20, 0.49), (0.41, 0.39)); arrow(b, (0.80, 0.49), (0.59, 0.39))
    b.text(0.5, 0.24, 'Rank the same 45 species pairs', ha='center', fontsize=8.5)
    rel = load('rho_cross_panel_both_estimators')
    b.text(0.5, 0.06, f"Mean split-half reliability: {rel['rho_reliability_halfset']:.3f}",
           ha='center', color=BLUE, weight='bold')
    b.text(0.5, -0.12, f"Full-subject extrapolation: {rel['rho_reliability_fullset']:.3f}",
           ha='center', fontsize=8.5)
    c = axes(fig, 3.32, 3.36, 2.90, 1.35, diagram=True)
    panel(c, 'c', 'How much agreement is retained?')
    c.text(0.5, 0.97, 'Across a time, species or system boundary', ha='center', fontsize=8.5)
    box(c, (0.01, 0.51), 0.37, 0.29, 'Side A replicates\n'+r'$\mathrm{rel}_A$', fontsize=8.5)
    box(c, (0.61, 0.51), 0.37, 0.29, 'Side B replicates\n'+r'$\mathrm{rel}_B$', fontsize=8.5)
    arrow(c, (0.20, 0.49), (0.41, 0.31)); arrow(c, (0.80, 0.49), (0.59, 0.31))
    c.text(0.5, 0.18, r'Fraction of ceiling = $\dfrac{\mathrm{observed}}{\sqrt{\mathrm{rel}_A\mathrm{rel}_B}}$',
           ha='center', va='center', fontsize=10)
    c.text(0.5, -0.12, 'Use control-disjoint contrasts (Table 2)', ha='center', fontsize=8.5)
    audit, ci = load('reconstruct_metric_audit'), load('fig1_ci')
    rows = sorted(((k, v) for k, v in audit.items() if k != 'uniform (all 9)'),
                  key=lambda kv: -kv[1]['r'])
    ax = axes(fig, 1.10, 0.62, 5.05, 2.11)
    names = [k for k, _ in rows]; vals = [v['r'] for _, v in rows]
    errors = [1.96*ci[n]['se'] for n in names]; y = np.arange(len(names))[::-1]
    cols = [PURPLE if n in ('Pan troglodytes', 'Macaca', 'Papio anubis') else BLUE for n in names]
    ax.barh(y, vals, color=cols, height=0.64, xerr=errors,
            error_kw={'ecolor': '#333333', 'elinewidth': 0.8, 'capsize': 2.2})
    for yy, val, err in zip(y, vals, errors):
        ax.text(val+err+0.012, yy, f'{val:.3f}', va='center', fontsize=8.5)
    ax.set_yticks(y, [NAMES[n] for n in names], fontsize=8.5)
    ax.tick_params(axis='y', length=0)
    ax.set(xlim=(0, 0.79), xlabel='Correlation with the held-out human response')
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    panel(ax, 'd', 'Measured agreement for each donor species')
    fig.text(0.5, 0.017, 'Readouts: panel size (Fig. 2)  |  context transfer (Fig. 3)  |  specificity (Fig. 4)',
             ha='center', fontsize=8.5, color=BLUE)
    save(fig, out, 1)


def figure2(out):
    core = load('core_numbers'); c = core['collectible six']
    fig = plt.figure(figsize=(WIDTH, 7.20))
    a = axes(fig, 0.06, 5.25, 6.16, 1.62, diagram=True)
    panel(a, 'a', 'Why more species give diminishing returns')
    a.text(0.23, 0.93, 'Equal-weight pooling', ha='center', weight='bold', fontsize=9)
    for j, color in enumerate([BLUE, PURPLE, GOLD]):
        y = 0.69-j*0.19
        box(a, (0.005, y-0.065), 0.13, 0.14, f'Donor {j+1}', color='#edf4f6', fontsize=8.5)
        arrow(a, (0.15, y), (0.205, 0.51), color=color)
    box(a, (0.21, 0.39), 0.11, 0.25, 'Mean\nprofile', fontsize=8.5)
    arrow(a, (0.33, 0.51), (0.385, 0.51))
    box(a, (0.39, 0.39), 0.10, 0.25, 'Human\ntarget', color='#f1ecf6', fontsize=8.5)
    a.text(0.23, 0.04, r'Mean donor-human $\bar{r}$: '+f"{c['rbar']:.3f}", ha='center', fontsize=8.5)
    a.text(0.77, 0.93, 'Correlated donors set the limit', ha='center', weight='bold')
    m = axes(fig, 3.48, 5.81, 0.78, 0.78)
    m.set(xlim=(0, 3), ylim=(3, 0), aspect='equal'); m.axis('off')
    for i, j in np.ndindex(3, 3):
        m.add_patch(Rectangle((j, i), 1, 1, fc='#eddcbb' if i == j else '#c9e0e8', ec='white', lw=1))
        m.text(j+0.5, i+0.5, '1' if i == j else r'$\bar{\rho}$', ha='center', va='center', fontsize=10)
    a.text(0.77, 0.04, r'Mean donor-donor $\bar{\rho}$: '+f"{c['rho']:.3f}", ha='center', fontsize=8.5)
    a.text(0.86, 0.55, r'$R(k)=\dfrac{k\bar{r}}{\sqrt{k+k(k-1)\bar{\rho}}}$', ha='center', fontsize=10.5)
    a.text(0.86, 0.26, 'Expected average', ha='center', fontsize=8.5)
    a.text(0.5, -0.12, 'Illustrative 3-donor matrix; unit variances and equal pair correlations.', ha='center', fontsize=8.5, color=GREY)
    ax = axes(fig, 0.64, 2.88, 5.49, 1.72)
    k = np.arange(1, 420)
    ax.plot(k, k*c['rbar']/np.sqrt(k+k*(k-1)*c['rho']), color=BLUE, lw=1.5, label='Predicted mean')
    ks = [r['k'] for r in c['rows']]
    ax.plot(ks, [r['mean'] for r in core['spread_collectible six']], 's--', color=GREY,
            ms=4, lw=1.1, label='Measured mean')
    ax.plot(ks, [r['measured_best'] for r in c['rows']], 'o-', color=RED, ms=4, lw=1.2, label='Measured best')
    ax.axhline(c['asymptote'], color=BLUE, ls='--', lw=1, label='Pool asymptote')
    ax.axhline(core['macaque'], color=PURPLE, ls=':', lw=1.2, label='One macaque')
    ax.axhline(core['chimp'], color=GREY, ls=':', lw=1.2, label='One chimpanzee')
    ax.plot(core['k_to_match_macaque'], core['macaque'], '*', color=PURPLE, ms=11)
    ax.set(xscale='log', xlim=(0.9, 420), ylim=(0.18, 0.74),
           xlabel='Number of species in the panel', ylabel='Correlation with human')
    ax.set_xticks([1, 2, 6, 20, 100, 400]); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='x', which='minor', bottom=False)
    panel(ax, 'b', 'Measured panels and the analytic expectation')
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.54, 2.22/7.20),
               ncol=3, frameon=False, columnspacing=1.2, handlelength=2.0)
    axs = [axes(fig, 0.64, 0.70, 2.49, 1.07), axes(fig, 3.64, 0.70, 2.49, 1.07)]
    for a, key, title, letter in zip(axs, ('spread_collectible six', 'spread_all nine'),
                                    ('Six collectible species', 'All nine donors'), 'cd'):
        rows = core[key]; kk = [r['k'] for r in rows]
        a.fill_between(kk, [r['worst'] for r in rows], [r['best'] for r in rows], color=BLUE, alpha=0.12)
        for field, color, marker, label in [('best', RED, 'o-', 'Best panel'),
                ('mean', GREY, 's--', 'Average panel'), ('worst', PURPLE, '^:', 'Worst panel')]:
            a.plot(kk, [r[field] for r in rows], marker, color=color, ms=3.5, lw=1.1, label=label)
        a.set(xlabel='Panel size $k$', ylim=(0.18, 0.72)); a.set_xticks(kk)
        panel(a, letter, title)
    axs[0].set_ylabel('Correlation with human'); axs[1].tick_params(labelleft=False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3,
               bbox_to_anchor=(0.54, 0.0), frameon=False)
    save(fig, out, 2)


def figure3(out):
    fig = plt.figure(figsize=(WIDTH, 6.65))
    a = axes(fig, 0.06, 4.62, 6.16, 1.68, diagram=True)
    panel(a, 'a', 'How to compare two redundancy matrices')
    # Teaching matrices only. The earlier rho_cross_context pair-level run was
    # superseded by the two-estimator analysis; do not mix its cells with that run.
    x = toy_responses()
    z = x + np.array([[0.1, 0.4, -0.1, 0.2, -0.1, 0.3],
                      [0.0, -0.2, 0.5, -0.1, 0.3, -0.4],
                      [-0.2, 0.1, 0.6, -0.3, 0.4, -0.1],
                      [0.3, -0.2, 0.1, 0.4, -0.1, -0.3]])
    for j, data in enumerate([x, z]):
        values = np.corrcoef(data)
        values[np.tril_indices(4)] = np.nan
        m = axes(fig, 0.36+j*1.82, 4.70, 1.36, 1.28)
        matrix(m, values, rowlabels=list('ABCD'), collabels=list('ABCD'),
               vmin=-1, vmax=1, numbers=True, fontsize=8.5)
        m.add_patch(Rectangle((2, 0), 1, 1, fill=False, ec=GOLD, lw=1.5))
        a.text(0.16+j*0.296, 0.95, ['Context 1', 'Context 2'][j],
               ha='center', fontsize=9, weight='bold')
    arrow(a, (0.27, 0.50), (0.33, 0.50))
    a.text(0.78, 0.88, 'One cell = one species pair', ha='center', weight='bold', fontsize=9)
    a.text(0.78, 0.50, 'Match cells by species pair\nRank their correlation values\nCompare ranks with Spearman',
           ha='center', va='center', linespacing=1.7, fontsize=9)
    a.text(0.78, 0.12, 'Gold outline: pair A-C in both contexts', ha='center', fontsize=8.5)
    a.text(0.5, -0.06, 'Illustrative matrices; diagonal and duplicate cells omitted.',
           ha='center', fontsize=8.5, color=GREY)
    rows = collect(load('rho_cross_panel_both_estimators'))
    reliability = [r for _, r, isrel in rows if isrel]
    times = [r for _, r, _ in rows if r.get('kind') == 'cross_time']
    tissues = [r for _, r, _ in rows if r.get('kind', '').startswith('cross_panel_macrophage')]
    assert len(reliability) == 3 and len(times) == 3 and len(tissues) == 2
    configs = [('spearman_all', 'p_all', GREY, 'o', 'All pairs'),
               ('spearman_cross', 'p_cross', BLUE, 's', 'Cross-order only'),
               ('partial_taxdist', 'p_partial', PURPLE, 'D', 'Partial on taxonomy')]

    def compare_rows(ax, data, labels, *, numbers=False):
        y = np.arange(len(data))[::-1]
        for yy, row in zip(y, data):
            for j, (field, pfield, color, marker, _) in enumerate(configs):
                v, pval = row.get(field), row.get(pfield)
                if v is None or not np.isfinite(v):
                    continue
                off = (1-j)*0.24
                ax.plot(v, yy+off, marker, color=color, ms=4)
                star = '*' if pval is not None and np.isfinite(pval) and pval < 0.05 else ''
                if numbers or star:
                    label = (f'{v:+.3f}' if numbers else '') + star
                    ax.text(v+0.035, yy+off, label, fontsize=8.5, va='center', color=color)
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.tick_params(axis='y', length=0)
        ax.set(ylim=(-0.6, len(data)-0.4), xlabel='Matrix agreement (Spearman)')
        ax.axvline(0, color='#9aa0a6', lw=0.7)
        for yy in y:
            ax.axhline(yy, color='#e8edf0', lw=0.6, zorder=0)

    b = axes(fig, 0.56, 2.52, 2.27, 1.43)
    compare_rows(b, reliability, ['2 h', '6 h', '24 h'])
    b.set(xlim=(-0.05, 1.13)); b.set_xticks([0, 0.5, 1])
    panel(b, 'b', 'Within-timepoint reliability')
    c = axes(fig, 3.92, 2.52, 2.20, 1.43)
    compare_rows(c, times, ['2 h vs 6 h', '2 h vs 24 h', '6 h vs 24 h'])
    c.set(xlim=(-0.05, 1.13)); c.set_xticks([0, 0.5, 1])
    panel(c, 'c', 'Across blood timepoints')
    fig.text(0.54, 2.00/6.65, 'Blood: 45 species pairs; 35 cross-order pairs', ha='center', fontsize=8.5)
    d = axes(fig, 1.60, 0.68, 4.52, 0.90)
    compare_rows(d, tissues, ['Pool profiles,\nthen correlate', 'Correlate by condition,\nthen average'], numbers=True)
    d.set(xlim=(-0.45, 0.84)); d.set_xticks([-0.4, 0, 0.4, 0.8])
    panel(d, 'd', 'Across tissues: blood vs macrophages')
    fig.legend(handles=[Line2D([], [], marker=marker, color=color, ls='', ms=4, label=label)
                        for _, _, color, marker, label in configs], loc='lower center',
               bbox_to_anchor=(0.53, 0.005), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.3)
    save(fig, out, 3)

def figure4(out):
    fig = plt.figure(figsize=(WIDTH, 6.60))
    a = axes(fig, 0.06, 4.54, 6.16, 1.72, diagram=True)
    panel(a, 'a', 'Separate the shared response from condition-specific changes')
    # Illustrative blood-panel operation: subtract each gene's condition mean.
    # This is NOT the projection used in the separate TG-GATEs analysis.
    x = np.array([[1.7, 1.2, 0.8, 1.5], [-1.2, -0.4, -0.8, -1.6],
                  [0.8, -0.4, 0.5, -0.1], [-0.3, 0.9, -0.9, 0.3], [0.2, 0.7, 0.9, 0.2]])
    shared = np.repeat(x.mean(axis=1, keepdims=True), x.shape[1], axis=1)
    residual = x-shared
    for i, (v, title) in enumerate(zip([x, shared, residual],
                                      ['Response', 'Across-condition mean', 'Specific residual'])):
        m = axes(fig, 0.63+i*2.08, 4.92, 1.12, 1.12)
        matrix(m, v, rowlabels=['g1', 'g2', 'g3', 'g4', 'g5'] if i == 0 else None,
               collabels=['1', '2', '3', '4'], vmin=-2, vmax=2)
        a.text((1.10+i*2.08)/6.16, 0.97, title, ha='center', fontsize=9, weight='bold')
    a.text(0.33, 0.51, '-', ha='center', fontsize=23)
    a.text(0.67, 0.51, '=', ha='center', fontsize=20)
    a.text(0.5, 0.09, 'Illustrative values; columns = conditions, rows = genes. Repeat within each species.',
           ha='center', fontsize=8.5, color=GREY)
    a.text(0.5, -0.08, 'Blue: negative response     White: zero     Red: positive response', ha='center', fontsize=8.5)
    b = axes(fig, 0.06, 3.36, 6.16, 0.67, diagram=True)
    panel(b, 'b', 'Test whether the condition identity matters')
    box(b, (0.01, 0.12), 0.19, 0.59, 'Donor\ncondition 1', color='#edf4f6', fontsize=8.5)
    box(b, (0.28, 0.12), 0.19, 0.59, 'Human\ncondition 1', color='#edf4f6', fontsize=8.5)
    arrow(b, (0.21, 0.41), (0.27, 0.41), color=BLUE)
    b.text(0.24, 0.84, 'Matched', color=BLUE, ha='center', fontsize=8.5)
    box(b, (0.54, 0.12), 0.19, 0.59, 'Donor\ncondition 2', fontsize=8.5)
    box(b, (0.81, 0.12), 0.18, 0.59, 'Human\ncondition 1', fontsize=8.5)
    arrow(b, (0.74, 0.41), (0.80, 0.41), color=GREY, style='--')
    b.text(0.77, 0.84, 'Mismatched', color=GREY, ha='center', fontsize=8.5)
    d = load('reconstruct_specificity')['blocks']; raw, res = d['raw'], d['residual']
    order = sorted(raw, key=lambda k: -raw[k]['matched']); y = np.arange(len(order))[::-1]
    left = axes(fig, 1.43, 0.55, 2.19, 2.16)
    right = axes(fig, 4.01, 0.55, 2.13, 2.16)
    total = np.array([raw[k]['matched'] for k in order])
    mis = np.array([raw[k]['mismatched_median'] for k in order])
    left.hlines(y, mis, total, color='#bac5ce', lw=1.5)
    left.plot(mis, y, 's', color=GREY, ms=4, label='Mismatched')
    left.plot(total, y, 'o', color=BLUE, ms=4.5, label='Matched')
    left.set_yticks(y, ['All nine (mean)' if k == 'uniform (all 9)'
                       else NAMES[k.removeprefix('only ')] for k in order], fontsize=8.5)
    left.set(xlim=(-0.12, 0.54), ylim=(-0.65, 9.65), xlabel='Raw recovery')
    left.set_xticks([-0.1, 0.1, 0.3, 0.5]); left.tick_params(axis='y', length=0)
    panel(left, 'c', 'Raw response')
    right.barh(y, [res[k]['residual_recovery'] for k in order], color=RED, height=0.54)
    right.set(xlim=(-0.45, 0.23), ylim=(-0.65, 9.65), xlabel='Residual recovery')
    right.set_yticks(y, []); right.set_xticks([-0.4, -0.2, 0, 0.2])
    panel(right, 'd', 'Consensus removed')
    for ax in [left, right]:
        ax.axvline(0, color='#555555', lw=0.7)
        ax.axvspan(ax.get_xlim()[0], 0, color='#f9eeeb', zorder=0)
        for yy in y:
            ax.axhline(yy, color='#edf0f2', lw=0.5, zorder=0)
    fig.legend(*left.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.40, 3.07/6.60),
               ncol=2, frameon=False, fontsize=8.5, columnspacing=1.0, handletextpad=0.4)
    save(fig, out, 4)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/paper/tex/figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    for draw in (figure1, figure2, figure3, figure4):
        draw(args.out)


if __name__ == "__main__":
    main()
