#!/usr/bin/env python3
"""The phylogeny control: every comparison, read three ways.

The drawing lives in draw(), so make_report_figures.py can place this panel inside the
composite figure by calling it rather than keeping a second copy of the code. A second copy
is how Figure 5 went stale against its own artifact earlier in this revision: it was still
drawn from a superseded file while the text quoted the current one.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED, PURPLE, GREY = '#1f6f8b', '#c0392b', '#7d3c98', '#9aa0a6'
plt.rcParams.update({'font.size': 11})


def collect(d):
    """The rows of the panel: reliabilities first, then every comparison drawn."""
    rows = []
    for r in d['reliability']:
        rows.append((r['label'].replace('blood ', '').replace(
            ': half of subjects vs the other half', ' — split-half'), r, True))
    for r in d['comparisons']:
        # Draw every cross-panel estimator, not only the per-condition one. Section 2.4
        # reports both because they disagree in sign, and a figure that shows the favourable
        # one alone would contradict that. The estimator tag stays in the label.
        if 'cross_panel' in r.get('kind', '') or 'cross_time' == r.get('kind'):
            rows.append((r['label'].replace('blood ', ''), r, False))
    return rows


def draw(ax, d, legend=True, fontsize=9.5, show_n=True):
    rows = collect(d)
    y = np.arange(len(rows))[::-1].astype(float)
    h = 0.24
    for (lbl, r, isrel), yy in zip(rows, y):
        vals = [('all pairs', r.get('spearman_all'), r.get('p_all'), GREY),
                ('cross-order only', r.get('spearman_cross'), r.get('p_cross'), BLUE),
                ('partial on taxonomy', r.get('partial_taxdist'), r.get('p_partial'), PURPLE)]
        for j, (nm, v, p, c) in enumerate(vals):
            if v is None or not np.isfinite(v):
                continue
            off = (1 - j) * h
            ax.barh(yy + off, v, height=h * 0.86, color=c,
                    alpha=0.35 if isrel else 1.0,
                    label=nm if yy == y[0] else None)
            star = '*' if (p is not None and np.isfinite(p) and p < 0.05) else ''
            ax.text(v + (0.012 if v >= 0 else -0.012), yy + off, f'{v:+.2f}{star}',
                    va='center', ha='left' if v >= 0 else 'right', fontsize=fontsize - 0.5)
    ax.axvline(0, color='#888', lw=1.0)
    ax.set_yticks(y)
    # Inside a composite the pair counts go in the caption, where they do not eat the
    # left margin the row labels need.
    labs = [f'{l}\n(n={r["n_pairs"]}, cross-order {r["n_cross_order"]})' if show_n else l
            for l, r, _ in rows]
    ax.set_yticklabels(labs, fontsize=fontsize)
    ax.set_xlim(-0.45, 1.12)
    ax.set_xlabel('Spearman between the two redundancy matrices')
    if legend:
        ax.legend(fontsize=fontsize, loc='lower left', frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return rows


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    rows = draw(ax, d)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace('.png', '.pdf'))
    print('wrote', out)
    for l, r, isrel in rows:
        print(f'  {l:34s} n={r["n_pairs"]:3d} cross={r["n_cross_order"]:3d}  '
              f'all {r.get("spearman_all", float("nan")):+.3f}  '
              f'cross {r.get("spearman_cross", float("nan")):+.3f}  '
              f'partial {r.get("partial_taxdist", float("nan")):+.3f}')
