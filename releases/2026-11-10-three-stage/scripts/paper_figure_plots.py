"""Numerical manuscript figures generated from deposited analysis records."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
PURPLE = '#4267A3'
GREEN = '#199B8D'
ROSE = '#D86273'
GOLD = '#B49A44'
GRAY = '#8C8F96'
C = {'human': '#199B8D', 'animal': '#4267A3', 'rose': '#D86273', 'gold': '#B49A44', 'ink': '#454C54', 'pale': '#E7EBF0', 'rabbit': '#A78374'}
COLORS = {'H': '#555D64', 'Human': '#137F79', 'M56': '#7355A0', 'M82': '#B14C73', 'Rabbit': '#9B6E5D', 'Fixed': '#84929D'}
_INFORMATION_STYLE = {'font.family': 'Arial', 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 1, 'lines.linewidth': 1.2, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white', 'savefig.dpi': 600, 'xtick.major.width': 1, 'xtick.minor.width': 1, 'ytick.major.width': 1, 'ytick.minor.width': 1, 'lines.markeredgewidth': 1}
_CORRESPONDENCE_STYLE = {'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.75, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'}
_STATE_STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 8, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'}

def _information_canvas(height=145):
    (f, ax) = plt.subplots(2, 2, figsize=(180 / 25.4, height / 25.4))
    f.subplots_adjust(left=0.12, right=0.975, bottom=0.1, top=0.94, wspace=0.48, hspace=0.55)
    for (i, a) in enumerate(ax.flat):
        a.text(-0.19, 1.08, 'abcd'[i], transform=a.transAxes, fontsize=8, fontweight='bold', ha='left', va='bottom')
        a.tick_params(width=1, length=3)
        a.set_axisbelow(True)
    return (f, ax.ravel())

def _state_layout(height=139):
    (f, axes) = plt.subplots(2, 2, figsize=(180 / 25.4, height / 25.4))
    f.subplots_adjust(left=0.105, right=0.975, bottom=0.125, top=0.91, wspace=0.47, hspace=0.62)
    return (f, list(axes.flat))

def _state_title(ax, letter, text):
    ax.text(-0.18, 1.07, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(text, loc='left', pad=10, fontweight='normal')

def _paired_conditions(ax, records, metric, labels, colors, panel, rows):
    keys = sorted(records[0]['by_condition'])
    assert all((sorted(r['by_condition']) == keys for r in records))
    vals = np.array([[r['by_condition'][k][metric] for r in records] for k in keys])
    xx = np.arange(len(records))
    for (i, k) in enumerate(keys):
        ax.plot(xx, vals[i], color='#CED4DA', lw=0.65, zorder=1)
        for (j, v) in enumerate(vals[i]):
            rows.append({'panel': panel, 'condition': k, 'method': labels[j].replace('\n', ' '), 'metric': metric, 'value': v})
    for (j, r) in enumerate(records):
        ax.scatter(np.full(len(keys), j), vals[:, j], s=12, color=colors[j], edgecolors='white', linewidths=0.3, zorder=3)
        mean = float(r['macro'][metric])
        assert np.isclose(vals[:, j].mean(), mean, atol=2e-07)
        ax.plot([j - 0.22, j + 0.22], [mean, mean], color='#182126', lw=1.6, zorder=4)
    ax.set_xticks(xx, labels)
    ax.set_xlim(-0.5, len(records) - 0.5)
    ax.grid(axis='y', color='#E5E9EC', lw=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals.max() * 1.1, 0.1))

def build_figure_2(ctx):
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update(_INFORMATION_STYLE)
        main = ctx.load_json('outputs/feedback_nested_chemical_20261107_absolute_v1/report.json')
        joint = ctx.load_json('outputs/feedback_joint_specificity_20261107_absolute/report.json')
        budget = ctx.load_json('outputs/feedback_label_budget_20261107_absolute/report.json')
        ident = ctx.load_json('outputs/feedback_joint_specificity_20261107_absolute/identification.json')
        (f, axes) = _information_canvas()
        (a, b, c, d) = axes
        prior = main['metrics']['chemical_context']
        source = main['metrics']['source_plus_chemical_context']
        assert prior['group_ids'] == source['group_ids']
        x = np.asarray(prior['group_losses'])
        y = np.asarray(source['group_losses'])
        limits = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
        a.plot(limits, limits, color=GRAY, linestyle='--', zorder=1)
        a.scatter(x, y, c=np.where(y < x, GREEN, ROSE), s=16, alpha=0.85, edgecolor='white', linewidth=1, zorder=3)
        a.set(xscale='log', yscale='log', xlim=limits, ylim=limits, xlabel='Chemical / exposure prior MSE', ylabel='Animal-informed MSE')
        a.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        a.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        a.minorticks_off()
        jd = np.asarray(joint['joint_residual_gain']['group_deltas'])
        order = np.argsort(jd)
        b.bar(np.arange(len(jd)), jd[order] * 1000, color=np.where(jd[order] >= 0, GREEN, ROSE), width=1)
        b.axhline(0, color='black', linewidth=1)
        b.set(xlabel='Compounds ordered by error difference', ylabel='MSE reduction with source residual\n(×0.001)', xlim=(-1, len(jd)))
        budget_data = []
        for (i, key) in enumerate(['16', '32', '64', 'all']):
            v = budget['results'][key]['aggregate']
            bp = np.asarray(v['seed_macro_mse']['chemical_context'])
            ap = np.asarray(v['seed_macro_mse']['source_plus_chemical_context'])
            gain = 1 - ap / bp
            offsets = np.linspace(-0.07, 0.07, len(gain)) if len(gain) > 1 else np.zeros(1)
            c.scatter(i + offsets, gain * 100, color=GREEN, s=20, zorder=3)
            budget_data.append({'reference_budget': key, 'gain_percent_per_draw': (gain * 100).tolist(), 'mean_gain_percent': float((1 - v['mean_seed_macro_mse']['source_plus_chemical_context'] / v['mean_seed_macro_mse']['chemical_context']) * 100)})
        c.plot(range(4), [v['mean_gain_percent'] for v in budget_data], color=GREEN)
        c.set(xticks=range(4), xticklabels=['16', '32', '64', '100–101'], xlabel='Paired reference compounds per fold', ylabel='MSE reduction versus prior (%)', ylim=(0, 6))
        names = ['chemical_context_plus_common', 'chemical_context_plus_common_residual']
        vals = [ident['identification'][n]['common_defined_queries'] for n in names]
        d.bar([0, 1], [v['top1'] * 100 for v in vals], color=[PURPLE, GREEN], width=0.55)
        d.axhline(vals[0]['chance_top1'] * 100, color=GRAY, linestyle='--')
        d.set(xticks=[0, 1], xticklabels=['Prior + amplitude', 'Add source residual'], ylabel='Correct response identification (%)', ylim=(0, 27))
        ctx.write_json('Fig2_information.json', {'a': {'compound_ids': prior['group_ids'], 'prior_mse': x.tolist(), 'animal_mse': y.tolist()}, 'b': {'compound_ids': joint['joint_residual_gain']['group_ids'], 'residual_gain': jd.tolist()}, 'c': budget_data, 'd': {'models': names, 'scores': vals}})
        ctx.save(f, 'Fig2_information', list(axes), png_dpi=240, pdf_dpi=600, portable_svg=False)

def build_figure_3(ctx):
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update(_CORRESPONDENCE_STYLE)
        runs = [ctx.load_json('outputs/crossspecies_direct_03/' + s + '/report.json') for s in ['source_exposure_v1', 'paired_exposure_v1']]
        retrieval = ctx.load_json('outputs/biorxiv_ready_20261108/metric_checks/retrieval_report.json')['reports']
        external = ctx.load_json('outputs/external_transfer_04/report.json')
        post = ctx.load_json('docs/multisource_20261108/external_transfer_04/posthoc_scalar_correspondence/report_v1.json')
        (fig, axs) = plt.subplots(2, 2, figsize=(180 / 25.4, 148 / 25.4))
        fig.subplots_adjust(left=0.16, right=0.975, bottom=0.13, top=0.91, wspace=0.54, hspace=0.69)
        (a, b, c, d) = axs.flat
        axes = list(axs.flat)
        source = {}
        titles = ['Training / query correspondence', 'Response identification', 'Frozen transfer: three drugs', 'Fixed-scalar source swap']
        for (letter, ax, panel_title) in zip('abcd', axes, titles):
            ax.text(-0.22, 1.09, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
            ax.set_title(panel_title, loc='left', pad=10, fontsize=7.5)
        labels = ['No animal', 'Correct / correct', 'Correct / wrong', 'Wrong / correct', 'Wrong / wrong']
        rows = []
        for (j, (r, col, mk)) in enumerate(zip(runs, [C['animal'], C['human']], ['o', 'D'])):
            base = r['macro_mse_per_seed']['prior'][0]
            for (i, arm) in enumerate(r['arms']):
                vv = np.asarray(r['macro_mse_per_seed'][arm], float) / base
                yy = i + (-0.17 if j == 0 else 0.17)
                if i >= 2:
                    a.scatter(vv, yy + np.linspace(-0.055, 0.055, len(vv)), s=8, color=col, alpha=0.45, zorder=3)
                a.scatter(np.median(vv), yy, marker=mk, s=20, facecolor=col, edgecolor='white', lw=0.5, zorder=4)
                rows.append({'regime': r['regime'], 'arm': arm, 'relative_mse': vv.tolist()})
        a.axvline(1, color=C['ink'], lw=0.8, ls='--')
        a.set(yticks=range(5), yticklabels=labels, ylim=(4.6, -0.6), xlim=(0.94, 1.07), xlabel='MSE / no-animal MSE', xticks=[0.94, 0.98, 1.02, 1.06])
        a.legend(handles=[Line2D([], [], ls='', marker=mk, color=co, markersize=3, label=lab) for (mk, co, lab) in zip(['o', 'D'], [C['animal'], C['human']], ['107 compounds', '49 compounds'])], frameon=False, loc='lower left', bbox_to_anchor=(-0.03, -0.39), ncol=2, fontsize=6.5, columnspacing=0.7, handletextpad=0.25)
        source['correspondence'] = rows
        br = []
        for (i, (r, v)) in enumerate(zip(runs, retrieval)):
            p = v['summary']['prior'][0]
            s = v['summary']['correct_fit_correct_query'][0]
            b.bar(i - 0.17, p['top1'] * 100, 0.3, color='#6B737D', label='No animal' if i == 0 else None)
            b.bar(i + 0.17, s['top1'] * 100, 0.3, color=C['rose'], label='Correct animal' if i == 0 else None)
            b.hlines(p['chance_top1'] * 100, i - 0.39, i + 0.39, colors='#8D949C', ls='--', lw=0.9)
            br.append({'regime': r['regime'], 'prior': p, 'correct': s})
        b.set(xticks=[0, 1], xticklabels=['Animal\nexposure', 'Paired\nexposure'], ylabel='Correct identification (%)', ylim=(0, 80), yticks=[0, 20, 40, 60, 80])
        b.legend(frameon=False, loc='upper left', fontsize=6.7, handlelength=1.2)
        source['identification'] = br
        names = ['CDDO-Me', 'KI696', 'Thapsigargin']
        rules = ['zero', 'human_mean', 'chemical_context', 'source_plus_context', 'scalar_nointercept']
        colors = ['#91969D', C['ink'], C['animal'], C['human'], C['rose']]
        labs = ['Zero', 'Human history', 'Chemical prior', 'Fusion', 'Scalar']
        marks = ['|', 's', 'o', 'D', '^']
        er = []
        for (j, (rule, co, lab, mk)) in enumerate(zip(rules, colors, labs, marks)):
            vv = [x['rules'][rule]['mse'] / x['rules']['zero']['mse'] for x in external['external_tasks']]
            c.scatter(np.arange(3) + (j - 2) * 0.13, vv, s=19, marker=mk, color=co, label=lab, zorder=4)
            er.append({'rule': rule, 'relative_mse': vv})
        c.axhline(1, lw=0.8, ls='--', color='#959CA4')
        c.set(xticks=range(3), xticklabels=names, ylabel='MSE / zero-response MSE', xlim=(-0.55, 2.55), ylim=(0.7, 1.9), yticks=[0.8, 1, 1.2, 1.4, 1.6, 1.8])
        c.legend(frameon=False, ncol=2, loc='upper center', fontsize=6.5, columnspacing=0.65, handletextpad=0.3)
        source['external'] = er
        matrix = np.asarray(post['mse_query_by_donor'])
        assert matrix.shape == (3, 3)
        norm = matrix / np.diag(matrix)[:, None]
        assert np.isfinite(norm).all()
        cm = LinearSegmentedColormap.from_list('rose', ['#FBF5F6', '#EAB7C0', C['rose']])
        d.pcolormesh(np.arange(4) - 0.5, np.arange(4) - 0.5, norm, cmap=cm, vmin=0.98, vmax=1.12)
        d.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), xticks=range(3), xticklabels=names, yticks=range(3), yticklabels=names, xlabel='Animal source', ylabel='Human query')
        for i in range(3):
            for j in range(3):
                d.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=7)
        d.text(0.5, -0.33, 'MSE / matched-source MSE', transform=d.transAxes, ha='center', fontsize=7)
        source['source_swap'] = {'raw_mse': matrix.tolist(), 'ratios': norm.tolist(), 'fixed_beta': post.get('beta', post.get('fixed_beta'))}
        ctx.write_json('Fig3_correspondence_transport.json', source)
        ctx.save(fig, 'Fig3_correspondence_transport', axes, png_dpi=450, pdf_dpi=450, portable_svg=True)

def build_figure_4(ctx):
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update(_STATE_STYLE)
        pre = 'outputs/remote_gpu_20261108/'
        native = ctx.load_json(pre + 'state_native_formal_64/report.json')['test']
        mis = ctx.load_json(pre + 'state_mispaired_posthoc/report.json')['mismatched_test']
        policy = ctx.load_json(pre + 'state_gaussian_policy_formal_64/report.json')['test']
        sc = ctx.load_json(pre + 'scdfm_nft_formal/report.json')['test']
        wi = ctx.load_json(pre + 'scdfm_witness_formal/evaluation_float64.json')['evaluation']
        (f, aa) = _state_layout(142)
        rows = []
        specs = [([native['released_checkpoint'], native['native_energy_sft'], mis], 'native_energy', ['Released', 'Paired', 'Mispaired'], [COLORS['H'], COLORS['Human'], COLORS['M82']], 'State: target correspondence', 'Native energy'), ([policy[k] for k in ['zero_update_gaussian_reference', 'pathwise', 'reinforce']], 'expected_native_energy', ['Zero update', 'Pathwise', 'REINFORCE'], [COLORS['H'], COLORS['Human'], COLORS['M82']], 'State: matched update comparison', 'Expected native energy'), ([sc[k] for k in ['released_checkpoint', 'native_fm', 'nft', 'terminal_gradient']], 'energy_u', ['Released', 'Supervised', 'Reward\nupdate', 'Direct\ngradient'], [COLORS['H'], COLORS['Human'], COLORS['M82'], COLORS['M56']], 'scDFM: population update', 'Energy U'), ([wi[k] for k in ['released_checkpoint', 'witness', 'native_fm']], 'energy_v', ['Released', 'Witness', 'Matched\nsupervision'], [COLORS['H'], COLORS['M56'], COLORS['Human']], 'scDFM: witness-only sensitivity', 'Energy V')]
        for (l, ax, (rr, met, lab, col, ttl, yl)) in zip('abcd', aa, specs):
            _state_title(ax, l, ttl)
            _paired_conditions(ax, rr, met, lab, col, l, rows)
            ax.set_ylabel(yl)
        ctx.write_csv('Fig4_adaptation.csv', pd.DataFrame(rows))
        ctx.save(f, 'Fig4_adaptation', aa, png_dpi=600, pdf_dpi=600, portable_svg=False)

def build_figure_6(ctx):
    with plt.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update(_STATE_STYLE)
        dirs = {'exp2': 'outputs/gse325993_state_20261110/state_primary_v1/audit', 'exp3': 'outputs/gse325993_screen_validation_20261110/validation_v1/audit'}
        summaries = {k: ctx.load_json(v + '/summary.json') for (k, v) in dirs.items()}
        scores = {k: ctx.load_csv(v + '/line_seed_scores.csv') for (k, v) in dirs.items()}
        rr = ctx.load_csv('outputs/nonrodent_singlecell_20261110/rabbit_primary_v1/audit/donor_scores.csv')
        rs = ctx.load_json('outputs/nonrodent_singlecell_20261110/rabbit_primary_v1/audit/summary.json')
        (f, aa) = _state_layout(151)
        groups = ['H', 'Hextra_cells', 'Monkey56', 'Monkey56A0', 'Monkey82', 'Monkey82A0', 'Three_quarter_H_response']
        labs = ['H', '+Human', '+M56', '+M56\nA0', '+M82', '+M82\nA0', '0.75H']
        cols = [COLORS[k] for k in ['H', 'Human', 'M56', 'M56', 'M82', 'M82', 'Fixed']]
        lines = ['63Ab2.2-02', '29B5-39s']
        rows = []
        for (ax, (screen, d), letter) in zip(aa[:2], scores.items(), 'ab'):
            _state_title(ax, letter, 'Initial screen: exp2' if screen == 'exp2' else 'Frozen evaluation: exp3')
            for (i, (g, col)) in enumerate(zip(groups, cols)):
                values = []
                for (j, line) in enumerate(lines):
                    vals = d[(d['variant'] == g) & (d['target_line'] == line)]['mean_response_mse'].to_numpy()
                    assert len(vals) == 3
                    v = vals.mean()
                    values.append(v)
                    ax.scatter(i - 0.1 + j * 0.2, v * 1000, s=18, marker=['o', 'D'][j], facecolor='white' if 'A0' in g else col, edgecolor=col, linewidth=0.7, zorder=3)
                    rows.append({'screen': screen, 'group': g, 'target_line': line, 'seed_mean_mse': v})
                mean = summaries[screen]['variants'][g]['macro']['mean_response_mse']
                assert np.isclose(np.mean(values), mean, atol=1e-14)
                ax.plot([i - 0.2, i + 0.2], [mean * 1000] * 2, color='#172027', lw=1.35)
            ax.set_xticks(range(7), labs)
            ax.set_xlim(-0.55, 6.55)
            ax.set_ylabel('Response MSE (×10$^{-3}$)')
            ax.set_ylim(0, 5 if screen == 'exp2' else 15)
            ax.set_yticks([0, 1, 2, 3, 4, 5] if screen == 'exp2' else [0, 5, 10, 15])
            ax.grid(axis='y', color='#E5E9EC', lw=0.45)
            ax.set_axisbelow(True)
        handles = [Line2D([], [], marker='o', ls='', color='#444', markersize=3, label=lines[0]), Line2D([], [], marker='D', ls='', color='#444', markersize=3, label=lines[1]), Line2D([], [], color='#172027', lw=1.4, label='Equal-line mean')]
        f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 0.995), ncol=3, frameon=False, fontsize=7, columnspacing=1.4)
        c = aa[2]
        _state_title(c, 'c', 'M56 versus additional human cells in exp3')
        d = scores['exp3']
        diffs = []
        paired = []
        for (i, line) in enumerate(lines):
            x = d[d['target_line'] == line].pivot(index='seed', columns='variant', values='mean_response_mse')
            vals = (x['Hextra_cells'] - x['Monkey56']).to_numpy() * 1000
            diffs.extend(vals)
            c.scatter(i + np.linspace(-0.12, 0.12, 3), vals, s=20, marker=['o', 'D'][i], color=COLORS['M56'], zorder=3)
            c.plot([i - 0.2, i + 0.2], [vals.mean()] * 2, color='#172027', lw=1.4)
            for (seed, v) in zip(x.index, vals):
                paired.append({'line': line, 'seed': int(seed), 'mse_extra_human_minus_m56': v / 1000})
        c.scatter(2, np.mean(diffs), color='#172027', marker='_', s=110)
        c.axhline(0, color='#89939B', lw=0.65, ls='--')
        c.set_xticks([0, 1, 2], [lines[0], lines[1], 'Equal-line\nmean'])
        c.set_ylabel('MSE advantage of M56 (×10$^{-3}$)')
        c.set_xlim(-0.45, 2.45)
        c.set_ylim(-0.7, 1.8)
        c.set_yticks([-0.5, 0, 0.5, 1, 1.5])
        c.grid(axis='y', color='#E5E9EC', lw=0.45)
        c.set_axisbelow(True)
        ax = aa[3]
        _state_title(ax, 'd', 'Rabbit transfer across ten human donors')
        wide = rr.pivot(index='donor', columns='variant', values='mean_response_mse')
        groupsr = ['H+Hextra', 'H+Rabbit', 'H+RabbitA0']
        rp = []
        for (j, g) in enumerate(groupsr):
            vv = (wide['H'] - wide[g]).to_numpy() * 1000
            col = COLORS['Human'] if j == 0 else COLORS['Rabbit']
            offset = np.linspace(-0.14, 0.14, len(vv))
            ax.scatter(j + offset, vv, s=15, facecolor='white' if j == 2 else col, edgecolor=col, linewidth=0.7, zorder=3)
            ax.plot([j - 0.22, j + 0.22], [vv.mean()] * 2, color='#172027', lw=1.4)
            assert np.isclose(wide[g].mean(), rs['variants'][g]['mean_response_mse']['mean'], atol=1e-14)
            for (donor, val) in zip(wide.index, vv):
                rp.append({'donor': donor, 'group': g, 'mse_H_minus_group': val / 1000})
        assert all(wide['H'] < wide['H+Rabbit']) and all(wide['H'] > wide['H+Hextra'])
        ax.axhline(0, color='#89939B', lw=0.65, ls='--')
        ax.set_xticks([0, 1, 2], ['+Human', '+Rabbit', '+Rabbit A0'])
        ax.set_ylabel('MSE reduction from H (×10$^{-3}$)')
        ax.set_ylim(-2.4, 1.2)
        ax.set_yticks([-2, -1, 0, 1])
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis='y', color='#E5E9EC', lw=0.45)
        ax.set_axisbelow(True)
        ctx.write_csv('Fig6_macaque_line_means.csv', pd.DataFrame(rows))
        ctx.write_csv('Fig6_m56_human_paired.csv', pd.DataFrame(paired))
        ctx.write_csv('Fig6_rabbit_paired.csv', pd.DataFrame(rp))
        ctx.save(f, 'Fig6_augmentation_results', aa, png_dpi=600, pdf_dpi=600, portable_svg=False)
