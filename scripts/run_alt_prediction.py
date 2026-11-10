#!/usr/bin/env python3
"""Evaluate exploratory compound-held-out predictions of rat aminotransferase outcomes.

Similarity-weighted predictions use other compounds' log2 endpoints. Inputs are cultured
human/rat hepatocytes at 24 h and high-dose rat liver at 4 days. Serum endpoints are maxima
across doses and days: the legacy result label '(same animals)' denotes the same resource,
not uniform individual-animal matching. The historical key is retained for compatibility.
Response magnitude and changed-gene counts provide label-independent baselines.

Permutation tests recompute label-dependent predictions after each target permutation.
Historical outputs produced before 2026-11-06 held predictions fixed; their predictive
p-values require recomputation. Bootstrap intervals remain conditional on fixed predictions.
"""
import argparse, json, math, sys
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from run_system_vs_species import bare, load   # noqa: E402
from run_dili_concordance import normalise      # noqa: E402


def profile_matrix(ds, dose, time, compounds):
    """genes x compounds, replicate-averaged, NaN preserved."""
    cols = []
    for c in compounds:
        i = ds['index'][(c, dose, time)]
        with np.errstate(invalid='ignore'):
            cols.append(np.nanmean(ds['values'][:, i, :], axis=1))
    return np.vstack(cols).T


def pairwise_corr(X, min_n=200):
    """Pairwise-complete Pearson between every pair of columns."""
    n = X.shape[1]
    S = np.full((n, n), np.nan)
    for i in range(n):
        xi = X[:, i]
        fi = np.isfinite(xi)
        for j in range(i, n):
            xj = X[:, j]
            m = fi & np.isfinite(xj)
            if m.sum() < min_n:
                continue
            a = xi[m] - xi[m].mean()
            b = xj[m] - xj[m].mean()
            d = math.sqrt(float(a @ a) * float(b @ b))
            S[i, j] = S[j, i] = float(a @ b) / d if d > 0 else np.nan
    return S


def loo_similarity_predict(S, y, power=3.0, top_k=None):
    """Leave one out: weight every other compound by its similarity, raised to a power so
    that near-zero similarities contribute almost nothing."""
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(n):
        w = S[i].copy()
        w[i] = np.nan
        ok = np.isfinite(w) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        ww = np.clip(w[ok], 0, None) ** power
        yy = y[ok]
        if top_k is not None and top_k < ok.sum():
            keep = np.argsort(-ww)[:top_k]
            ww, yy = ww[keep], yy[keep]
        if ww.sum() <= 0:
            pred[i] = yy.mean()
        else:
            pred[i] = float(ww @ yy / ww.sum())
    return pred


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float('nan')
    x = rankdata(a[m], method='average')
    z = rankdata(b[m], method='average')
    x -= x.mean()
    z -= z.mean()
    d = math.sqrt(float(x @ x) * float(z @ z))
    return float(x @ z) / d if d > 0 else float('nan')


def auc(score, label):
    m = np.isfinite(score) & np.isfinite(label)
    s, l = score[m], label[m].astype(bool)
    npos, nneg = int(l.sum()), int((~l).sum())
    if npos == 0 or nneg == 0:
        return float('nan'), npos, nneg
    r = rankdata(s, method='average')
    return float((r[l].sum() - npos * (npos + 1) / 2) / (npos * nneg)), npos, nneg


def boot_auc(pred, ybin, rng, n_boot=2000):
    """Percentile interval over compounds. With 13 to 23 positives the point estimate on
    its own invites more confidence than it has earned."""
    m = np.isfinite(pred) & np.isfinite(ybin)
    p, l = pred[m], ybin[m]
    n = len(p)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(p[idx], l[idx])[0]
        if np.isfinite(a):
            vals.append(a)
    if len(vals) < 100:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(name, pred, y, ybin, rng, n_perm, report, *, predict_from_labels=None):
    """Score predictions and test the full label-dependent prediction procedure.

    Supply predict_from_labels for supervised predictors. Fixed scores are valid only
    for label-independent baselines, such as the transcriptome response magnitude.
    """
    sp = spearman(pred, y)
    au, npos, nneg = auc(pred, ybin)
    lo, hi = boot_auc(pred, ybin, rng)
    ok = np.isfinite(pred)
    null_sp, null_au = [], []
    for _ in range(n_perm):
        p = rng.permutation(len(y))
        perm_y = y[p]
        perm_pred = pred if predict_from_labels is None else predict_from_labels(perm_y)
        null_sp.append(spearman(perm_pred, perm_y))
        null_au.append(auc(perm_pred, ybin[p])[0])
    null_sp = np.array([x for x in null_sp if np.isfinite(x)])
    null_au = np.array([x for x in null_au if np.isfinite(x)])
    p_sp = float((1 + (np.abs(null_sp) >= abs(sp)).sum()) / (1 + null_sp.size)) if null_sp.size and np.isfinite(sp) else float('nan')
    p_au = float((1 + (null_au >= au).sum()) / (1 + null_au.size)) if null_au.size and np.isfinite(au) else float('nan')
    ci = f'[{lo:.3f}, {hi:.3f}]' if lo is not None else '[--]'
    print(f'  {name:34s} n={int(ok.sum()):3d}  rho {sp:+.3f} (p={p_sp:.3f})   '
          f'AUC {au:.3f} {ci} (p={p_au:.3f})   [{npos}+/{nneg}-]')
    report[name] = {'n': int(ok.sum()), 'spearman': sp, 'p_spearman': p_sp,
                    'auc': au, 'auc_ci95': [lo, hi], 'p_auc': p_au,
                    'n_pos': npos, 'n_neg': nneg,
                    'auc_null_p95': float(np.percentile(null_au, 95)) if null_au.size else None,
                    'permutation_method': ('recompute_predictions_per_label_permutation'
                                           if predict_from_labels is not None
                                           else 'fixed_label_independent_scores'),
                    'interval_method': 'compound_bootstrap_of_fixed_predictions'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-hep', required=True)
    ap.add_argument('--rat-hep', required=True)
    ap.add_argument('--rat-vivo', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--alt', required=True, help='markers json from build_vivo_markers.py')
    ap.add_argument('--endpoint', default='ALT(IU/L)')
    ap.add_argument('--go', help='gene to GO BP table; adds pathway-aggregated predictors')
    ap.add_argument('--vitro-dose', default='High',
                    help='in vitro dose level for the hepatocyte predictors, or "sweep"')
    ap.add_argument('--exp-dir', help='annotation directory; adds dose itself as a predictor')
    ap.add_argument('--alt-threshold', type=float, default=2.0)
    ap.add_argument('--power', type=float, default=3.0)
    ap.add_argument('--n-perm', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    axis = json.load(open(a.axis))['axis']
    H = load(a.human_hep, 'homo_sapiens', axis)
    R = load(a.rat_hep, 'rattus_norvegicus', axis)
    V = load(a.rat_vivo, 'rattus_norvegicus', axis)
    raw = json.load(open(a.alt))
    first = next(iter(raw.values()))
    if 'alt_fold_max' in first:            # the older ALT-only file
        alt = {normalise(k): v['alt_fold_max'] for k, v in raw.items()}
        ep = 'ALT(IU/L)'
    else:
        ep = a.endpoint
        alt = {normalise(k): v[ep]['fold_max'] for k, v in raw.items() if ep in v}
    print(f'endpoint: {ep} at {a.alt_threshold:g}x')

    def have(ds, dose, time):
        return {k[0] for k in ds['index'] if k[1] == dose and k[2] == time}
    comps = sorted(have(H, 'High', '24 hr') & have(R, 'High', '24 hr')
                   & have(V, 'High', '4 day') & set(alt))
    comps = [c for c in comps if alt[c] and alt[c] > 0]
    y = np.log2(np.array([alt[c] for c in comps]))
    ybin = (y >= math.log2(a.alt_threshold)).astype(float)
    print(f'compounds usable in all three systems with a value: {len(comps)}')
    print(f'positives: {int(ybin.sum())}   negatives: {int((1-ybin).sum())}')

    print('\nbuilding profiles and similarity matrices ...', flush=True)
    levels = ['Low', 'Middle', 'High'] if a.vitro_dose == 'sweep' else [a.vitro_dose]
    P = {'rat liver in vivo (same animals)': profile_matrix(V, 'High', '4 day', comps)}
    for lv in levels:
        tag = '' if len(levels) == 1 else f' [{lv}]'
        for nm_, ds in (('rat hepatocyte', R), ('human hepatocyte', H)):
            if all((c, lv, '24 hr') in ds['index'] for c in comps):
                P[f'{nm_}{tag}'] = profile_matrix(ds, lv, '24 hr', comps)
            else:
                miss = sum(1 for c in comps if (c, lv, '24 hr') not in ds['index'])
                print(f'  {nm_} at {lv}/24 hr: {miss} of {len(comps)} compounds absent, skipped')
    if a.go:
        # the obvious rebuttal to an in vitro predictor that fails is that genes are the
        # wrong unit. Aggregating to external pathway terms is the cheapest test of it.
        import collections as _c
        pos = {bare(p['homo_sapiens']): i for i, p in enumerate(axis)}
        by = _c.defaultdict(list)
        for line in open(a.go, encoding='utf-8', errors='replace'):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 2 and bare(f[0]) in pos:
                by[f[1]].append(pos[bare(f[0])])
        terms = [np.array(sorted(set(v))) for v in by.values() if 5 <= len(set(v)) <= 500]
        print(f'  aggregating to {len(terms):,} GO BP terms')
        def agg(X):
            out = np.full((len(terms), X.shape[1]), np.nan)
            for ti, idx in enumerate(terms):
                sub = X[idx]
                with np.errstate(invalid='ignore'):
                    cnt = np.isfinite(sub).sum(axis=0)
                    mean = np.nanmean(sub, axis=0)
                out[ti] = np.where(cnt >= 5, mean, np.nan)
            return out
        for k in list(P):
            P[f'{k} [GO terms]'] = agg(P[k])
    S = {k: pairwise_corr(v, min_n=30 if 'GO terms' in k else 200)
         for k, v in P.items()}
    if 'rat hepatocyte' in S and 'human hepatocyte' in S:
        S['both hepatocytes (mean similarity)'] = np.nanmean(
            np.stack([S['rat hepatocyte'], S['human hepatocyte']]), axis=0)

    rng = np.random.default_rng(a.seed)
    report = {'endpoint': ep, 'n_compounds': len(comps), 'n_positive': int(ybin.sum()),
              'threshold': a.alt_threshold, 'power': a.power, 'results': {}}
    print(f'\n--- leave-one-compound-out, similarity-weighted (power {a.power:g}) ---')
    for name, s in S.items():
        evaluate(name, loo_similarity_predict(s, y, a.power), y, ybin, rng, a.n_perm,
                 report['results'],
                 predict_from_labels=lambda labels, similarity=s:
                     loo_similarity_predict(similarity, labels, a.power))

    print('\n--- baselines that use no similarity structure ---')
    for lbl, X in P.items():
        with np.errstate(invalid='ignore'):
            rms = np.sqrt(np.nanmean(X ** 2, axis=0))
            moved = np.nansum(np.abs(X) > 1.0, axis=0).astype(float)
        evaluate(f'{lbl}: response magnitude', rms, y, ybin, rng, a.n_perm, report['results'])
        evaluate(f'{lbl}: genes |log2|>1', moved, y, ybin, rng, a.n_perm, report['results'])

    if a.exp_dir:
        # The in vivo high dose was set near what the animal tolerates, so dose on its own
        # already encodes how toxic the compound is. If dose predicts injury as well as the
        # transcriptome does, then the transcriptome is not what is doing the work.
        import csv as _csv
        dose = {}
        for fn in ('rat_vivo_repeat_exp.txt',):
            for r in _csv.DictReader(open(f'{a.exp_dir}/{fn}', encoding='utf-8',
                                          errors='replace'), delimiter='\t'):
                if r['DOSE_LEVEL'].strip() == 'High':
                    try:
                        v = float(r['DOSE'])
                    except (TypeError, ValueError):
                        continue
                    if v > 0:
                        dose[normalise(r['COMPOUND_NAME'])] = v
        have = np.array([dose.get(c, np.nan) for c in comps], dtype=float)
        if np.isfinite(have).sum() > 50:
            print('\n--- dose itself, with no transcriptome at all ---')
            lg = np.where(np.isfinite(have), np.log10(np.where(have > 0, have, np.nan)), np.nan)
            evaluate('in vivo High dose (log mg/kg)', lg, y, ybin, rng, a.n_perm, report['results'])
            evaluate('in vivo High dose, negated', -lg, y, ybin, rng, a.n_perm, report['results'])

    print('\n--- does the human arm add anything over the rat arm? ---')
    rr = report['results'].get('rat hepatocyte')
    bb = report['results'].get('both hepatocytes (mean similarity)')
    hh = report['results'].get('human hepatocyte')
    if rr and bb and hh:
        print(f"  rat hepatocyte alone   AUC {rr['auc']:.3f}")
        print(f"  human hepatocyte alone AUC {hh['auc']:.3f}")
        print(f"  both                   AUC {bb['auc']:.3f}   "
              f"delta vs rat alone {bb['auc'] - rr['auc']:+.3f}")
    else:
        print('  (the combined predictor is only built when a single in vitro dose is used)')
    report['compounds'] = comps
    report['alt_log2'] = y.tolist()
    json.dump(report, open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
