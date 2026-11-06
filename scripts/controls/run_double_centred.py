"""Compare timepoints after subtracting off-diagonal species row means.

Uses the same tie-aware Spearman/Mantel implementation as the primary rho analysis.
Input and output paths are explicit so the deposited result can be regenerated.
This is not an orthogonal additive-effects fit: excluding the diagonal leaves
nonzero residual row means. Historical output keys are retained for compatibility.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_rho_reliability_and_phylogeny as rho


def to_matrix(d):
    sp = sorted({s for pr in d for s in pr})
    n = len(sp); idx = {s: i for i, s in enumerate(sp)}
    A = np.full((n, n), np.nan)
    for (a, b), v in d.items():
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = v
    return sp, A


def double_centre(A):
    """r_ij - r_i. - r_.j + r_.. over off-diagonal entries."""
    B = A.copy()
    n = len(B)
    rm = np.array([np.nanmean(np.delete(B[i], i)) for i in range(n)])
    gm = np.nanmean(B[~np.eye(n, dtype=bool)])
    R = np.full_like(B, np.nan)
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(B[i, j]):
                R[i, j] = B[i, j] - rm[i] - rm[j] + gm
    return R, rm, gm


def additive_share(A):
    """Off-diagonal variance reduction, not an independent variance component."""
    n = len(A)
    mask = ~np.eye(n, dtype=bool) & np.isfinite(A)
    R, _, _ = double_centre(A)
    tot = np.nanvar(A[mask]); res = np.nanvar(R[mask])
    return float(1 - res / tot) if tot > 0 else np.nan


def as_dict(sp, A):
    return {tuple(sorted((sp[i], sp[j]))): A[i, j]
            for i in range(len(sp)) for j in range(i + 1, len(sp)) if np.isfinite(A[i, j])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--blood', type=Path, default=ROOT / 'artifacts/gse249010_v1/processed')
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts/biology/concordance/rho_double_centred.json')
    ap.add_argument('--seed', type=int, default=20261026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    # This comparison uses the blood panel's full, unchanged gene axis; no symbol bridge.
    rows, data, _ = rho.load_blood(args.blood)
    times = sorted({r['time_h'] for r in rows}, key=int)
    mats = {}
    out = {'additive_share': {}, 'raw': [], 'double_centred': []}
    for t in times:
        select_time = lambda r, time=t: r['time_h'] == time
        pairs = rho.rho_matrix(rho.species_vectors(rows, data, select_time))
        sp, A = to_matrix(pairs)
        mats[t] = (sp, A)
        out['additive_share'][t] = additive_share(A)

    for a, b in (('2', '6'), ('2', '24'), ('6', '24')):
        spa, A = mats[a]; spb, B = mats[b]
        raw = rho.compare(as_dict(spa, A), as_dict(spb, B), f'blood {a}h vs {b}h [raw]', rng)
        Ra, _, _ = double_centre(A); Rb, _, _ = double_centre(B)
        centred = rho.compare(as_dict(spa, Ra), as_dict(spb, Rb), f'blood {a}h vs {b}h [double-centred]', rng)
        if raw:
            out['raw'].append(raw)
        if centred:
            out['double_centred'].append(centred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
