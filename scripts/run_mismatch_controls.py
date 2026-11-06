#!/usr/bin/env python3
"""How much of the in-vitro-against-in-vivo penalty is the system, and how much is only
that the two grids cannot be aligned in time or exposure?

The species comparison puts High/24 hr against High/24 hr. The system comparison has to
put High/24 hr against High/4 day, because the in vitro grid is 2/8/24 hr and the in vivo
grid is 4/8/15/29 day, with no overlap. Time and dose mismatch therefore penalise the
system comparison alone, and that penalty has to be measured before the two are ranked.

Each control holds species and system fixed and changes one thing:
  time   High/24 hr against High/2 hr and High/8 hr, within one dataset
  dose   High/24 hr against Middle/24 hr and Low/24 hr, within one dataset
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from run_system_vs_species import compare, load, summarise, NORMALISATION_NOTE   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human-hep', required=True)
    ap.add_argument('--rat-hep', required=True)
    ap.add_argument('--rat-vivo', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--min-rel', type=float, default=0.10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    print(NORMALISATION_NOTE)

    axis = json.load(open(a.axis))['axis']
    sets = {'human hepatocyte': (load(a.human_hep, 'homo_sapiens', axis), 'hr'),
            'rat hepatocyte': (load(a.rat_hep, 'rattus_norvegicus', axis), 'hr'),
            'rat liver in vivo': (load(a.rat_vivo, 'rattus_norvegicus', axis), 'day')}
    print(f'axis verified: {len(axis):,} ortholog pairs')
    keys = ['gene', 'direction']
    out = {}

    for name, (ds, unit) in sets.items():
        times = ['2 hr', '8 hr'] if unit == 'hr' else ['8 day', '15 day', '29 day']
        ref_t = '24 hr' if unit == 'hr' else '4 day'
        for t in times:
            k = f'{name} | time {ref_t} vs {t}'
            out[k] = compare(ds, ds, 'High', ref_t, 'High', t, k, [], a.min_rel)
            summarise(out[k], keys)
        for d in ('Middle', 'Low'):
            k = f'{name} | dose High vs {d} at {ref_t}'
            out[k] = compare(ds, ds, 'High', ref_t, d, ref_t, k, [], a.min_rel)
            summarise(out[k], keys)

    print('\n==================== what this bounds ====================')
    def med(k, stat='gene'):
        fr = [r[stat]['frac'] for r in out[k]['per_compound'] if r.get(stat, {}).get('frac') is not None]
        return float(np.median(fr)) * 100 if fr else float('nan')
    rows = [(k, med(k)) for k in out]
    for k, v in rows:
        print(f'  {v:6.1f}%   {k}')
    json.dump(out, open(a.out, 'w'), indent=1, allow_nan=True)
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
