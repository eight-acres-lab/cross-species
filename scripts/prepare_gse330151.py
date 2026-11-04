#!/usr/bin/env python3
"""Response vectors for the hepatocyte stress panel, on a five-way ortholog axis.

GSE330151: primary hepatocytes from human, cynomolgus macaque, dog, mouse and rat,
treated with three compounds across two stress pathways -- bardoxolone (CDDO-Me) and
Ki696 on NRF2, thapsigargin on the unfolded protein response -- each against its own
vehicle control at its own matched incubation. One raw-count matrix per species,
keyed by that species' own Ensembl ids.

Which vehicle goes with which treatment is read out of the GEO sample titles rather
than inferred from the library names. The library names do not say: the three vehicle
arms are called CTRL1, CTRL2 and CTRL, and the numbering does not follow the
treatments' order in the file. The titles say it outright -- "Vehicle control,
matched to CDDO-Me treatment" -- and the series matrix also carries the incubation,
so the pairing can be checked against it. The first version of this script hardcoded
the pairing from the library names and got all three arms wrong: CDDO (24 h) was
differenced against the thapsigargin vehicle (8 h), thapsigargin (8 h) against the
CDDO vehicle (24 h), and Ki696 (24 h) against the 8 h vehicle too. Primary
hepatocytes dedifferentiate quickly in culture, so that subtracted an 8-versus-24-hour
drift into every response vector; it showed up as HMOX1 falling under an NRF2
activator, which is how the error was caught.

The titles also carry a donor index (n=1, n=2, n=3) that survives the gaps in the
library numbering -- human CTRL1_Human_1/2/4 and Thap_Human_1/2/4 are n=1/2/3 of the
same three donors. Differencing is therefore done per donor and then averaged, which
removes the donor effect instead of leaving it in the arm means, and leaves three
paired estimates per contrast from which single-donor reliability can be read.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SPECIES_FILE = {'Homo sapiens': 'human', 'Macaca fascicularis': 'cyno',
                'Canis lupus familiaris': 'dog', 'Mus musculus': 'mouse',
                'Rattus norvegicus': 'rat'}
AXIS_KEY = {'Homo sapiens': 'human', 'Macaca fascicularis': 'macaca_fascicularis',
            'Canis lupus familiaris': 'canis_lupus_familiaris',
            'Mus musculus': 'mus_musculus', 'Rattus norvegicus': 'rattus_norvegicus'}
# canonical agent name -> the token to use as the task id, so the three compounds get
# stable names independent of how each species' libraries were spelled
AGENTS = {'cddo-me': 'CDDO', 'thapsigargin': 'THAP', 'ki696': 'KI696'}


def read_matrix(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        samples = [h.strip('"') for h in header[1:]]
        genes, rows = [], []
        for row in reader:
            if not row:
                continue
            genes.append(row[0].strip('"').split('.')[0])
            rows.append([float(x) if x not in ('', 'NA') else 0.0 for x in row[1:]])
    return samples, genes, np.asarray(rows, dtype=np.float64)


def series_fields(path: Path):
    """field name -> list of per-sample values, from a GEO series matrix header."""
    out = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if not line.startswith('!Sample_'):
                continue
            parts = line.rstrip('\n').split('\t')
            out[parts[0][1:]].append([p.strip('"') for p in parts[1:]])
    return out


def parse_series(path: Path):
    """One series matrix -> (organism, [{library, agent, donor, incubation}]).

    A vehicle sample is identified by its title naming the treatment it is matched
    to; a treated sample by its title naming the agent directly. Anything whose title
    matches neither pattern is returned with agent None and is dropped by the caller
    rather than guessed at.
    """
    f = series_fields(path)
    organism = f['Sample_organism_ch1'][0][0]
    titles = f['Sample_title'][0]
    libs = [re.sub(r'^Library name:\s*', '', v)
            for v in next(r for r in f['Sample_description'] if 'Library name' in r[0])]
    treat = next((r for r in f['Sample_characteristics_ch1']
                  if r and r[0].startswith('treatment:')), [''] * len(titles))
    samples = []
    for title, lib, tr in zip(titles, libs, treat):
        low = title.lower()
        donor = re.search(r'n\s*=\s*(\d+)', title)
        hours = re.search(r'(\d+(?:\.\d+)?)\s*h\b', tr)
        agent, role = None, None
        if 'vehicle control' in low:
            m = re.search(r'matched to\s+(.+?)\s+treatment', low)
            if m:
                agent, role = AGENTS.get(m.group(1).strip()), 'vehicle'
        else:
            for name, token in AGENTS.items():
                if name in low:
                    agent, role = token, 'treated'
                    break
        samples.append({'library': lib, 'agent': agent, 'role': role,
                        'donor': int(donor.group(1)) if donor else None,
                        'incubation_h': float(hours.group(1)) if hours else None,
                        'title': title})
    return organism, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--orthologs', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--min-cpm', type=float, default=1.0)
    ap.add_argument('--pseudocount', type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    axis = json.loads(Path(args.orthologs).read_text())['axis']
    symbols = [a['symbol'] or a['human'] for a in axis]

    # the design, read from GEO rather than assumed
    design = {}
    for p in sorted(data.glob('GSE330151*series_matrix.txt.gz')):
        organism, samples = parse_series(p)
        design[organism] = samples
    missing_org = set(SPECIES_FILE) - set(design)
    if missing_org:
        raise SystemExit(f'no series matrix for {sorted(missing_org)}')

    print('design read from the GEO series matrices:')
    for organism in SPECIES_FILE:
        pairs = defaultdict(dict)
        for s in design[organism]:
            if s['agent']:
                pairs[s['agent']][s['role']] = s['incubation_h']
        line = '  '.join(f"{a} {v.get('treated')}h vs vehicle {v.get('vehicle')}h"
                         for a, v in sorted(pairs.items()))
        unassigned = [s['library'] for s in design[organism] if not s['agent']]
        print(f'  {organism:26s} {line}' + (f'  UNASSIGNED {unassigned}' if unassigned else ''))

    per_species, prov = {}, {}
    for species, tag in SPECIES_FILE.items():
        hits = sorted(data.glob(f'*_{tag}_expression_matrix*raw_counts.csv.gz'))
        if not hits:
            raise SystemExit(f'no matrix for {species} ({tag}) in {data}')
        samples, genes, counts = read_matrix(hits[0])
        key = AXIS_KEY[species]
        wanted = [a[key].split('.')[0] for a in axis]
        index = {}
        for i, g in enumerate(genes):
            index.setdefault(g, i)      # first occurrence; duplicates are rare and dropped
        rowsel = np.array([index.get(g, -1) for g in wanted])
        present = rowsel >= 0
        # CPM on the species' own full transcriptome, then restricted to the axis:
        # normalising after restriction would make the denominator depend on how many
        # orthologs that species happens to have
        libsize = counts.sum(axis=0)
        cpm = counts / np.maximum(libsize, 1) * 1e6
        mat = np.full((len(axis), len(samples)), np.nan)
        mat[present] = cpm[rowsel[present]]

        bylib = {s: j for j, s in enumerate(samples)}
        unmatched = [s['library'] for s in design[species] if s['library'] not in bylib]
        if unmatched:
            raise SystemExit(f'{species}: series-matrix libraries absent from the count '
                             f'matrix: {unmatched}')
        per_species[species] = {'mat': mat, 'bylib': bylib, 'present': present}
        prov[species] = {'file': hits[0].name, 'n_samples': len(samples),
                         'n_genes_in_file': len(genes),
                         'axis_genes_found': int(present.sum()),
                         'axis_genes_missing': int((~present).sum()),
                         'median_library_size': float(np.median(libsize))}
        print(f'{species:26s} {len(samples):3d} samples  axis {int(present.sum()):6,}/{len(axis):,}')

    rows, armmean, meta, skipped = [], [], [], []
    donorwise = {}
    for species in SPECIES_FILE:
        mat = per_species[species]['mat']
        bylib = per_species[species]['bylib']
        grid = defaultdict(dict)          # agent -> donor -> role -> column
        incub = defaultdict(dict)
        for s in design[species]:
            if not s['agent'] or s['donor'] is None:
                continue
            grid[s['agent']].setdefault(s['donor'], {})[s['role']] = bylib[s['library']]
            incub[s['agent']][s['role']] = s['incubation_h']
        for agent, donors in sorted(grid.items()):
            paired = {d: v for d, v in donors.items() if 'treated' in v and 'vehicle' in v}
            if not paired:
                skipped.append(f'{species}:{agent} has no donor with both arms')
                continue
            if incub[agent].get('treated') != incub[agent].get('vehicle'):
                skipped.append(f'{species}:{agent} incubation mismatch '
                               f'{incub[agent]}; refusing to difference across timepoints')
                continue
            tcols = [v['treated'] for _, v in sorted(paired.items())]
            ccols = [v['vehicle'] for _, v in sorted(paired.items())]
            # detection filter on this contrast's own vehicle arm, so a gene silent in
            # this species' hepatocytes cannot contribute a response
            ctrl_mean = np.nanmean(mat[:, ccols], axis=1)
            keep = (ctrl_mean >= args.min_cpm) & np.isfinite(ctrl_mean)
            # per-donor log ratios, then averaged: the donor effect cancels within each
            # pair instead of being carried by the arm means
            lp = np.log2(mat[:, tcols] + args.pseudocount)
            lc = np.log2(mat[:, ccols] + args.pseudocount)
            diff = lp - lc
            diff[~keep] = np.nan
            resp = np.nanmean(diff, axis=1)
            resp[~keep] = np.nan
            rows.append(resp.astype(np.float32))
            # the same contrast under the blood panel's arithmetic -- arm means first,
            # then one log ratio -- so a comparison between the two panels is not
            # confounded by which differencing scheme each one received
            treat_mean = np.nanmean(mat[:, tcols], axis=1)
            alt = np.full(len(mat), np.nan)
            ok = keep & np.isfinite(treat_mean)
            alt[ok] = (np.log2(treat_mean[ok] + args.pseudocount)
                       - np.log2(ctrl_mean[ok] + args.pseudocount))
            armmean.append(alt.astype(np.float32))
            donorwise[f'{species}|{agent}'] = diff.astype(np.float32)
            meta.append({'species': species, 'task_id': agent, 'treatment': agent,
                         'control': 'matched vehicle', 'subject': f'{species}_donor_mean',
                         'n_donor_pairs': len(paired),
                         'incubation_h': incub[agent].get('treated'),
                         'n_genes_passing_detection': int(keep.sum())})

    responses = np.vstack(rows)
    complete = np.isfinite(responses).all(axis=0)
    np.save(out / 'responses.npy', responses)
    np.save(out / 'genes.npy', np.asarray(symbols))
    np.save(out / 'responses_armmean.npy', np.vstack(armmean))
    np.savez_compressed(out / 'donorwise.npz', **donorwise)
    with (out / 'response_metadata.tsv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=list(meta[0]), delimiter='\t')
        w.writeheader()
        w.writerows(meta)

    idx = {s: i for i, s in enumerate(symbols)}
    sanity = {}
    for g in ('NQO1', 'HMOX1', 'GCLM', 'TXNRD1', 'SRXN1',        # NRF2 targets
              'HSPA5', 'DDIT3', 'ATF3', 'XBP1', 'HERPUD1',       # UPR targets
              'ACTB', 'GAPDH', 'TBP'):                           # housekeeping
        i = idx.get(g)
        if i is None:
            sanity[g] = None
            continue
        per = defaultdict(dict)
        for r, m in zip(responses, meta):
            per[m['treatment']][m['species']] = float(r[i])
        sanity[g] = dict(per)

    (out / 'provenance.json').write_text(json.dumps({
        'panel': 'GSE330151, primary hepatocytes, NRF2 and UPR stress',
        'species': list(SPECIES_FILE), 'ortholog_axis': args.orthologs,
        'n_axis_genes': len(axis), 'n_response_rows': len(meta),
        'n_genes_finite_in_every_row': int(complete.sum()),
        'transform': 'counts to CPM on the full transcriptome, then restricted to the axis, then '
                     'per-donor log2(CPM + pseudocount) treated minus its own matched vehicle, '
                     'averaged over donors',
        'control_matching': 'read from the GEO sample titles ("Vehicle control, matched to X '
                            'treatment"), and the incubation in Sample_characteristics is checked '
                            'to agree before any contrast is formed',
        'design_as_read': {sp: sorted({(s['agent'], s['role'], s['incubation_h'])
                                       for s in design[sp] if s['agent']})
                           for sp in SPECIES_FILE},
        'detection_filter': f'matched-vehicle mean CPM >= {args.min_cpm}',
        'responses_armmean': 'the same contrasts computed the way the blood panel was -- arm '
                             'means then one log ratio -- so the two panels can be compared '
                             'under identical arithmetic',
        'pseudocount': args.pseudocount,
        'skipped': skipped, 'per_species': prov,
        'sanity_pathway_and_housekeeping_genes': sanity,
    }, indent=2), encoding='utf-8')

    print(f'\nresponse rows {len(meta)}  axis {len(axis):,}  '
          f'finite everywhere {int(complete.sum()):,}')
    if skipped:
        print('skipped:', skipped)
    for g, v in sanity.items():
        if v:
            line = '  '.join(f'{t}:' + '/'.join(f'{v[t][s]:+.1f}' for s in sorted(v[t]))
                             for t in sorted(v))
            print(f'  {g:8s} {line}')


if __name__ == '__main__':
    main()
