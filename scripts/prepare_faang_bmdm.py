#!/usr/bin/env python3
"""Prepare the FAANG/Roslin BMDM panel: nine mammals, LPS, one tissue.

Source: PRJEB22535 and siblings, harmonised to one-to-one orthologs and published
as Supplementary Table S3 of PMC7386301 (Frontiers in Cell and Developmental
Biology 8:661). The table gives TPM at 0 h and at the stimulated timepoint for
each of nine species on a shared gene axis, together with the authors' own
log2 fold change.

The response vector is computed here from the TPM columns rather than taken from
the published fold-change column, so that the transform is the same log2(TPM + 1)
difference used for GSE249010 and so that the pseudocount can be varied. The
published column is then used as an independent check that the recomputation is
right: any gene where the two disagree beyond rounding is a parsing error.

Two heterogeneities in the source are recorded rather than smoothed over: the two
Mus species are stimulated with Kdo2-lipid A for 6 h while the other seven get LPS
for 7 h, and the species are not all annotated for every gene.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openpyxl

HEADER_NAME_ROW, HEADER_SPECIES_ROW, HEADER_TIME_ROW = 9, 10, 11
FIRST_DATA_ROW = 19

LABEL = {
    'sheep': 'Ovis aries', 'buffalo': 'Bubalus bubalis', 'cow': 'Bos taurus',
    'goat': 'Capra hircus', 'horse': 'Equus caballus',
    'mouse (M. musculus)': 'Mus musculus', 'mouse (M. spretus)': 'Mus spretus',
    'pig': 'Sus scrofa', 'rat': 'Rattus norvegicus',
}
KDO2 = {'Mus musculus', 'Mus spretus'}


def num(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in {'NA', '', 'null', '-'}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--pseudocount', type=float, default=1.0)
    ap.add_argument('--max-published-disagreement', type=float, default=0.02,
                    help='the published log2FC is rounded to 3 decimals, so exact equality is not '
                         'expected; anything beyond this is a parsing error')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.load_workbook(args.table, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [r for r in ws.iter_rows(values_only=True)]
    wb.close()

    names, spp, times = rows[HEADER_NAME_ROW], rows[HEADER_SPECIES_ROW], rows[HEADER_TIME_ROW]

    # The three header rows are merged: the species row labels TPM pairs and then
    # single columns for fold change and rank. Which block a column belongs to is
    # decided by the names row, which marks each block's first column.
    block_start = {j: str(v) for j, v in enumerate(names) if v is not None}
    tpm_start = next(j for j, v in block_start.items() if v.startswith('Expression level'))
    fc_start = next(j for j, v in block_start.items() if v.startswith('log2 fold change'))
    rank_start = next(j for j, v in block_start.items() if v.startswith('Rank'))

    tpm_cols: dict[str, dict[str, int]] = {}
    current = None
    for j in range(tpm_start, fc_start):
        if spp[j] is not None:
            current = LABEL[str(spp[j])]
            tpm_cols[current] = {}
        tpm_cols[current][str(times[j])] = j
    published_fc = {LABEL[str(spp[j])]: j for j in range(fc_start, rank_start)
                    if spp[j] is not None}

    species = sorted(tpm_cols)
    if sorted(published_fc) != species:
        raise SystemExit('the TPM and fold-change blocks disagree on the species list')
    for s, cols in tpm_cols.items():
        if len(cols) != 2 or '0h' not in cols:
            raise SystemExit(f'{s} does not have a 0 h control and one stimulated timepoint: {cols}')

    # Every gene is read for every species first; which genes survive depends on the
    # species set required, and that choice is consequential enough to measure.
    all_names, all_vals, all_pub = [], [], []
    for r in rows[FIRST_DATA_ROW:]:
        if r[0] is None:
            continue
        vals, pvals = [], []
        for s in species:
            cols = tpm_cols[s]
            t_stim = next(t for t in cols if t != '0h')
            c, x = num(r[cols['0h']]), num(r[cols[t_stim]])
            vals.append(np.log2(x + args.pseudocount) - np.log2(c + args.pseudocount)
                        if np.isfinite(c) and np.isfinite(x) else np.nan)
            pvals.append(num(r[published_fc[s]]))
        all_names.append(str(r[0]))
        all_vals.append(vals)
        all_pub.append(pvals)

    ALL = np.asarray(all_vals, dtype=np.float32)
    finite = np.isfinite(ALL)
    # Which species is costing the gene axis? Requiring complete coverage is only
    # defensible once it is clear what it removes and whether one species drives it.
    per_species_missing = {s: int((~finite[:, j]).sum()) for j, s in enumerate(species)}
    complete_all = finite.all(axis=1)
    cost_of_each_species = {}
    for j, s in enumerate(species):
        keep = [i for i in range(len(species)) if i != j]
        cost_of_each_species[s] = int(finite[:, keep].all(axis=1).sum() - complete_all.sum())

    keep_idx = np.flatnonzero(complete_all)
    gene_names = [all_names[i] for i in keep_idx]
    resp = [all_vals[i] for i in keep_idx]
    pub = [all_pub[i] for i in keep_idx]
    dropped_incomplete = int(len(all_names) - len(gene_names))

    R = np.asarray(resp, dtype=np.float32)           # genes x species
    P = np.asarray(pub, dtype=np.float32)
    ok = np.isfinite(P)
    disagreement = float(np.nanmax(np.abs(R[ok] - P[ok]))) if ok.any() else float('nan')
    if not (disagreement <= args.max_published_disagreement):
        raise SystemExit(
            f'recomputed log2FC disagrees with the published column by up to {disagreement:.4f}; '
            'the TPM columns are not being read as intended')

    # Downstream code is sample x gene with one row per (species, task).
    responses = R.T.astype(np.float32)
    meta = [{'species': s, 'task_id': 'LPS|BMDM',
             'subject': 'published_species_mean',
             'stimulus': 'Kdo2-lipid A' if s in KDO2 else 'LPS',
             'time_h': '6h' if s in KDO2 else '7h'} for s in species]

    np.save(out / 'responses.npy', responses)
    np.save(out / 'genes.npy', np.asarray(gene_names))
    with (out / 'response_metadata.tsv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0]), delimiter='\t')
        w.writeheader(); w.writerows(meta)

    # Biological sanity: the classic LPS-induced genes must go up in most species
    # and the housekeepers must stay flat. This is a check on the whole pipeline,
    # not a result.
    idx = {g: i for i, g in enumerate(gene_names)}
    def row(g):
        i = idx.get(g)
        return None if i is None else {s: float(R[i, j]) for j, s in enumerate(species)}
    induced = {g: row(g) for g in ('IL1B', 'IL6', 'CXCL8', 'SOD2', 'CCL5', 'NFKB1')}
    flat = {g: row(g) for g in ('ACTB', 'GAPDH', 'RPL13A')}

    # ACTB, GAPDH and RPL13A are annotated in eight of the nine species and absent
    # only in Mus spretus, so the all-nine axis drops exactly the genes that make the
    # negative control. Evaluate it on the axis where they exist instead of losing it.
    worst = max(per_species_missing, key=per_species_missing.get)
    kept8 = [j for j, s_ in enumerate(species) if s_ != worst]
    species8 = [species[j] for j in kept8]
    complete8 = finite[:, kept8].all(axis=1)
    idx8 = {all_names[i]: i for i in np.flatnonzero(complete8)}
    def row8(g):
        i = idx8.get(g)
        return None if i is None else {species[j]: float(ALL[i, j]) for j in kept8}
    flat8 = {g: row8(g) for g in ('ACTB', 'GAPDH', 'RPL13A')}

    prov = {
        'panel': 'FAANG/Roslin bone-marrow-derived macrophages',
        'accession': 'PRJEB22535 and sibling projects',
        'harmonised_table': 'PMC7386301 Supplementary Table S3',
        'journal_reference': 'Frontiers in Cell and Developmental Biology 8:661',
        'retrieved': '2026-10-05 from the Europe PMC supplementaryFiles endpoint',
        'species': species, 'n_species': len(species),
        'n_genes_complete_in_all_species': len(gene_names),
        'n_genes_dropped_for_incomplete_species_coverage': dropped_incomplete,
        'pseudocount': args.pseudocount,
        'transform': 'log2(TPM + pseudocount) stimulated minus the matched 0 h control',
        'max_disagreement_with_published_log2fc': disagreement,
        'published_log2fc_rounding': '3 decimals, which bounds the agreement achievable',
        'n_tasks': 1,
        'single_task_consequence':
            'one response vector per species, so nothing here can be resampled over tasks and the '
            'residual-reproducibility arm cannot be run on this panel',
        'stimulus_heterogeneity':
            'seven species receive LPS for 7 h; Mus musculus and Mus spretus receive Kdo2-lipid A '
            '(a defined TLR4 agonist) for 6 h. Both rodent rows therefore differ from the other '
            'seven in both agonist and timing, which is a confound specific to them',
        'sanity_induced_genes_log2fc': induced,
        'sanity_housekeeping_genes_log2fc_all_nine_axis': flat,
        'gene_axis_cost_by_species': {
            'per_species_genes_with_a_missing_value': per_species_missing,
            'genes_regained_by_dropping_that_species': cost_of_each_species,
            'species_costing_the_most': worst,
            'note': 'the all-nine axis is the primary arm because more species is the point of this '
                    'panel, but one species sets the cost and the eight-species axis is reported '
                    'alongside so the difference is visible rather than implicit',
        },
        'eight_species_axis': {
            'species': species8,
            'n_genes': int(complete8.sum()),
            'why': f'{worst} is annotated from a de-novo Mouse Genomes Project assembly, so it '
                   'is missing many genes the other eight have, including every housekeeping gene '
                   'used as the negative control',
            'sanity_housekeeping_genes_log2fc': flat8,
        },
    }
    (out / 'provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf-8')

    print(f'species : {len(species)}  {species}')
    print(f'genes   : {len(gene_names)} complete, {dropped_incomplete} dropped for incompleteness')
    print(f'max |recomputed - published| log2FC : {disagreement:.5f}')
    for g, v in induced.items():
        if v:
            print(f'  {g:7s} {min(v.values()):+.2f} .. {max(v.values()):+.2f}')
    print(f'gene axis cost: {worst} alone removes '
          f'{cost_of_each_species[worst]} genes that the other eight share')
    print(f'eight-species axis ({len(species8)} species): {int(complete8.sum())} genes')
    for g, v in flat8.items():
        if v:
            print(f'  {g:7s} {min(v.values()):+.2f} .. {max(v.values()):+.2f}  '
                  f'(expected flat; eight-species axis)')


if __name__ == '__main__':
    main()
