#!/usr/bin/env python3
"""Put a TG-GATEs expression matrix onto the human-rat one-to-one ortholog axis.

Three things here are easy to get wrong and each one is checked rather than assumed:

* The axis stores versioned Ensembl ids (ENSG00000000003.18) while the array annotation
  stores bare ones (ENSG00000000003). Stripping a trailing ".N" is only safe for ids that
  look like Ensembl gene accessions -- the same careless strip turned S. pombe's
  SPAC212.11 into SPAC212 and collapsed that proteome, so the strip is guarded by shape.
* 43% of the matrix is the literal token NaN (probes MAS5 called absent). A non-finite
  value poisons comparisons, so every read goes through a finite check; two stray NaNs
  were enough to make a sorted 6,765-element array report a median below its own first
  quartile.
* Probes matching more than one gene on the selected axis are dropped. A probe with
  one on-axis hit can still have off-axis annotations, so this is not proof of unique
  genomic specificity.

Output is gene x condition x replicate, where a condition is (compound, dose level, time).
The matrix is already expressed as log2 ratio to the matched control, which was verified
by the control arms sitting at median +0.002 with 0.23% of values beyond |1|, against
4.02% in the high-dose arms.
"""
import argparse, collections, csv, json, math, os, re, sys
import numpy as np

ENS_VERSIONED = re.compile(r'^(ENS[A-Z]*G\d+)\.\d+$')
MISSING = {'', '---', 'NA', 'na', 'null'}


def bare(gene_id):
    """Drop an Ensembl version suffix, and only an Ensembl version suffix."""
    m = ENS_VERSIONED.match(gene_id.strip())
    return m.group(1) if m else gene_id.strip()


def load_axis(path, species):
    doc = json.load(open(path))
    pairs = doc['axis']
    genes, seen = [], set()
    for p in pairs:
        g = bare(p[species])
        if g in seen:                       # a versioned pair that collapses is a defect
            raise SystemExit(f'{g} appears twice on the axis after version stripping')
        seen.add(g)
        genes.append(g)
    return genes


def load_probe_map(path, axis_set):
    """probe id -> gene, keeping exactly one on-axis hit (off-axis hits may exist)."""
    rd = csv.DictReader(open(path, encoding='utf-8', errors='replace'), delimiter='\t')
    id_col = rd.fieldnames[0]
    ens_col = next(c for c in rd.fieldnames if 'ensembl' in c.lower())
    out, stat = {}, collections.Counter()
    for r in rd:
        raw = (r.get(ens_col) or '').strip()
        if raw in MISSING:
            stat['no_ensembl'] += 1
            continue
        ids = {bare(x) for x in re.split(r'[\s:,;|]+', raw) if x.strip() not in MISSING}
        hits = sorted(ids & axis_set)
        if not hits:
            stat['off_axis'] += 1
        elif len(hits) > 1:
            stat['ambiguous_dropped'] += 1
        else:
            out[r[id_col].strip()] = hits[0]
            stat['kept'] += 1
    return out, stat


def load_conditions(path):
    """array id -> (compound, dose level, time); replicate index assigned by sorted id."""
    rows = list(csv.DictReader(open(path, encoding='utf-8', errors='replace'), delimiter='\t'))
    hyb_col = rows[0].keys().__iter__().__next__()
    cells = collections.defaultdict(list)
    for r in rows:
        key = (r['COMPOUND_NAME'].strip(), r['DOSE_LEVEL'].strip(), r['SACRI_PERIOD'].strip())
        cells[key].append(r[hyb_col].strip())
    arr2cell = {}
    for key, ids in cells.items():
        for rep, aid in enumerate(sorted(ids)):
            arr2cell[aid] = (key, rep)
    return arr2cell, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', required=True)
    ap.add_argument('--probes', required=True)
    ap.add_argument('--exp', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--species', required=True, help='key used inside the axis file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-reps', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true', help='report join rates and stop')
    a = ap.parse_args()

    genes = load_axis(a.axis, a.species)
    gene_ix = {g: i for i, g in enumerate(genes)}
    probe2gene, pstat = load_probe_map(a.probes, set(genes))
    print(f'axis genes           {len(genes):,}')
    print(f'probe annotation     {dict(pstat)}')
    print(f'axis genes with >=1 probe  {len({g for g in probe2gene.values()}):,} '
          f'({len({g for g in probe2gene.values()})/len(genes):.1%})')

    arr2cell, cells = load_conditions(a.exp)
    print(f'arrays annotated     {len(arr2cell):,} in {len(cells):,} conditions')

    with open(a.matrix, encoding='utf-8', errors='replace') as h:
        while True:
            line = h.readline()
            if not line:
                raise SystemExit('matrix ended before a header row')
            if not line.startswith('#'):
                header = line.rstrip('\n').split('\t')
                break
        col_ids = [c.strip() for c in header[1:]]
        matched = sum(1 for c in col_ids if c in arr2cell)
        print(f'matrix columns       {len(col_ids):,}, annotated {matched:,} '
              f'({matched/len(col_ids):.1%})')
        if matched < 0.95 * len(col_ids):
            raise SystemExit('fewer than 95% of matrix columns carry an annotation -- '
                             'the array id join is wrong, not merely incomplete')
        if a.dry_run:
            probe_col = []
            for i, line in enumerate(h):
                if i >= 5:
                    break
                probe_col.append(line.split('\t', 1)[0].strip())
            hit = sum(1 for p in probe_col if p in probe2gene)
            print(f'first 5 matrix row ids {probe_col} -> {hit} in the probe map')
            return

        cell_list = sorted(cells)
        cell_ix = {c: i for i, c in enumerate(cell_list)}
        n_g, n_c, n_r = len(genes), len(cell_list), a.max_reps
        total = np.zeros((n_g, n_c, n_r), dtype=np.float64)
        count = np.zeros((n_g, n_c, n_r), dtype=np.int32)

        col_gene_slot = []
        for c in col_ids:
            if c in arr2cell:
                (key, rep) = arr2cell[c]
                col_gene_slot.append((cell_ix[key], rep) if rep < n_r else None)
            else:
                col_gene_slot.append(None)
        keep_cols = np.array([s is not None for s in col_gene_slot])
        ci = np.array([s[0] if s else 0 for s in col_gene_slot])
        ri = np.array([s[1] if s else 0 for s in col_gene_slot])

        n_rows = n_used = 0
        for line in h:
            n_rows += 1
            cut = line.find('\t')
            probe = line[:cut].strip()
            g = probe2gene.get(probe)
            if g is None:
                continue
            # np.fromstring with sep is deprecated and returns a short array on a
            # parse error instead of raising; an explicit cast handles the NaN token
            vals = np.array(line[cut + 1:].rstrip('\n').split('\t'), dtype=np.float64)
            if vals.size != len(col_ids):
                raise SystemExit(f'row {probe} has {vals.size} values, expected {len(col_ids)}')
            ok = np.isfinite(vals) & keep_cols
            if not ok.any():
                continue
            n_used += 1
            gi = gene_ix[g]
            np.add.at(total[gi], (ci[ok], ri[ok]), vals[ok])
            np.add.at(count[gi], (ci[ok], ri[ok]), 1)
            if n_rows % 10000 == 0:
                print(f'  {n_rows:,} rows, {n_used:,} on axis', flush=True)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    print(f'rows read {n_rows:,}; rows on axis {n_used:,}')
    print(f'cells with data: {np.isfinite(mean).mean():.1%} of {mean.size:,} gene-cell-rep slots')
    per_gene = np.isfinite(mean).reshape(n_g, -1).mean(axis=1)
    print(f'genes with >=50% slots observed: {(per_gene >= .5).sum():,}/{n_g:,}')

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(
        a.out, values=mean.astype(np.float32), n_probes=count.astype(np.int16),
        genes=np.array(genes), compounds=np.array([c[0] for c in cell_list]),
        doses=np.array([c[1] for c in cell_list]), times=np.array([c[2] for c in cell_list]),
        species=a.species)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
