#!/usr/bin/env python3
"""Response vectors for the selenite time course, on a seven-way yeast ortholog axis.

E-MTAB-7022/7023/7044/7045/7046/7047/7049/7053: eight yeast species, each exposed to
sodium selenite and sampled at 10, 20, 30, 40, 50, 60, 70 and 80 minutes, with a
mock-treated culture co-hybridised on the same two-colour array at every timepoint,
four biological replicates, dye-swapped. One accession per species.

Why this panel is wanted. Every timepoint carries its own mock on its own array, so
two timepoints share no control sample. That is what neither of the other panels can
offer: the blood panel's doses are differenced against a common same-subject control,
and GSE38478's timepoints share their replicate's T0. A dose-split of the blood panel
returned 125% of its attenuation bound, which cannot happen, and that is how the
shared control was found. Here the design removes the problem rather than requiring a
trick to avoid it, and eight timepoints give 28 pairwise comparisons rather than 3.

Lachancea kluyveri is dropped: its array (A-MTAB-645) carries SAKL gene identifiers,
but no annotated proteome exists in Ensembl Fungi, UniProt or NCBI -- three
field-qualified queries and a GenBank flat file with zero CDS features agree -- so it
cannot join a sequence-derived ortholog axis. Seven species remain.

Two file formats, one mapping route. S. cerevisiae and C. glabrata were scanned to
GenePix .gpr and the rest to Agilent Feature Extraction .txt, but in both the data
file carries only an array-internal probe name (Sace02_000A1F9, Yali02_000A76C) and
the array design file is what turns it into a gene. The Agilent SystematicName column
looks like a gene identifier and is not one -- for Y. lipolytica it holds genomic
intervals such as chryali0f:730873-730933, which match nothing and would have left
five of seven species silently empty. ProbeName joins the design file for 51,730 of
51,730 probes. C. albicans is the exception twice over: its design file has no
database-entry column, so the mapping comes from the companion _comments.txt, and it
names genes as orf19.*, which Ensembl does not use, so it is then translated through
the Candida Genome Database feature table.

In both formats channel 635/red is Cy5 and 532/green is Cy3; which dye carried
selenite is read per array from the SDRF, so dye swaps are handled by construction
rather than assumed to cancel.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

SPECIES_ACC = {
    'saccharomyces_cerevisiae': 'E-MTAB-7022', 'candida_glabrata': 'E-MTAB-7023',
    'yarrowia_lipolytica': 'E-MTAB-7044', 'lachancea_thermotolerans': 'E-MTAB-7045',
    'kluyveromyces_lactis': 'E-MTAB-7047', 'debaryomyces_hansenii': 'E-MTAB-7049',
    'candida_albicans': 'E-MTAB-7053',
}
TREATED = 'sodium selenite'
MOCK = 'none'
# Genolevures ids written as PREFIX_BODY by Ensembl for some species and PREFIXBODY by
# the arrays. Narrow on purpose: C. albicans ids like C4_04510W_A must keep their
# underscore, and a general strip would silently merge genes.
GENOLEVURES = re.compile(r'^([A-Z]{4}\d)_([A-Z]\d+[a-z]?)$')


def canonical(gene: str) -> str:
    m = GENOLEVURES.match(gene)
    return f'{m.group(1)}{m.group(2)}' if m else gene


def read_sdrf(path: Path):
    """Array Data File -> {time, cy5_compound, cy3_compound, adf}."""
    rows = [l.split('\t') for l in path.read_text(encoding='utf-8', errors='replace')
            .rstrip('\n').split('\n')]
    hdr = [h.strip() for h in rows[0]]
    idx = lambda p: next((i for i, h in enumerate(hdr) if re.search(p, h, re.I)), None)
    c_file = idx(r'Array\s*Data\s*File')
    c_lab = idx(r'^Label$|^Label\b')
    c_cmp = idx(r'Factor\s*Value\s*\[\s*compound')
    c_time = idx(r'Factor\s*Value\s*\[\s*time')
    c_adf = idx(r'Array\s*Design\s*REF')
    c_dose = idx(r'Factor\s*Value\s*\[\s*dose')
    out = defaultdict(dict)
    for r in rows[1:]:
        if len(r) <= c_file or not r[c_file].strip():
            continue
        f = r[c_file].strip()
        lab = r[c_lab].strip() if c_lab is not None and len(r) > c_lab else ''
        out[f][f'{lab}_compound'] = r[c_cmp].strip() if len(r) > c_cmp else ''
        out[f][f'{lab}_time'] = r[c_time].strip() if len(r) > c_time else ''
        out[f]['adf'] = r[c_adf].strip() if c_adf is not None and len(r) > c_adf else ''
        d = r[c_dose].strip() if c_dose is not None and len(r) > c_dose else ''
        if d:
            out[f]['dose'] = d
    return out


def adf_probe_to_gene(path: Path):
    """Reporter Name -> gene id. Some designs put the mapping in a companion
    _comments.txt instead of a database-entry column in the design file itself."""
    # the design is A-MTAB-642.adf.txt, so the companion is A-MTAB-642_comments.txt --
    # Path.stem leaves ".adf" on and names a file that does not exist
    comments = path.with_name(path.name.split('.')[0] + '_comments.txt')
    if comments.exists():
        rows = [l.split('\t') for l in
                comments.read_text(encoding='utf-8', errors='replace').split('\n')[1:] if l.strip()]
        m = {r[0].strip(): r[1].strip() for r in rows if len(r) > 1 and r[1].strip()}
        if m:
            return m
    lines = path.read_text(encoding='utf-8', errors='replace').split('\n')
    hi = next(i for i, l in enumerate(lines)
              if l.split('\t')[0].strip() in ('Block Column', 'Reporter Name', 'Block'))
    hdr = [h.strip() for h in lines[hi].split('\t')]
    ni = hdr.index('Reporter Name')
    db = [i for i, h in enumerate(hdr) if 'Database Entry' in h]
    if not db:
        return {}
    gi = db[0]
    m = {}
    for l in lines[hi + 1:]:
        if not l.strip():
            continue
        f = l.split('\t')
        if len(f) > max(ni, gi) and f[ni].strip() and f[gi].strip():
            m.setdefault(f[ni].strip(), f[gi].strip())
    return m


def read_gpr(path: Path):
    """probe name, Cy5 and Cy3 background-subtracted medians, from a GenePix table."""
    with path.open(encoding='utf-8', errors='replace') as h:
        for l in h:
            if l.lstrip('"').startswith('Block'):
                cols = [c.strip().strip('"') for c in l.rstrip('\n').split('\t')]
                break
        else:
            raise ValueError(f'{path.name}: no GenePix column header')
        need = ['Name', 'F635 Median', 'B635 Median', 'F532 Median', 'B532 Median']
        ix = {c: cols.index(c) for c in need}
        fl = cols.index('Flags') if 'Flags' in cols else None
        name, r_, g_ = [], [], []
        for l in h:
            f = [c.strip().strip('"') for c in l.rstrip('\n').split('\t')]
            if len(f) < len(cols):
                continue
            if fl is not None:
                try:
                    if float(f[fl]) < 0:      # GenePix marks bad or absent spots negative
                        continue
                except ValueError:
                    continue
            try:
                rv = float(f[ix['F635 Median']]) - float(f[ix['B635 Median']])
                gv = float(f[ix['F532 Median']]) - float(f[ix['B532 Median']])
            except ValueError:
                continue
            name.append(f[ix['Name']]); r_.append(rv); g_.append(gv)
    return np.array(name, dtype=object), np.array(r_), np.array(g_)


def read_agilent(path: Path):
    """probe NAME, Cy5 and Cy3 background-subtracted medians, from Agilent FE.

    ProbeName, not SystematicName: the latter holds genomic intervals on some of
    these designs and joins nothing.
    """
    with path.open(encoding='utf-8', errors='replace') as h:
        cols = None
        for l in h:
            if l.startswith('FEATURES'):
                cols = l.rstrip('\n').split('\t')
                break
        if cols is None:
            raise ValueError(f'{path.name}: no FEATURES header')
        ix = {c: i for i, c in enumerate(cols)}
        need = ['ProbeName', 'ControlType', 'rMedianSignal', 'gMedianSignal',
                'rBGMedianSignal', 'gBGMedianSignal']
        for c in need:
            if c not in ix:
                raise ValueError(f'{path.name}: missing column {c}')
        name, r_, g_ = [], [], []
        for l in h:
            if not l.startswith('DATA'):
                continue
            f = l.rstrip('\n').split('\t')
            if len(f) < len(cols):
                continue
            try:
                if int(float(f[ix['ControlType']])) != 0:   # 0 is a real probe
                    continue
                rv = float(f[ix['rMedianSignal']]) - float(f[ix['rBGMedianSignal']])
                gv = float(f[ix['gMedianSignal']]) - float(f[ix['gBGMedianSignal']])
            except ValueError:
                continue
            name.append(f[ix['ProbeName']]); r_.append(rv); g_.append(gv)
    return np.array(name, dtype=object), np.array(r_), np.array(g_)


def normalise(m, a, frac=0.3):
    """Within-array loess of M on A -- the standard correction for the intensity-
    dependent dye bias that no amount of averaging removes."""
    ok = np.isfinite(m) & np.isfinite(a)
    if ok.sum() < 200:
        return m
    fit = lowess(m[ok], a[ok], frac=frac, it=2, return_sorted=True,
                 delta=0.01 * (np.nanmax(a[ok]) - np.nanmin(a[ok])))
    out = m.copy()
    out[ok] = m[ok] - np.interp(a[ok], fit[:, 0], fit[:, 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--orthologs', required=True)
    ap.add_argument('--calbicans-map', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    data = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    axis = json.loads(Path(args.orthologs).read_text())['axis']
    species = sorted(SPECIES_ACC)
    for s in species:
        if s not in axis[0]:
            raise SystemExit(f'ortholog axis has no column for {s}; it has {sorted(axis[0])}')
    pos = {s: {canonical(row[s]): i for i, row in enumerate(axis)} for s in species}
    orf19 = json.loads(Path(args.calbicans_map).read_text())

    prov, per_array, rows, meta = {}, {}, [], []
    for sp in species:
        acc = SPECIES_ACC[sp]
        d = data / sp
        sdrf = read_sdrf(d / f'{acc}.sdrf.txt')
        adf_map = {}
        adfs = {v.get('adf') for v in sdrf.values() if v.get('adf')}
        for a in adfs:
            p = data / 'adf' / f'{a}.adf.txt'
            if p.exists():
                adf_map.update(adf_probe_to_gene(p))
        by_time = defaultdict(list)
        skipped = []
        for fn, info in sorted(sdrf.items()):
            p = d / fn
            if not p.exists():
                skipped.append(f'{fn}: absent'); continue
            cy5, cy3 = info.get('Cy5_compound', ''), info.get('Cy3_compound', '')
            t5, t3 = info.get('Cy5_time', ''), info.get('Cy3_time', '')
            if t5 != t3:
                skipped.append(f'{fn}: channels disagree on time ({t5} vs {t3})'); continue
            if {cy5, cy3} != {TREATED, MOCK}:
                skipped.append(f'{fn}: channels are {cy5!r}/{cy3!r}'); continue
            try:
                name, red, green = (read_gpr(p) if fn.endswith('.gpr') else read_agilent(p))
            except Exception as exc:                      # noqa: BLE001 - recorded, not raised
                skipped.append(f'{fn}: {type(exc).__name__}: {exc}'); continue
            treated = red if cy5 == TREATED else green
            mock = green if cy5 == TREATED else red
            ok = (treated > 0) & (mock > 0)
            m = np.full(len(name), np.nan)
            a = np.full(len(name), np.nan)
            m[ok] = np.log2(treated[ok] / mock[ok])
            a[ok] = 0.5 * np.log2(treated[ok] * mock[ok])
            m = normalise(m, a)
            genes = [adf_map.get(n, n) for n in name]
            if sp == 'candida_albicans':
                genes = [orf19.get(g, g) for g in genes]
            agg = defaultdict(list)
            for g, v in zip(genes, m):
                if g and np.isfinite(v):
                    agg[canonical(g)].append(v)
            vec = np.full(len(axis), np.nan)
            hit = 0
            for g, vs in agg.items():
                i = pos[sp].get(g)
                if i is not None:
                    vec[i] = float(np.median(vs)); hit += 1
            by_time[t5].append((fn, vec, hit, len(agg)))
        hits = [h for v in by_time.values() for _, _, h, _ in v]
        if not hits or np.median(hits) < 0.2 * len(axis):
            raise SystemExit(
                f'{sp}: arrays reach a median of {int(np.median(hits)) if hits else 0} of '
                f'{len(axis):,} axis genes. That is an identifier-namespace mismatch, not a '
                f'sparse array -- refusing to build a panel with a silently empty species.')
        times = sorted(by_time, key=lambda x: float(x))
        prov[sp] = {'accession': acc, 'arrays_used': sum(len(v) for v in by_time.values()),
                    'timepoints': times, 'arrays_per_timepoint': {t: len(by_time[t]) for t in times},
                    'format': 'GenePix .gpr' if any(f.endswith('.gpr') for f in sdrf) else 'Agilent FE .txt',
                    'axis_genes_hit_median': int(np.median([h for v in by_time.values() for _, _, h, _ in v]))
                                             if by_time else 0,
                    'dose': next((v['dose'] for v in sdrf.values() if v.get('dose')), None),
                    'skipped': skipped}
        print(f'{sp:28s} {acc}  {prov[sp]["format"]:16s} '
              f'{prov[sp]["arrays_used"]:2d} arrays  times {times}  '
              f'axis hit ~{prov[sp]["axis_genes_hit_median"]:,}/{len(axis):,}', flush=True)
        if skipped:
            print(f'    skipped {len(skipped)}: {skipped[:3]}', flush=True)
        for t in times:
            stack = np.vstack([v for _, v, _, _ in by_time[t]])
            per_array[f'{sp}|{t}'] = stack.astype(np.float32)
            with np.errstate(invalid='ignore'):
                rows.append(np.nanmean(stack, axis=0).astype(np.float32))
            meta.append({'species': sp, 'task_id': f't{int(float(t)):02d}',
                         'timepoint_min': t, 'n_arrays': stack.shape[0],
                         'dose_mM': prov[sp]['dose'], 'accession': acc})

    responses = np.vstack(rows)
    symbols = [a['saccharomyces_cerevisiae'] for a in axis]
    np.save(out / 'responses.npy', responses)
    np.save(out / 'genes.npy', np.asarray(symbols))
    np.savez_compressed(out / 'per_array.npz', **per_array)
    with (out / 'response_metadata.tsv').open('w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=list(meta[0]), delimiter='\t')
        w.writeheader(); w.writerows(meta)

    complete = np.isfinite(responses).all(axis=0)
    (out / 'provenance.json').write_text(json.dumps({
        'panel': 'selenite time course in seven yeasts (E-MTAB-7022 and siblings)',
        'species': species, 'ortholog_axis': args.orthologs, 'n_axis_genes': len(axis),
        'n_response_rows': len(meta), 'n_genes_finite_in_every_row': int(complete.sum()),
        'design': 'sodium selenite against a mock culture co-hybridised on the same array at '
                  'every timepoint, four biological replicates, dye-swapped; so two timepoints '
                  'share no control sample',
        'dose_caveat': 'the dose is per species (0.5 to 10 mM), chosen by the depositors to be '
                       'comparably effective rather than identical. The shared axis is time since '
                       'a comparable insult; within a species the dose is constant across '
                       'timepoints, so timepoint comparisons are unaffected',
        'transform': 'background-subtracted median signals, log2(selenite/mock) per probe with '
                     'the dye orientation read per array from the SDRF, within-array loess of M '
                     'on A, median over probes of a gene, mean over replicate arrays',
        'dropped_species': {'lachancea_kluyveri': 'no annotated proteome in Ensembl Fungi, '
                                                  'UniProt or NCBI, so it cannot join a '
                                                  'sequence-derived ortholog axis'},
        'per_species': prov,
    }, indent=2), encoding='utf-8')
    print(f'\nresponse rows {len(meta)}  axis {len(axis):,}  '
          f'finite everywhere {int(complete.sum()):,}')


if __name__ == '__main__':
    main()
