#!/usr/bin/env python3
"""A one-to-one ortholog axis across the GSE38478 yeasts, built from sequence.

The study these data come from used the Fungal Orthogroups Repository
(http://www.broadinstitute.org/regev/orthogroups/, Synergy; Wapinski 2007). That
resource now returns 403 and has no live mirror, and YGOB -- the fallback the paper
used for its one missing species -- covers Saccharomycetaceae only, so it does not
span Schizosaccharomyces or Candida. The axis therefore has to be rebuilt rather than
reused.

The rule is the one this project already applies to its mammalian axis: reciprocal
best hits anchored on one reference, kept only when the pairing is unique in BOTH
directions, then intersected across species so every axis position is one gene in
every species. Rebuilding it has an advantage over reusing the lost table: the rule is
identical to the one the mammalian panels use, so an agreement measured between a
yeast panel and a mammalian panel is not partly an artefact of two different
orthology definitions.

Anchor is S. cerevisiae: the arrays are annotated against it, it is the best annotated
of the eight, and anchoring everything on one species is what makes "unique in both
directions" well defined.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def read_fasta_ids(path: Path):
    """protein id -> gene id, from the gene: token in Ensembl pep headers.

    The gene id is taken VERBATIM. A first version stripped a trailing ".N" as though
    it were a version suffix, which is true for Ensembl-style ids but false for
    S. pombe systematic names: gene:SPAC212.11 became SPAC212, collapsing every gene
    on a cosmid into one. 5,146 pombe proteins became 489 genes, the reciprocal hits
    against S. cerevisiae fell to 32, and the intersected axis came out at 24 genes
    instead of thousands. The caller checks the collapse ratio so this class of defect
    announces itself rather than returning a small plausible number.
    """
    out = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as h:
        for line in h:
            if not line.startswith('>'):
                continue
            parts = line[1:].split()
            gene = next((p.split(':', 1)[1] for p in parts if p.startswith('gene:')), None)
            out[parts[0]] = gene or parts[0]
    return out


def best_hits(query: Path, db: Path, out: Path, threads: int, evalue: float):
    """One best hit per query protein, by bitscore."""
    if not out.exists():
        subprocess.run(
            ['diamond', 'blastp', '-q', str(query), '-d', str(db), '-o', str(out),
             '--max-target-seqs', '1', '--evalue', str(evalue), '--threads', str(threads),
             '--outfmt', '6', 'qseqid', 'sseqid', 'bitscore', '--quiet'],
            check=True)
    best = {}
    with out.open() as h:
        for line in h:
            q, s, b = line.rstrip('\n').split('\t')[:3]
            b = float(b)
            if q not in best or b > best[q][1]:
                best[q] = (s, b)
    return {q: s for q, (s, _) in best.items()}


def collapse_to_genes(prot_map, qgenes, sgenes):
    """Protein-level best hits -> gene-level pairs kept only when unambiguous.

    A gene with several protein isoforms can send its isoforms to different targets;
    such a gene is dropped rather than resolved by a rule, because any rule here would
    be a silent judgement call on an axis the whole analysis rests on.
    """
    per = defaultdict(set)
    for q, s in prot_map.items():
        if q in qgenes and s in sgenes:
            per[qgenes[q]].add(sgenes[s])
    return {g: next(iter(t)) for g, t in per.items() if len(t) == 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pep-dir', required=True)
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--anchor', default='saccharomyces_cerevisiae')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--evalue', type=float, default=1e-5)
    ap.add_argument('--only', default='',
                    help='comma-separated species stems to restrict the axis to; the pep '
                         'directory is shared between panels, so an axis must say which '
                         'species it is for rather than take whatever happens to be there')
    args = ap.parse_args()

    pep = Path(args.pep_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    files = {p.name.split('.pep')[0]: p for p in sorted(pep.glob('*.pep.fa.gz'))}
    if args.only:
        want = [x.strip() for x in args.only.split(',') if x.strip()]
        missing = [w for w in want if w not in files]
        if missing:
            raise SystemExit(f'--only names species with no proteome: {missing}')
        files = {k: v for k, v in files.items() if k in want}
    if args.anchor not in files:
        raise SystemExit(f'anchor {args.anchor} not among {sorted(files)}')
    others = [s for s in files if s != args.anchor]
    print(f'anchor {args.anchor}; {len(others)} other species: {others}')

    genes = {}
    for s, p in files.items():
        genes[s] = read_fasta_ids(p)
        n_prot, n_gene = len(genes[s]), len(set(genes[s].values()))
        print(f'  {s:32s} {n_prot:6,} proteins  {n_gene:6,} genes')
        # The guard is on the absolute count, not the collapse ratio. A ratio test was
        # written for yeasts, where proteins and genes are nearly one to one, and it
        # refused the human proteome, where 382,428 isoforms collapsing to 23,879 genes
        # is simply correct. No eukaryotic reference proteome has fewer than about a
        # thousand protein-coding genes, so that is what a parsing failure looks like:
        # the S. pombe bug this guard exists for turned 5,146 proteins into 489 "genes".
        if n_gene < 1000:
            raise SystemExit(f'{s}: {n_prot:,} proteins yielded only {n_gene:,} distinct '
                             'gene ids. No eukaryotic proteome is that small -- this is a '
                             'header-parsing defect, not biology.')
        db = work / f'{s}.dmnd'
        if not db.exists():
            subprocess.run(['diamond', 'makedb', '--in', str(p), '-d', str(db),
                            '--threads', str(args.threads), '--quiet'], check=True)

    pairs, stats = {}, {}
    for s in others:
        fwd = best_hits(files[args.anchor], work / f'{s}.dmnd',
                        work / f'{args.anchor}__to__{s}.tsv', args.threads, args.evalue)
        rev = best_hits(files[s], work / f'{args.anchor}.dmnd',
                        work / f'{s}__to__{args.anchor}.tsv', args.threads, args.evalue)
        a2b = collapse_to_genes(fwd, genes[args.anchor], genes[s])
        b2a = collapse_to_genes(rev, genes[s], genes[args.anchor])
        rbh = {a: b for a, b in a2b.items() if b2a.get(b) == a}
        # unique in both directions: a target claimed by two anchors is not one-to-one
        seen = defaultdict(list)
        for a, b in rbh.items():
            seen[b].append(a)
        rbh = {a: b for a, b in rbh.items() if len(seen[b]) == 1}
        pairs[s] = rbh
        stats[s] = {'anchor_best_hits': len(fwd), 'target_best_hits': len(rev),
                    'gene_level_forward': len(a2b), 'gene_level_reverse': len(b2a),
                    'reciprocal_one_to_one': len(rbh)}
        print(f'  {args.anchor} <-> {s:32s} one-to-one {len(rbh):5,}')

    core = set.intersection(*(set(p) for p in pairs.values())) if pairs else set()
    axis = []
    for a in sorted(core):
        row = {args.anchor: a}
        row.update({s: pairs[s][a] for s in others})
        axis.append(row)
    payload = {
        'what': 'one-to-one ortholog axis across the GSE38478 yeasts, from reciprocal best hits',
        'why_rebuilt': 'the Fungal Orthogroups Repository the source study used returns 403 and '
                       'has no live mirror; YGOB does not span Schizosaccharomyces or Candida',
        'rule': 'DIAMOND blastp best hit each way, collapsed to genes, kept only when the gene '
                'pairing is unambiguous in both directions and the target is claimed by exactly '
                'one anchor gene; then intersected across all species',
        'anchor': args.anchor, 'species': sorted(files), 'evalue': args.evalue,
        'per_species': stats, 'n_axis_genes': len(axis), 'axis': axis,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\naxis: {len(axis):,} genes present one-to-one in all {len(files)} species')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
