#!/usr/bin/env python3
"""A five-way one-to-one ortholog axis for the hepatocyte stress panel.

GSE330151 ships one raw-count matrix per species, each keyed by that species' own
Ensembl gene ids, so a shared axis has to be built before any cross-species
comparison. Only one-to-one orthologs are admitted, unique in both directions:
many-to-many pairs would let paralogue expansion appear as a species difference.

Human-anchored, because the pairwise maps Ensembl publishes are human-anchored, then
intersected. Every filtering step is counted, because the size of the surviving axis
bounds everything measured on it.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

HOST = 'https://www.ensembl.org/biomart/martservice'
REST = 'https://rest.ensembl.org/info/data?content-type=application/json'
PARTNERS = {'macaca_fascicularis': 'mfascicularis', 'canis_lupus_familiaris': 'clfamiliaris',
            'mus_musculus': 'mmusculus', 'rattus_norvegicus': 'rnorvegicus'}


class Permanent308(urllib.request.HTTPRedirectHandler):
    """Python's urllib does not follow 308, which is what Ensembl answers with."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)


def query(prefix):
    return ('<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
            '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
            'count="" datasetConfigVersion="0.6">'
            '<Dataset name="hsapiens_gene_ensembl" interface="default">'
            '<Attribute name="ensembl_gene_id"/><Attribute name="external_gene_name"/>'
            f'<Attribute name="{prefix}_homolog_ensembl_gene"/>'
            f'<Attribute name="{prefix}_homolog_orthology_type"/>'
            f'<Attribute name="{prefix}_homolog_orthology_confidence"/>'
            '</Dataset></Query>')


def fetch(prefix):
    opener = urllib.request.build_opener(Permanent308)
    url = HOST + '?' + urllib.parse.urlencode({'query': query(prefix)})
    with opener.open(url, timeout=900) as handle:
        body = handle.read().decode('utf-8', errors='replace')
    rows = [l.split('\t') for l in body.splitlines() if l.strip()]
    if len(rows) < 10000:
        raise SystemExit(f'only {len(rows)} rows for {prefix}; the query did not succeed')
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    with urllib.request.urlopen(REST, timeout=90) as h:
        releases = json.loads(h.read().decode())['releases']

    steps, maps, symbols = {}, {}, {}
    for species, prefix in PARTNERS.items():
        rows = fetch(prefix)
        one, dropped_type, no_partner = {}, 0, 0
        for row in rows:
            row = row + [''] * (5 - len(row))
            human, symbol, partner, otype, _conf = (c.strip() for c in row[:5])
            if not human:
                continue
            if symbol:
                symbols.setdefault(human, symbol)
            if not partner:
                no_partner += 1
                continue
            if otype != 'ortholog_one2one':
                dropped_type += 1
                continue
            if human in one and one[human] != partner:
                one[human] = None      # contradictory source rows: drop both
            else:
                one.setdefault(human, partner)
        contradictory = sum(1 for v in one.values() if v is None)
        one = {k: v for k, v in one.items() if v}
        back = {}
        for h, p in one.items():
            back.setdefault(p, []).append(h)
        non_unique = {p for p, hs in back.items() if len(hs) > 1}
        one = {h: p for h, p in one.items() if p not in non_unique}
        steps[species] = {'rows_returned': len(rows), 'human_genes_with_no_partner': no_partner,
                          'dropped_not_one_to_one': dropped_type,
                          'dropped_contradictory_source_rows': contradictory,
                          'dropped_partner_claimed_by_two_human_genes': len(non_unique),
                          'surviving_pairs': len(one)}
        maps[species] = one
        print(f'{species:26s} {len(one):6,} one-to-one pairs', flush=True)

    shared = sorted(set.intersection(*(set(m) for m in maps.values())))
    axis = [dict({'human': h, 'symbol': symbols.get(h, '')},
                 **{s: maps[s][h] for s in PARTNERS}) for h in shared]
    Path(args.output).write_text(json.dumps({
        'source': HOST, 'release_reported_by_rest_api': releases, 'retrieved': '2026-10-20',
        'why': 'GSE330151 ships one count matrix per species keyed by that species own Ensembl '
               'ids, so a shared axis must be built before any cross-species comparison',
        'admission_rule': 'one-to-one orthologs only, unique in both directions',
        'species': ['homo_sapiens'] + list(PARTNERS),
        'per_pair_filtering': steps, 'five_way_axis_size': len(axis), 'axis': axis,
    }, indent=2), encoding='utf-8')
    print(f'\nfive-way axis: {len(axis):,} genes -> {args.output}')


if __name__ == '__main__':
    main()
