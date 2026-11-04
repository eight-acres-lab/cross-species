#!/usr/bin/env python3
"""Ensembl gene id to HGNC symbol, for matching panels that use different gene keys.

GSE249010's processed matrix is keyed by human Ensembl gene id; the FAANG BMDM
harmonised table is keyed by gene symbol. Matching them needs a map, and the map
has to be independent of either panel: taking it from another panel's gene list
would restrict the matched axis to genes that panel happened to express, which
biases a cross-panel comparison toward whatever that panel was measuring.

The release 109 archive host used elsewhere in this project refuses TLS from this
machine, so the map comes from the current release and the release number is read
back from the REST API and recorded rather than assumed. A symbol map is far less
release-sensitive than a quantification index: symbols are renamed occasionally, and
any id whose symbol changed simply fails to match, which costs coverage rather than
introducing a wrong pairing.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

HOST = 'https://www.ensembl.org/biomart/martservice'
REST = 'https://rest.ensembl.org/info/data?content-type=application/json'
QUERY = ('<?xml version="1.0" encoding="UTF-8"?>'
         '<!DOCTYPE Query>'
         '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" count="" '
         'datasetConfigVersion="0.6">'
         '<Dataset name="hsapiens_gene_ensembl" interface="default">'
         '<Attribute name="ensembl_gene_id"/>'
         '<Attribute name="external_gene_name"/>'
         '</Dataset></Query>')


class Permanent308(urllib.request.HTTPRedirectHandler):
    """Python 3.9's urllib does not follow 308, which is what Ensembl answers with."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    with urllib.request.urlopen(REST, timeout=60) as r:
        releases = json.loads(r.read().decode())['releases']
    opener = urllib.request.build_opener(Permanent308)
    url = HOST + '?' + urllib.parse.urlencode({'query': QUERY})
    with opener.open(url, timeout=300) as r:
        resolved = r.geturl()
        body = r.read().decode('utf-8', errors='replace')
    pairs, blank = {}, 0
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 2 or not parts[1].strip():
            blank += 1
            continue
        pairs[parts[0].strip()] = parts[1].strip()
    if len(pairs) < 20000:
        raise SystemExit(f'only {len(pairs)} id-symbol pairs returned; the query did not succeed')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'source': HOST, 'resolved_url': resolved, 'release_reported_by_rest_api': releases,
        'release_note': 'the project pins Ensembl 109 for quantification; the 109 BioMart archive '
                        'host refuses TLS from this machine, so this map is from the current release '
                        'and the number above is what the API reported at fetch time',
        'dataset': 'hsapiens_gene_ensembl',
        'attributes': ['ensembl_gene_id', 'external_gene_name'],
        'retrieved': '2026-10-05',
        'n_pairs': len(pairs), 'n_ids_without_a_symbol': blank,
        'why': 'matching GSE249010 (Ensembl ids) to the FAANG BMDM table (symbols) with a map that '
               'is independent of both panels',
        'map': pairs,
    }, indent=2), encoding='utf-8')
    print(f'{len(pairs)} id->symbol pairs, {blank} ids without a symbol -> {out}')


if __name__ == '__main__':
    main()
