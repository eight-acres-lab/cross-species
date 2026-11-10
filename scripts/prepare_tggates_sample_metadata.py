#!/usr/bin/env python3
"""Join official public TG-GATEs barcode annotations to the sample RMA matrix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--attributes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(args.matrix) as f:
        samples = f.readline().rstrip('\n').split('\t')[1:]
    with open(args.attributes, encoding='cp932') as f:
        attributes = list(csv.DictReader(f))
    by_id = {}
    for row in attributes:
        barcode = row['BARCODE']
        if barcode in samples:
            if barcode in by_id:
                raise ValueError(f'Duplicate annotation for {barcode}')
            by_id[barcode] = row
    missing = set(samples) - by_id.keys()
    if missing:
        raise ValueError(f'Missing official annotation: {len(missing)}, {sorted(missing)[:10]}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['sample_id', 'species', 'compound', 'dose_level', 'time_h',
              'dose_value', 'dose_unit', 'experiment_id', 'group_id', 'replicate_id']
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for sample in samples:
            row = by_id[sample]
            if row['EXP_TEST_TYPE'] != 'in vitro' or row['ORGAN'] != 'Liver':
                raise ValueError(f'Unexpected sample type {sample}')
            period = row['SACRIFICE_PERIOD'].split()
            if len(period) != 2 or period[1] != 'hr':
                raise ValueError(f'Unexpected time unit {period}')
            writer.writerow(dict(zip(fields, [sample, row['SPECIES'].lower(), row['COMPOUND_NAME'],
                row['DOSE_LEVEL'], str(float(period[0])), row['DOSE'], row['DOSE_UNIT'],
                row['EXP_ID'], row['GROUP_ID'], row['INDIVIDUAL_ID']])))
    result = {'samples_matched': len(samples), 'official_attributes': len(attributes),
              'matrix': {'path': args.matrix, 'sha256': sha256(args.matrix)},
              'attributes': {'path': args.attributes, 'sha256': sha256(args.attributes)},
              'sources': ['https://zenodo.org/records/20204514',
                          'https://dbarchive.biosciencedbc.jp/en/open-tggates/data-4.html'],
              'replicate_note': 'INDIVIDUAL_ID identifies a sample within an assay group; it does not establish independent human donors.'}
    output.with_suffix('.provenance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
