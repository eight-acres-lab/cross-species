#!/usr/bin/env python3
"""Collect Figure 2 metadata without changing or rerunning any scientific analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOOD = 'artifacts/gse249010_v1/processed/response_metadata.tsv'


def collect_dataset_overview():
    sources = {}

    def read(path, table=False):
        payload = (ROOT / path).read_bytes()
        sources[path] = hashlib.sha256(payload).hexdigest()
        if table:
            return list(csv.DictReader(payload.decode().splitlines(), delimiter='\t'))
        return json.loads(payload)

    blood = read(BLOOD, table=True)
    datasets = []
    metadata = [
        ('Blood', BLOOD, 'Whole blood', 'Core analyses', 'GSE249010'),
        ('Hepatocytes', 'artifacts/biology/gse330151/response_metadata.tsv',
         'Primary hepatocytes', 'Additional panels', 'GSE330151'),
        ('Macrophages', 'artifacts/faang_bmdm/processed/response_metadata.tsv',
         'Macrophages', 'Additional panels', 'PRJEB22535 and related projects'),
        ('Yeasts', 'artifacts/biology/selenite/response_metadata.tsv',
         'Yeast culture', 'Additional panels', 'E-MTAB-7022 and related accessions'),
    ]
    for name, source, system, role, accession in metadata:
        rows = blood if source == BLOOD else read(source, table=True)
        species = sorted({r['species'] for r in rows})
        datasets.append(dict(name=name, species=species, n_species=len(species),
                             system=system, role=role, accession=accession, source=source))

    # Organism identities are study annotations, not inferred from cell-type counts.
    datasets.insert(1, dict(name='TG-GATEs', species=['Homo sapiens', 'Rattus norvegicus'],
                           n_species=2, system='Hepatocytes / liver', role='Core analyses',
                           accession='S-DIXA-AN-004/005/007'))
    lincs = read('artifacts/lincs_scale/scale_test.json')
    parse = read('artifacts/rhaister_benchmark/donor_ceiling.json')
    datasets.extend([
        dict(name='LINCS', species=['Homo sapiens'], n_species=1,
             system=' / '.join(lincs['pair']), role='External check', accession='GSE70138'),
        dict(name='Parse PBMC', species=['Homo sapiens'], n_species=1,
             system='PBMCs', role='External check', accession='Parse / Rhaister release',
             n_cell_types=len(parse['by_cell_type'])),
    ])

    species = {r['species'] for r in blood}
    tasks = sorted(set.intersection(*[
        {r['task_id'] for r in blood if r['species'] == sp} for sp in species
    ]))
    selected = [r for r in blood if r['task_id'] in tasks]
    doses = sorted({float(r['dose_ng_ml']) for r in selected})
    times = sorted({float(r['time_h']) for r in selected})
    counts = [[sum(float(r['dose_ng_ml']) == dose and float(r['time_h']) == time
                   for r in selected) for time in times] for dose in doses]
    if len({r['sample_id'] for r in blood}) != len(blood):
        raise ValueError('Blood response metadata has duplicate sample identifiers.')

    comparisons = []
    choices = [
        ('Human, 24 vs 8 h', 'mismatch_controls', 'human hepatocyte | time 24 hr vs 8 hr'),
        ('Rat, 24 vs 8 h', 'mismatch_controls', 'rat hepatocyte | time 24 hr vs 8 hr'),
        ('Human, 24 vs 2 h', 'mismatch_controls', 'human hepatocyte | time 24 hr vs 2 hr'),
        ('Rat, 24 vs 2 h', 'mismatch_controls', 'rat hepatocyte | time 24 hr vs 2 hr'),
        ('Human vs rat', 'system_vs_species', 'species_changed'),
        ('Culture vs liver', 'matched_time_system',
         'SYSTEM, TIME MATCHED: rat hep 24h vs rat liver 24h'),
    ]
    for name, stem, key in choices:
        path = f'artifacts/biology/dili/{stem}.json'
        entry = read(path)[key]
        rows = entry.get('per_compound', entry.get('rows'))
        unique = {r['compound'] for r in rows}
        if len(unique) != len(rows) or len(rows) != entry.get('n_compounds_shared', entry.get('n')):
            raise ValueError(f'Inconsistent compound count: {key}')
        values = [r.get('gene', r) for r in rows]
        kept = [v for v in values if v.get('frac') is not None and math.isfinite(v['frac'])
                and v.get('bound') is not None and .60 <= v['bound'] < .78]
        comparisons.append(dict(label=name, source=path, source_key=key,
                                n_shared=len(rows), n_in_band=len(kept)))

    return dict(
        schema_version=1, datasets=datasets,
        blood=dict(source=BLOOD, n_deposited_rows=len(blood), n_shared_rows=len(selected),
                   n_excluded_rows=len(blood)-len(selected), n_shared_conditions=len(tasks),
                   n_subjects=len({(r['species'], r['subject']) for r in selected}),
                   doses_ng_ml=doses, times_h=times, counts=counts,
                   counting_unit='One subject response under one dose and time; not an independent subject.'),
        tggates=dict(ceiling_band=[.60, .78], upper_bound_inclusive=False,
                     comparisons=comparisons, counting_unit='Distinct compounds within each comparison.'),
        sources_sha256=sources,
    )
