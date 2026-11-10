#!/usr/bin/env python3
"""Independently rebuild named State inputs from the preserved native counts.

Does not import preparation/training code, read predictions, or compute a
biological response. Checks every output value in bounded row blocks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    before = {str(args.manifest): sha(args.manifest)}
    for record in [*manifest['source_files'].values(), manifest['arrays'], manifest['axis']]:
        assert sha(record['path']) == record['sha256'], record['path']
        before[record['path']] = record['sha256']
    annotation = pd.read_csv(manifest['source_files']['gencode32_annotation']['path'], sep='\t', dtype=str)
    coding = set(annotation.loc[annotation.gene_type == 'protein_coding', 'gene_id_stable'])
    axis = pd.read_csv(manifest['axis']['path'], sep='\t')
    assert axis.native_index.is_unique and axis.human_ensembl_gene_id.is_unique
    assert len(axis) == 6357 and np.all(np.diff(axis.native_index) > 0)
    assert set(axis.native_index).issubset(range(6546))
    result = {'status': 'passed', 'scope': 'Independent reconstruction of all prepared expression values and native metadata; no responses or predictions read',
              'script_sha256': sha(__file__), 'manifest_sha256': before[str(args.manifest)], 'species': {}}
    with np.load(manifest['arrays']['path'], allow_pickle=False) as saved:
        assert np.array_equal(saved['native_gene_indices'], axis.native_index.to_numpy())
        for species in ['human', 'cynomolgus']:
            counts = ad.read_h5ad(manifest['source_files'][species + '_counts']['path'])
            assert counts.obs_names.is_unique and counts.var_names.is_unique
            assert np.array_equal(counts.obs_names.astype(str), saved[species + '_cell_ids'])
            for target, column in [('donors', 'individual'), ('cell_lines', 'cell_line'), ('condition', 'condition'),
                                   ('guide', 'dominant_guide'), ('capture', 'capture'), ('experiment', 'experiment')]:
                assert np.array_equal(counts.obs[column].astype(str), saved[species + '_' + target]), (species, column)
            columns = counts.var_names.get_indexer(axis.human_ensembl_gene_id)
            assert np.all(columns >= 0)
            assert np.array_equal(saved[species + '_gene_ids'], axis.human_ensembl_gene_id.to_numpy())
            denominator_mask = np.array([gene in coding for gene in counts.var_names], dtype=bool)
            library = np.asarray(counts.X[:, denominator_mask].sum(axis=1)).ravel().astype(np.float64)
            assert np.all(library > 0) and np.array_equal(library, saved[species + '_coding_library_size'])
            observed = saved[species + '_expression']
            assert observed.shape == (counts.n_obs, len(axis)) and observed.dtype == np.float32
            compared = 0
            for start in range(0, counts.n_obs, 256):
                stop = min(start + 256, counts.n_obs)
                block = counts.X[start:stop, columns].toarray().astype(np.float64)
                rebuilt = np.log1p(block * 10000.0 / library[start:stop, None]).astype(np.float32)
                assert np.array_equal(rebuilt, observed[start:stop]), (species, start, float(np.max(np.abs(rebuilt - observed[start:stop]))))
                compared += rebuilt.size
            result['species'][species] = {'cells': counts.n_obs, 'observed_genes': len(axis), 'coding_denominator_genes': int(denominator_mask.sum()),
                                           'values_compared': compared, 'all_float32_values_exact': True, 'maximum_absolute_difference': 0.0,
                                           'metadata_and_gene_coordinates_exact': True}
    for path, digest in before.items():
        assert sha(path) == digest, 'Input changed during verification: ' + path
    result['total_values_recomputed'] = sum(v['values_compared'] for v in result['species'].values())
    result['all_source_hashes_unchanged'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
