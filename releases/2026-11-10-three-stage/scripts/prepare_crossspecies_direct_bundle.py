#!/usr/bin/env python3
"""Freeze the verified TG-GATEs cohort for a remote animal-information control.

The input and gene axes must exactly reproduce the existing absolute-dose run.
This program defines no new scientific performance result.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_feedback_nested_benchmark as nested
from cross_species.feedback_data import load_feedback_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--compound-features", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Use a new frozen bundle directory")
    reference = json.loads((args.reference_dir / "report.json").read_text())
    data = load_feedback_data(args.data_dir, out_dir=args.cache_dir, seed=reference["seed"])
    if data.input_hashes != reference["input_hashes"]:
        raise ValueError("Raw input hashes differ from the existing strict analysis")
    data, chemical, chemical_provenance = nested.align_compound_features(data, args.compound_features)
    data, exposure = nested.absolute_concentration_data(data)
    chemical = {name: value[exposure["original_row_indices"]] for name, value in chemical.items()}
    context = nested.model_context(data)
    if (len(data.sample_ids) != reference["rows"]
            or len(set(data.compounds)) != reference["compounds"]
            or len(data.feature_ids) != reference["target_genes"]
            or chemical_provenance["sha256"] != reference["molecular_features"]["sha256"]):
        raise ValueError("Cohort, fixed gene axis or molecular features differ")
    with (args.reference_dir / "fold_assignments.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [row["sample_id"] for row in rows] != data.sample_ids.tolist():
        raise ValueError("Frozen outer sample order differs")
    outer = np.asarray([int(row["outer_fold"]) for row in rows])
    regenerated = nested.group_folds(data.compounds, reference["outer_folds"], reference["seed"])
    if not np.array_equal(outer, regenerated):
        raise ValueError("Frozen outer assignment differs from the original rule")
    inner_metadata = json.loads((args.reference_dir / "inner_folds.json").read_text())
    inner = np.full((reference["outer_folds"], len(outer)), -1, dtype=np.int64)
    for fold in range(reference["outer_folds"]):
        train = np.flatnonzero(outer != fold)
        record = inner_metadata[str(fold)]
        if record["train_sample_ids"] != data.sample_ids[train].tolist():
            raise ValueError("Original inner-fold training sample order differs")
        inner[fold, train] = record["inner_fold"]
        expected = nested.group_folds(data.compounds[train], reference["inner_folds"],
                                      reference["seed"] + 1009 * (fold + 1))
        if not np.array_equal(inner[fold, train], expected):
            raise ValueError("Frozen inner assignment differs from the original rule")
    with np.load(args.reference_dir / "oof_predictions.npz", allow_pickle=False) as old:
        for name, actual in {"sample_ids": data.sample_ids, "compounds": data.compounds,
                             "human_gene_ids": data.feature_ids, "source_gene_ids": data.source_feature_ids,
                             "truth": data.target_delta.astype(np.float32),
                             "source_copy": data.source_delta.astype(np.float32), "outer_fold": outer}.items():
            if not np.array_equal(old[name], actual):
                raise ValueError("Rebuilt array differs from the original frozen run: " + name)
        original_predictions = {name: old[name].copy() for name in
                                ["chemical_context", "source_plus_chemical_context"]}
    args.out_dir.mkdir(parents=True)
    np.savez_compressed(args.out_dir / "bundle.npz", source=data.source_delta, target=data.target_delta,
                        context=context, fingerprints=chemical["fingerprints"], descriptors=chemical["descriptors"],
                        sample_ids=data.sample_ids, compounds=data.compounds,
                        human_gene_ids=data.feature_ids, rat_gene_ids=data.source_feature_ids,
                        outer_fold=outer, inner_fold_by_outer=inner)
    nested.save_json(args.out_dir / "metadata.json", exposure["retained_metadata"])
    np.savez_compressed(args.out_dir / "legacy_predictions.npz", **original_predictions)
    manifest = {
        "stage": "verified_fixed_crossspecies_inputs", "source_species": "rat", "target_species": "human",
        "rows": len(outer), "compounds": len(set(data.compounds)), "genes": len(data.feature_ids),
        "source_quantity": "rat treatment minus matched-control condition-mean RMA log2 expression",
        "target_quantity": "human treatment minus matched-control condition-mean RMA log2 expression",
        "context_feature_names": data.context_feature_ids.tolist(), "input_hashes": data.input_hashes,
        "chemical_provenance": chemical_provenance, "raw_rebuild_equals_existing_input_arrays": True,
        "outer_and_inner_assignments_match_saved_original": True,
        "reference_report_sha256": nested.file_hash(args.reference_dir / "report.json"),
        "reference_predictions_sha256": nested.file_hash(args.reference_dir / "oof_predictions.npz"),
        "prepare_script_sha256": nested.file_hash(__file__),
        "legacy_kernel_script_sha256": nested.file_hash(nested.__file__),
        "warnings": list(data.warnings), "files": {},
    }
    manifest["files"] = {name: nested.file_hash(args.out_dir / name) for name in
                         ["bundle.npz", "metadata.json", "legacy_predictions.npz"]}
    nested.save_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["rows", "compounds", "genes",
          "raw_rebuild_equals_existing_input_arrays", "outer_and_inner_assignments_match_saved_original"]}, indent=2))


if __name__ == "__main__":
    main()
