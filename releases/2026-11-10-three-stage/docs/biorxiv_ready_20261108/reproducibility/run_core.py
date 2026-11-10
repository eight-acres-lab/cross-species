#!/usr/bin/env python3
"""Portable command plan for the integrated manuscript's main analyses.

Default mode prints commands and input gaps. --execute runs the selected phase
in a new work root; original packaged results are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys


def copy_if_identical_or_missing(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.read_bytes() != target.read_bytes():
        raise ValueError("Existing staged metadata differs: " + str(target))
    if not target.exists():
        shutil.copy2(source, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--phase", choices=["blood", "tg_baseline", "tg_supplement", "direct", "external", "metrics", "q_audit", "all"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(); root = args.package_root.resolve(); work = args.work_root.resolve()
    py = sys.executable
    def script(name): return str(root / "scripts" / name)
    def run(cmd):
        print(json.dumps(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=root, check=True)
    def stage(source, target):
        if args.execute: copy_if_identical_or_missing(root / source, work / target)
    phases = ["blood", "tg_baseline", "tg_supplement", "direct", "external", "metrics", "q_audit"] if args.phase == "all" else [args.phase]
    if args.execute: work.mkdir(parents=True, exist_ok=True)
    tg = work / "raw/tggates/data"; chemistry = root / "artifacts/feedback_compound_features_20261107/features.npz"
    baseline = work / "tg_absolute_baseline"
    for phase in phases:
        if phase in {"tg_baseline", "tg_supplement"}:
            for source, target in [
                ("artifacts/tg_gates_rat_human_v1/ortholog_map/rat_to_human_one2one.tsv", "raw/tggates/ortholog_map/rat_to_human_one2one.tsv"),
                ("artifacts/tg_gates_rat_human_v1/metadata/open_tggates_cel_file_attribute.csv", "raw/tggates/metadata/open_tggates_cel_file_attribute.csv")]:
                stage(source, target)
        if phase == "tg_baseline":
            run([py, script("run_feedback_nested_benchmark.py"), "--data-dir", str(tg), "--compound-features", str(chemistry),
                 "--absolute-concentrations", "--out-dir", str(baseline), "--cache-dir", str(work / "cache/tg"), "--threads", "8"])
        elif phase == "tg_supplement":
            common = ["--data-dir", str(tg), "--compound-features", str(chemistry), "--absolute-concentrations", "--cache-dir", str(work / "cache/tg"), "--threads", "8"]
            run([py, script("run_feedback_residual_adapter.py"), *common, "--direct-run-dir", str(baseline), "--out-dir", str(work / "tg_residual_adapter")])
            run([py, script("run_feedback_joint_specificity.py"), *common, "--feature-manifest", str(root / "artifacts/feedback_compound_features_20261107/manifest.json"), "--out-dir", str(work / "tg_joint_specificity")])
            run([py, script("audit_feedback_joint_identification.py"), "--run-dir", str(work / "tg_joint_specificity")])
            run([py, script("run_feedback_label_budget.py"), *common, "--full-run-dir", str(baseline), "--out-dir", str(work / "tg_label_budget")])
        elif phase == "direct":
            # Recreate the original fold metadata from the newly executed baseline.
            reference = work / "direct_reference"
            bundle = work / "crossspecies_direct_03/bundle_v1"
            if args.execute and not bundle.exists():
                import numpy as np
                reference.mkdir(exist_ok=False)
                for filename in ["report.json", "fold_assignments.tsv", "oof_predictions.npz"]:
                    shutil.copy2(baseline / filename, reference / filename)
                folds = {}
                for i in range(5):
                    with np.load(baseline / f"fold_{i}_transforms_models.npz", allow_pickle=False) as z:
                        folds[str(i)] = {"train_sample_ids": z["train_sample_ids"].tolist(), "inner_fold": z["inner_fold"].tolist()}
                (reference / "inner_folds.json").write_text(json.dumps(folds, indent=2) + "\n")
            if not bundle.exists():
                run([py, script("prepare_crossspecies_direct_bundle.py"), "--data-dir", str(tg), "--compound-features", str(chemistry),
                     "--reference-dir", str(reference), "--cache-dir", str(work / "cache/tg"), "--out-dir", str(bundle)])
            protocol = root / "docs/multisource_20261108/crossspecies_direct_03/PROTOCOL_v1.zh.md"
            for mode in ["--fit-only", "--score-only"]:
                for regime in ["source_exposure", "paired_exposure"]:
                    run([py, script("run_crossspecies_pairing_benchmark.py"), "--bundle", str(bundle / "bundle.npz"), "--metadata", str(bundle / "metadata.json"),
                         "--protocol", str(protocol), "--out-dir", str(work / "crossspecies_direct_03/runs" / (regime + "_v1")),
                         "--device", args.device, "--regime", regime, "--threads", "8", mode])
        elif phase == "external":
            panel = work / "multisource/data/GSE330151"
            raw = work / "raw/g330"
            if not panel.exists():
                for name in ["fetch_manifest.json", "sample_design.tsv"]:
                    stage("outputs/biorxiv_sprint_20261107/gse330151_fetch/" + name, "raw/g330/" + name)
                run([py, script("prepare_multisource_panels.py"), "--g330-dir", str(raw), "--out-dir", str(panel.parent),
                     "--orthologs", str(root / "outputs/multisource_20261108/curated_panels/orthology_release_freeze/axis.tsv")])
            inputs = work / "external_transfer_04/inputs_v1"
            if not inputs.exists():
                audit = root / "docs/multisource_20261108/external_transfer_04"
                run([py, script("prepare_crossstudy_transfer_inputs.py"), "--training-dir", str(work / "crossspecies_direct_03/bundle_v1"),
                     "--panel-dir", str(panel), "--bridge-dir", str(audit / "gene_bridge"), "--task-manifest", str(audit / "hepatocyte_audit/gse330151_task_manifest.json"),
                     "--chemical-dir", str(audit / "chemical_features"), "--out-dir", str(inputs)])
            for mode in ["--fit-only", "--score-only"]:
                run([py, script("run_crossstudy_kernel_transfer.py"), "--input-dir", str(inputs), "--protocol", str(root / "docs/multisource_20261108/external_transfer_04/PROTOCOL_v1.zh.md"),
                     "--out-dir", str(work / "external_transfer_04/run_v1"), "--device", args.device, "--threads", "8", mode])
        elif phase == "metrics":
            run([py, script("run_manuscript_metric_checks.py"), "--project-root", str(work), "--protocol", str(root / "docs/biorxiv_ready_20261108/metric_checks/PROTOCOL_v1.zh.md"),
                 "--out-dir", str(work / "metric_checks")])
        elif phase == "q_audit":
            run([py, script("audit_replicate_normalisation.py"), "--out", str(work / "q_interpretation_audit.json")])
        elif phase == "blood":
            blood = work / "biorxiv_ready_20261108/legacy_blood"
            srcdir = root / "docs/biorxiv_ready_20261108/legacy_blood_audit"
            if args.execute:
                blood.mkdir(parents=True, exist_ok=True)
                shutil.copy2(srcdir / "PROTOCOL_v1.zh.md", blood / "PROTOCOL_v1.zh.md")
                audit = json.loads((srcdir / "input_audit.json").read_text())
                (blood / "input").mkdir(exist_ok=True)
                provenance_path = blood / "input/provenance.json"
                if provenance_path.exists():
                    previous = json.loads(provenance_path.read_text())
                    for name, expected in audit["source_hashes"].items():
                        if previous["downloaded_files"][name]["sha256"] != expected:
                            raise ValueError("Existing blood provenance differs")
                else:
                    provenance_path.write_text(json.dumps({"downloaded_files": {name: {"sha256": value} for name, value in audit["source_hashes"].items()}}, indent=2) + "\n")
            for name in ["prepare_missing_aware.py", "run_missing_aware_analysis.py"]:
                src = srcdir / name; patched = blood / "portable_code" / name
                if args.execute:
                    original = src.read_text()
                    literal = "ROOT=Path('/hy-tmp/cross-species-feedback-20261108/biorxiv_ready_20261108/legacy_blood')"
                    if original.count(literal) != 1:
                        raise ValueError("Expected one historical path assignment")
                    transformed = original.replace(literal, "ROOT=Path(" + repr(str(blood)) + ")")
                    patched.parent.mkdir(exist_ok=True)
                    patched.write_text(transformed)
                    note = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "patched_sha256": hashlib.sha256(patched.read_bytes()).hexdigest(),
                            "change": "Only ROOT path literal relocated; algorithm and parameters unchanged"}
                    patched.with_suffix(".path_patch.json").write_text(json.dumps(note, indent=2) + "\n")
                run([py, str(patched)])


if __name__ == "__main__":
    main()
