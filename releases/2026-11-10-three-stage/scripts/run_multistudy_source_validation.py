#!/usr/bin/env python3
"""Amplitude-aware small-panel source comparison and whole-study transfer.

All source subsets and fixed calibration modes are reported. Historical source
selection excludes query human treatment outcomes. Cross-study normalization and
exposure differences remain visible; few perturbations do not establish a general
source-selection law.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.multisource_data import load_curated_panel, file_hash
from cross_species.multistudy_source_models import HistoricalPanelModel, evaluate_predictions, error_geometry, subsets


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def panel_view(panel, genes, source_ids):
    species = ("human",) + tuple(source_ids)
    positions = {str(g): i for i, g in enumerate(panel.feature_ids)}
    idx = np.array([positions[g] for g in genes])
    contrast = {str(row["contrast_id"]): i for i, row in enumerate(panel.contrast_metadata)}
    rows = [r for r in panel.contrast_metadata if r["species"] in species
            and r.get("contrast_kind", "drug") == "drug"]
    lookup = {}
    for row in rows:
        key = (row["species"], row["perturbation"])
        if key in lookup:
            raise ValueError("Multiple contexts for one species/drug; define a separate context task explicitly")
        lookup[key] = row
    tasks = sorted(set.intersection(*[{r["perturbation"] for r in rows if r["species"] == s} for s in species]))
    if len(tasks) < 3:
        raise ValueError("At least three fully covered panel tasks needed for this fixed analysis")
    means, controls, records = [], [], []
    for task in tasks:
        tmeans, tcontrols, trecords = [], [], []
        for s in species:
            row = lookup[s, task]
            values = panel.contrast_means[contrast[row["contrast_id"]], idx]
            control_rows = panel.arm_indices(row["contrast_id"], "vehicle")
            treated_rows = panel.arm_indices(row["contrast_id"], "treated")
            baseline = panel.arm_logcpm[control_rows][:, idx].mean(0)
            if not np.isfinite(values).all() or not np.isfinite(baseline).all():
                raise ValueError("Fixed common axis has missing values")
            tmeans.append(values); tcontrols.append(baseline)
            trecords.append({"species": s, "contrast_id": row["contrast_id"], "perturbation": task,
                             "treated_rows": treated_rows.tolist(), "control_rows": control_rows.tolist(),
                             "time_h": row.get("time_h"), "dose_value": row.get("dose_value"),
                             "dose_unit": row.get("dose_unit"), "units": row.get("units"),
                             "biological_independence": row.get("independent_unit_status"),
                             "context": row.get("cell_model"), "contrast_kind": row.get("contrast_kind", "drug")})
        means.append(tmeans); controls.append(tcontrols); records.append(trecords)
    values, baselines = np.asarray(means), np.asarray(controls)
    return {"task_ids": tasks, "sources": values[:, 1:], "human": values[:, 0],
            "source_controls": baselines[:, 1:], "human_controls": baselines[:, 0],
            "records": records, "gene_indices": idx}


def control_selected_predictions(model, view, query_indices):
    source = view["sources"][query_indices]
    chosen, similarity = model.choose_from_controls(view["source_controls"][query_indices], view["human_controls"][query_indices])
    values = {"raw": source, "scaled": model.calibrators["scaled"].predict(source),
              "affine_gene_mean": model.calibrators["affine_gene_mean"].predict(source)}
    result = {}
    linear = {}
    for mode, predictions in values.items():
        output = np.repeat(model.target_mean[None], len(chosen), axis=0)
        weights, offsets = [], []
        for row, selected in enumerate(chosen):
            if selected >= 0:
                output[row] = predictions[row, selected]
                coeff, offset = model.linear_estimators()[f"{mode}/equal/{model.source_ids[selected]}"]
            else:
                coeff, offset = np.zeros(len(model.source_ids)), model.target_mean
            weights.append(coeff); offsets.append(offset)
        label = f"{mode}/known_control_selected"
        result[label] = output
        linear[label] = (np.asarray(weights), np.asarray(offsets))
    return result, linear, {"chosen_source_indices": chosen, "control_pearson": similarity,
                           "control_availability": "All candidate source and target control profiles assumed already measured; no treated human outcome used. Not a zero-cost source acquisition recommendation."}


def run_case(train_view, test_view, source_ids, *, lopo=False):
    predictions, coefficients, offsets, fit_records = {}, {}, {}, []
    task_count, genes = test_view["human"].shape
    for q in range(task_count):
        train_idx = np.flatnonzero(np.arange(len(train_view["human"])) != q) if lopo else np.arange(len(train_view["human"]))
        model = HistoricalPanelModel.fit(train_view["sources"][train_idx], train_view["human"][train_idx], source_ids)
        current = model.predict(test_view["sources"][[q]])
        forms = {name: (coef[None], offset[None]) for name, (coef, offset) in model.linear_estimators().items()}
        cpred, cforms, selection = control_selected_predictions(model, test_view, np.array([q]))
        current.update(cpred); forms.update(cforms)
        for name, array in current.items():
            predictions.setdefault(name, np.empty((task_count, genes)))[q] = array[0]
            coefficients.setdefault(name, np.empty((task_count, len(source_ids))))[q] = forms[name][0][0]
            offsets.setdefault(name, np.empty((task_count, genes)))[q] = forms[name][1][0]
        fit_records.append({"test_task": test_view["task_ids"][q],
                            "training_tasks": [train_view["task_ids"][i] for i in train_idx],
                            "fitted_state": model.state(), "control_selection": selection})
    scores = evaluate_predictions(predictions, test_view["human"])
    return predictions, coefficients, offsets, scores, fit_records


def measured_arm_sensitivity(panel, view, coefficients, offsets, *, n_draws, seed, device, chunk_size=8):
    """Resample original arm records; shared control index sets share each draw.

    Fixed fit coefficients are not refitted. Percentiles describe sensitivity to
    measured records; they are not task/population confidence intervals.
    """
    unique_groups, group_ids, role_groups = [], {}, []
    for trecord in view["records"]:
        task_groups = []
        for record in trecord:
            pair = []
            for role in ["treated_rows", "control_rows"]:
                key = tuple(record[role])
                if key not in group_ids:
                    group_ids[key] = len(unique_groups); unique_groups.append(key)
                pair.append(group_ids[key])
            task_groups.append(pair)
        role_groups.append(task_groups)
    rng = np.random.default_rng(seed)
    counts = []
    for rows in unique_groups:
        draw = rng.integers(len(rows), size=(n_draws, len(rows)))
        counts.append(np.stack([(draw == i).sum(1) for i in range(len(rows))], 1))
    source_names = list(coefficients)
    weights = np.stack([coefficients[name] for name in source_names])
    biases = np.stack([offsets[name] for name in source_names])
    raw = [panel.arm_logcpm[list(rows)][:, view["gene_indices"]] for rows in unique_groups]
    tg = np.asarray(role_groups)

    def compute(target_device, draw_limit):
        W = torch.tensor(weights, dtype=torch.float64, device=target_device)
        C = torch.tensor(biases, dtype=torch.float64, device=target_device)
        profiles = [torch.tensor(a, dtype=torch.float64, device=target_device) for a in raw]
        out = np.empty((draw_limit, len(view["task_ids"]), len(source_names)))
        with torch.no_grad():
            for start in range(0, draw_limit, chunk_size):
                end = min(start + chunk_size, draw_limit)
                means = torch.stack([torch.tensor(n[start:end] / len(a), dtype=torch.float64, device=target_device) @ a
                                     for n, a in zip(counts, profiles)], dim=1)
                response = means[:, tg[:, :, 0]] - means[:, tg[:, :, 1]]
                predicted = torch.einsum("mts,btsg->btmg", W, response[:, :, 1:]) + C.transpose(0, 1)[None]
                out[start:end] = (predicted - response[:, :, 0, None]).square().mean(-1).cpu().numpy()
        return out

    cpu = compute("cpu", min(4, n_draws))
    start = time.monotonic()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    result = compute(device, n_draws)
    if device == "cuda": torch.cuda.synchronize()
    difference = float(np.max(np.abs(cpu - result[:len(cpu)])))
    if not np.allclose(cpu, result[:len(cpu)], atol=1e-10, rtol=1e-10):
        raise ValueError("CPU/GPU sensitivity computation differs")
    summary = {name: {"per_query_percentiles_2p5_50_97p5": np.quantile(result[:, :, m], [.025, .5, .975], axis=0),
                      "macro_percentiles_2p5_50_97p5": np.quantile(result[:, :, m].mean(1), [.025, .5, .975])}
               for m, name in enumerate(source_names)}
    return result, summary, {"device": device, "seconds": time.monotonic() - start,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                            "CPU_parity_max_abs": difference, "resamples": n_draws,
                            "unique_arm_groups": len(unique_groups),
                            "single_record_groups": [i for i, rows in enumerate(unique_groups) if len(rows) == 1],
                            "shared_controls_preserved": True,
                            "scope": "Fixed-estimator measured-assay sensitivity; no donor identity or population interval, no additional independent perturbations."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", action="append", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=["rat", "cyno", "dog"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261108)
    args = ap.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()): ap.error("Use a new output directory")
    if len(set(args.sources)) != len(args.sources) or "human" in args.sources or not args.sources: ap.error("Unique nonhuman sources required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels = {path.name: load_curated_panel(path) for path in args.panel}
    if len(panels) != len(args.panel): raise ValueError("Panel directory basenames must be unique")
    available = [set(map(str, panel.feature_ids[panel.global_eligible_mask])) for panel in panels.values()]
    genes = sorted(set.intersection(*available))
    if not genes: raise ValueError("No shared complete feature axis")
    views = {name: panel_view(panel, genes, args.sources) for name, panel in panels.items()}
    code = [Path(__file__), ROOT / "src/cross_species/multistudy_source_models.py", ROOT / "src/cross_species/multisource_models.py", ROOT / "src/cross_species/multisource_data.py"]
    specification = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Retrospective small-panel exploration. Existing G330 outcomes were inspected. Whole-study tests are external protocol shifts, not prospective trials or isolated species causal effects.",
        "data": {path.name: {"path": str(path), "manifest_sha256": file_hash(path / 'manifest.json'), "panel_sha256": file_hash(path / 'panel.npz')} for path in args.panel},
        "source_order": args.sources, "genes": len(genes), "gene_axis_sha256": hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
        "cohort_rule": "Common complete global mask intersection from every panel; no effect-size selection; only complete drug contrasts across human and all named sources.",
        "queries": {name: view["task_ids"] for name, view in views.items()},
        "methods": {"scaled": "One nonnegative no-intercept slope per source, fitted to historical task/gene loss.",
                    "affine_gene_mean": "Training-human gene mean + one nonnegative source slope times deviation from its training-source gene mean.",
                    "fusion": "All nonempty source subsets; equal and simplex shrinkage. Objective=historicalOOF MSE / historical human RMS² + ||w-uniform||². Fixed multiplier1.",
                    "selection": "Raw best fixed source from historical raw-copy error; calibrated source choice from historical leave-one-task-out predictions. No query treated human outcome.",
                    "control_selection": "Pearson of already available source and target untreated controls; fixed original source ordering resolves ties. It uses all source control information and has a different acquisition budget from history-only source choice.",
                    "zero_slope_fallback": "Zero historical source innovation variance yields beta0. Affine calibration with one reference task reduces to its human mean."},
        "scoring": "Per-drug full-axis MSE, Pearson, response RMS; no gene/sample-count significance claims.",
        "budget": "Original available arm profiles per acquired source; added sources generally increase measurement budget. This study-transfer analysis does not equalize acquisition costs.",
        "sensitivity": {"draws": args.resamples, "unit": "Original treated/control arm records; shared original control sets reuse a draw; fitted historical coefficients stay fixed.", "limitation": "Rat CPZ may have one record; resampling cannot estimate its unobserved replicate variability. Donor identities unresolved."},
        "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in code}}
    save(args.out_dir / "specification.json", specification)
    for path in code:
        target = args.out_dir / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    np.savez_compressed(args.out_dir / "common_axis.npz", human_gene_ids=np.array(genes))
    cases = [(name, name, True) for name in panels]
    cases += [(train, test, False) for train in panels for test in panels if train != test]
    result = {}
    for train_name, test_name, lopo in cases:
        case = train_name + ("_LOPO" if lopo else "_TO_" + test_name)
        print("Fitting fixed case " + case, flush=True)
        case_dir = args.out_dir / case; case_dir.mkdir()
        prediction, weights, offsets, scores, states = run_case(views[train_name], views[test_name], args.sources, lopo=lopo)
        save(case_dir / "historical_fits.json", states)
        score_path = case_dir / "scores.json"; save(score_path, scores)
        diagnostics = {mode: error_geometry(np.stack([prediction[f"{mode}/equal/{s}"] for s in args.sources], 1), views[test_name]["human"], args.sources)
                       for mode in ['raw', 'scaled', 'affine_gene_mean']}
        save(case_dir / "terminal_error_geometry.json", diagnostics)
        np.savez_compressed(case_dir / "predictions.npz", names=np.array(list(prediction)), predictions=np.stack(list(prediction.values())),
                            targets=views[test_name]["human"], source_responses=views[test_name]["sources"], tasks=np.array(views[test_name]["task_ids"]),
                            human_gene_ids=np.array(genes), source_ids=np.array(args.sources),
                            coefficients=np.stack([weights[k] for k in prediction]), offsets=np.stack([offsets[k] for k in prediction]))
        resamples, summary, device = measured_arm_sensitivity(panels[test_name], views[test_name], weights, offsets,
            n_draws=args.resamples, seed=args.seed, device=args.device)
        save(case_dir / "measured_arm_sensitivity.json", summary); save(case_dir / "device.json", device)
        np.savez_compressed(case_dir / "measured_arm_resampling.npz", losses=resamples, method_names=np.array(list(prediction)))
        result[case] = {"training_study": train_name, "test_study": test_name,
                       "scope": "Within-study leave-one-drug exploratory" if lopo else "Whole-study protocol shift, with three reference drugs; no broad species-selection claim",
                       "test_queries": views[test_name]["task_ids"], "scores": scores, "device": device,
                       "training_records": views[train_name]["records"], "test_records": views[test_name]["records"]}
    save(args.out_dir / "report.json", {"specification_sha256": file_hash(args.out_dir / 'specification.json'), "cases": result})
    with (args.out_dir / "all_method_losses.csv").open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['case','method','query','MSE','Pearson'])
        for case, value in result.items():
            for method, score in value['scores'].items():
                for i, query in enumerate(value['test_queries']):
                    writer.writerow([case,method,query,score['mse'][i],score['pearson'][i] if np.isfinite(score['pearson'][i]) else ''])
    print(args.out_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
