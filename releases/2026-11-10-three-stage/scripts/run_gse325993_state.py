#!/usr/bin/env python3
"""Prepare metadata-only pools, then fit/freeze the declared36State comparisons.

This module has no scorer. Real fitting requires a reviewed hash-bound contract,
the sealed normalized arrays, checked pool manifests and >18GiB free storage.
Public-cohort QC limitations are inherited from the scientific contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("H", "Hextra_cells", "Monkey56", "Monkey56A0", "Monkey82", "Monkey82A0")
SEEDS = (20261110, 20261111, 20261112)
SPLIT_SALT = "gse325993_state_20261110/v1/metadata_splits"
BATCH_SALT = "gse325993_state_20261110/v1/step_cells"
METADATA_FIELDS = ("cell_ids", "donors", "cell_lines", "condition", "guide", "capture", "experiment")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def hashed(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def contract_and_manifest(args):
    require(platform.system() == "Linux", "Real-data commands must use the remote Linux runtime")
    require(sha256(args.contract) == args.contract_sha256, "Reviewed contract SHA mismatch")
    contract = read(args.contract)
    design = contract["design"]
    require(tuple(design["arms"]) == ARMS and tuple(design["seeds"]) == SEEDS, "Arms/seeds differ from implemented protocol")
    require(design["fits"] == 36 and len(design["outer_folds"]) == 2, "Expected36fits across2held-out human lines")
    for fold in design["outer_folds"]:
        require(fold["train_line"] != fold["test_line"] and
                {fold["train_line"], fold["test_line"]} == set(design["line_ids"]["human"]), "Human line leakage")
    settings = contract["training"]
    require(settings["optimizer_steps"] == 200 and settings["candidate_checkpoints"] == [0, 50, 100, 200], "Training budget differs")
    require(settings["added_source_weight"] == .5, "Added-source coefficient differs")
    require(contract["calibration"]["post_calibration_shrink"]["alpha_grid"] == [0, .25, .5, .75, 1], "Shrink grid differs")
    require(contract["pools"]["per_condition"] == 128, "Per-condition pool budget differs")
    require(contract["pools"]["hash_salt"] == SPLIT_SALT and contract["pools"]["strata"] == ["capture", "guide"], "Metadata split definition changed")
    require(contract["pools"]["human_training_pools"] == ["base", "extra", "cal"], "Human pool labels changed")
    binding = contract["data"]
    require(args.prepared.resolve() == Path(binding["prepared_root"]).resolve(), "Prepared root differs from contract")
    require(isinstance(binding["manifest_sha256"], str) and isinstance(binding["arrays_sha256"], str), "Scientific contract still lacks finalized normalized-input hashes")
    manifest_path = args.prepared / "manifest.json"
    require(sha256(manifest_path) == binding["prepared_manifest_sha256"], "Prepared manifest not contract-bound")
    manifest = read(manifest_path)
    arrays_path = args.prepared / Path(manifest["arrays"]["path"]).name
    actual = sha256(arrays_path)
    require(actual == manifest["arrays"]["sha256"] == binding["arrays_sha256"], "Normalized arrays SHA mismatch")
    require(arrays_path.name == binding["arrays_file"], "Normalized array filename changed")
    return contract, manifest, arrays_path


def load_metadata(arrays_path):
    """Read only metadata/coordinate arrays; expression arrays remain unopened."""
    result = {}
    with np.load(arrays_path, allow_pickle=False) as z:
        for species in ("human", "cynomolgus"):
            values = {k: z[species + "_" + k].astype(str) for k in METADATA_FIELDS}
            n = len(values["cell_ids"])
            require(all(x.shape == (n,) for x in values.values()), "Metadata arrays disagree")
            require(len(set(values["cell_ids"])) == n, "Repeated actual cell IDs")
            require(set(values["condition"]) == {"NTC", "POU5F1"}, "Unexpected assay conditions")
            require(set(values["experiment"]) == {"exp2"}, "Use matched public exp2 only")
            result[species] = values
        ix = z["native_gene_indices"].astype(np.int64)
    require(ix.ndim == 1 and len(ix) == len(set(ix)) and len(ix) > 0 and np.all(np.diff(ix) > 0), "Invalid native gene axis")
    require(ix.min() >= 0 and ix.max() < 6546, "Native gene coordinate outside checkpoint")
    return result, ix


def validate_metadata(contract, metadata, ix):
    require(len(ix) == contract["data"]["primary_gene_count"] == 6357, "Observed common gene budget changed")
    for species in ("human", "cynomolgus"):
        m = metadata[species]
        expected = contract["design"]["known_counts"][species]
        require(set(m["cell_lines"]) == set(expected), "Line labels differ from reviewed metadata")
        for line, counts in expected.items():
            require(len(set(m["donors"][m["cell_lines"] == line])) == 1, "One cell line maps to multiple donor identities")
            for condition, n in counts.items():
                require(int(np.sum((m["cell_lines"] == line) & (m["condition"] == condition))) == n, "Line/condition population size changed")
        require(set(m["capture"]) <= {f"exp2_chip1_lane{i}" for i in range(1, 9)}, "Unexpected capture label")
    require(sum(len(m["cell_ids"]) for m in metadata.values()) == 9732, "Public-cell population changed")


def split_stratified(metadata, eligible, pool_names, per_pool, seed, line, condition):
    """Proportional Hamilton quotas followed by balanced per-stratum pools.

    Uses capture, guide, stable cell IDs and seed only. Quotas/rounding use
    integers. Every stratum's pool counts differ by at most1; global pool sizes
    are exact. Selected cells are never reused across pools.
    """
    eligible = [int(i) for i in eligible]
    needed, population = len(pool_names) * per_pool, len(eligible)
    require(population >= needed, f"Insufficient {line}/{condition}: need{needed}, have{population}")
    strata = defaultdict(list)
    for i in eligible:
        strata[(str(metadata["capture"][i]), str(metadata["guide"][i]))].append(i)
    quota = {s: needed * len(v) // population for s, v in strata.items()}
    remainder = needed - sum(quota.values())
    order = sorted(strata, key=lambda s: (-(needed * len(strata[s]) % population),
                                          hashed(SPLIT_SALT, seed, line, condition, *s, "quota")))
    for s in order[:remainder]:
        quota[s] += 1
    pools = {p: [] for p in pool_names}
    audit = []
    # Deterministic stratum order also controls sequential remainder balancing.
    for s in sorted(strata):
        candidates = sorted(strata[s], key=lambda i: hashed(SPLIT_SALT, seed, line, condition, *s, metadata["cell_ids"][i], "select"))
        chosen = candidates[:quota[s]]
        each, extras = divmod(len(chosen), len(pool_names))
        assigned = {p: each for p in pool_names}
        # Giving the equal floor to every pool preserves the existing ordering
        # of their total sizes. Remainders go to distinct currently smallest pools.
        remaining = set(pool_names)
        for _ in range(extras):
            p = min(remaining, key=lambda p: (len(pools[p]) + assigned[p],
                                             hashed(SPLIT_SALT, seed, line, condition, *s, p, "remainder")))
            assigned[p] += 1
            remaining.remove(p)
        cursor = 0
        for p in pool_names:
            pools[p].extend(chosen[cursor:cursor + assigned[p]])
            cursor += assigned[p]
        require(max(assigned.values()) - min(assigned.values()) <= 1, "Per-stratum pool balance failed")
        audit.append({"capture": s[0], "guide": s[1], "available": len(candidates), "selected": quota[s], "pool_counts": assigned})
    require(all(len(v) == per_pool for v in pools.values()), "Exact global pool size failed")
    flat = [i for rows in pools.values() for i in rows]
    require(len(flat) == len(set(flat)) == needed, "Metadata pool overlap")
    return pools, audit


def pool_record(metadata, rows_by_condition, line, role):
    return {"line": line, "role": role,
            **{part + "_rows": list(rows_by_condition[label]) for part, label in [("control", "NTC"), ("target", "POU5F1")]},
            **{part + "_cell_ids": metadata["cell_ids"][rows_by_condition[label]].tolist()
               for part, label in [("control", "NTC"), ("target", "POU5F1")]}}


def construct_splits(contract, metadata):
    design, budget = contract["design"], contract["pools"]["per_condition"]
    seeds = []
    for seed in design["seeds"]:
        human, animal, audits = {}, {}, []
        for line in design["line_ids"]["human"]:
            m = metadata["human"]
            selected = {}
            for condition in ("NTC", "POU5F1"):
                eligible = np.flatnonzero((m["cell_lines"] == line) & (m["condition"] == condition))
                selected[condition], detail = split_stratified(m, eligible, ("base", "extra", "cal"), budget, seed, line, condition)
                audits.append({"line": line, "condition": condition, "strata": detail})
            human[line] = {p: pool_record(m, {c: selected[c][p] for c in selected}, line, p) for p in ("base", "extra", "cal")}
        for source in ("Monkey56", "Monkey82"):
            line, m = design["line_ids"][source], metadata["cynomolgus"]
            controls = np.flatnonzero((m["cell_lines"] == line) & (m["condition"] == "NTC"))
            c, detail = split_stratified(m, controls, ("input", "null_target"), budget, seed, line, "NTC")
            audits.append({"line": line, "condition": "NTC", "strata": detail})
            targets = np.flatnonzero((m["cell_lines"] == line) & (m["condition"] == "POU5F1"))
            t, detail = split_stratified(m, targets, ("treated_target",), budget, seed, line, "POU5F1")
            audits.append({"line": line, "condition": "POU5F1", "strata": detail})
            animal[source] = {
                "real": pool_record(m, {"NTC": c["input"], "POU5F1": t["treated_target"]}, line, source),
                "null": pool_record(m, {"NTC": c["input"], "POU5F1": c["null_target"]}, line, source + "A0"),
            }
        folds = []
        for fold in design["outer_folds"]:
            m, test_line = metadata["human"], fold["test_line"]
            test = {c: np.flatnonzero((m["cell_lines"] == test_line) & (m["condition"] == c)).tolist() for c in ("NTC", "POU5F1")}
            require(all(test.values()), "Held-out line lacks a required condition")
            folds.append({**fold, "human": human[fold["train_line"]], "animal": animal,
                          "test": pool_record(m, test, test_line, "test_all_cells")})
        seeds.append({"seed": seed, "folds": folds, "stratum_quota_audit": audits})
    return {"seed_splits": seeds, "split_salt": SPLIT_SALT, "split_features": ["capture", "guide", "cell_ids", "seed"],
            "expression_values_read": False, "expression_values_used_for_pool_selection": False,
            "heldout_line_all_cell_evaluation_rows_identified_by_metadata_only": True,
            "source_pools_identical_across_outer_folds_for_same_seed": True}


def prepare_splits(args):
    contract, manifest, arrays = contract_and_manifest(args)
    require(not args.out.exists(), "Use a new split directory")
    metadata, ix = load_metadata(arrays)
    validate_metadata(contract, metadata, ix)
    splits = construct_splits(contract, metadata)
    args.out.mkdir(parents=True)
    save(args.out / "splits.json", splits)
    save(args.out / "manifest.json", {"contract_sha256": args.contract_sha256, "code_sha256": sha256(__file__),
                                     "prepared_manifest_sha256": sha256(args.prepared / "manifest.json"),
                                     "arrays_sha256": manifest["arrays"]["sha256"],
                                     "splits_sha256": sha256(args.out / "splits.json"), "observed_genes": len(ix),
                                     "metadata_only": True, "expression_values_read": False})
    print(json.dumps({"phase": "metadata_pools_frozen", "seeds": 3, "outer_folds": 2, "expression_values_read": False}))


def load_training_inputs(args):
    contract, manifest, arrays = contract_and_manifest(args)
    split_manifest, splits = read(args.splits / "manifest.json"), read(args.splits / "splits.json")
    require(split_manifest["contract_sha256"] == args.contract_sha256 and split_manifest["code_sha256"] == sha256(__file__), "Split contract/code binding changed")
    require(split_manifest["arrays_sha256"] == manifest["arrays"]["sha256"] and
            split_manifest["splits_sha256"] == sha256(args.splits / "splits.json"), "Split data/hash binding changed")
    metadata, ix = load_metadata(arrays)
    validate_metadata(contract, metadata, ix)
    require(splits == plain(construct_splits(contract, metadata)), "Frozen pools differ from metadata-only algorithm")
    with np.load(arrays, allow_pickle=False) as z:
        data = {s: z[s + "_expression"].copy() for s in ("human", "cynomolgus")}
        require(all(np.isfinite(z[s + "_coding_library_size"]).all() and (z[s + "_coding_library_size"] > 0).all()
                    for s in ("human", "cynomolgus")), "Invalid coding library normalization denominator")
    for s, x in data.items():
        require(x.dtype == np.float32 and x.shape == (len(metadata[s]["cell_ids"]), len(ix)), "Normalized expression shape/dtype differs")
        require(np.isfinite(x).all() and np.min(x) >= 0, "Invalid normalized counts")
    return contract, manifest, splits, data, ix


def positions(n, seed, step, population):
    return np.asarray(sorted(range(n), key=lambda i: hashed(BATCH_SALT, seed, step, population, i))[:min(64, n)], dtype=int)


def batch_rows(record, seed, step, slot):
    return {part: np.asarray(record[part + "_rows"], dtype=int)[positions(len(record[part + "_rows"]), seed, step, slot + "|" + part)]
            for part in ("control", "target")}


def response_scale(expression, record):
    delta = expression[record["target_rows"]].mean(0, dtype=np.float64) - expression[record["control_rows"]].mean(0, dtype=np.float64)
    value = float(np.square(delta).mean())
    require(np.isfinite(value), "Nonfinite fitting-only response scale")
    return value


def get_runtime():
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from cross_species import state_assay_adapter as adapter
    return torch, adapter


def initialize(args, seed):
    torch, adapter = get_runtime()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    native, metadata = adapter.load_native_model(args.checkpoint, device="cpu", source_root=args.state_source)
    model = adapter.extend_assay_model(native).to(args.device).eval()
    return model, metadata


def pad(values, ix, device):
    import torch
    x = torch.zeros((len(values), 6546), dtype=torch.float32, device=device)
    x[:, ix] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return x


def infer(model, expression, record, ix, mask, device):
    torch, adapter = get_runtime()
    rows = np.asarray(record["control_rows"], dtype=int)
    predictions, contrasts, reconstructions = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            controls = pad(expression[rows[start:start + 64]], ix, device)
            output = adapter.predict_assay_contrast(model, controls, mask, context="human_iPSC")
            predictions.append(output["prediction"][:, ix].cpu().numpy())
            contrasts.append(output["contrast"][:, ix].cpu().numpy())
            reconstructions.append(output["native_control"][:, ix].cpu().numpy())
    return {"prediction": np.concatenate(predictions), "contrast": np.concatenate(contrasts),
            "native_control": np.concatenate(reconstructions), "control_rows": rows,
            "target_rows": np.asarray(record["target_rows"], dtype=int),
            "donor_line": np.asarray(record["line"]), "native_gene_indices": ix.detach().cpu().numpy()}


def mse(prediction, target):
    return float(np.square(np.asarray(prediction, dtype=np.float64).mean(0) - np.asarray(target, dtype=np.float64).mean(0)).mean())


def calibrate(model, expression, record, ix, mask, args):
    output = infer(model, expression, record, ix, mask, args.device)
    return mse(output["prediction"], expression[record["target_rows"]]), output


def progress(root, **record):
    record["unix_time"] = time.time()
    with (root / "progress.jsonl").open("a") as f:
        f.write(json.dumps(plain(record)) + "\n")
    print(json.dumps(plain(record)), flush=True)


def fit(args, contract, data, ix_array, fold, seed, arm, folder):
    torch, adapter = get_runtime()
    model, model_metadata = initialize(args, seed)
    ix = torch.as_tensor(ix_array, device=args.device)
    mask = torch.zeros(6546, dtype=torch.bool, device=args.device)
    mask[ix] = True
    base, extra, cal = [fold["human"][r] for r in ("base", "extra", "cal")]
    scales = {"H": response_scale(data["human"], base), "Hextra_cells": response_scale(data["human"], extra),
              **{s: response_scale(data["cynomolgus"], fold["animal"][s]["real"]) for s in ("Monkey56", "Monkey82")}}
    source = "Hextra_cells" if arm == "Hextra_cells" else arm.removesuffix("A0")
    coefficient = 0. if arm == "H" else .5 * scales["H"] / max(scales[source], 1e-8)
    if arm != "H":
        source_data = data["human"] if source == "Hextra_cells" else data["cynomolgus"]
        source_record = extra if source == "Hextra_cells" else fold["animal"][source]["null" if arm.endswith("A0") else "real"]
        context = "human_iPSC" if source == "Hextra_cells" else "cynomolgus_iPSC"
    settings = contract["training"]
    lr = settings["learning_rates"]
    optimizer, parameters = adapter.make_assay_optimizer(model, condition_lr=lr["explicit_new_condition_weights"],
        core_lr=lr["all_other_connected_trainable_parameters"], weight_decay=settings["weight_decay"],
        betas=tuple(settings["betas"]), eps=settings["epsilon"])
    old_batch = adapter.preserve_old_batch_rows(model)
    first = next(m for m in model.pert_encoder.original.modules() if isinstance(m, torch.nn.Linear))
    old_pert = first.weight.detach().clone()
    best, selected_output = calibrate(model, data["human"], cal, ix, mask, args)
    require(np.array_equal(selected_output["prediction"], data["human"][cal["control_rows"]]), "Initial masked contrast changed controls")
    selected_step, history, witnesses = 0, [], []
    calibration = [{"step": 0, "mse": best}]
    torch.save(model.state_dict(), folder / "selected.pt")
    started = time.monotonic()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        hb = batch_rows(base, seed, step, "base")
        controls = pad(data["human"][hb["control"]], ix, args.device)
        targets = torch.as_tensor(data["human"][hb["target"]], device=args.device)
        prediction = adapter.predict_assay_contrast(model, controls, mask, context="human_iPSC")["prediction"][:, ix]
        human_loss = (prediction.mean(0) - targets.mean(0)).square().mean()
        source_loss = None
        if arm != "H":
            sb = batch_rows(source_record, seed, step, "source")
            sc = pad(source_data[sb["control"]], ix, args.device)
            st = torch.as_tensor(source_data[sb["target"]], device=args.device)
            sp = adapter.predict_assay_contrast(model, sc, mask, context=context)["prediction"][:, ix]
            source_loss = (sp.mean(0) - st.mean(0)).square().mean()
            if step in (0, 2, 9):
                chosen = [(n, p) for n, p in model.named_parameters() if p.requires_grad and (
                    n == "pert_encoder.extension.weight" or n.endswith("layers.0.self_attn.v_proj.weight")
                    or (n.startswith("basal_encoder.") and n.endswith("weight")))]
                source_grad = torch.autograd.grad(coefficient * source_loss, [p for _, p in chosen], retain_graph=True, allow_unused=True)
                human_grad = torch.autograd.grad(human_loss, [p for _, p in chosen], retain_graph=True, allow_unused=True)
                detail = {}
                for (name, _), a, h in zip(chosen, source_grad, human_grad):
                    an, hn = (None if a is None else float(a.norm().cpu())), (None if h is None else float(h.norm().cpu()))
                    cosine = None if an is None or hn is None or an == 0 or hn == 0 else float((a.flatten() @ h.flatten()).cpu() / (an * hn))
                    detail[name] = {"weighted_source_norm": an, "human_norm": hn, "cosine": cosine}
                witnesses.append({"step": step + 1, "parameters": detail})
        loss = human_loss if source_loss is None else human_loss + coefficient * source_loss
        loss.backward()
        require(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in parameters if p.grad is not None), "Nonfinite training objective or gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_global_norm_clip"])
        optimizer.step()
        adapter.restore_old_batch_rows(model, old_batch)
        history.append({"step": step + 1, "human_loss": float(human_loss.detach().cpu()),
                        "source_loss": None if source_loss is None else float(source_loss.detach().cpu()),
                        "source_coefficient": coefficient, "gradient_norm": float(grad_norm.cpu())})
        if step + 1 in (50, 100, 200):
            score, output = calibrate(model, data["human"], cal, ix, mask, args)
            calibration.append({"step": step + 1, "mse": score})
            if score < best:
                best, selected_step, selected_output = score, step + 1, output
                torch.save(model.state_dict(), folder / "selected.pt")
            progress(args.out, phase="calibration", fold=fold["fold"], seed=seed, arm=arm, step=step + 1, mse=score, selected_step=selected_step)
    require(torch.equal(first.weight, old_pert) and torch.equal(model.batch_encoder.weight[:56], old_batch), "Unused original interface moved")
    if arm.startswith("Monkey"):
        require(any(x["step"] >= 3 and any(v["weighted_source_norm"] is not None and v["weighted_source_norm"] > 0
                    for k, v in x["parameters"].items() if "basal_encoder" in k or "v_proj" in k) for x in witnesses), "No actual source gradient in connected shared model")
    c = data["human"][cal["control_rows"]]
    shrink = [{"alpha": alpha, "mse": mse(c + alpha * (selected_output["prediction"] - c), data["human"][cal["target_rows"]])}
              for alpha in contract["calibration"]["post_calibration_shrink"]["alpha_grid"]]
    alpha = min(shrink, key=lambda x: (x["mse"], x["alpha"]))["alpha"]
    np.savez_compressed(folder / "calibration.npz", **selected_output)
    result = {"fold": fold["fold"], "seed": seed, "arm": arm, "train_line": fold["train_line"], "test_line": fold["test_line"],
              "selected_step": selected_step, "selected_calibration_mse": best, "selected_shrink": alpha,
              "calibration_history": calibration, "shrink_grid": shrink, "training_history": history,
              "source_scales": scales, "source_coefficient": coefficient, "source_gradient_witness": witnesses,
              "old_pert_columns_and_batch_rows_unchanged": True, "target_outcomes_scored": False,
              "model_metadata": model_metadata, "selected_weight_sha256": sha256(folder / "selected.pt"),
              "calibration_predictions_sha256": sha256(folder / "calibration.npz"), "elapsed_seconds": time.monotonic() - started}
    save(folder / "training.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def fit_folder(root, fold, seed, arm):
    return root / f"fold_{fold}_seed_{seed}_{arm}"


def train_predict(args):
    require(not args.out.exists(), "Use a new36fit output directory")
    disk_root = args.out.parent
    while not disk_root.exists():
        disk_root = disk_root.parent
    free = shutil.disk_usage(disk_root).free
    require(free > 18 * 1024**3, "Require more than18GiB free before writing36selected checkpoints")
    contract, manifest, splits, data, ix_array = load_training_inputs(args)
    torch, adapter = get_runtime()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(args.device == "cuda" and torch.cuda.is_available(), "Real model fitting requires the reviewed remote GPU runtime")
    args.out.mkdir(parents=True)
    code = [Path(__file__), ROOT / "src/cross_species/state_assay_adapter.py", ROOT / "src/cross_species/named_gene_state.py"]
    specification = {"contract": contract, "contract_sha256": args.contract_sha256,
                     "prepared_manifest": manifest, "split_manifest": read(args.splits / "manifest.json"),
                     "splits_sha256": sha256(args.splits / "splits.json"), "checkpoint_sha256": sha256(args.checkpoint),
                     "code": {str(p.relative_to(ROOT)): sha256(p) for p in code}, "initial_free_disk_bytes": free,
                     "fold_specific_test_outcomes_used_for_training_or_selection": False,
                     "all_fits_and_selections_before_target_inference": True, "actual_primary_target_units": 2,
                     "seeds_are_computational_repeats": True}
    save(args.out / "specification.json", specification)
    fits = []
    for item in splits["seed_splits"]:
        for fold in item["folds"]:
            for arm in ARMS:
                folder = fit_folder(args.out, fold["fold"], item["seed"], arm)
                folder.mkdir()
                fits.append(fit(args, contract, data, ix_array, fold, item["seed"], arm, folder))
    require(len(fits) == 36, "All36fits must complete before target prediction")
    save(args.out / "ALL_FITS_SELECTED.json", {"fits": fits, "target_outcomes_scored": False,
                                               "specification_sha256": sha256(args.out / "specification.json")})
    artifacts = {}
    ix = torch.as_tensor(ix_array, device=args.device)
    mask = torch.zeros(6546, dtype=torch.bool, device=args.device)
    mask[ix] = True
    for result in fits:
        item = next(s for s in splits["seed_splits"] if s["seed"] == result["seed"])
        fold = next(f for f in item["folds"] if f["fold"] == result["fold"])
        folder = fit_folder(args.out, result["fold"], result["seed"], result["arm"])
        require(sha256(folder / "selected.pt") == result["selected_weight_sha256"], "Selected model changed before target prediction")
        model, _ = initialize(args, result["seed"])
        model.load_state_dict(torch.load(folder / "selected.pt", map_location=args.device, weights_only=True), strict=True)
        output = infer(model, data["human"], fold["test"], ix, mask, args.device)
        base = fold["human"]["base"]
        base_delta = data["human"][base["target_rows"]].mean(0, dtype=np.float64) - data["human"][base["control_rows"]].mean(0, dtype=np.float64)
        # Test target row indices are recorded without dereferencing outcomes.
        # Baselines are losslessly factorized: frozen query NTC controls, P,
        # alpha and training-only base_delta completely specify them. Avoid
        # duplicating five large arrays for each of36fits.
        path = folder / "target.npz"
        np.savez_compressed(path, **output, selected_shrink=np.asarray(result["selected_shrink"], dtype=np.float64),
                            base_mean_delta=base_delta)
        artifacts[str(path.relative_to(args.out))] = sha256(path)
        del model
        torch.cuda.empty_cache()
    save(args.out / "ALL_TARGET_PREDICTIONS_FROZEN.json", {"artifacts": artifacts, "fits": 36, "target_units": 2,
        "fits_selected_sha256": sha256(args.out / "ALL_FITS_SELECTED.json"), "target_outcomes_scored": False,
        "contract_sha256": args.contract_sha256, "unix_time": time.time()})
    progress(args.out, phase="all_target_predictions_frozen", fits=36, target_outcomes_scored=False)


def self_test():
    # Deliberately unbalanced strata exercise Hamilton rounding and pool totals.
    strata = [("cap1", "g1", 13), ("cap1", "g2", 8), ("cap2", "g1", 20), ("cap3", "g3", 11)]
    cells, capture, guide = [], [], []
    for cap, g, n in strata:
        for i in range(n):
            cells.append(f"{cap}/{g}/{i}"); capture.append(cap); guide.append(g)
    m = {"cell_ids": np.array(cells), "capture": np.array(capture), "guide": np.array(guide)}
    a, detail = split_stratified(m, np.arange(len(cells)), ("base", "extra", "cal"), 12, 20261110, "toy", "NTC")
    b, again = split_stratified(m, np.arange(len(cells)), ("base", "extra", "cal"), 12, 20261110, "toy", "NTC")
    assert a == b and detail == again and len(set(sum(a.values(), []))) == 36
    assert all(len(v) == 12 for v in a.values())
    assert all(max(x["pool_counts"].values()) - min(x["pool_counts"].values()) <= 1 for x in detail)
    for row in detail:
        exact = 36 * row["available"] / len(cells)
        assert row["selected"] in (int(np.floor(exact)), int(np.ceil(exact)))
    animal, _ = split_stratified(m, np.arange(len(cells)), ("input", "null_target"), 16, 20261110, "animal", "NTC")
    assert not set(animal["input"]) & set(animal["null_target"])
    one, _ = split_stratified(m, np.arange(len(cells)), ("target",), 16, 20261110, "animal", "POU5F1")
    assert len(one["target"]) == 16
    for step in (0, 2, 9, 199):
        ranks = positions(128, 20261110, step, "source|target")
        assert len(set(ranks)) == 64 and np.array_equal(ranks, positions(128, 20261110, step, "source|target"))
    try:
        split_stratified(m, np.arange(2), ("base", "extra", "cal"), 12, 1, "toy", "NTC")
    except ValueError:
        pass
    else:
        raise AssertionError("Insufficient metadata pool must fail")
    print(json.dumps({"artificial_metadata_tests": "PASS", "expression_data_opened": False,
                      "checks": ["Hamilton proportional quotas", "balanced exact pools", "no overlap", "same-seed determinism", "two-pool A/A0", "matched step ranks", "insufficient-budget rejection"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-splits", "train-predict"):
        p = sub.add_parser(command)
        for key in ("prepared", "contract", "out"):
            p.add_argument("--" + key, type=Path, required=True)
        p.add_argument("--contract-sha256", required=True)
        if command == "train-predict":
            for key in ("splits", "checkpoint", "state-source"):
                p.add_argument("--" + key, type=Path, required=True)
            p.add_argument("--device", default="cuda")
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "prepare-splits":
        prepare_splits(args)
    elif args.command == "train-predict":
        train_predict(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
