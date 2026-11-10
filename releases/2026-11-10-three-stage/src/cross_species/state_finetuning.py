"""Native State energy fine-tuning with deterministic reference-output anchoring.

This module contains supervised distributional calibration, with no policy
likelihood, policy-gradient estimator, or diffusion/flow RL claim.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F


DEVELOPMENT = {"4-1BBL", "IFN-gamma", "IL-2", "TGF-beta1"}


def condition_split(conditions, allowed, *, seed=20261108):
    eligible = sorted(set(conditions) & set(allowed))
    if len(eligible) >= 30:
        counts = (16, 6, 8)
    elif len(eligible) >= 10:
        counts = (4, 2, 4)
    else:
        raise ValueError("At least ten metadata-complete official test cytokines are required")
    development = sorted(set(eligible) & DEVELOPMENT)
    key = lambda name: hashlib.sha256((str(seed) + "|state-parse-context|" + name).encode()).hexdigest()
    remaining = sorted(set(eligible) - set(development), key=key)
    n_train, n_calib, n_test = counts
    train = development + remaining[:n_train - len(development)]
    remaining = remaining[n_train - len(development):]
    calib, test = remaining[:n_calib], remaining[n_calib:n_calib + n_test]
    unused = sorted(set(eligible) - set(train + calib + test))
    return {"train": train, "calib": calib, "test": test, "unused": unused,
            "seed": seed, "counts": counts, "development_forced_into_train": development}


def population_index(payload, conditions, donors, cell_count):
    labels, persons = np.asarray(payload["cytokine"]).astype(str), np.asarray(payload["donor"]).astype(str)
    control = np.asarray(payload["is_control"], dtype=bool)
    populations = {}
    for condition in conditions:
        for donor in donors:
            basal = np.flatnonzero(control & (persons == donor))
            targets = np.flatnonzero((~control) & (labels == condition) & (persons == donor))
            if len(basal) < cell_count or len(targets) < cell_count:
                raise ValueError("Insufficient distinct real cells for " + condition + "/" + donor)
            populations[(condition, donor)] = {"controls": basal, "targets": targets}
    return populations


def training_schedule(populations, train_conditions, donors, *, steps, cell_count, seed):
    rng, result = np.random.default_rng(seed), []
    for step in range(steps):
        condition, donor = str(rng.choice(train_conditions)), str(rng.choice(donors))
        available = populations[(condition, donor)]
        result.append({"step": step, "condition": condition, "donor": donor,
                       "control_rows": rng.choice(available["controls"], cell_count, replace=False),
                       "target_rows": rng.choice(available["targets"], cell_count, replace=False)})
    return result


def deterministic_anchor(model, reference, batch):
    previous = model.training
    model.eval()
    try:
        live = model(batch, padded=False)
        with torch.no_grad():
            target = reference(batch, padded=False)
        return F.mse_loss(live, target)
    finally:
        model.train(previous)


def fit_arm(model, reference, expression, schedule, perturbations, *, weight,
            device, lr=1e-5, seed=20261108):
    """Only the explicitly listed training populations are read for optimization."""
    if not schedule or weight < 0:
        raise ValueError("A nonempty training schedule and nonnegative anchor weight are required")
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model.train()
    reference.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=lr, weight_decay=.0005)
    history, initial_anchor, started = [], None, time.monotonic()
    for record in schedule:
        basal = torch.as_tensor(expression[record["control_rows"]], dtype=torch.float32, device=device)
        target = torch.as_tensor(expression[record["target_rows"]], dtype=torch.float32, device=device)
        pert = perturbations[record["condition"]].to(device).expand(len(basal), -1).contiguous()
        batch = {"ctrl_cell_emb": basal, "pert_emb": pert}
        optimizer.zero_grad(set_to_none=True)
        if initial_anchor is None:
            with torch.no_grad():
                initial_anchor = float(deterministic_anchor(model, reference, batch).cpu())
            if initial_anchor > 1e-8:
                raise ValueError("Reference anchor is not zero at the common initial checkpoint")
        prediction = model(batch, padded=False)
        energy = model._compute_distribution_loss(prediction, target).mean()
        # Compute this same extra path for lambda zero, preserving the compute/RNG schedule.
        anchor = deterministic_anchor(model, reference, batch)
        loss = energy + weight * anchor
        loss.backward()
        active = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if not torch.isfinite(loss) or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise ValueError("Nonfinite State fine-tuning objective or gradient")
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        history.append({"step": record["step"] + 1, "condition": record["condition"],
                        "donor": record["donor"], "native_energy": float(energy.detach().cpu()),
                        "reference_output_mse": float(anchor.detach().cpu()),
                        "objective": float(loss.detach().cpu()), "gradient_norm": float(gradient.detach().cpu())})
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"history": history, "initial_reference_anchor": initial_anchor,
            "gradient_receiving_parameter_tensors": len(active),
            "gradient_receiving_parameters": sum(parameter.numel() for parameter in active),
            "requires_grad_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "updates": len(schedule), "lambda_anchor": weight, "seconds": time.monotonic() - started}


def distribution_metrics(prediction, target, *, bandwidth):
    from scipy.spatial.distance import cdist
    xy = cdist(prediction, target)
    xx, yy = cdist(prediction, prediction), cdist(target, target)
    n, m = len(prediction), len(target)
    energy = 2 * xy.mean() - xx.sum() / (n * (n - 1)) - yy.sum() / (m * (m - 1))
    kxx, kyy, kxy = [np.exp(-np.square(distance) / (2 * bandwidth ** 2)) for distance in [xx, yy, xy]]
    mmd = ((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
           + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return {"energy_u": float(energy), "rbf_mmd2_u": float(mmd),
            "mean_mse": float(np.square(prediction.mean(axis=0) - target.mean(axis=0)).mean()),
            "variance_mae": float(np.abs(prediction.var(axis=0, ddof=1) - target.var(axis=0, ddof=1)).mean())}


def evaluate_model(model, expression, populations, conditions, donors, perturbations, *,
                   device, cell_count, bandwidth, save_predictions=False):
    model.eval()
    rows, predictions, actual_targets, ids = [], [], [], []
    with torch.no_grad():
        for condition in conditions:
            for donor in donors:
                available = populations[(condition, donor)]
                control_rows, target_rows = available["controls"][:cell_count], available["targets"][:cell_count]
                basal = torch.as_tensor(expression[control_rows], dtype=torch.float32, device=device)
                target = expression[target_rows]
                pert = perturbations[condition].to(device).expand(len(basal), -1).contiguous()
                prediction = model({"ctrl_cell_emb": basal, "pert_emb": pert}, padded=False).cpu().numpy()
                metrics = distribution_metrics(prediction, target, bandwidth=bandwidth)
                metrics["native_energy"] = float(model._compute_distribution_loss(
                    torch.as_tensor(prediction, device=device),
                    torch.as_tensor(target, device=device)).mean().cpu())
                rows.append({"condition": condition, "donor": donor, **metrics})
                if save_predictions:
                    predictions.append(prediction)
                    actual_targets.append(target)
                    ids.append({"condition": condition, "donor": donor,
                                "control_rows": control_rows, "target_rows": target_rows})
    grouped = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        grouped[condition] = {metric: float(np.mean([row[metric] for row in subset]))
                              for metric in ["native_energy", "energy_u", "rbf_mmd2_u", "mean_mse", "variance_mae"]}
    macro = {metric: float(np.mean([row[metric] for row in grouped.values()])) for metric in next(iter(grouped.values()))}
    result = {"macro": macro, "by_condition": grouped, "by_condition_donor": rows}
    if save_predictions:
        result.update(predictions=np.stack(predictions), targets=np.stack(actual_targets), query_indices=ids)
    return result
