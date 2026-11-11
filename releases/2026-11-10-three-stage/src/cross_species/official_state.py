"""Load the released Arc State architecture and weights without redefining it.

State ST is a set-to-set perturbation transformer, not a flow velocity network.
This module exposes loading and small operator checks; it does not implement RL.
"""
from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import inspect
import io
from pathlib import Path

import numpy as np
import torch


STATE_SOURCE_COMMIT = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"
PARSE_SPLIT0_SHA256 = "244acf009b4b0e1a87451e8a155ce2bbcaa3d4d1ad25910e9d735ae76877fccb"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_state(checkpoint_path, *, device="cpu", expected_sha256=PARSE_SPLIT0_SHA256):
    """Instantiate the official class from released hparams and strictly load it."""
    path = Path(checkpoint_path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Official checkpoint SHA256 differs from the pinned artifact")
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not {"state_dict", "hyper_parameters"} <= set(checkpoint):
        raise ValueError("Released State checkpoint must include state_dict and hyper_parameters")
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        model = StateTransitionPerturbationModel(**deepcopy(checkpoint["hyper_parameters"]))
    status = model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    metadata = {
        "checkpoint_sha256": actual, "source_commit": STATE_SOURCE_COMMIT,
        "official_class": type(model).__module__ + "." + type(model).__name__,
        "official_class_file": inspect.getfile(type(model)),
        "official_class_file_sha256": sha256_file(inspect.getfile(type(model))),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "strict_missing_keys": status.missing_keys, "strict_unexpected_keys": status.unexpected_keys,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "input_dim": model.input_dim, "output_dim": model.output_dim,
        "pert_dim": model.pert_dim, "training_cell_set_len": model.cell_sentence_len,
        "architecture_scope": "Released set-to-set State ST; no flow-matching or RL algorithm is implied.",
    }
    return model, metadata


def load_perturbation_map(path, expected_dim=91):
    """Read the official mapping with restricted NumPy scalar allowlisting."""
    from numpy._core.multiarray import scalar
    allowed = [scalar, np.dtype, np.dtypes.StrDType, np.dtypes.Float64DType,
               np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Official perturbation mapping must be a dictionary")
    result = {}
    for key, value in payload.items():
        key = str(key)
        if key in result or not isinstance(value, torch.Tensor) or value.shape != (expected_dim,):
            raise ValueError("Invalid or duplicate perturbation representation")
        value = value.detach().float().clone()
        if not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all() or value.sum() != 1:
            raise ValueError("Official perturbation representation must be finite one-hot")
        result[key] = value
    return result


def select_real_population_pair(payload, target_condition, max_cells):
    """Select distinct actual control/treated cells sharing a donor and cell type."""
    required = {"expression", "is_control", "cytokine", "cell_type", "donor", "cell_ids", "gene_ids"}
    if not required <= set(payload):
        raise ValueError("Missing genuine-cell smoke fields: " + ", ".join(sorted(required - set(payload))))
    expression = np.asarray(payload["expression"])
    if expression.ndim != 2 or not np.isfinite(expression).all():
        raise ValueError("Expression must be a finite actual cell-by-feature matrix")
    n = len(expression)
    for name in required - {"expression", "gene_ids"}:
        if np.asarray(payload[name]).shape != (n,):
            raise ValueError("Cell metadata must have one entry per actual cell")
    if len(set(np.asarray(payload["cell_ids"]).astype(str))) != n:
        raise ValueError("Duplicate actual cells cannot inflate the smoke population")
    control = np.asarray(payload["is_control"], dtype=bool)
    condition, donor, cell_type = [np.asarray(payload[name]).astype(str) for name in ["cytokine", "donor", "cell_type"]]
    for person, lineage in sorted(set(zip(donor.tolist(), cell_type.tolist()))):
        basal = np.flatnonzero(control & (donor == person) & (cell_type == lineage))
        target = np.flatnonzero((~control) & (condition == target_condition) & (donor == person) & (cell_type == lineage))
        size = min(len(basal), len(target), int(max_cells))
        if size >= 2:
            return basal[:size], target[:size], {"donor": person, "cell_type": lineage,
                                               "cells_per_population": size,
                                               "pairing": "Unpaired actual cell populations; same donor/cell type, no cell-to-cell matching claim."}
    raise ValueError("No donor/cell-type stratum has at least two real controls and target cells")


def population_operator_probe(model, controls, targets, perturbation, *, device, backward=True):
    """Run the model's native forward and energy loss; never optimize parameters."""
    if controls.shape != targets.shape or controls.ndim != 2 or controls.shape[1] != model.input_dim:
        raise ValueError("Control/target operator populations must share the released feature axis")
    model.eval()
    basal = torch.as_tensor(controls, dtype=torch.float32, device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    # MPS linear kernels require actual storage for this repeated condition batch.
    pert = perturbation.to(device=device, dtype=torch.float32).expand(len(basal), -1).contiguous()
    model.zero_grad(set_to_none=True)
    prediction = model({"ctrl_cell_emb": basal, "pert_emb": pert}, padded=False)
    loss = model._compute_distribution_loss(prediction, target).mean()
    if backward:
        loss.backward()
    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise ValueError("Native forward/loss contains nonfinite values")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("Native backward contains nonfinite gradients")
    return prediction.detach().cpu().numpy(), {
        "native_energy_loss": float(loss.detach().cpu()), "backward": bool(backward),
        "gradient_parameter_tensors": len(gradients), "finite_output_and_gradients": True,
        "gradient_receiving_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.grad is not None),
        "optimizer_steps": 0, "variable_cell_set_inference": True,
        "scope": "Operator and gradient checks on unpaired cell populations using released State weights.",
    }


def population_optimizer_probe(model, controls, targets, perturbation, *, device, steps=1, lr=1e-5):
    """Perform actual full-parameter native-loss updates for hardware verification.

    These development updates are not a held-out performance experiment. A formal
    benchmark must reload the original checkpoint after this probe.
    """
    if steps < 1:
        raise ValueError("At least one optimizer step is required")
    model.train()
    basal = torch.as_tensor(controls, dtype=torch.float32, device=device)
    target = torch.as_tensor(targets, dtype=torch.float32, device=device)
    pert = perturbation.to(device=device, dtype=torch.float32).expand(len(basal), -1).contiguous()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.0005)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model({"ctrl_cell_emb": basal, "pert_emb": pert}, padded=False)
        loss = model._compute_distribution_loss(prediction, target).mean()
        loss.backward()
        gradients = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
        if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for _, p in gradients):
            raise ValueError("Native optimizer probe contains nonfinite loss or gradients")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        parameter_name, witness = gradients[0]
        before = witness.detach().clone()
        optimizer.step()
        change = float((witness.detach() - before).square().sum().sqrt().cpu())
        history.append({"step": step + 1, "native_energy_loss": float(loss.detach().cpu()),
                        "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                        "gradient_parameter_tensors": len(gradients),
                        "gradient_receiving_parameters": sum(parameter.numel() for _, parameter in gradients),
                        "updated_parameter_witness": parameter_name,
                        "updated_parameter_l2_change": change})
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device) == "mps":
        torch.mps.synchronize()
    return prediction.detach().cpu().numpy(), {
        "optimizer_steps": steps, "optimizer": "AdamW", "lr": lr, "weight_decay": .0005,
        "native_update_history": history, "finite_output_and_gradients": True,
        "scope": "Actual native-loss full-parameter development updates; not held-out performance, RL, or a formal tuned checkpoint.",
    }
