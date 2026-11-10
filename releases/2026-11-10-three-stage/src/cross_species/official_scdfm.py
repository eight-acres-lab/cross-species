"""Strict loading and native flow-matching checks for the released scDFM model."""
from __future__ import annotations

import contextlib
import inspect
import io
from pathlib import Path
import sys

import numpy as np
import torch

from cross_species.official_state import sha256_file

SCDFM_SOURCE_COMMIT = "8de47e7d3d443939f93f768f7902c66e0e82eda2"
COMBOSCIPLEX_SHA256 = "7d0744740d435f49c315b2d3d2b16e21004e2d463886d87fe3c18e8ccec6a2d9"


def load_official_scdfm(checkpoint, vendor_root, mask_path, *, device="cpu"):
    checkpoint, vendor_root, mask_path = map(Path, [checkpoint, vendor_root, mask_path])
    if sha256_file(checkpoint) != COMBOSCIPLEX_SHA256:
        raise ValueError("Released scDFM checkpoint SHA256 mismatch")
    mask = torch.load(mask_path, map_location="cpu", weights_only=True)
    if not isinstance(mask, torch.Tensor) or mask.shape != (5004, 5004) or mask.dtype != torch.bool:
        raise ValueError("Require the reconstructed official 5004-token coexpression mask")
    source = vendor_root / "src/models/origin/model.py"
    if not source.is_file():
        raise ValueError("Pinned official scDFM source is missing")
    sys.path.insert(0, str(vendor_root.resolve()))
    from src.models.origin.model import model as OfficialModel
    archive = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in archive:
        raise ValueError("Released checkpoint does not contain model_state_dict")
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        model = OfficialModel(ntoken=6000, d_model=512, nhead=8, nlayers=4, dropout=.1,
            fusion_method="differential_perceiver", perturbation_function="drug",
            use_perturbation_interaction=True, mask_path=str(mask_path))
    status = model.load_state_dict(archive["model_state_dict"], strict=True)
    model.to(device)
    metadata = {"checkpoint_sha256": sha256_file(checkpoint), "source_commit": SCDFM_SOURCE_COMMIT,
        "official_class_file": inspect.getfile(OfficialModel), "official_class_file_sha256": sha256_file(inspect.getfile(OfficialModel)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "state_tensors": len(archive["model_state_dict"]), "strict_missing_keys": status.missing_keys,
        "strict_unexpected_keys": status.unexpected_keys, "checkpoint_iteration": archive.get("iteration"),
        "mask_sha256": sha256_file(mask_path), "mask_shape": list(mask.shape),
        "mask_provenance": "Reconstructed from official training cells with published/code k30, absolute positive/negative correlation; original mask bytes were absent from checkpoint.",
        "architecture": {"ntoken": 6000, "d_model": 512, "heads": 8, "layers": 4, "dropout": .1,
                         "fusion": "differential_perceiver", "perturbation_function": "drug"}}
    return model, metadata


def native_flow_loss(model, controls, targets, gene_tokens, perturbation_tokens, *, generator=None, gamma=.5):
    """Official Gaussian affine conditional FM plus one-step multi-kernel MMD."""
    n = len(controls)
    time = torch.rand(n, device=controls.device, dtype=controls.dtype, generator=generator)
    noise = torch.randn(controls.shape, device=controls.device, dtype=controls.dtype, generator=generator)
    xt = (1 - time[:, None]) * noise + time[:, None] * targets
    velocity = targets - noise
    prediction = model(gene_tokens, xt, time, controls, perturbation_tokens, gene_tokens, mode="predict_y")
    fm = (prediction - velocity).square().mean()
    endpoint = xt + prediction * (1 - time[:, None])
    # Exact published train_step kernel scales; median bandwidth is detached.
    target_distances = torch.cdist(targets, targets).square()
    off_diagonal = target_distances[~torch.eye(n, device=targets.device, dtype=torch.bool)]
    median = off_diagonal.detach().median().clamp_min(1e-12)
    dxx, dyy, dxy = torch.cdist(endpoint, endpoint).square(), target_distances, torch.cdist(endpoint, targets).square()
    mmd_values = []
    for scale in [.5, 1., 2., 4.]:
        denominator = 2 * scale * median + 1e-12
        kxx, kyy, kxy = [torch.exp(-distance / denominator) for distance in [dxx, dyy, dxy]]
        mmd_values.append((kxx.sum() - kxx.diag().sum()) / (n * (n - 1) + 1e-12)
            + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1) + 1e-12) - 2 * kxy.mean())
    mmd = torch.stack(mmd_values).mean()
    return fm + gamma * mmd, {"flow_mse": fm.detach(), "one_step_mmd2_u": mmd.detach(),
                             "time": time.detach(), "noise": noise.detach(), "noised_target": xt.detach(),
                             "prediction": prediction}


def flow_optimizer_probe(model, controls, targets, gene_tokens, perturbation_tokens, *, lr=1e-5, seed=20261108):
    if controls.shape != targets.shape or len(controls) < 2:
        raise ValueError("Need two unpaired real populations on an identical gene axis")
    model.train()
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = native_flow_loss(model, controls, targets, gene_tokens, perturbation_tokens,
        generator=torch.Generator(device=controls.device).manual_seed(seed))
    loss.backward()
    active = [(name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None]
    if not torch.isfinite(loss) or any(not torch.isfinite(parameter.grad).all() for _, parameter in active):
        raise ValueError("Nonfinite released flow native objective or gradient")
    witness_name, witness = active[0]
    before = witness.detach().clone()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    if controls.device.type == "cuda":
        torch.cuda.synchronize()
    return {"optimizer_steps": 1, "native_total_loss": float(loss.detach().cpu()),
            "native_flow_mse": float(diagnostics["flow_mse"].cpu()), "native_one_step_mmd2_u": float(diagnostics["one_step_mmd2_u"].cpu()),
            "gamma": .5, "optimizer": "Adam", "lr": lr, "gradient_norm": float(gradient_norm.detach().cpu()),
            "gradient_receiving_parameters": sum(parameter.numel() for _, parameter in active), "gradient_receiving_tensors": len(active),
            "updated_parameter_witness": witness_name, "updated_parameter_l2_change": float((witness-before).detach().square().sum().sqrt().cpu()),
            "scope": "Released scDFM architecture/weights, real-cell native Gaussian affine flow-matching plus MMD development update; no held-out improvement claim."}
