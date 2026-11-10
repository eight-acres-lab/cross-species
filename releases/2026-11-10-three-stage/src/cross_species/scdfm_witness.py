"""Witness-only sensitivity following PerturbCellRL v2 Appendix C.

The omitted expression-support/pathway terms, fixed 64-cell real pools, and
retained scDFM Adam optimizer prevent a full-reproduction claim.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np
import torch

from cross_species.official_scdfm import native_flow_loss
from cross_species.scdfm_feedback import frozen_velocity_matching_autograd, model_digest


def group_standardize(values, group_ids):
    if values.ndim != 1 or group_ids.shape != values.shape or not torch.isfinite(values).all():
        raise ValueError("Need one finite scalar and one event-group ID per endpoint")
    result = torch.empty_like(values)
    for group in torch.unique(group_ids):
        take = group_ids == group
        if int(take.sum()) < 2:
            raise ValueError("Each independent control event needs at least two rollouts")
        std = values[take].std(unbiased=False)
        if not torch.isfinite(std) or std <= 0:
            raise ValueError("Witness has zero/nonfinite within-event standard deviation")
        result[take] = (values[take] - values[take].mean()) / std
    return result


@torch.no_grad()
def witness_preferences(raw_generated, real_reference, group_ids):
    """Euclidean energy witness with the entire query sampling event excluded."""
    if raw_generated.ndim != 2 or real_reference.ndim != 2 or raw_generated.shape[1] != real_reference.shape[1]:
        raise ValueError("Generated and measured cells must share a gene axis")
    if group_ids.shape != (len(raw_generated),) or len(real_reference) < 2:
        raise ValueError("Invalid event groups or measured reference pool")
    if not torch.isfinite(raw_generated).all() or not torch.isfinite(real_reference).all():
        raise ValueError("Nonfinite expression")
    generated, real = raw_generated.float().clamp_min(0), real_reference.float()
    distances = torch.cdist(generated, generated)
    eligible = group_ids[:, None] != group_ids[None, :]
    if (eligible.sum(1) == 0).any():
        raise ValueError("Need at least two independent control-sampling events")
    witness = (distances * eligible).sum(1) / eligible.sum(1) - torch.cdist(generated, real).mean(1)
    pairs = torch.pdist(real).square()
    scale = pairs.median().sqrt()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Measured reference distance scale must be positive")
    reward = group_standardize(witness / scale, group_ids)
    advantage = group_standardize(reward, group_ids)
    preference = .5 + advantage.clamp(-2, 2) / 4
    return preference, {"witness": witness, "reward": reward, "advantage": advantage,
                        "real_reference_scale": scale, "eligible_generated_count": eligible.sum(1)}


def witness_velocity_loss(live, old, reference, target_velocity, preference, *, gamma=1., clip_c=2., reference_weight=1.):
    if live.ndim != 2 or any(x.shape != live.shape for x in [old, reference, target_velocity]):
        raise ValueError("Velocity tensors must share cell and gene axes")
    if preference.shape != (len(live),) or not torch.isfinite(preference).all() or (preference < 0).any() or (preference > 1).any():
        raise ValueError("Need one preference in [0,1] per generated cell")
    live, old, reference, target = live.float(), old.detach().float(), reference.detach().float(), target_velocity.detach().float()
    positive = gamma * live + (1 - gamma) * old
    negative = -gamma * live + (1 + gamma) * old
    per_cell = preference.detach() * (positive-target).square().mean(1) + (1-preference.detach()) * (negative-target).square().mean(1)
    policy = clip_c * per_cell.mean()
    penalty = (live-reference).square().mean()
    return policy + reference_weight * penalty, {"policy": policy.detach(), "reference_velocity_mse": penalty.detach()}


def autocast_for(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()


@torch.no_grad()
def rk4_endpoint(model, controls, genes, perturbations, initial_noise):
    """Author torchdiffeq RK4 on exactly twenty grid points (19 intervals)."""
    from torchdiffeq import odeint
    model.eval()
    with autocast_for(controls.device):
        def velocity(t, value):
            times = t.expand(len(value)).contiguous()
            return model(genes, value, times, controls, perturbations, genes, mode="predict_y").float()
        result = odeint(velocity, initial_noise.float(), torch.linspace(0, 1, 20, device=controls.device),
                        method="rk4", atol=1e-4, rtol=1e-4)[-1]
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite RK4 terminal state")
    return result


@torch.no_grad()
def update_eval_ema(ema, live, zero_based_step):
    decay = min((1 + zero_based_step) / (10 + zero_based_step), .9)
    for left, right in zip(ema.parameters(), live.parameters(), strict=True):
        left.mul_(decay).add_(right, alpha=1-decay)
    return decay


def fit_witness(model, old, reference, ema, matrix, schedule, pools, columns, token_ids, *, method, device, seed=0):
    if method not in {"witness", "native_fm"}:
        raise ValueError("Unknown fixed sensitivity arm")
    model.eval()
    for fixed in [old, reference, ema]:
        fixed.eval().requires_grad_(False)
    reference_before = model_digest(reference)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-7)
    history, start, witness_parameter, before, initial_difference = [], time.monotonic(), None, None, None
    gene_vector = torch.as_tensor(token_ids[columns], device=device)
    for step, record in enumerate(schedule):
        controls = torch.as_tensor(matrix[record["control_rows"]][:, columns].toarray(), device=device, dtype=torch.float32)
        real = torch.as_tensor(matrix[pools[record["condition"]]][:, columns].toarray(), device=device, dtype=torch.float32)
        genes = gene_vector.expand(64, -1).contiguous()
        perturbations = torch.as_tensor(record["drug_ids"], device=device).expand(64, -1).contiguous()
        generator = torch.Generator(device=device).manual_seed(seed + 100000 * step)
        group_ids = torch.arange(8, device=device).repeat_interleave(8)
        if method == "witness":
            initial = torch.randn(controls.shape, device=device, generator=generator)
            clean = rk4_endpoint(old, controls, genes, perturbations, initial).detach()
            preference, reward_info = witness_preferences(clean, real, group_ids)
        else:
            clean, preference, reward_info = None, None, None
        optimizer.zero_grad(set_to_none=True)
        loss_values, penalty_values, objective_values = [], [], []
        for offset in [0, 32]:
            sl = slice(offset, offset+32)
            basal, g, p = controls[sl], genes[sl], perturbations[sl]
            if method == "witness":
                # Fresh independent noise/time per endpoint, distinct from ODE initial noise.
                epsilon = torch.randn(clean[sl].shape, device=device, generator=generator)
                tau = torch.rand(32, device=device, generator=generator)
                xt = (1-tau[:, None]) * epsilon + tau[:, None] * clean[sl]
                target = clean[sl] - epsilon
                with autocast_for(device):
                    u_old = frozen_velocity_matching_autograd(old, g, xt, tau, basal, p)
                    u_ref = frozen_velocity_matching_autograd(reference, g, xt, tau, basal, p)
                    u_live = model(g, xt, tau, basal, p, g, mode="predict_y")
                if initial_difference is None:
                    initial_difference = float((u_live.float()-u_old.float()).abs().max().detach().cpu())
                    if initial_difference > 1e-6:
                        raise ValueError("Initial live/old vector fields differ")
                loss, info = witness_velocity_loss(u_live, u_old, u_ref, target, preference[sl])
                primary = info["policy"]
            else:
                targets = real[torch.as_tensor(record["target_pool_order"][offset:offset+32], device=device)]
                with autocast_for(device):
                    native, native_info = native_flow_loss(model, basal, targets, g, p, generator=generator, gamma=.5)
                    u_ref = frozen_velocity_matching_autograd(reference, g, native_info["noised_target"], native_info["time"], basal, p)
                penalty = (native_info["prediction"].float()-u_ref.float()).square().mean()
                loss, info, primary = native.float()+penalty, {"reference_velocity_mse": penalty.detach()}, native.detach()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sensitivity objective")
            (loss/2).backward()
            loss_values.append(float(loss.detach().cpu())); penalty_values.append(float(info["reference_velocity_mse"].cpu())); objective_values.append(float(primary.cpu()))
        active = [(name,p) for name,p in model.named_parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for _,p in active):
            raise ValueError("Nonfinite sensitivity gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        if witness_parameter is None:
            witness_name, witness_parameter = active[0]
            before = witness_parameter.detach().clone()
        optimizer.step()
        old.load_state_dict(model.state_dict(), strict=True)
        decay = update_eval_ema(ema, model, step)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        record_out = {"update": step+1, "condition": record["condition"], "loss": float(np.mean(loss_values)),
            "native_or_nft_objective": float(np.mean(objective_values)), "reference_velocity_mse": float(np.mean(penalty_values)),
            "gradient_norm_before_clip": float(norm.detach().cpu()), "eval_ema_decay": decay,
            "elapsed_seconds": time.monotonic()-start}
        if reward_info is not None:
            record_out.update(witness_mean=float(reward_info["witness"].mean().cpu()),
                              real_reference_scale=float(reward_info["real_reference_scale"].cpu()),
                              preference_min=float(preference.min().cpu()), preference_max=float(preference.max().cpu()))
        history.append(record_out)
        if (step+1) % 5 == 0 or len(schedule) == 1:
            print({"method": method, **record_out}, flush=True)
    if reference_before != model_digest(reference):
        raise ValueError("Original reference model changed")
    return {"method": method, "updates": len(schedule), "history": history, "wall_seconds": time.monotonic()-start,
            "initial_live_old_velocity_max_difference": initial_difference, "frozen_reference_sha256": reference_before,
            "actual_gradient_receiving_parameters": sum(p.numel() for _,p in active),
            "updated_parameter_witness": witness_name, "updated_parameter_l2_change": float((witness_parameter-before).square().sum().sqrt().detach().cpu())}
