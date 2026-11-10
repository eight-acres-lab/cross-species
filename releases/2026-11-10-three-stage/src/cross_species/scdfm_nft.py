# Adapted from NVIDIA DiffusionNFT (Apache-2.0), commit
# cbb14f84b8312b620390dfcbe2fab69c1383b104, scripts/train_nft_sd3.py:910-970.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0;
# https://www.apache.org/licenses/LICENSE-2.0
"""Negative-aware flow fine-tuning adapted to scDFM's noise-to-data convention.

The loss follows the pinned NFT positive/implicit-negative construction. scDFM
uses tau=0 noise and tau=1 data, so NFT t=1-tau and NFT velocity v=-u.
The reference term is velocity MSE, not an exact likelihood or policy KL.
"""
from __future__ import annotations

import torch


def group_advantages(rewards):
    """One condition/population group per round; global and group std coincide."""
    if rewards.ndim != 1 or len(rewards) < 2 or not torch.isfinite(rewards).all():
        raise ValueError("Need finite rewards for two or more generated populations")
    values = rewards.detach().double()
    return ((values - values.mean()) / (values.std(unbiased=False) + 1e-4)).float()


def negative_aware_loss(clean, noised, tau, current_velocity, old_velocity, reference_velocity,
                        advantages, *, mixture_beta=1., advantage_clip=5., reference_weight=1e-4):
    """Population axis first; each population comprises all its cell/gene values."""
    if (clean.shape != noised.shape or clean.shape != current_velocity.shape
            or clean.shape != old_velocity.shape or clean.shape != reference_velocity.shape):
        raise ValueError("Every NFT tensor must share population and feature axes")
    if clean.ndim < 2 or advantages.shape != (len(clean),) or mixture_beta <= 0 or advantage_clip <= 0:
        raise ValueError("Invalid NFT population dimensions or coefficients")
    if reference_weight < 0 or not torch.isfinite(advantages).all():
        raise ValueError("NFT reference weight and advantages must be valid")
    if tau.shape != (len(clean),) or not torch.isfinite(tau).all() or (tau < 0).any() or (tau > 1).any():
        raise ValueError("tau must be a valid noise-to-data time per population")
    expand = (len(clean),) + (1,) * (clean.ndim - 1)
    remaining = (1 - tau).reshape(expand)
    target, old, reference = clean.detach(), old_velocity.detach(), reference_velocity.detach()
    positive_velocity = mixture_beta * current_velocity + (1 - mixture_beta) * old
    negative_velocity = (1 + mixture_beta) * old - mixture_beta * current_velocity
    positive_clean = noised + remaining * positive_velocity
    negative_clean = noised + remaining * negative_velocity
    axes = tuple(range(1, clean.ndim))
    with torch.no_grad():
        positive_weight = (positive_clean.double() - target.double()).abs().mean(axes, keepdim=True).clamp_min(1e-5)
        negative_weight = (negative_clean.double() - target.double()).abs().mean(axes, keepdim=True).clamp_min(1e-5)
    positive_loss = ((positive_clean - target).square() / positive_weight).mean(axes)
    negative_loss = ((negative_clean - target).square() / negative_weight).mean(axes)
    reward_weight = ((advantages.detach().clamp(-advantage_clip, advantage_clip) / advantage_clip) / 2 + .5).clamp(0, 1)
    policy = (advantage_clip * (reward_weight * positive_loss + (1 - reward_weight) * negative_loss) / mixture_beta).mean()
    reference_mse = (current_velocity - reference).square().mean()
    total = policy + reference_weight * reference_mse
    return total, {"negative_aware_objective": policy.detach(), "reference_velocity_mse": reference_mse.detach(),
                   "reward_weight": reward_weight, "positive_reconstruction_loss": positive_loss.detach(),
                   "implicit_negative_reconstruction_loss": negative_loss.detach()}


def old_policy_decay(update):
    """Pinned NFT return_decay(decay_type=1)."""
    return min(.001 * update, .5)


@torch.no_grad()
def refresh_old(old, live, update):
    decay = old_policy_decay(update)
    for old_parameter, live_parameter in zip(old.parameters(), live.parameters(), strict=True):
        old_parameter.mul_(decay).add_(live_parameter, alpha=1 - decay)
    return decay


def euler_endpoint(model, controls, gene_tokens, perturbation_tokens, *, steps=20, generator=None,
                   initial_noise=None, retain_grad=False, checkpoint_steps=False):
    """Noise-start scDFM ODE with an explicitly fixed Euler solver."""
    if steps <= 0:
        raise ValueError("Euler integration needs a positive fixed step count")
    model.eval()
    state = (torch.randn(controls.shape, device=controls.device, dtype=controls.dtype, generator=generator)
             if initial_noise is None else initial_noise.clone())
    if state.shape != controls.shape:
        raise ValueError("Noise and control gene axes differ")
    with torch.set_grad_enabled(retain_grad):
        for step in range(steps):
            tau = torch.full((len(controls),), step / steps, device=controls.device, dtype=controls.dtype)
            if retain_grad and checkpoint_steps:
                from torch.utils.checkpoint import checkpoint
                velocity = checkpoint(lambda value, current_tau: model(gene_tokens, value, current_tau,
                    controls, perturbation_tokens, gene_tokens, mode="predict_y"), state, tau, use_reentrant=False)
            else:
                velocity = model(gene_tokens, state, tau, controls, perturbation_tokens, gene_tokens, mode="predict_y")
            state = state + velocity / steps
    if not torch.isfinite(state).all():
        raise ValueError("The released flow produced a nonfinite terminal state")
    return state
