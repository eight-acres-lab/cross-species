"""Fixed-information scDFM continuation, NFT adaptation and terminal-gradient control."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from cross_species.official_scdfm import native_flow_loss
from cross_species.scdfm_nft import euler_endpoint, group_advantages, negative_aware_loss, refresh_old


def energy_u(prediction, target):
    n, m = len(prediction), len(target)
    if min(n, m) < 2:
        raise ValueError("Energy U requires two actual points per population")
    xx, yy, xy = torch.cdist(prediction, prediction), torch.cdist(target, target), torch.cdist(prediction, target)
    return 2 * xy.mean() - xx.sum() / (n * (n - 1)) - yy.sum() / (m * (m - 1))


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def seeded_generator(device, seed, update, rollout, purpose):
    return torch.Generator(device=device).manual_seed(seed + 100000 * update + 100 * rollout + purpose)


def frozen_velocity_matching_autograd(model, genes, noised, times, basal, perturbations):
    """Match live forward kernels without retaining gradients for frozen weights.

    CUDA may use different matrix kernels for grad-enabled and no-grad linear
    layers. Detached parameter views enable the same forward path; the output
    graph is discarded immediately. Original parameters stay frozen and receive
    no gradients, updates, or storage copies.
    """
    from torch.func import functional_call
    views = {name: parameter.detach().requires_grad_(True) for name, parameter in model.named_parameters()}
    with torch.enable_grad():
        value = functional_call(model, views, (genes, noised, times, basal, perturbations, genes), {"mode": "predict_y"})
    return value.detach()


def fit_feedback(model, old, reference, matrix, schedule, gene_columns, gene_token_ids, *,
                 method, device, steps=20, rollouts=4, lr=1e-5, seed=20261108):
    if method not in {"native_fm", "nft", "terminal_gradient"} or not schedule:
        raise ValueError("Invalid fixed scDFM feedback procedure")
    model.eval()
    old.eval().requires_grad_(False)
    reference.eval().requires_grad_(False)
    if any(a.data_ptr() == b.data_ptr() for a, b in zip(model.parameters(), old.parameters(), strict=True)):
        raise ValueError("Old and current models must have independent storage")
    reference_before = model_digest(reference)
    initial_witness_name, witness, witness_before = None, None, None
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr)
    history, initial_velocity_difference, started = [], None, time.monotonic()
    for update, record in enumerate(schedule):
        basal = torch.as_tensor(matrix[record["control_rows"]][:, gene_columns].toarray(), device=device, dtype=torch.float32)
        targets = torch.as_tensor(matrix[record["target_rows"]][:, gene_columns].toarray(), device=device, dtype=torch.float32)
        genes = torch.as_tensor(gene_token_ids[gene_columns], device=device).expand(len(basal), -1).contiguous()
        perturbations = torch.as_tensor(record["drug_ids"], device=device).expand(len(basal), -1).contiguous()
        # All procedures acquire the same number of unique old-policy trajectories.
        buffers, rewards, initial_noises = [], [], []
        for rollout in range(rollouts):
            initial_noise = torch.randn(basal.shape, device=device, generator=seeded_generator(device, seed, update, rollout, 1))
            endpoint = euler_endpoint(old, basal, genes, perturbations, steps=steps, initial_noise=initial_noise)
            with torch.no_grad():
                rewards.append(-energy_u(endpoint.clamp_min(0), targets))
            buffers.append(endpoint.detach()); initial_noises.append(initial_noise)
        rewards = torch.stack(rewards)
        advantages = group_advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        losses, penalties, diagnostics = [], [], []
        for rollout, clean in enumerate(buffers):
            generator = seeded_generator(device, seed, update, rollout, 2)
            # Independent forward-corruption noise, never the generation initial noise.
            tau = torch.rand((), device=device, generator=generator)
            epsilon = torch.randn(clean.shape, device=device, generator=generator)
            noised = (1 - tau) * epsilon + tau * clean
            times = tau.expand(len(basal)).contiguous()
            old_velocity = frozen_velocity_matching_autograd(old, genes, noised, times, basal, perturbations)
            reference_velocity = frozen_velocity_matching_autograd(reference, genes, noised, times, basal, perturbations)
            current_velocity = model(genes, noised, times, basal, perturbations, genes, mode="predict_y")
            if initial_velocity_difference is None:
                initial_velocity_difference = float((current_velocity - old_velocity).detach().abs().max().cpu())
                if initial_velocity_difference > 1e-6:
                    raise ValueError("Identical initial current/old vector fields must agree in eval mode")
            reference_penalty = (current_velocity - reference_velocity).square().mean()
            if method == "nft":
                loss, information = negative_aware_loss(clean[None], noised[None], tau[None], current_velocity[None],
                    old_velocity[None], reference_velocity[None], advantages[rollout:rollout+1],
                    mixture_beta=1., advantage_clip=5., reference_weight=1e-4)
                diagnostics.append(float(information["negative_aware_objective"].cpu()))
            elif method == "native_fm":
                native, information = native_flow_loss(model, basal, targets, genes, perturbations,
                    generator=seeded_generator(device, seed, update, rollout, 3), gamma=.5)
                loss = native + 1e-4 * reference_penalty
                diagnostics.append(float(information["flow_mse"].cpu()))
            else:
                # Replaying the same unique rollout seed enables exact pathwise gradients.
                live_endpoint = euler_endpoint(model, basal, genes, perturbations, steps=steps,
                    initial_noise=initial_noises[rollout], retain_grad=True, checkpoint_steps=True)
                terminal = energy_u(live_endpoint.clamp_min(0), targets)
                loss = terminal + 1e-4 * reference_penalty
                diagnostics.append(float(terminal.detach().cpu()))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite flow post-training objective")
            (loss / rollouts).backward()
            losses.append(float(loss.detach().cpu())); penalties.append(float(reference_penalty.detach().cpu()))
        active = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise ValueError("Nonfinite flow post-training gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if witness is None:
            initial_witness_name, witness = next((name, parameter) for name, parameter in model.named_parameters() if parameter.grad is not None)
            witness_before = witness.detach().clone()
        optimizer.step()
        decay = refresh_old(old, model, update + 1)
        row = {"update": update + 1, "condition": record["condition"], "objective": float(np.mean(losses)),
            "old_policy_mean_reward": float(rewards.mean().cpu()), "reward_std": float(rewards.std(unbiased=False).cpu()),
            "advantages": advantages.cpu().tolist(), "reference_velocity_mse": float(np.mean(penalties)),
            "gradient_norm": float(gradient_norm.detach().cpu()), "old_ema_decay": decay,
            "method_diagnostic": float(np.mean(diagnostics)), "elapsed_seconds": time.monotonic()-started}
        history.append(row)
        if (update + 1) % 10 == 0:
            print({"method": method, **row}, flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    reference_after = model_digest(reference)
    if reference_before != reference_after:
        raise ValueError("Frozen original reference changed")
    return {"method": method, "updates": len(schedule), "unique_generation_rollouts_per_update": rollouts,
            "euler_steps": steps, "initial_current_old_velocity_difference": initial_velocity_difference,
            "frozen_reference_sha256_before": reference_before, "frozen_reference_sha256_after": reference_after,
            "updated_parameter_witness": initial_witness_name,
            "updated_parameter_l2_change": float((witness-witness_before).detach().square().sum().sqrt().cpu()),
            "actual_gradient_receiving_parameters": sum(parameter.numel() for parameter in active),
            "wall_seconds": time.monotonic()-started, "history": history,
            "direct_gradient_extra_compute": method == "terminal_gradient"}


def evaluate_flow(model, matrix, populations, gene_columns, gene_token_ids, *, device, steps=20, rollouts=4, seed=20261108):
    from cross_species.state_finetuning import distribution_metrics
    model.eval()
    by_condition, predictions, raw_predictions, actual_targets = {}, [], [], []
    for condition, record in populations.items():
        controls = matrix[record["control_rows"]][:, gene_columns].toarray()
        targets = matrix[record["target_rows"]][:, gene_columns].toarray()
        per_rollout, raw_values = [], []
        for rollout in range(rollouts):
            digest = hashlib.sha256((str(seed) + "|scdfm-evaluation|" + condition + "|" + str(rollout)).encode()).hexdigest()
            generator = torch.Generator(device=device).manual_seed(int(digest[:15], 16))
            # Initial noise is drawn once in frozen row order, then computation is chunked.
            initial = torch.randn(controls.shape, device=device, generator=generator)
            pieces = []
            for start in range(0, len(controls), 4):
                basal = torch.as_tensor(controls[start:start+4], device=device, dtype=torch.float32)
                genes = torch.as_tensor(gene_token_ids[gene_columns], device=device).expand(len(basal), -1).contiguous()
                perturbations = torch.as_tensor(record["drug_ids"], device=device).expand(len(basal), -1).contiguous()
                raw = euler_endpoint(model, basal, genes, perturbations, steps=steps, initial_noise=initial[start:start+4])
                pieces.append(raw.detach().cpu().numpy())
            raw_values.append(np.concatenate(pieces))
            per_rollout.append(np.maximum(raw_values[-1], 0))
        # Fixed training-only RBF bandwidth supplied by caller in each record.
        metrics = [distribution_metrics(value, targets, bandwidth=record["rbf_bandwidth"]) for value in per_rollout]
        by_condition[condition] = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
        predictions.append(np.stack(per_rollout)); raw_predictions.append(np.stack(raw_values)); actual_targets.append(targets)
        print({"evaluation_condition": condition, "metrics": by_condition[condition]}, flush=True)
    return {"macro": {key: float(np.mean([row[key] for row in by_condition.values()])) for key in next(iter(by_condition.values()))},
            "by_condition": by_condition, "predictions": np.stack(predictions), "raw_predictions": np.stack(raw_predictions),
            "targets": np.stack(actual_targets), "condition_order": list(populations)}
