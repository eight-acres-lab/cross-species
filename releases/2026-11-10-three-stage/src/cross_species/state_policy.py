"""Custom Gaussian latent policy over the released State decoder.

This is an explicit experimental adaptation of State, not native State RL or a
reproduction of a diffusion/flow post-training algorithm. The Gaussian is
optimization exploration; it is not a model of biological cell variability.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def decoder_input(model, batch):
    """Capture the exact official project_out input without changing its values."""
    captured = []
    handle = model.project_out.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        model(batch, padded=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one official output-decoder invocation")
    return captured[0].reshape(-1, model.hidden_dim)


class StateLatentPolicy(nn.Module):
    """Keep the released architecture; optimize only its final transformer blocks."""
    def __init__(self, model, *, last_blocks=2, sigma=.05):
        super().__init__()
        if not 0 < sigma < 1:
            raise ValueError("Exploration sigma must be fixed, finite, and positive")
        self.model = model.eval().requires_grad_(False)
        layers = model.transformer_backbone.layers
        if not 1 <= last_blocks <= len(layers):
            raise ValueError("last_blocks must name existing official transformer layers")
        for layer in layers[-last_blocks:]:
            layer.requires_grad_(True)
        self.last_blocks = last_blocks
        witness = next(model.parameters())
        self.register_buffer("sigma", torch.tensor(float(sigma), device=witness.device, dtype=witness.dtype))

    def mean(self, batch):
        self.model.eval()  # Gradients remain enabled; Gaussian defines all policy randomness.
        return decoder_input(self.model, batch)

    def decode(self, latent):
        values = self.model.project_out(latent)
        # The supported release is X_hvg and applies this exact official ReLU.
        if self.model.output_space != "gene" or self.model.hparams["embed_key"] != "X_hvg":
            raise ValueError("This policy is scoped to the released Parse X_hvg decoder")
        return self.model.relu(values).reshape(-1, self.model.output_dim)

    def distribution(self, mean):
        if mean.device != self.sigma.device:
            raise ValueError("Policy Gaussian parameters must share the model device")
        return Normal(mean, self.sigma)

    def reference_kl_per_dimension(self, mean, reference_mean):
        # Identical fixed diagonal variances, so this is the analytic KL / (N * H).
        return .5 * ((mean - reference_mean) / self.sigma).square().mean()

    def parameter_metadata(self):
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        return {"trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
                "trainable_parameter_names": names, "last_blocks": self.last_blocks,
                "fixed_exploration_sigma": float(self.sigma.cpu()),
                "frozen_decoder": not any(parameter.requires_grad for parameter in self.model.project_out.parameters())}


def reinforce_loss(distribution, actions, rewards, *, kl_per_dimension, beta):
    """Unbiased K-rollout LOO REINFORCE; actions must be detached from sampling."""
    if actions.ndim != distribution.loc.ndim + 1 or len(actions) < 2:
        raise ValueError("Require at least two joint population rollouts")
    if rewards.shape != (len(actions),) or not torch.isfinite(rewards).all():
        raise ValueError("Require a finite scalar population reward per rollout")
    log_probability = distribution.log_prob(actions.detach()).flatten(1).sum(1)
    detached = rewards.detach()
    baseline = (detached.sum() - detached) / (len(detached) - 1)
    advantage = detached - baseline
    objective = -(advantage * log_probability).mean() + beta * kl_per_dimension
    return objective, {"joint_log_probability": log_probability.detach(), "advantages": advantage,
                       "kl_per_dimension": kl_per_dimension.detach()}


def policy_training_loss(policy, batch, targets, reference_mean, *, estimator,
                         beta=.01, rollouts=4, generator=None):
    """Compare likelihood-ratio and pathwise estimators on the same objective."""
    if estimator not in {"reinforce", "pathwise"} or rollouts < 2 or beta < 0:
        raise ValueError("Invalid fixed policy-training specification")
    mean = policy.mean(batch)
    distribution = policy.distribution(mean)
    noise = torch.randn((rollouts,) + tuple(mean.shape), device=mean.device,
                        dtype=mean.dtype, generator=generator)
    actions = mean.unsqueeze(0) + policy.sigma * noise
    if estimator == "reinforce":
        # Do not differentiate rewards through sampled actions or the decoder.
        with torch.no_grad():
            rewards = torch.stack([-policy.model._compute_distribution_loss(policy.decode(action.detach()), targets).mean()
                                   for action in actions])
    else:
        rewards = torch.stack([-policy.model._compute_distribution_loss(policy.decode(action), targets).mean()
                               for action in actions])
    kl = policy.reference_kl_per_dimension(mean, reference_mean.detach())
    if estimator == "reinforce":
        loss, diagnostics = reinforce_loss(distribution, actions, rewards, kl_per_dimension=kl, beta=beta)
    else:
        loss = -rewards.mean() + beta * kl
        diagnostics = {"kl_per_dimension": kl.detach()}
    diagnostics.update(rewards=rewards.detach(), expected_native_energy=-rewards.mean().detach(),
                       objective_estimator=estimator, joint_action_dimensions=mean.numel())
    return loss, diagnostics


def fit_policy(policy, reference, expression, schedule, perturbations, *, estimator,
               beta, device, lr=1e-5, seed=20261108, rollouts=4):
    """Optimize on caller-listed training target populations only."""
    import hashlib
    import time
    if not schedule:
        raise ValueError("A nonempty metadata-frozen training schedule is required")
    reference.eval().requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed(seed)
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    def frozen_digest():
        digest = hashlib.sha256()
        for name, parameter in policy.model.named_parameters():
            if not parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    frozen_before = frozen_digest()
    witness_name, witness = next((name, parameter) for name, parameter in policy.model.named_parameters() if parameter.requires_grad)
    witness_before = witness.detach().clone()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.0005)
    history, started, initial_difference = [], time.monotonic(), None
    for record in schedule:
        basal = torch.as_tensor(expression[record["control_rows"]], device=device, dtype=torch.float32)
        target = torch.as_tensor(expression[record["target_rows"]], device=device, dtype=torch.float32)
        pert = perturbations[record["condition"]].to(device).expand(len(basal), -1).contiguous()
        batch = {"ctrl_cell_emb": basal, "pert_emb": pert}
        with torch.no_grad():
            reference_mean = decoder_input(reference, batch)
            if initial_difference is None:
                initial_difference = float((policy.mean(batch) - reference_mean).abs().max().cpu())
                if initial_difference > 1e-7:
                    raise ValueError("Policy must start at the unchanged official predecoder mean")
        optimizer.zero_grad(set_to_none=True)
        loss, values = policy_training_loss(policy, batch, target, reference_mean,
            estimator=estimator, beta=beta, rollouts=rollouts, generator=generator)
        loss.backward()
        active = [parameter for parameter in parameters if parameter.grad is not None]
        if not torch.isfinite(loss) or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise ValueError("Nonfinite policy objective or gradient")
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 10.)
        optimizer.step()
        history.append({"step": record["step"] + 1, "condition": record["condition"], "donor": record["donor"],
                        "mean_native_energy": float(values["expected_native_energy"].cpu()),
                        "kl_per_dimension": float(values["kl_per_dimension"].cpu()),
                        "kl_joint": float(values["kl_per_dimension"].cpu()) * values["joint_action_dimensions"],
                        "joint_action_dimensions": values["joint_action_dimensions"],
                        "estimator_surrogate_loss": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient.detach().cpu())})
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    frozen_after = frozen_digest()
    if frozen_before != frozen_after:
        raise ValueError("Frozen official State parameters changed during policy optimization")
    return {"estimator": estimator, "updates": len(schedule), "rollouts_per_update": rollouts,
            "beta_kl_per_dimension": beta, "seconds": time.monotonic() - started,
            "initial_mean_max_difference": initial_difference, "history": history,
            "frozen_parameter_sha256_before": frozen_before, "frozen_parameter_sha256_after": frozen_after,
            "updated_parameter_witness": witness_name,
            "updated_parameter_l2_change": float((witness - witness_before).detach().square().sum().sqrt().cpu()),
            "actual_gradient_parameters": sum(parameter.numel() for parameter in active),
            **policy.parameter_metadata()}


def predict_policy(policy, controls, perturbation, *, device, rollouts=4, seed=20261108):
    """Inference consumes actual PBS inputs and planned perturbation identity only."""
    with torch.no_grad():
        basal = torch.as_tensor(controls, device=device, dtype=torch.float32)
        pert = perturbation.to(device).expand(len(basal), -1).contiguous()
        mean = policy.mean({"ctrl_cell_emb": basal, "pert_emb": pert})
        noise = torch.randn((rollouts,) + tuple(mean.shape), device=device, dtype=mean.dtype,
                            generator=torch.Generator(device=device).manual_seed(seed))
        deterministic = policy.decode(mean)
        sampled = torch.stack([policy.decode(mean + policy.sigma * draw) for draw in noise])
    return deterministic.cpu().numpy(), sampled.cpu().numpy()


def evaluate_policy(policy, expression, populations, conditions, donors, perturbations, *,
                    device, cell_count, bandwidth, rollouts=4, seed=20261108, save_predictions=False):
    import hashlib
    import numpy as np
    from cross_species.state_finetuning import distribution_metrics
    policy.eval()
    records, samples, deterministic_values, targets, queries = [], [], [], [], []
    for condition in conditions:
        for donor in donors:
            index = populations[(condition, donor)]
            control_rows, target_rows = index["controls"][:cell_count], index["targets"][:cell_count]
            # The per-query seed is fixed from metadata, independently of method and truth.
            text = str(seed) + "|state-policy-evaluation|" + condition + "|" + donor
            query_seed = int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)
            deterministic, draws = predict_policy(policy, expression[control_rows], perturbations[condition],
                device=device, rollouts=rollouts, seed=query_seed)
            target = expression[target_rows]
            metrics = [distribution_metrics(draw, target, bandwidth=bandwidth) for draw in draws]
            with torch.no_grad():
                target_tensor = torch.as_tensor(target, device=device)
                energies = [float(policy.model._compute_distribution_loss(torch.as_tensor(draw, device=device), target_tensor).mean().cpu()) for draw in draws]
                deterministic_energy = float(policy.model._compute_distribution_loss(torch.as_tensor(deterministic, device=device), target_tensor).mean().cpu())
            row = {"condition": condition, "donor": donor, "expected_native_energy": float(np.mean(energies)),
                   "deterministic_native_energy": deterministic_energy,
                   "deterministic_mean_mse": float(np.square(deterministic.mean(0) - target.mean(0)).mean()),
                   **{key: float(np.mean([entry[key] for entry in metrics])) for key in metrics[0]}}
            records.append(row)
            if save_predictions:
                samples.append(draws); deterministic_values.append(deterministic); targets.append(target)
                queries.append({"condition": condition, "donor": donor, "control_rows": control_rows,
                                "target_rows": target_rows, "query_seed": query_seed})
    grouped = {condition: {key: float(np.mean([row[key] for row in records if row["condition"] == condition]))
                           for key in records[0] if key not in {"condition", "donor"}} for condition in conditions}
    result = {"macro": {key: float(np.mean([row[key] for row in grouped.values()])) for key in next(iter(grouped.values()))},
              "by_condition": grouped, "by_condition_donor": records}
    if save_predictions:
        result.update(sampled_predictions=np.stack(samples), deterministic_predictions=np.stack(deterministic_values),
                      targets=np.stack(targets), query_indices=queries)
    return result
