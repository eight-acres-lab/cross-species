"""Small, explicit models for a cross-species feedback pilot.

No function reads files, chooses a split, constructs test labels, or loads pretrained
weights. Callers supply train/calibration arrays and generate reference candidates
out of fold. Human test responses belong exclusively to ``evaluate_predictions``.
The Gaussian actor is a newly initialized network until explicitly supervised-fit.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence
from torch.nn import functional as F


def _matrix(value, name):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.ndim != 2 or value.shape[0] == 0 or not np.isfinite(value).all():
        raise ValueError("%s must be a nonempty finite matrix" % name)
    return value


class Standardizer:
    """Fit explicitly on training rows; transform never updates fitted statistics."""

    def fit(self, train_x):
        x = _matrix(train_x, "train_x")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, x):
        return (_matrix(x, "x") - self.mean_) / self.scale_

    def inverse_transform(self, z):
        return _matrix(z, "z") * self.scale_ + self.mean_


class TrainOnlyPCA:
    """Standardized PCA of training target deltas, with an explicit inverse map.

    PCA/latent MSE differs from original-gene MSE. Decode before reporting gene-level
    human loss. This class cannot enforce split provenance: the caller owns it.
    """

    def fit(self, train_delta, n_components):
        x = _matrix(train_delta, "train_delta")
        if not 1 <= n_components <= min(x.shape[0] - 1, x.shape[1]):
            raise ValueError("n_components must be between 1 and min(n_train-1, n_genes)")
        self.standardizer_ = Standardizer().fit(x)
        z = self.standardizer_.transform(x)
        _, singular, vt = np.linalg.svd(z, full_matrices=False)
        self.components_ = vt[:n_components].copy()
        self.explained_variance_ = singular[:n_components] ** 2 / (len(x) - 1)
        return self

    def transform(self, delta):
        return self.standardizer_.transform(delta) @ self.components_.T

    def inverse_transform(self, latent):
        return self.standardizer_.inverse_transform(
            _matrix(latent, "latent") @ self.components_)


class RidgePredictor:
    """Multi-output ridge of concatenated source/context, with an unpenalized intercept."""

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = float(alpha)

    def fit(self, source, context, target_latent):
        x = np.concatenate([_matrix(source, "source"), _matrix(context, "context")], axis=1)
        y = _matrix(target_latent, "target_latent")
        if len(x) != len(y):
            raise ValueError("source/context and targets must have equal row counts")
        self.mean_x_, self.mean_y_ = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - self.mean_x_, y - self.mean_y_
        self.coef_ = np.linalg.lstsq(
            np.vstack([xc, np.sqrt(self.alpha) * np.eye(x.shape[1])]),
            np.vstack([yc, np.zeros((x.shape[1], y.shape[1]))]), rcond=None)[0]
        self.intercept_ = self.mean_y_ - self.mean_x_ @ self.coef_
        return self

    def predict(self, source, context):
        x = np.concatenate([_matrix(source, "source"), _matrix(context, "context")], axis=1)
        return x @ self.coef_ + self.intercept_


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, output_dim))


def _features(source, context):
    if source.ndim != 2 or context.ndim != 2 or len(source) != len(context):
        raise ValueError("source and context must be matrices with equal row counts")
    return torch.cat([source, context], dim=-1)


class MLPPredictor(nn.Module):
    def __init__(self, source_dim, context_dim, target_dim, hidden_dim=64):
        super().__init__()
        self.network = _mlp(source_dim + context_dim, hidden_dim, target_dim)

    def forward(self, source, context):
        return self.network(_features(source, context))


def clone_frozen(module):
    """Independent parameter/buffer storage; input gradients remain possible."""
    result = deepcopy(module).eval()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    return result


class GaussianActor(nn.Module):
    """Trainable diagonal Gaussian; optionally a residual around a frozen predictor."""

    def __init__(self, source_dim, context_dim, target_dim, hidden_dim=64,
                 initial_std=0.1, min_std=0.01, max_std=1.0, base_predictor=None):
        super().__init__()
        if not 0 < min_std <= initial_std <= max_std:
            raise ValueError("require 0 < min_std <= initial_std <= max_std")
        self.network = _mlp(source_dim + context_dim, hidden_dim, target_dim)
        self.base_predictor = None if base_predictor is None else clone_frozen(base_predictor)
        if base_predictor is not None:
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        self.log_std = nn.Parameter(torch.full((target_dim,), float(np.log(initial_std))))
        self.min_log_std, self.max_log_std = float(np.log(min_std)), float(np.log(max_std))

    def train(self, mode=True):
        super().train(mode)
        if self.base_predictor is not None:
            self.base_predictor.eval()
        return self

    def forward(self, source, context):
        mean = self.network(_features(source, context))
        if self.base_predictor is not None:
            mean = mean + self.base_predictor(source, context)
        return mean

    def distribution(self, source, context):
        mean = self(source, context)
        scale = self.log_std.clamp(self.min_log_std, self.max_log_std).exp().expand_as(mean)
        return Independent(Normal(mean, scale), 1)

    def sample(self, source, context, n_samples=1, reparameterized=False):
        distribution = self.distribution(source, context)
        # sample() detaches the action, while log_prob retains the score-function graph.
        samples = (distribution.rsample((n_samples,)) if reparameterized
                   else distribution.sample((n_samples,)))
        return samples.transpose(0, 1), distribution.log_prob(samples).transpose(0, 1)


def _candidate_inputs(source, context, candidates):
    if candidates.ndim == 2:
        return torch.cat([_features(source, context), candidates], dim=-1)
    if candidates.ndim != 3 or len(candidates) != len(source):
        raise ValueError("candidates must have shape (tasks, latent) or (tasks, K, latent)")
    features = _features(source, context)[:, None, :].expand(-1, candidates.shape[1], -1)
    return torch.cat([features, candidates], dim=-1)


class RewardScorer(nn.Module):
    """Reward is negative predicted human loss. Inference takes no human labels."""

    def __init__(self, source_dim, context_dim, target_dim, hidden_dim=64):
        super().__init__()
        self.network = _mlp(source_dim + context_dim + target_dim, hidden_dim, 1)

    def predicted_loss(self, source, context, candidates):
        values = self.network(_candidate_inputs(source, context, candidates)).squeeze(-1)
        return F.softplus(values)

    def forward(self, source, context, candidates):
        return -self.predicted_loss(source, context, candidates)


class _FrozenRidge(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.register_buffer("coef", torch.tensor(model.coef_.copy(), dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(model.intercept_.copy(), dtype=torch.float32))

    def forward(self, source, context):
        return _features(source, context) @ self.coef + self.intercept


class TeacherDistanceReward(nn.Module):
    """Degenerate control: its unconstrained optimum is the teacher's own prediction."""

    def __init__(self, teacher):
        super().__init__()
        self.teacher = (_FrozenRidge(teacher) if isinstance(teacher, RidgePredictor)
                        else clone_frozen(teacher))

    def forward(self, source, context, candidates):
        target = self.teacher(source, context)
        if candidates.ndim == 3:
            target = target[:, None, :]
        return -(candidates - target).square().mean(dim=-1)


class MeasurementConsistencyReward(nn.Module):
    """Frozen reverse-ridge consistency, an alternative proxy without human test labels.

    Fit reverse on reference human latents + context -> observed animal features.
    Reward agreement with the observed animal measurement. This is a cycle-style
    proxy, not a calibrated human loss; a rank-deficient reverse map may give many
    different human candidates identical rewards.
    """

    def __init__(self, reverse):
        super().__init__()
        if not isinstance(reverse, RidgePredictor) or not hasattr(reverse, "coef_"):
            raise ValueError("reverse must be a fitted RidgePredictor")
        self.reverse = _FrozenRidge(reverse)

    def forward(self, source, context, candidates):
        if source.ndim != 2 or source.shape[1] == 0:
            raise ValueError("source must be a nonempty-feature matrix")
        if candidates.ndim == 3:
            n, k, d = candidates.shape
            repeated_context = context[:, None, :].expand(-1, k, -1).reshape(n * k, context.shape[1])
            predicted = self.reverse(candidates.reshape(n * k, d), repeated_context)
            predicted = predicted.reshape(n, k, -1)
            observed = source[:, None, :]
        elif candidates.ndim == 2:
            predicted = self.reverse(candidates, context)
            observed = source
        else:
            raise ValueError("candidates must have shape (N,D) or (N,K,D)")
        if len(predicted) != len(source) or predicted.shape[-1] != source.shape[1]:
            raise ValueError("reverse outputs must match observed source features")
        return -(predicted - observed).square().mean(dim=-1)


def _device(module):
    value = next(iter(module.parameters()), None)
    if value is None:
        value = next(iter(module.buffers()), torch.empty(0))
    return value.device


def _data(module, *values):
    return [torch.as_tensor(value, dtype=torch.float32, device=_device(module)).detach()
            for value in values]


def _check_finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("inputs and labels must be finite")
    if not values or len(values[0]) == 0 or any(len(v) != len(values[0]) for v in values):
        raise ValueError("inputs and labels need equal, nonzero task counts")


def _optimizer(module, lr, weight_decay=0.0):
    return torch.optim.Adam([p for p in module.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)


def fit_supervised(model, source, context, target_latent, *, epochs=100, lr=1e-3,
                   batch_size=64, weight_decay=0.0, seed=0):
    """MSE-fit a newly initialized MLP or actor mean on supplied training labels only."""
    source, context, target = _data(model, source, context, target_latent)
    _check_finite(source, context, target)
    optimizer = _optimizer(model, lr, weight_decay)
    rng, history = np.random.default_rng(seed), []
    model.train()
    for epoch in range(epochs):
        order, total = rng.permutation(len(source)), 0.0
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=source.device)
            loss = F.mse_loss(model(source[idx], context[idx]), target[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        history.append({"epoch": epoch, "supervised_mse": total / len(source)})
    return history


def supervised_pretrain(actor, source, context, target_latent, **kwargs):
    return fit_supervised(actor, source, context, target_latent, **kwargs)


def fit_supervised_decoded(model, source, context, gene_targets, decoder_weight,
                           decoder_bias, *, loss_scale=1.0, epochs=100, lr=1e-3,
                           batch_size=64, weight_decay=0.0, seed=0):
    """Supervise the latent predictor using the same decoded gene loss as the reward.

    decoder_weight has shape (genes, latent); decoder_bias has shape (genes,).
    They are detached, copied constants, including when supplied as tensors. They
    are never optimizer parameters. Human labels must be train/reference rows.
    """
    source, context, target, weight, bias = _data(
        model, source, context, gene_targets, decoder_weight, decoder_bias)
    weight, bias = weight.clone(), bias.clone()
    _check_finite(source, context, target)
    if (target.ndim != 2 or weight.ndim != 2 or bias.ndim != 1
            or weight.shape[0] != target.shape[1] or len(bias) != target.shape[1]):
        raise ValueError("require gene_targets (N,G), decoder_weight (G,D), decoder_bias (G,)")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("decoder constants must be finite")
    if not np.isfinite(loss_scale) or loss_scale <= 0 or batch_size < 1 or epochs < 0:
        raise ValueError("require finite positive loss_scale/batch_size and nonnegative epochs")
    optimizer = _optimizer(model, lr, weight_decay)
    rng, history = np.random.default_rng(seed), []
    model.train()
    for epoch in range(epochs):
        order, total = rng.permutation(len(source)), 0.0
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=source.device)
            decoded = F.linear(model(source[idx], context[idx]), weight, bias)
            gene_mse = F.mse_loss(decoded, target[idx])
            objective = gene_mse / loss_scale
            optimizer.zero_grad()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(gene_mse.detach()) * len(idx)
        history.append({"epoch": epoch, "supervised_gene_mse": total / len(source),
                        "supervised_objective": total / len(source) / loss_scale})
    return history


def fit_reward_scorer(model, source, context, candidates, true_losses, *, epochs=100,
                      lr=1e-3, batch_size=64, ranking_weight=0.2, seed=0):
    """Fit externally supplied OOF candidate losses, with within-task pairwise ranking.

    true_losses are nonnegative human losses computed by the caller on reference
    tasks. No candidate bank or test responses are constructed in this trainer.
    """
    source, context, candidates, labels = _data(model, source, context, candidates, true_losses)
    _check_finite(source, context, candidates, labels)
    if candidates.ndim != 3 or labels.shape != candidates.shape[:2] or (labels < 0).any():
        raise ValueError("require candidates (N,K,D) and nonnegative true_losses (N,K)")
    if ranking_weight < 0:
        raise ValueError("ranking_weight must be nonnegative")
    optimizer, rng, history = _optimizer(model, lr), np.random.default_rng(seed), []
    model.train()
    for epoch in range(epochs):
        order, total, total_rank = rng.permutation(len(source)), 0.0, 0.0
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=source.device)
            predicted = model.predicted_loss(source[idx], context[idx], candidates[idx])
            regression = F.mse_loss(predicted, labels[idx])
            better = labels[idx, :, None] < labels[idx, None, :]
            margin = predicted[:, None, :] - predicted[:, :, None]
            pair_loss = F.softplus(-margin)
            rank_loss = (pair_loss * better).sum() / better.sum().clamp_min(1)
            loss = regression + ranking_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(regression.detach()) * len(idx)
            total_rank += float(rank_loss.detach()) * len(idx)
        history.append({"epoch": epoch, "loss_regression": total / len(source),
                        "within_task_ranking": total_rank / len(source)})
    return history


def _feedback_finetune(actor, reference, reward, source, context, *, method, steps=100,
                       lr=1e-4, batch_size=32, n_samples=4, kl_weight=0.01, seed=0):
    """Shared budget interface; there is no human-target argument in either optimizer."""
    if n_samples < 2 or steps < 0 or batch_size < 1 or kl_weight < 0:
        raise ValueError("require n_samples >= 2, steps >= 0, batch_size >= 1, KL >= 0")
    source, context = _data(actor, source, context)
    _check_finite(source, context)
    frozen_reward = clone_frozen(reward).to(_device(actor))
    frozen_reference = clone_frozen(reference).to(_device(actor))
    optimizer, rng, history = _optimizer(actor, lr), np.random.default_rng(seed), []
    actor.train()
    devices = [source.device.index or 0] if source.device.type == "cuda" else []
    # Restore the caller's RNG state after sampling, so independent trials remain isolated.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for step in range(steps):
            idx = torch.as_tensor(rng.choice(len(source), min(batch_size, len(source)),
                                            replace=False), device=source.device)
            s, c = source[idx], context[idx]
            actions, log_prob = actor.sample(s, c, n_samples,
                                            reparameterized=(method == "direct"))
            if method == "reinforce":
                with torch.no_grad():
                    rewards = frozen_reward(s, c, actions)
                # Other samples from this same task form an action-independent baseline.
                baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (n_samples - 1)
                advantage = (rewards - baseline).detach()
                objective = -(advantage * log_prob).mean()
            else:
                rewards = frozen_reward(s, c, actions)
                objective = -rewards.mean()
            with torch.no_grad():
                ref_distribution = frozen_reference.distribution(s, c)
            kl = kl_divergence(actor.distribution(s, c), ref_distribution).mean()
            loss = objective + kl_weight * kl
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite feedback objective")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite actor gradient")
            optimizer.step()
            with torch.no_grad():
                actor.log_std.clamp_(actor.min_log_std, actor.max_log_std)
            history.append({"step": step, "proxy_reward": float(rewards.detach().mean()),
                            "reference_kl": float(kl.detach()), "objective": float(loss.detach()),
                            "sampled_candidates": int(len(idx) * n_samples)})
    return history


def reinforce_finetune(actor, reference, reward, source, context, **kwargs):
    """REINFORCE with detached actions, leave-one-out baselines, and analytic reference KL."""
    return _feedback_finetune(actor, reference, reward, source, context,
                              method="reinforce", **kwargs)


def direct_reward_finetune(actor, reference, reward, source, context, **kwargs):
    """Pathwise gradient through the same frozen reward; a required differentiable control."""
    return _feedback_finetune(actor, reference, reward, source, context,
                              method="direct", **kwargs)


def predict_numpy(model, source, context):
    if isinstance(model, RidgePredictor):
        return model.predict(source, context)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            s, c = _data(model, source, context)
            return model(s, c).cpu().numpy().copy()
    finally:
        model.train(was_training)


def evaluate_predictions(predictions, human_targets, proxy_rewards=None):
    """Terminal evaluation: keep observed human loss separate from a proxy reward.

    The caller should inverse-transform latent predictions and supply gene-space
    targets for gene-space claims. These labels are never used to update a model.
    """
    predictions = _matrix(predictions, "predictions")
    targets = _matrix(human_targets, "human_targets")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and human_targets must have identical shapes")
    per_task = ((predictions - targets) ** 2).mean(axis=1)
    result = {"human_loss": float(per_task.mean()), "human_loss_per_task": per_task}
    if proxy_rewards is not None:
        rewards = np.array(proxy_rewards, dtype=float, copy=True)
        if rewards.shape != per_task.shape or not np.isfinite(rewards).all():
            raise ValueError("proxy_rewards must be finite with one reward per task")
        result.update(proxy_reward=float(rewards.mean()), proxy_reward_per_task=rewards)
    return result
