"""Small, explicit estimators for replicate-preserving multisource comparisons.

Training tasks, test source measurements and terminal human scoring are separate
interfaces. A gene axis is a feature axis, never an independent task axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import torch


def finite_array(value, ndim, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or any(size == 0 for size in result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def source_subsets(source_ids):
    identifiers = tuple(str(value) for value in source_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers) or len(identifiers) > 8:
        raise ValueError("Need unique source IDs, at most eight for exact subset enumeration")
    return [tuple(combo) for size in range(1, len(identifiers)+1)
            for combo in combinations(range(len(identifiers)), size)]


def simplex_weights(train_sources, train_human, *, ridge=1.):
    """Exact small-S nonnegative sum-one ridge, shrinking toward equal weights.

    Objective = mean_task,gene prediction MSE + ridge*||w-uniform||².
    Ridge is fixed before terminal scoring; there is no intercept or gene fitting.
    """
    x = finite_array(train_sources, 3, "train_sources")
    y = finite_array(train_human, 2, "train_human")
    if y.shape != (x.shape[0], x.shape[2]) or not np.isfinite(ridge) or ridge <= 0 or x.shape[1] > 8:
        raise ValueError("Incompatible tasks/genes or invalid fixed ridge")
    tasks, sources, genes = x.shape
    gram = np.einsum("tsg,tug->su", x, x)/(tasks*genes)
    cross = np.einsum("tsg,tg->s", x, y)/(tasks*genes)
    uniform = np.full(sources, 1/sources)
    matrix, rhs = gram + ridge*np.eye(sources), cross + ridge*uniform
    best, best_value = None, float("inf")
    for active in source_subsets(range(sources)):
        idx = np.asarray(active)
        kkt = np.block([[matrix[np.ix_(idx, idx)], np.ones((len(idx), 1))],
                        [np.ones((1, len(idx))), np.zeros((1, 1))]])
        solution = np.linalg.solve(kkt, np.r_[rhs[idx], 1.])[:-1]
        if (solution < -1e-10).any():
            continue
        weights = np.zeros(sources)
        weights[idx] = np.maximum(solution, 0)
        weights /= weights.sum()
        objective = float(weights@gram@weights - 2*cross@weights + ridge*np.square(weights-uniform).sum())
        if objective < best_value-1e-14:
            best, best_value = weights, objective
    if best is None:
        raise RuntimeError("No feasible simplex solution")
    return best.copy()


def combine_sources(source_responses, weights):
    source = finite_array(source_responses, 3, "source_responses")
    weights = finite_array(weights, 1, "weights")
    if len(weights) != source.shape[1] or (weights < -1e-12).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Source weights must be nonnegative and sum to one")
    return np.einsum("tsg,s->tg", source, weights)


@dataclass(frozen=True)
class HistoricalSourceRule:
    source_ids: tuple
    training_losses: np.ndarray
    training_task_count: int

    def choose(self):
        # Ties follow the caller's frozen source order.
        return int(np.argmin(self.training_losses))

    def predict(self, test_sources):
        values = finite_array(test_sources, 3, "test_sources")
        if values.shape[1] != len(self.source_ids):
            raise ValueError("Source order/count changed")
        return values[:, self.choose()].copy()


def fit_historical_source_rule(train_sources, train_human, source_ids):
    source = finite_array(train_sources, 3, "train_sources")
    human = finite_array(train_human, 2, "train_human")
    if source.shape[0] < 1 or human.shape != (source.shape[0], source.shape[2]) or len(source_ids) != source.shape[1]:
        raise ValueError("Invalid historical reference tasks")
    losses = np.square(source-human[:, None]).mean(axis=(0, 2))
    losses.setflags(write=False)
    return HistoricalSourceRule(tuple(source_ids), losses, len(source))


def response_scores(predictions, human):
    predictions, human = finite_array(predictions, 2, "predictions"), finite_array(human, 2, "human")
    if predictions.shape != human.shape:
        raise ValueError("Prediction and human query axes differ")
    mse = np.square(predictions-human).mean(1)
    pearson, cosine = [], []
    for x, y in zip(predictions, human):
        xc, yc = x-x.mean(), y-y.mean()
        denominator, norm = np.linalg.norm(xc)*np.linalg.norm(yc), np.linalg.norm(x)*np.linalg.norm(y)
        pearson.append(float(xc@yc/denominator) if denominator > 0 else None)
        cosine.append(float(x@y/norm) if norm > 0 else None)
    return {"mse_by_query": mse, "macro_mse": float(mse.mean()), "pearson_by_query": pearson,
            "cosine_by_query": cosine, "defined_pearson_queries": sum(v is not None for v in pearson)}


def fit_lopo_predictions(sources, human, source_ids, *, ridge=1., known_context=None):
    """Exploratory leave-one-perturbation-out predictions; no held human in fitting.

    All subsets are reported, never selected with terminal loss. Context matching
    is a fixed exact-category rule and falls back to all historical human tasks.
    """
    source, human = finite_array(sources, 3, "sources"), finite_array(human, 2, "human")
    tasks, count, genes = source.shape
    if tasks < 2 or human.shape != (tasks, genes) or len(source_ids) != count:
        raise ValueError("Need matched feature axes and at least two perturbations")
    if known_context is not None and len(known_context) != tasks:
        raise ValueError("One premeasurement context category per perturbation is required")
    subsets = source_subsets(source_ids)
    predictions = {"zero": np.zeros_like(human), "human_history_mean": np.empty_like(human),
                   "human_context_history_mean": np.empty_like(human), "best_fixed_source_history": np.empty_like(human)}
    weights = {name: np.zeros((tasks, count)) for name in predictions if name not in {"zero", "human_history_mean", "human_context_history_mean"}}
    for subset in subsets:
        label = "+".join(str(source_ids[i]) for i in subset)
        for method in ["equal", "simplex"]:
            name = method+"/"+label
            predictions[name] = np.empty_like(human)
            weights[name] = np.zeros((tasks, count))
    provenance = []
    for held in range(tasks):
        train = np.flatnonzero(np.arange(tasks) != held)
        predictions["human_history_mean"][held] = human[train].mean(0)
        same = train if known_context is None else train[np.asarray(known_context, dtype=str)[train] == str(known_context[held])]
        if not len(same):
            same = train
        predictions["human_context_history_mean"][held] = human[same].mean(0)
        rule = fit_historical_source_rule(source[train], human[train], source_ids)
        predictions["best_fixed_source_history"][held] = rule.predict(source[held:held+1])[0]
        weights["best_fixed_source_history"][held, rule.choose()] = 1
        for subset in subsets:
            idx = np.asarray(subset)
            label = "+".join(str(source_ids[i]) for i in subset)
            for name, local_weights in [("equal/"+label, np.full(len(idx), 1/len(idx))),
                                         ("simplex/"+label, simplex_weights(source[train][:, idx], human[train], ridge=ridge))]:
                predictions[name][held] = combine_sources(source[held:held+1][:, idx], local_weights)[0]
                weights[name][held, idx] = local_weights
        provenance.append({"held_query": held, "training_queries": train.tolist(), "context_reference_queries": same.tolist(),
                           "selected_source_before_measurement": str(source_ids[rule.choose()]),
                           "source_training_losses": rule.training_losses.tolist(),
                           "context_selector_status": "Context-dependent source selection is not fitted; each split has two training tasks." if len(train)<4 else "Not fitted in this fixed baseline protocol"})
    return predictions, weights, provenance


def independent_arm_counts(n_boot, n_tasks, n_species, n_replicates, seed=20261108):
    if min(n_boot, n_tasks, n_species, n_replicates) < 1:
        raise ValueError("Positive dimensions required")
    rng = np.random.default_rng(seed)
    draws = rng.integers(n_replicates, size=(n_boot,n_tasks,n_species,2,n_replicates))
    return np.stack([(draws == j).sum(-1) for j in range(n_replicates)], axis=-1)


def arm_resampling_losses(arm_profiles, source_weights, draw_counts, *, device="cpu", chunk_size=32):
    """Batched, fixed-estimator measured-arm resampling; no task/gene bootstrap.

    arm_profiles[T,H+S,2,R,G], role0 treated/role1 vehicle;
    weights[M,T,S]; independent multinomial sample counts[B,T,H+S,2,R].
    Every method reuses the same sampled human and source arms within each draw.
    """
    arms = finite_array(arm_profiles, 5, "arm_profiles")
    weights = finite_array(source_weights, 3, "source_weights")
    counts = np.array(draw_counts, dtype=np.int64, copy=True)
    tasks, species, roles, replicates, genes = arms.shape
    if roles != 2 or weights.shape[1:] != (tasks,species-1) or counts.ndim != 5 or counts.shape[1:] != (tasks,species,2,replicates):
        raise ValueError("Arm, fixed-model weights, or draw-count axes mismatch")
    if (counts < 0).any() or not np.all(counts.sum(-1) == replicates) or chunk_size < 1:
        raise ValueError("Each independently sampled arm must have exactly R measured-profile draws")
    if (weights < -1e-12).any() or not np.allclose(weights.sum(-1),1):
        raise ValueError("Every fixed estimator must have simplex source weights")
    arm_tensor = torch.as_tensor(arms, device=device, dtype=torch.float64)
    weight_tensor = torch.as_tensor(weights, device=device, dtype=torch.float64)
    output = np.empty((len(counts),tasks,len(weights)))
    with torch.no_grad():
        for start in range(0,len(counts),chunk_size):
            draw = torch.as_tensor(counts[start:start+chunk_size],device=device,dtype=torch.float64)/replicates
            means = torch.einsum("btsar,tsarg->btsag",draw,arm_tensor)
            response = means[:,:,:,0]-means[:,:,:,1]
            predicted = torch.einsum("mts,btsg->btmg",weight_tensor,response[:,:,1:])
            losses = (predicted-response[:,:,0,None]).square().mean(-1)
            output[start:start+len(draw)] = losses.cpu().numpy()
    return output


def fixed_total_profile_budget(arm_profiles, subsets, *, budget_per_role=3):
    """Exact expected loss under balanced, without-replacement finite-arm sampling.

    Each method receives B treated and B vehicle profiles in total across chosen
    sources. Each source has R available profiles/arm. Human truth is the fixed
    complete human-arm mean. Source-estimator weights stay equal. The expectation
    averages every balanced allocation and independent subset selection; it does
    not average predictions from separately paid experiments into a cheaper one.
    """
    arms=finite_array(arm_profiles,5,"arm_profiles")
    tasks,species,roles,replicates,genes=arms.shape
    if roles!=2 or replicates<2 or not isinstance(budget_per_role,int) or budget_per_role<1:
        raise ValueError("Invalid measured-profile budget")
    means=arms.mean(3)
    response=means[:,:,0]-means[:,:,1]
    variance=arms[:,:,0].var(2,ddof=1)+arms[:,:,1].var(2,ddof=1)
    result=[]
    for subset in subsets:
        subset=tuple(subset);size=len(subset)
        if size>budget_per_role or budget_per_role>size*replicates:
            result.append({"subset":list(subset),"eligible":False,"reason":"Fixed total profile budget cannot cover each source or exceeds available records"})
            continue
        allocations=[a for a in product(range(1,replicates+1),repeat=size)
                     if sum(a)==budget_per_role and max(a)-min(a)<=1]
        source_idx=np.array(subset)+1
        point=np.square(response[:,source_idx].mean(1)-response[:,0]).mean(1)
        extra=[]
        for allocation in allocations:
            n=np.asarray(allocation)
            factor=(1-n/replicates)/n
            extra.append((variance[:,source_idx].mean(-1)*factor[None]/size**2).sum(1))
        expected=point+np.mean(extra,axis=0)
        result.append({"subset":list(subset),"eligible":True,"treated_profiles_total":budget_per_role,
                       "vehicle_profiles_total":budget_per_role,"balanced_allocations_per_role":[list(a) for a in allocations],
                       "fixed_full_mean_mse_by_query":point.tolist(),"finite_sampling_variance_by_query":np.mean(extra,axis=0).tolist(),
                       "expected_mse_by_query":expected.tolist(),"macro_expected_mse":float(expected.mean())})
    return result
