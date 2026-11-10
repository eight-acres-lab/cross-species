#!/usr/bin/env python3
"""Same-sample native versus HumanMapped source-response sensitivity.

No fitted predictor or source selection. The native route remains primary. Both
routes score the identical human samples and fixed gene axis; mouse is excluded
from both because no alternative HumanMapped file was deposited.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[key] = "2"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_species.multisource_data import file_hash, load_curated_panel


def plain(value):
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return plain(value.item())
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def ranks(values):
    values = np.asarray(values)
    result = np.empty(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]: stop += 1
        result[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return result


def pearson(left, right):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a * b).sum(-1), denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def subset_weights(source_count):
    subsets = [s for k in range(1, source_count + 1) for s in itertools.combinations(range(source_count), k)]
    weights = np.zeros((len(subsets), source_count))
    for row, subset in enumerate(subsets): weights[row, list(subset)] = 1 / len(subset)
    return subsets, weights


def load_aligned(primary_dir, sensitivity_dir):
    primary = load_curated_panel(primary_dir)
    m = json.loads((Path(sensitivity_dir) / "manifest.json").read_text())
    if m["primary_panel_manifest_sha256"] != file_hash(Path(primary_dir) / "manifest.json"):
        raise ValueError("Technical sensitivity references a different primary panel")
    for name, expected in m["files"].items():
        if file_hash(Path(sensitivity_dir) / name) != expected:
            raise ValueError("Technical input checksum mismatch")
    with np.load(Path(sensitivity_dir) / "same_sample_alignment.npz", allow_pickle=False) as a:
        arrays = {key: a[key].copy() for key in a.files}
    axis = primary.feature_ids[primary.global_eligible_mask]
    if not np.array_equal(axis, arrays["gene_ids"]):
        raise ValueError("Technical and primary gene order differ")
    with np.load(Path(primary_dir) / "donor_responses.npz", allow_pickle=False) as donor:
        donor_index = {(s, d): i for i, (s, d) in enumerate(zip(donor["species"], donor["donors"]))}
        rows = [donor_index[s, d] for s, d in zip(arrays["donor_species"], arrays["donor_ids"])]
        # Mean(treated-baseline) and mean(treated)-baseline differ only by
        # floating-point summation; identifiers and gene order are exact.
        if not np.allclose(donor["response"][rows][:, primary.global_eligible_mask], arrays["native_donor_response"], rtol=0, atol=1e-12):
            raise ValueError("Native responses differ from primary curation")
    species = [str(s) for s in primary.species_order if s != "mouse"]
    if species[0] != "human" or len(species) != 9:
        raise ValueError("Expected human plus eight alternative-mapping animal sources")
    route_arrays = []
    donor_order = []
    for route in ["native_donor_response", "humanmapped_donor_response"]:
        blocks = []
        for source in species:
            rows = np.flatnonzero(arrays["donor_species"] == source)
            if len(rows) != 3:
                raise ValueError("Expected three donors per source in both routes")
            blocks.append(arrays[route][rows])
            if route == "native_donor_response": donor_order.append(arrays["donor_ids"][rows].tolist())
        values = np.stack(blocks)
        if not np.isfinite(values).all(): raise ValueError("Nonfinite comparison data")
        route_arrays.append(values)
    native, humanmapped = route_arrays
    if not np.array_equal(native[0], humanmapped[0]):
        raise ValueError("Human reference differs between mapping routes")
    # Raw arm identities/order are also the same source records, not pseudoreplicates.
    metadata = json.loads((Path(sensitivity_dir) / "arm_metadata.json").read_text())
    if arrays["sample_ids"].tolist() != [row["sample_id"] for row in metadata]:
        raise ValueError("Technical arm sample identities/order differ")
    if len(set(arrays["sample_ids"].tolist())) != 162:
        raise ValueError("Technical matrix duplicates biological samples")
    return native, humanmapped, species, donor_order, axis


def score_subsets(donor_response, weights):
    means = donor_response.mean(1)
    human, sources = means[0], means[1:]
    prediction = weights @ sources
    return {"mean_target_mse": ((prediction - human) ** 2).mean(1),
            "mean_individual_target_mse": np.mean([((prediction - person) ** 2).mean(1) for person in donor_response[0]], axis=0),
            "pearson_mean_target": pearson(prediction, np.broadcast_to(human, prediction.shape)),
            "response_rms": np.sqrt(np.square(prediction).mean(1)), "predictions": prediction}


def edge_comparison(subsets, native_loss, alternative_loss):
    index = {subset: i for i, subset in enumerate(subsets)}
    source_count = max(max(s) for s in subsets) + 1
    result = []
    for before in subsets:
        for added in range(source_count):
            if added in before: continue
            after = tuple(sorted(before + (added,)))
            old, new = index[before], index[after]
            ng = float(native_loss[old] - native_loss[new])
            ag = float(alternative_loss[old] - alternative_loss[new])
            ns, als = np.sign(ng) if abs(ng) > 1e-12 else 0, np.sign(ag) if abs(ag) > 1e-12 else 0
            result.append({"before_indices": before, "added_index": added, "after_indices": after,
                           "native_gain": ng, "humanmapped_gain": ag, "native_harmful": ng < -1e-12,
                           "humanmapped_harmful": ag < -1e-12, "direction_agrees": bool(ns == als)})
    return result


def donor_draws(draws, groups, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(3, [1 / 3] * 3, size=(draws, groups)).astype(np.int64)


def resampled_losses(donors, weights, counts, device="cpu", batch_size=16):
    """Evaluate fixed equal weights using donor-mean Gram matrices in float64."""
    donors, weights, counts = np.asarray(donors), np.asarray(weights), np.asarray(counts)
    if donors.ndim != 3 or donors.shape[1] != 3 or counts.ndim != 3 or counts.shape[1:] != donors.shape[:2]:
        raise ValueError("Incompatible donor arrays and donor-count draws")
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all() or not np.all(counts.sum(2) == 3):
        raise ValueError("Every species draw must contain three sampled donors")
    if weights.shape[1] != len(donors) - 1:
        raise ValueError("Wrong fixed-source weight axis")
    tensor = torch.tensor(donors, dtype=torch.float64, device=device)
    # Each estimator subtracts the same resampled human response.
    combined = torch.tensor(np.concatenate([-np.ones((len(weights), 1)), weights], axis=1), dtype=torch.float64, device=device)
    result = []
    for start in range(0, len(counts), batch_size):
        count = torch.tensor(counts[start:start + batch_size], dtype=torch.float64, device=device)
        means = torch.einsum("bsd,sdg->bsg", count / 3., tensor)
        gram = torch.bmm(means, means.transpose(1, 2)) / donors.shape[2]
        loss = torch.einsum("ms,bst,mt->bm", combined, gram, combined)
        result.append(loss.cpu().numpy())
    return np.maximum(np.concatenate(result), 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-panel", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE105160")
    parser.add_argument("--technical-panel", type=Path, default=ROOT / "outputs/multisource_20261108/curated_panels/GSE105160_alignment_sensitivity")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/multisource_20261108/benchmarks/GSE105160_mapping_sensitivity")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()): parser.error("Output must be empty")
    if args.n_resamples < 2: parser.error("Need at least two measured-donor sensitivity draws")
    native, alt, species, donors, genes = load_aligned(args.primary_panel, args.technical_panel)
    subsets, weights = subset_weights(len(species) - 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specification = {"written_before_scoring_utc": datetime.now(timezone.utc).isoformat(), "study": "GSE105160",
        "primary_route": "native mapped counts with explicit one-to-one orthology", "alternative_route": "Same samples aligned to human reference",
        "sample_species_order": species, "donor_ids": donors, "sources": species[1:], "source_subsets": subsets,
        "genes": len(genes), "gene_axis_sha256": hashlib.sha256("\n".join(genes.tolist()).encode()).hexdigest(),
        "estimators": "All 255 nonempty subsets, equal means of source means; no parameters fitted or source chosen using outcomes",
        "primary_metric": "Gene MSE to the identical three-human-donor mean, squared log2CPM response units",
        "secondary": "Individual human target mean MSE, Pearson profile correlation, response RMS, source/subset ranks and added-source edge directions",
        "edge_gain": "MSE before adding source minus MSE after; negative means specified equal-mean estimator worsens; sign tolerance1e-12",
        "measured_donor_sensitivity": {"draws": args.n_resamples, "seed": args.seed, "sampled_unit": "Three deposited donors resampled with replacement separately in each fixed species, with own-control response intact; identical donor draws used for both routes", "percentiles": [.025,.5,.975], "interpretation": "Finite measured-donor sensitivity only; three donors/source, one stimulus, no population/generalization confidence guarantee; no species or gene bootstrap"},
        "leave_one_donor_out": "For each single source, all3 human-donor deletions×all3 source-donor deletions, paired across mapping routes; other sources irrelevant to each single-source case",
        "budget": "Each subset member supplies allthree source donor means; total source measurement budget grows with number of sources",
        "mapping_caveat": "The two routes differ in alignment/reference annotation and full-gene normalization denominators. This is route sensitivity, not an isolated causal mapping effect.",
        "scope": "Descriptive same-sample/same-gene mapping sensitivity; native remains primary. No unseen perturbation or source-recommendation validation. Mouse excluded from BOTH routes because alternative data absent.",
        "inputs": {str(args.primary_panel / "manifest.json"): file_hash(args.primary_panel / "manifest.json"), str(args.technical_panel / "manifest.json"): file_hash(args.technical_panel / "manifest.json")},
        "code_sha256": file_hash(Path(__file__)), "device": args.device}
    save(args.out_dir / "specification.json", specification)
    (args.out_dir / "code_snapshot.py").write_bytes(Path(__file__).read_bytes())
    scores = {"native": score_subsets(native, weights), "humanmapped": score_subsets(alt, weights)}
    single = {}
    for route in scores:
        s = scores[route]
        single[route] = [{"source": species[i + 1], "mse": s["mean_target_mse"][i], "pearson": s["pearson_mean_target"][i], "response_rms": s["response_rms"][i], "mse_rank": ranks(s["mean_target_mse"][:8])[i]} for i in range(8)]
    edges = edge_comparison(subsets, scores["native"]["mean_target_mse"], scores["humanmapped"]["mean_target_mse"])
    counts = donor_draws(args.n_resamples, len(species), args.seed)
    start = time.monotonic()
    sensitivity = {route: resampled_losses(values, weights, counts, device=args.device) for route, values in [("native", native), ("humanmapped", alt)]}
    if args.device == "cuda": torch.cuda.synchronize()
    seconds = time.monotonic() - start
    # Literal direct predictions independently validate quadratic scoring.
    max_error = 0.
    for route, values in [("native", native), ("humanmapped", alt)]:
        for i in range(min(4, len(counts))):
            means = np.einsum("sd,sdg->sg", counts[i] / 3., values)
            actual = ((weights @ means[1:] - means[0]) ** 2).mean(1)
            max_error = max(max_error, float(np.max(np.abs(actual - sensitivity[route][i]))))
    if max_error > 1e-10: raise ValueError("Quadratic/literal resampling loss parity failed")
    lodo = []
    for source in range(1, len(species)):
        for h in range(3):
            for s in range(3):
                row = {"source": species[source], "human_donor_removed": donors[0][h], "source_donor_removed": donors[source][s]}
                for route, values in [("native", native), ("humanmapped", alt)]:
                    target = np.delete(values[0], h, axis=0).mean(0)
                    pred = np.delete(values[source], s, axis=0).mean(0)
                    row[route + "_mse"] = float(np.square(pred - target).mean())
                lodo.append(row)
    records = []
    for i, subset in enumerate(subsets):
        record = {"sources": "+".join(species[j + 1] for j in subset), "source_count": len(subset)}
        for route in scores:
            for metric in ["mean_target_mse", "mean_individual_target_mse", "pearson_mean_target", "response_rms"]:
                record[route + "_" + metric] = float(scores[route][metric][i])
            record[route + "_measured_donor_quantiles"] = np.quantile(sensitivity[route][:,i], [.025,.5,.975])
        record["humanmapped_minus_native_mse"] = record["humanmapped_mean_target_mse"] - record["native_mean_target_mse"]
        record["paired_route_difference_quantiles"] = np.quantile(sensitivity["humanmapped"][:,i] - sensitivity["native"][:,i], [.025,.5,.975])
        records.append(record)
    report = {"specification_sha256": file_hash(args.out_dir / "specification.json"), "sources": species[1:], "donors": 27, "genes": len(genes), "subsets": len(subsets),
        "human_routes_identical": True, "human_response_rms": float(np.sqrt(np.square(native[0].mean(0)).mean())),
        "single_sources": single, "single_source_mse_rank_spearman": float(pearson(ranks(scores["native"]["mean_target_mse"][:8]), ranks(scores["humanmapped"]["mean_target_mse"][:8]))),
        "all_subset_mse_rank_spearman": float(pearson(ranks(scores["native"]["mean_target_mse"]), ranks(scores["humanmapped"]["mean_target_mse"]))),
        "source_subset_records": records, "added_source_edges": edges,
        "edge_summary": {"edges": len(edges), "native_harmful": sum(r["native_harmful"] for r in edges), "humanmapped_harmful": sum(r["humanmapped_harmful"] for r in edges), "direction_agrees": sum(r["direction_agrees"] for r in edges), "harm_status_agrees": sum(r["native_harmful"] == r["humanmapped_harmful"] for r in edges)},
        "single_source_donor_deletion_sensitivity": lodo,
        "computation": {"device": args.device, "float": "float64", "resampling_seconds": seconds, "literal_parity_max_abs_error": max_error},
        "inference_limits": specification["scope"] + " " + specification["mapping_caveat"] + " " + specification["measured_donor_sensitivity"]["interpretation"]}
    save(args.out_dir / "report.json", report)
    with (args.out_dir / "all_subset_scores.tsv").open("w", newline="") as handle:
        fields = [k for k in records[0] if "quantiles" not in k]
        writer = csv.DictWriter(handle, fields, delimiter="\t");writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
    np.savez_compressed(args.out_dir / "donor_sensitivity.npz", counts=counts, weights=weights,
                        native_losses=sensitivity["native"], humanmapped_losses=sensitivity["humanmapped"],
                        human_gene_ids=genes, source_order=np.asarray(species[1:]))
    print(json.dumps(plain({"single_sources": single, "single_rank_spearman": report["single_source_mse_rank_spearman"], "all_subset_rank_spearman": report["all_subset_mse_rank_spearman"], "edges": report["edge_summary"], "computation": report["computation"]}), indent=2))


if __name__ == "__main__": main()
