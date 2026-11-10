#!/usr/bin/env python3
"""Published-witness-only scDFM sensitivity; fixed settings, no test selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cross_species.official_scdfm import load_official_scdfm
from cross_species.official_state import sha256_file
from cross_species.scdfm_feedback import energy_u
from cross_species.scdfm_witness import fit_witness, rk4_endpoint
from run_scdfm_nft_benchmark import make_roles
from run_official_state_benchmark import write_json


def build_schedule(frame, roles, *, updates=200, seed=0):
    rng = np.random.default_rng(seed)
    controls = np.flatnonzero(frame.condition.eq("control").to_numpy())
    pools, drug_ids = {}, {}
    for condition in roles["train"]:
        rows = np.flatnonzero(frame.condition.eq(condition).to_numpy())
        if len(rows) < 64:
            raise ValueError("Need 64 distinct measured target cells per fitting condition")
        pools[condition] = rng.choice(rows, 64, replace=False)
        drug_ids[condition] = frame.loc[rows[0], ["drug_id_1", "drug_id_2"]].to_numpy(dtype="int64")
    schedule = []
    for step in range(updates):
        condition = str(rng.choice(roles["train"]))
        # Independent event IDs survive repeated draws of the same physical cell.
        events = rng.choice(controls, 8, replace=True)
        schedule.append({"update": step, "condition": condition, "control_event_rows": events,
                         "control_rows": np.repeat(events, 8), "drug_ids": drug_ids[condition],
                         "target_pool_order": rng.permutation(64)})
    return pools, schedule


@torch.no_grad()
def evaluate_one(model, matrix, record, columns, token_ids, *, device, seed=0):
    controls = matrix[record["control_rows"]][:, columns].toarray()
    real = torch.as_tensor(matrix[record["target_rows"]][:, columns].toarray(), device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randn(controls.shape, device=device, generator=generator)
    pieces = []
    for start in range(0, 128, 64):
        basal = torch.as_tensor(controls[start:start+64], device=device)
        genes = torch.as_tensor(token_ids[columns], device=device).expand(len(basal), -1).contiguous()
        pert = torch.as_tensor(record["drug_ids"], device=device).expand(len(basal), -1).contiguous()
        pieces.append(rk4_endpoint(model, basal, genes, pert, initial[start:start+64]))
    raw = torch.cat(pieces)
    prediction = raw.clamp_min(0)
    xx, yy, xy = torch.cdist(prediction, prediction), torch.cdist(real, real), torch.cdist(prediction, real)
    energy_v = 2*xy.mean()-xx.mean()-yy.mean()
    metrics = {"energy_v": float(energy_v.cpu()), "energy_u": float(energy_u(prediction, real).cpu()),
               "mean_mse": float((prediction.mean(0)-real.mean(0)).square().mean().cpu()),
               "generated_cells": 128, "real_cells": len(real)}
    return metrics, raw.cpu().numpy(), real.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "vendor-root", "processed-dir", "out-dir"]:
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="Exactly one update per arm and train-only evaluation, no held-out labels")
    parser.add_argument("--runtime-probe", type=Path, help="Completed CUDA probe report required before a formal sensitivity run")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output directory must be empty")
    if not args.probe and (args.runtime_probe is None or not args.runtime_probe.is_file()):
        parser.error("Formal run requires the completed runtime probe")
    resource_probe = None if args.runtime_probe is None else json.loads(args.runtime_probe.read_text())
    if resource_probe is not None and (not resource_probe.get("probe_only") or resource_probe["runtime_gate"]["total_seconds"] > 6000):
        parser.error("The completed probe must satisfy the approved 100-minute gate")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.processed_dir/"preprocessing_manifest.json").read_text())
    for name, value in manifest["files"].items():
        if sha256_file(args.processed_dir/name) != value["sha256"]:
            raise ValueError("Processed input SHA mismatch: "+name)
    frame = pd.read_csv(args.processed_dir/"metadata.csv", keep_default_na=False)
    matrix = sparse.load_npz(args.processed_dir/"expression_lognorm_hvg5000.npz")
    all_genes = np.load(args.processed_dir/"gene_names.npy", allow_pickle=False)
    genes = np.load(args.processed_dir/"official_test_hvg1000_gene_names.npy", allow_pickle=False)
    token_ids = np.load(args.processed_dir/"gene_token_ids.npy", allow_pickle=False)
    lookup = {str(g):i for i,g in enumerate(all_genes)}
    columns = np.array([lookup[str(g)] for g in genes])
    roles = make_roles(frame, 20261108)  # Preserve previously frozen original-seven-test allocation.
    pools, full_schedule = build_schedule(frame, roles)
    schedule = full_schedule[:1] if args.probe else full_schedule
    controls = np.flatnonzero(frame.condition.eq("control").to_numpy())
    populations = {}
    eval_roles = roles["train"][:1] if args.probe else roles["test"]
    for condition in eval_roles:
        rows = np.flatnonzero(frame.condition.eq(condition).to_numpy())
        seed = int(hashlib.sha256(("0|witness-rk4-eval|"+condition).encode()).hexdigest()[:15],16)
        rng = np.random.default_rng(seed)
        populations[condition] = {"control_rows": rng.choice(controls,128,replace=True),
            "target_rows": pools[condition] if args.probe else rows,
            "drug_ids": frame.loc[rows[0],["drug_id_1","drug_id_2"]].to_numpy(dtype="int64"), "noise_seed": seed}
    spec = {"scope": "Witness-only sensitivity aligned with PerturbCellRL v2 Appendix C velocity objective, on the released scDFM architecture and weights. No expression-support/pathway terms; no complete reproduction claim.",
        "source_paper": "https://arxiv.org/html/2606.27752v2#SSc0.SS3", "post_hoc": True,
        "design_timing": "Designed after previous custom population-NFT and native-FM test scores were viewed. New outcomes are not used to alter this recipe.",
        "roles": roles, "role_allocation_seed": 20261108, "training_and_noise_seed": 0,
        "fixed_real_reference_pools": pools, "reference_pool_M_per_condition":64,
        "actual_unique_measured_fitting_cells":len(np.unique(np.concatenate(list(pools.values())))),
        "available_original_cells":len(frame), "schedule": full_schedule, "evaluation_populations":populations,
        "gene_names":genes, "gene_columns":columns, "gene_token_ids":token_ids[columns],
        "updates":200, "groups":8, "rollouts_per_group":8, "microbatch":32, "accumulation":2,
        "control_sampling":"Eight iid draws with replacement per update; group ID denotes sampling event, including repeated physical cells.",
        "optimizer":{"name":"Adam", "lr":5e-7,"betas":[.9,.999],"eps":1e-8,"weight_decay":0,"gradient_clip_norm":.25},
        "precision":"Autocast BF16 forward, FP32 master parameters, ODE state and loss operands; TF32 disabled.",
        "nft":{"gamma":1,"C":2,"reference_velocity_mse_weight":1,"old_update":"Exact live copy after each optimizer update", "eval_ema":"min((1+s)/(10+s),.9), zero-based s; first decay .1", "objective":"Eq57 velocity MSE, without adaptive clean reconstruction denominator", "reward":"Euclidean witness excludes entire control event; generated readout clip>=0; real scale sqrt median squared pair distance; two group z scores, population std divisor8, reject zero std"},
        "native_control":"Official scDFM affine Gaussian FM plus MMD(.5), plus the same weight1 original-reference velocity MSE; same real64 pool, 64 examples, two32 microbatches, LR, updates and eval EMA. It does not acquire unused generated trajectories, so FLOPs differ.",
        "ode":{"library":"torchdiffeq.odeint","method":"rk4","grid_points":20,"intervals":19,"atol":1e-4,"rtol":1e-4},
        "evaluation":"All seven original held-out conditions; 128 independently generated cells and all original real cells per condition, published energy V including diagonal as primary. One fixed draw; no condition/hyperparameter selection.",
        "deviations":["Witness-only; omitted support/pathway terms.","Fixed64 real-reference cells per fitting condition chosen for matched finite label budget; paper permits available real reference responses.","Adam retained from scDFM; PerturbCellRL appendix does not specify optimizer identity.","Original author transductive5000/test1000 HVG preprocessing retained; continuation and readout use the author1000-HVG axis, released pretraining used5000.","FP32 master weights and BF16 autocast; native FM shares reference penalty as our matched control.","Our fixed18-fit/6-original-training-calibration split; calibration is not used to tune any coefficient.","One seed and one A549 context; no biological-replicate or cross-species claim."],
        "resource_gate":{"approved_max_seconds":6000,"original_gate_seconds":3600,"revision_reason":"Completed genuine CUDA probe projected99.2min; parent explicitly authorized100min before formal run, retaining all scientific settings.","probe_report_sha256":None if args.runtime_probe is None else sha256_file(args.runtime_probe),"probe_runtime_projection":None if resource_probe is None else resource_probe["runtime_gate"]},
        "probe_only":args.probe,"input_hashes":{"checkpoint":sha256_file(args.checkpoint),"manifest":sha256_file(args.processed_dir/"preprocessing_manifest.json")},
        "code_hashes":{str(path.relative_to(ROOT)):sha256_file(path) for path in [Path(__file__),ROOT/"src/cross_species/scdfm_witness.py",ROOT/"src/cross_species/official_scdfm.py",ROOT/"src/cross_species/scdfm_feedback.py",ROOT/"scripts/run_scdfm_nft_benchmark.py",ROOT/"scripts/run_official_state_benchmark.py"]}}
    write_json(args.out_dir/"specification.json",spec)
    reference,metadata = load_official_scdfm(args.checkpoint,args.vendor_root,args.processed_dir/"mask_k30_signed_pad4.pt",device="cuda")
    reference.eval().requires_grad_(False)
    fits={}
    for method in ["witness","native_fm"]:
        models=[load_official_scdfm(args.checkpoint,args.vendor_root,args.processed_dir/"mask_k30_signed_pad4.pt",device="cuda")[0] for _ in range(3)]
        model,old,ema=models
        print(json.dumps({"event":"fit_started","method":method,"probe":args.probe}),flush=True)
        fits[method]=fit_witness(model,old,reference,ema,matrix,schedule,pools,columns,token_ids,method=method,device="cuda")
        write_json(args.out_dir/(method+"_fit.json"),fits[method])
        torch.save({"state_dict":{k:v.detach().cpu() for k,v in ema.state_dict().items()},"raw_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"specification_sha256":sha256_file(args.out_dir/"specification.json")},args.out_dir/(method+".pt"))
        del model,old,ema,models
        torch.cuda.empty_cache()
    results={}
    for method in ["released_checkpoint","witness","native_fm"]:
        model,_=load_official_scdfm(args.checkpoint,args.vendor_root,args.processed_dir/"mask_k30_signed_pad4.pt",device="cuda")
        if method!="released_checkpoint":
            model.load_state_dict(torch.load(args.out_dir/(method+".pt"),weights_only=True,map_location="cpu")["state_dict"],strict=True)
        model.eval()
        by_condition={}
        start=time.monotonic()
        for condition,record in populations.items():
            metrics,raw,real=evaluate_one(model,matrix,record,columns,token_ids,device="cuda",seed=record["noise_seed"])
            by_condition[condition]=metrics
            np.savez_compressed(args.out_dir/(method+"_"+condition.replace("/","_")+"_predictions.npz"),raw_predictions=raw,predictions=np.maximum(raw,0),targets=real,gene_names=genes,condition=np.array(condition))
            print(json.dumps({"event":"evaluation","method":method,"condition":condition,"metrics":metrics}),flush=True)
        results[method]={"by_condition":by_condition,"macro":{k:float(np.mean([v[k] for v in by_condition.values()])) for k in ["energy_v","energy_u","mean_mse"]},"evaluation_seconds":time.monotonic()-start}
        write_json(args.out_dir/(method+"_evaluation.json"),results[method])
        del model
        torch.cuda.empty_cache()
    comparisons={}
    if not args.probe:
        for baseline,method in [("released_checkpoint","witness"),("released_checkpoint","native_fm"),("native_fm","witness")]:
            delta=np.array([results[baseline]["by_condition"][c]["energy_v"]-results[method]["by_condition"][c]["energy_v"] for c in roles["test"]])
            rng=np.random.default_rng(0)
            draws=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
            comparisons[baseline+"_minus_"+method]={"macro_gain":float(delta.mean()),"conditional_condition_bootstrap_95":np.quantile(draws,[.025,.975]),"per_condition_gain":delta,"positive_conditions":int((delta>0).sum()),"warning":"Seven fixed conditions, fixed trained predictions; no biological-replicate/population coverage guarantee."}
    report={"scope":spec["scope"],"probe_only":args.probe,"fits":fits,"evaluation":results,"comparisons":comparisons,"checkpoint_metadata":metadata,"specification_sha256":sha256_file(args.out_dir/"specification.json"),"cuda":{"device":torch.cuda.get_device_name(),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}}
    if args.probe:
        train_projection=200*sum(row["wall_seconds"] for row in fits.values())
        eval_projection=7*sum(row["evaluation_seconds"] for row in results.values())
        report["runtime_gate"]={"projected_two_arm_training_seconds":train_projection,"projected_three_arm_evaluation_seconds":eval_projection,"plus_setup_margin_seconds":180,"total_seconds":train_projection+eval_projection+180,"complete_if_at_most_seconds":3600}
    write_json(args.out_dir/"report.json",report)
    print(json.dumps({"event":"complete","probe":args.probe,"results":{k:v["macro"] for k,v in results.items()},"runtime_gate":report.get("runtime_gate")}),flush=True)


if __name__=="__main__":
    main()
