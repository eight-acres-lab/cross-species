#!/usr/bin/env python3
"""Frozen small-panel multisource baselines and measured-arm sensitivity."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cross_species.multisource_data import load_curated_panel,file_hash
from cross_species.multisource_models import (source_subsets,fit_lopo_predictions,response_scores,
    independent_arm_counts,arm_resampling_losses,fixed_total_profile_budget)


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def save(path,value):
    Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+"\n")


def array_contract(panel):
    species=[str(v) for v in panel.species_order]
    if species[0]!="human" or len(species)!=5:
        raise ValueError("Expected human followed by four ordered source species")
    tasks=sorted({r["perturbation"] for r in panel.contrast_metadata})
    lookup={(r["species"],r["perturbation"]):r for r in panel.contrast_metadata}
    if len(lookup)!=len(species)*len(tasks):
        raise ValueError("Require complete same-query coverage across all sources")
    mask=np.asarray(panel.global_eligible_mask,dtype=bool)
    if not mask.any() or mask.shape!=panel.feature_ids.shape:
        raise ValueError("Empty/incompatible globally frozen axis")
    arms,records=[],[]
    for task in tasks:
        values=[]
        for s in species:
            record=lookup[s,task];cid=record["contrast_id"]
            rows=[panel.arm_indices(cid,role) for role in ["treated","vehicle"]]
            if any(len(r)!=3 for r in rows):
                raise ValueError("The fixed profile budget requires three records per arm")
            values.append(np.stack([panel.arm_logcpm[r][:,mask] for r in rows]))
            records.append({"perturbation":task,"species":s,"contrast_id":cid,"treated_rows":rows[0].tolist(),"vehicle_rows":rows[1].tolist(),
                            "time_h":record["time_h"],"dose_value":record["dose_value"],"dose_unit":record["dose_unit"]})
        arms.append(values)
    arms=np.asarray(arms,dtype=float)
    if not np.isfinite(arms).all():
        raise ValueError("Global axis contains a missing expression value")
    return arms,tasks,species,records,panel.feature_ids[mask]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--n-resamples",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261108)
    args=parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Output must be a new/empty directory")
    if args.n_resamples<2:
        parser.error("Need at least two sensitivity resamples")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    panel=load_curated_panel(args.panel)
    arms,tasks,species,records,genes=array_contract(panel)
    if tasks!=["CDDO","KI696","THAP"]:
        raise ValueError("The GSE330151 analysis expects three documented drugs")
    source_ids=species[1:];subsets=source_subsets(source_ids)
    contexts=["NRF2_24h","NRF2_24h","UPR_8h"]
    spec={"schema":"multisource_baseline_v1","frozen_utc":datetime.now(timezone.utc).isoformat(),"dataset":"GSE330151",
        "retrospective":True,"prior_human_outcomes_inspected":True,"scope":"Descriptive3-drug/2-pathway primary-hepatocyte panel and exploratory LOPO; no independent validation or stable population CI.",
        "human_queries":tasks,"source_order":source_ids,"source_subsets":subsets,"global_gene_count":len(genes),"gene_axis_sha256":hashlib.sha256("\n".join(genes.astype(str)).encode()).hexdigest(),
        "records":records,"feature_selection":"All90arms complete on same fresh five-way axis; never change axis by subset or response effect.",
        "input_manifest_sha256":file_hash(args.panel/"manifest.json"),"panel_sha256":file_hash(args.panel/"panel.npz"),
        "training":"For each held perturbation, only other2 perturbations' human/source means fit scalar simplex weights and bestfixedsource. No intercept/gene-specific parameters.",
        "simplex_ridge":1.,"simplex_penalty":"Mean_task,gene MSE +1*||w-uniform||², nonnegative sum1. Fixed without terminal tuning.",
        "source_selection_visibility":"Bestfixedsource selects its source using training-task source-copy losses before acquiring the held-out source response. Held-out human and source responses are excluded from source selection.",
        "context_rule":{"fixed_known_categories":contexts,"human_baseline":"Mean human response among training tasks in the same pathway/time category; if none are available, the mean across all training tasks is used.","limitation":"Only3drugs with pathway/time confounding. This is not a validated strong molecular/context prior or causal factor attribution."},
        "context_selector":"Bestfixedhistory selects a source using the two training tasks before held-out source measurement; context-dependent selection is not fitted.",
        "added_source_budget":"Each acquired source uses all3treated+3vehicle profiles. More sources therefore add nominal measured-profile budget.",
        "fixed_total_budget":"Exact finite-record expected MSE for3treated+3vehicle profiles total across sources, independent without-replacement draws within each arm, balanced allocations, equal source weights. Four sources cannot satisfy this6-profile budget and are marked ineligible.",
        "random_control":"Expected loss of uniformly choosing one source, mean of four single-source losses; not loss of their mean prediction.",
        "phylogeny_proxy_control":"Fixed cyno single source, named control only; no fitted phylogenetic model.",
        "oracle":"Best single source or subset chosen using heldhuman is terminal diagnostic only; excluded from deployable rankings.",
        "primary":"Perperturbation gene MSE and equal-perturbation macro; Pearson/cosine secondary with undefined values preserved.",
        "measurement_resampling":{"draws":args.n_resamples,"seed":args.seed,"unit":"Labelled original samples resampled independently within treated/vehicle arms of each species×drug. Allmethods share sampled human/source arms.","fixed_estimator":True,"paired_reference_controls":["best_fixed_source_history","uniform_random_single_source_expected_loss"],"warning":"Biological IDs/independence unknown; percentile ranges are measured-arm sensitivity, not donor/generalization confidence intervals. No task/gene bootstrap or significance claim."},
        "device_requested":args.device,"precision":"Torch float64 batched resampling, CPU same-count first4draws allmethods parity guard rtol1e-10/atol1e-10.",
        "code_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/"src/cross_species/multisource_models.py",ROOT/"src/cross_species/multisource_data.py"]}}
    save(args.out_dir/"specification.json",spec)
    means=arms.mean(3);responses=means[:,:,0]-means[:,:,1]
    sources,human=responses[:,1:],responses[:,0]
    predictions,weights,history=fit_lopo_predictions(sources,human,source_ids,ridge=1.,known_context=contexts)
    scores={name:response_scores(value,human) for name,value in predictions.items()}
    source_methods=["equal/"+s for s in source_ids]
    source_losses=np.stack([scores[name]["mse_by_query"] for name in source_methods],axis=1)
    random_loss=source_losses.mean(1)
    fixed_cyno=scores["equal/cyno"]
    oracle_single=np.argmin(source_losses,axis=1)
    equal_names=[name for name in predictions if name.startswith("equal/")]
    equal_losses=np.stack([scores[name]["mse_by_query"] for name in equal_names],axis=1)
    oracle_subset=np.argmin(equal_losses,axis=1)
    fixed_total=fixed_total_profile_budget(arms,subsets,budget_per_role=3)
    edges=[]
    for before in subsets:
        for extra in range(4):
            if extra in before:continue
            after=tuple(sorted(before+(extra,)))
            before_name="equal/"+"+".join(source_ids[i] for i in before)
            after_name="equal/"+"+".join(source_ids[i] for i in after)
            gain=scores[before_name]["mse_by_query"]-scores[after_name]["mse_by_query"]
            edges.append({"before":before_name,"added_source":source_ids[extra],"after":after_name,
                          "mse_gain_by_query":gain,"macro_gain":float(gain.mean()),"harmful_queries":int((gain<0).sum()),
                          "interpretation":"Added-source measured-profile budget; equal-mean estimator changes allweights. This is empirical estimator harm/benefit, not intrinsic or causal source utility."})
    methods=list(weights);weight_array=np.stack([weights[name] for name in methods])
    draw_counts=independent_arm_counts(args.n_resamples,len(tasks),len(species),3,args.seed)
    np.savez_compressed(args.out_dir/"frozen_inputs_predictions.npz",arm_profiles=arms,human=human,source_responses=sources,
                        gene_ids=genes,source_ids=np.array(source_ids),query_ids=np.array(tasks),
                        prediction_names=np.array(list(predictions)),predictions=np.stack(list(predictions.values())),
                        resampling_method_names=np.array(methods),fixed_source_weights=weight_array,draw_counts=draw_counts)
    cpu_start=time.monotonic()
    cpu=arm_resampling_losses(arms,weight_array,draw_counts[:4],device="cpu",chunk_size=4)
    cpu_seconds=time.monotonic()-cpu_start
    start=time.monotonic()
    bootstrap=arm_resampling_losses(arms,weight_array,draw_counts,device=args.device,chunk_size=16)
    if args.device=="cuda":torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    difference=float(np.max(np.abs(cpu-bootstrap[:4])))
    if not np.allclose(cpu,bootstrap[:4],rtol=1e-10,atol=1e-10):
        raise ValueError("Batched device arithmetic differs from CPU fixed draws")
    save(args.out_dir/"device_verification.json",{"requested_device":args.device,"actual_device":torch.cuda.get_device_name() if args.device=="cuda" else "cpu",
         "torch_version":torch.__version__,"float_dtype":"float64","cpu_first4_seconds":cpu_seconds,"all_draw_seconds":elapsed,"cpu_device_max_abs_difference":difference,
         "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else None})
    np.savez_compressed(args.out_dir/"measurement_resampling.npz",losses=bootstrap,method_names=np.array(methods),query_ids=np.array(tasks))
    resampling={name:{"per_query_percentile_2_5_97_5":np.quantile(bootstrap[:,:,i],[.025,.975],axis=0),
                       "macro_percentile_2_5_97_5":np.quantile(bootstrap[:,:,i].mean(1),[.025,.975])} for i,name in enumerate(methods)}
    paired_sensitivity={}
    fixed_loss=bootstrap[:,:,methods.index("best_fixed_source_history")]
    random_losses=np.stack([bootstrap[:,:,methods.index(name)] for name in source_methods]).mean(0)
    for i,name in enumerate(methods):
        paired_sensitivity[name]={}
        for baseline,losses in [("best_fixed_source_history",fixed_loss),("uniform_random_single_source_expected_loss",random_losses)]:
            gains=losses-bootstrap[:,:,i]
            point=(scores["best_fixed_source_history"]["mse_by_query"] if baseline=="best_fixed_source_history" else random_loss)-scores[name]["mse_by_query"]
            paired_sensitivity[name][baseline]={"point_mse_gain_by_query":point,"point_macro_mse_gain":float(point.mean()),
                "measurement_resampling_per_query_percentiles":np.quantile(gains,[.025,.975],axis=0),
                "measurement_resampling_macro_percentiles":np.quantile(gains.mean(1),[.025,.975]),
                "warning":"Fixed-estimator arm-record sensitivity, not a task/generalization confidence interval; source acquisition budgets differ where subset sizes differ."}
    drivers=[]
    for t,task in enumerate(tasks):
        for s,source in enumerate(source_ids):
            baseline_delta=means[t,s+1,1]-means[t,0,1]
            drivers.append({"query":task,"source":source,"known_time_h":next(r["time_h"] for r in records if r["perturbation"]==task and r["species"]==source),
                            "source_vehicle_human_vehicle_mse":float(np.square(baseline_delta).mean()),
                            "source_response_rms":float(np.sqrt(np.square(sources[t,s]).mean())),"human_response_rms_terminal":float(np.sqrt(np.square(human[t]).mean())),
                            "human_prediction_mse_terminal":float(source_losses[t,s]),"scope":"Descriptive association only; source vehicle/magnitude uses acquired data, human outcomes terminal. One tissue, pathway/time/drug confounding; cannot identify causal drivers or validate premeasurement prediction."})
    report={"specification_sha256":file_hash(args.out_dir/"specification.json"),"query_ids":tasks,"source_order":source_ids,"genes":len(genes),
        "scores":scores,"fitting_and_premeasurement_selection":history,
        "uniform_random_single_source":{"mse_by_query":random_loss,"macro_mse":float(random_loss.mean()),"unit_budget":"One source,3treated+3vehicle profiles"},
        "fixed_cyno_control":fixed_cyno,"oracle_diagnostics":{"single_source_by_query":[source_ids[i] for i in oracle_single],"single_mse_by_query":source_losses[np.arange(3),oracle_single],
        "subset_by_query":[equal_names[i] for i in oracle_subset],"subset_mse_by_query":equal_losses[np.arange(3),oracle_subset],"deployable":False},
        "added_source_edges":edges,"fixed_total_profile_budget":fixed_total,"postmeasurement_descriptive_factors":drivers,
        "fixed_estimator_measured_arm_sensitivity":resampling,"paired_fixed_estimator_arm_sensitivity":paired_sensitivity,"sensitivity_warning":spec["measurement_resampling"]["warning"],
        "remaining_identifiability_limits":["3drugs/2pathways; no stable generalization confidence interval.","Retrospective evaluation on published human outcomes.","No validated strong human molecular prior; zero/history/contextmean controls are limited.","Each leave-one-perturbation-out split supplies two training tasks for source selection.","Biological donor identities and cross-arm pairing unresolved.","Four-source totalbudget3-per-role comparison impossible with at least one profile/source.","Source species, cell system, protocol and donor mixture are not factorially separated.","Arm normalization uses a fixed complete native gene support; missing565dog genes were excluded from its library support, never imputed aszero."]}
    save(args.out_dir/"report.json",report)
    print(json.dumps({"queries":tasks,"genes":len(genes),"methods":len(scores),"source_subsets":len(subsets),"device":args.device,"resampling_seconds":elapsed,"cpu_device_max_abs_difference":difference,
                      "single_source_macro_mse":{s:scores['equal/'+s]['macro_mse'] for s in source_ids},"bestfixed_history_macro_mse":scores['best_fixed_source_history']['macro_mse'],
                      "allsource_equal_macro_mse":scores['equal/'+'+'.join(source_ids)]['macro_mse']},indent=2),flush=True)


if __name__=="__main__":main()
