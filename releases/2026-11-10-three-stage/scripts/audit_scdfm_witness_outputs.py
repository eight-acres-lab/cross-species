#!/usr/bin/env python3
"""Independently verify saved witness-sensitivity predictions with SciPy float64.

Preserves the original FP32 CUDA report and reports numerical precision deltas.
No prediction or fitted model is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(prediction,truth):
    x,y=prediction.astype("float64"),truth.astype("float64")
    xx,yy,xy=cdist(x,x),cdist(y,y),cdist(x,y)
    return {"energy_v":float(2*xy.mean()-xx.mean()-yy.mean()),
            "energy_u":float(2*xy.mean()-xx.sum()/(len(x)*(len(x)-1))-yy.sum()/(len(y)*(len(y)-1))),
            "mean_mse":float(np.mean((x.mean(0)-y.mean(0))**2))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--processed-dir",type=Path,required=True)
    args=parser.parse_args()
    spec_path=args.run_dir/"specification.json"
    spec=json.loads(spec_path.read_text())
    report_path=args.run_dir/"report.json"
    report=json.loads(report_path.read_text())
    if report["specification_sha256"]!=sha(spec_path):
        raise ValueError("Specification seal mismatch")
    if report["probe_only"]:
        conditions=list(spec["evaluation_populations"])
    else:
        conditions=spec["roles"]["test"]
    matrix=sparse.load_npz(args.processed_dir/"expression_lognorm_hvg5000.npz")
    columns=np.asarray(spec["gene_columns"])
    by_method={}
    for method,original in report["evaluation"].items():
        by_condition={}
        paths={}
        differences=[]
        precision_deltas={}
        for condition in conditions:
            path=args.run_dir/(method+"_"+condition.replace("/","_")+"_predictions.npz")
            with np.load(path,allow_pickle=False) as archive:
                pred,raw,truth=[archive[key].copy() for key in ["predictions","raw_predictions","targets"]]
                genes=archive["gene_names"].astype(str).tolist()
                saved_condition=str(archive["condition"])
            if pred.shape!=(128,1000) or raw.shape!=pred.shape or truth.ndim!=2 or truth.shape[1]!=1000:
                raise ValueError("Unexpected cell/gene shape")
            if not all(np.isfinite(v).all() for v in [pred,raw,truth]) or not np.array_equal(pred,np.maximum(raw,0)):
                raise ValueError("Nonfinite values or raw/clipped readout mismatch")
            if genes!=spec["gene_names"] or saved_condition!=condition:
                raise ValueError("Gene order or condition mismatch")
            expected=matrix[np.asarray(spec["evaluation_populations"][condition]["target_rows"])][:,columns].toarray()
            if not np.array_equal(expected,truth):
                raise ValueError("Saved target differs from original measured cells")
            values=metrics(pred,truth)
            precision_deltas[condition]={}
            for key,value in values.items():
                delta=value-original["by_condition"][condition][key]
                precision_deltas[condition][key]=delta
                differences.append(abs(delta))
                if abs(delta)>2e-4:
                    raise ValueError("Recomputed metric exceeds predeclared2e-4 precision tolerance: "+method+"/"+condition+"/"+key)
            by_condition[condition]=values
            paths[condition]=sha(path)
        macro={key:float(np.mean([v[key] for v in by_condition.values()])) for key in ["energy_v","energy_u","mean_mse"]}
        for key in macro:
            if abs(macro[key]-original["macro"][key])>2e-4:
                raise ValueError("Macro metric precision check fails")
        by_method[method]={"by_condition_float64":by_condition,"macro_float64":macro,"maximum_abs_difference_from_original_fp32":max(differences),"prediction_file_sha256":paths,"float64_minus_fp32_by_condition":precision_deltas,
                           "float64_minus_fp32_macro":{key:macro[key]-original["macro"][key] for key in macro}}
    contrasts={}
    if not report["probe_only"]:
        for key,original in report["comparisons"].items():
            baseline,method=key.split("_minus_")
            original_delta=np.array([report["evaluation"][baseline]["by_condition"][c]["energy_v"]-report["evaluation"][method]["by_condition"][c]["energy_v"] for c in conditions])
            if not np.array_equal(original_delta,np.asarray(original["per_condition_gain"])):
                raise ValueError("Reported paired gains are inconsistent")
            draws=np.random.default_rng(0).integers(0,7,(2000,7))
            interval=np.quantile(original_delta[draws].mean(1),[.025,.975])
            if not np.allclose(interval,original["conditional_condition_bootstrap_95"],rtol=0,atol=1e-14):
                raise ValueError("Bootstrap replay differs")
            delta=np.array([by_method[baseline]["by_condition_float64"][c]["energy_v"]-by_method[method]["by_condition_float64"][c]["energy_v"] for c in conditions])
            contrasts[key]={"float64_gain":float(delta.mean()),"float64_interval":np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),"per_condition_gain_float64":delta.tolist(),"conditions":conditions,"positive_conditions_float64":int((delta>0).sum()),"original_fp32_gain":original["macro_gain"],"absolute_gain_precision_difference":abs(float(delta.mean())-original["macro_gain"])}
    audit={"original_report_sha256":sha(report_path),"specification_sha256":sha(spec_path),"methods":by_method,"contrasts":contrasts,
           "tolerance_abs":2e-4,"predictions_and_models_changed":False,"original_report_preserved":True,
           "precision_note":"SciPy Euclidean distances use float64 and exact-zero diagonals; original torch.cdist/means use FP32. Precision tolerance was declared using fitting-only probe before formal test scoring.","audit_script_sha256":sha(__file__)}
    target=args.run_dir/"output_audit.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    precision_amendment=args.run_dir/"evaluation_precision_amendment.json"
    float64_report={"evaluation":{method:{"by_condition":row["by_condition_float64"],"macro":row["macro_float64"]} for method,row in by_method.items()},
        "comparisons":{key:{"macro_gain":row["float64_gain"],"conditional_condition_bootstrap_95":row["float64_interval"],"per_condition_gain":row["per_condition_gain_float64"],"conditions":row["conditions"],"positive_conditions":row["positive_conditions_float64"]} for key,row in contrasts.items()},
        "arithmetic":"Independent SciPy float64 Euclidean distances and means from identical frozen saved predicted/measured values; self-distances zero; no model/prediction/selection change.",
        "original_report_sha256":sha(report_path),"specification_sha256":sha(spec_path),"audit_script_sha256":sha(__file__),
        "precision_amendment_sha256":sha(precision_amendment) if precision_amendment.exists() else None,
        "float64_minus_fp32":{method:{"by_condition":row["float64_minus_fp32_by_condition"],"macro":row["float64_minus_fp32_macro"]} for method,row in by_method.items()},
        "interval_warning":"Descriptive resampling of seven fixed chemical-condition losses, conditional on fitted predictions, one optimization/generation seed, shared drug components and one A549 context. No biological-replicate/population coverage guarantee."}
    float64_path=args.run_dir/"evaluation_float64.json"
    if float64_path.exists():
        raise FileExistsError(float64_path)
    float64_path.write_text(json.dumps(float64_report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"verified_methods":list(by_method),"contrasts":contrasts,"max_precision_delta":max(v["maximum_abs_difference_from_original_fp32"] for v in by_method.values())},indent=2))


if __name__=="__main__":
    main()
