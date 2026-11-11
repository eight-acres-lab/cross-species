#!/usr/bin/env python3
"""Frozen donor-held-out rabbit/human LPS comparison using named State weights.

The training and prediction phase never scores target LPS outcomes. Every fold,
arm, selected checkpoint, diagnostic shrink value and target prediction is
frozen before the separate scoring phase can read those outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_species.named_gene_state import load_named_gene_state,extend_named_model,predict_contrast,sha256_file

ARMS=('H','H+Rabbit','H+Hextra','H+RabbitA0')
CONTRACT_SHA='b0675040fde2a096770303591984689058f8d30cba1bc5af20c0cf9dff89f885'


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def progress(args,**record):
    record={'unix_time':time.time(),**record}
    with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(plain(record))+'\n')
    print(json.dumps(plain(record)),flush=True)


def load_data(args):
    if sha256_file(args.contract)!=CONTRACT_SHA:raise ValueError('Contract differs from the reviewed frozen v3')
    contract=json.loads(args.contract.read_text());manifest=json.loads((args.prepared/'manifest.json').read_text())
    for name in ['arrays','selections']:
        item=manifest[name];path=args.prepared/Path(item['path']).name
        if sha256_file(path)!=item['sha256']:raise ValueError('Prepared artifact changed: '+name)
    with np.load(args.prepared/'lps_primary_arrays.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    selections=json.loads((args.prepared/'selections.json').read_text())
    ix=data['native_gene_indices']
    if ix.shape!=(4633,) or len(set(ix))!=4633 or not np.all(np.diff(ix)>0) or min(ix)<0 or max(ix)>=6546:
        raise ValueError('Invalid fixed measured-gene mask')
    if data['human_expression'].shape!=(1644,4633) or data['rabbit_expression'].shape!=(638,4633):raise ValueError('Wrong populations')
    for species in ['human','rabbit']:
        if not np.isfinite(data[species+'_expression']).all() or (data[species+'_expression']<0).any():raise ValueError('Invalid coding-normalized expression')
        if len(set(data[species+'_cell_ids'].astype(str)))!=len(data[species+'_expression']):raise ValueError('Repeated actual cells')
    role_names={'human_base':'base_human','human_extra':'extra_human','human_cal':'calibration','human_test':'test'}
    for fold in selections['folds']:
        expected=contract['splitting']['folds'][fold['fold']]
        combined=[]
        for role,name in role_names.items():
            donors=[r['donor'] for r in fold[role]]
            if donors!=expected[name]:raise ValueError('Human donor-role order differs from contract')
            combined.extend(donors)
            for r in fold[role]:
                for part,label in [('control','Baseline'),('target','LPS')]:
                    rows=np.asarray(r[part+'_rows'],int)
                    if data['human_cell_ids'][rows].astype(str).tolist()!=r[part+'_cell_ids']:raise ValueError('Human row ID mismatch')
                    if not np.all(data['human_donors'][rows].astype(str)==r['donor']) or not np.all(data['human_condition'][rows].astype(str)==label):raise ValueError('Human donor/condition mismatch')
        if len(set(combined))!=10:raise ValueError('Human donor leakage')
        for e,a,n,ep in zip(fold['human_extra'],fold['rabbit_source'],fold['rabbit_null'],expected['source_pairing']):
            if (e['donor']!=ep['extra_human_donor'] or a['donor']!=ep['rabbit_donor'] or n['donor']!=a['donor']):raise ValueError('Animal/human source pair changed')
            for part in ['control','target']:
                if len(e[part+'_rows'])!=len(a[part+'_rows']) or len(a[part+'_rows'])!=len(n[part+'_rows']):raise ValueError('Unequal source budgets')
            if a['control_rows']!=n['control_rows'] or set(n['control_rows'])&set(n['target_rows']):raise ValueError('A0 input/target cells not disjoint')
            for r,tt in [(a,6),(n,0)]:
                for part,tm in [('control',0),('target',tt)]:
                    rows=np.asarray(r[part+'_rows'],int)
                    if data['rabbit_cell_ids'][rows].astype(str).tolist()!=r[part+'_cell_ids']:raise ValueError('Rabbit cell provenance mismatch')
                    if not np.all(data['rabbit_donors'][rows].astype(str)==r['donor']) or not np.all(data['rabbit_time_h'][rows]==tm):raise ValueError('Rabbit sample/time mismatch')
    return contract,manifest,selections,data


def pad(values,ix,device):
    x=torch.zeros((len(values),6546),dtype=torch.float32,device=device)
    x[:,torch.as_tensor(ix,device=device)]=torch.as_tensor(values,dtype=torch.float32,device=device)
    return x


def cell_positions(n,seed,step,key):
    salt='nonrodent_singlecell_20261110/v1/step_cells'
    positions=sorted(range(n),key=lambda p:hashlib.sha256(f'{salt}|{seed}|{step}|{key}|{p}'.encode()).hexdigest())
    return np.asarray(positions[:min(64,n)],int)


def batch_record(record,seed,step,key):
    return {part:np.asarray(record[part+'_rows'],int)[cell_positions(len(record[part+'_rows']),seed,step,key+'|'+part)]
            for part in ['control','target']}


def fitting_scale(expression,records):
    values=[float(np.square(expression[r['target_rows']].mean(0,dtype=np.float64)-expression[r['control_rows']].mean(0,dtype=np.float64)).mean()) for r in records]
    if not np.isfinite(values).all():raise ValueError('Nonfinite fitting-only response scale')
    return float(np.mean(values))


def initialize(args,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,metadata=load_named_gene_state(args.checkpoint,device='cpu',source_root=args.state_source)
    model=extend_named_model(model).to(args.device).eval()
    # Original perturbation columns get zero inputs in this assay; preserve them
    # exactly rather than moving them via AdamW's weight decay alone.
    first=next(m for m in model.pert_encoder.original.modules() if isinstance(m,torch.nn.Linear))
    first.weight.requires_grad_(False)
    decoder=getattr(model,'gene_decoder',None)
    if isinstance(decoder,torch.nn.Module):decoder.requires_grad_(False)
    return model,metadata


@torch.no_grad()
def infer(model,expression,record,ix,device):
    controls=expression[record['control_rows']];pred=[];contrast=[];reconstruction=[]
    for start in range(0,len(controls),64):
        x=pad(controls[start:start+64],ix,device)
        result=predict_contrast(model,x,batch='human_new_assay')
        pred.append(result['prediction'][:,ix].cpu().numpy())
        contrast.append(result['contrast'][:,ix].cpu().numpy())
        reconstruction.append(result['native_control'][:,ix].cpu().numpy())
    return {'prediction':np.concatenate(pred),'contrast':np.concatenate(contrast),'native_control':np.concatenate(reconstruction),
            'control_rows':np.asarray(record['control_rows'],int),'target_rows':np.asarray(record['target_rows'],int),
            'donor':record['donor']}


def mse(pred,target):return float(np.square(np.asarray(pred,dtype=np.float64).mean(0)-np.asarray(target,dtype=np.float64).mean(0)).mean())


def calibrate(model,data,fold,ix,args):
    values=[];outputs=[]
    for rec in fold['human_cal']:
        out=infer(model,data['human_expression'],rec,ix,args.device)
        values.append(mse(out['prediction'],data['human_expression'][rec['target_rows']]))
        outputs.append(out)
    return float(np.mean(values)),outputs,values


def make_optimizer(model,contract):
    high=list(model.pert_encoder.extension.parameters());ids={id(p) for p in high}
    core=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    if len(set(map(id,high+core)))!=len(high+core):raise AssertionError('Duplicate optimizer parameters')
    settings=contract['training']
    optimizer=torch.optim.AdamW([{'params':high,'lr':.001},{'params':core,'lr':1e-5}],
                                weight_decay=settings['weight_decay'],betas=tuple(settings['betas']),eps=settings['epsilon'])
    return optimizer,high+core


def fit_arm(args,contract,data,fold,arm,folder):
    seed=20261110+fold['fold'];model,metadata=initialize(args,seed)
    ix=data['native_gene_indices'];human=data['human_expression'];rabbit=data['rabbit_expression']
    scales={'H':fitting_scale(human,fold['human_base']),'Rabbit':fitting_scale(rabbit,fold['rabbit_source']),
            'Hextra':fitting_scale(human,fold['human_extra'])}
    coefficient=0. if arm=='H' else .5*scales['H']/max(scales['Hextra' if arm=='H+Hextra' else 'Rabbit'],1e-8)
    source=human if arm=='H+Hextra' else rabbit
    records=fold['human_extra'] if arm=='H+Hextra' else fold['rabbit_null'] if arm=='H+RabbitA0' else fold['rabbit_source']
    context='human_new_assay' if arm=='H+Hextra' else 'rabbit_new_assay'
    before_batch=model.batch_encoder.weight[:56].detach().clone()
    old_pert=next(m for m in model.pert_encoder.original.modules() if isinstance(m,torch.nn.Linear)).weight.detach().clone()
    score,calibration_predictions,by_donor=calibrate(model,data,fold,ix,args)
    best=score;selected_step=0;calibration=[{'step':0,'macro_mse':score,'donor_mse':by_donor}]
    # Initial prediction must exactly equal observed control cells, not a learned
    # cross-study reconstruction. This check uses calibration controls only.
    for out in calibration_predictions:
        if not np.array_equal(out['prediction'],human[out['control_rows']]):raise AssertionError('Zero-step anchored prediction changed controls')
    torch.save(model.state_dict(),folder/'selected.pt')
    optimizer,parameters=make_optimizer(model,contract)
    history=[];gradient_witness=[];started=time.monotonic()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        hrec=fold['human_base'][step%4];hb=batch_record(hrec,seed,step,hrec['donor'])
        hc=pad(human[hb['control']],ix,args.device);ht=torch.as_tensor(human[hb['target']],device=args.device)
        out=predict_contrast(model,hc,batch='human_new_assay')['prediction'][:,ix]
        human_loss=(out.mean(0)-ht.mean(0)).square().mean()
        source_loss=None
        if arm!='H':
            sr=records[step%2];sb=batch_record(sr,seed,step,'source_slot'+str(step%2))
            sc=pad(source[sb['control']],ix,args.device);st=torch.as_tensor(source[sb['target']],device=args.device)
            sout=predict_contrast(model,sc,batch=context)['prediction'][:,ix]
            source_loss=(sout.mean(0)-st.mean(0)).square().mean()
            if step in (0,2,9):
                chosen=[(n,p) for n,p in model.named_parameters() if p.requires_grad and (
                        n=='pert_encoder.extension.weight' or n.endswith('layers.0.self_attn.v_proj.weight')
                        or n.startswith('basal_encoder.') and n.endswith('weight'))]
                gs=torch.autograd.grad(coefficient*source_loss,[p for _,p in chosen],retain_graph=True,allow_unused=True)
                gradient_witness.append({'step':step+1,'norms':{n:None if g is None else float(g.norm().cpu()) for (n,_),g in zip(chosen,gs)}})
        loss=human_loss if source_loss is None else human_loss+coefficient*source_loss
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Nonfinite training objective or gradient')
        norm=torch.nn.utils.clip_grad_norm_(parameters,10.)
        optimizer.step()
        with torch.no_grad():model.batch_encoder.weight[:56].copy_(before_batch)
        if step==199 and not torch.equal(next(m for m in model.pert_encoder.original.modules() if isinstance(m,torch.nn.Linear)).weight,old_pert):raise AssertionError('Unused old condition columns changed')
        history.append({'step':step+1,'human_donor':hrec['donor'],'source_donor':None if arm=='H' else records[step%2]['donor'],
                        'human_loss':float(human_loss.detach().cpu()),'source_loss':None if source_loss is None else float(source_loss.detach().cpu()),
                        'source_coefficient':coefficient,'gradient_norm':float(norm.cpu())})
        if step+1 in (50,100,200):
            score,outs,by_donor=calibrate(model,data,fold,ix,args)
            calibration.append({'step':step+1,'macro_mse':score,'donor_mse':by_donor})
            if score<best:
                best=score;selected_step=step+1;calibration_predictions=outs;torch.save(model.state_dict(),folder/'selected.pt')
            progress(args,phase='calibration',fold=fold['fold'],arm=arm,step=step+1,mse=score,selected_step=selected_step)
    shrink=[]
    for alpha in contract['calibration']['post_calibration_shrink']['alpha_grid']:
        scores=[]
        for out in calibration_predictions:
            c=human[out['control_rows']];p=c+alpha*(out['prediction']-c)
            scores.append(mse(p,human[out['target_rows']]))
        shrink.append({'alpha':alpha,'macro_mse':float(np.mean(scores))})
    chosen=min(shrink,key=lambda v:(v['macro_mse'],v['alpha']))
    for i,out in enumerate(calibration_predictions):np.savez_compressed(folder/f'calibration_{i}.npz',**{k:v for k,v in out.items() if k!='donor'})
    result={'fold':fold['fold'],'arm':arm,'seed':seed,'selected_step':selected_step,'selected_calibration_mse':best,
            'calibration_history':calibration,'shrink_grid':shrink,'selected_shrink':chosen['alpha'],
            'source_scale':scales,'source_coefficient':coefficient,'training_history':history,
            'animal_gradient_witness':gradient_witness,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in parameters),'weight_sha256':sha256_file(folder/'selected.pt'),
            'unused_old_perturbation_columns_unchanged':True,'unused_old_batch_rows_unchanged':torch.equal(model.batch_encoder.weight[:56],before_batch),
            'target_outcomes_scored':False,'elapsed_seconds':time.monotonic()-started}
    if arm in ('H+Rabbit','H+RabbitA0'):
        later=[row for row in gradient_witness if row['step']>=3]
        if not any(any(v is not None and np.isfinite(v) and v>0 for k,v in row['norms'].items()
                       if 'v_proj' in k or 'basal_encoder' in k) for row in later):
            raise AssertionError('Actual source data produced no later connected shared gradient')
    save(folder/'training.json',result)
    del model;torch.cuda.empty_cache()
    return result


def train_predict(args):
    contract,manifest,selections,data=load_data(args)
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Use a new frozen experiment directory')
    args.out.mkdir(parents=True)
    spec={'contract_sha256':CONTRACT_SHA,'contract':contract,'prepared_manifest':manifest,
          'inputs':{str(p.resolve()):sha256_file(p) for p in [args.contract,args.prepared/'manifest.json',args.prepared/'selections.json',args.prepared/'lps_primary_arrays.npz',args.checkpoint]},
          'code':{str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__),ROOT/'src/cross_species/named_gene_state.py']},
          'official_source_root':str(args.state_source),'original_checkpoint':str(args.checkpoint),
          'torch':str(torch.__version__),'device':torch.cuda.get_device_name(),
          'fold_specific_target_outcomes_used_in_that_fold_fitting_or_selection':False,
          'target_outcomes_scored_before_global_freeze':False}
    save(args.out/'specification.json',spec)
    fits=[]
    for fold in selections['folds']:
        for arm in ARMS:
            folder=args.out/f"fold_{fold['fold']}_{arm.replace('+','_')}";folder.mkdir()
            fits.append(fit_arm(args,contract,data,fold,arm,folder))
    save(args.out/'ALL_FITS_SELECTED.json',{'fits':fits,'specification_sha256':sha256_file(args.out/'specification.json'),'target_scored':False})
    artifacts={}
    for fit in fits:
        fold=selections['folds'][fit['fold']];folder=args.out/f"fold_{fit['fold']}_{fit['arm'].replace('+','_')}"
        model,_=initialize(args,fit['seed']);model.load_state_dict(torch.load(folder/'selected.pt',map_location=args.device,weights_only=True),strict=True)
        base_delta=np.mean([data['human_expression'][r['target_rows']].mean(0,dtype=np.float64)-data['human_expression'][r['control_rows']].mean(0,dtype=np.float64) for r in fold['human_base']],axis=0)
        for i,record in enumerate(fold['human_test']):
            out=infer(model,data['human_expression'],record,data['native_gene_indices'],args.device)
            c=data['human_expression'][record['control_rows']]
            path=folder/f'target_{i}.npz'
            # Store indices but never dereference target LPS rows in this phase.
            np.savez_compressed(path,**out,calibrated_shrink=c+fit['selected_shrink']*(out['prediction']-c),
                                zero_response=c,base_mean_response=np.maximum(c+base_delta,0),half_response=c+.5*(out['prediction']-c),
                                native_gene_indices=data['native_gene_indices'])
            artifacts[str(path.relative_to(args.out))]=sha256_file(path)
        del model;torch.cuda.empty_cache()
    save(args.out/'ALL_TARGET_PREDICTIONS_FROZEN.json',{'artifacts':artifacts,'fits_selected_sha256':sha256_file(args.out/'ALL_FITS_SELECTED.json'),
         'target_outcomes_scored':False,'unix_time':time.time()})
    progress(args,phase='all_target_predictions_frozen',fits=len(fits),populations=len(artifacts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['prepared','contract','checkpoint','state-source','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train_predict(args)


if __name__=='__main__':main()
