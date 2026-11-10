#!/usr/bin/env python3
"""Editable integrated manuscript figures from verified, frozen result summaries."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter,FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/biorxiv_ready_20261108/figures'
SRC=ROOT/'docs/biorxiv_ready_20261108/figures/results'
PURPLE='#7864B5';GREEN='#26856B';ROSE='#CE536B';GOLD='#B8A23E';GRAY='#8C8F96';LIGHT='#D7D7DE'
ARM_COLORS=[PURPLE,GREEN,ROSE,'#A08FC2',GRAY]
plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':1,'lines.linewidth':1.2,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
SOURCES={}
def load(rel):
    p=ROOT/rel;SOURCES[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def canvas(height=145):
    f,ax=plt.subplots(2,2,figsize=(180/25.4,height/25.4))
    f.subplots_adjust(left=.12,right=.975,bottom=.10,top=.94,wspace=.48,hspace=.55)
    for i,a in enumerate(ax.flat):
        a.text(-.19,1.08,'abcd'[i],transform=a.transAxes,fontsize=8,fontweight='bold',ha='left',va='bottom')
        a.tick_params(width=1,length=3);a.set_axisbelow(True)
    return f,ax.ravel()
def export(f,name,data,caption):
    OUT.mkdir(parents=True,exist_ok=True);SRC.mkdir(parents=True,exist_ok=True)
    f.savefig(OUT/(name+'.pdf'));f.savefig(OUT/(name+'.svg'));f.savefig(OUT/(name+'.png'),dpi=450)
    plt.close(f)
    (SRC/(name+'_source_data.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (SRC/(name+'_legend.md')).write_text(caption+'\n')

def figure5():
    main=load('outputs/feedback_nested_chemical_20261107_absolute_v1/report.json')
    joint=load('outputs/feedback_joint_specificity_20261107_absolute/report.json')
    budget=load('outputs/feedback_label_budget_20261107_absolute/report.json')
    ident=load('outputs/feedback_joint_specificity_20261107_absolute/identification.json')
    f,axes=canvas();a,b,c,d=axes
    prior=main['metrics']['chemical_context'];source=main['metrics']['source_plus_chemical_context']
    assert prior['group_ids']==source['group_ids']
    x=np.asarray(prior['group_losses']);y=np.asarray(source['group_losses']);limits=[min(x.min(),y.min())*.8,max(x.max(),y.max())*1.2]
    a.plot(limits,limits,color=GRAY,linestyle='--',zorder=1)
    a.scatter(x,y,c=np.where(y<x,GREEN,ROSE),s=16,alpha=.85,edgecolor='white',linewidth=.35,zorder=3)
    a.set(xscale='log',yscale='log',xlim=limits,ylim=limits,xlabel='Chemical / exposure prior MSE',ylabel='Animal-informed MSE')
    a.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    a.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    a.minorticks_off()
    jd=np.asarray(joint['joint_residual_gain']['group_deltas']);order=np.argsort(jd)
    b.bar(np.arange(len(jd)),jd[order]*1000,color=np.where(jd[order]>=0,GREEN,ROSE),width=1)
    b.axhline(0,color='black',linewidth=1);b.set(xlabel='Compounds ordered by error difference',ylabel='MSE reduction with source residual\n(×0.001)',xlim=(-1,len(jd)))
    budget_data=[]
    for i,key in enumerate(['16','32','64','all']):
        v=budget['results'][key]['aggregate'];bp=np.asarray(v['seed_macro_mse']['chemical_context']);ap=np.asarray(v['seed_macro_mse']['source_plus_chemical_context']);gain=1-ap/bp
        offsets=np.linspace(-.07,.07,len(gain)) if len(gain)>1 else np.zeros(1)
        c.scatter(i+offsets,gain*100,color=GREEN,s=20,zorder=3)
        budget_data.append({'reference_budget':key,'gain_percent_per_draw':(gain*100).tolist(),'mean_gain_percent':float((1-v['mean_seed_macro_mse']['source_plus_chemical_context']/v['mean_seed_macro_mse']['chemical_context'])*100)})
    c.plot(range(4),[v['mean_gain_percent'] for v in budget_data],color=GREEN)
    c.set(xticks=range(4),xticklabels=['16','32','64','100–101'],xlabel='Paired reference compounds per fold',ylabel='MSE reduction versus prior (%)',ylim=(0,6))
    names=['chemical_context_plus_common','chemical_context_plus_common_residual'];vals=[ident['identification'][n]['common_defined_queries'] for n in names]
    d.bar([0,1],[v['top1']*100 for v in vals],color=[PURPLE,GREEN],width=.55)
    d.axhline(vals[0]['chance_top1']*100,color=GRAY,linestyle='--');d.set(xticks=[0,1],xticklabels=['Prior + amplitude','Add source residual'],ylabel='Correct response identification (%)',ylim=(0,27))
    export(f,'Fig5_information',{'a':{'compound_ids':prior['group_ids'],'prior_mse':x.tolist(),'animal_mse':y.tolist()},'b':{'compound_ids':joint['joint_residual_gain']['group_ids'],'residual_gain':jd.tolist()},'c':budget_data,'d':{'models':names,'scores':vals}},'**Figure 5. Predictive information in measured rat responses.** a, Compound-level errors on 126 compounds and 678 conditions, with all 10,358 orthologues retained. Green and rose points indicate lower and higher error after adding the rat measurement; both axes are logarithmic. b, Error reduction from the source residual beyond the specified chemical, exposure, projection-amplitude and RMS control; all 126 compounds are shown. c, Reference-coverage sensitivity. Points are three fixed nested subsample draws at budgets 16–64; the full-budget result is reused once. The line reports the relative difference of mean losses. These draws share data and are not biological replicates. d, Terminal response identification for the joint-control and residual models among other compounds in the same outer-test time/nominal-dose stratum. The dashed line is query-specific chance averaged over compounds. Candidate absolute concentrations can differ. No feature or outcome-based panel selection is used.')

def figure6():
    runs=[load('outputs/crossspecies_direct_03/'+s+'/report.json') for s in ['source_exposure_v1','paired_exposure_v1']]
    retrieval=load('outputs/biorxiv_ready_20261108/metric_checks/retrieval_report.json')['reports']
    f,axes=canvas(150);data={}
    labels=['No animal','Correct / correct','Correct / wrong','Wrong / correct','Wrong / wrong']
    arms=runs[0]['arms']
    for a,r,letter in zip(axes[:2],runs,['a','b']):
        base=r['macro_mse_per_seed']['prior'][0];records=[]
        for i,arm in enumerate(arms):
            v=np.asarray(r['macro_mse_per_seed'][arm])/base
            if i>=2:
                a.scatter(v,i+np.linspace(-.1,.1,len(v)),s=12,c=ARM_COLORS[i],alpha=.75,zorder=3)
            a.scatter(np.median(v),i,marker='D',s=25,c=ARM_COLORS[i],edgecolor='white',linewidth=.4,zorder=4)
            records.append({'arm':arm,'relative_mse_per_seed':v.tolist()})
        a.axvline(1,color='black',linestyle='--');a.set(yticks=range(5),yticklabels=labels,xlabel='MSE / no-animal MSE',ylim=(4.55,-.55),xlim=(.94,1.075))
        data[letter]={'regime':r['regime'],'conditions':r['rows'],'compounds':r['compounds'],'values':records}
    c=axes[2];records=[]
    for i,(r,v) in enumerate(zip(runs,retrieval)):
        p=v['summary']['prior'][0];s=v['summary']['correct_fit_correct_query'][0]
        c.bar(i-.16,p['top1']*100,width=.29,color=PURPLE,label='No animal' if i==0 else None)
        c.bar(i+.16,s['top1']*100,width=.29,color=GREEN,label='Correct animal' if i==0 else None)
        c.hlines(p['chance_top1']*100,i-.38,i+.38,colors=GRAY,linestyles='--',linewidth=1.2)
        records.append({'regime':r['regime'],'prior':p,'correct':s})
    c.set(xticks=[0,1],xticklabels=['Animal exposure','Paired exposure'],ylabel='Correct response identification (%)',ylim=(0,75));c.legend(frameon=False,loc='upper left',ncol=1)
    d=axes[3]
    for r,color in zip(runs,[GREEN,PURPLE]):
        vals=np.sort(r['comparisons']['animal_increment']['compound_deltas_per_seed'][0]);d.step(vals*1000,np.arange(1,len(vals)+1)/len(vals),where='post',color=color,label=f"{r['compounds']} compounds")
    d.axvline(0,color=GRAY,linestyle='--');d.set(xlabel='MSE reduction from correct animal input\n(×0.001)',ylabel='Cumulative fraction of compounds',ylim=(0,1.02));d.legend(frameon=False,loc='lower right')
    data['c']=records;data['d']=[{'regime':r['regime'],'compound_deltas':r['comparisons']['animal_increment']['compound_deltas_per_seed'][0]} for r in runs]
    export(f,'Fig6_correspondence',data,'**Figure 6. Training and query correspondence determine predictive increment.** a,b, Five information arms evaluated on identical queries within each exposure-matching regime. a matches animal time and actual molar concentration (467 conditions, 107 compounds); b additionally matches the paired record’s human concentration (208 conditions, 49 compounds). Labels give fit/query correspondence. Each deterministic arm is shown once as a diamond. For the three mismatch arms, dots show 20 prespecified mismatches and diamonds give their medians. Dashed vertical lines mark the no-animal baseline; lower ratios are better. c, Post-hoc terminal identification among candidates within the same outer-test fold and exact exposure stratum. Dashed segments give compound-averaged query-specific chance. d, All compound-level error differences. The two cohorts overlap; seed dispersion and candidate pairs do not add independent biological tasks. Wrong-fit models received the same human-label budget and independently selected regularization within their mismatched inner folds.')

def figure7():
    r=load('outputs/external_transfer_04/report.json');ref=load('outputs/biorxiv_ready_20261108/metric_checks/measurement_reference_report.json')
    post=load('docs/multisource_20261108/external_transfer_04/posthoc_scalar_correspondence/report_v1.json')
    f,axes=canvas(155);a,b,c,d=axes;tasks=r['external_tasks'];names=['CDDO-Me','KI696','Thapsigargin'];rules=['zero','human_mean','chemical_context','source_plus_context','scalar_nointercept']
    colors=[GRAY,'#555860',PURPLE,GREEN,ROSE];lines=['--',':','-','-','-']
    ratio=[]
    for name,col,ls in zip(rules,colors,lines):
        vals=[x['rules'][name]['mse']/x['rules']['zero']['mse'] for x in tasks]
        a.plot(range(3),vals,color=col,linestyle=ls,marker='o',markersize=3,label={'zero':'Zero','human_mean':'Human history','chemical_context':'Chemical prior','source_plus_context':'Fusion','scalar_nointercept':'Scalar'}[name]);ratio.append({'rule':name,'relative_mse':vals})
    a.set(xticks=range(3),xticklabels=names,ylabel='MSE / zero-response MSE',ylim=(.72,1.94));a.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.02),columnspacing=.7,handlelength=1.5)
    beta=post.get('beta',post.get('fixed_beta'))
    # Read saved report structure rather than infer a correspondence matrix.
    matrix=np.asarray(post['mse_query_by_donor'],float)
    assert matrix.shape==(3,3) and post['task_order']==['CDDO','KI696','THAP']
    norm=matrix/np.diag(matrix)[:,None]
    cmap=LinearSegmentedColormap.from_list('muted_purple_rose',['#EEEAF5','#BBA7D3',ROSE])
    im=b.imshow(norm,cmap=cmap,vmin=.98,vmax=1.12,aspect='auto');b.set(xticks=range(3),yticks=range(3),xticklabels=names,yticklabels=names,xlabel='Animal response source',ylabel='Human query')
    for i in range(3):
        for j in range(3):b.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='black',fontsize=7)
    cb=f.colorbar(im,ax=b,fraction=.05,pad=.25,orientation='horizontal',ticks=[.98,1.05,1.12]);cb.set_label('MSE / matched-source MSE',fontsize=7);cb.ax.tick_params(labelsize=7)
    refs=[]
    for i,t in enumerate(ref['tasks']):
        vals=[t['rules'][k]['mean_mse'] for k in ['zero','same_condition_human_reference','source_plus_context','scalar_nointercept']]
        base=vals[0];values=np.asarray(vals)/base
        for j,(v,color) in enumerate(zip(values,[GRAY,GOLD,GREEN,ROSE])):c.bar(i+(j-1.5)*.17,v,width=.16,color=color,label=['Zero','Human record reference','Fusion','Scalar'][j] if i==0 else None)
        refs.append({'task':t['task_id'],'relative_held_record_mse':values.tolist(),'raw_mse':vals})
    c.set(xticks=range(3),xticklabels=names,ylabel='Held-record MSE / zero MSE',ylim=(0,3.0));c.legend(frameon=False,ncol=1,loc='upper right',fontsize=7)
    support=r['external_support'];ds=[]
    for i,x in enumerate(support):
        dr=x['context'][1];d.scatter(dr,i,c=[PURPLE,GREEN,ROSE][i],s=32,zorder=4);ds.append({'task':x['task_id'],'log10_rat_M':dr})
    d.axvspan(-6,-2,color=LIGHT,alpha=.5);d.set(yticks=range(3),yticklabels=names,xlabel='Planned animal concentration (log10 M)',xlim=(-7.4,-1.8),ylim=(2.6,-.6));d.axvline(-6,color=GRAY,linestyle='--')
    export(f,'Fig7_transport',{'a':ratio,'b':{'mse_matrix':matrix.tolist(),'row_normalized_mse':norm.tolist(),'beta':beta},'c':refs,'d':ds},'**Figure 7. Frozen rules have heterogeneous effects in another hepatocyte study.** a, Errors relative to the zero-response baseline on three new chemicals in GSE330151; all models were fitted using TG-GATEs only on 7,307 shared genes and three continuous exposure features. Lines connect task values and do not imply an ordered exposure trajectory. b, Post-hoc fixed-scalar query-by-source diagnostic, normalized to each query’s matched-source error. No coefficient was refitted. Off-diagonal inputs also change animal dose and sometimes time. c, Human-record sensitivity: each of nine choices of one treated and one control record is scored against a reference from the remaining two treated and two control records, and against frozen model predictions. Held and reference records are disjoint within each choice; the nine choices overlap. Donor identities remain unknown, and this reference does not define a biological noise ceiling. d, Query animal concentrations and the TG training range (shaded). CDDO-Me is below that range. Three drugs, two pathway families and prior public-data use limit external generalization claims.')

def figure8():
    fib=load('outputs/multisource_20261108/benchmarks/GSE105160_v1/report.json')
    hep=load('outputs/multisource_20261108/benchmarks/GSE330151_v1/report.json')
    comp=load('docs/multisource_20261108/compartment_benchmark_02/report_interpretation_corrected_v1.json')
    imm=load('docs/multisource_20261108/e5919_recovery_02/POSTHOC_SCALAR_report_v1.json')
    f,axes=canvas(150);a,b,c,d=axes;rng=np.random.default_rng(818)
    subsets=fib['all_subset_records'];curve=[]
    for k in range(1,10):
        vals=np.asarray([x['mean_target_mse'] for x in subsets if x['source_count']==k]);a.scatter(k+rng.uniform(-.14,.14,len(vals)),vals,s=7,alpha=.25,color=PURPLE,edgecolor='none');curve.append([k,float(vals.mean()),float(vals.min())])
    curve=np.asarray(curve);a.plot(curve[:,0],curve[:,1],color=PURPLE,marker='o',markersize=3,label='Mean subset');a.plot(curve[:,0],curve[:,2],color=GREEN,marker='s',markersize=3,linestyle='--',label='Post-outcome minimum');a.set(xlabel='Animal sources in subset',ylabel='Human mean-response MSE',xticks=[1,3,5,7,9]);a.legend(frameon=False,fontsize=7)
    budget=hep['fixed_total_profile_budget'];full=[];fixed=[]
    for key,v in hep['scores'].items():
        if key.startswith('equal/'):
            full.append([len(key.split('/')[1].split('+')),v['macro_mse']])
    for row in budget:
        if row['eligible']:fixed.append([len(row['subset']),row['macro_expected_mse']])
    for values,color,marker,label in [(full,PURPLE,'o','All measured records'),(fixed,ROSE,'s','Fixed 3 + 3 records')]:
        ar=np.asarray(values);means=[]
        for k in sorted(set(ar[:,0])):
            v=ar[ar[:,0]==k,1];b.scatter(k+rng.uniform(-.08,.08,len(v)),v,color=color,marker=marker,s=15,alpha=.5);means.append([k,v.mean()])
        means=np.asarray(means);b.plot(means[:,0],means[:,1],color=color,marker=marker,markersize=3,label=label)
    b.axhline(hep['fixed_cyno_control']['macro_mse'],color=GREEN,linestyle='--');b.set(xlabel='Animal sources in subset',ylabel='Three-drug macro MSE',xticks=[1,2,3,4]);b.legend(frameon=False,fontsize=7,loc='upper right')
    cdata=[]
    for i,context in enumerate(['HepG2','K562']):
        m=next(x['macro_mse'] for x in comp['macro_results'] if x['group']=='primary|'+context)
        for j,(key,color) in enumerate([('human_history_mean',GRAY),('whole_raw',PURPLE),('cyto_raw',ROSE),('equal_scaled',GREEN)]):
            c.bar(i+(j-1.5)*.18,m[key]/m['zero'],width=.17,color=color,label=['Human history','Whole-cell','Cytoplasmic','Calibrated mean'][j] if i==0 else None)
        cdata.append({'context':context,'macro_mse':m})
    c.axhline(1,color='black',linestyle='--');c.set(xticks=[0,1],xticklabels=['HepG2','K562'],ylabel='MSE / zero-response MSE',ylim=(0,1.8));c.legend(frameon=False,ncol=2,fontsize=7,loc='upper left',columnspacing=.5,handlelength=1)
    idata=[]
    for i,x in enumerate(imm['results']):
        base=x['rules']['zero_response']['mse'];vals=[x['rules'][k]['mse']/base for k in ['source_raw','source_scaled','other_stimulus_human_mean']]
        for j,(v,col) in enumerate(zip(vals,[PURPLE,GREEN,GRAY])):d.bar(i+(j-1)*.23,v,width=.21,color=col,label=['Raw animal','Historical scalar','Human history'][j] if i==0 else None)
        idata.append({'task':x['held_stimulus'],'relative_mse':vals})
    d.axhline(1,color='black',linestyle='--');d.set(xticks=[0,1],xticklabels=['IFNβ','poly(I:C)'],ylabel='MSE / zero-response MSE',ylim=(0,1.8));d.legend(frameon=False,ncol=1,fontsize=7,loc='upper right')
    export(f,'Fig8_scope',{'a':{'all_subsets':subsets,'summary':curve.tolist()},'b':{'full_records':full,'fixed_budget':fixed,'fixed_cyno_mse':hep['fixed_cyno_control']['macro_mse']},'c':cdata,'d':idata},'**Figure 8. Source composition and calibration have limited domains of usefulness.** a, All 511 equal-weight subsets of nine animal sources for one fibroblast stimulus in GSE105160. Jitter separates overlapping points and has no scientific meaning. The minimum is selected after observing the human endpoint. b, GSE330151 source subsets under full measured-arm availability or a fixed total of 3 treated and 3 vehicle records. Points are all feasible subsets; lines average them by size. The dashed reference is the fixed cynomolgus source. Four sources are infeasible under the fixed budget. c, SRSF comparisons in two human contexts with three principal genetic targets; the two P19 measurements are compartments of one cell model. d, Post-hoc E-MTAB-5919 author-version sensitivity using a scalar fitted on the other of two related stimuli. IFNβ uses an 8 h treatment and 0 h baseline, retaining a culture-time difference; poly(I:C) uses a 4 h matched mock. Black dashed lines in c,d indicate zero response. Different panels use different gene axes and experimental units, so their absolute errors are not pooled.')

if __name__=='__main__':
    for fn in [figure5,figure6,figure7,figure8]:fn()
    (SRC/'source_manifest.json').write_text(json.dumps({'source_files':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'design':'180mm,blackArial,editableSVG/PDF,450dpiPNG;no_panel_titles'},indent=2)+'\n')
