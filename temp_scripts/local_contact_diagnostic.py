"""Supplemental diagnosis of the gap between any-contact labels and task success."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss
from threadpoolctl import threadpool_limits
from temp_scripts.data_access import DataRepository
from temp_scripts.project_paths import default_index
from temp_scripts.physical_runtime import NumericSAE
from temp_scripts.local_interpretability import PROJECT_ROOT,PHYSICAL,STUDY,infer,fit_readout,sample_weights,dump


def run(root):
    root=Path(root);torch.set_num_threads(1)
    f=pd.read_csv(root/'development_frames.csv');strict=np.zeros(len(f),dtype=bool)
    repo=DataRepository(default_index())
    for tid,g in f.groupby('trajectory',sort=False):
        traj=repo.load(tid)
        if traj.meta['split'] not in ('train','validation'):raise ValueError('Non-development input')
        strict[g.index.to_numpy()]=traj.state('contact')[g.state_index.to_numpy()]
    masks=[('local',f.local_window.to_numpy()),('post_onset',f.post_onset.to_numpy())]
    counts=[]
    for scope,mask in masks:
        for split in ['train','validation']:
            q=mask&f.split.eq(split).to_numpy()
            counts.append(dict(scope=scope,split=split,states=int(q.sum()),strict_positive_states=int(strict[q].sum()),
                trajectories=f.trajectory[q].nunique(),positive_trajectories=f.trajectory[q&strict].nunique()))
    pd.DataFrame(counts).to_csv(root/'strict_contact_label_coverage.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dict(trajectory=f.trajectory,state_index=f.state_index,strict_contact=strict)).to_csv(root/'strict_contact_labels.csv',index=False,encoding='utf-8-sig')
    with np.load(root/'development_arrays.npz') as a:h=a['h']
    with np.load(STUDY/'controls/ridge_rotation_10p0_coordinate/projection.npz') as a:
        ridge=((h-a['h_mean'])/a['h_scale'])@a['rotation']
    reps={'raw':h,'ridge':ridge}
    for folder in (PHYSICAL/'bundle/models').iterdir():reps[folder.name]=infer(NumericSAE(folder),h)[0]
    ranks=json.loads((root/'frozen_rankings.json').read_text())
    mask=f.post_onset.to_numpy();train=mask&f.split.eq('train').to_numpy();val=mask&f.split.eq('validation').to_numpy()
    groups=f.trajectory.to_numpy();vw=sample_weights(groups[val]);rows=[];arts={};preds={}
    with threadpool_limits(limits=1):
        for method,x in reps.items():
            for representation,coords in [('opening_top4',ranks[method]['opening_rad'][:4]),('full',list(range(64)))]:
                art,p,value,notices=fit_readout(x[:,coords],strict.astype(float),train,val,groups,True)
                name=method+'__'+representation;arts[name]={**art,'coordinates':coords};preds[name]=p
                rows.append(dict(method=method,representation=representation,average_precision=float(average_precision_score(strict[val],p[val],sample_weight=vw)),
                    log_loss=float(log_loss(strict[val],np.column_stack([1-p[val],p[val]]),labels=[0,1],sample_weight=vw)),
                    regularization=value,warning_count=len(notices),validation_states=int(val.sum()),positive_states=int(strict[val].sum())))
    prevalence=float(np.average(strict[val],weights=vw))
    pd.DataFrame(rows).to_csv(root/'strict_contact_probe_metrics.csv',index=False,encoding='utf-8-sig')
    joblib.dump(arts,root/'strict_contact_probes.joblib');np.savez_compressed(root/'strict_contact_predictions.npz',**preds)
    dump(root/'strict_contact_diagnostic.json',dict(status='completed',purpose='any-contact supervision is not the strict contact predicate used by task success',
        frozen_sae=True,new_target='new post-onset diagnostic logistic probes; no SAE supervision update',
        coordinates='opening Top4 frozen on original local train, not selected on strict labels',
        train_fit_only=True,validation_selected=True,test_numeric_loaded=0,validation_trajectory_weighted_prevalence=prevalence))
    # Exact contact group patterns when the highlighted branches reach target progress.
    rows=[]
    for name in ['natural','assisted_seed1_top1_d5_nominal_target','assisted_seed1_top2_d5_nominal_target','assisted_seed1_top3_d5_nominal_target','assisted_seed1_top4_d5_nominal_target']:
        path=PHYSICAL/'remote/campaign/object11477_seed2026110301'/name/(name+'.npz')
        with np.load(path) as a:
            goal=np.flatnonzero(a['post_progress']>.95)
            for t in goal:
                rows.append(dict(name=name,step=int(t),progress=float(a['post_progress'][t]),opening_degrees=float(np.rad2deg(a['post_opening'][t])),
                    distance=float(a['post_distance'][t]),strict_contact=bool(a['post_strict_contact'][t]),success=bool(a['post_success'][t]),
                    finger_groups=int(a['post_contacts'][t,:-1].sum()),palm=bool(a['post_contacts'][t,-1])))
    pd.DataFrame(rows).to_csv(root/'example_goal_contact_patterns.csv',index=False,encoding='utf-8-sig')
    print('STRICT CONTACT DIAGNOSTIC COMPLETE',rows,flush=True)
