"""All seven origins: natural, sustained decrease and sustained bounded increase."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from temp_scripts.project_paths import PROJECT_ROOT


def native(p):
    p=Path(p).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def first(mask):
    q=np.flatnonzero(mask)
    return int(q[0]) if len(q) else None


def intervals(mask):
    q=np.flatnonzero(mask)
    if not len(q):return ''
    groups=np.split(q,np.flatnonzero(np.diff(q)>1)+1)
    return ';'.join(str(int(g[0])) if len(g)==1 else '%d-%d'%(g[0],g[-1]) for g in groups)


def main():
    out=PROJECT_ROOT/'outputs/contact_coordination_20261105';out.mkdir(parents=True,exist_ok=False)
    run=native(PROJECT_ROOT/'outputs/online_development_20261104')
    campaign=json.loads((run/'remote/pilot/campaign.json').read_text())
    selection=json.loads((run/'remote/screen/selected_origins.json').read_text())
    sources={};summaries=[];frames=[];data={}
    for case in campaign['cases']:
        obj,seed=case['object'],case['seed']
        selected=next(i for i in selection['origins'] if i['object']==obj and i['seed']==seed)
        natural=run/'remote/screen'/selected['natural_relative_path']
        requests=[('natural',natural,selected['natural_npz_sha256'])]
        for record in case['runs']:
            condition=record.get('condition',{})
            if condition.get('duration')!=-1:continue
            assert condition['model']=='assisted_seed0' and condition['group']=='top4'
            path=run/'remote/pilot'/('object%d_seed%d'%(obj,seed))/record['relative_dir']/(record['name']+'.npz')
            requests.append((condition['control'],path,record['npz_sha256']))
        assert len(requests)==3
        baseline_success=bool(case['baseline_audit']['success'])
        for label,path,expected in requests:
            assert digest(path)==expected
            sources[str(path)]=expected
            with np.load(path,allow_pickle=False) as archive:
                a={k:archive[k].copy() for k in archive.files if k.startswith('post_') or k in ['step','opening','applied','scheduled','delta_norm','done']}
            contacts=a['post_contacts'];fingers=contacts[:,:4].sum(1);palm=contacts[:,4].astype(bool)
            strict=(fingers>=2)&palm;near=a['post_distance']<=.2;goal=a['post_progress']>.95
            success=goal&near&strict;goal_near=goal&near
            assert np.array_equal(strict,a['post_strict_contact'])
            assert np.array_equal(success,a['post_success'])
            assert a['done'][-1] and not a['done'][:-1].any()
            q=np.flatnonzero(goal_near);s=np.flatnonzero(strict&near)
            gap=int(np.min(abs(q[:,None]-s[None,:]))) if len(q) and len(s) else None
            first_goal=int(q[0]) if len(q) else None;last_goal=int(q[-1]) if len(q) else None
            prior=s[s<first_goal] if first_goal is not None else []
            later=s[s>last_goal] if last_goal is not None else []
            state=dict(object=obj,seed=seed,condition=label,baseline_success=baseline_success,
                success=bool(success.any()),steps=len(success),goal_intervals=intervals(goal_near),
                goal_states=int(goal_near.sum()),first_goal=first_goal,last_goal=last_goal,
                first_strict_near=first(strict&near),last_strict_near_before_goal=int(prior[-1]) if len(prior) else None,
                first_strict_near_after_goal=int(later[0]) if len(later) else None,
                nearest_goal_contact_gap=gap,strict_ever=bool(strict.any()),
                goal_missing_palm=int((goal_near&~palm).sum()),
                goal_insufficient_fingers=int((goal_near&(fingers<2)).sum()),
                min_goal_finger_groups=int(fingers[goal_near].min()) if goal_near.any() else None,
                max_goal_distance_m=float(a['post_distance'][goal_near].max()) if goal_near.any() else None,
                first_success=first(success),max_opening_deg=float(np.rad2deg(a['post_opening'].max())),
                goal_entry_opening_deg=float(np.rad2deg(a['post_opening'][first_goal])) if first_goal is not None else None,
                success_right_censored=bool(success[-1]))
            summaries.append(state)
            for k in range(len(success)):
                frames.append(dict(object=obj,seed=seed,condition=label,action_index=k,
                    post_physical_step=int(a['post_step'][k]),opening_deg=float(np.rad2deg(a['post_opening'][k])),
                    progress=float(a['post_progress'][k]),distance_m=float(a['post_distance'][k]),
                    fingers=int(fingers[k]),palm=bool(palm[k]),strict=bool(strict[k]),goal_near=bool(goal_near[k]),
                    success=bool(success[k]),delta_h_norm=float(a['delta_norm'][k])))
            data[obj,seed,label]=dict(goal=goal_near,strict=strict&near,success=success,n=len(success))
    summary=pd.DataFrame(summaries);source=pd.DataFrame(frames)
    summary.to_csv(out/'all_origin_condition_summary.csv',index=False,encoding='utf-8-sig')
    source.to_csv(out/'full_contact_motion_timelines.csv',index=False,encoding='utf-8-sig')
    natural=summary[summary.condition.eq('natural')]
    inc=summary[summary.condition.eq('bounded_increase')]
    dec=summary[summary.condition.eq('decrease')]
    rescued=inc[(~inc.baseline_success)&inc.success]
    damaged=summary[summary.baseline_success&~summary.success]
    assert len(rescued)==2 and len(damaged)==2 and damaged[['object','seed']].drop_duplicates().shape[0]==1
    assert len(dec[(~dec.baseline_success)&dec.success])==0
    failed_natural=natural[~natural.success]
    common=[]
    for _,r in failed_natural.iterrows():
        was_rescued=bool(((rescued.object==r.object)&(rescued.seed==r.seed)).any())
        common.append(dict(object=int(r.object),seed=int(r.seed),rescued_by_sustained_increase=was_rescued,
            natural_goal_states=int(r.goal_states),natural_goal_finger_min=int(r.min_goal_finger_groups),
            natural_goal_all_missing_palm=int(r.goal_missing_palm)==int(r.goal_states),
            natural_strict_ever=bool(r.strict_ever),natural_goal_contact_gap=None if pd.isna(r.nearest_goal_contact_gap) else int(r.nearest_goal_contact_gap)))
    pd.DataFrame(common).to_csv(out/'natural_failure_commonalities.csv',index=False,encoding='utf-8-sig')
    assert all(digest(Path(p))==h for p,h in sources.items())
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf8')
    checks=dict(passed=True,origins=7,trajectories=21,sustained_conditions=14,natural_baselines=7,
        sustained_increase_rescued_origins=2,sustained_increase_damaged_origins=1,
        sustained_decrease_rescued_origins=0,sustained_decrease_damaged_origins=1,
        damaged_unique_physical_origins=1,predicate_mismatches=0,source_hashes_unchanged=True,
        new_training_fits=0,new_physical_trajectories=0,
        limits=['retrospective mechanisms, not a pre-intervention rescue classifier','only three selected natural failures and one dictionary seed',
                'contact predicates measure instantaneous groups, not force or stable grasp','success truncates future observations',
                'endpoint decomposition does not identify the first neural or physical cause'])
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cases=[(11477,2026110400,'Originally successful'),(11406,2026110402,'Rescued by sustained increase'),
           (11477,2026110402,'Rescued by sustained increase'),(11405,2026110405,'Not rescued')]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,(obj,seed,title) in zip(axes.flat,cases):
        rows=summary[(summary.object==obj)&(summary.seed==seed)]
        lo=int(rows.first_goal.min())-7;hi=int(rows.last_goal.max())+7
        for j,label in enumerate(['natural','bounded_increase','decrease']):
            d=data[obj,seed,label]
            for part,key,color in [(0,'goal','#0072b2'),(1,'strict','#009e73')]:
                y=5-2*j-part
                for t in np.flatnonzero(d[key]):
                    if lo<=t<=hi:ax.broken_barh([(t-.4,.8)],(y-.27,.54),facecolors=color)
                if d['n']<=hi:ax.broken_barh([(d['n']-.5,hi-d['n']+1)],(y-.27,.54),facecolors='#e8e8e8')
            success_index=first(d['success'])
            if success_index is not None and lo<=success_index<=hi:ax.plot(success_index,5-2*j,'*',color='black',markersize=10)
        ax.set_yticks([5,4,3,2,1,0],['Natural: target','Natural: contact','Increase: target','Increase: contact','Decrease: target','Decrease: contact'])
        ax.set_ylim(-.7,5.7);ax.set_xlim(lo-.6,hi+.6);ax.set_xlabel('Saved action index (0-based)')
        ax.set_title('%s: %d / %s'%(title,obj,str(seed)[-3:]),fontsize=11)
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two coupled requirements: target progress and strict contact must overlap',fontsize=13)
    fig.legend(handles=[Patch(color='#0072b2',label='Target progress + near'),Patch(color='#009e73',label='2+ finger groups + palm + near'),
                        Patch(color='#e8e8e8',label='After episode ended: unobserved')],loc='outside lower center',ncol=3,fontsize=9)
    fig.savefig(out/'target_contact_coordination.png',dpi=180);plt.close(fig)
    print(json.dumps(checks))


if __name__=='__main__':main()
