"""Development-only angle-matched hand-distance snapshots; no causal labels."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from temp_scripts.data_access import DataRepository
from temp_scripts.physical_runtime import NumericSAE
from temp_scripts.project_paths import PROJECT_ROOT, default_index


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def qrot(q):
    return Rotation.from_quat(np.asarray(q)[:,[1,2,3,0]])


def actor(h,w):
    pre=np.asarray(h,float)@w['mlp_extractor.policy_net.2.weight'].T+w['mlp_extractor.policy_net.2.bias']
    h2=np.maximum(pre,0)
    return pre,h2,h2@w['action_net.weight'].T+w['action_net.bias']


def tail_path(h0,h1,w):
    """Exact integrated ReLU slope on the straight hidden-state path."""
    p0,_,m0=actor(h0,w);p1,_,m1=actor(h1,w)
    gate=np.zeros(64)
    gate[(p0>0)&(p1>0)]=1
    up=(p0<=0)&(p1>0);down=(p0>0)&(p1<=0)
    gate[up]=p1[up]/(p1[up]-p0[up])
    gate[down]=p0[down]/(p0[down]-p1[down])
    jac=(w['action_net.weight']*gate[None,:])@w['mlp_extractor.policy_net.2.weight']
    np.testing.assert_allclose(jac@(h1-h0),m1-m0,atol=1e-10,rtol=1e-10)
    return jac,m1-m0


def match_pair(a,b,tier):
    # Calipers frozen before reading activations or outcomes.
    angle=abs(a['angle'][:,None]-b['angle'][None,:])
    speed=abs(a['hinge_v'][:,None]-b['hinge_v'][None,:])
    time=abs(a['step'][:,None]-b['step'][None,:])
    orient=2*np.arccos(np.clip(abs(a['relative_q']@b['relative_q'].T),0,1))
    distance=abs(a['distance'][:,None]-b['distance'][None,:])
    basic=(angle<=.01)&(speed<=.15)&(time<=10)&(orient<=np.deg2rad(15))&(distance>=.03)&(distance<=.15)
    score=(angle/.01)**2+(speed/.15)**2+(time/10)**2+(orient/np.deg2rad(15))**2
    if tier=='strict':
        contacts=np.all(a['contacts'][:,None,:]==b['contacts'][None,:,:],axis=2)
        finger=np.sqrt(np.mean((a['finger'][:,None,:]-b['finger'][None,:,:])**2,axis=2))
        velocity=np.linalg.norm(a['palm_v'][:,None,:]-b['palm_v'][None,:,:],axis=2)
        scene_pos=np.linalg.norm(a['object_pos'][:,None,:]-b['object_pos'][None,:,:],axis=2)
        scene_q=2*np.arccos(np.clip(abs(a['object_q']@b['object_q'].T),0,1))
        basic &= contacts&(finger<=.25)&(velocity<=.25)&(scene_pos<=.025)&(scene_q<=np.deg2rad(10))
        score += (finger/.25)**2+(velocity/.25)**2+(scene_pos/.025)**2+(scene_q/np.deg2rad(10))**2
    ia,ib=np.nonzero(basic)
    if not len(ia):return None
    chosen=np.argmin(score[ia,ib]);i,j=int(ia[chosen]),int(ib[chosen])
    return dict(i=i,j=j,score=float(score[i,j]),angle_gap=float(angle[i,j]),speed_gap=float(speed[i,j]),
                step_gap=int(time[i,j]),orientation_gap_degrees=float(np.rad2deg(orient[i,j])),
                distance_gap=float(distance[i,j]),candidate_frame_pairs=len(ia))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=PROJECT_ROOT/'outputs/snapshot_matching_20261105')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    repo=DataRepository(default_index());records=repo.filter(('train','validation'))
    sources={};trajectories={};coverage=[]
    for m in records:
        t=repo.load(m['trajectory_id']);a=t.arrays;n=t.n
        path=repo.root/m['numeric_archive'];sources[str(path)]=sha(path)
        keep=np.flatnonzero((t.opening[:n]>=.05)&(t.opening[:n]<=1.6))
        coverage.append(dict(trajectory=t.tid,split=m['split'],object=int(m['object_id']),all_states=n,eligible_states=len(keep)))
        if not len(keep):continue
        obj=qrot(a['object_quaternion_wxyz'][keep]); palm=qrot(a['palm_quaternion_wxyz'][keep])
        rel=obj.inv().apply(a['palm_position_world'][keep]-a['handle_position_world'][keep])
        finger=a['robot_qpos'][keep,6:]
        trajectories[t.tid]=dict(tid=t.tid,split=m['split'],object=int(m['object_id']),step=keep,
            angle=a['object_qpos'][keep,0].astype(float),opening=t.opening[keep],hinge_v=a['object_qvel'][keep,0].astype(float),
            distance=a['palm_handle_distance'][keep].astype(float),relative_q=(obj.inv()*palm).as_quat(),relative_palm=rel,
            palm_world=a['palm_position_world'][keep],handle_world=a['handle_position_world'][keep],
            palm_v=obj.inv().apply(a['palm_velocity_world'][keep]),contacts=a['finger_palm_contact'][keep],finger=finger,
            object_pos=a['object_position_world'][keep],object_q=a['object_quaternion_wxyz'][keep],
            h=a['activations'][keep].astype(float),h2=a['activations_l2'][keep].astype(float),mu=a['action_mean_raw'][keep].astype(float))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False,encoding='utf-8-sig')
    protocol=dict(scope='original train/validation natural episodes only; no test trajectories or edited branches',
        phases='pre-action opening from reset between .05 and 1.6 rad',
        basic=dict(same_object=True,different_trajectories=True,actual_hinge_angle_gap_rad=.01,object_velocity_gap_rad_s=.15,
                   step_gap=10,object_frame_palm_orientation_gap_deg=15,palm_handle_distance_gap_m=[.03,.15]),
        strict_additions=dict(same_contact_pattern=True,finger_qpos_rms_rad=.25,object_frame_palm_velocity_gap_m_s=.25,
                              object_root_position_gap_m=.025,object_root_rotation_gap_deg=10),
        selection='minimum physical-control mismatch per unordered trajectory pair, then greedy disjoint trajectories per object/split; no outcome or activation selection',
        candidate_selection='train pairs only, rank standardized mean far-minus-near activation difference; validation is reused development corroboration',
        limits=['observational matching, not experimental control of hand position','other state, visual noise and history can differ',
                'strict tier may be sparse; no threshold expansion to manufacture pairs','per-neuron activation differences and exact actor path decompositions are not physical causal mediation'])
    dump(out/'protocol.json',protocol)
    matches=[];candidate_counts=[]
    for tier in ['basic','strict']:
        for split in ['train','validation']:
            for obj in sorted({a['object'] for a in trajectories.values()}):
                ts=[a for a in trajectories.values() if a['split']==split and a['object']==obj]
                candidates=[]
                for a,b in itertools.combinations(ts,2):
                    result=match_pair(a,b,tier)
                    if result is not None:candidates.append(dict(a=a['tid'],b=b['tid'],**result))
                used=set();chosen=[]
                for c in sorted(candidates,key=lambda x:(x['score'],x['a'],x['b'],x['i'],x['j'])):
                    if c['a'] in used or c['b'] in used:continue
                    chosen.append(c);used.update([c['a'],c['b']])
                candidate_counts.append(dict(tier=tier,split=split,object=obj,eligible_trajectories=len(ts),candidate_trajectory_pairs=len(candidates),selected_pairs=len(chosen)))
                for c in chosen:
                    a,b=trajectories[c['a']],trajectories[c['b']];i,j=c['i'],c['j']
                    if a['distance'][i]<=b['distance'][j]:near,far,ni,fi=a,b,i,j
                    else:near,far,ni,fi=b,a,j,i
                    matches.append(dict(pair_id=len(matches),tier=tier,split=split,object=obj,near_tid=near['tid'],far_tid=far['tid'],
                        near_local=ni,far_local=fi,near_step=int(near['step'][ni]),far_step=int(far['step'][fi]),
                        near_angle=float(near['angle'][ni]),far_angle=float(far['angle'][fi]),
                        near_distance=float(near['distance'][ni]),far_distance=float(far['distance'][fi]),
                        relative_position_gap=float(np.linalg.norm(near['relative_palm'][ni]-far['relative_palm'][fi])),
                        contact_pattern_equal=bool(np.array_equal(near['contacts'][ni],far['contacts'][fi])),
                        **{k:v for k,v in c.items() if k not in ['a','b','i','j']}))
    pairs=pd.DataFrame(matches);pairs.to_csv(out/'matched_pairs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(candidate_counts).to_csv(out/'matching_coverage.csv',index=False,encoding='utf-8-sig')
    if pairs.empty:
        dump(out/'verification.json',dict(passed=True,no_common_support=True,source_hashes=sources));return
    bundle=PROJECT_ROOT/'outputs/online_development_20261104/package/bundle'
    with np.load(bundle/'actor.npz') as w:weights={k:w[k].astype(float) for k in w.files}
    sources[str(bundle/'actor.npz')]=sha(bundle/'actor.npz')
    # Infer only selected snapshots and train natural states needed for scale.
    train_h=np.concatenate([a['h'] for a in trajectories.values() if a['split']=='train'])
    near_h=np.asarray([trajectories[r.near_tid]['h'][int(r.near_local)] for _,r in pairs.iterrows()])
    far_h=np.asarray([trajectories[r.far_tid]['h'][int(r.far_local)] for _,r in pairs.iterrows()])
    reps={'raw':(near_h,far_h,np.std(train_h,axis=0))};decoders={}
    for folder in sorted((bundle/'models').iterdir()):
        model=NumericSAE(folder)
        ztrain=np.concatenate([model.infer(train_h[k:k+256])[0] for k in range(0,len(train_h),256)])
        zn=np.asarray([model.infer(h)[0][0] for h in near_h]);zf=np.asarray([model.infer(h)[0][0] for h in far_h])
        reps[folder.name]=(zn,zf,np.std(ztrain,axis=0))
        decoders[folder.name]=model.w['decoder_weight']*model.w['scale'][:,None]
        for name in ['weights.npz','model.json']:sources[str(folder/name)]=sha(folder/name)
    effects=[];shortlist=[];tail_rows=[];max_tail_error=0.;max_actor_error=0.
    for k,r in pairs.iterrows():
        hn,hf=near_h[k],far_h[k];jac,delta_mu=tail_path(hn,hf,weights)
        saved_n=trajectories[r.near_tid]['mu'][int(r.near_local)];saved_f=trajectories[r.far_tid]['mu'][int(r.far_local)]
        max_actor_error=max(max_actor_error,float(np.max(abs(actor(hn,weights)[2]-saved_n))),float(np.max(abs(actor(hf,weights)[2]-saved_f))))
        raw_contributions=jac*(hf-hn)[None,:]
        max_tail_error=max(max_tail_error,float(np.max(abs(raw_contributions.sum(1)-delta_mu))))
        for name,(n,f,scale) in reps.items():
            delta=f[k]-n[k];standard=np.divide(delta,scale,out=np.zeros(64),where=scale>1e-8)
            for c in range(64):effects.append(dict(pair_id=int(r.pair_id),tier=r.tier,split=r.split,object=int(r.object),method=name,coordinate=c,
                near_value=float(n[k,c]),far_value=float(f[k,c]),delta=float(delta[c]),standardized_delta=float(standard[c])))
            if name=='raw':
                contributions=raw_contributions;residual=np.zeros(22)
            else:
                d=decoders[name];contributions=(jac@d)*delta[None,:]
                residual=jac@((hf-hn)-d@delta)
            reconstruction=contributions.sum(1)+residual
            max_tail_error=max(max_tail_error,float(np.max(abs(reconstruction-delta_mu))))
            for c in range(64):tail_rows.append(dict(pair_id=int(r.pair_id),tier=r.tier,split=r.split,method=name,coordinate=c,
                path_action_vector_l2=float(np.linalg.norm(contributions[:,c])),actual_mu_difference_l2=float(np.linalg.norm(delta_mu)),
                residual_path_vector_l2=float(np.linalg.norm(residual))))
    table=pd.DataFrame(effects);table.to_csv(out/'all_coordinate_pair_differences.csv',index=False,encoding='utf-8-sig')
    tail=pd.DataFrame(tail_rows);tail.to_csv(out/'actor_tail_path_contributions.csv',index=False,encoding='utf-8-sig')
    for (tier,method),group in table[table.split.eq('train')].groupby(['tier','method']):
        # Equal objects for selection; original neuron/feature identities remain distinct.
        rank=group.groupby(['object','coordinate']).standardized_delta.mean().groupby('coordinate').mean()
        selected=rank.abs().sort_values(ascending=False,kind='stable').head(3).index.tolist()
        for order,c in enumerate(selected,1):
            direction=float(np.sign(rank.loc[c]))
            for split in ['train','validation']:
                q=table[(table.tier==tier)&(table.method==method)&(table.coordinate==c)&(table.split==split)]
                values=q.groupby('object').standardized_delta.mean()
                shortlist.append(dict(tier=tier,method=method,coordinate=int(c),train_rank=order,train_direction=direction,split=split,
                    pairs=len(q),objects=len(values),mean_standardized_delta_equal_objects=float(values.mean()) if len(values) else None,
                    sign_concordant_pairs=int((direction*q.standardized_delta>0).sum()),
                    median_absolute_delta=float(q.delta.abs().median()) if len(q) else None))
    pd.DataFrame(shortlist).to_csv(out/'train_selected_candidates.csv',index=False,encoding='utf-8-sig')
    # Match balance and disjointness are checked independently of feature scores.
    for (tier,split),g in pairs.groupby(['tier','split']):
        tids=list(g.near_tid)+list(g.far_tid);assert len(tids)==len(set(tids))
        assert (abs(g.far_angle-g.near_angle)<=.01+1e-12).all()
        assert ((g.far_distance-g.near_distance)>=.03-1e-12).all()
        assert (g.speed_gap<=.15+1e-12).all()
        if tier=='strict':assert g.contact_pattern_equal.all()
    assert max_tail_error<1e-10
    assert max_actor_error<2e-5
    assert all(sha(Path(p))==h for p,h in sources.items())
    dump(out/'source_hashes.json',sources)
    dump(out/'verification.json',dict(passed=True,development_trajectories=len(records),matched_pairs=len(pairs),
        counts={str(k):int(v) for k,v in pairs.groupby(['tier','split']).size().items()},
        trajectory_reuse_within_tier_split=False,candidate_selection_uses_train_only=True,test_trajectories_loaded=0,
        max_actor_recording_error=max_actor_error,max_exact_tail_vector_residual=max_tail_error,
        source_files_unchanged=True,new_sae_or_probe_fits=0,new_physical_rollouts=0))
    np.savez_compressed(out/'matched_snapshot_vectors.npz',h_near=near_h,h_far=far_h,
        relative_near=np.asarray([trajectories[r.near_tid]['relative_palm'][int(r.near_local)] for _,r in pairs.iterrows()]),
        relative_far=np.asarray([trajectories[r.far_tid]['relative_palm'][int(r.far_local)] for _,r in pairs.iterrows()]))
    print(json.dumps(json.loads((out/'verification.json').read_text()),ensure_ascii=False))


if __name__=='__main__':main()
