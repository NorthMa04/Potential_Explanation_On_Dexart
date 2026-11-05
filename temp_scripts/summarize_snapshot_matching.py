"""Render measured snapshot geometry and disclose residual matching imbalance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from temp_scripts.data_access import DataRepository
from temp_scripts.project_paths import PROJECT_ROOT, default_index


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])


def main():
    out=PROJECT_ROOT/'outputs/snapshot_matching_20261105'
    pairs=pd.read_csv(out/'matched_pairs.csv');repo=DataRepository(default_index())
    details=[];geometry={}
    for _,r in pairs.iterrows():
        near=repo.load(r.near_tid);far=repo.load(r.far_tid);i=int(r.near_step);j=int(r.far_step)
        a,b=near.arrays,far.arrays
        oa=rotation(a['object_quaternion_wxyz'][i]);ob=rotation(b['object_quaternion_wxyz'][j])
        root_angle=float(np.rad2deg((oa.inv()*ob).magnitude()))
        vdiff=float(np.linalg.norm(oa.inv().apply(a['palm_velocity_world'][i])-ob.inv().apply(b['palm_velocity_world'][j])))
        finger=float(np.sqrt(np.mean((a['robot_qpos'][i,6:]-b['robot_qpos'][j,6:])**2)))
        scene=float(np.linalg.norm(a['object_position_world'][i]-b['object_position_world'][j]))
        details.append(dict(pair_id=int(r.pair_id),tier=r.tier,split=r.split,object=int(r.object),
                            scene_position_gap_m=scene,scene_rotation_gap_deg=root_angle,finger_tail16_rms_rad=finger,
                            object_frame_palm_velocity_gap_m_s=vdiff,contact_pattern_equal=bool(r.contact_pattern_equal),
                            strict_scene_position_ok=scene<=.025,strict_scene_orientation_ok=root_angle<=10,
                            strict_finger_ok=finger<=.25,strict_palm_velocity_ok=vdiff<=.25))
        points=[]
        for data,k,o in [(a,i,oa),(b,j,ob)]:
            palm=o.inv().apply(data['palm_position_world'][k]-data['handle_position_world'][k])
            tips=o.inv().apply(data['fingertip_position_world'][k]-data['handle_position_world'][k])
            points.append((palm,tips))
        geometry[int(r.pair_id)]=points
    balance=pd.DataFrame(details);balance.to_csv(out/'residual_covariate_balance.csv',index=False,encoding='utf-8-sig')
    # Example is selected by physical matching score among validation pairs, not activations.
    example=pairs[pairs.split.eq('validation')].sort_values('score').iloc[0]
    k=int(example.pair_id)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(11,4.8),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    for (palm,tips),color,label in zip(geometry[k],['#0072b2','#d55e00'],['Near','Far']):
        ax.scatter(*palm,s=65,color=color,label=label+' palm',marker='s')
        ax.scatter(tips[:,0],tips[:,1],tips[:,2],s=20,color=color,marker='o')
        for tip in tips:ax.plot([palm[0],tip[0]],[palm[1],tip[1]],[palm[2],tip[2]],color=color,alpha=.45)
    ax.scatter(0,0,0,s=80,marker='*',color='black',label='Handle')
    ax.set_xlabel('Object-frame x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)')
    ax.set_title('Measured palm / fingertip locations\n(lines are schematic, not hand anatomy)',fontsize=10)
    ax.legend(fontsize=7,loc='upper left')
    ax=fig.add_subplot(122)
    effects=pd.read_csv(out/'all_coordinate_pair_differences.csv')
    candidates=pd.read_csv(out/'train_selected_candidates.csv')
    fixed=candidates[candidates.tier.eq('basic')&candidates.method.eq('assisted_seed0')&candidates.split.eq('train')].sort_values('train_rank').coordinate.to_numpy()
    rows=effects[effects.pair_id.eq(k)&effects.method.eq('assisted_seed0')].set_index('coordinate')
    xs=np.arange(len(fixed));ax.bar(xs-.18,rows.loc[fixed,'near_value'],.36,color='#0072b2',label='Near')
    ax.bar(xs+.18,rows.loc[fixed,'far_value'],.36,color='#d55e00',label='Far')
    ax.set_xticks(xs,['z'+str(c) for c in fixed]);ax.set_ylabel('Frozen SAE code value')
    ax.set_title('Train-selected candidates, shown on one validation pair\nOne pair cannot establish replication',fontsize=10)
    ax.legend();ax.spines[['top','right']].set_visible(False);ax.set_ylim(bottom=0)
    fig.suptitle('Angle-matched observational snapshots: object %d, angle gap %.3f deg\nPalm-handle distance %.1f vs %.1f cm; residual covariates still differ' %
                 (example.object,np.rad2deg(example.angle_gap),example.near_distance*100,example.far_distance*100),fontsize=11)
    fig.savefig(out/'measured_snapshot_example.png',dpi=180);plt.close(fig)
    summary=dict(example_pair_id=k,example_selection='lowest physical mismatch validation pair',
                 warning='measured geometry schematic, not simulator render; no causal feature lock',
                 strict_failure_counts={c:int((~balance[c]).sum()) for c in ['contact_pattern_equal','strict_scene_position_ok','strict_scene_orientation_ok','strict_finger_ok','strict_palm_velocity_ok']})
    (out/'snapshot_example.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary))


if __name__=='__main__':main()
