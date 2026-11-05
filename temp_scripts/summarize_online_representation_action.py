"""Sensitivity and origin-weighted interpretation of frozen readout diagnostics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from temp_scripts.analyze_online_representation_action import native, actor, digest


def main():
    root=native('outputs/online_development_20261104')
    out=native('outputs/deeper_analysis_20261105/representation')
    first=pd.read_csv(out/'first_assigned_step.csv')
    all_rows=pd.read_csv(out/'scheduled_readout_action_rows.csv')
    primary=first[first.family.eq('assisted') & first.group.eq('top4') & first.duration.eq(5)].copy()
    targets=primary[primary.control.isin(['decrease','bounded_increase'])].copy()
    before_hashes={str(p):digest(p) for p in [out/'first_assigned_step.csv',out/'scheduled_readout_action_rows.csv',root/'package/bundle/actor.npz']}
    with np.load(root/'package/bundle/actor.npz',allow_pickle=False) as a:weights={k:a[k].astype(float) for k in a.files}
    raw_errors=[];gate_ambiguity=[]
    for _,row in primary.iterrows():
        path=root/row.archive
        natural=root/'remote/screen'/('object%d_seed%d' % (row.object,row.environment_seed))/'natural/natural.npz'
        with np.load(path,allow_pickle=False) as edited,np.load(natural,allow_pickle=False) as base:
            trigger=int(row.frame)
            delta=float(edited['post_opening'][trigger+4]-base['post_opening'][trigger+4])
            raw_errors.append(abs(delta-row.physical_opening_delta_after5_rad))
            before=actor(weights,edited['h_original'][trigger])[0]
            after=actor(weights,edited['h_applied'][trigger])[0]
            flips=(before>0)!=(after>0)
            ambiguous=(np.abs(before)<=2e-5)|(np.abs(after)<=2e-5)
            gate_ambiguity.append(dict(object=row.object,environment_seed=row.environment_seed,model=row.model,control=row.control,
                                       h2_gate_flips=int(flips.sum()),robust_gate_flips=int((flips&~ambiguous).sum()),
                                       ambiguous_near_zero_gates=int(ambiguous.sum()),
                                       ambiguous_gate_flips=int((flips&ambiguous).sum())))
    pd.DataFrame(gate_ambiguity).to_csv(out/'first_step_gate_boundary_checks.csv',index=False,encoding='utf-8-sig')
    thresholds=[];distribution=[]
    for subset,frame in [('all_assigned',targets),('first_step_nonzero',targets[targets.nonzero_h])]:
        for control,g in frame.groupby('control'):
            x=g.physical_opening_delta_after5_rad
            distribution.append(dict(subset=subset,control=control,assigned_model_origin_rows=len(g),physical_origins=g[['object','environment_seed']].drop_duplicates().shape[0],
                                      mean=float(x.mean()),median=float(x.median()),q25=float(x.quantile(.25)),q75=float(x.quantile(.75)),minimum=float(x.min()),maximum=float(x.max())))
            for eps in [0.,1e-4,1e-3,.01]:
                thresholds.append(dict(subset=subset,control=control,near_zero_threshold_rad=eps,assigned_model_origin_rows=len(g),
                                       positive=int((x>eps).sum()),negative=int((x<-eps).sum()),near_zero=int((abs(x)<=eps).sum())))
    distribution=pd.DataFrame(distribution);sensitivity=pd.DataFrame(thresholds)
    distribution.to_csv(out/'physical_five_step_delta_distribution.csv',index=False,encoding='utf-8-sig')
    sensitivity.to_csv(out/'physical_five_step_delta_zero_sensitivity.csv',index=False,encoding='utf-8-sig')
    pulse=all_rows[all_rows.duration.eq(5)].groupby(['object','environment_seed','model','control','group']).agg(
        five_step_nonzero_count=('nonzero_h','sum'),five_step_cumulative_h_norm=('actual_h_norm','sum')).reset_index()
    targets=targets.merge(pulse,on=['object','environment_seed','model','control','group'],validate='one_to_one')
    origin=targets.groupby(['object','environment_seed','control']).agg(
        assigned_seeds=('model','size'),first_nonzero_seeds=('nonzero_h','sum'),
        first_opening_readout_delta_rad=('opening_probe_delta_rad','mean'),
        physical_opening_delta_after5_rad=('physical_opening_delta_after5_rad','mean'),
        mean_five_step_nonzero_count=('five_step_nonzero_count','mean'),
        mean_five_step_cumulative_h_norm=('five_step_cumulative_h_norm','mean'),
        first_clipped_action_delta_l2=('clipped_action_delta_l2','mean')).reset_index()
    origin.to_csv(out/'readout_vs_physical_origin.csv',index=False,encoding='utf-8-sig')
    later_support=targets[(~targets.nonzero_h)&targets.five_step_nonzero_count.gt(0)]
    later_support.to_csv(out/'first_zero_but_later_exposed.csv',index=False,encoding='utf-8-sig')
    action_summary=primary.groupby('control').agg(
        assigned_model_origins=('model','size'),nonzero_h=('nonzero_h','sum'),
        h_norm=('actual_h_norm','mean'),raw_mu_l2=('raw_mu_delta_l2','mean'),clipped_action_l2=('clipped_action_delta_l2','mean'),
        opening_readout_delta_rad=('opening_probe_delta_rad','mean'),any_contact_delta=('any_contact_probe_delta','mean'),strict_contact_delta=('strict_contact_probe_delta','mean')).reset_index()
    action_summary.to_csv(out/'first_step_action_magnitude_allocation_summary.csv',index=False,encoding='utf-8-sig')
    strict_file=native('outputs/local_interpretability/20261104_mainline_v3/strict_contact_probes.joblib')
    probes=joblib.load(strict_file)
    strict_before_hash=digest(strict_file)
    with np.load(out/'scheduled_readout_action_vectors.npz',allow_pickle=False) as vectors:
        logit_rows=[]
        for _,r in first.iterrows():
            if r.group!='top4' or r.duration!=5:continue
            probe=probes[r.model+'__opening_top4'];coords=probe['coordinates']
            slope=probe['model'].coef_.reshape(-1)/probe['scaler'].scale_
            dz=vectors['z_after'][int(r.row),coords]-vectors['z_before'][int(r.row),coords]
            logit_rows.append(dict(object=r.object,environment_seed=r.environment_seed,model=r.model,family=r.family,control=r.control,
                                   nonzero_h=bool(r.nonzero_h),p_before=float(r.strict_contact_probe_before),
                                   p_after=float(r.strict_contact_probe_before+r.strict_contact_probe_delta),
                                   probability_delta=float(r.strict_contact_probe_delta),logit_delta=float(dz@slope)))
    strict_rows=pd.DataFrame(logit_rows)
    strict_rows.to_csv(out/'first_step_strict_contact_logit_rows.csv',index=False,encoding='utf-8-sig')
    strict_summary=[]
    for key,g in strict_rows.groupby(['family','control']):
        strict_summary.append(dict(family=key[0],control=key[1],assigned_rows=len(g),nonzero_rows=int(g.nonzero_h.sum()),
                                   p_before_min=float(g.p_before.min()),p_before_median=float(g.p_before.median()),p_before_max=float(g.p_before.max()),
                                   logit_delta_min=float(g.logit_delta.min()),logit_delta_q25=float(g.logit_delta.quantile(.25)),
                                   logit_delta_median=float(g.logit_delta.median()),logit_delta_q75=float(g.logit_delta.quantile(.75)),logit_delta_max=float(g.logit_delta.max()),
                                   probability_delta_mean=float(g.probability_delta.mean())))
    pd.DataFrame(strict_summary).to_csv(out/'first_step_strict_contact_logit_summary.csv',index=False,encoding='utf-8-sig')
    original_hashes_match=all(digest(Path(p))==v for p,v in before_hashes.items())
    assert original_hashes_match
    assert max(raw_errors)<1e-15
    check=dict(passed=True,first_primary_assisted_rows_checked=len(raw_errors),maximum_raw_five_step_delta_error=max(raw_errors),
               source_artifacts_unchanged=original_hashes_match,
               ambiguous_h2_gate_flips=sum(r['ambiguous_gate_flips'] for r in gate_ambiguity),
               near_zero_gate_definition='Either side abs(h2 preactivation)<=2e-5; approximate float64 gate counts not used as exact causal evidence',
               first_zero_later_exposed_target_rows=len(later_support),source_hashes=before_hashes,
               frozen_strict_probe_hash_unchanged=(digest(strict_file)==strict_before_hash),
               summary_implementation_sha256=digest(native(__file__)))
    (out/'secondary_verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['## 进一步解读：内部状态读数并非未来开盖命令','',
           '全部231条件/2640个调度帧的数值核验通过，不含7个专用noop。主Top4首步辅助各方向21个模型—起点行中17个非零：增加16正/1负/4零，下降16负/1正/4零；同预算Ridge非零17行全部符合其标记方向。辅助组实际码L2首步下降后平均剩40.3%、增加后为161.9%（这两个比例仅对17个非零、原目标码L2可定义者），不是精确归零或纯状态概念。', '',
           '首步唯一读出反号起点是11395/2026110400的assisted_seed0：自然当前开启.13454rad，旧局部探针读.01515rad；码下降使读出+.00832，码增加使读出−.00661rad。其前5步物理开启与自然完全相同。这是局部读出/阶段适用性的反例，不是数学实现错误。', '',
           '对首步非零的17行，码增加的第5个动作后真实开启差中位−.01963rad，IQR[−.02386,−.00622]，范围[−.04533,+.06535]；码下降中位+.01530rad，IQR[+.01150,+.03669]，范围[−.00231,+.11455]。当前状态探针的变化与五步闭环物理后果时间含义不同，不能把这种对应直接叫作负反馈控制机制。', '',
           '|首步非零子集|近零阈值(rad)|五步物理差正/负/近零|', '|---|---:|---|']
    for _,r in sensitivity[sensitivity.subset.eq('first_step_nonzero')].iterrows():
        lines.append('|%s|%g|%d/%d/%d|'%(r.control,r.near_zero_threshold_rad,r.positive,r.negative,r.near_zero))
    lines += ['', '上述17行是首步支持的诊断子集，不能替代全分配结果。把7个起点内的3个模型种子先平均、零支持照常保留，可见增加在4起点使五步物理开启较少、2较多、1相同；下降在6起点较多、1相同。', '',
              '|起点/种子|操作|首步非零模型数/3|首步开启状态读出变化(rad)|第5动作后物理开启差(rad)|',
              '|---|---|---:|---:|---:|']
    for _,r in origin.iterrows():
        lines.append('|%d/%d|%s|%d/3|%+.5f|%+.5f|'%(r.object,r.environment_seed,r.control,r.first_nonzero_seeds,r.first_opening_readout_delta_rad,r.physical_opening_delta_after5_rad))
    lines += ['', '一部分首步零支持的分支在后续步骤变为有支持，因此不能将首步0等同整段无处理；详见first_zero_but_later_exposed.csv。所有21行的第5步物理差分布和近零敏感度也保留在两个physical_five_step_delta表中。', '',
              '同一h和相同实际h范数下，首步辅助目标码下降/增加的平均裁剪动作变化L2为.06845/.07474；三个随机方向分别.05527/.05710/.06233，Ridge两符号约.05896/.05910。等h范数并非等动作强度；这是一般动作敏感性的剩余解释，不能从破坏/救援数量反推出语义特异性。', '',
              '目标与同符号Ridge的首步裁剪动作方向cosine在17个非零模型—起点对中中位数分别为下降.303、增加.521，既有共享成分也非同一个方向。对照cosine仅在双方非零时定义；无支持完整留在主分母，条件诊断不作新的处理效应估计。', '',
              '严格接触开发探针在7起点触发时的实际标签均为负。辅助Top4的21个模型—起点p_before范围7.06e-6至.0014285，中位9.42e-5；小概率下有sigmoid压缩，因此绝对概率变化1e-4量级不能表示logit未变。码增加的logit delta中位+.22124、IQR[+.17662,+.27947]、范围[0,+.66478]；码下降中位−.25605、IQR[−.38764,−.11047]、范围[−.59409,+.40037]（含零支持行）。底表见first_step_strict_contact_logit_summary.csv。不能据小概率变化排除当前接触读出、相关表示或未来接触控制的作用，也不能把旧任一接触读出变化等同严格复合成功概念。', '',
              '建议把现有机制表述收紧为：少量码方向可稳定改变冻结局部状态读数与即时动作，但局部探针标度、实际动作增益、触发阶段及闭环后果各有独立环节；当前证据尚未将这些环节识别为单一、可泛化的开盖/接触概念控制通路。']
    report=out/'REPRESENTATION_ACTION_2026-11-05.md'
    current=report.read_text(encoding='utf-8')
    marker='\n## 进一步解读：内部状态读数并非未来开盖命令\n'
    if marker in current:current=current.split(marker)[0]
    report.write_text(current+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(check,ensure_ascii=False))


if __name__=='__main__':
    main()
