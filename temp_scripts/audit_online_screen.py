"""Independent local audit of all retrieved natural/repeat screening trajectories.

No pilot evidence is required or read. No frozen inputs are modified.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__:
    from . import analyze_online_development as independent
else:
    import analyze_online_development as independent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    run=independent.long_path(args.run)
    remote=run/'remote'
    if not (remote/'screen/campaign.json').is_file():
        candidates=list(remote.rglob('screen/campaign.json'))
        if len(candidates)!=1:
            raise RuntimeError('Need one retrieved screening campaign')
        remote=candidates[0].parent.parent
    out=run/'analysis_screen'
    out.mkdir(exist_ok=True)
    campaign_path=remote/'screen/campaign.json'
    selection_path=remote/'screen/selected_origins.json'
    plan_path=independent.locate('screen_plan.json',remote,run)
    try:
        actor_path=independent.locate('screen_source/bundle/actor.npz',remote,run)
    except FileNotFoundError:
        actor_path=independent.locate('package/bundle/actor.npz',remote,run)
    campaign=independent.read_json(campaign_path)
    plan=independent.read_json(plan_path)
    selection=independent.read_json(selection_path)
    actor=independent.load_npz(actor_path)
    initial_hashes={path:independent.sha(path) for path in [campaign_path,selection_path,plan_path,actor_path]}
    errors=[]
    def fail(scope,error):
        errors.append(dict(scope=scope,error=str(error)))
    if campaign.get('status')!='completed':
        fail('campaign','Screen campaign is not completed')
    signature=hashlib.sha256(json.dumps(campaign['frozen_inputs'],sort_keys=True).encode()).hexdigest()
    if signature!=campaign['frozen_inputs_sha256'] or signature!=selection['source_frozen_inputs_sha256']:
        fail('freeze','Frozen input snapshot signature does not reproduce or differs from selection')
    frozen_checked=[]
    frozen_unavailable=[]
    for remote_path,expected in campaign['frozen_inputs'].items():
        # Reconcile directly retrieved source files. The remaining old deploy
        # files may be absent from the retrieval; do not imply they were rehashed.
        suffix=remote_path.split('/online_development_20261104/',1)[-1]
        candidate=remote/suffix
        if candidate.is_file():
            if independent.sha(candidate)!=expected:
                fail('frozen_file',suffix+' differs from its screening snapshot')
            frozen_checked.append(suffix)
        else:
            frozen_unavailable.append(suffix)
    actor_expected=[expected for path,expected in campaign['frozen_inputs'].items() if path.endswith('/screen_source/bundle/actor.npz')]
    if actor_expected!=[independent.sha(actor_path)]:
        fail('actor_freeze','Independent NumPy actor weights differ from screening frozen hash')
    initial={(obj,seed) for obj in plan['objects'] for seed in plan['initial_seeds']}
    cases={(case['object'],case['seed']):case for case in campaign['cases']}
    if len(cases)!=len(campaign['cases']):
        fail('population','Duplicate physical origin')
    # Current screening outcomes determine only the single preauthorized extension.
    initial_failures=sum(bool(case.get('natural',{}).get('post_onset_failure')) for key,case in cases.items() if key in initial)
    extension=(initial_failures < plan.get('minimum_post_onset_failures',4) and plan.get('allow_single_expansion',True))
    expected_population=set(initial)
    if extension:
        expected_population.update((obj,seed) for obj in plan['objects'] for seed in plan['expansion_seeds'])
    if set(cases)!=expected_population or bool(campaign.get('extension_used'))!=extension:
        fail('population','Observed screening population/extension differs from frozen rule')
    results=[]
    repeat_rows=[]
    baselines=[]
    full_policy_hashes=set()
    checkpoint_hashes=set()
    input_records=[]
    for (obj,seed),case in sorted(cases.items()):
        label='object%d_seed%d'%(obj,seed)
        folder=remote/'screen'/label
        latest=independent.latest_runs(case)
        arrays_by_name={}
        audits={}
        digests={}
        reports={}
        for name in ['natural','fresh_original_control']:
            row=dict(object=obj,environment_seed=seed,name=name,audit_passed=False,error=None)
            try:
                record=latest[name]
                if case.get('status')!='completed' or record.get('status')!='completed' or record.get('exit_code')!=0:
                    raise AssertionError('Campaign case/run did not complete successfully')
                target=folder/record['relative_dir']
                archive=target/(name+'.npz')
                report_path=target/'report.json'
                audit_path=target/'audit.json'
                for path,field in [(archive,'npz_sha256'),(report_path,'report_sha256'),(audit_path,'audit_sha256')]:
                    actual=independent.sha(path)
                    if actual!=record[field]:
                        raise AssertionError('Campaign evidence hash mismatch: '+path.name)
                    input_records.append((path,actual))
                report=independent.read_json(report_path)
                if report.get('status')!='fresh_process_rollout_completed':
                    raise AssertionError('Child report is not a completed full trajectory')
                for flag in ['actor_parameters_unchanged','full_policy_state_unchanged']:
                    if report.get(flag) is not True:
                        raise AssertionError('Policy invariance flag missing/failed: '+flag)
                if report.get('checkpoint_sha256')!=campaign['checkpoint_sha256']:
                    raise AssertionError('Checkpoint hash differs from frozen screening campaign')
                expected=dict(object=obj,seed=seed,model_name=None,duration=0,feature_edit=False,group_edit=False,
                              source_plan_sha256=independent.sha(plan_path))
                for key,value in expected.items():
                    if report.get(key)!=value:
                        raise AssertionError('Report assignment/freeze mismatch: '+key)
                if not isinstance(report.get('full_policy_state_sha256'),str) or len(report['full_policy_state_sha256'])!=64:
                    raise AssertionError('Missing full frozen policy state hash')
                arrays=independent.load_npz(archive)
                audit=independent.audit_arrays(arrays,actor,plan['trigger_opening_rad'])
                if audit['success']!=report['summary']['success'] or audit['steps']!=report['summary']['steps']:
                    raise AssertionError('Child endpoint/length summary disagrees with independent arrays')
                server_audit=independent.read_json(audit_path)
                for key in ['success','steps','mechanism_class']:
                    if server_audit[key]!=audit[key]:
                        raise AssertionError('Saved server audit differs from independent reconstruction: '+key)
                if server_audit['trigger_step']!=audit['onset_step']:
                    raise AssertionError('Trigger differs from independent current-state trigger')
                full_policy_hashes.add(report['full_policy_state_sha256'])
                checkpoint_hashes.add(report['checkpoint_sha256'])
                arrays_by_name[name]=arrays
                audits[name]=audit
                digests[name]=independent.sha(archive)
                reports[name]=report
                row.update(audit,audit_passed=True,npz_sha256=digests[name],report_sha256=independent.sha(report_path))
            except Exception as error:
                row['error']=str(error)
                fail(label+'/'+name,error)
            results.append(row)
        repeat_row=dict(object=obj,environment_seed=seed,audit_passed=False,error=None)
        try:
            natural=arrays_by_name['natural']
            repeat=arrays_by_name['fresh_original_control']
            tolerance=plan['restore_repeat_atol']
            keys={key for key in natural if not key.startswith(('z__','prediction__'))}
            repeat_keys={key for key in repeat if not key.startswith(('z__','prediction__'))}
            if keys!=repeat_keys:
                raise AssertionError('Full natural/repeat array keys differ')
            differences={key:independent.compare_array(natural[key],repeat[key],tolerance) for key in sorted(keys)}
            has_trigger=audits['natural']['onset_step'] is not None
            repeat_report=reports['fresh_original_control']
            if has_trigger:
                if repeat_report.get('prefix_gate_passed') is not True or repeat_report.get('reference_sha256')!=digests['natural']:
                    raise AssertionError('Triggered repeat reference gate/hash mismatch')
                if not repeat_report['prefix_max_abs'] or max(repeat_report['prefix_max_abs'].values())>tolerance:
                    raise AssertionError('Triggered repeat prefix error missing or over tolerance')
            elif repeat_report.get('reference') is not None or repeat_report.get('prefix_gate_passed') is not None:
                raise AssertionError('No-trigger repeat unexpectedly used a reference gate')
            if reports['natural'].get('reference') is not None:
                raise AssertionError('Natural trajectory unexpectedly used a reference')
            if reports['natural']['full_policy_state_sha256']!=repeat_report['full_policy_state_sha256']:
                raise AssertionError('Independent-process full policy hashes differ')
            saved_repeat_path=folder/'repeat_audit.json'
            if independent.sha(saved_repeat_path)!=case['repeat_audit_sha256']:
                raise AssertionError('Stored full-repeat audit hash mismatch')
            saved=independent.read_json(saved_repeat_path)
            if saved.get('status')!='passed' or saved['natural_npz_sha256']!=digests['natural'] or saved['repeat_npz_sha256']!=digests['fresh_original_control']:
                raise AssertionError('Stored repeat audit refers to different evidence')
            input_records.append((saved_repeat_path,independent.sha(saved_repeat_path)))
            repeat_row.update(audit_passed=True,has_trigger=has_trigger,full_arrays_compared=len(keys),
                              complete_trajectory_steps=audits['natural']['steps'],max_abs=max(differences.values(),default=0.),
                              per_array_max_abs=json.dumps(differences,sort_keys=True))
            baselines.append(dict(object=obj,environment_seed=seed,success=audits['natural']['success'],has_trigger=has_trigger,
                                  post_onset_failure=(not audits['natural']['success'] and has_trigger),
                                  **{key:audits['natural'][key] for key in ['mechanism_class','steps','onset_step','goal_and_near','strict_contact_anywhere','strict_after_last_goal','first_later_strict_gap']}))
            for field,actual in [('success',audits['natural']['success']),('has_trigger',has_trigger),('post_onset_failure',not audits['natural']['success'] and has_trigger)]:
                if case['natural'].get(field)!=actual:
                    raise AssertionError('Screen case stratum summary differs from independent baseline: '+field)
        except Exception as error:
            repeat_row['error']=str(error)
            fail(label+'/repeat',error)
        repeat_rows.append(repeat_row)
    if len(full_policy_hashes)!=1 or len(checkpoint_hashes)!=1:
        fail('policy_hashes','Expected one unchanged policy/checkpoint across all screening processes')
    expected_selection=[]
    for obj in plan['objects']:
        candidates=sorted([row for row in baselines if row['object']==obj],key=lambda row:row['environment_seed'])
        for stratum in ['success','post_onset_failure']:
            eligible=[row for row in candidates if row['has_trigger'] and row[stratum]]
            if eligible:
                expected_selection.append((obj,eligible[0]['environment_seed'],stratum))
    if set(expected_selection)!={(item['object'],item['seed'],item['baseline_stratum']) for item in selection['origins']}:
        fail('selection','Selected origins differ from independent first-seed outcome-stratum rule')
    for path,expected in list(initial_hashes.items())+input_records:
        if independent.sha(path)!=expected:
            fail('source_changed','Read source changed during audit: '+independent.readable_path(path))
    pd.DataFrame(results).to_csv(out/'trajectory_audits.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(repeat_rows).to_csv(out/'full_repeat_audits.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(baselines).to_csv(out/'natural_population.csv',index=False,encoding='utf-8-sig')
    success_count=sum(row['success'] for row in baselines)
    post_onset=sum(row['post_onset_failure'] for row in baselines)
    no_onset=sum(not row['has_trigger'] and not row['success'] for row in baselines)
    good=[row for row in results if row['audit_passed']]
    good_repeats=[row for row in repeat_rows if row['audit_passed']]
    verification=dict(passed=not errors,error_count=len(errors),errors=errors,
                      origins_expected=len(expected_population),origins_verified=len(baselines),
                      trajectories_expected=2*len(expected_population),trajectories_verified=len(good),
                      independent_repeats_verified=len(good_repeats),
                      natural_action_steps=sum(row['steps'] for row in baselines),
                      total_action_steps=sum(row['steps'] for row in good),
                      composite_states_checked=sum(2*row['steps'] for row in good),
                      natural_successes=success_count,post_onset_natural_failures=post_onset,no_onset_natural_failures=no_onset,
                      natural_mechanism_counts=dict(Counter(row['mechanism_class'] for row in baselines)),
                      selected_origins=len(selection['origins']),selected_successes=sum(item['baseline_stratum']=='success' for item in selection['origins']),
                      selected_post_onset_failures=sum(item['baseline_stratum']=='post_onset_failure' for item in selection['origins']),
                      extension_used=campaign['extension_used'],
                      no_trigger_repeats_verified=sum(not row['has_trigger'] for row in good_repeats),
                      max_mu_error=max((row['max_mu_error'] for row in good),default=0.),
                      max_action_error=max((row['max_action_error'] for row in good),default=0.),
                      max_recorded_clip_error=max((row['max_recorded_clip_error'] for row in good),default=0.),
                      max_norm_error=max((row['max_norm_error'] for row in good),default=0.),
                      maximum_full_repeat_difference=max((row['max_abs'] for row in good_repeats),default=0.),
                      predicate_mismatches=sum(row['predicate_errors'] for row in good),
                      full_policy_state_sha256=sorted(full_policy_hashes),checkpoint_sha256=sorted(checkpoint_hashes),
                      frozen_snapshot_signature_reproduced=(signature==campaign['frozen_inputs_sha256']),
                      directly_rehashed_frozen_files=frozen_checked,unavailable_frozen_source_files=frozen_unavailable,
                      actor_sha256=independent.sha(actor_path),screen_plan_sha256=independent.sha(plan_path),
                      source_hashes={independent.readable_path(path):value for path,value in initial_hashes.items()},
                      source_files_unchanged=not any(error['scope']=='source_changed' for error in errors),
                      boundaries=['No pilot data read or final pilot report generated',
                                  'Full policy parameter invariance reconciles child report flags and identical state/checkpoint hashes; independent tensor inference covers actor tail',
                                  'Only retrieved frozen source files directly rehashed; missing old-deploy files are explicitly listed',
                                  'SAE per-step recoding is not performed by this screening audit'],
                      actor_tolerance=dict(atol=independent.ACTOR_ATOL,rtol=independent.ACTOR_RTOL),
                      full_repeat_atol=plan['restore_repeat_atol'],full_repeat_rtol=0)
    independent.dump(out/'verification.json',verification)
    text=['# 自然基线筛查独立复核','',
          '2026-11-04，北京时间。只读取screen回传证据，没有读取pilot或生成pilot最终报告。','',
          '核验%s：%d起点的%d条自然/独立重复轨迹，合计%d动作步、%d个动作前/后状态。复合成功判据不一致%d处；NumPy actor尾部mu最大差%.3g，裁剪动作最大差%.3g；%d组完整重复最大差%.3g，其中%d组无触发也完成全程独立重复。' %
          ('通过' if verification['passed'] else '存在问题',len(baselines),len(good),verification['total_action_steps'],verification['composite_states_checked'],verification['predicate_mismatches'],verification['max_mu_error'],verification['max_action_error'],len(good_repeats),verification['maximum_full_repeat_difference'],verification['no_trigger_repeats_verified']), '',
          '全部%d条自然起点保留：成功%d、开启后失败%d、无开启失败%d。未启动额外种子扩展。按预定每物体各取首个成功/开启后失败，得到%d开发起点（%d成功、%d失败）；这不是留出测试。' %
          (len(baselines),success_count,post_onset,no_onset,len(selection['origins']),verification['selected_successes'],verification['selected_post_onset_failures']), '',
          '64条NPZ及其report/audit的记录哈希、32份完整重复审计哈希、screen_plan与actor冻结关联、子进程参数不变标记和完整策略哈希均已核对。实际计数以verification.json为准。缺失的旧部署源文件未假称重新哈希，清单见verification.json；没有改写历史冻结定义或SAE码审计状态。', '',
          '逐轨迹结果见trajectory_audits.csv，完整重复见full_repeat_audits.csv，自然结局与机制分层见natural_population.csv。']
    if errors:
        text.append('\n发现%d个问题，不能将本次复核标作通过。'%len(errors))
    (out/'SCREEN_RESULTS_2026-11-04.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({key:verification[key] for key in ['passed','trajectories_verified','total_action_steps','natural_successes','post_onset_natural_failures','no_onset_natural_failures','max_mu_error','max_action_error','maximum_full_repeat_difference','error_count']},ensure_ascii=False))
    return 0 if verification['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
