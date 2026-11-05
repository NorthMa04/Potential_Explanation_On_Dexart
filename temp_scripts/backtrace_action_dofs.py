"""Read-only numerical tracing from actuator channels through a frozen actor.

The finite edit example uses original/applied h1 at the SAME physical state.
It explains the actor output, not the later closed-loop physical outcome.
"""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

from temp_scripts.analyze_sustained_contact_coordination import native
from temp_scripts.data_access import DataRepository
from temp_scripts.match_angle_hand_snapshots import actor, tail_path
from temp_scripts.online_bidirectional import OnlineBidirectionalEditor
from temp_scripts.physical_runtime import NumericSAE
from temp_scripts.project_paths import PROJECT_ROOT, default_index


def sha(p):
    return hashlib.sha256(native(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def write_csv(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_urdf(archive):
    member = 'assets/robot/xarm6_description/xarm6_allegro_wrist_mounted_rotate.urdf'
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    joints = []
    for e in ET.fromstring(raw).findall('joint'):
        origin, axis, limit = e.find('origin'), e.find('axis'), e.find('limit')
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        joints.append(dict(name=e.get('name'), type=e.get('type'), parent=e.find('parent').get('link'),
            child=e.find('child').get('link'), transform=transform,
            axis=np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' '),
            limits=None if limit is None else [float(limit.get('lower')), float(limit.get('upper'))]))
    return joints, member, hashlib.sha256(raw).hexdigest()


def forward_kinematics(joints, q):
    active = [j for j in joints if j['type'] != 'fixed']
    values = dict(zip([j['name'] for j in active], q))
    roots = {j['parent'] for j in joints} - {j['child'] for j in joints}
    known = {r: np.eye(4) for r in roots}
    for r in roots:
        known[r][:3, 3] = [-.5, 0, 0]
    remaining = list(range(len(joints)))
    while remaining:
        done = []
        for i in remaining:
            j = joints[i]
            if j['parent'] not in known:
                continue
            rotation = np.eye(4)
            if j['type'] in ('revolute', 'continuous'):
                rotation[:3, :3] = Rotation.from_rotvec(j['axis'] * values[j['name']]).as_matrix()
            known[j['child']] = known[j['parent']] @ j['transform'] @ rotation
            done.append(i)
        if not done:
            raise ValueError('Disconnected URDF')
        remaining = [i for i in remaining if i not in done]
    return known


def main():
    out = PROJECT_ROOT / 'outputs/action_backtrace_20261105'
    out.mkdir(parents=True, exist_ok=True)
    online = PROJECT_ROOT / 'outputs/online_development_20261104'
    actor_path = online / 'package/bundle/actor.npz'
    with np.load(actor_path, allow_pickle=False) as a:
        w = {k: a[k].astype(float) for k in a.files}
    output_w, output_b = w['action_net.weight'], w['action_net.bias']
    second_w, second_b = w['mlp_extractor.policy_net.2.weight'], w['mlp_extractor.policy_net.2.bias']
    model_path = online / 'package/bundle/models/assisted_seed0'
    model = NumericSAE(model_path)
    decoder = model.w['decoder_weight'].astype(float) * model.w['scale'].astype(float)[:, None]

    joints, member, urdf_sha = read_urdf(Path('D:/homework/dexart_assets.zip'))
    active = [j for j in joints if j['type'] != 'fixed']
    assert [j['name'] for j in active] == ['joint%d' % i for i in range(1, 7)] + ['joint_%d.0' % i for i in range(16)]
    channels = []
    for i in range(22):
        if i < 6:
            label = ('ee_vx', 'ee_vy', 'ee_vz', 'ee_wx', 'ee_wy', 'ee_wz')[i]
            kind, unit, limits, group = 'end_link_spatial_velocity_before_IK', ('m/s' if i < 3 else 'rad/s'), [-1., 1.], ('translation' if i < 3 else 'rotation')
        else:
            slot = i - 6
            group = ('index', 'middle', 'ring', 'thumb')[slot // 4]
            label = group + '_joint' + str(slot % 4)
            kind, unit, limits = 'finger_position_target', 'rad', active[i]['limits']
        channels.append(dict(action_index=i, label=label, group=group, command_type=kind, unit=unit,
            joint_name=active[i]['name'] if i >= 6 else '', lower=limits[0], upper=limits[1],
            target_derivative_per_clipped_action=(limits[1] - limits[0]) / 2))
    write_csv(out / 'action_dof_map.csv', channels)
    np.savetxt(out / 'output_weights_22x64.csv', output_w, delimiter=',')
    np.savetxt(out / 'output_bias_22.csv', output_b, delimiter=',')
    rows = []
    for i, channel in enumerate(channels):
        for rank, k in enumerate(np.argsort(-abs(output_w[i]))[:8], 1):
            rows.append(dict(action_index=i, action_label=channel['label'], rank_abs_weight=rank,
                h2_index=int(k), weight=float(output_w[i, k])))
    write_csv(out / 'largest_weights_per_action.csv', rows)

    campaign_path = online / 'remote/pilot/campaign.json'
    campaign = json.loads(campaign_path.read_text())
    case = next(c for c in campaign['cases'] if c['object'] == 11477 and c['seed'] == 2026110400)
    record = next(r for r in case['runs'] if r.get('condition', {}).get('duration') == -1
        and r['condition']['control'] == 'bounded_increase')
    source = online / 'remote/pilot/object11477_seed2026110400' / record['relative_dir'] / (record['name'] + '.npz')
    assert sha(source) == record['npz_sha256']
    with np.load(native(source), allow_pickle=False) as a:
        logged = {k: a[k].copy() for k in ('h_original', 'h_applied', 'h2', 'mu', 'action',
            'edit_z_before', 'edit_z_after', 'applied', 'robot_qpos', 'observation_state')}
    step = 28  # A documented pre-goal state; illustrative, not selected as a confirmatory endpoint.
    assert logged['applied'][step]
    h0, h1 = logged['h_original'][step].astype(float), logged['h_applied'][step].astype(float)
    pre0, h20, mu0 = actor(h0, w)
    pre1, h21, mu1 = actor(h1, w)
    a0, a1 = np.clip(mu0, -1, 1), np.clip(mu1, -1, 1)
    np.testing.assert_allclose(h21, logged['h2'][step], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(mu1, logged['mu'][step], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(a1, logged['action'][step], atol=2e-6, rtol=2e-6)
    last_contributions = output_w * (h21 - h20)[None, :]
    jac, delta_mu = tail_path(h0, h1, w)
    h1_contributions = jac * (h1 - h0)[None, :]
    local_jacobian = (output_w * (pre0 > 0)[None, :]) @ second_w
    local_sae = local_jacobian @ decoder
    reencoded_dz = logged['edit_z_after'][step].astype(float) - logged['edit_z_before'][step].astype(float)
    editor = OnlineBidirectionalEditor(online / 'package/bundle', {'assisted_seed0': model})
    reproduced, detail = editor.edit(h0, 'assisted_seed0', 'top4', 'full', 'bounded_increase', 'unused_for_target')
    np.testing.assert_array_equal(reproduced, logged['h_applied'][step])
    _, indices, _, direction, request_up, _, _, _, budget = editor._context(h0, 'assisted_seed0', 'top4')
    # Recover the actual scalar on the requested decoder ray, with the same
    # projected-norm bisection. Re-encoded z_after is a diagnostic, NOT the edit.
    length = np.linalg.norm(direction)
    unit_direction = direction / length
    low, high = 0., max(budget, 1e-8)
    for _ in range(60):
        if np.linalg.norm(np.maximum(h0 + high * unit_direction, 0) - h0) >= budget - 1e-10:
            break
        high *= 2
    for _ in range(65):
        mid = (low + high) / 2
        if np.linalg.norm(np.maximum(h0 + mid * unit_direction, 0) - h0) < budget:
            low = mid
        else:
            high = mid
    ray_scale = high / length
    dz = np.zeros(64)
    dz[indices] = ray_scale * request_up
    # Nonnegative h1 projection plus float32 application rounding are retained
    # as a separately named term, rather than assigned to an SAE feature.
    residual_h = (h1 - h0) - decoder @ dz
    sae_contributions = (jac @ decoder) * dz[None, :]
    residual_mu = jac @ residual_h
    np.testing.assert_allclose(last_contributions.sum(1), delta_mu, atol=1e-12)
    np.testing.assert_allclose(h1_contributions.sum(1), delta_mu, atol=1e-12)
    np.testing.assert_allclose(sae_contributions.sum(1) + residual_mu, delta_mu, atol=1e-12)
    effects, contributors = [], []
    for i, channel in enumerate(channels):
        scale = channel['target_derivative_per_clipped_action']
        offset = (channel['lower'] + channel['upper']) / 2
        effects.append(dict(action_index=i, action_label=channel['label'], command_unit=channel['unit'],
            mu_before=float(mu0[i]), mu_after=float(mu1[i]), delta_mu=float(delta_mu[i]),
            clipped_action_before=float(a0[i]), clipped_action_after=float(a1[i]),
            delta_clipped_action=float(a1[i]-a0[i]),
            target_before=float(scale*a0[i]+offset), target_after=float(scale*a1[i]+offset),
            delta_target=float(scale*(a1[i]-a0[i])),
            clipped_before=bool(abs(mu0[i]) > 1), clipped_after=bool(abs(mu1[i]) > 1)))
        for rank, k in enumerate(np.argsort(-abs(last_contributions[i]))[:8], 1):
            contributors.append(dict(action_index=i, action_label=channel['label'], rank_abs_contribution=rank,
                h2_index=int(k), weight=float(output_w[i, k]), h2_before=float(h20[k]), h2_after=float(h21[k]),
                delta_mu_contribution=float(last_contributions[i, k])))
    write_csv(out / 'example_same_state_action_effects.csv', effects)
    write_csv(out / 'example_last_layer_contributors.csv', contributors)
    write_csv(out / 'example_sae_feature_contributors.csv', [dict(action_index=i, action_label=channels[i]['label'],
        sae_feature=k, delta_z=float(dz[k]), delta_mu_contribution=float(sae_contributions[i,k]))
        for i in range(22) for k in np.flatnonzero(dz)])
    np.savez_compressed(out / 'example_exact_backtrace.npz', h_original=h0, h_applied=h1,
        h2_original=h20, h2_applied=h21, mu_original=mu0, mu_applied=mu1,
        h2_contributions=last_contributions, h1_contributions=h1_contributions,
        local_h1_jacobian=local_jacobian, local_sae_jacobian=local_sae,
        integrated_h1_jacobian=jac, sae_feature_contributions=sae_contributions,
        residual_h=residual_h, residual_mu=residual_mu, requested_scaled_delta_z=dz,
        observed_reencoded_delta_z=reencoded_dz)

    # Independently check active-joint order against observed FK, including fingertips.
    t = DataRepository(default_index()).load('object_11477__episode_00000')
    checks = []
    tips = ['link_15.0_tip', 'link_3.0_tip', 'link_7.0_tip', 'link_11.0_tip']
    for k in [0, 15, 25]:
        fk = forward_kinematics(joints, t.arrays['robot_qpos'][k])
        checks.append(dict(trajectory=t.tid, action_index=k, point='palm',
            position_error_m=float(np.linalg.norm(fk['base_link'][:3,3]-t.arrays['palm_position_world'][k]))))
        for i, tip in enumerate(tips):
            checks.append(dict(trajectory=t.tid, action_index=k, point=tip,
                position_error_m=float(np.linalg.norm(fk[tip][:3,3]-t.arrays['fingertip_position_world'][k,i]))))
    fk = forward_kinematics(joints, logged['robot_qpos'][step])
    checks.append(dict(trajectory='online_object11477_seed2026110400_c032', action_index=step, point='palm',
        position_error_m=float(np.linalg.norm(fk['base_link'][:3,3]-logged['observation_state'][step,28:31]))))
    assert max(c['position_error_m'] for c in checks) < 1e-6
    write_csv(out / 'joint_order_fk_verification.csv', checks)
    singular = np.linalg.svd(output_w, compute_uv=False)
    np.testing.assert_array_less(0, singular)
    max_log_error = float(max(np.max(abs(h21-logged['h2'][step])), np.max(abs(mu1-logged['mu'][step]))))
    summary = dict(passed=True, output_weight_shape=list(output_w.shape),
        rank=int(np.linalg.matrix_rank(output_w)), linear_nullspace_dimension=64-int(np.linalg.matrix_rank(output_w)),
        condition_number=float(singular[0]/singular[-1]), example_object=11477, example_seed=2026110400,
        example_action_index=step, same_physical_state_counterfactual=True,
        source_condition=record['condition'], source=str(source), source_sha256=sha(source),
        max_replay_log_error=max_log_error,
        max_last_layer_sum_error=float(np.max(abs(last_contributions.sum(1)-delta_mu))),
        max_h1_path_sum_error=float(np.max(abs(h1_contributions.sum(1)-delta_mu))),
        max_sae_path_sum_error=float(np.max(abs(sae_contributions.sum(1)+residual_mu-delta_mu))),
        projection_and_rounding_residual_norm=float(np.linalg.norm(residual_h)),
        actual_decoder_ray_scale=float(ray_scale), exact_edit_reproduction=True,
        max_fk_position_error_m=max(c['position_error_m'] for c in checks),
        fk_checked_points=len(checks), requested_z_coordinates_changed=np.flatnonzero(dz).tolist(),
        reencoded_z_coordinates_changed=np.flatnonzero(reencoded_dz).tolist(),
        note='Exact actor decomposition; actuator targets are not achieved motion or task success.')
    dump(out / 'verification.json', summary)
    backup = PROJECT_ROOT.parents[1] / 'research/embodied_sae/runs/laptop_resume_bundle/third_party/dexart-release'
    source_files = [actor_path, campaign_path, model_path/'weights.npz', model_path/'model.json',
        backup/'dexart/env/rl_env/base.py', backup/'dexart/env/create_env.py',
        backup/'dexart/utils/kinematics_helper.py', backup/'dexart/utils/common_robot_utils.py',
        online/'package/bundle/online_statistics.npz', online/'package/bundle/online_intervention.json',
        PROJECT_ROOT/'temp_scripts/online_bidirectional.py', PROJECT_ROOT/'temp_scripts/constrained_intervention.py']
    dump(out / 'source_hashes.json', dict(files={str(p):sha(p) for p in source_files},
        urdf_zip='D:/homework/dexart_assets.zip', urdf_member=member, urdf_member_sha256=urdf_sha,
        frozen_trajectory_id=t.tid, frozen_trajectory_provenance=t.provenance))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('example action channels:', json.dumps([effects[i] for i in [0,5,6,18]], ensure_ascii=False))


if __name__ == '__main__':
    main()
