"""Development-object local reconstruction pilot; original policy parameters stay frozen."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from physical_runtime import NumericSAE, verify_bundle


def dump(path, obj):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    pending.replace(path)


def capture(env, obs):
    attributes = {}
    for key, value in env.__dict__.items():
        if isinstance(value, (int, float, bool, str, np.number, np.ndarray)) or value is None:
            attributes[key] = copy.deepcopy(value)
        elif type(value).__name__ == 'Pose':
            attributes[key] = type(value)(value.p.copy(), value.q.copy())
    return {'scene': copy.deepcopy(env.scene.pack()), 'attributes': attributes,
            'imaginations': copy.deepcopy(env.imaginations), 'last_obs': copy.deepcopy(env.last_obs),
            'drives': [(a.get_drive_target().copy(), a.get_drive_velocity_target().copy(), a.get_qf().copy())
                       for a in (env.robot, env.instance)],
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'env_rng': env.np_random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(), 'obs': copy.deepcopy(obs)}


def restore(env, saved):
    env.scene.unpack(copy.deepcopy(saved['scene']))
    for key, value in saved['attributes'].items():
        if type(value).__name__ == 'Pose':
            value = type(value)(value.p.copy(), value.q.copy())
        else:
            value = copy.deepcopy(value)
        setattr(env, key, value)
    env.imaginations = copy.deepcopy(saved['imaginations'])
    env.last_obs = copy.deepcopy(saved['last_obs'])
    for a, (pos, vel, force) in zip((env.robot, env.instance), saved['drives']):
        a.set_drive_target(pos)
        a.set_drive_velocity_target(vel)
        a.set_qf(force)
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    env.np_random.set_state(saved['env_rng'])
    torch.set_rng_state(saved['torch'])
    torch.cuda.set_rng_state_all(saved['cuda'])
    return copy.deepcopy(saved['obs'])


def measure(env):
    limits = env.joint_limits_dict[str(env.index)]
    opening = np.sign(limits['middle'] - limits['left']) * (env.instance.get_qpos()[0] - limits['left'] - .2)
    return {'step': int(env.current_step), 'opening': float(opening),
            'distance': float(np.linalg.norm(env.palm_pose.p - env.handle_pose.p)),
            'contact': bool(np.any(env.robot_object_contact)), 'strict_contact': bool(env.is_contact),
            'contacts': np.array(env.robot_object_contact, copy=True),
            'robot_qpos': env.robot.get_qpos().copy(), 'object_qpos': env.instance.get_qpos().copy(),
            'robot_qvel': env.robot.get_qvel().copy(), 'object_qvel': env.instance.get_qvel().copy(),
            'success': bool(env.is_eval_done), 'progress': float(env.progress)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('/hy-tmp/embodied_sae/third_party/dexart-release'))
    p.add_argument('--object', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--single-condition')
    p.add_argument('--reference', type=Path)
    p.add_argument('--model-name')
    p.add_argument('--replacement', choices=('raw', 'relu'), default='raw')
    p.add_argument('--duration', type=int, default=0)
    p.add_argument('--feature-edit', action='store_true')
    p.add_argument('--group-edit', action='store_true')
    p.add_argument('--group-name')
    p.add_argument('--dose', choices=('nominal', 'common_norm'))
    p.add_argument('--operation', choices=('half_suppress', 'suppress', 'enhance'))
    p.add_argument('--control', choices=('target', 'random0', 'random1', 'random2', 'raw', 'pure', 'ridge',
                                        'random_group0', 'random_group1', 'random_group2'))
    args = p.parse_args()
    if args.group_edit and (args.feature_edit or not args.single_condition):
        p.error('Group edits require a single independent-process condition')
    if args.group_edit and args.model_name and (not args.group_name or not args.dose or not args.control):
        p.error('Group treatment requires group-name, dose and control')
    if args.group_edit and args.model_name and args.duration not in (1, 5, -1):
        p.error('Group duration must be 1, 5 or -1')
    plan = json.loads(args.plan.read_text())
    assert args.object in plan['objects'] and args.seed in plan['environment_seeds']
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo))
    from dexart.env.create_env import create_env
    from dexart.env.task_setting import RANDOM_CONFIG
    from examples.train import get_3d_policy_kwargs
    from stable_baselines3 import PPO
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    numeric = {d.name: NumericSAE(d) for d in (args.bundle / 'models').iterdir()}
    editor = None
    group_editor = None
    if args.feature_edit:
        from constrained_intervention import ConstrainedEditor
        editor = ConstrainedEditor(args.bundle, numeric)
    if args.group_edit:
        from combined_intervention import CombinedEditor
        group_editor = CombinedEditor(args.bundle, numeric)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = RANDOM_CONFIG['laptop']
    env = create_env(task_name='laptop', use_visual_obs=True, use_gui=False, is_eval=True,
                     pc_noise=True, pc_seg=True, index=[args.object], img_type='robot',
                     rand_pos=cfg['rand_pos'], rand_degree=cfg['rand_degree'])
    env.seed(args.seed)
    checkpoint = args.repo / 'assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip'
    model = PPO.load(str(checkpoint), env, 'cuda:0',
                     policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                     check_obs_space=False, force_load=True)
    model.set_random_seed(args.seed)
    model.policy.eval()
    for v in model.policy.parameters():
        v.requires_grad_(False)
    with np.load(args.bundle / 'actor.npz') as actor:
        for key in actor.files:
            np.testing.assert_array_equal(model.policy.state_dict()[key].cpu().numpy(), actor[key])
    weight_hash_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def full_policy_hash():
        digest = hashlib.sha256()
        for key, tensor in sorted(model.policy.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    full_policy_before = full_policy_hash()
    ctx = {'condition': 'original', 'model': None, 'start': 0, 'duration': 0, 'capture': {}}
    reference = None
    prefix_errors = {}
    replay_gate_end = -1
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as data:
            reference = {k: data[k].copy() for k in data.files}
        hits = np.flatnonzero(reference['feature_trigger_eligible'] if editor is not None
                              else reference['opening'] >= plan['trigger_opening_rad'])
        if not len(hits):
            dump(args.out / 'report.json', {'status': 'reference_no_trigger'})
            return
        ctx.update(start=int(hits[0]), condition=args.replacement,
                   model=args.model_name, duration=args.duration)
        if args.group_edit and args.duration == -1:
            ctx['duration'] = env.horizon
        replay_gate_end = ctx['start'] if args.model_name else env.horizon

    def check_prefix(key, actual):
        if reference is None or env.current_step > replay_gate_end:
            return
        expected = reference[key][env.current_step]
        error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
        prefix_errors[key] = max(prefix_errors.get(key, 0.), error)
        if error > plan['restore_repeat_atol']:
            dump(args.out / 'prefix_failure.json', {'status': 'fresh_prefix_mismatch_no_valid_comparison',
                  'step': int(env.current_step), 'key': key, 'max_abs': error, 'errors': prefix_errors})
            raise RuntimeError('Fresh-process prefix gate failed: ' + key)

    def h1_hook(_m, _i, output):
        h = output.detach().cpu().numpy().copy()
        check_prefix('h_original', h[0])
        emitted = h.copy()
        active = ctx['start'] <= env.current_step < ctx['start'] + ctx['duration']
        applied = bool(ctx['model'] and active)
        if group_editor is not None:
            ctx['capture'].update(scheduled=applied, edit_status_code=0, edit_budget=0.,
                edit_actual_norm=0., edit_norm_error=0., edit_target_before=0., edit_target_after=0.,
                edit_nontarget_code_l2=0., edit_cap_active=False, edit_common_supported_candidates=0,
                edit_z_before=np.zeros(64), edit_z_after=np.zeros(64))
        if editor is not None:
            ctx['capture']['feature_trigger_eligible'] = bool(measure(env)['opening'] >= plan['trigger_opening_rad']
                                                            and editor.trigger_eligible(h[0]))
            defaults = {'scheduled': bool(applied), 'edit_status_code': 0, 'edit_budget': 0.,
                        'edit_actual_norm': 0., 'edit_norm_error': 0., 'edit_target_before': 0.,
                        'edit_target_after': 0., 'edit_actual_code_change': 0.,
                        'edit_requested_code_change': 0., 'edit_code_retention': 0.,
                        'edit_nontarget_code_l2': 0., 'edit_cap_active': False}
            ctx['capture'].update(defaults)
        if ctx['model'] and active:
            if group_editor is not None:
                value, detail = group_editor.edit(h[0], ctx['model'], args.group_name, args.dose, args.control,
                                                 '%d|%d|%d' % (args.object, args.seed, ctx['start']))
                applied = detail['status'] == 'ok'
                emitted = value[None, :] if applied else h.copy()
                reasons = {'ok': 1, 'inactive_or_no_feasible_budget': 2, 'zero_budget': 3,
                           'zero_direction': 4, 'insufficient_feasible_capacity': 5, 'norm_match_failed': 6}
                ctx['capture']['edit_status_code'] = reasons[detail['status']]
                for key in ('budget', 'actual_norm', 'norm_error', 'target_before', 'target_after',
                            'nontarget_code_l2', 'cap_active', 'common_supported_candidates', 'z_before', 'z_after'):
                    ctx['capture']['edit_' + key] = detail[key]
            elif editor is not None:
                value, detail = editor.edit(h[0], ctx['model'], args.operation, args.control,
                                           '%d|%d|%d' % (args.object, args.seed, ctx['start']))
                applied = detail['status'] == 'ok'
                emitted = value[None, :].astype(np.float32) if applied else h.copy()
                ctx['capture']['edit_status_code'] = 1 if applied else 2
                for key in ('budget', 'actual_norm', 'norm_error', 'target_before', 'target_after',
                            'actual_code_change', 'requested_code_change', 'code_retention',
                            'nontarget_code_l2', 'cap_active'):
                    ctx['capture']['edit_' + key] = detail.get(key, 0.)
            else:
                z, rec, pred = numeric[ctx['model']].infer(h)
                emitted = np.maximum(rec, 0) if ctx['condition'] == 'relu' else rec
                emitted = emitted.astype(np.float32)
                ctx['capture']['unprojected_rec'] = rec[0]
        ctx['capture'].update(h_original=h[0], h_applied=emitted[0], applied=applied)
        if applied:
            return torch.as_tensor(emitted, device=output.device, dtype=output.dtype)
        return None

    handles = [model.policy.mlp_extractor.policy_net[1].register_forward_hook(h1_hook)]
    def mu_hook(_m, _i, output):
        ctx['capture']['mu'] = output.detach().cpu().numpy()[0].copy()
    handles.append(model.policy.action_net.register_forward_hook(mu_hook))
    image_dirs = set()

    def step(obs, records, name):
        before = measure(env)
        for key, value in before.items():
            check_prefix(key, value)
        for key, value in obs.items():
            check_prefix('obs__' + key, value)
        ctx['capture'] = {}
        batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        action = model.predict(observation=batched, deterministic=True)[0].reshape(env.action_space.shape)
        row = dict(before)
        row.update({k: copy.deepcopy(v) for k, v in ctx['capture'].items() if k != 'unprojected_rec'})
        row['negative_fraction'] = float(np.mean(row['h_applied'] < 0))
        row['delta_norm'] = float(np.linalg.norm(row['h_applied'] - row['h_original']))
        row['observation_state'] = np.asarray(obs['state']).copy()
        if reference is not None and (ctx['model'] is None or env.current_step < ctx['start']):
            check_prefix('action', action)
            check_prefix('mu', row['mu'])
        # Record the policy's full noisy visual input for reproducibility.
        for key, value in obs.items():
            row['obs__' + key] = np.asarray(value).copy()
        if name == 'natural':
            for key, sae in numeric.items():
                z, rec, pred = sae.infer(row['h_original'])
                row['z__' + key], row['prediction__' + key] = z[0], pred[0]
        obs, reward, done, _ = env.step(action)
        for key, value in measure(env).items():
            row['post_' + key] = value
        row.update(action=action.copy(), reward=float(reward), done=bool(done))
        for key, value in row.items():
            records[key].append(value)
        if name in plan['render_conditions'] and before['step'] % 3 == 0:
            from PIL import Image
            image_dir = args.out / 'frames' / name
            if name not in image_dirs:
                image_dir.mkdir(parents=True, exist_ok=True)
                image_dirs.add(name)
            env.scene.update_render()
            camera = env.cameras[sorted(env.cameras)[-1]]
            camera.take_picture()
            color = np.clip(camera.get_float_texture('Color')[..., :3], 0, 1)
            Image.fromarray((color * 255).astype(np.uint8)).save(image_dir / ('%04d.png' % before['step']))
        return obs, done

    def save(records, name):
        arrays = {k: np.stack(v) for k, v in records.items()}
        assert arrays['done'][-1], 'Partial rollout'
        assert all(np.isfinite(a).all() for a in arrays.values())
        np.savez_compressed(args.out / (name + '.npz'), **arrays)
        summary = {'name': name, 'steps': len(arrays['step']), 'success': bool(np.any(arrays['post_success'])),
                   'final_opening_rad': float(arrays['post_opening'][-1]),
                   'max_opening_rad': float(np.max(arrays['post_opening'])),
                   'applied_steps': int(np.sum(arrays['applied'])),
                   'max_negative_fraction': float(np.max(arrays['negative_fraction'])),
                   'return': float(np.sum(arrays['reward']))}
        if group_editor is not None:
            summary.update(scheduled_steps=int(arrays['scheduled'].sum()),
                           failed_edit_steps=int(np.sum(arrays['scheduled'] & ~arrays['applied'])),
                           cumulative_delta_norm=float(arrays['delta_norm'].sum()))
        print(json.dumps(summary), flush=True)
        return arrays, summary

    # Reseed after policy setup, matching the official collector's episode loop.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.seed(args.seed)
    obs = env.reset()
    if args.single_condition:
        records = defaultdict(list)
        while True:
            obs, done = step(obs, records, args.single_condition)
            if done:
                break
        arrays, summary = save(records, args.single_condition)
        if reference is not None and ctx['model'] is None:
            for key in ('step', 'opening', 'post_opening', 'action', 'mu', 'done', 'post_success'):
                np.testing.assert_allclose(arrays[key], reference[key], atol=plan['restore_repeat_atol'], rtol=0)
        # Directly verify policy parameters after each independent-process rollout.
        with np.load(args.bundle / 'actor.npz') as actor:
            for key in actor.files:
                np.testing.assert_array_equal(model.policy.state_dict()[key].cpu().numpy(), actor[key])
        assert weight_hash_before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert full_policy_before == full_policy_hash()
        dump(args.out / 'report.json', {'status': 'fresh_process_rollout_completed',
             'object': args.object, 'seed': args.seed, 'summary': summary,
             'prefix_gate_passed': True if reference is not None else None,
             'prefix_max_abs': prefix_errors, 'trigger_step': ctx['start'] if reference is not None else None,
             'reference': str(args.reference) if args.reference else None,
             'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
             'checkpoint_sha256': weight_hash_before, 'actor_parameters_unchanged': True,
             'full_policy_state_sha256': full_policy_before, 'full_policy_state_unchanged': True,
             'source_plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
             'feature_edit': args.feature_edit, 'operation': args.operation, 'control': args.control,
             'model_name': args.model_name,
             'group_edit': args.group_edit, 'group_name': args.group_name, 'dose': args.dose,
             'duration': args.duration,
             'combined_metadata_sha256': hashlib.sha256((args.bundle / 'combined_intervention.json').read_bytes()).hexdigest() if group_editor else None,
             'intervention_metadata_sha256': hashlib.sha256((args.bundle / 'intervention.json').read_bytes()).hexdigest() if editor is not None else None})
        return
    natural_records, saved, trigger = defaultdict(list), None, None
    while True:
        before = measure(env)
        if saved is None and before['opening'] >= plan['trigger_opening_rad'] and not env.is_done():
            saved, trigger = capture(env, obs), before
        obs, done = step(obs, natural_records, 'natural')
        if done:
            break
    natural, natural_summary = save(natural_records, 'natural')
    report = {'object': args.object, 'seed': args.seed, 'trigger': trigger,
              'source_plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              'checkpoint_sha256': weight_hash_before, 'conditions': [natural_summary],
              'arbitrary_state_restore_claim': False}
    if saved is None:
        report['status'] = 'no_trigger_all_outcomes_retained'
        dump(args.out / 'report.json', report)
        return
    # Convert physical vectors before serialization; original arrays remain in archives.
    report['trigger'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in trigger.items()}

    def branch(name, model_name=None, condition='original', duration=0):
        branch_obs = restore(env, saved)
        ctx.update(model=model_name, condition=condition, start=trigger['step'], duration=duration)
        records = defaultdict(list)
        while True:
            branch_obs, done = step(branch_obs, records, name)
            if done:
                break
        arrays, summary = save(records, name)
        report['conditions'].append(summary)
        return arrays

    first = branch('restore_control_1')
    second = branch('restore_control_2')
    checks = {}
    for key in ('robot_qpos', 'object_qpos', 'robot_qvel', 'object_qvel', 'action', 'mu'):
        checks[key] = float(np.max(np.abs(first[key] - second[key]))) if first[key].shape == second[key].shape else float('inf')
    # Full visual-input replay is checked separately from policy/physics values.
    for key in first:
        if key.startswith('obs__'):
            checks[key] = float(np.max(np.abs(first[key] - second[key]))) if first[key].shape == second[key].shape else float('inf')
    repeat_passed = all(v <= plan['restore_repeat_atol'] for v in checks.values())
    nat = {k: v[trigger['step']:] for k, v in natural.items()}
    natural_errors = {}
    for key in ('robot_qpos', 'object_qpos', 'action'):
        a, b = nat[key], first[key]
        n = min(len(a), len(b))
        natural_errors[key] = float(np.max(np.abs(a[:n] - b[:n])))
    report['restore_audit'] = {'repeat_passed': repeat_passed, 'full_continuation_max_abs': checks,
                               'natural_vs_restore_max_abs': natural_errors,
                               'natural_length_equal': len(nat['action']) == len(first['action']),
                               'snapshot_fields': list(saved['attributes']),
                               'scope': 'this scene and trigger only; not arbitrary-state exact restore'}
    dump(args.out / 'report.json', report)
    if not repeat_passed:
        report['status'] = 'restore_repeat_gate_failed_no_replacements'
        dump(args.out / 'report.json', report)
        return
    for seed in plan['sae_seeds']:
        for kind in ('pure', 'assisted'):
            for condition in ('raw', 'relu'):
                for duration in plan['durations']:
                    name = '%s_seed%d_%s_%d' % (kind, seed, condition, duration)
                    branch(name, '%s_seed%d' % (kind, seed), condition, duration)
                    dump(args.out / 'report.json', report)
    assert weight_hash_before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with np.load(args.bundle / 'actor.npz') as actor:
        for key in actor.files:
            np.testing.assert_array_equal(model.policy.state_dict()[key].cpu().numpy(), actor[key])
    report['actor_parameters_unchanged'] = True
    report['status'] = 'local_reconstruction_pilot_completed'
    dump(args.out / 'report.json', report)
    for handle in handles:
        handle.remove()


if __name__ == '__main__':
    main()
