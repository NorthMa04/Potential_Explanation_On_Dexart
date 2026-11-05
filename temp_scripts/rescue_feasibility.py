"""Fixed-input clean/corrupt/feature-restored action diagnostic, never env.step.

Clean codes are an experimental oracle, not an available deployment input.
Synthetic point corruption is not a physical state intervention.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from constrained_intervention import project_delta, match_projected_norm, stable_random_directions
from physical_runtime import NumericSAE, actor_tail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('/hy-tmp/embodied_sae/third_party/dexart-release'))
    p.add_argument('--dataset', type=Path, default=Path('/hy-tmp/embodied_sae/runs/laptop_dataset_001'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    assert all(c['split'] == 'train' for c in cases)
    sys.path.insert(0, str(args.repo))
    import gym
    from stable_baselines3 import PPO
    from examples.train import get_3d_policy_kwargs
    first = cases[0]
    def archive(case):
        return args.dataset / (case['trajectory'].replace('__', '/') + '.npz')
    def observation(case):
        with np.load(archive(case)) as data:
            return {k[5:]: data[k][case['state_index']].copy() for k in data.files if k.startswith('obs__')}
    first_obs = observation(first)
    class ReadOnlyEnv(gym.Env):
        action_space = gym.spaces.Box(-1., 1., shape=(22,), dtype=np.float32)
        observation_space = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=v.dtype) for k, v in first_obs.items()})
        def reset(self): return {k: v.copy() for k, v in first_obs.items()}
        def seed(self, seed=None): return [seed]
        def step(self, action): raise RuntimeError('No action execution allowed in this diagnostic')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    model = PPO.load(str(args.repo / 'assets/rl_checkpoints/laptop/laptop_nopretrain_0.zip'),
                     ReadOnlyEnv(), 'cuda:0', policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                     check_obs_space=False, force_load=True)
    model.policy.eval()
    for parameter in model.policy.parameters(): parameter.requires_grad_(False)
    capture = {}
    def hook_h(_m, _i, y): capture['h'] = y.cpu().numpy()[0].copy()
    def hook_mu(_m, _i, y): capture['mu'] = y.cpu().numpy()[0].copy()
    handles = [model.policy.mlp_extractor.policy_net[1].register_forward_hook(hook_h),
               model.policy.action_net.register_forward_hook(hook_mu)]
    def forward(obs):
        capture.clear()
        action = np.asarray(model.predict({k: v[None] for k, v in obs.items()}, deterministic=True)[0]).reshape(22)
        np.testing.assert_array_equal(action, np.clip(capture['mu'], -1, 1))
        return capture['h'].copy(), capture['mu'].copy(), action.copy()
    numeric = {d.name: NumericSAE(d) for d in (args.bundle / 'models').iterdir()}
    with np.load(args.bundle / 'actor.npz') as data: actor = {k: data[k].copy() for k in data.files}
    def parameter_hash():
        d = hashlib.sha256()
        for key, value in sorted(model.policy.state_dict().items()):
            d.update(key.encode()); d.update(value.cpu().numpy().tobytes())
        return d.hexdigest()
    before = parameter_hash()
    rows, skips = [], []
    max_clean = 0.
    for case in cases:
        obs = observation(case)
        clean_h, clean_mu, clean_action = forward(obs)
        with np.load(archive(case)) as data:
            error = float(np.max(np.abs(clean_mu - data['action_mean_raw'][case['state_index']])))
        max_clean = max(max_clean, error)
        assert error < 2e-5, ('clean replay', case, error)
        np.testing.assert_allclose(actor_tail(actor, clean_h[None])[0], clean_mu, atol=2e-5, rtol=2e-5)
        handle = obs['instance_1-seg_gt'][:, 0] > .5
        if not handle.any():
            skips.append({'case': case, 'reason': 'no_handle_points'}); continue
        for corruption in ('handle_jitter_5mm', 'handle_jitter_20mm', 'handle_dropout_half'):
            changed = {k: v.copy() for k, v in obs.items()}
            points = changed['instance_1-point_cloud']
            key = case['trajectory'] + '|' + str(case['state_index']) + '|' + corruption
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            if 'jitter' in corruption:
                sigma = .005 if '5mm' in corruption else .02
                points[handle] += rng.normal(0, sigma, size=(int(handle.sum()), 3))
            else:
                indices = np.flatnonzero(handle)
                selected = rng.choice(indices, size=max(1, len(indices) // 2), replace=False)
                donors = np.flatnonzero(~handle)
                if not len(donors):
                    skips.append({'case': case, 'corruption': corruption, 'reason': 'no_nonhandle_donors'}); continue
                source = rng.choice(donors, size=len(selected), replace=True)
                points[selected] = points[source]
                changed['instance_1-seg_gt'][selected] = changed['instance_1-seg_gt'][source]
            corrupt_h, corrupt_mu, corrupt_action = forward(changed)
            baseline_error = float(np.linalg.norm(corrupt_action - clean_action))
            for sae_seed in (0, 1, 2):
                sae = numeric['assisted_seed%d' % sae_seed]
                z_clean, z_corrupt = sae.infer(np.vstack((clean_h, corrupt_h)))[0]
                for size in (1, 4):
                    coordinates = sae.meta['probes'][1]['coordinates'][:size]
                    delta_z = np.zeros_like(z_corrupt); delta_z[coordinates] = z_clean[coordinates] - z_corrupt[coordinates]
                    raw = (sae.w['decoder_weight'] * sae.w['scale'][:, None]) @ delta_z
                    delta = project_delta(corrupt_h, raw)
                    cap = .1 * np.linalg.norm(corrupt_h)
                    if np.linalg.norm(delta) > cap: delta *= cap / np.linalg.norm(delta)
                    budget = float(np.linalg.norm(delta))
                    variants = [('feature_restore', delta, 'ok')]
                    for j, direction in enumerate(stable_random_directions(key + '|%d' % size, 64)):
                        d, status = match_projected_norm(corrupt_h, direction, budget)
                        variants.append(('random%d' % j, d, status))
                    for variant, d, status in variants:
                        edited_h = np.maximum(corrupt_h + d, 0).astype(np.float32)
                        mu = actor_tail(actor, edited_h[None])[0]
                        after_error = float(np.linalg.norm(mu.clip(-1, 1) - clean_action))
                        rows.append({'trajectory': case['trajectory'], 'state_index': case['state_index'],
                            'corruption': corruption, 'sae_seed': sae_seed, 'coordinates': size, 'variant': variant,
                            'status': status, 'baseline_action_error_l2': baseline_error, 'after_action_error_l2': after_error,
                            'absolute_error_reduction': baseline_error - after_error,
                            'error_recovery_fraction': (baseline_error - after_error) / baseline_error if baseline_error > 1e-6 else None,
                            'budget': budget, 'norm_error': float(abs(np.linalg.norm(edited_h - corrupt_h) - budget))})
    assert before == parameter_hash()
    with (args.out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'scope': 'exploratory fixed-input action diagnostic only, no physical action execution',
        'data_scope': 'train only, chosen before this diagnostic by first consensus-supported state per trajectory',
        'oracle': 'clean code accessed only as experimental restoration oracle; not deployment capability',
        'cases': len(cases), 'rows': len(rows), 'skips': skips, 'clean_replay_max_mu_abs': max_clean,
        'full_policy_unchanged': True, 'groups': []}
    for corruption in ('handle_jitter_5mm', 'handle_jitter_20mm', 'handle_dropout_half'):
        for size in (1, 4):
            for variant in ('feature_restore', 'random0', 'random1', 'random2'):
                r = [r for r in rows if r['corruption'] == corruption and r['coordinates'] == size and r['variant'] == variant and r['status'] == 'ok']
                summary['groups'].append({'corruption': corruption, 'coordinates': size, 'variant': variant,
                    'rows': len(r), 'mean_absolute_error_reduction': float(np.mean([v['absolute_error_reduction'] for v in r]))})
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for h in handles: h.remove()


if __name__ == '__main__': main()
