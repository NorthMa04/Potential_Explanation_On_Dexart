"""Portable bounded bidirectional group edits for the 2026-11-04 pilot.

SAE-coordinate increase is not a claim about physical opening direction. Bounds
are frozen from the original train local window. Old experiments are unchanged.
"""
import json
from pathlib import Path

import numpy as np

if __package__:
    from .constrained_intervention import project_delta, match_projected_norm, stable_random_directions
    from .physical_runtime import NumericSAE, actor_tail
else:
    from constrained_intervention import project_delta, match_projected_norm, stable_random_directions
    from physical_runtime import NumericSAE, actor_tail


class OnlineBidirectionalEditor:
    CONTROLS = ('decrease', 'bounded_increase', 'random0', 'random1', 'random2',
                'ridge_decrease', 'ridge_increase', 'noop')
    DOSES = {'full': 1., 'half': .5, 'nominal': 1., 'common_norm': .5}

    def __init__(self, bundle, numeric):
        bundle = Path(bundle)
        self.meta = json.loads((bundle / 'online_intervention.json').read_text(encoding='utf-8'))
        with np.load(bundle / 'online_statistics.npz', allow_pickle=False) as arrays:
            self.stats = {k: arrays[k].copy() for k in arrays.files}
        with np.load(bundle / 'combined_directions.npz', allow_pickle=False) as arrays:
            self.ridge = arrays['ridge_oriented'].copy()
        self.numeric = numeric
        self._cache_key = None

    def _context(self, h, name, group):
        key = (h.tobytes(), name, group)
        if key == self._cache_key:
            return self._cache
        model = self.numeric[name]
        z = model.infer(h)[0][0]
        n = self.meta['groups'][group]
        indices = np.asarray(self.meta['models'][name]['opening_coordinates'][:n], dtype=int)
        decoder = model.w['decoder_weight'] * model.w['scale'][:, None]
        std = self.stats[name + '__active_std'][indices]
        upper = self.stats[name + '__active_q99'][indices]
        request_up = np.where(z[indices] > 0, np.maximum(0., np.minimum(.5 * std, upper - z[indices])), 0.)
        decrease = decoder[:, indices] @ (-z[indices])
        increase = decoder[:, indices] @ request_up
        down_norm = float(np.linalg.norm(project_delta(h, decrease)))
        up_norm = float(np.linalg.norm(project_delta(h, increase)))
        cap = self.meta['relative_norm_cap'] * float(np.linalg.norm(h))
        budget = min(down_norm, up_norm, cap)
        self._cache_key = key
        self._cache = (z, indices, decrease, increase, request_up, down_norm, up_norm, cap, budget)
        return self._cache

    def edit(self, h, model_name, group, dose, control, random_key):
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (64,) or not np.isfinite(h).all() or (h < 0).any():
            raise ValueError('Expected a finite, nonnegative 64-dimensional h1')
        if dose not in self.DOSES or control not in self.CONTROLS:
            raise ValueError('Unknown online dose/control')
        if group not in self.meta['groups'] or model_name not in self.meta['models']:
            raise ValueError('Unknown online model/group')
        z, indices, down, up, request_up, down_norm, up_norm, cap, common = self._context(h, model_name, group)
        budget = self.DOSES[dose] * common
        if control == 'decrease':
            direction = down
        elif control == 'bounded_increase':
            direction = up
        elif control.startswith('random'):
            direction = stable_random_directions(random_key, len(h), 3)[int(control[6:])]
        elif control.startswith('ridge_'):
            sign = -1. if control == 'ridge_decrease' else 1.
            direction = sign * self.ridge[:, :len(indices)].sum(axis=1)
        else:
            direction = np.zeros_like(h)
        if control == 'noop':
            delta, status = np.zeros_like(h), 'ok'
            budget = 0.
        elif budget <= 1e-9:
            delta, status = np.zeros_like(h), 'inactive_or_no_feasible_budget'
        else:
            delta, status = match_projected_norm(h, direction, budget)
        emitted = np.maximum(h + delta, 0).astype(np.float32) if status == 'ok' else h.astype(np.float32)
        actual = emitted.astype(np.float64) - h
        after = self.numeric[model_name].infer(emitted)[0][0]
        other = np.ones(len(z), dtype=bool)
        other[indices] = False
        detail = {
            'status': status, 'budget': float(budget), 'actual_norm': float(np.linalg.norm(actual)),
            'norm_error': float(abs(np.linalg.norm(actual) - budget)),
            'target_before': float(np.linalg.norm(z[indices])),
            'target_after': float(np.linalg.norm(after[indices])),
            'nontarget_code_l2': float(np.linalg.norm((after - z)[other])),
            'cap_active': bool(min(down_norm, up_norm) > cap),
            'common_supported_candidates': int(common > 1e-9) * 2,
            'common_total_candidates': 2, 'target_indices': indices.tolist(),
            'control_indices': [], 'z_before': z.tolist(), 'z_after': after.tolist(),
            'negative_fraction': float(np.mean(emitted < 0)),
            'requested_increase': request_up.tolist(),
            'requested_decrease': (-z[indices]).tolist(),
            'native_decrease_norm': down_norm, 'native_increase_norm': up_norm,
            'protocol': self.meta['version'], 'operation': control,
        }
        return emitted, detail


# Allows a copied physical hook to replace its import without other API changes.
CombinedEditor = OnlineBidirectionalEditor


def verify_online_bundle(bundle):
    """Check frozen reference emissions, feasibility, codes and actor outputs."""
    bundle = Path(bundle)
    numeric = {d.name: NumericSAE(d) for d in sorted((bundle / 'models').iterdir())}
    editor = OnlineBidirectionalEditor(bundle, numeric)
    rows = json.loads((bundle / 'online_reference.json').read_text(encoding='utf-8'))
    maxima = {'h': 0., 'mu': 0., 'z': 0., 'norm_error': 0.}
    with np.load(bundle / 'online_reference.npz', allow_pickle=False) as ref, np.load(bundle / 'actor.npz', allow_pickle=False) as actor:
        actual = []
        for i, row in enumerate(rows):
            h, detail = editor.edit(ref['h'][i], row['model'], row['group'], row['dose'], row['control'], row['random_key'])
            if detail['status'] != row['status']:
                raise RuntimeError('Cross-runtime feasibility mismatch at row %d' % i)
            maxima['h'] = max(maxima['h'], float(np.max(abs(h - ref['emitted'][i]))))
            maxima['z'] = max(maxima['z'], float(np.max(abs(np.asarray(detail['z_after']) - ref['z_after'][i]))))
            np.testing.assert_allclose(h, ref['emitted'][i], rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(detail['z_after'], ref['z_after'][i], rtol=2e-5, atol=2e-4)
            if detail['status'] == 'ok':
                maxima['norm_error'] = max(maxima['norm_error'], detail['norm_error'])
                if detail['norm_error'] > 2e-6:
                    raise RuntimeError('Actual emitted norm mismatch')
            actual.append(h)
        mu = actor_tail({k: actor[k] for k in actor.files}, np.asarray(actual))
        maxima['mu'] = float(np.max(abs(mu - ref['mu'])))
        np.testing.assert_allclose(mu, ref['mu'], rtol=2e-5, atol=2e-5)
    return {'passed': True, 'rows': len(rows), 'max_abs': maxima,
            'scope': 'single-state numeric inference, reference h/code/actor emissions and feasibility'}
