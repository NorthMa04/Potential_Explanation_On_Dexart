"""Nonnegative residual-preserving direction edits and post-projection norm controls.

This module is portable to Python 3.8. Re-encoding audits actual feature changes;
projected decoder edits are never assumed to be exact latent ablations.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def project_delta(h, delta):
    h, delta = np.asarray(h, float), np.asarray(delta, float)
    if h.shape != delta.shape or h.ndim != 1 or not np.isfinite(h).all() or not np.isfinite(delta).all() or (h < 0).any():
        raise ValueError('Need matching finite vectors and nonnegative original h')
    return np.maximum(h + delta, 0.) - h


def match_projected_norm(h, direction, budget, atol=1e-10):
    """Find a nonnegative projected ray with the requested ACTUAL L2 norm.

    Negative-only rays have finite capacity; failure is explicit, never silently
    called matched. The zero original coordinates may only move nonnegatively.
    """
    h, direction = np.asarray(h, float), np.asarray(direction, float)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('Invalid budget')
    if budget <= atol:
        return np.zeros_like(h), 'zero_budget'
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-15:
        return np.zeros_like(h), 'zero_direction'
    direction = direction / length
    low, high = 0., max(budget, 1e-8)
    for _ in range(60):
        delta = project_delta(h, high * direction)
        if np.linalg.norm(delta) >= budget - atol:
            break
        high *= 2
    else:
        return np.zeros_like(h), 'insufficient_feasible_capacity'
    for _ in range(65):
        mid = (low + high) / 2
        delta = project_delta(h, mid * direction)
        if np.linalg.norm(delta) < budget:
            low = mid
        else:
            high = mid
    delta = project_delta(h, high * direction)
    if abs(np.linalg.norm(delta) - budget) > max(atol * 10, budget * 1e-8):
        return np.zeros_like(h), 'norm_match_failed'
    return delta, 'ok'


def stable_random_directions(key, dimension, count=3):
    seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(count, dimension))
    return draws / np.linalg.norm(draws, axis=1, keepdims=True)


class ConstrainedEditor:
    def __init__(self, bundle, numeric):
        bundle = Path(bundle)
        self.meta = json.loads((bundle / 'intervention.json').read_text())
        with np.load(bundle / 'directions.npz') as arrays:
            self.directions = {k: arrays[k].copy() for k in arrays.files}
        self.numeric = numeric

    def trigger_eligible(self, h):
        # Consensus support gives all three SAE seeds exactly the same states.
        for seed in self.meta['sae_seeds']:
            item = self.meta['models']['assisted_seed%d' % seed]
            z = self.numeric['assisted_seed%d' % seed].infer(h)[0][0, item['feature']]
            if not item['q80'] <= z < item['q99']:
                return False
        return True

    def edit(self, h, model_name, operation, control, random_key):
        h = np.asarray(h, dtype=float).reshape(-1)
        model, info = self.numeric[model_name], self.meta['models'][model_name]
        z = model.infer(h)[0][0]
        j = info['feature']
        if operation == 'suppress':
            requested = -float(z[j])
        elif operation == 'half_suppress':
            requested = -.5 * float(z[j])
        elif operation == 'enhance':
            requested = min(.5 * info['active_std'], info['q99'] - float(z[j]))
        else:
            raise ValueError('Unknown operation')
        sign = -1. if operation.endswith('suppress') else 1.
        raw_direction = self.directions[model_name + '__target']
        target_delta = project_delta(h, raw_direction * requested)
        nominal_norm = np.linalg.norm(target_delta)
        cap = self.meta['relative_norm_cap'] * np.linalg.norm(h)
        if nominal_norm > cap:
            target_delta *= cap / nominal_norm
        budget = float(np.linalg.norm(target_delta))
        details = {'feature': j, 'requested_code_change': requested,
                   'nominal_unprojected_norm': float(np.linalg.norm(raw_direction * requested)),
                   'budget': budget, 'cap_active': bool(nominal_norm > cap)}
        if (operation == 'enhance' and requested <= 1e-9) or z[j] <= 1e-9 or budget <= 1e-9:
            details['status'] = 'inactive_or_no_feasible_budget'
            return h.copy(), details
        if control == 'target':
            delta, status = target_delta, 'ok'
        else:
            if control.startswith('random'):
                index = int(control[6:])
                direction = stable_random_directions(random_key, len(h), self.meta['random_directions'])[index] * sign
            elif control in ('raw', 'pure', 'ridge'):
                direction = self.directions[model_name + '__' + control] * sign
            else:
                raise ValueError('Unknown control')
            delta, status = match_projected_norm(h, direction, budget)
        # Use the actual float32 emission in all reported realized-change audits.
        emitted = np.maximum(h + delta, 0).astype(np.float32)
        actual_delta = emitted.astype(float) - h
        z_after = model.infer(emitted)[0][0]
        details.update(status=status, actual_norm=float(np.linalg.norm(actual_delta)),
                       norm_error=float(abs(np.linalg.norm(actual_delta) - budget)),
                       target_before=float(z[j]), target_after=float(z_after[j]),
                       actual_code_change=float(z_after[j] - z[j]),
                       code_retention=float((z_after[j] - z[j]) / requested) if abs(requested) > 1e-12 else 0.,
                       nontarget_code_l2=float(np.linalg.norm(np.delete(z_after - z, j))),
                       negative_fraction=float(np.mean(emitted < 0)),
                       projected_direction_cosine=float(np.dot(actual_delta, raw_direction * sign) /
                            max(np.linalg.norm(actual_delta) * np.linalg.norm(raw_direction), 1e-15)))
        return emitted, details
