"""Portable multi-coordinate suppression and post-projection matched controls."""
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from .constrained_intervention import project_delta, match_projected_norm, stable_random_directions
else:
    from constrained_intervention import project_delta, match_projected_norm, stable_random_directions


def group_definitions(mode):
    groups = {'top%d' % n: list(range(n)) for n in (1, 2, 3, 4, 6)}
    if mode == 'factorial':
        groups.update(A=[0, 1], B=[2, 3], C=[4, 5], AB=[0, 1, 2, 3],
                      AC=[0, 1, 4, 5], BC=[2, 3, 4, 5], ABC=list(range(6)))
        # Top2/4/6 already implement A/AB/ABC.
        groups = {k: v for k, v in groups.items() if k not in ('A', 'AB', 'ABC')}
    return groups


class CombinedEditor:
    def __init__(self, bundle, numeric):
        self.meta = json.loads((Path(bundle) / 'combined_intervention.json').read_text(encoding='utf-8'))
        with np.load(Path(bundle) / 'combined_directions.npz', allow_pickle=False) as data:
            self.directions = {k: data[k].copy() for k in data.files}
        self.numeric = numeric
        self._cache_key = None

    def _context(self, h):
        key = h.tobytes()
        if key == self._cache_key:
            return self._cache
        candidates, codes = {}, {}
        cap = self.meta['relative_norm_cap'] * np.linalg.norm(h)
        for name, info in self.meta['models'].items():
            model = self.numeric[name]
            z = model.infer(h)[0][0]
            codes[name] = z
            decoder = model.w['decoder_weight'] * model.w['scale'][:, None]
            for group, ranks in self.meta['groups'].items():
                indices = np.asarray(info['opening_coordinates'])[ranks]
                raw = decoder[:, indices] @ (-self.meta['suppression_fraction'] * z[indices])
                delta = project_delta(h, raw)
                native_norm = float(np.linalg.norm(delta))
                delta *= min(1., cap / max(native_norm, 1e-30))
                candidates[name, group] = (indices, raw, delta, native_norm)
        positive = [np.linalg.norm(v[2]) for v in candidates.values() if np.linalg.norm(v[2]) > 1e-9]
        self._cache = (codes, candidates, min(positive) if positive else 0., len(positive))
        self._cache_key = key
        return self._cache

    def edit(self, h, model_name, group, dose, control, random_key):
        h = np.asarray(h, dtype=np.float64)
        if h.shape != (64,) or not np.isfinite(h).all() or (h < 0).any():
            raise ValueError('Expected finite nonnegative h1 with 64 coordinates')
        if dose not in ('nominal', 'common_norm'):
            raise ValueError('Unknown dose')
        codes, candidates, common, supported = self._context(h)
        indices, raw, native, native_norm = candidates[model_name, group]
        z = codes[model_name]
        budget = float(np.linalg.norm(native)) if dose == 'nominal' else float(common)
        delta = np.zeros_like(h)
        status = 'inactive_or_no_feasible_budget'
        control_indices = []
        if np.linalg.norm(native) > 1e-9 and budget > 1e-9:
            if control == 'target' and dose == 'nominal':
                delta, status = native, 'ok'
            else:
                direction = raw
                if control.startswith('random_group'):
                    seed = int.from_bytes(hashlib.sha256((str(random_key) + '|' + model_name + '|' + control).encode()).digest()[:8], 'little')
                    rng = np.random.default_rng(seed)
                    pool = np.setdiff1d(np.arange(len(z)), self.meta['models'][model_name]['opening_coordinates'])
                    control_indices = rng.choice(pool, len(indices), replace=False).tolist()
                    model = self.numeric[model_name]
                    decoder = model.w['decoder_weight'] * model.w['scale'][:, None]
                    direction = decoder[:, control_indices] @ (-z[control_indices])
                elif control.startswith('random'):
                    direction = stable_random_directions(random_key, len(h), 3)[int(control[6:])]
                elif control == 'ridge':
                    direction = -self.directions['ridge_oriented'][:, :len(indices)].sum(axis=1)
                elif control != 'target':
                    raise ValueError('Unknown control')
                delta, status = match_projected_norm(h, direction, budget)
        emitted = np.maximum(h + delta, 0).astype(np.float32) if status == 'ok' else h.astype(np.float32)
        actual = emitted.astype(float) - h
        after = self.numeric[model_name].infer(emitted)[0][0]
        other = np.ones(len(z), dtype=bool)
        other[indices] = False
        details = {
            'status': status, 'budget': budget, 'actual_norm': float(np.linalg.norm(actual)),
            'norm_error': float(abs(np.linalg.norm(actual) - budget)),
            'nominal_projected_norm': native_norm,
            'cap_active': bool(native_norm > self.meta['relative_norm_cap'] * np.linalg.norm(h)),
            'common_supported_candidates': supported, 'common_total_candidates': len(candidates),
            'target_indices': indices.tolist(), 'control_indices': control_indices,
            'target_before': float(np.linalg.norm(z[indices])),
            'target_after': float(np.linalg.norm(after[indices])),
            'nontarget_code_l2': float(np.linalg.norm((after - z)[other])),
            'z_before': z.tolist(), 'z_after': after.tolist(),
            'negative_fraction': float(np.mean(emitted < 0))}
        return emitted, details
