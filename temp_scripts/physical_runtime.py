"""Portable numeric SAE inference, compatible with simulation's Python 3.8/Torch 1.12."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class NumericSAE:
    def __init__(self, folder, device='cpu'):
        folder = Path(folder)
        self.meta = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
        with np.load(folder / 'weights.npz', allow_pickle=False) as data:
            self.w = {key: data[key].copy() for key in data.files}
        self.device = torch.device(device)
        self.t = {key: torch.as_tensor(self.w[key], device=self.device)
                  for key in ('encoder_weight', 'encoder_bias', 'decoder_weight', 'decoder_bias')}

    def infer(self, h):
        h = np.atleast_2d(np.asarray(h, dtype=np.float64))
        x = torch.as_tensor(((h - self.w['mean']) / self.w['scale']).astype(np.float32), device=self.device)
        with torch.no_grad():
            pre = F.linear(x, self.t['encoder_weight'], self.t['encoder_bias'])
            values, indices = torch.topk(pre, self.meta['top_k'], dim=-1)
            z = torch.zeros_like(pre).scatter(-1, indices, torch.relu(values))
            rec = F.linear(z, self.t['decoder_weight'], self.t['decoder_bias'])
        z = z.cpu().numpy()
        rec = rec.cpu().numpy().astype(np.float64) * self.w['scale'] + self.w['mean']
        return z, rec, self.predict(z)

    def predict(self, z):
        standardized = (np.asarray(z, dtype=float) - self.w['z_mean']) / self.w['z_scale']
        result = []
        for i, probe in enumerate(self.meta['probes']):
            p = standardized[:, probe['coordinates']] @ self.w['probe%d_coef' % i].reshape(-1)
            p = p + self.w['probe%d_intercept' % i].reshape(-1)[0]
            if probe['classification']:
                p = 1. / (1. + np.exp(-np.clip(p, -700, 700)))
            else:
                p = p * probe['target_scale'] + probe['target_mean']
            result.append(p)
        return np.column_stack(result)


def actor_tail(weights, h, device='cpu'):
    t = {key: torch.as_tensor(value, device=device) for key, value in weights.items()}
    x = torch.as_tensor(np.asarray(h, dtype=np.float32), device=device)
    with torch.no_grad():
        y = F.linear(torch.relu(F.linear(x, t['mlp_extractor.policy_net.2.weight'],
              t['mlp_extractor.policy_net.2.bias'])), t['action_net.weight'], t['action_net.bias'])
    return y.cpu().numpy()


def verify_bundle(bundle, device='cpu'):
    bundle = Path(bundle)
    with np.load(bundle / 'reference.npz') as ref, np.load(bundle / 'actor.npz') as a:
        actor = {k: a[k] for k in a.files}
        h = ref['h']
        checks = []
        for folder in sorted((bundle / 'models').iterdir()):
            model = NumericSAE(folder, device)
            z, rec, pred = model.infer(h)
            mu = actor_tail(actor, rec, device)
            row = {'model': folder.name, 'device': device}
            for key, actual, atol in [('z', z, 2e-4), ('rec', rec, 2e-5),
                                       ('pred', pred, 2e-5), ('mu', mu, 2e-5)]:
                expected = ref[folder.name + '__' + key]
                row[key + '_max_abs'] = float(np.max(np.abs(actual - expected)))
                row[key + '_ok'] = bool(np.allclose(actual, expected, rtol=2e-5, atol=atol))
            row['passed'] = all(row[k + '_ok'] for k in ('z', 'rec', 'pred', 'mu'))
            checks.append(row)
    return {'passed': all(r['passed'] for r in checks), 'rows': len(h), 'checks': checks,
            'tolerances_frozen': {'rtol': 2e-5, 'z_atol': 2e-4, 'other_atol': 2e-5}}
