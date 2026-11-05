"""Export frozen development-selected SAEs/probes as numeric arrays for simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import joblib
import numpy as np
import torch
from temp_scripts.actor_model import FrozenActor
from temp_scripts.state_assisted_model import load_assisted_checkpoint
from temp_scripts.physical_runtime import NumericSAE, verify_bundle


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, default=Path(__file__).parent / 'outputs/state_assisted/20261102_122123_738477')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.study_dir.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((root / 'frozen_evaluation_plan.json').read_text())
    shutil.copy2(root / 'frozen_evaluation_plan.json', out / 'source_frozen_plan.json')
    source = np.load(root / 'selected_arrays.npz')
    h = source['h']
    assert set(source['split']) == {'train', 'validation'}
    actor = FrozenActor().eval().requires_grad_(False)
    actor.load_state_dict(torch.load(root / 'model/frozen_actor_state.pt', weights_only=True))
    np.savez_compressed(out / 'actor.npz', **{k: v.numpy() for k, v in actor.state_dict().items()})
    reference, entries = {'h': h}, []
    for kind, config in [('pure', 'dense_l0p0'), ('assisted', 'slots_s0.3_a1.0')]:
        for seed in (0, 1, 2):
            eid = config + '_seed' + str(seed)
            folder = root / eid
            for filename in ('best_model.pt', 'fitted_probes.joblib', 'summary.json'):
                assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == frozen['artifact_hashes'][eid][filename]
            model, scaler, payload = load_assisted_checkpoint(folder / 'best_model.pt')
            artifacts = joblib.load(folder / 'fitted_probes.joblib')
            weights = {key.replace('.', '_'): value.numpy() for key, value in model.state_dict().items()
                       if key.startswith(('encoder.', 'decoder.'))}
            weights.update(mean=scaler.mean, scale=scaler.scale,
                           z_mean=artifacts['z_scaler'].mean_, z_scale=artifacts['z_scaler'].scale_)
            probes, predictions = [], []
            with torch.no_grad():
                z, reconstruction = model(torch.from_numpy(scaler.transform(h)))
                z = z.numpy()
            rec = scaler.inverse_transform(reconstruction.numpy())
            for i, target in enumerate(('distance_m', 'opening_rad', 'any_contact')):
                choices = [v for k, v in artifacts['models'].items()
                           if k.startswith(target + '__z_top4__') and not v.get('with_time', False)]
                assert len(choices) == 1
                fitted = choices[0]
                est, ts = fitted['estimator'], fitted['target_scaler']
                assert type(est).__name__ in ('Ridge', 'LogisticRegression')
                weights['probe%d_coef' % i] = est.coef_
                weights['probe%d_intercept' % i] = np.atleast_1d(est.intercept_)
                standardized = artifacts['z_scaler'].transform(z)[:, fitted['coordinates']]
                p = est.predict_proba(standardized)[:, list(est.classes_).index(1)] if i == 2 else ts.inverse_transform(est.predict(standardized)[:, None]).ravel()
                predictions.append(p)
                probes.append({'target': target, 'coordinates': fitted['coordinates'], 'classification': i == 2,
                               'target_scale': float(ts.scale_[0]) if ts else None,
                               'target_mean': float(ts.mean_[0]) if ts else None})
            name = kind + '_seed' + str(seed)
            dest = out / 'models' / name
            dest.mkdir(parents=True)
            np.savez_compressed(dest / 'weights.npz', **weights)
            dump(dest / 'model.json', {'name': name, 'experiment_id': eid, 'seed': seed,
                 'top_k': model.top_k, 'probes': probes, 'source_hashes': frozen['artifact_hashes'][eid]})
            with torch.no_grad():
                mu = actor.from_h1(torch.as_tensor(rec, dtype=torch.float32)).numpy()
            reference.update({name + '__z': z, name + '__rec': rec,
                              name + '__pred': np.column_stack(predictions), name + '__mu': mu})
            entries.append(name)
    np.savez_compressed(out / 'reference.npz', **reference)
    result = verify_bundle(out)
    dump(out / 'local_verification.json', result)
    assert result['passed'], result
    dump(out / 'bundle_manifest.json', {'format': 'numeric_physical_v1', 'models': entries,
        'data_scope': '1365 train/validation event-window frames only; no test data',
        'source_study': str(root), 'actor_checkpoint_unchanged': True,
        'files': {str(p.relative_to(out)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
