"""Cross-runtime edit verification against frozen development reference cases."""
import argparse
import json
from pathlib import Path

import numpy as np
from constrained_intervention import ConstrainedEditor
from physical_runtime import NumericSAE


def verify(bundle):
    bundle = Path(bundle)
    cases = json.loads((bundle / 'edit_reference.json').read_text())
    with np.load(bundle / 'edit_reference.npz') as data:
        h, expected = data['h'], data['emitted']
    numeric = {p.name: NumericSAE(p) for p in (bundle / 'models').iterdir()}
    editor = ConstrainedEditor(bundle, numeric)
    errors, statuses = [], []
    for i, case in enumerate(cases):
        actual, info = editor.edit(h[i], case['model'], case['operation'], case['control'], case['random_key'])
        errors.append(float(np.max(np.abs(actual - expected[i]))))
        statuses.append(info['status'] == case['status'])
    return {'passed': all(statuses) and max(errors) < 2e-5, 'cases': len(cases),
            'max_emission_abs': max(errors), 'all_statuses_match': all(statuses),
            'tolerance': 2e-5}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = verify(args.bundle)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert result['passed'], result
