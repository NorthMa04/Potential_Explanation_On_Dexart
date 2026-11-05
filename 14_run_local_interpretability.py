"""Local interpretability follow-up: frozen models, development data, saved rollouts."""
import sys
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
from temp_scripts.project_paths import bootstrap
bootstrap()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-edits', action='store_true')
    args = p.parse_args()
    from temp_scripts.local_interpretability import run
    run(args.out, skip_edits=args.skip_edits)
    from temp_scripts.local_contact_diagnostic import run as diagnose_contact
    diagnose_contact(args.out)
    if not args.skip_edits:
        from temp_scripts.validate_local_interpretability import validate
        validate(args.out)
    import shutil, json, hashlib
    from temp_scripts.project_paths import PROJECT_ROOT
    sources=['14_run_local_interpretability.py','temp_scripts/local_interpretability.py',
             'temp_scripts/local_contact_diagnostic.py','temp_scripts/validate_local_interpretability.py']
    source_hashes={}
    for relative in sources:
        source=PROJECT_ROOT/relative
        shutil.copy2(source,args.out/('source_'+source.name))
        source_hashes[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
    (args.out/'implementation_hashes.json').write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
