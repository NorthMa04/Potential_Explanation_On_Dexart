"""Step 2 maintenance: catalog existing assets without running experiments."""
import sys
sys.dont_write_bytecode = True

from temp_scripts.project_assets import main

if __name__ == "__main__":
    raise SystemExit(main())
