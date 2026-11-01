"""Step 2: read-only queries of the state/activation information table."""
import sys
sys.dont_write_bytecode = True

from temp_scripts.information_queries import main


if __name__ == "__main__":
    raise SystemExit(main())
