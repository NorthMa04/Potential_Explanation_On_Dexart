"""All generated files live below this project's outputs/ directory."""
from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def bootstrap():
    cache = OUTPUT_ROOT / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(cache / "pycache")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)


def new_run(label="review"):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = OUTPUT_ROOT / "runs" / (stamp + "_" + label)
    folder.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "latest_run.txt").write_text(str(folder) + "\n", encoding="utf-8")
    return folder


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Output must stay inside project outputs/: " + str(path))
    return path


def default_index():
    return PROJECT_ROOT.parents[1] / "research" / "embodied_sae" / "runs" / "laptop_physical_index_v1"


def source_path(name):
    """Resolve an entry/document at project root or a shared implementation module."""
    candidate = PROJECT_ROOT / name
    return candidate if candidate.is_file() else MODULE_ROOT / name
