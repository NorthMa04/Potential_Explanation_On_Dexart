"""Keep Windows awake during this retrieval without changing its power plan."""
import ctypes
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1] / 'outputs/combined_physical/20261103_140606_955714'
status = root / 'runtime/keep_awake.json'
stop = root / 'runtime/keep_awake.stop'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
kernel.SetThreadExecutionState.restype = ctypes.c_uint32
previous = kernel.SetThreadExecutionState(0x80000001)
if not previous:
    raise ctypes.WinError(ctypes.get_last_error())
status.write_text(json.dumps({'status': 'system_sleep_prevented', 'started_unix': time.time()}))
try:
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline and not stop.exists():
        time.sleep(10)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    status.write_text(json.dumps({'status': 'released', 'ended_unix': time.time()}))
