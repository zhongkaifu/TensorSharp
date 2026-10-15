"""Sample one owned process and board VRAM. Observations are not allocation ownership."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=7200)
a = p.parse_args()
owner = psutil.Process(a.pid)
birth = owner.create_time()
started = time.monotonic()
with a.output.open('x', encoding='utf-8') as stream:
    while time.monotonic()-started < a.seconds:
        row = {'unix': time.time(), 'elapsed': time.monotonic()-started, 'pid': a.pid, 'created': birth}
        try:
            if owner.create_time() != birth or not owner.is_running(): break
            memory = owner.memory_info()
            row.update(rss=memory.rss, private=getattr(memory, 'private', None), peak_rss=getattr(memory, 'peak_wset', None))
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=3,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            row['board_gpu'] = gpu.stdout.strip()
            if gpu.returncode: row['gpu_error'] = gpu.stderr.strip()
        except psutil.NoSuchProcess: break
        except Exception as ex: row['error'] = str(ex)
        stream.write(json.dumps(row)+'\n'); stream.flush()
        time.sleep(1)
