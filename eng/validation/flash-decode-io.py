#!/usr/bin/env python3
"""Diagnostic process/system I/O sampler; use --process-counters true in the probe.

Physical disk counters cover ALL processes, not only the measured model. Windows
process faults include soft faults and process I/O excludes some mapped-file I/O.
This does not flush caches or claim controlled cold-storage measurements.
Requires psutil. Writes generated evidence only under artifacts/ or docs/validation/.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0:
        parser.error('Supply a positive timeout and -- executable [arguments]')
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to(root / base) for base in ('artifacts','docs/validation')):
        parser.error('Generated evidence must stay under artifacts/ or docs/validation/')
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    errors = []
    def sample(process=None):
        row = dict(unix_ms=time.time_ns()//1_000_000,
            system_memory=psutil.virtual_memory()._asdict(),
            whole_disk={k:v._asdict() for k,v in psutil.disk_io_counters(perdisk=True).items()})
        if process is not None:
            try:
                row['process_memory'] = process.memory_info()._asdict()
                row['process_io'] = process.io_counters()._asdict()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                row['process_error'] = str(error)
        samples.append(row)
    sample()
    started = time.monotonic()
    timed_out = False
    with (output/'process.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                sample(process)
                if time.monotonic()-started > args.timeout:
                    timed_out = True
                    child.kill()
                    break
                time.sleep(0.25)
            exit_code = child.wait(timeout=10)
        except BaseException:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            raise
        finally:
            sample()
            result = dict(command=command, pid=child.pid, exit_code=child.returncode,
                timed_out=timed_out, wall_seconds=time.monotonic()-started,
                sample_interval_seconds=0.25, samples=samples, errors=errors,
                limitations=['Disk counters include all processes and trace/other file I/O.',
                    'Process faults include soft faults; process IO is not mapped-weight physical disk IO.',
                    'No OS cache flush. Diagnostic run, not quiet throughput.'])
            (output/'io.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(pid=child.pid, exit_code=exit_code, timed_out=timed_out, samples=len(samples))))
    return 0 if exit_code == 0 and not timed_out else 1


if __name__ == '__main__':
    raise SystemExit(main())
