#!/usr/bin/env python3
"""Observe whole-disk I/O while one identified process lives; never attribute it exclusively to that process."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

import psutil

SPEC = importlib.util.spec_from_file_location('storage_bench', Path(__file__).with_name('multimodel-quality-bench.py'))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--created', type=float, required=True, help='Expected psutil process creation timestamp')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=2700)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.created, args.seconds, args.interval)):
        parser.error('Require positive finite timestamps and observation durations')
    directory = bench.evidence_directory(args.output.parent)
    directory.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(args.pid)
    if abs(process.create_time() - args.created) > .001:
        parser.error('PID identity changed before observation')
    started = time.monotonic()
    header = dict(kind='scope', pid=args.pid, process_created=args.created, unix=time.time(),
                  observer_sha256=bench.sha(__file__), interval_seconds=args.interval,
                  scope='Whole physical-disk counters include all processes and OS page faults. '
                        'They are not exclusive process I/O, mapped-page ownership or a physical memory cap. '
                        'Observation can start after model loading; no earlier I/O is inferred.')
    with args.output.open('x', encoding='utf-8', buffering=1) as output:
        def emit(row):
            output.write(json.dumps(row) + '\n')
        emit(header)
        reason = 'deadline'
        while time.monotonic() - started < args.seconds:
            try:
                if not process.is_running():
                    reason = 'process exited'
                    break
                counters = psutil.disk_io_counters(perdisk=True, nowrap=True)
                emit(dict(kind='sample', unix=time.time(), elapsed_seconds=time.monotonic() - started,
                          disks={name: value._asdict() for name, value in counters.items()},
                          process_io=process.io_counters()._asdict()))
            except psutil.NoSuchProcess:
                reason = 'process exited'
                break
            except (OSError, psutil.Error) as error:
                emit(dict(kind='error', unix=time.time(), error=repr(error)))
            time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - started))))
        emit(dict(kind='finished', reason=reason, elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
