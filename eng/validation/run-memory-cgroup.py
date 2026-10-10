#!/usr/bin/env python3
"""Run one owned Linux validation process in a fresh RAM+swap-limited cgroup.

Requires a delegated memory controller. Never changes an existing cgroup's limits
or launches an unbounded fallback. Exit 77 means unavailable, not passed. GPU
VRAM is NOT capped by this tool. Shared file pages can be charged to their first
touching cgroup; cgroup usage is not the same as process RSS.
Sources: https://docs.kernel.org/admin-guide/cgroup-v2.html
         https://docs.kernel.org/admin-guide/cgroup-v1/memory.html
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cgroup-parent', type=Path, required=True)
    parser.add_argument('--ram-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not command or args.ram_bytes < 1 << 20 or args.timeout <= 0:
        parser.error('A command, positive timeout and at least 1 MiB RAM are required.')
    if not any(output.is_relative_to(root / p) for p in ('artifacts', 'docs/validation')):
        parser.error('Evidence must be inside artifacts/ or docs/validation/.')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(command=command, requested_ram_bytes=args.ram_bytes, swap_bytes=0,
                  enforcement='unavailable', passed=False, exit_code=None, samples=[], errors=[])
    group = None
    created = False
    child = None
    result = 77
    try:
        if sys.platform != 'linux':
            raise RuntimeError('A delegated Linux memory cgroup is required.')
        parent = args.cgroup_parent.resolve(strict=True)
        # Require a real cgroup filesystem, never write lookalike files in an
        # arbitrary directory. The operator explicitly selects the delegation.
        mounts = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            fields = line.split()
            sep = fields.index('-')
            if fields[sep + 1] in ('cgroup', 'cgroup2'):
                mount = Path(fields[4].replace('\\040', ' ')).resolve()
                if parent.is_relative_to(mount):
                    mounts.append((len(str(mount)), fields[sep + 1]))
        if not mounts:
            raise RuntimeError('Parent is not on a cgroup filesystem.')
        v2 = max(mounts)[1] == 'cgroup2'
        if v2 and 'memory' not in (parent / 'cgroup.subtree_control').read_text().split():
            raise RuntimeError('Parent has not delegated memory to children; it was left unchanged.')
        group = parent / ('tensorsharp-validation-' + uuid.uuid4().hex)
        group.mkdir()  # Unique, empty child; permission refusal stops before launch.
        created = True
        if v2:
            (group / 'memory.max').write_text(str(args.ram_bytes))
            (group / 'memory.swap.max').write_text('0')
            limit_file, swap_file = 'memory.max', 'memory.swap.max'
            metrics = ('memory.current', 'memory.peak', 'memory.events', 'memory.stat', 'memory.swap.current')
        else:
            (group / 'memory.limit_in_bytes').write_text(str(args.ram_bytes))
            # Without memsw, the requested no-swap hard limit is unverified.
            (group / 'memory.memsw.limit_in_bytes').write_text(str(args.ram_bytes))
            limit_file, swap_file = 'memory.limit_in_bytes', 'memory.memsw.limit_in_bytes'
            metrics = ('memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.failcnt',
                       'memory.oom_control', 'memory.stat', 'memory.memsw.usage_in_bytes')
        effective = int((group / limit_file).read_text())
        swap = int((group / swap_file).read_text())
        if effective > args.ram_bytes or swap != (0 if v2 else effective):
            raise RuntimeError('Read-back limits do not enforce the requested RAM/no-swap ceiling.')
        report.update(enforcement='cgroup-v2' if v2 else 'cgroup-v1', cgroup=str(group),
                      effective_ram_bytes=effective, configured_swap_limit=swap)
        def sample():
            values = {name: (group / name).read_text().strip() for name in metrics if (group / name).exists()}
            if child is not None and Path('/proc', str(child.pid), 'status').exists():
                try:
                    values['process_status'] = '\n'.join(line for line in Path('/proc', str(child.pid), 'status').read_text().splitlines()
                                                        if line.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssFile:', 'VmSwap:')))
                except FileNotFoundError:
                    pass
            report['samples'].append(dict(unix_ms=time.time_ns() // 1_000_000, values=values))
        sample()
        def enter_group():
            (group / 'cgroup.procs').write_text(str(os.getpid()))
        started = time.monotonic()
        with (output / 'process.log').open('w', encoding='utf-8') as log:
            # The wrapper is single-threaded; join before exec so the model is
            # never initialized outside its assigned group.
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     preexec_fn=enter_group, start_new_session=True)
            while child.poll() is None:
                sample()
                if time.monotonic() - started >= args.timeout:
                    raise TimeoutError('Owned validation process exceeded its deadline.')
                time.sleep(0.25)
            sample()
        report.update(exit_code=child.returncode, seconds=time.monotonic() - started, passed=child.returncode == 0)
        result = 0 if report['passed'] else 1
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        result = 77 if child is None else 1
    finally:
        if child is not None:
            # Also stop descendants in this wrapper-owned process group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if created:
            try:
                group.rmdir()  # Never recurse or alter any pre-existing group.
            except OSError as error:
                report['errors'].append(f'Cleanup: {error}')
                report['passed'] = False
                result = 1
        (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('enforcement', 'passed', 'errors')}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
