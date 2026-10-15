#!/usr/bin/env python3
"""Prove a scoped v2 parent RAM limit; never modify a pre-existing group's limits.

Runs the GLM observer safely, then deliberately OOM-kills only a disposable child.
The parent is 128 MiB, child 256 MiB, both swap disabled. An observer returning 0
is recorded as unknown/fail-closed, not as proof of exact available-byte telemetry.
Exit 77 means the environment cannot perform this validation, never a pass.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid


def proves_parent_limit(before, after, exit_code):
    def counters(value):
        return {key: int(number) for key, number in (line.split() for line in value.splitlines())}
    parent0 = counters(before['parent']['memory.events.local'])
    parent1 = counters(after['parent']['memory.events.local'])
    child0 = counters(before['child']['memory.events.local'])
    child1 = counters(after['child']['memory.events.local'])
    # Linux SIGKILL is 9; keep this pure evidence classifier runnable on Windows.
    # Limit events belong to the parent, while oom_kill is charged to the killed
    # process's actual child group.
    return (exit_code == -9 and
            all(parent1[name] > parent0[name] for name in ('max', 'oom')) and
            child1['oom_kill'] > child0['oom_kill'] and
            all(child1[name] == child0[name] for name in ('max', 'oom')) and
            all(int(after[level]['memory.swap.current']) == 0 for level in ('parent', 'child')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cgroup-parent', type=Path, default=Path('/sys/fs/cgroup'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, kernel=platform.release(),
                  environment='WSL' if 'microsoft' in platform.release().lower() else platform.system(),
                  parentLimit=128 << 20, childLimit=256 << 20, swapLimit=0, runs=[], errors=[])
    outer = inner = None
    created = []
    process = None
    code = 77

    def read(directory):
        names = ('memory.max', 'memory.current', 'memory.peak', 'memory.events', 'memory.events.local',
                 'memory.swap.max', 'memory.swap.current', 'cgroup.procs')
        return {name: (directory/name).read_text().strip() for name in names}

    try:
        if sys.platform != 'linux':
            raise RuntimeError('Linux with a delegated cgroup-v2 memory controller is required.')
        parent = args.cgroup_parent.resolve(strict=True)
        real_mount = False
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            if right.split()[0] == 'cgroup2':
                mount = Path(left.split()[4].replace('\\040', ' ')).resolve()
                real_mount |= parent.is_relative_to(mount)
        if not real_mount or 'memory' not in (parent/'cgroup.subtree_control').read_text().split():
            raise RuntimeError('Selected parent does not expose delegated cgroup-v2 memory.')
        result['supervisorMembership'] = Path('/proc/self/cgroup').read_text()
        outer = parent/('tensorsharp-glm-canary-' + uuid.uuid4().hex)
        outer.mkdir()
        created.append(outer)
        (outer/'memory.max').write_text(str(result['parentLimit']))
        (outer/'memory.swap.max').write_text('0')
        (outer/'cgroup.subtree_control').write_text('+memory')
        inner = outer/'child'
        inner.mkdir()
        created.append(inner)
        (inner/'memory.max').write_text(str(result['childLimit']))
        (inner/'memory.swap.max').write_text('0')
        result.update(cgroup=str(outer), before={'parent': read(outer), 'child': read(inner)})
        if (int((outer/'memory.max').read_text()) != result['parentLimit'] or
                int((inner/'memory.max').read_text()) != result['childLimit'] or
                int((outer/'memory.swap.max').read_text()) != 0 or int((inner/'memory.swap.max').read_text()) != 0):
            raise RuntimeError('Kernel limits did not match requested parent/child RAM and swap ceilings.')
        for mode in ('observe', 'overrun'):
            def enter():
                (inner/'cgroup.procs').write_text(str(os.getpid()))
            started = time.monotonic()
            with (output/(mode+'.log')).open('w') as log:
                process = subprocess.Popen([str(args.binary.resolve()), mode, str(result['parentLimit'])],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           preexec_fn=enter, start_new_session=True)
                status = process.wait(timeout=30)
            result['runs'].append(dict(mode=mode, exitCode=status, seconds=time.monotonic()-started,
                                       parent=read(outer), child=read(inner)))
            process = None
            if mode == 'observe' and status != 0:
                raise RuntimeError('The bounded observer failed; the OOM test was not launched.')
        observations = [json.loads(line) for line in (output/'observe.log').read_text().splitlines()]
        if not observations[0].get('membership', '').endswith('/' + outer.name + '/child'):
            raise RuntimeError('Actual child process membership did not match the owned nested group.')
        result['observer'] = observations
        result['observerKnown'] = all(row.get('knownBefore', row.get('knownAfter', False)) for row in observations)
        result['passed'] = proves_parent_limit(result['before'], result['runs'][1], result['runs'][1]['exitCode'])
        if not result['passed']:
            raise RuntimeError('No verified parent-limit OOM kill: enforcement is not proven.')
        code = 0
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
        code = 1 if result['runs'] else 77
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for directory in reversed(created):
            if directory.exists():
                try:
                    directory.rmdir()  # Only the exact newly created empty groups.
                except OSError as error:
                    result['errors'].append(f'Cleanup: {error}')
                    result['passed'] = False
                    code = 1
        result['ownGroupsRemoved'] = outer is not None and not outer.exists()
        (output/'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
