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
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cgroup-parent', type=Path, required=True)
    parser.add_argument('--ram-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.command or args.ram_bytes < 1 << 20 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('A command, finite positive timeout and at least 1 MiB RAM are required.')
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    if not any(args.output.is_relative_to((root / p).resolve()) for p in ('artifacts', 'docs/validation')):
        parser.error('Evidence must be inside artifacts/ or docs/validation/.')
    return args


def live_group_members(group, proc_root=Path('/proc')):
    """Read state without reaping the session leader or resolving a reused PGID."""
    members = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ('Z', 'X'):
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(members)


def cgroup_members(group):
    values = (group / 'cgroup.procs').read_text().split()
    if any(not value.isdecimal() or int(value) <= 0 for value in values):
        raise RuntimeError('Invalid cgroup membership observation.')
    return sorted({int(value) for value in values})


def cleanup_owned(child, group, report, grace=.5, wait_seconds=10):
    """All killpg calls precede reap; the unreaped leader reserves its PID/PGID."""
    errors = report['cleanup_errors']
    pgid = child.pid
    try:
        try:
            os.killpg(pgid, signal.SIGTERM)
            end = time.monotonic() + grace
            while live_group_members(pgid) and time.monotonic() < end:
                time.sleep(.05)
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append('Owned group TERM: ' + repr(error))
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append('Owned group KILL: ' + repr(error))
        # This path is only the unique cgroup created by this invocation. v2
        # cgroup.kill also covers descendants which start another session.
        # Never enumerate PIDs and signal them after a possible PID reuse.
        if group is not None:
            try:
                if (group / 'cgroup.kill').exists():
                    (group / 'cgroup.kill').write_text('1')
                    report['owned_cgroup_kill_used'] = True
            except BaseException as error:
                errors.append('Owned cgroup kill: ' + repr(error))
        try:
            end = time.monotonic() + wait_seconds
            members = live_group_members(pgid)
            while members and time.monotonic() < end:
                time.sleep(.05)
                members = live_group_members(pgid)
            report['live_group_members_after_kill'] = members
            if members:
                errors.append('Owned process group did not exit before cleanup deadline.')
        except BaseException as error:
            errors.append('Final group observation: ' + repr(error))
    finally:
        # No more process-group signals after this may reap/release the leader.
        try:
            report['exit_code'] = child.wait(timeout=wait_seconds)
        except BaseException as error:
            errors.append('Owned leader reap: ' + repr(error))


def remove_empty_owned_group(group, report, wait_seconds=10):
    """Keep failed groups and their limits intact; never recurse or alter a parent."""
    try:
        end = time.monotonic() + wait_seconds
        members = cgroup_members(group)
        while members and time.monotonic() < end:
            time.sleep(.05)
            members = cgroup_members(group)
        report['cgroup_members_after_cleanup'] = members
        if members:
            raise RuntimeError('Owned cgroup still has members; limits and group retained.')
        group.rmdir()
        report['owned_cgroup_removed'] = True
    except BaseException as error:
        report['cleanup_errors'].append('Owned cgroup removal: ' + repr(error))
        report['owned_cgroup_removed'] = False


def interrupt_once(report, context):
    """Publish cancellation before raising; repeated signals cannot break its log."""
    def interrupted(number, _frame):
        if 'interrupted_signal' in report:
            return
        report['interrupted_signal'] = number
        raise InterruptedError(f'{context} received signal {number}.')
    return interrupted


def supervise_owned(command, output, timeout, enter_group, sample, group=None):
    """Linux process supervision, invoked only after main has enforced limits.

    The group=None path is for CPU process-lifecycle unit tests, not a CLI
    fallback. v1 descendants must stay in the inherited process group; escaping
    descendants prevent cgroup removal and qualification, never pass silently.
    """
    report = dict(passed=False, exit_code=None, timed_out=False, errors=[], cleanup_errors=[],
                  live_group_members_after_kill=None)
    child = None
    handled = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    interrupted = interrupt_once(report, 'Validation supervisor')
    previous = {number: signal.signal(number, interrupted) for number in handled}
    started = time.monotonic()
    try:
        with (output / 'process.log').open('w', encoding='utf-8') as log:
            # Block cancellation from before fork through publishing the owned
            # PID. The CLI is single-threaded; join the cgroup before exec.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
            def before_exec():
                enter_group()
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            try:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=before_exec)
                report.update(pid=child.pid, process_group=child.pid)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            if os.getpgid(child.pid) != child.pid or os.getsid(child.pid) != child.pid:
                raise RuntimeError('Child did not receive its own process group/session.')
            (output / 'started.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            deadline = started + timeout
            while True:
                # poll()/wait() would reap and expose a reused PID to killpg.
                status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                sample(child.pid)
                expired = time.monotonic() >= deadline
                if status is not None:
                    report['timed_out'] = expired
                    report['exit_code_before_cleanup'] = status.si_status if status.si_code == os.CLD_EXITED else -status.si_status
                    report['stray_live_descendants_before_cleanup'] = live_group_members(child.pid)
                    break
                if expired:
                    report['timed_out'] = True
                    break
                time.sleep(.25)
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        # Repeated TERM/HUP/INT cannot interrupt draining/reaping/removal.
        cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled)
        try:
            for number in handled:
                signal.signal(number, signal.SIG_IGN)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        try:
            if child is not None:
                cleanup_owned(child, group, report)
            if group is not None:
                remove_empty_owned_group(group, report)
            report['seconds'] = time.monotonic() - started
            report['passed'] = (report.get('exit_code_before_cleanup') == 0 and not report['timed_out']
                and 'interrupted_signal' not in report
                and not report['errors'] and not report['cleanup_errors']
                and not report.get('stray_live_descendants_before_cleanup')
                and report['live_group_members_after_kill'] == []
                and (group is None or report.get('owned_cgroup_removed') is True))
            (output / 'supervision.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return report


def main(argv=None):
    args = arguments(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = dict(command=args.command, requested_ram_bytes=args.ram_bytes, swap_bytes=0,
                  enforcement='unavailable', passed=False, exit_code=None, samples=[], errors=[], cleanup_errors=[])
    group = None
    created = False
    supervised = False
    supervision_returned = False
    previous = {}
    result = 77
    try:
        if sys.platform != 'linux' or not hasattr(os, 'WNOWAIT'):
            raise RuntimeError('A delegated Linux memory cgroup and waitid(WNOWAIT) are required.')
        interrupted = interrupt_once(report, 'Cgroup setup')
        previous = {number: signal.signal(number, interrupted)
                    for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        parent = args.cgroup_parent.resolve(strict=True)
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
        creation_mask = signal.pthread_sigmask(signal.SIG_BLOCK, tuple(previous))
        try:
            group.mkdir()
            created = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, creation_mask)
        if v2:
            (group / 'memory.max').write_text(str(args.ram_bytes))
            (group / 'memory.swap.max').write_text('0')
            limit_file, swap_file = 'memory.max', 'memory.swap.max'
            metrics = ('memory.current', 'memory.peak', 'memory.events', 'memory.stat', 'memory.swap.current')
        else:
            (group / 'memory.limit_in_bytes').write_text(str(args.ram_bytes))
            (group / 'memory.memsw.limit_in_bytes').write_text(str(args.ram_bytes))
            limit_file, swap_file = 'memory.limit_in_bytes', 'memory.memsw.limit_in_bytes'
            metrics = ('memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.failcnt',
                       'memory.oom_control', 'memory.stat', 'memory.memsw.usage_in_bytes')
        effective = int((group / limit_file).read_text())
        swap = int((group / swap_file).read_text())
        if not 0 < effective <= args.ram_bytes or swap != (0 if v2 else effective):
            raise RuntimeError('Read-back limits do not enforce the requested RAM/no-swap ceiling.')
        report.update(enforcement='cgroup-v2' if v2 else 'cgroup-v1', cgroup=str(group),
                      effective_ram_bytes=effective, configured_swap_limit=swap)
        def sample(pid=None):
            values = {name: (group / name).read_text().strip() for name in metrics if (group / name).exists()}
            values['cgroup.procs'] = (group / 'cgroup.procs').read_text().strip()
            if pid is not None:
                try:
                    values['process_status'] = '\n'.join(line for line in Path('/proc', str(pid), 'status').read_text().splitlines()
                                                        if line.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssFile:', 'VmSwap:')))
                    values['process_cgroup'] = Path('/proc', str(pid), 'cgroup').read_text().strip()
                except FileNotFoundError:
                    pass
            report['samples'].append(dict(unix_ms=time.time_ns() // 1_000_000, values=values))
        sample()
        def enter_group():
            (group / 'cgroup.procs').write_text(str(os.getpid()))
        # The helper owns child cleanup and removal even when cancellation or
        # sampling fails. Never retry it without the enforced cgroup.
        supervised = True
        lifecycle = supervise_owned(args.command, output, args.timeout, enter_group, sample, group)
        supervision_returned = True
        report.update(lifecycle)
        result = (77 if 'pid' not in lifecycle and not report['cleanup_errors']
                  else 0 if report['passed'] else 1)
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        result = 1 if supervised or isinstance(error, (InterruptedError, KeyboardInterrupt)) else 77
    finally:
        if previous:
            cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, tuple(previous))
            try:
                for number in previous:
                    signal.signal(number, signal.SIG_IGN)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        try:
            if created and not supervision_returned and group.exists():
                remove_empty_owned_group(group, report)
            if report['errors'] or report['cleanup_errors']:
                report['passed'] = False
            if report['cleanup_errors']:
                result = 1
            (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    print(json.dumps({k: report[k] for k in ('enforcement', 'passed', 'errors', 'cleanup_errors')}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
