#!/usr/bin/env python3
"""Bound one Linux build process group and retain logs even when it fails.

The child starts a new session. Its leader remains unreaped until group cleanup,
so the owned group identifier cannot be reused while cleanup signals are sent.
This covers ordinary cmake/ninja/nvcc descendants in that group, not a program
that deliberately escapes with setsid(). It is not a model benchmark runner.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--grace', type=float, default=5)
    parser.add_argument('--space-floor', nargs=2, action='append', default=[], metavar=('PATH', 'BYTES'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not args.command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('A command and finite positive timeout are required.')
    if not math.isfinite(args.grace) or not 0 <= args.grace <= 30:
        parser.error('Grace must be finite and within 0..30 seconds.')
    try:
        args.space_floor = [(Path(path), int(floor)) for path, floor in args.space_floor]
        if any(floor <= 0 for _, floor in args.space_floor):
            raise ValueError()
    except ValueError:
        parser.error('Disk space floors must be positive integer bytes.')
    return args


def sample_space(floors):
    samples = []
    for path, minimum in floors:
        free = shutil.disk_usage(path).free
        samples.append(dict(path=str(path), free_bytes=free, minimum_bytes=minimum))
    return samples


def require_space(samples):
    for sample in samples:
        if sample['free_bytes'] < sample['minimum_bytes']:
            raise OSError('Low disk space: ' + json.dumps(sample))


def live_group_members(group, proc_root=Path('/proc')):
    result = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces or parentheses; fields after the final ')'
            # start with state, ppid, pgrp. Read only membership/state diagnostics.
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ('Z', 'X'):
                result.append(int(entry.name))
        except FileNotFoundError:
            continue  # Process exited during enumeration.
        except ProcessLookupError:
            continue
    return sorted(result)


def main(argv=None):
    args = arguments(argv)
    if sys.platform != 'linux' or not hasattr(os, 'WNOWAIT'):
        raise RuntimeError('This process-group supervisor requires Linux waitid(WNOWAIT).')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(command=args.command, timeout_seconds=args.timeout, grace_seconds=args.grace,
                  space_samples=[], space_sample_count=0, minimum_observed_free_bytes={}, space_sample_interval_seconds=1,
                  passed=False, timed_out=False, errors=[], cleanup_errors=[],
                  scope='Owned Linux build process group only; no setsid escape or model/performance qualification.')
    child = None
    group = None
    def interrupted(number, _frame):
        # A second signal can run while the first exception is being formatted
        # in except, before finally masks cancellation. Preserve the first one.
        if 'interrupted_signal' in report:
            return
        report['interrupted_signal'] = number
        raise InterruptedError(f'Build supervisor received signal {number}.')
    handled_signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous_handlers = {number: signal.signal(number, interrupted) for number in handled_signals}
    started = time.monotonic()
    def observe_space():
        samples = sample_space(args.space_floor)
        report['space_samples'] = samples
        report['space_sample_count'] += 1
        minima = report['minimum_observed_free_bytes']
        for sample in samples:
            path = sample['path']
            minima[path] = min(minima.get(path, sample['free_bytes']), sample['free_bytes'])
        require_space(samples)
    try:
        observe_space()
        with (output / 'process.log').open('wb') as log:
            # A cancellation between spawn and assigning child must not orphan
            # an otherwise untracked group. Deliver it after identity publication.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled_signals)
            try:
                child = subprocess.Popen(args.command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         # This CLI creates no threads. Restore the
                                         # child's inherited mask before exec;
                                         # otherwise TERM would stay blocked there.
                                         preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
                group = child.pid
                report.update(pid=child.pid, process_group=group)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            if os.getpgid(child.pid) != group or os.getsid(child.pid) != group:
                raise RuntimeError('Child did not receive its own process group/session.')
            (output / 'started.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            deadline = started + args.timeout
            next_space_sample = started
            while True:
                status = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                expired = time.monotonic() >= deadline
                if time.monotonic() >= next_space_sample or status is not None:
                    observe_space()
                    next_space_sample = time.monotonic() + 1
                if status is not None:
                    report['timed_out'] = expired
                    report['exit_code_before_cleanup'] = status.si_status if status.si_code == os.CLD_EXITED else -status.si_status
                    report['stray_live_descendants_before_cleanup'] = live_group_members(group)
                    break
                if expired:
                    report['timed_out'] = True
                    break
                time.sleep(0.1)
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        # Repeated external cancellation cannot interrupt owned-child cleanup.
        cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled_signals)
        try:
            for number in handled_signals:
                signal.signal(number, signal.SIG_IGN)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        if child is not None:
            # Do not call Popen.poll/wait before these signals: the retained
            # leader PID protects the process-group identity from PID reuse.
            try:
                os.killpg(group, signal.SIGTERM)
                grace_end = time.monotonic() + args.grace
                while live_group_members(group) and time.monotonic() < grace_end:
                    time.sleep(0.1)
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                report['cleanup_errors'].append(f'Owned group: {type(error).__name__}: {error}')
                # A failed TERM/enumeration must not skip the final owned kill.
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as kill_error:
                    report['cleanup_errors'].append(f'Owned group KILL: {type(kill_error).__name__}: {kill_error}')
            try:
                # Check before reaping, retaining the same group identity.
                stop_deadline = time.monotonic() + 10
                members = live_group_members(group)
                while members and time.monotonic() < stop_deadline:
                    time.sleep(0.1)
                    members = live_group_members(group)
                report['live_group_members_after_kill'] = members
            except BaseException as error:
                report['cleanup_errors'].append(f'Final group observation: {type(error).__name__}: {error}')
            finally:
                try:
                    report['exit_code'] = child.wait(timeout=10)
                except BaseException as error:
                    report['cleanup_errors'].append(f'Child reap: {type(error).__name__}: {error}')
        report['seconds'] = time.monotonic() - started
        report['passed'] = (report.get('exit_code_before_cleanup') == 0 and not report['timed_out']
                            and not report['errors'] and not report['cleanup_errors']
                            and not report.get('stray_live_descendants_before_cleanup')
                            and report.get('live_group_members_after_kill') == [])
        try:
            (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
    print(json.dumps({key: report[key] for key in ('passed', 'timed_out', 'errors', 'cleanup_errors')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
