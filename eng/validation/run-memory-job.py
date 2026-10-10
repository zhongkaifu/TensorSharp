#!/usr/bin/env python3
"""Run an owned Windows child under a verified Job private-commit ceiling.

This is NOT a physical RAM/RSS, file-cache or VRAM cap. The job constrains
committed virtual memory for the process tree. Assignment happens before the
initial thread resumes; no unbounded fallback is launched. Exit 77 means the
requested enforcement was unavailable. Core TensorSharp does not require Jobs.

https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information
https://learn.microsoft.com/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    root = Path(__file__).resolve().parents[2]
    # Keep evidence at the requested ignored path, including owned junctions to
    # a larger volume. No existing directory or external process is modified.
    output = Path(os.path.abspath(args.output))
    if not any(output.is_relative_to(root / p) for p in ('artifacts', 'docs/validation')):
        parser.error('Evidence must be inside artifacts/ or docs/validation/.')
    if not command or args.commit_bytes < 64 << 20 or args.timeout <= 0:
        parser.error('A command, positive timeout and at least 64 MiB commit are required.')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(command=command, requested_commit_bytes=args.commit_bytes,
                  enforcement='unavailable', passed=False, errors=[], samples=[], processes=[],
                  scope='Process-tree private commit only; excludes mapped-file residency, shared file cache and VRAM.')
    job = None
    result = 77
    try:
        if sys.platform != 'win32':
            raise RuntimeError('Windows Job enforcement is unavailable on this platform.')
        import msvcrt
        kernel = c.WinDLL('kernel32', use_last_error=True)
        size = c.c_size_t

        class Basic(c.Structure):
            _fields_ = [('per_process_time', c.c_int64), ('per_job_time', c.c_int64),
                        ('flags', w.DWORD), ('minimum_ws', size), ('maximum_ws', size),
                        ('active_processes', w.DWORD), ('affinity', size),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]

        class Io(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ('read', 'write', 'other', 'read_bytes', 'write_bytes', 'other_bytes')]

        class Extended(c.Structure):
            _fields_ = [('basic', Basic), ('io', Io), ('process_limit', size), ('job_limit', size),
                        ('process_peak', size), ('job_peak', size)]

        class Startup(c.Structure):
            _fields_ = [('cb', w.DWORD), ('reserved', w.LPWSTR), ('desktop', w.LPWSTR), ('title', w.LPWSTR),
                        ('x', w.DWORD), ('y', w.DWORD), ('width', w.DWORD), ('height', w.DWORD),
                        ('xchars', w.DWORD), ('ychars', w.DWORD), ('fill', w.DWORD), ('flags', w.DWORD),
                        ('show', w.WORD), ('reserved_size', w.WORD), ('reserved_data', c.c_void_p),
                        ('stdin', w.HANDLE), ('stdout', w.HANDLE), ('stderr', w.HANDLE)]

        class Process(c.Structure):
            _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE), ('pid', w.DWORD), ('tid', w.DWORD)]

        def api(name, result_type, *parameters):
            fn = getattr(kernel, name)
            fn.restype, fn.argtypes = result_type, parameters
            return fn

        create_job = api('CreateJobObjectW', w.HANDLE, c.c_void_p, w.LPCWSTR)
        set_job = api('SetInformationJobObject', w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD)
        query_job = api('QueryInformationJobObject', w.BOOL, w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p)
        assign = api('AssignProcessToJobObject', w.BOOL, w.HANDLE, w.HANDLE)
        member = api('IsProcessInJob', w.BOOL, w.HANDLE, w.HANDLE, c.POINTER(w.BOOL))
        create = api('CreateProcessW', w.BOOL, w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p,
                     w.BOOL, w.DWORD, c.c_void_p, w.LPCWSTR, c.POINTER(Startup), c.POINTER(Process))
        resume = api('ResumeThread', w.DWORD, w.HANDLE)
        wait = api('WaitForSingleObject', w.DWORD, w.HANDLE, w.DWORD)
        code = api('GetExitCodeProcess', w.BOOL, w.HANDLE, c.POINTER(w.DWORD))
        terminate = api('TerminateProcess', w.BOOL, w.HANDLE, w.UINT)
        terminate_job = api('TerminateJobObject', w.BOOL, w.HANDLE, w.UINT)
        close = api('CloseHandle', w.BOOL, w.HANDLE)

        def check(ok):
            if not ok:
                raise c.WinError(c.get_last_error())

        def sample():
            info = Extended()
            check(query_job(job, 9, c.byref(info), c.sizeof(info), None))
            report['samples'].append(dict(unix_ms=time.time_ns() // 1_000_000,
                                          process_peak_commit_bytes=info.process_peak,
                                          job_peak_commit_bytes=info.job_peak))
            return info

        job = create_job(None, None)  # Unnamed, non-inherited, owned only by this wrapper.
        check(job)
        limits = Extended()
        limits.basic.flags = 0x100 | 0x200 | 0x2000  # PROCESS_MEMORY, JOB_MEMORY, KILL_ON_JOB_CLOSE
        limits.process_limit = limits.job_limit = args.commit_bytes
        check(set_job(job, 9, c.byref(limits), c.sizeof(limits)))
        actual = sample()
        if actual.basic.flags & limits.basic.flags != limits.basic.flags or actual.job_limit != args.commit_bytes or actual.process_limit != args.commit_bytes:
            raise RuntimeError('Job limit read-back mismatch.')
        report['effective_commit_bytes'] = actual.job_limit

        def run(argv, name, timeout):
            proc = Process()
            started = time.monotonic()
            try:
                with (output / (name + '.log')).open('wb') as log, open(os.devnull, 'rb') as null:
                    startup = Startup(cb=c.sizeof(Startup), flags=0x100)
                    startup.stdout = startup.stderr = msvcrt.get_osfhandle(log.fileno())
                    startup.stdin = msvcrt.get_osfhandle(null.fileno())
                    os.set_handle_inheritable(startup.stdout, True)
                    os.set_handle_inheritable(startup.stdin, True)
                    line = c.create_unicode_buffer(subprocess.list2cmdline(argv))
                    check(create(None, line, None, None, True, 0x4 | 0x08000000, None, None, c.byref(startup), c.byref(proc)))
                    check(assign(job, proc.process))
                    belongs = w.BOOL()
                    check(member(proc.process, job, c.byref(belongs)))
                    if not belongs.value:
                        raise RuntimeError('Suspended child is not in the owned job.')
                    report['processes'].append(dict(name=name, pid=proc.pid, membership_verified=True, assigned_before_resume=True))
                    if resume(proc.thread) == 0xffffffff:
                        raise c.WinError(c.get_last_error())
                    while True:
                        state = wait(proc.process, 250)
                        sample()
                        if state == 0:
                            break
                        if state != 258:
                            raise c.WinError(c.get_last_error())
                        if time.monotonic() - started > timeout:
                            raise TimeoutError(f'{name} exceeded its deadline.')
                    exit_code = w.DWORD()
                    check(code(proc.process, c.byref(exit_code)))
                    return exit_code.value, time.monotonic() - started
            finally:
                if proc.process:
                    # Also covers assignment failures before the suspended child ran.
                    if wait(proc.process, 0) == 258:
                        terminate(proc.process, 1)
                        wait(proc.process, 10000)
                    close(proc.process)
                if proc.thread:
                    close(proc.thread)

        # An over-limit private allocation must actually fail, not just read back
        # a configured number. No pages are touched if enforcement is broken.
        canary = ("import ctypes as c,sys,json; k=c.WinDLL('kernel32',use_last_error=True); "
                  "k.VirtualAlloc.argtypes=[c.c_void_p,c.c_size_t,c.c_ulong,c.c_ulong]; "
                  "k.VirtualAlloc.restype=c.c_void_p; "
                  f"p=k.VirtualAlloc(None,{args.commit_bytes + 4096},0x3000,4); e=c.get_last_error(); "
                  "print(json.dumps({'refused':not bool(p),'win32_error':e})); "
                  "sys.exit(0 if not p and e in (8,1455) else 1)")
        canary_code, _ = run([sys.executable, '-c', canary], 'canary', 30)
        report['canary'] = json.loads((output / 'canary.log').read_text().strip())
        if canary_code != 0 or not report['canary']['refused']:
            raise RuntimeError('Allocation refusal canary did not confirm enforcement.')
        report['enforcement'] = 'windows-job-private-commit'
        result = 1
        exit_code, seconds = run(command, 'process', args.timeout)
        report.update(exit_code=exit_code, seconds=seconds, passed=exit_code == 0)
        result = 0 if report['passed'] else 1
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        if job:
            if not terminate_job(job, 1):
                report['errors'].append(f'Job cleanup: {c.WinError(c.get_last_error())}')
                report['passed'], result = False, 1
            close(job)
        (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('enforcement', 'passed', 'errors')}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
