#!/usr/bin/env python3
"""Run one directly owned validation process with a hard deadline and durable outcome.

This is for a single-process probe (or server), not a shell/process-tree launcher.
Native calls cannot safely be cancelled in-process. A deadline kills only this
child and records incomplete execution; it never turns partial evidence into PASS.

Default invocations retain the original deadline/exit behavior. Opt-in --telemetry
adds live GgmlOps module discovery, an explicit SHA256 pin, application-file
before/after hashes, and sampled RSS/private/board VRAM. It requires psutil and
nvidia-smi on Windows/Linux; samples are not allocator peaks or hard limits.
Native observation errors reject the owned process; missing telemetry never
qualifies as zero usage. Scenario token/EOS/semantic gates remain separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

RUNTIME_CAPTURE_SCOPE = dict(Schema='tensorsharp.probe-runtime-capture.v1', Traversal='recursive',
    FileKinds=['dll', 'exe', 'deps.json', 'runtimeconfig.json', 'ptx', 'so', 'versioned-so', 'dylib'],
    ReparsePoints='reject')


def runtime_artifact(path):
    name = Path(path).name.lower()
    return name.endswith(('.dll', '.exe', '.deps.json', '.runtimeconfig.json', '.ptx', '.so', '.dylib')) \
        or re.search(r'\.so\.[0-9]+(?:\.[0-9]+)*$', name) is not None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


class ProbeTelemetry:
    """Optional live native identity and sampled memory; never model qualification."""
    def __init__(self, managed_directory, expected_native_sha256, bench=None):
        self.directory = Path(managed_directory).resolve()
        if len(expected_native_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_native_sha256):
            raise ValueError('Expected a lowercase native SHA256 digest')
        if not self.directory.is_dir():
            raise ValueError('Managed directory does not exist')
        if bench is None:
            source = Path(__file__).with_name('multimodel-quality-bench.py')
            spec = importlib.util.spec_from_file_location('bounded_probe_bench', source)
            bench = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bench)
        self.bench, self.expected = bench, expected_native_sha256
        self.stop = threading.Event()
        self.native, self.error, self.sampler, self.worker = None, None, None, None
        self.report = {'passed': False, 'native_verified': False, 'loaded_managed_modules_verified': False,
                       'capture_scope': RUNTIME_CAPTURE_SCOPE,
                       'expected_native_sha256': self.expected,
                       'runner_sha256': digest(__file__),
                       'sampler_sha256': digest(bench.__file__) if getattr(bench, '__file__', None) else None,
                       'native_observation_mode': 'Stop module polling after first verified live observation; hash observed file again after exit.',
                       'scope': 'Native path observed once in the owned live process, with on-disk hash at observation and after exit; later module replacements/unloads are not continuously monitored. Declared recursive runtime artifact set hashed before/after, not loaded managed-module enumeration. Memory is sampled; board VRAM includes other processes. Sampling/identity overhead is unquantified. No model hashes, hard-limit or semantic/performance qualification.'}

    def managed_files(self):
        files = {}
        for path in sorted(self.directory.rglob('*')):
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
                raise ValueError('Runtime capture refuses reparse/symlink entries: ' + str(path))
            if path.is_file() and runtime_artifact(path):
                files[str(path)] = digest(path)
        if not any(name.endswith('.dll') for name in files):
            raise ValueError('Managed directory contains no DLLs')
        return files

    def prepare(self, output):
        self.output = output
        self.report['managed_files_before'] = self.managed_files()
        self.report['environment'] = {k: v for k, v in os.environ.items()
            if k.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'NCCL_', 'OMP_', 'GOMP_'))
            or k in ('CUDA_VISIBLE_DEVICES', 'MAX_CONTEXT', 'KV_CACHE_DTYPE')}

    def observe_native(self, pid):
        paths = self.bench.native_module_paths(pid)
        if not paths:
            return
        if len(paths) != 1:
            raise ValueError('Expected exactly one loaded native library: ' + repr(paths))
        path = str(Path(paths[0]).resolve())
        if self.native is not None:
            if path != self.native['path']:
                raise ValueError('Loaded native path changed during execution')
            return
        actual = digest(path)
        if actual != self.expected:
            raise ValueError('Loaded native SHA256 differs from the required pin: ' + path)
        self.native = {'path': path, 'sha256': actual, 'observed_utc': datetime.now(timezone.utc).isoformat(),
                       'pid': pid, 'capture_phase': 'owned process alive'}
        (self.output / 'native-observation.json').write_text(json.dumps(self.native, indent=2) + '\n', encoding='utf-8')

    def start(self, child):
        self.sampler = self.bench.Sampler(self.output / 'memory.jsonl')
        self.sampler.process = self.bench.psutil.Process(child.pid)
        self.sampler.thread.start()
        def observe():
            while not self.stop.is_set() and child.poll() is None:
                try:
                    self.observe_native(child.pid)
                    if self.native is not None:
                        return  # Avoid recurrent PowerShell overhead on host-offloaded models.
                except Exception as error:
                    # A process disappearing during module enumeration is unknown,
                    # not evidence that a different native binary was loaded.
                    if isinstance(error, ValueError) or child.poll() is None:
                        self.error = repr(error)
                    return
                self.stop.wait(1)
        self.worker = threading.Thread(target=observe, daemon=True)
        self.worker.start()

    def finish(self):
        self.stop.set()
        if self.sampler is not None:
            self.sampler.stop.set()
            if self.sampler.thread.ident is not None:
                self.sampler.thread.join(timeout=15)
        if self.worker is not None:
            self.worker.join(timeout=35)
        errors = [self.error] if self.error else []
        if ((self.worker is not None and self.worker.is_alive()) or
                (self.sampler is not None and self.sampler.thread.is_alive())):
            errors.append('Telemetry worker did not stop; observations are incomplete')
        rows = list(self.sampler.rows) if self.sampler is not None else []
        if self.sampler is not None:
            errors.extend(self.sampler.errors)
        peak = lambda key: max((r['memory'][key] for r in rows if key in r.get('memory', {})), default=None)
        self.report['memory'] = {'samples': len(rows), 'peak_rss_bytes': peak('rss'),
            'peak_private_bytes': peak('private'),
            'peak_board_mib': max((d['used_mib'] for r in rows for d in r.get('gpu', [])), default=None)}
        self.report['loaded_native'] = self.native
        try:
            self.report['managed_files_after'] = self.managed_files()
            self.report['managed_files_unchanged'] = self.report['managed_files_before'] == self.report['managed_files_after']
            self.report['native_verified'] = self.native is not None and digest(self.native['path']) == self.expected
        except Exception as error:
            errors.append(repr(error))
        self.report['errors'] = errors
        self.report['passed'] = (not errors and self.report.get('managed_files_unchanged') is True
            and self.report['native_verified'] and peak('rss') is not None and self.report['memory']['peak_board_mib'] is not None)
        return self.report


def run(command, output, timeout, telemetry=None):
    output.mkdir(parents=True, exist_ok=False)
    report = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": timeout, "complete": False, "timed_out": False,
              "exit_code": None, "quality_passed": None, "cleanup_errors": []}

    def publish():
        temporary = output / "execution.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "execution.json")

    publish()
    start = time.monotonic()
    child = None
    try:
        if telemetry is not None:
            telemetry.prepare(output)
            report['telemetry'] = telemetry.report
            publish()
        with (output / "process.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = time.monotonic() + timeout
            report["pid"] = child.pid
            publish()
            if telemetry is not None:
                telemetry.start(child)
            try:
                if telemetry is None:
                    report["exit_code"] = child.wait(timeout=max(0, deadline - time.monotonic()))
                    report["complete"] = True
                else:
                    while True:
                        if telemetry.error:
                            report['error'] = telemetry.error
                            child.kill()
                            report['exit_code'] = child.wait(timeout=15)
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout)
                        try:
                            report['exit_code'] = child.wait(timeout=min(.2, remaining))
                            report['complete'] = True
                            break
                        except subprocess.TimeoutExpired:
                            continue
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
                child.kill()
                report["exit_code"] = child.wait(timeout=15)
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if child is not None:
            try:
                if child.poll() is None:
                    child.kill()
                report['exit_code'] = child.wait(timeout=15)
            except Exception as error:
                report['cleanup_errors'].append('owned child: ' + repr(error))
        if telemetry is not None:
            try:
                report['telemetry'] = telemetry.finish()
            except Exception as error:
                report['telemetry'] = dict(telemetry.report, passed=False, finish_error=repr(error))
                report['cleanup_errors'].append('telemetry.finish: ' + repr(error))
        report["wall_seconds"] = time.monotonic() - start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["qualification"] = "Exit zero is completed execution only; consult the probe's numerical/semantic evidence."
        publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument('--telemetry', action='store_true', help='Opt-in live native identity and sampled memory on Windows/Linux; requires psutil and nvidia-smi')
    parser.add_argument('--managed-directory', type=Path, help='Immutable application directory whose declared recursive runtime artifacts are hashed before/after')
    parser.add_argument('--expected-native-sha256', help='Required lowercase loaded-native SHA256 when telemetry is enabled')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Supply a nonempty command and positive finite timeout")
    if args.telemetry != bool(args.managed_directory and args.expected_native_sha256) or (not args.telemetry and (args.managed_directory or args.expected_native_sha256)):
        parser.error('--telemetry requires both --managed-directory and --expected-native-sha256; they are invalid without it')
    telemetry = ProbeTelemetry(args.managed_directory, args.expected_native_sha256) if args.telemetry else None
    report = run(command, args.output.resolve(), args.timeout, telemetry)
    print(json.dumps(report, indent=2))
    if report['timed_out']:
        return 124
    if report['cleanup_errors']:
        return report['exit_code'] or 1
    if args.telemetry and not report['telemetry']['passed']:
        return report['exit_code'] or 1
    return report['exit_code']


if __name__ == "__main__":
    raise SystemExit(main())
