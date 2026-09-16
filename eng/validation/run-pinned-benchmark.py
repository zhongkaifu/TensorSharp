#!/usr/bin/env python3
"""Own one explicitly pinned benchmark process and retain unqualified evidence.

This runner establishes process/file identity and collects resource samples.
Scenario-specific comparators must still validate engagement, full outputs and
performance. A successful process alone never qualifies a release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from release_application_identity import check_application, digest, read_pinned_json


def check_managed_dependencies(application, assembly):
    """Check the entry point's declared app-local runtime files before loading weights.

    Framework assemblies are supplied by .NET and are not listed as runtime
    assets in this manifest. A server directory cannot safely stand in for a
    console benchmark's own dependency closure.
    """
    application, assembly = Path(application).resolve(), Path(assembly).resolve()
    if assembly.parent != application or not assembly.is_file():
        raise ValueError('Benchmark assembly must belong to the pinned application')
    manifest = assembly.with_suffix('.deps.json')
    data = json.loads(manifest.read_text())
    target = data['targets'][data['runtimeTarget']['name']]
    found, missing = {}, []
    for library in target.values():
        for name in library.get('runtime', {}):
            if not name.endswith('.dll'):
                continue
            candidates = (application / name, application / Path(name).name)
            path = next((p for p in candidates if p.is_file() and p.resolve().is_relative_to(application)), None)
            if path is None:
                missing.append(name)
            else:
                found[str(path.relative_to(application))] = digest(path)
    if missing:
        raise ValueError('Missing benchmark runtime dependencies: ' + ', '.join(sorted(missing)))
    return found


def wait_command_line(read_bytes, poll, expected, timeout=5):
    """An exec transition can briefly expose an empty /proc/PID/cmdline."""
    deadline = time.monotonic() + timeout
    while True:
        raw = read_bytes()
        if raw:
            observed = raw.decode().rstrip('\0').split('\0')
            if observed != expected:
                raise ValueError('Observed process command differs from the pinned invocation')
            return observed
        if poll() is not None:
            raise RuntimeError('Benchmark exited before command identity could be verified')
        if time.monotonic() >= deadline:
            raise TimeoutError('Benchmark command identity remained empty')
        time.sleep(.01)


def inspect_maps(text, expected, native, native_sha):
    """Reject unexpected or deleted mapped project assemblies and native code."""
    mapped = {}
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        name = fields[5]
        deleted = name.endswith(' (deleted)')
        clean = name[:-10] if deleted else name
        path = Path(clean)
        relevant = path.name == 'libGgmlOps.so' or (path.name.startswith('TensorSharp.') and path.suffix == '.dll')
        if not relevant:
            continue
        if deleted:
            raise ValueError('Mapped input deleted: ' + clean)
        if path.name == 'libGgmlOps.so' and path != native:
            raise ValueError('Unexpected mapped native library: ' + clean)
        wanted = native_sha if path == native else expected.get(str(path))
        if wanted is None:
            raise ValueError('Unexpected mapped input: ' + clean)
        if clean not in mapped:
            actual = digest(path)
            if actual != wanted:
                raise ValueError('Mapped input changed: ' + clean)
            mapped[clean] = actual
    return mapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = read_pinned_json(a.plan, a.plan_sha256, 'Benchmark plan')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    app = Path(plan['application']).resolve()
    native = app / 'libGgmlOps.so'
    native_sha = plan['native_sha256']
    report = {'status': 'running', 'release_qualified': False, 'performance_qualified': False,
              'runner_sha256': digest(__file__), 'plan_sha256': a.plan_sha256,
              'plan': plan, 'started_unix': time.time(), 'mapped_inputs': {},
              'limitations': ['Process success is not output/engagement or performance acceptance.',
                             'Resource telemetry is sampled, not an allocator peak measurement.']}
    process = telemetry = None

    def save():
        temporary = out / 'owner.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'owner.json')

    def audit_inputs():
        result = check_application(app, plan['application_manifest'], plan['application_manifest_sha256'])
        if digest(native) != native_sha:
            raise ValueError('Native library differs from plan')
        assets = {}
        for name, expected in plan['assets_sha256'].items():
            actual = digest(name)
            if actual != expected:
                raise ValueError('Checkpoint/head/media input differs: ' + name)
            assets[name] = actual
        return {'application': result, 'native_sha256': native_sha, 'assets_sha256': assets}

    try:
        save()
        argv = plan['argv']
        if len(argv) < 2:
            raise ValueError('Benchmark invocation requires an entry assembly')
        report['managed_dependency_files_sha256'] = check_managed_dependencies(app, argv[1])
        save()
        report['identity_before'] = audit_inputs()
        expected = {str(app / name): value for name, value in
                    report['identity_before']['application']['non_native_files_sha256'].items()}
        if len(argv) < 2 or Path(argv[1]).resolve().parent != app or str(Path(argv[1]).resolve()) not in expected:
            raise ValueError('Benchmark assembly must belong to the pinned application')
        env = {**os.environ, **plan['env'], 'PYTHONDONTWRITEBYTECODE': '1'}
        report['environment_sha256'] = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
        started = time.monotonic()
        with (out / 'process.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=app, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            proc = Path('/proc') / str(process.pid)
            report['pid'] = process.pid
            report['start_ticks'] = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19])
            report['command_line'] = wait_command_line((proc / 'cmdline').read_bytes, process.poll, argv)
            report['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            telemetry = subprocess.Popen([sys.executable, str(Path(__file__).with_name('sample-release-telemetry.py')),
                '--pid', str(process.pid), '--output', str(out / 'telemetry.jsonl'), '--interval', '1'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            previous_maps = None
            while process.poll() is None:
                try:
                    maps = (proc / 'maps').read_text()
                    if maps != previous_maps:
                        report['mapped_inputs'].update(inspect_maps(maps, expected, native, native_sha))
                        previous_maps = maps
                        save()
                except FileNotFoundError:
                    if process.poll() is None:
                        raise
                if time.monotonic() - started > plan.get('timeout_seconds', 7200):
                    raise TimeoutError('Owned benchmark timed out')
                time.sleep(.1)
            report['exit_code'] = process.wait()
        if report['exit_code'] != 0:
            raise RuntimeError('Benchmark failed with exit ' + str(report['exit_code']))
        if report['mapped_inputs'].get(str(native)) != native_sha:
            raise ValueError('Expected native library was never observed in the process')
        if not any(Path(name).name == 'TensorSharp.Models.dll' for name in report['mapped_inputs']):
            raise ValueError('Expected managed model assembly was never observed')
        report['identity_after'] = audit_inputs()
        report['status'] = 'process-completed-scenario-gates-pending'
    except Exception as error:
        report.update(status='failed', error=repr(error))
    finally:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            report['exit_code'] = process.returncode
            report['process_gone'] = not (Path('/proc') / str(process.pid)).exists()
        if telemetry is not None:
            try:
                telemetry.wait(timeout=15)
            except subprocess.TimeoutExpired:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            report['telemetry_exit_code'] = telemetry.returncode
            if telemetry.returncode != 0:
                report.update(status='failed', telemetry_error='Telemetry failed')
        report['finished_unix'] = time.time()
        save()
    print(report['status'], flush=True)
    return int(report['status'] != 'process-completed-scenario-gates-pending')


if __name__ == '__main__':
    raise SystemExit(main())
