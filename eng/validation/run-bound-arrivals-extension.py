#!/usr/bin/env python3
"""Run a separate bounded arrivals extension, then release a held agent result.

No model, build or SSH connection is started. The exact held 46-case completion
is published unchanged; extension outcome and aggregate qualification are separate.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def active_waiter_deadline(runtime, completion, nonce):
    for entry in runtime.get('suites', []):
        argv = entry.get('command', [])
        if not any(part.endswith('/wait-external-suite.py') for part in argv):
            continue
        if argv[argv.index('--nonce') + 1] != nonce:
            continue
        if Path(argv[argv.index('--completion') + 1]).resolve() != completion.resolve():
            continue
        require('exit_code' not in entry, 'The matching external waiter has already completed')
        return entry['started_at_unix'] + float(argv[argv.index('--timeout') + 1])
    raise ValueError('No matching active external waiter in the runtime profile')


def publish_bytes_fresh(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication, never replaces an existing completion.
    finally:
        Path(temporary).unlink(missing_ok=True)


def fits_waiter_budget(deadline, wall_seconds, margin_seconds, now):
    return deadline - now >= wall_seconds + margin_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'manifest', 'ready-identity', 'owner-report', 'held-completion',
                 'completion', 'server-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('manifest-sha256', 'owner-report-sha256', 'nonce'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(digest(args.manifest) == args.manifest_sha256, 'Extension manifest changed')
    manifest = read(args.manifest)
    require(args.nonce == manifest['nonce'], 'Wrong extension campaign nonce')
    require(manifest['repeats'] == 3 and manifest['request_timeout_seconds'] == 1800,
            'Preserved three-repeat workload bounds are required')
    require(0 < manifest['wall_seconds'] <= 3600 and manifest['coordination_margin_seconds'] >= 120,
            'Extension needs a bounded wall time and coordination margin')
    require(not args.output.exists() and not os.path.lexists(args.completion), 'Use fresh extension/completion outputs')
    for name, expected in manifest['program_sha256'].items():
        require(digest(args.repo / name) == expected, 'Pinned extension/import changed: ' + name)
    require(digest(args.owner_report) == args.owner_report_sha256, 'Completed owner report changed')
    owner_report = read(args.owner_report)
    require(owner_report.get('finished_at_unix') and owner_report.get('ssh_agent_stopped') is True
            and owner_report.get('owned_tunnel_stopped') is True, 'The 46-case owner must finish and stop its tunnel first')
    held_bytes = args.held_completion.read_bytes()
    held = json.loads(held_bytes)
    require(held == owner_report.get('completion') and held.get('nonce') == args.nonce,
            'Held completion does not match the finished owner/nonce')
    ready = read(args.ready_identity)
    require(ready.get('nonce') == args.nonce and owner_report.get('nonce') == args.nonce, 'Wrong campaign nonce')
    require(owner_report.get('ready_identity') == ready, 'Finished owner belongs to a different ready identity')
    require(ready['expected_native_sha256'] == manifest['native_sha256'], 'Wrong ready native pin')
    managed = ready['profile']['qualified_managed_build']
    require(managed['sha256'] == manifest['managed_build_sha256'], 'Wrong ready managed build pin')
    require({Path(name).name: value for name, value in ready['available_managed_libraries'].items()}
            == managed['managed_assemblies'], 'Ready managed identities differ from the reviewed build')
    require(digest(Path(managed['path'])) == manifest['managed_build_sha256'], 'Managed build record changed')
    runtime_path = Path(ready['runtime_profile_path'])
    require(runtime_path.resolve() == Path(manifest['runtime_profile_path']).resolve(), 'Wrong extension runtime profile')
    runtime = read(runtime_path)
    require(Path(owner_report['remote_results']).resolve() == runtime_path.parent.resolve(), 'Finished owner belongs to a different runtime')
    require(runtime['status'] == 'running' and runtime['server_pid'] == ready['remote_pid'], 'Server runtime identity changed')
    require(args.completion.parent.resolve() == runtime_path.parent.resolve()
            and args.held_completion.parent.resolve() == runtime_path.parent.resolve(), 'Completion paths must belong to this runtime')
    require(args.server_log.resolve() == runtime_path.with_name('server.log').resolve(), 'Wrong server log')
    deadline = active_waiter_deadline(runtime, args.completion, args.nonce)
    spec = importlib.util.spec_from_file_location('extension_identity', args.repo / 'eng/validation/capture-release-server-identity.py')
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    capture.OWNER.verify_ready_identity(ready, args.nonce)
    before = capture.observe(ready['remote_pid'], ready['remote_port'])
    capture.OWNER.verify_identity(before, ready, manifest['native_sha256'], ready['remote_port'])
    args.output.mkdir(parents=True)
    report = {'status': 'starting', 'release_qualified': False, 'nonce': args.nonce,
              'scope': 'Separate arrivals/cancellation extension, not part of original nine suites or 46 agent cases; HTTP overlap does not establish fused native batching.',
              'manifest_sha256': args.manifest_sha256, 'owner_report_sha256': args.owner_report_sha256,
              'held_completion_sha256': hashlib.sha256(held_bytes).hexdigest(), 'ready_identity_sha256': digest(args.ready_identity),
              'native_sha256': manifest['native_sha256'], 'managed_build': managed, 'before_identity': before,
              'started_at_unix': time.time(), 'waiter_deadline_unix': deadline,
              'remaining_waiter_seconds_at_start': deadline - time.time(), 'program_sha256': manifest['program_sha256']}
    save = lambda: (args.output / 'extension-qualification.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    process = None
    extension_passed = False
    try:
        needed = manifest['wall_seconds'] + manifest['coordination_margin_seconds']
        if not fits_waiter_budget(deadline, manifest['wall_seconds'], manifest['coordination_margin_seconds'], time.time()):
            report.update(status='deferred-insufficient-waiter-budget', executed=False,
                          required_remaining_seconds=needed)
        else:
            command = [sys.executable, str(args.repo / 'eng/validation/validate-release-arrivals.py'),
                       '--url', 'http://127.0.0.1:' + str(ready['remote_port']), '--model', ready['model_id'],
                       '--repeats', str(manifest['repeats']), '--timeout', str(manifest['request_timeout_seconds']),
                       '--server-log', str(args.server_log), '--output', str(args.output / 'arrivals.json')]
            report.update(status='running', executed=True, command=command,
                          wall_timeout_seconds=manifest['wall_seconds'])
            save()
            with (args.output / 'arrivals.log').open('w') as log:
                process = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, BENCH_CONFIG=str(args.repo / 'benchmarks/engine_comparison/benchmark_config.json')))
                report['extension_pid'] = process.pid
                save()
                try:
                    report['exit_code'] = process.wait(timeout=manifest['wall_seconds'])
                except subprocess.TimeoutExpired:
                    report['timed_out'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    report['exit_code'] = process.returncode
            result = read(args.output / 'arrivals.json')
            report['request_run_complete'] = result.get('run_complete') is True
            cancellations = [row for row in result.get('cases', []) if row.get('scenario') == 'client-disconnect']
            extension_passed = (report['exit_code'] == 0 and result.get('run_complete') is True
                                and len(result['cases']) == 21 and len(result['waves']) == 3
                                and all(row['status'] == 'ok' for row in result['cases'])
                                and all(row['actual_overlap'] and row['replacement_while_peer_active'] for row in result['waves'])
                                and len(cancellations) == 3
                                and all(row.get('peer_active_at_disconnect') is True and row.get('session_cleanup_passed') is True
                                        for row in cancellations))
            report.update(status='passed' if extension_passed else 'failed',
                          arrivals_sha256=digest(args.output / 'arrivals.json'))
    except Exception as error:
        report.update(status='failed', error=repr(error))
    finally:
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            require(process is None or process.poll() is not None, 'Extension process still running')
            after = capture.observe(ready['remote_pid'], ready['remote_port'])
            capture.OWNER.verify_identity(after, before, manifest['native_sha256'], ready['remote_port'])
            report['after_identity'] = after
            require(not report.get('timed_out'), 'Timed-out clients do not prove server requests drained; automatic hand-off is withheld')
            require(not report.get('executed') or report.get('request_run_complete'),
                    'An incomplete request run needs owner review before completion hand-off')
            for name, expected in manifest['program_sha256'].items():
                require(digest(args.repo / name) == expected, 'Extension/import changed during execution: ' + name)
            require(args.held_completion.read_bytes() == held_bytes, 'Held completion changed during extension')
            require(digest(args.owner_report) == args.owner_report_sha256, 'Completed owner report changed during extension')
            require(digest(args.ready_identity) == report['ready_identity_sha256'], 'Ready identity changed during extension')
            require(digest(Path(managed['path'])) == manifest['managed_build_sha256'], 'Managed build record changed during extension')
            require(time.time() < deadline - 30, 'Too close to waiter expiry for safe hand-off')
            active_waiter_deadline(read(runtime_path), args.completion, args.nonce)
            report['after_identity_checks_passed'] = True
            report['extension_passed'] = extension_passed
            original_passed = (held.get('client_exit_code') == 0 and held.get('independent_code_exit_code') == 0
                               and held.get('remote_binding_postcheck_passed') is True
                               and re.fullmatch('[0-9a-f]{64}', held.get('client_report_sha256') or '') is not None)
            report['original_agent_completion_passed'] = original_passed
            report['aggregate_passed'] = bool(original_passed and extension_passed)
            report['completion_handoff'] = 'Publishing original agent completion unchanged; extension outcome must be reviewed separately.'
            report['finished_at_unix'] = time.time()
            save()  # Persist separate failure/defer/aggregate outcome before releasing the waiter.
            publish_bytes_fresh(args.completion, held_bytes)
            report['completion_published'] = True
            report['published_completion_sha256'] = digest(args.completion)
        except Exception as error:
            report.update(status='failed', handoff_error=repr(error), after_identity_checks_passed=False)
        report['finished_at_unix'] = time.time()
        save()
    print(json.dumps({key: report.get(key) for key in ('status', 'extension_passed', 'aggregate_passed', 'completion_published', 'release_qualified')}))
    return (0 if report.get('aggregate_passed') and report.get('completion_published') else
            2 if report['status'] == 'deferred-insufficient-waiter-budget' and report.get('completion_published') else 1)


if __name__ == '__main__':
    raise SystemExit(main())
