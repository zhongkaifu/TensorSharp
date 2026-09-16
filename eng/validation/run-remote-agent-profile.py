#!/usr/bin/env python3
"""Run an already-built model-free AgentHost client, then independent code checks.

This runner never builds, starts a model, or opens an SSH connection. The root
stage owns the exclusive remote lane, tunnel, and native PID mapping checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dotnet', 'assembly', 'fixtures', 'identity', 'skills-root', 'artifact-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('endpoint', 'model', 'expected-native-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--scripted-fixture', action='store_true', help='Local plumbing check only; no model quality or native identity qualification')
    parser.add_argument('--artifact-port', type=int, default=18481)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--max-rounds', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--bwrap', default='/usr/local/bin/bwrap')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / 'profile.json'
    if summary.exists() or (args.output / 'actual-agent-workflows.json').exists():
        parser.error('Use a fresh output directory')
    command = [str(args.dotnet), str(args.assembly)]
    for name in ('fixtures', 'identity', 'skills_root', 'artifact_root', 'output', 'endpoint', 'model',
                 'expected_native_sha256', 'artifact_port', 'timeout', 'max_rounds', 'max_tokens'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    if args.thinking:
        command.append('--thinking')
    if args.scripted_fixture:
        command.append('--scripted-fixture')
    report = {'status': 'running', 'release_qualified': False, 'command': command,
              'topology': 'WSL local real AgentHost tools; remote model over pre-existing loopback tunnel',
              'started_at_unix': time.time(), 'harness_sha256': hashlib.sha256(args.assembly.read_bytes()).hexdigest(),
              'remote_identity_sha256': hashlib.sha256(args.identity.read_bytes()).hexdigest(),
              'remote_post_run_binding_check': 'required from the owning root stage'}
    if args.scripted_fixture:
        report.update(scripted_fixture=True, topology='Scripted loopback endpoint; real local tools and sandbox; no model',
                      remote_post_run_binding_check='Not applicable: synthetic fixture, no remote process')
    summary.write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'client.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report['client_exit_code'] = result.returncode
    workflow = args.output / 'actual-agent-workflows.json'
    if workflow.exists():
        verification = [sys.executable, str(Path(__file__).with_name('verify-agent-code-artifacts.py')),
                        '--workflow-report', str(workflow), '--artifact-store', str(args.artifact_root),
                        '--output', str(args.output / 'independent-code-checks.json'), '--bwrap', args.bwrap]
        with (args.output / 'independent-code-checks.log').open('w') as log:
            checked = subprocess.run(verification, stdout=log, stderr=subprocess.STDOUT)
        report['independent_code_exit_code'] = checked.returncode
    report['status'] = ('workflow-and-code-passed-pending-remote-postcheck'
                        if result.returncode == 0 and report.get('independent_code_exit_code') == 0 else 'failed')
    if args.scripted_fixture and report['status'] != 'failed':
        report['status'] = 'scripted-plumbing-passed-no-model-qualification'
    report['finished_at_unix'] = time.time()
    summary.write_text(json.dumps(report, indent=2) + '\n')
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
