#!/usr/bin/env python3
"""Own an ephemeral WSL SSH agent/tunnel and the scheduled 46-case client.

Does not start a remote model. Run only after the lane owner has explicitly
scheduled the already-ready server. Reads the existing private key through
ssh-add stdin; never copies or logs key contents. Completion is nonce-bound and
published atomically only after recording the remote process postcheck.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

# The preserved 46-case fixture contains 16 sequential waves. Each case keeps
# its 1,200-second gate; this owner budget adds time for setup and independent
# artifact checks without aborting a valid later wave. The VM waits 21,600 s.
CAMPAIGN_TIMEOUT_SECONDS = 21000
import urllib.error


REMOTE_OBSERVER = r'''
import hashlib,json,os,sys,time
from pathlib import Path
pid,port=int(sys.argv[1]),int(sys.argv[2])
proc=Path('/proc')/str(pid)
def start_ticks(): return int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
start=start_ticks()
maps=(proc/'maps').read_text()
native=set();managed=set()
for line in maps.splitlines():
    parts=line.split(maxsplit=5)
    if len(parts)<6 or not parts[5].startswith('/'):continue
    path=parts[5]
    if 'GgmlOps' in path or 'libggml' in path:
        if path.endswith(' (deleted)'):raise RuntimeError('A mapped native library was deleted')
        native.add(path)
    elif Path(path).name.startswith('TensorSharp.') and path.endswith('.dll'):managed.add(path)
sockets=set()
for fd in (proc/'fd').iterdir():
    try:link=os.readlink(fd)
    except FileNotFoundError:continue
    if link.startswith('socket:['):sockets.add(link[8:-1])
listeners=[]
for family in ('tcp','tcp6'):
    network=proc/'net'/family
    if not network.exists():continue
    for line in network.read_text().splitlines()[1:]:
        fields=line.split()
        if int(fields[1].rsplit(':',1)[1],16)==port and fields[3]=='0A' and fields[9] in sockets:
            listeners.append({'family':family,'address':fields[1],'inode':fields[9]})
result={'remote_pid':pid,'remote_start_ticks':start,'remote_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'captured_at_unix':time.time(),'command_line':(proc/'cmdline').read_bytes().decode().split('\0')[:-1],
        'mapped_native_libraries':{p:digest(p) for p in sorted(native)},
        'mapped_managed_libraries':{p:digest(p) for p in sorted(managed)},
        'available_managed_libraries':{str(p):digest(p) for parent in sorted({Path(p).parent for p in managed})
                                       for p in sorted(parent.glob('TensorSharp.*.dll'))},
        'native_maps_sha256':hashlib.sha256('\n'.join(line for line in maps.splitlines() if any(p in line for p in native)).encode()).hexdigest(),
        'owned_listening_sockets':listeners,'remote_port':port}
if start_ticks()!=start:raise RuntimeError('Process identity changed during capture')
print(json.dumps(result))
'''

REMOTE_COMPLETE = r'''
import json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); result=json.load(sys.stdin)
if not path.parent.is_dir():raise RuntimeError('Completion parent does not exist')
if path.exists():raise RuntimeError('Completion already exists')
temporary=path.with_name(path.name+'.'+result['nonce']+'.tmp')
with temporary.open('x') as output:
    json.dump(result,output,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
if path.exists():raise RuntimeError('Completion appeared concurrently')
os.replace(temporary,path)
print(json.dumps({'completion':str(path),'nonce':result['nonce']}))
'''


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_identity(actual, expected, native_hash, port):
    if actual['remote_pid'] != expected['remote_pid'] or actual['remote_start_ticks'] != expected['remote_start_ticks']:
        raise ValueError('Remote PID/start time changed')
    if expected.get('remote_boot_id') and actual['remote_boot_id'] != expected['remote_boot_id']:
        raise ValueError('Remote machine boot identity changed')
    if actual['mapped_native_libraries'] != expected['mapped_native_libraries']:
        raise ValueError('Mapped native library paths or hashes changed')
    ops = {value for name, value in actual['mapped_native_libraries'].items() if 'GgmlOps' in name}
    if ops != {native_hash}:
        raise ValueError('Actual GgmlOps mapping does not match explicit native pin')
    if actual['remote_port'] != port or not actual['owned_listening_sockets']:
        raise ValueError('The pinned remote process does not own the endpoint listening socket')
    if expected.get('available_managed_libraries'):
        available = expected['available_managed_libraries']
        if actual.get('available_managed_libraries') != available:
            raise ValueError('Available managed library paths or hashes changed')
        if any(available.get(path) != digest for path, digest in actual['mapped_managed_libraries'].items()):
            raise ValueError('A mapped managed library is outside the pinned application manifest')
    elif expected.get('mapped_managed_libraries') and actual['mapped_managed_libraries'] != expected['mapped_managed_libraries']:
        raise ValueError('Mapped managed library paths or hashes changed')


def verify_ready_identity(ready, nonce):
    if ready.get('scripted_fixture') or ready.get('remote_pid', 0) <= 0 or not ready.get('remote_start_ticks'):
        raise ValueError('Actual owner-supplied ready process identity is required')
    if ready.get('nonce') != nonce:
        raise ValueError('Ready process identity nonce does not match the scheduled campaign')


def verify_release_fixtures(fixtures):
    cases = fixtures['cases']
    original_names = {'skill_selection', 'skill_run', 'skill_script_run', 'shell_run', 'code_generation_run', 'code_edit_run'}
    distinct_names = {'skill_script_run', 'shell_run', 'code_generation_run', 'code_edit_run'}
    expected = {('original', scenario, trial) for scenario in original_names
                for trial in ('c1-i0', 'c4-i0', 'c4-i1', 'c4-i2', 'c4-i3')}
    expected |= {('distinct', scenario, f'c4-i{index}') for scenario in distinct_names for index in range(4)}
    if len(cases) != 46 or {(c['variant'], c['scenario'], c['trial']) for c in cases} != expected:
        raise ValueError('Scheduled release runner requires the exact 46-case campaign')


def capture_completion(output, nonce, before, native_hash, port, observe):
    """Retain failed workflow evidence and independently attempt the binding check."""
    completion = {'nonce': nonce, 'client_exit_code': -1, 'independent_code_exit_code': -1,
                  'remote_binding_postcheck_passed': False, 'client_report_sha256': None}
    details = {}
    try:
        profile_path = output / 'client/profile.json'
        workflow_path = output / 'client/actual-agent-workflows.json'
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            completion['client_exit_code'] = profile.get('client_exit_code', -1)
            completion['independent_code_exit_code'] = profile.get('independent_code_exit_code', -1)
        if workflow_path.exists():
            raw = workflow_path.read_bytes()
            completion['client_report_sha256'] = hashlib.sha256(raw).hexdigest()
            workflow = json.loads(raw)
            verify_release_fixtures(workflow)
            if (workflow.get('scripted_fixture') or not workflow.get('run_complete')
                    or any(case.get('status') != 'ok' for case in workflow['cases'])):
                raise ValueError('Actual full campaign did not complete')
            details['workflow_validation_passed'] = True
        else:
            raise FileNotFoundError('Actual workflow report is absent')
    except Exception as error:
        details['workflow_validation_error'] = repr(error)
        if completion['client_exit_code'] == 0:
            completion['client_exit_code'] = -1
    # Deliberately independent: failed or truncated workflow JSON must not
    # suppress post-run binding evidence from a still-healthy model process.
    try:
        if before is not None:
            after = observe()
            save(output / 'remote-after.json', after)
            verify_identity(after, before, native_hash, port)
            completion['remote_binding_postcheck_passed'] = True
    except Exception as error:
        details['postcheck_error'] = repr(error)
    return completion, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assembly', 'fixtures', 'skills-root', 'ready-identity', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('model', 'expected-native-sha256', 'remote-results', 'nonce'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--remote-completion-name', default='external-client-complete.json')
    parser.add_argument('--ssh-host', default='root@194.68.245.119')
    parser.add_argument('--ssh-port', type=int, default=22050)
    parser.add_argument('--private-key', type=Path, default=Path('/mnt/c/Users/monke/.ssh/id_ed25519'))
    parser.add_argument('--known-hosts', type=Path, default=Path('/mnt/c/Users/monke/.ssh/known_hosts'))
    parser.add_argument('--local-port', type=int, default=18080)
    parser.add_argument('--remote-port', type=int, default=5100)
    parser.add_argument('--artifact-port', type=int, default=18481)
    parser.add_argument('--dotnet', default='/usr/bin/dotnet')
    parser.add_argument('--bwrap', default='/usr/local/bin/bwrap')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--max-rounds', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.expected_native_sha256):
        parser.error('Expected an explicit lowercase native SHA256')
    if not re.fullmatch('[A-Za-z0-9_-]{8,128}', args.nonce):
        parser.error('Nonce must be 8-128 letters/digits/underscore/hyphen')
    if not re.fullmatch('[A-Za-z0-9._-]+', args.remote_completion_name):
        parser.error('Completion name must be a filename')
    if not args.remote_results.startswith('/') or '..' in Path(args.remote_results).parts:
        parser.error('Remote result directory must be an absolute explicit path')
    if args.output.exists():
        parser.error('Use a fresh output directory')
    ready = json.loads(args.ready_identity.read_text())
    verify_ready_identity(ready, args.nonce)
    verify_release_fixtures(json.loads(args.fixtures.read_text()))
    # Check availability without contacting the VM; do not attach to an existing tunnel.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.local_port))
    args.output.mkdir(parents=True)
    report = {'status': 'starting', 'release_qualified': False, 'nonce': args.nonce,
              'scope': 'Remote TensorSharp model, local WSL real AgentHost tools and required sandbox',
              'ready_identity': ready, 'started_at_unix': time.time(), 'remote_results': args.remote_results,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'campaign_timeout_seconds': CAMPAIGN_TIMEOUT_SECONDS,
              'campaign_timeout_basis': '16 sequential waves at 1200 seconds each, plus setup and independent verification; individual case limits unchanged'}
    save(args.output / 'owner-profile.json', report)
    agent = tunnel = client_process = None
    before = None
    env = dict(os.environ)
    log = (args.output / 'ssh-control.log').open('w')
    with tempfile.TemporaryDirectory(prefix='ts-agent-') as directory:
        try:
            agent_socket = str(Path(directory) / 'agent.sock')
            agent = subprocess.Popen(['ssh-agent', '-D', '-a', agent_socket], stdout=log, stderr=subprocess.STDOUT)
            for _ in range(50):
                if Path(agent_socket).exists(): break
                if agent.poll() is not None: raise RuntimeError('Ephemeral SSH agent exited')
                time.sleep(.1)
            else: raise TimeoutError('Ephemeral SSH agent did not create its socket')
            env['SSH_AUTH_SOCK'] = agent_socket
            # Stream the existing private file directly to ssh-add; no key data is
            # read into this Python process, emitted, or written to another file.
            with args.private_key.open('rb') as key:
                subprocess.run(['ssh-add', '-'], stdin=key, stdout=log, stderr=subprocess.STDOUT,
                               env=env, timeout=10, check=True)
            ssh = ['ssh', '-p', str(args.ssh_port), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'UserKnownHostsFile=' + str(args.known_hosts), '-o', 'IdentityAgent=' + agent_socket,
                   '-o', 'IdentityFile=none', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=3']

            def remote(script, argv, input_text=None):
                command = 'python3 -c ' + shlex.quote(script) + ' ' + ' '.join(shlex.quote(str(a)) for a in argv)
                result = subprocess.run([*ssh, args.ssh_host, command], input=input_text, text=True,
                                        stdout=subprocess.PIPE, stderr=log, env=env, timeout=120, check=True)
                return json.loads(result.stdout)

            before = remote(REMOTE_OBSERVER, [ready['remote_pid'], args.remote_port])
            save(args.output / 'remote-before.json', before)
            verify_identity(before, ready, args.expected_native_sha256, args.remote_port)
            identity = {**ready, **before, 'model': args.model, 'nonce': args.nonce,
                        'endpoint_mapping': f'127.0.0.1:{args.local_port} -> remote 127.0.0.1:{args.remote_port}'}
            identity_file = args.output / 'client-identity.json'
            save(identity_file, identity)
            tunnel = subprocess.Popen([*ssh, '-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                                       f'127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}', args.ssh_host],
                                      stdout=log, stderr=subprocess.STDOUT, env=env)
            for _ in range(100):
                if tunnel.poll() is not None: raise RuntimeError('Owned SSH tunnel exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.local_port}/v1/models', timeout=2) as response:
                        models = json.load(response)
                    if args.model not in {model['id'] for model in models.get('data', [])}:
                        raise ValueError('Scheduled model is absent from the pinned endpoint')
                    report['endpoint_models'] = models
                    break
                except (ConnectionError, urllib.error.URLError, TimeoutError):
                    time.sleep(.1)
            else: raise TimeoutError('Owned tunnel endpoint did not become ready')
            report.update(status='running', ssh_agent_pid=agent.pid, ssh_tunnel_pid=tunnel.pid)
            save(args.output / 'owner-profile.json', report)
            command = [sys.executable, str(Path(__file__).with_name('run-remote-agent-profile.py')),
                       '--dotnet', args.dotnet, '--assembly', str(args.assembly), '--fixtures', str(args.fixtures),
                       '--identity', str(identity_file), '--skills-root', str(args.skills_root),
                       '--artifact-root', str(args.output / 'artifacts'), '--output', str(args.output / 'client'),
                       '--endpoint', f'http://127.0.0.1:{args.local_port}/v1/chat/completions', '--model', args.model,
                       '--expected-native-sha256', args.expected_native_sha256, '--artifact-port', str(args.artifact_port),
                       '--bwrap', args.bwrap, '--timeout', str(args.timeout), '--max-rounds', str(args.max_rounds),
                       '--max-tokens', str(args.max_tokens)]
            if args.thinking: command.append('--thinking')
            report['client_command'] = command
            with (args.output / 'client-runner.log').open('w') as client_log:
                client_process = subprocess.Popen(command, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
                report['client_runner_exit_code'] = client_process.wait(timeout=CAMPAIGN_TIMEOUT_SECONDS)
        except Exception as error:
            report.update(status='failed', detail=repr(error))
        finally:
            if client_process is not None and client_process.poll() is None:
                os.killpg(client_process.pid, signal.SIGTERM)
                try: client_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(client_process.pid, signal.SIGKILL)
                    client_process.wait(timeout=10)
            completion, post_details = capture_completion(args.output, args.nonce, before,
                args.expected_native_sha256, args.remote_port,
                lambda: remote(REMOTE_OBSERVER, [ready['remote_pid'], args.remote_port]))
            report.update(post_details)
            report['completion'] = completion
            # Sending a failed completion also releases the owner's wait stage;
            # it never turns partial or unverified work into a passing suite.
            if before is not None:
                try:
                    report['completion_published'] = remote(REMOTE_COMPLETE,
                        [args.remote_results.rstrip('/') + '/' + args.remote_completion_name], json.dumps(completion))
                except Exception as error:
                    report['completion_publish_error'] = repr(error)
            for process in (tunnel, agent):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            log.close()
            passed = (completion['client_exit_code'] == 0 and completion['independent_code_exit_code'] == 0
                      and completion['remote_binding_postcheck_passed'] and completion['client_report_sha256']
                      and report.get('workflow_validation_passed') and report.get('completion_published')
                      and report.get('client_runner_exit_code') == 0)
            report.update(status='passed-supported-client-topology' if passed else 'failed', finished_at_unix=time.time(),
                          ssh_agent_stopped=agent is None or agent.poll() is not None,
                          owned_tunnel_stopped=tunnel is None or tunnel.poll() is not None)
            save(args.output / 'owner-profile.json', report)
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
