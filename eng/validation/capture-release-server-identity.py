#!/usr/bin/env python3
"""Capture a fresh, nonce-bound identity for an already-running release server.

Run locally on the server machine as a before_suites hook. This never starts a
server or an SSH connection. Mapping/socket observation and identity validation
come from run-remote-agent-over-ssh.py, so the producer and consumer agree.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request


OWNER_PATH = Path(__file__).with_name('run-remote-agent-over-ssh.py')
_spec = importlib.util.spec_from_file_location('release_identity_owner', OWNER_PATH)
OWNER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OWNER)


def observe(pid, port):
    result = subprocess.run([sys.executable, '-c', OWNER.REMOTE_OBSERVER, str(pid), str(port)],
                            check=True, capture_output=True, text=True, timeout=90)
    return json.loads(result.stdout)


def available_models(port):
    # Bypass proxy variables for the local endpoint owned by the observed PID.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
        return json.load(response)


def validate_profile(profile, native_hash):
    if profile.get('status') != 'running' or profile.get('finished_at_unix'):
        raise ValueError('A currently running runtime profile is required')
    settings = profile['profile']
    if settings.get('native_policy', 'exact') != 'exact':
        raise ValueError('This identity requires an exact native policy')
    if settings.get('expected_native_sha256') != native_hash:
        raise ValueError('The declared profile native pin differs from the explicit pin')
    pid, port = profile['server_pid'], settings.get('port', 5100)
    if type(pid) is not int or pid <= 0 or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid runtime PID or port')
    if not isinstance(settings.get('model'), str) or not settings['model'] or not profile.get('model_id'):
        raise ValueError('The ready model identity and declared model path are required')
    if not profile.get('loaded_native_libraries') or not profile.get('managed_assemblies') or not profile.get('command'):
        raise ValueError('Runtime native, managed and command snapshots are required')
    for name, digest in profile['managed_assemblies'].items():
        if Path(name).name != name or not name.startswith('TensorSharp.') or not name.endswith('.dll'):
            raise ValueError('Invalid managed snapshot filename')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid managed snapshot digest')
    return pid, port


def validate_observation(actual, profile, native_hash, port):
    if actual.get('command_line') != profile['command']:
        raise ValueError('Observed process command differs from the launched server command')
    available = actual.get('available_managed_libraries', {})
    declared = profile['managed_assemblies']
    if not available or {Path(path).name for path in available} != set(declared):
        raise ValueError('Available managed assemblies differ from the profile snapshot')
    expected_available = {}
    for path in available:
        if not Path(path).is_absolute():
            raise ValueError('A managed mapping path is not absolute')
        expected_available[path] = declared[Path(path).name]
    if not actual.get('mapped_managed_libraries'):
        raise ValueError('No TensorSharp managed assembly is mapped')
    # The runtime report did not record /proc start ticks at launch. Capture
    # them now, then independently reobserve them before publishing the file.
    expected = dict(actual, remote_pid=profile['server_pid'],
                    mapped_native_libraries=profile['loaded_native_libraries'],
                    available_managed_libraries=expected_available)
    OWNER.verify_identity(actual, expected, native_hash, port)


def publish_fresh(path, value):
    """Atomic publication without ever replacing an existing ready identity."""
    if not path.parent.is_dir():
        raise ValueError('The output parent directory must already exist')
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(value, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        # Unlike replace(), link() atomically fails when a concurrent publisher
        # creates the destination after the initial freshness check.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture(profile_path, output_path, native_hash, nonce, observer=observe, model_probe=available_models):
    if not re.fullmatch('[0-9a-f]{64}', native_hash):
        raise ValueError('Expected an explicit lowercase native SHA256')
    if not re.fullmatch('[A-Za-z0-9_-]{8,128}', nonce):
        raise ValueError('Nonce must be 8-128 letters/digits/underscore/hyphen')
    if os.path.lexists(output_path):
        raise FileExistsError('Use a fresh ready identity output path')
    raw = profile_path.read_bytes()
    profile = json.loads(raw)
    pid, port = validate_profile(profile, native_hash)
    before = observer(pid, port)
    validate_observation(before, profile, native_hash, port)
    models = model_probe(port)
    if not any(model.get('id') == profile['model_id'] for model in models.get('data', [])):
        raise ValueError('The ready model ID is absent from the observed endpoint')
    after = observer(pid, port)
    validate_observation(after, profile, native_hash, port)
    OWNER.verify_identity(after, before, native_hash, port)
    if profile_path.read_bytes() != raw:
        raise ValueError('The runtime profile changed during identity capture')
    settings = profile['profile']
    identity = dict(after, nonce=nonce, expected_native_sha256=native_hash,
                    model_id=profile['model_id'], model_path=settings['model'],
                    profile_id=settings.get('id'), profile=settings,
                    environment_overrides=profile.get('environment_overrides', settings.get('env', {})),
                    runtime_profile_path=str(profile_path.resolve()),
                    runtime_profile_sha256=hashlib.sha256(raw).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    observer_source_sha256=hashlib.sha256(OWNER_PATH.read_bytes()).hexdigest(),
                    observer_program_sha256=hashlib.sha256(OWNER.REMOTE_OBSERVER.encode()).hexdigest(),
                    process_observed_twice=True, endpoint_model_id_verified=True,
                    model_artifact_digest_status='not_computed_by_identity_capture')
    # Preserve supplied inventory metadata verbatim; it is provenance from the
    # profile owner, not a claim that this helper hashed large model artifacts.
    for key in ('model_inventory', 'model_inventory_metadata', 'inventory', 'inventory_metadata', 'model_artifacts'):
        if key in profile:
            identity[key] = profile[key]
    publish_fresh(output_path, identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True, help='Runtime output/profile.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh ready-identity.json')
    parser.add_argument('--expected-native-sha256', required=True)
    parser.add_argument('--nonce', required=True)
    args = parser.parse_args()
    result = capture(args.profile, args.output, args.expected_native_sha256, args.nonce)
    print(json.dumps({'output': str(args.output), 'nonce': args.nonce,
                      'remote_pid': result['remote_pid'], 'remote_start_ticks': result['remote_start_ticks'],
                      'expected_native_sha256': args.expected_native_sha256}))


if __name__ == '__main__':
    main()
