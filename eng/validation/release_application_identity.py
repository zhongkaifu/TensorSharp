"""Bind isolated HTTP application files to an independently pinned build.

Only the top-level libGgmlOps.so may differ between native comparison labels.
Dependency DLLs, configuration, runtime files and nested assets are all checked.
This establishes file identity, not numerical or release qualification.
"""
import hashlib
import http.client
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


_IDENTITY_HELPER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected a lowercase SHA256 digest')
    return value


def read_pinned_json(path, expected, label):
    # Parse precisely the bytes that passed the explicit pin, even if another
    # process replaces the pathname immediately after this read.
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_sha(expected):
        raise ValueError(label + ' SHA256 mismatch')
    return json.loads(raw.decode('utf-8-sig'))


def application_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Application directory must be a real directory')
    files = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise ValueError('Application symlink is not an immutable snapshot: ' + relative)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError('Application contains a non-regular file: ' + relative)
        if relative != 'libGgmlOps.so':
            files[relative] = digest(path)
    if 'TensorSharp.Server.Host.dll' not in files:
        raise ValueError('Application is missing TensorSharp.Server.Host.dll')
    return files


def managed_files(files):
    return {name: value for name, value in files.items()
            if '/' not in name and name.startswith('TensorSharp.') and name.endswith('.dll')}


def checked_build(path, expected):
    build = read_pinned_json(path, expected, 'Managed build manifest')
    if build.get('status') != 'passed' or not build.get('managed_assemblies'):
        raise ValueError('Managed build has not passed or lacks assembly identities')
    for name, value in build['managed_assemblies'].items():
        if Path(name).name != name or not name.startswith('TensorSharp.') or not name.endswith('.dll'):
            raise ValueError('Invalid managed build assembly name')
        require_sha(value)
    return build['managed_assemblies']


def make_manifest(directory, build_path, build_sha256):
    expected = checked_build(build_path, build_sha256)
    files = application_files(directory)
    if managed_files(files) != expected:
        raise ValueError('Application assemblies differ from the pinned managed build')
    return {'schema_version': 1, 'native_exception': 'libGgmlOps.so',
            'managed_build': {'path': str(Path(build_path).resolve()), 'sha256': build_sha256},
            'non_native_files_sha256': files}


def check_application(directory, manifest_path, expected_sha256):
    manifest = read_pinned_json(manifest_path, expected_sha256, 'Application manifest')
    if manifest.get('schema_version') != 1 or manifest.get('native_exception') != 'libGgmlOps.so':
        raise ValueError('Unsupported application manifest contract')
    build = manifest['managed_build']
    expected_managed = checked_build(Path(build['path']), build['sha256'])
    expected = manifest['non_native_files_sha256']
    actual = application_files(directory)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(name for name in actual.keys() & expected.keys() if actual[name] != expected[name])
        raise ValueError(f'Application files changed: missing={missing}, added={added}, changed={changed}')
    if managed_files(actual) != expected_managed:
        raise ValueError('Application assemblies differ from the pinned managed build')
    return {'status': 'passed', 'directory': str(Path(directory).resolve()),
            'manifest': str(Path(manifest_path).resolve()), 'manifest_sha256': expected_sha256,
            'managed_build': build, 'file_count': len(actual),
            'managed_assemblies': expected_managed, 'non_native_files_sha256': actual}


def _reject_deleted_managed_mappings(maps):
    # The frozen shared observer predates deleted-managed-map rejection. Keep
    # that observer unchanged while refusing this otherwise omitted identity.
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or not fields[5].endswith(' (deleted)'):
            continue
        path = fields[5][:-len(' (deleted)')]
        if Path(path).name.startswith('TensorSharp.') and path.endswith('.dll'):
            raise ValueError('A mapped TensorSharp managed library was deleted: ' + path)


def _observe_application_process(pid, port):
    """Use the same process observer as the remote agent identity owner."""
    owner_path = Path(__file__).with_name('run-remote-agent-over-ssh.py')
    maps_path = Path('/proc') / str(pid) / 'maps'
    _reject_deleted_managed_mappings(maps_path.read_text())
    owner_source = owner_path.read_bytes()
    spec = importlib.util.spec_from_file_location('application_identity_owner', owner_path)
    owner = importlib.util.module_from_spec(spec)
    # Execute the unchanged bytes whose hash is reported, avoiding a second
    # pathname read or stale bytecode-cache substitution during provenance capture.
    exec(compile(owner_source, str(owner_path), 'exec'), owner.__dict__)
    result = subprocess.run([sys.executable, '-c', owner.REMOTE_OBSERVER, str(pid), str(port)],
                            check=True, capture_output=True, text=True, timeout=90)
    observation = json.loads(result.stdout)
    _reject_deleted_managed_mappings(maps_path.read_text())
    observation['observer_source_sha256'] = hashlib.sha256(owner_source).hexdigest()
    observation['observer_program_sha256'] = hashlib.sha256(owner.REMOTE_OBSERVER.encode()).hexdigest()
    return observation


def _proc_endpoint(value, family):
    """Decode Linux proc TCP address words, including dual-stack IPv4 peers.

    The kernel emits IPv4 addresses or four IPv6 words in native byte order.
    See https://docs.kernel.org/networking/proc_net_tcp.html.
    """
    address, port = value.rsplit(':', 1)
    words = 1 if family == 'tcp' else 4
    if len(address) != words * 8:
        raise ValueError('Malformed proc TCP address')
    packed = b''.join(int(address[index:index + 8], 16).to_bytes(4, sys.byteorder)
                      for index in range(0, len(address), 8))
    parsed = ipaddress.ip_address(packed)
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        parsed = parsed.ipv4_mapped
    return str(parsed), int(port, 16)


def _connection_from_snapshot(tables, owned_inodes, client, server):
    """Match an established server-side tuple and reject competing listeners.

    /proc/PID/net lists a whole network namespace. The PID's fd inode set is
    essential: a matching tuple in that table alone does not establish owner.
    IPv6 wildcard listeners are conservatively IPv4-capable, because these
    tables do not expose IPV6_V6ONLY. Ambiguity fails rather than qualifying.
    """
    established = []
    competing = []
    for family, table in tables.items():
        if family not in ('tcp', 'tcp6'):
            raise ValueError('Unknown proc TCP table family')
        for row in table.splitlines()[1:]:
            fields = row.split()
            if len(fields) < 10:
                raise ValueError('Malformed proc TCP connection row')
            local, remote = _proc_endpoint(fields[1], family), _proc_endpoint(fields[2], family)
            state, inode = fields[3], fields[9]
            if state == '0A' and local[1] == server[1] and local[0] in ('0.0.0.0', '::', server[0]):
                if inode not in owned_inodes:
                    competing.append({'family': family, 'local': list(local), 'inode': inode})
            if state == '01' and local == server and remote == client and inode in owned_inodes:
                established.append({'family': family, 'local': list(local), 'remote': list(remote), 'inode': inode})
    if competing:
        raise RuntimeError('Unqualified endpoint: another or ambiguous IPv4-capable listener shares the tested port: '
                           + json.dumps(competing, sort_keys=True))
    if len(established) > 1:
        raise RuntimeError('Unqualified endpoint: accepted connection ownership is ambiguous')
    return established[0] if established else None


def _inspect_loopback_connection(pid, client, server):
    proc = Path('/proc') / str(pid)
    if not proc.is_dir():
        raise RuntimeError('Unqualified endpoint: Linux process/socket identity is unavailable')
    owned = set()
    for descriptor in (proc / 'fd').iterdir():
        try:
            link = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if link.startswith('socket:[') and link.endswith(']'):
            owned.add(link[8:-1])
    tables = {family: (proc / 'net' / family).read_text() for family in ('tcp', 'tcp6')
              if (proc / 'net' / family).exists()}
    if not tables:
        raise RuntimeError('Unqualified endpoint: process TCP tables are unavailable')
    return _connection_from_snapshot(tables, owned, client, server)


def _check_runtime_observation(observed, pid, port, command, application, expected):
    if observed.get('remote_pid') != pid or observed.get('remote_port') != port:
        raise ValueError('Observed process PID/port differs from the launched application')
    if type(observed.get('remote_start_ticks')) is not int or observed['remote_start_ticks'] <= 0:
        raise ValueError('Observed process lacks a valid start identity')
    if not isinstance(observed.get('remote_boot_id'), str) or not observed['remote_boot_id']:
        raise ValueError('Observed process lacks a boot identity')
    if observed.get('command_line') != list(command):
        raise ValueError('Observed command differs from the launched application')
    if not observed.get('owned_listening_sockets'):
        raise ValueError('The launched application does not own a listener at its port')
    directory = Path(application['directory'])
    available = {str(directory / name): value for name, value in application['managed_assemblies'].items()}
    if observed.get('available_managed_libraries') != available:
        raise ValueError('Available managed paths/hashes differ from the pinned isolated application')
    mapped = observed.get('mapped_managed_libraries', {})
    host = str(directory / 'TensorSharp.Server.Host.dll')
    if host not in mapped or any(path not in available or available[path] != value for path, value in mapped.items()):
        raise ValueError('Mapped managed paths/hashes differ from the pinned isolated application')
    if expected:
        for key in ('remote_pid', 'remote_start_ticks', 'remote_boot_id', 'command_line'):
            if observed.get(key) != expected.get(key):
                raise ValueError('Application process identity changed: ' + key)


def check_application_runtime(pid, port, command, application_identity, expected=None, *,
                              observer=None, socket_inspector=None, timeout=5):
    """Bind the pinned isolated app to the exact IPv4 origin used by suites.

    The existing native policy remains the caller's responsibility. Returned
    observations include the original observer's native maps for that check.
    A keepalive GET is fully read with bounded size/time; its socket stays open
    across both process observations and exact accepted-socket checks. No model
    inference, compilation, or server lifecycle is performed here.
    """
    if type(pid) is not int or pid <= 0 or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid application PID or port')
    if not 0 < timeout <= 30 or not command or application_identity.get('status') != 'passed':
        raise ValueError('A checked application, command and bounded timeout are required')
    if digest(__file__) != _IDENTITY_HELPER_SOURCE_SHA256:
        raise ValueError('Identity helper source changed after import')
    observer = observer or _observe_application_process
    socket_inspector = socket_inspector or _inspect_loopback_connection
    before = observer(pid, port)
    _check_runtime_observation(before, pid, port, command, application_identity, expected)
    with socket.create_connection(('127.0.0.1', port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        client, server = connection.getsockname(), connection.getpeername()
        connection.sendall(b'GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n')
        response = http.client.HTTPResponse(connection)
        try:
            response.begin()
            body = response.read(1024 * 1024 + 1)
            if response.status != 200 or response.will_close or len(body) > 1024 * 1024:
                raise RuntimeError('Unqualified endpoint: bounded keepalive model probe failed')
            models = json.loads(body)
            if not isinstance(models, dict) or not models.get('data'):
                raise RuntimeError('Unqualified endpoint: held connection has no ready model')
            deadline = time.monotonic() + timeout
            accepted = socket_inspector(pid, client, server)
            while accepted is None and time.monotonic() < deadline:
                time.sleep(min(.02, max(0, deadline - time.monotonic())))
                accepted = socket_inspector(pid, client, server)
            if accepted is None:
                raise RuntimeError('Unqualified endpoint: launched PID does not own the exact accepted loopback connection')
            after = observer(pid, port)
            _check_runtime_observation(after, pid, port, command, application_identity, before)
            final_connection = socket_inspector(pid, client, server)
            if final_connection != accepted:
                raise RuntimeError('Unqualified endpoint: accepted connection changed during identity capture')
        finally:
            response.close()
    if digest(__file__) != _IDENTITY_HELPER_SOURCE_SHA256:
        raise ValueError('Identity helper source changed during capture')
    return dict(after, status='passed', accepted_loopback_connection=accepted,
                endpoint_origin=f'http://127.0.0.1:{port}', endpoint_models=models,
                application_manifest_sha256=application_identity['manifest_sha256'],
                identity_helper_source_sha256=_IDENTITY_HELPER_SOURCE_SHA256, process_observed_twice=True)
