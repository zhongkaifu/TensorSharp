"""Runtime app identity binds the actual IPv4 connection, not a port label."""
import copy
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_application_identity as identity


def endpoint(address, port):
    packed = ipaddress.ip_address(address).packed
    return ''.join(f'{int.from_bytes(packed[index:index + 4], sys.byteorder):08X}'
                   for index in range(0, len(packed), 4)) + f':{port:04X}'


def row(local, remote, state, inode):
    return f'0: {endpoint(*local)} {endpoint(*remote)} {state} 0:0 0:0 0 0 0 {inode}\n'


class KeepaliveServer:
    """One bounded local fake HTTP connection; no external network or model."""
    def __init__(self, keepalive=True):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.port = self.listener.getsockname()[1]
        self.closed = threading.Event()
        self.error = None
        self.keepalive = keepalive
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            with self.listener.accept()[0] as connection:
                connection.settimeout(2)
                request = b''
                while b'\r\n\r\n' not in request:
                    block = connection.recv(4096)
                    if not block:
                        raise AssertionError('Client closed before its request')
                    request += block
                if not request.startswith(b'GET /v1/models HTTP/1.1\r\n'):
                    raise AssertionError('Unexpected probe request')
                body = b'{"data":[{"id":"fake-model"}]}'
                header = (f'HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n'
                          f'Connection: {"keep-alive" if self.keepalive else "close"}\r\n\r\n').encode()
                connection.sendall(header + body)
                if self.keepalive and connection.recv(1) != b'':
                    raise AssertionError('Unexpected data after the probe')
        except Exception as error:
            self.error = error
        finally:
            self.closed.set()

    def finish(self):
        self.listener.close()
        self.thread.join(3)
        if self.thread.is_alive():
            raise AssertionError('Fake listener thread did not terminate')
        if self.error:
            raise self.error


class RuntimeApplicationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'app'
        self.directory.mkdir()
        self.assemblies = {'TensorSharp.Server.Host.dll': 'a' * 64, 'TensorSharp.Runtime.dll': 'b' * 64}
        self.application = {'status': 'passed', 'directory': str(self.directory),
                            'managed_assemblies': self.assemblies, 'manifest_sha256': 'c' * 64}
        self.command = ['dotnet', str(self.directory / 'TensorSharp.Server.Host.dll')]
        self.pid = 4242

    def observation(self, port):
        files = {str(self.directory / name): digest for name, digest in self.assemblies.items()}
        return {'remote_pid': self.pid, 'remote_port': port, 'remote_start_ticks': 12345,
                'remote_boot_id': 'fixture-boot', 'command_line': self.command,
                'available_managed_libraries': files, 'mapped_managed_libraries': files,
                'mapped_native_libraries': {}, 'owned_listening_sockets': [{'inode': '123'}],
                'observer_source_sha256': 'd' * 64, 'observer_program_sha256': 'e' * 64}

    def inspect(self, pid, client, server):
        self.assertEqual(pid, self.pid)
        self.assertEqual(client[0], '127.0.0.1')
        self.assertEqual(server[0], '127.0.0.1')
        return {'family': 'tcp', 'local': list(server), 'remote': list(client), 'inode': '345'}

    def call(self, server, observer=None, inspector=None, expected=None):
        return identity.check_application_runtime(self.pid, server.port, self.command, self.application,
            expected=expected, observer=observer or (lambda *_: self.observation(server.port)),
            socket_inspector=inspector or self.inspect, timeout=.2)

    def test_exact_held_connection_and_provenance(self):
        server = KeepaliveServer()
        try:
            result = self.call(server)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['accepted_loopback_connection']['local'], ['127.0.0.1', server.port])
            self.assertEqual(result['application_manifest_sha256'], 'c' * 64)
            self.assertEqual(result['observer_source_sha256'], 'd' * 64)
            self.assertEqual(result['identity_helper_source_sha256'], identity.digest(identity.__file__))
            self.assertTrue(result['process_observed_twice'])
            self.assertTrue(server.closed.wait(1))
        finally:
            server.finish()

    def test_observer_executes_and_hashes_one_source_snapshot(self):
        # A later pathname read could see a different owner implementation.
        # The returned source pin and executed observer must use this one read.
        source = b'REMOTE_OBSERVER = "print(123)"\n'
        with mock.patch.object(Path, 'read_bytes', return_value=source) as read, \
                mock.patch.object(Path, 'read_text', return_value='') as maps:
            with mock.patch.object(identity.subprocess, 'run',
                                   return_value=SimpleNamespace(stdout='{}')) as run:
                result = identity._observe_application_process(4242, 5123)
        self.assertEqual(read.call_count, 1)
        self.assertEqual(maps.call_count, 2)
        self.assertEqual(run.call_args.args[0], [sys.executable, '-c', 'print(123)', '4242', '5123'])
        self.assertEqual(result['observer_source_sha256'], hashlib.sha256(source).hexdigest())
        self.assertEqual(result['observer_program_sha256'], hashlib.sha256(b'print(123)').hexdigest())

    def test_deleted_managed_mapping_is_not_silently_omitted(self):
        row = '1000-2000 r--p 0000 08:01 12345 '
        identity._reject_deleted_managed_mappings(row + '/app/TensorSharp.Runtime.dll\n')
        identity._reject_deleted_managed_mappings(row + '/app/Other.dll (deleted)\n')
        with self.assertRaisesRegex(ValueError, 'managed library was deleted'):
            identity._reject_deleted_managed_mappings(row + '/app/TensorSharp.Runtime.dll (deleted)\n')

    def test_managed_mapping_deleted_during_observation_is_refused(self):
        source = b'REMOTE_OBSERVER = "print(123)"\n'
        maps = '1000-2000 r--p 0000 08:01 12345 /app/TensorSharp.Runtime.dll (deleted)\n'
        with mock.patch.object(Path, 'read_bytes', return_value=source), \
                mock.patch.object(Path, 'read_text', side_effect=['', maps]), \
                mock.patch.object(identity.subprocess, 'run', return_value=SimpleNamespace(stdout='{}')):
            with self.assertRaisesRegex(ValueError, 'managed library was deleted'):
                identity._observe_application_process(4242, 5123)

    def test_helper_source_drift_is_refused_before_endpoint_probe(self):
        with mock.patch.object(identity, 'digest', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'source changed after import'):
                identity.check_application_runtime(self.pid, 5123, self.command, self.application)

    def test_connection_closes_when_second_observer_fails(self):
        server = KeepaliveServer()
        calls = []
        def observer(*_):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('fixture observer failed')
            return self.observation(server.port)
        try:
            with self.assertRaisesRegex(RuntimeError, 'fixture observer failed'):
                self.call(server, observer=observer)
            self.assertTrue(server.closed.wait(1))
        finally:
            server.finish()

    def test_wrong_process_connection_refused_even_with_matching_port(self):
        server = KeepaliveServer()
        try:
            with self.assertRaisesRegex(RuntimeError, 'does not own the exact accepted'):
                self.call(server, inspector=lambda *_: None)
            self.assertTrue(server.closed.wait(1))
        finally:
            server.finish()

    def test_changed_process_start_and_command_are_rejected(self):
        before = self.observation(5123)
        for field, value in (('remote_pid', 123), ('remote_start_ticks', 54321),
                             ('remote_boot_id', 'different-boot'), ('command_line', ['dotnet', 'other.dll'])):
            with self.subTest(field=field):
                after = dict(before, **{field: value})
                with self.assertRaises(ValueError):
                    identity._check_runtime_observation(after, self.pid, 5123, self.command, self.application, before)

    def test_exact_managed_paths_and_hashes_not_just_file_names(self):
        for change in ('available_hash', 'mapped_hash', 'mapped_other_directory', 'missing_host'):
            actual = copy.deepcopy(self.observation(5123))
            host = str(self.directory / 'TensorSharp.Server.Host.dll')
            if change == 'available_hash':
                actual['available_managed_libraries'][host] = 'f' * 64
            elif change == 'mapped_hash':
                actual['mapped_managed_libraries'][host] = 'f' * 64
            elif change == 'mapped_other_directory':
                actual['mapped_managed_libraries'][str(self.directory.parent / 'TensorSharp.Server.Host.dll')] = 'a' * 64
            else:
                del actual['mapped_managed_libraries'][host]
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'managed paths/hashes'):
                identity._check_runtime_observation(actual, self.pid, 5123, self.command, self.application, None)

    def test_server_must_keep_proof_connection_open(self):
        server = KeepaliveServer(keepalive=False)
        try:
            with self.assertRaisesRegex(RuntimeError, 'keepalive model probe failed'):
                self.call(server)
        finally:
            server.finish()

    def test_namespace_tuple_requires_fd_owner_and_exact_client_port(self):
        client, server = ('127.0.0.1', 32123), ('127.0.0.1', 5123)
        table = 'header\n' + row(server, client, '01', '7')
        self.assertIsNone(identity._connection_from_snapshot({'tcp': table}, {'8'}, client, server))
        self.assertIsNone(identity._connection_from_snapshot({'tcp': table}, {'7'}, (client[0], client[1] + 1), server))
        proof = identity._connection_from_snapshot({'tcp': table}, {'7'}, client, server)
        self.assertEqual(proof['inode'], '7')

    def test_dual_stack_ipv4_tuple_and_unrelated_ipv6_listener(self):
        client, server = ('127.0.0.1', 32123), ('127.0.0.1', 5123)
        table = 'header\n' + row(('::ffff:127.0.0.1', 5123), ('::ffff:127.0.0.1', 32123), '01', '7')
        table += row(('::1', 5123), ('::', 0), '0A', '99')
        proof = identity._connection_from_snapshot({'tcp6': table}, {'7'}, client, server)
        self.assertEqual(proof['local'], list(server))
        self.assertEqual(proof['remote'], list(client))

    def test_competing_ipv4_or_ambiguous_ipv6_wildcard_refused(self):
        client, server = ('127.0.0.1', 32123), ('127.0.0.1', 5123)
        owned = 'header\n' + row(server, client, '01', '7')
        for address, family in (('127.0.0.1', 'tcp'), ('0.0.0.0', 'tcp'), ('::', 'tcp6')):
            tables = {'tcp': owned}
            tables[family] = tables.get(family, 'header\n') + row((address, 5123),
                ('0.0.0.0' if family == 'tcp' else '::', 0), '0A', '99')
            with self.subTest(address=address), self.assertRaisesRegex(RuntimeError, 'ambiguous IPv4-capable listener'):
                identity._connection_from_snapshot(tables, {'7'}, client, server)

    @unittest.skipUnless(sys.platform.startswith('linux') and Path('/proc/self/fd').exists(), 'Linux proc socket integration')
    def test_actual_local_accepted_socket_in_proc(self):
        server = KeepaliveServer()
        try:
            with socket.create_connection(('127.0.0.1', server.port), timeout=1) as connection:
                connection.sendall(b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n')
                connection.recv(4096)
                result = identity._inspect_loopback_connection(os.getpid(), connection.getsockname(), connection.getpeername())
                self.assertIsNotNone(result)
                self.assertEqual(result['local'], ['127.0.0.1', server.port])
        finally:
            server.finish()


if __name__ == '__main__':
    unittest.main()
