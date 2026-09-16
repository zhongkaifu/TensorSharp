"""Offline hand-off bounds and byte-preserving publication; no model/HTTP/SSH."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('bound_arrivals', Path(__file__).parents[1] / 'run-bound-arrivals-extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class BoundArrivalsTests(unittest.TestCase):
    def test_budget_includes_full_extension_and_coordination_margin(self):
        self.assertTrue(extension.fits_waiter_budget(3000, 1800, 120, 1080))
        self.assertFalse(extension.fits_waiter_budget(3000, 1800, 120, 1080.001))

    def test_only_exact_active_waiter_supplies_deadline(self):
        completion = Path('/fixture/external-client-complete.json')
        entry = {'started_at_unix': 100, 'command': ['/python', '/repo/wait-external-suite.py',
                 '--completion', str(completion), '--nonce', 'fixture', '--timeout', '21600']}
        self.assertEqual(extension.active_waiter_deadline({'suites': [entry]}, completion, 'fixture'), 21700)
        with self.assertRaises(ValueError):
            extension.active_waiter_deadline({'suites': [entry]}, completion, 'wrong')
        entry['exit_code'] = 0
        with self.assertRaises(ValueError):
            extension.active_waiter_deadline({'suites': [entry]}, completion, 'fixture')

    def test_completion_bytes_preserved_and_existing_file_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'complete.json'
            data = b'{ "nonce": "fixture", "client_exit_code": 1 }\n'
            extension.publish_bytes_fresh(path, data)
            self.assertEqual(path.read_bytes(), data)
            with self.assertRaises(FileExistsError):
                extension.publish_bytes_fresh(path, b'replacement')
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual([p.name for p in path.parent.iterdir()], ['complete.json'])

    def test_complete_controller_pass_failure_defer_and_timeout(self):
        for mode in ('pass', 'semantic-failure', 'defer', 'timeout'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                repo, runtime_dir = base / 'repo', base / 'runtime'
                (repo / 'eng/validation').mkdir(parents=True)
                runtime_dir.mkdir()
                program = repo / 'eng/validation/capture-release-server-identity.py'
                program.write_text('# isolated offline fixture; never executed\n')
                native = 'a' * 64
                managed_path = base / 'managed-build.json'
                managed_path.write_text('{"status":"passed"}\n')
                managed = {'path': str(managed_path), 'sha256': extension.digest(managed_path),
                           'managed_assemblies': {'TensorSharp.Test.dll': 'b' * 64}}
                runtime_path = runtime_dir / 'profile.json'
                ready_path = runtime_dir / 'ready.json'
                completion = runtime_dir / 'complete.json'
                held_path = runtime_dir / 'held.json'
                owner_path = base / 'owner.json'
                output = base / 'extension'
                ready = {'nonce': 'offline-fixture', 'expected_native_sha256': native,
                         'remote_pid': 987, 'remote_port': 5100, 'model_id': 'offline-model',
                         'runtime_profile_path': str(runtime_path),
                         'profile': {'qualified_managed_build': managed},
                         'available_managed_libraries': {str(base / 'TensorSharp.Test.dll'): 'b' * 64}}
                ready_path.write_text(json.dumps(ready))
                timeout = '1800' if mode == 'defer' else '21600'
                runtime_path.write_text(json.dumps({'status': 'running', 'server_pid': 987, 'suites': [
                    {'started_at_unix': 0, 'command': ['python', '/repo/wait-external-suite.py',
                     '--nonce', 'offline-fixture', '--completion', str(completion), '--timeout', timeout]}]}))
                held = {'nonce': 'offline-fixture', 'client_exit_code': 0, 'independent_code_exit_code': 0,
                        'remote_binding_postcheck_passed': True, 'client_report_sha256': 'c' * 64}
                held_bytes = (json.dumps(held, indent=3) + '\n').encode()
                held_path.write_bytes(held_bytes)
                owner_path.write_text(json.dumps({'finished_at_unix': 999, 'ssh_agent_stopped': True,
                    'owned_tunnel_stopped': True, 'completion': held, 'nonce': 'offline-fixture',
                    'ready_identity': ready, 'remote_results': str(runtime_dir)}))
                manifest_path = base / 'manifest.json'
                manifest_path.write_text(json.dumps({'repeats': 3, 'request_timeout_seconds': 1800,
                    'wall_seconds': 1800, 'coordination_margin_seconds': 120, 'native_sha256': native,
                    'nonce': 'offline-fixture', 'runtime_profile_path': str(runtime_path),
                    'managed_build_sha256': managed['sha256'],
                    'program_sha256': {'eng/validation/capture-release-server-identity.py': extension.digest(program)}}))
                argv = ['extension', '--repo', str(repo), '--manifest', str(manifest_path),
                        '--manifest-sha256', extension.digest(manifest_path), '--ready-identity', str(ready_path),
                        '--owner-report', str(owner_path), '--owner-report-sha256', extension.digest(owner_path),
                        '--held-completion', str(held_path), '--completion', str(completion),
                        '--server-log', str(runtime_dir / 'server.log'), '--output', str(output), '--nonce', 'offline-fixture']
                capture = SimpleNamespace(OWNER=SimpleNamespace(verify_ready_identity=mock.Mock(),
                    verify_identity=mock.Mock()), observe=mock.Mock(return_value={'observed': True}))
                loader = SimpleNamespace(exec_module=lambda value: None)
                def create_process(*args, **kwargs):
                    cases = [{'status': 'ok', 'scenario': 'short'} for _ in range(18)]
                    cases += [{'status': 'ok', 'scenario': 'client-disconnect', 'peer_active_at_disconnect': True,
                               'session_cleanup_passed': True} for _ in range(3)]
                    if mode == 'semantic-failure':
                        cases[0]['status'] = 'fail'
                    (output / 'arrivals.json').write_text(json.dumps({'run_complete': True, 'cases': cases,
                        'waves': [{'actual_overlap': True, 'replacement_while_peer_active': True}] * 3}))
                    process = mock.Mock(pid=123, returncode=-15 if mode == 'timeout' else 0)
                    process.poll.return_value = process.returncode
                    process.wait.side_effect = [subprocess.TimeoutExpired('offline', 1800), -15] if mode == 'timeout' else [0]
                    return process
                with mock.patch.object(extension.sys, 'argv', argv), mock.patch.object(extension.time, 'time', return_value=1000), \
                     mock.patch.object(extension.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)), \
                     mock.patch.object(extension.importlib.util, 'module_from_spec', return_value=capture), \
                     mock.patch.object(extension.subprocess, 'Popen', side_effect=create_process) as popen, \
                     mock.patch.object(extension.os, 'killpg', create=True), contextlib.redirect_stdout(io.StringIO()):
                    code = extension.main()
                report = json.loads((output / 'extension-qualification.json').read_text())
                self.assertEqual(code, {'pass': 0, 'semantic-failure': 1, 'defer': 2, 'timeout': 1}[mode])
                self.assertEqual(report.get('aggregate_passed', False), mode == 'pass')
                self.assertEqual(popen.call_count, 0 if mode == 'defer' else 1)
                self.assertEqual(capture.observe.call_count, 2)
                self.assertFalse(report['release_qualified'])
                if mode == 'timeout':
                    self.assertFalse(completion.exists())
                    self.assertIn('drained', report['handoff_error'])
                else:
                    self.assertEqual(completion.read_bytes(), held_bytes)
                    self.assertTrue(report['completion_published'])


if __name__ == '__main__':
    unittest.main()
