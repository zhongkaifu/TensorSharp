import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'capture-release-server-identity.py'
spec = importlib.util.spec_from_file_location('capture_identity', SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureReleaseServerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.profile_path = self.folder / 'profile.json'
        self.output = self.folder / 'ready.json'
        self.native_hash = 'a' * 64
        self.managed_hash = 'b' * 64
        self.command = ['/usr/bin/dotnet', '/repo/host/TensorSharp.Server.Host.dll', '--port', '5100']
        # Use absolute local paths so this pure test also runs on Windows.
        managed = str(self.folder / 'TensorSharp.Server.Host.dll')
        self.actual = {'remote_pid': 123, 'remote_start_ticks': 456, 'remote_boot_id': 'boot-1',
                       'remote_port': 5100, 'command_line': self.command,
                       'mapped_native_libraries': {'/repo/libGgmlOps.so': self.native_hash},
                       'mapped_managed_libraries': {managed: self.managed_hash},
                       'available_managed_libraries': {managed: self.managed_hash},
                       'owned_listening_sockets': [{'inode': '17'}], 'captured_at_unix': 123.5}
        self.profile = {'status': 'running', 'server_pid': 123, 'model_id': 'test-model',
                        'command': self.command, 'loaded_native_libraries': self.actual['mapped_native_libraries'],
                        'managed_assemblies': {'TensorSharp.Server.Host.dll': self.managed_hash},
                        'model_inventory': {'publisher_digest': 'supplied-by-owner'},
                        'profile': {'id': 'candidate', 'port': 5100, 'model': '/models/weights.gguf',
                                    'expected_native_sha256': self.native_hash, 'env': {'TS_TEST': '1'}}}
        self.save_profile()

    def save_profile(self):
        self.profile_path.write_text(json.dumps(self.profile))

    def run_capture(self, observations=None, models=None):
        observations = iter(observations or [copy.deepcopy(self.actual), copy.deepcopy(self.actual)])
        return capture.capture(self.profile_path, self.output, self.native_hash, 'nonce_123',
                               observer=lambda pid, port: next(observations),
                               model_probe=lambda port: models if models is not None else {'data': [{'id': 'test-model'}]})

    def test_captures_real_observation_contract_and_metadata(self):
        result = self.run_capture()
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertEqual(result['remote_start_ticks'], 456)
        self.assertEqual(result['nonce'], 'nonce_123')
        self.assertTrue(result['process_observed_twice'])
        self.assertEqual(result['model_inventory'], self.profile['model_inventory'])
        self.assertEqual(result['model_artifact_digest_status'], 'not_computed_by_identity_capture')
        self.assertNotIn('model_sha256', result)
        self.assertEqual(result['profile']['env'], {'TS_TEST': '1'})
        self.assertEqual(len(result['runtime_profile_sha256']), 64)

    def test_profile_pin_must_match_explicit_pin(self):
        self.profile['profile']['expected_native_sha256'] = 'c' * 64
        self.save_profile()
        with self.assertRaisesRegex(ValueError, 'declared profile native pin'):
            self.run_capture()
        self.assertFalse(self.output.exists())

    def test_observed_contract_failures_never_publish(self):
        mutations = [lambda a: a.update(remote_pid=124), lambda a: a.update(remote_port=5101),
                     lambda a: a.update(owned_listening_sockets=[]), lambda a: a.update(command_line=['other-server']),
                     lambda a: a['mapped_native_libraries'].update({'/repo/libGgmlOps.so': 'c' * 64}),
                     lambda a: a['available_managed_libraries'].update({next(iter(a['available_managed_libraries'])): 'c' * 64}),
                     lambda a: a['mapped_managed_libraries'].update({str(self.folder / 'TensorSharp.Unknown.dll'): 'd' * 64})]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.actual)
                mutate(changed)
                with self.assertRaises(ValueError):
                    self.run_capture([changed, changed])
                self.assertFalse(self.output.exists())

    def test_pid_reuse_boot_change_and_mapping_change_are_rejected(self):
        for key, value in [('remote_start_ticks', 457), ('remote_boot_id', 'boot-2')]:
            with self.subTest(key=key):
                after = copy.deepcopy(self.actual)
                after[key] = value
                with self.assertRaises(ValueError):
                    self.run_capture([copy.deepcopy(self.actual), after])
                self.assertFalse(self.output.exists())

    def test_current_endpoint_must_report_ready_model(self):
        with self.assertRaisesRegex(ValueError, 'ready model ID'):
            self.run_capture(models={'data': [{'id': 'another-model'}]})
        self.assertFalse(self.output.exists())

    def test_finished_profile_is_rejected(self):
        self.profile['status'] = 'passed'
        self.save_profile()
        with self.assertRaisesRegex(ValueError, 'currently running'):
            self.run_capture()

    def test_profile_change_during_capture_is_rejected(self):
        def probe(port):
            self.profile_path.write_text('{}')
            return {'data': [{'id': 'test-model'}]}
        with self.assertRaisesRegex(ValueError, 'profile changed'):
            capture.capture(self.profile_path, self.output, self.native_hash, 'nonce_123',
                            observer=lambda pid, port: copy.deepcopy(self.actual), model_probe=probe)
        self.assertFalse(self.output.exists())

    def test_existing_identity_is_never_replaced(self):
        self.output.write_text('prior identity')
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.assertEqual(self.output.read_text(), 'prior identity')

    def test_atomic_publication_preserves_concurrent_winner(self):
        original = capture.os.link
        def race(source, destination):
            Path(destination).write_text('concurrent identity')
            return original(source, destination)
        with patch.object(capture.os, 'link', side_effect=race):
            with self.assertRaises(FileExistsError):
                self.run_capture()
        self.assertEqual(self.output.read_text(), 'concurrent identity')
        self.assertEqual(list(self.folder.glob('*.tmp')), [])

    def test_nonce_required_before_observation(self):
        with self.assertRaisesRegex(ValueError, 'Nonce'):
            capture.capture(self.profile_path, self.output, self.native_hash, 'short',
                            observer=lambda *args: self.fail('Must not observe an invalid invocation'))


if __name__ == '__main__':
    unittest.main()
