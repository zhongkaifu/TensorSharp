"""Application drift must fail before launch or after even a failed lifecycle."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import release_application_identity as identity


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), SCRIPTS / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = load('run-release-profile')
prepare = load('prepare-release-application-manifest')


class ApplicationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.app = self.root / 'app'
        self.app.mkdir()
        for name in ('TensorSharp.Server.Host.dll', 'TensorSharp.Runtime.dll',
                     'TensorSharp.Server.Host.runtimeconfig.json', 'Dependency.dll', 'libGgmlOps.so'):
            (self.app / name).write_bytes(name.encode())
        (self.app / 'nested').mkdir()
        (self.app / 'nested/asset.bin').write_bytes(b'asset')
        self.build = self.root / 'build.json'
        self.build.write_text(json.dumps({'status': 'passed',
            'managed_assemblies': identity.managed_files(identity.application_files(self.app))}))
        self.build_sha = identity.digest(self.build)
        self.manifest = self.root / 'application.json'
        self.manifest.write_text(json.dumps(identity.make_manifest(self.app, self.build, self.build_sha)))
        self.manifest_sha = identity.digest(self.manifest)

    def check(self):
        return identity.check_application(self.app, self.manifest, self.manifest_sha)

    def args(self):
        profile = self.root / 'input-profile.json'
        profile.write_text(json.dumps({'id': 'fixture', 'model': 'fixture.gguf', 'telemetry': False,
                                       'suites': [['fixture-suite']]}))
        return ['run-release-profile.py', '--profile', str(profile), '--repo', str(self.root),
                '--dotnet', str(self.root / 'dotnet'), '--native', str(self.app / 'libGgmlOps.so'),
                '--output', str(self.root / 'result'), '--server-assembly',
                str(self.app / 'TensorSharp.Server.Host.dll'), '--application-manifest',
                str(self.manifest), '--application-manifest-sha256', self.manifest_sha]

    def test_complete_file_set_and_native_only_exception(self):
        self.assertEqual(self.check()['file_count'], 5)
        (self.app / 'libGgmlOps.so').write_bytes(b'a different native label')
        self.check()
        (self.app / 'nested/asset.bin').write_bytes(b'drift')
        with self.assertRaisesRegex(ValueError, 'nested/asset.bin'):
            self.check()

    def test_added_dependency_and_missing_configuration_fail(self):
        (self.app / 'injected.dll').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError, 'added=.*injected.dll'):
            self.check()
        (self.app / 'injected.dll').unlink()
        (self.app / 'TensorSharp.Server.Host.runtimeconfig.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing=.*runtimeconfig.json'):
            self.check()

    def test_manifest_and_independent_build_pins_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'Application manifest SHA256 mismatch'):
            identity.check_application(self.app, self.manifest, '0' * 64)
        self.build.write_text(self.build.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Managed build manifest SHA256 mismatch'):
            self.check()

    def test_self_recording_a_different_managed_build_is_refused(self):
        (self.app / 'TensorSharp.Runtime.dll').write_bytes(b'newer managed assembly')
        with self.assertRaisesRegex(ValueError, 'differ from the pinned managed build'):
            identity.make_manifest(self.app, self.build, self.build_sha)

    def test_replaced_manifest_is_not_reopened_after_hashing(self):
        original_read = Path.read_bytes
        reads = []
        def replace_after_read(path):
            raw = original_read(path)
            if path == self.manifest:
                reads.append(path)
                path.write_text('{"schema_version": -1}')
            return raw
        with mock.patch.object(Path, 'read_bytes', replace_after_read):
            self.assertEqual(self.check()['status'], 'passed')
        self.assertEqual(reads, [self.manifest])
        with self.assertRaisesRegex(ValueError, 'Application manifest SHA256 mismatch'):
            self.check()

    def test_replaced_build_manifest_is_not_reopened_after_hashing(self):
        original_read = Path.read_bytes
        reads = []
        def replace_after_read(path):
            raw = original_read(path)
            if path == self.build:
                reads.append(path)
                path.write_text('{"status": "failed"}')
            return raw
        with mock.patch.object(Path, 'read_bytes', replace_after_read):
            self.assertEqual(self.check()['status'], 'passed')
        self.assertEqual(reads, [self.build])
        with self.assertRaisesRegex(ValueError, 'Managed build manifest SHA256 mismatch'):
            self.check()

    def test_peer_mismatch_cannot_publish_manifest(self):
        peer = self.root / 'peer'
        shutil.copytree(self.app, peer)
        (peer / 'libGgmlOps.so').write_bytes(b'candidate')
        output = self.root / 'paired.json'
        args = ['prepare-release-application-manifest.py', '--application', str(self.app),
                '--peer', str(peer), '--managed-build', str(self.build),
                '--managed-build-sha256', self.build_sha, '--output', str(output)]
        with mock.patch.object(sys, 'argv', args):
            prepare.main()
        original = output.read_bytes()
        with mock.patch.object(sys, 'argv', args), self.assertRaises(FileExistsError):
            prepare.main()
        self.assertEqual(output.read_bytes(), original)
        (peer / 'Dependency.dll').write_bytes(b'changed dependency')
        args[-1] = str(self.root / 'rejected.json')
        with mock.patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'Comparison peer differs'):
            prepare.main()
        self.assertFalse((self.root / 'rejected.json').exists())

    def test_prelaunch_drift_stops_before_starting_any_process(self):
        args = self.args()
        (self.app / 'Dependency.dll').write_bytes(b'drift')
        with mock.patch.object(sys, 'argv', args), mock.patch.object(runner.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
            launch.assert_not_called()
        report = json.loads((self.root / 'result/profile.json').read_text())
        self.assertEqual(report['application_identity_after']['status'], 'failed')

    def test_final_audit_runs_when_startup_itself_fails(self):
        def failed_launch(*args, **kwargs):
            (self.app / 'Dependency.dll').write_bytes(b'changed during failed startup')
            raise RuntimeError('fixture startup failure')
        with mock.patch.object(sys, 'argv', self.args()), mock.patch.object(runner.subprocess, 'Popen', side_effect=failed_launch), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        report = json.loads((self.root / 'result/profile.json').read_text())
        self.assertEqual(report['application_identity_before']['status'], 'passed')
        self.assertEqual(report['application_identity_after']['status'], 'failed')
        self.assertEqual(report['error'], 'fixture startup failure')

    def test_successful_suite_cannot_mask_application_drift(self):
        process = mock.Mock(pid=123456789, returncode=0)
        process.poll.return_value = 0
        def suite(*args, **kwargs):
            (self.app / 'Dependency.dll').write_bytes(b'drift after a passing suite')
            return mock.Mock(returncode=0)
        # poll is checked during startup, after the suite, then during cleanup.
        process.poll.side_effect = [None, None, 0]
        def response(*args, **kwargs):
            return contextlib.closing(io.BytesIO(b'{"data":[{"id":"fixture"}]}'))
        native = self.app / 'libGgmlOps.so'
        with mock.patch.object(sys, 'argv', self.args()), \
                mock.patch.object(runner.subprocess, 'Popen', return_value=process), \
                mock.patch.object(runner.subprocess, 'run', side_effect=suite), \
                mock.patch.object(runner.urllib.request, 'urlopen', side_effect=response), \
                mock.patch.object(runner, 'mapped_native_libraries', return_value={str(native): identity.digest(native)}), \
                mock.patch.object(identity, 'check_application_runtime',
                    return_value={'mapped_native_libraries': {str(native): identity.digest(native)},
                                  'endpoint_models': {'data': [{'id': 'fixture'}]}}) as binding, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
            self.assertEqual(binding.call_count, 3)
        report = json.loads((self.root / 'result/profile.json').read_text())
        self.assertEqual(report['suites'][0]['exit_code'], 0)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['application_identity_after']['status'], 'failed')

    def test_unrelated_ready_endpoint_cannot_run_a_suite(self):
        process = mock.Mock(pid=123456789, returncode=0)
        process.poll.side_effect = [None, 0]
        def response(*args, **kwargs):
            return contextlib.closing(io.BytesIO(b'{"data":[{"id":"fixture"}]}'))
        native = self.app / 'libGgmlOps.so'
        with mock.patch.object(sys, 'argv', self.args()), \
                mock.patch.object(runner.subprocess, 'Popen', return_value=process), \
                mock.patch.object(runner.subprocess, 'run') as suite, \
                mock.patch.object(runner.urllib.request, 'urlopen', side_effect=response), \
                mock.patch.object(runner, 'mapped_native_libraries', return_value={str(native): identity.digest(native)}), \
                mock.patch.object(identity, 'check_application_runtime',
                    side_effect=ValueError('The launched process does not own this HTTP connection')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
            suite.assert_not_called()
        report = json.loads((self.root / 'result/profile.json').read_text())
        self.assertEqual(report['suites'], [])
        self.assertIn('does not own', report['error'])
        self.assertEqual(report['application_identity_after']['status'], 'passed')

    def test_pinned_process_with_another_model_still_fails(self):
        observation = {'mapped_native_libraries': {'/app/libGgmlOps.so': 'a' * 64},
                       'endpoint_models': {'data': [{'id': 'other-model'}]}}
        with self.assertRaisesRegex(RuntimeError, 'expected model'):
            runner.check_runtime_policy({}, observation, 'a' * 64, 'expected-model')


if __name__ == '__main__':
    unittest.main()
