"""Model-free tests for the Windows probe SDK closure checker; no SDK invocation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


tool = Path(__file__).resolve().parents[1] / 'stage-glm-session-probe-publish.py'
spec = importlib.util.spec_from_file_location('glm_session_sdk_stage', tool)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class RuntimeClosureTests(unittest.TestCase):
    def setUp(self):
        self.logging = 'Microsoft.Extensions.Logging.Abstractions.dll'
        self.linux = 'runtimes/linux-x64/native/library.so.164'
        self.mac = 'runtimes/osx-arm64/native/library.dylib'
        self.deps = {'targets': {'net10.0': {
            'Microsoft.Extensions.Logging.Abstractions/6.0.0': {'runtime': {
                'lib/net6.0/' + self.logging: {'assemblyVersion': '6.0.0.0'}}},
            'NativePackage/1.0': {'runtimeTargets': {
                self.linux: {'rid': 'linux-x64', 'assetType': 'native'},
                self.mac: {'rid': 'osx-arm64', 'assetType': 'native'}}}}}}
        self.files = {self.logging: {'bytes': 600, 'sha256': 'a' * 64},
                      self.linux: {'bytes': 9, 'sha256': 'b' * 64},
                      self.mac: {'bytes': 9, 'sha256': 'c' * 64}}

    def test_logging_six_and_each_portable_target_are_bound(self):
        result = stage.runtime_closure(self.deps, self.files, self.files)
        self.assertEqual(len(result['runtime_assemblies']), 1)
        self.assertEqual(result['runtime_targets'][self.mac]['sha256'], 'c' * 64)

    def test_root_basename_cannot_substitute_for_missing_rid_target(self):
        published = dict(self.files)
        published[Path(self.linux).name] = published.pop(self.linux)
        with self.assertRaisesRegex(ValueError, 'exact SDK runtimeTarget'):
            stage.runtime_closure(self.deps, published, self.files)

    def test_same_size_swapped_platform_payload_is_rejected(self):
        published = dict(self.files)
        published[self.linux] = self.files[self.mac]
        with self.assertRaisesRegex(ValueError, 'exact SDK runtimeTarget'):
            stage.runtime_closure(self.deps, published, self.files)

    def test_missing_logging_assembly_is_not_hidden_by_framework_assumption(self):
        published = dict(self.files)
        del published[self.logging]
        with self.assertRaisesRegex(ValueError, 'runtime assembly'):
            stage.runtime_closure(self.deps, published, self.files)

    def test_runtime_target_path_traversal_is_rejected(self):
        for path in ('runtimes/../outside.so', 'runtimes\\win-x64\\library.dll', '/runtimes/library.so'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'escaping'):
                stage.runtime_closure({'targets': {'net10.0': {'bad': {
                    'runtimeTargets': {path: {}}}}}}, {}, {})

    def test_snapshot_covers_versioned_native_payloads_and_runtime_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('library.so.164', 'library.dylib', 'runtimes/linux-x64/native/license.txt', 'run.log'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            self.assertEqual(set(stage.snapshot(root, True)),
                             {'library.so.164', 'library.dylib', 'runtimes/linux-x64/native/license.txt'})


if __name__ == '__main__':
    unittest.main()
