import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('pinned_benchmark', ROOT / 'run-pinned-benchmark.py')
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


class MappedInputIdentityTests(unittest.TestCase):
    def test_console_dependency_closure_cannot_be_replaced_by_server_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            assembly = root / 'Bench.dll'
            assembly.write_bytes(b'benchmark')
            dependency = 'Microsoft.Extensions.Logging.Abstractions.dll'
            assembly.with_suffix('.deps.json').write_text(json.dumps({
                'runtimeTarget': {'name': 'test'}, 'targets': {'test': {
                    'Bench/1': {'runtime': {'Bench.dll': {}}},
                    'Logging/6': {'runtime': {'lib/net6.0/' + dependency: {}}}}}}))
            # The server's framework reference does not supply a local package
            # declared by this different entry point's dependency manifest.
            (root / 'Server.dll').write_bytes(b'server')
            with self.assertRaisesRegex(ValueError, dependency):
                OWNER.check_managed_dependencies(root, assembly)
            (root / dependency).write_bytes(b'logging')
            result = OWNER.check_managed_dependencies(root, assembly)
            self.assertEqual(result, {'Bench.dll': OWNER.digest(assembly),
                                     dependency: OWNER.digest(root / dependency)})

    def test_entry_assembly_and_runtime_dependency_cannot_escape_application(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            app = root / 'app'
            app.mkdir()
            outside = root / 'Outside.dll'
            outside.write_bytes(b'outside')
            with self.assertRaisesRegex(ValueError, 'pinned application'):
                OWNER.check_managed_dependencies(app, outside)
            assembly = app / 'Bench.dll'
            assembly.write_bytes(b'bench')
            assembly.with_suffix('.deps.json').write_text(json.dumps({
                'runtimeTarget': {'name': 'test'}, 'targets': {'test': {
                    'Outside/1': {'runtime': {'../Outside.dll': {}}}}}}))
            with self.assertRaisesRegex(ValueError, 'Missing benchmark runtime'):
                OWNER.check_managed_dependencies(app, assembly)

    def test_empty_exec_transition_waits_for_exact_command(self):
        reads = iter([b'', b'/usr/bin/dotnet\0/pinned/bench.dll\0'])
        expected = ['/usr/bin/dotnet', '/pinned/bench.dll']
        self.assertEqual(OWNER.wait_command_line(lambda: next(reads), lambda: None, expected), expected)

    def test_missing_or_wrong_command_identity_never_passes(self):
        for raw, status, timeout, error in [(b'', 1, 5, RuntimeError),
                                             (b'', None, 0, TimeoutError),
                                             (b'wrong\0', None, 5, ValueError)]:
            with self.subTest(raw=raw, status=status):
                with self.assertRaises(error):
                    OWNER.wait_command_line(lambda: raw, lambda: status, ['expected'], timeout)

    def test_real_file_mapping_requires_exact_path_and_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            library = root / 'libGgmlOps.so'
            model = root / 'TensorSharp.Models.dll'
            library.write_bytes(b'fresh native')
            model.write_bytes(b'pinned managed')
            expected = {str(model): OWNER.digest(model)}
            native_sha = OWNER.digest(library)
            maps = '\n'.join(f'1000-2000 r--p 00000000 08:01 7 {path}' for path in (library, model))
            self.assertEqual(OWNER.inspect_maps(maps, expected, library, native_sha),
                             {str(library): native_sha, **expected})
            library.write_bytes(b'stale native')
            with self.assertRaisesRegex(ValueError, 'changed'):
                OWNER.inspect_maps(maps, expected, library, native_sha)

    def test_another_library_with_same_digest_does_not_establish_binding(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            expected = root / 'libGgmlOps.so'
            other = root / 'other' / expected.name
            other.parent.mkdir()
            expected.write_bytes(b'native')
            other.write_bytes(expected.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                OWNER.inspect_maps(f'1000-2000 r--p 00000000 08:01 7 {other}', {}, expected, OWNER.digest(expected))
            # A nested file pinned as an application asset still cannot become
            # the actual native bridge named by the process plan.
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                OWNER.inspect_maps(f'1000-2000 r--p 00000000 08:01 7 {other}',
                                   {str(other): OWNER.digest(other)}, expected, OWNER.digest(expected))

    def test_deleted_managed_mapping_is_failure_even_if_path_was_recreated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / 'TensorSharp.Models.dll'
            model.write_bytes(b'recreated')
            with self.assertRaisesRegex(ValueError, 'deleted'):
                OWNER.inspect_maps(f'1000-2000 r--p 00000000 08:01 7 {model} (deleted)',
                                   {str(model): OWNER.digest(model)}, root / 'libGgmlOps.so', '0' * 64)


if __name__ == '__main__':
    unittest.main()
