"""Failure-evidence tests; no WSL, compiler, dependency fetch or model is started."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load('prepare-wsl-cuda-validation')
DRIVER = load('build-wsl-cuda-validation')


def make_frozen(root, extra_member=None):
    archive = root / 'ggml-upstream.tar'
    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT,
                      pax_headers={'comment': FREEZE.REVISION}) as stream:
        for name in ('ExternalProjects', 'ExternalProjects/ggml'):
            member = tarfile.TarInfo(name); member.type = tarfile.DIRTYPE
            stream.addfile(member)
        member = tarfile.TarInfo('ExternalProjects/ggml/known.txt'); member.size = 4
        stream.addfile(member, io.BytesIO(b'test'))
        if extra_member:
            member = tarfile.TarInfo(extra_member); member.size = 1
            stream.addfile(member, io.BytesIO(b'x'))
    manifest = dict(upstream_revision=FREEZE.REVISION,
                    files=[dict(relative=archive.name, bytes=archive.stat().st_size, sha256=FREEZE.digest(archive))])
    (root / 'frozen-inputs.json').write_text(json.dumps(manifest))
    return manifest


class FreezeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('git'), 'Requires git for a tiny local archive fixture')
    def test_real_git_archive_prefix_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); repo = parent / 'repo'; repo.mkdir()
            root = parent / 'freeze'; root.mkdir()
            hooks = parent / 'empty-hooks'; hooks.mkdir()
            command = ['git', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=' + str(hooks),
                       '-c', 'user.name=Fixture', '-c', 'user.email=fixture@invalid', '-C', str(repo)]
            for arguments in (['init'],):
                subprocess.run(command + arguments, check=True, capture_output=True)
            (repo / 'known.txt').write_bytes(b'test')
            subprocess.run(command + ['add', 'known.txt'], check=True, capture_output=True)
            subprocess.run(command + ['commit', '-m', 'tiny local archive fixture'], check=True, capture_output=True)
            revision = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip()
            archive = root / 'ggml-upstream.tar'
            subprocess.run(command + ['archive', '--format=tar', '--prefix=ExternalProjects/ggml/',
                                      '--output=' + str(archive), revision], check=True, capture_output=True)
            manifest = dict(upstream_revision=revision, files=[dict(relative=archive.name,
                bytes=archive.stat().st_size, sha256=FREEZE.digest(archive))])
            (root / 'frozen-inputs.json').write_text(json.dumps(manifest))
            with tarfile.open(archive) as stream:
                members = {member.name for member in stream}
                self.assertIn('ExternalProjects/ggml', members)
                self.assertIn('ExternalProjects/ggml/known.txt', members)
            with mock.patch.object(FREEZE, 'REVISION', revision):
                FREEZE.verify(root)

    def test_archive_prefix_parent_is_accepted_and_payload_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); make_frozen(root)
            FREEZE.verify(root)
            upstream = root / 'src/ExternalProjects/ggml'; upstream.mkdir(parents=True)
            source = upstream / 'known.txt'; source.write_bytes(b'test')
            FREEZE.verify(root)
            source.write_bytes(b'edit')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                FREEZE.verify(root)

    def test_unexpected_upstream_file_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); make_frozen(root)
            upstream = root / 'src/ExternalProjects/ggml'; upstream.mkdir(parents=True)
            (upstream / 'known.txt').write_bytes(b'test'); (upstream / 'injected.h').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'Unexpected file'):
                FREEZE.verify(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); make_frozen(root, 'ExternalProjects/ggml/../../outside')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                FREEZE.verify(root)

    def test_wrong_revision_or_missing_archive_cannot_verify(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = make_frozen(root)
            manifest['upstream_revision'] = 'wrong'
            (root / 'frozen-inputs.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'revision'):
                FREEZE.verify(root)
            manifest['upstream_revision'] = FREEZE.REVISION; manifest['files'] = []
            (root / 'frozen-inputs.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'no upstream archive'):
                FREEZE.verify(root)

    def test_prepare_mid_copy_failure_retains_execution(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            parent = Path(temporary); repo = parent / 'repo'; repo.mkdir()
            source = repo / 'TensorSharp.GGML.Native/CMakeLists.txt'; source.parent.mkdir(); source.write_bytes(b'changed')
            proposed = parent / 'manifest.json'
            proposed.write_text(json.dumps(dict(files=[dict(relative='TensorSharp.GGML.Native/CMakeLists.txt', sha256='0' * 64)])))
            args = argparse.Namespace(repo=repo, output=parent / 'output', source_manifest=proposed,
                                      adaptive_app=parent, upload_app=parent, memory=parent / 'missing.dll')
            with mock.patch.object(FREEZE.subprocess, 'check_output', side_effect=[FREEZE.REVISION, '']), \
                 mock.patch.object(FREEZE.subprocess, 'run') as run:
                self.assertEqual(1, FREEZE.prepare(args)); run.assert_not_called()
            report = json.loads((args.output / 'prepare-execution.json').read_text())
            self.assertFalse(report['passed']); self.assertIn('proposed pin', report['errors'][0])


class DriverFailureTests(unittest.TestCase):
    def test_low_disk_saves_evidence_without_launching_any_process(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            with mock.patch.object(DRIVER.shutil, 'disk_usage', return_value=SimpleNamespace(free=31)), \
                 mock.patch.object(DRIVER, 'sibling') as sibling:
                code = DRIVER.main(['--root', str(root), '--minimum-output-free-bytes', '32', '--system-volume', str(root)])
            self.assertEqual(1, code); sibling.assert_not_called()
            report = json.loads((root / 'build-evidence/execution.json').read_text())
            self.assertFalse(report['passed']); self.assertEqual('disk-space-preflight', report['phase'])
            self.assertIn('Low disk space', report['errors'][0])

    def run_mock_phase_failure(self, raised):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            freezer = SimpleNamespace(verify=mock.Mock(return_value={}))
            def fail_phase(argv):
                output = Path(argv[argv.index('--output') + 1]); output.mkdir()
                (output / 'process.log').write_text('retained compiler diagnostic')
                (output / 'execution.json').write_text(json.dumps(dict(passed=False, errors=['injected phase failure'])))
                cache = root / 'build/CMakeCache.txt'; cache.parent.mkdir(exist_ok=True); cache.write_text('retained cache')
                if raised: raise OSError('injected supervisor bookkeeping failure')
                return 1
            supervisor = SimpleNamespace(main=mock.Mock(side_effect=fail_phase))
            with mock.patch.object(DRIVER, 'is_wsl_output_path', return_value=True), \
                 mock.patch.object(DRIVER.sys, 'platform', 'linux'), \
                 mock.patch.object(DRIVER.signal, 'SIGHUP', 1, create=True), \
                 mock.patch.object(DRIVER.signal, 'signal'), \
                 mock.patch.object(DRIVER, 'check_space', return_value={}), \
                 mock.patch.object(DRIVER, 'sibling', side_effect=[freezer, supervisor]), \
                 mock.patch.dict(DRIVER.os.environ, {}, clear=False):
                self.assertEqual(1, DRIVER.main(['--root', str(root), '--system-volume', str(root)]))
            self.assertEqual(2, freezer.verify.call_count)
            self.assertEqual(1, supervisor.main.call_count)
            report = json.loads((root / 'build-evidence/execution.json').read_text())
            self.assertFalse(report['passed']); self.assertTrue(report['source_identity_unchanged'])
            self.assertTrue((root / 'build-evidence/CMakeCache.txt').is_file())
            self.assertTrue((root / 'build-evidence/cmake-version/process.log').is_file())
            self.assertIn('injected' if raised else 'phase failed', report['errors'][0])

    def test_phase_failure_saves_log_cache_and_final_identity(self):
        self.run_mock_phase_failure(False)

    def test_intermediate_exception_still_saves_final_execution(self):
        self.run_mock_phase_failure(True)


if __name__ == '__main__':
    unittest.main()
