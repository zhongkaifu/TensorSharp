#!/usr/bin/env python3
"""Freeze owned native sources and exact upstream archive for an isolated WSL build.

Preparation does not launch WSL, compile, load a model or modify a dependency.
--verify checks the frozen tree and, after extraction, every upstream archive
entry. All generated evidence belongs in the supplied fresh validation directory.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

REVISION = 'ffa4e8b80930029a35991f94e7c8a93cd67730ab'
TOOLS = ('prepare-wsl-cuda-validation.py', 'build-wsl-cuda-validation.py',
         'run-bounded-build.py', 'run-bounded-probe.py',
         'multimodel-quality-bench.py', 'linux-cgroup-telemetry.py',
         'tests/test_bounded_build.py', 'tests/test_wsl_cuda_build_tools.py')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Input escapes its declared root: ' + str(relative))
    return path


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'frozen-inputs.json').read_text(encoding='utf-8'))
    if manifest.get('upstream_revision') != REVISION:
        raise ValueError('Frozen manifest has the wrong upstream revision.')
    names = [item['relative'] for item in manifest['files']]
    if len(set(names)) != len(names) or 'ggml-upstream.tar' not in names:
        raise ValueError('Frozen manifest has duplicate paths or no upstream archive.')
    for item in manifest['files']:
        path = contained(root, item['relative'])
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['relative'])
    upstream = root / 'src/ExternalProjects/ggml'
    with tarfile.open(root / 'ggml-upstream.tar') as archive:
        if archive.pax_headers.get('comment') != REVISION:
            raise ValueError('Archive does not identify the required commit.')
        expected_paths = set()
        for member in archive:
            relative = PurePosixPath(member.name)
            # git archive emits its prefix's parent directory as an explicit
            # entry. Only the directory itself is allowed, never other files.
            if relative.parts == ('ExternalProjects',) and member.isdir():
                continue
            if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] != ('ExternalProjects', 'ggml'):
                raise ValueError('Upstream archive path escapes its source root.')
            if not (member.isdir() or member.issym() or member.isfile()):
                raise ValueError('Unexpected upstream archive entry: ' + member.name)
            if member.issym():
                # Check without resolving the (not-yet-extracted) path. Git's
                # known symlink stays inside the archive's source directory.
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise ValueError('Upstream archive contains an escaping symlink.')
            expected_paths.add(relative.as_posix().rstrip('/'))
            if upstream.exists():
                path = root / 'src' / relative
                if member.isdir():
                    if not path.is_dir() or path.is_symlink():
                        raise ValueError('Missing upstream directory: ' + member.name)
                elif member.issym():
                    if not path.is_symlink() or str(path.readlink()) != member.linkname:
                        raise ValueError('Upstream symlink changed: ' + member.name)
                elif member.isfile():
                    with archive.extractfile(member) as stream:
                        expected = hashlib.sha256(stream.read()).hexdigest()
                    if path.is_symlink() or digest(path) != expected:
                        raise ValueError('Upstream source changed: ' + member.name)
        if upstream.exists():
            for path in upstream.rglob('*'):
                if path.relative_to(root / 'src').as_posix() not in expected_paths:
                    raise ValueError('Unexpected file added to upstream tree: ' + str(path))
    return manifest


def prepare(args):
    repo = args.repo.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, errors=[], scope='Source/managed-file freeze only; no build, WSL or GPU execution.')
    items = []
    def copy(source, relative, expected=None):
        before = digest(source)
        if expected is not None and before != expected.lower():
            raise ValueError('Source differs from proposed pin: ' + str(source))
        target = contained(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(source) != before or digest(target) != before:
            raise ValueError('Source/copy changed during freeze: ' + str(source))
        items.append(dict(relative=relative, bytes=target.stat().st_size, sha256=before, original=str(source)))
    try:
        dependency = repo / 'ExternalProjects/ggml'
        revision = subprocess.check_output(['git', '-C', str(dependency), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(dependency), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if revision != REVISION or status:
            raise ValueError('Expected exact, tracked-clean upstream ggml ' + REVISION)
        proposed = json.loads(args.source_manifest.read_text(encoding='utf-8-sig'))
        if not proposed.get('files'):
            raise ValueError('The proposed native source manifest is empty.')
        for item in proposed['files']:
            if not item['relative'].startswith('TensorSharp.GGML.Native/'):
                raise ValueError('Unexpected proposed source root.')
            copy(contained(repo, item['relative']), 'src/' + item['relative'], item['sha256'])
        copy(args.source_manifest, 'proposed-source-inputs.json')
        for source in sorted((repo / 'ExternalProjects/cudnn/include').glob('*.h')):
            copy(source, 'src/ExternalProjects/cudnn/include/' + source.name)
        for relative in TOOLS:
            copy(repo / 'eng/validation' / relative, 'tools/' + relative)
        for name, directory in (('adaptive-app', args.adaptive_app), ('upload-app', args.upload_app)):
            app = 'AdaptiveMemoryProbe' if name == 'adaptive-app' else 'GlmUploadBudgetProbe'
            required_files = [app + '.dll', app + '.deps.json', app + '.runtimeconfig.json',
                              'TensorSharp.Memory.dll', 'TensorSharp.Core.dll', 'TensorSharp.Backends.GGML.dll']
            if name == 'adaptive-app':
                required_files += ['TensorSharp.Models.dll', 'TensorSharp.Runtime.dll']
            for required in required_files:
                if not (directory / required).is_file():
                    raise FileNotFoundError('Required managed app input missing: ' + str(directory / required))
            for source in sorted(directory.iterdir()):
                if not source.is_file() or source.name.lower() == 'ggmlops.dll':
                    continue
                if source.suffix == '.dll' or source.name.endswith(('.deps.json', '.runtimeconfig.json')):
                    selected = args.memory if source.name == 'TensorSharp.Memory.dll' else source
                    copy(selected, name + '/' + source.name)
        subprocess.run(['git', '-C', str(dependency), 'archive', '--format=tar',
                        '--prefix=ExternalProjects/ggml/', '--output=' + str(root / 'ggml-upstream.tar'),
                        REVISION], check=True, timeout=120)
        archive_path = root / 'ggml-upstream.tar'
        with tarfile.open(archive_path) as archive:
            if archive.pax_headers.get('comment') != REVISION:
                raise ValueError('Archive does not identify the required commit.')
        items.append(dict(relative='ggml-upstream.tar', bytes=archive_path.stat().st_size,
                          sha256=digest(archive_path), original='git archive ' + REVISION))
        manifest = dict(upstream_revision=REVISION, files=items,
                        memory_substitution=str(args.memory.resolve()),
                        scope='Managed copies are pinned independently; no native .so exists until the isolated build succeeds.')
        (root / 'frozen-inputs.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        verify(root)
        report['passed'] = True
        report['files'] = len(items)
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        (root / 'prepare-execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--adaptive-app', type=Path)
    parser.add_argument('--upload-app', type=Path)
    parser.add_argument('--memory', type=Path)
    args = parser.parse_args()
    if args.verify:
        if any(getattr(args, key) for key in ('repo', 'output', 'source_manifest', 'adaptive_app', 'upload_app', 'memory')):
            parser.error('--verify must be used alone.')
        verify(args.verify)
        print('Frozen source, archive and managed identities verified.')
        return 0
    if not all(getattr(args, key) for key in ('repo', 'output', 'source_manifest', 'adaptive_app', 'upload_app', 'memory')):
        parser.error('Preparation requires repo, fresh output, source manifest, both app directories and memory DLL.')
    return prepare(args)


if __name__ == '__main__':
    raise SystemExit(main())
