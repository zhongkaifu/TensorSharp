#!/usr/bin/env python3
"""Stage the already compiled Server.Host closure without build or restore.

Never launches the host, injects native binaries, or modifies user output. A
fresh destination and the actual compiled OutDir are mandatory. SDK publish
metadata may be generated in obj; all source and user bin hashes are preserved.
The native library must be supplied from a separately qualified receipt later.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def snapshot(path, include=None):
    path = Path(path)
    return {str(p.relative_to(path)).replace('\\', '/'): {
                'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(path.rglob('*')) if p.is_file() and (include is None or include(p))}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def compiled_input(path):
    # Compare complete executables/assemblies and their launch metadata, not
    # mutable validation logs/uploads from the mixed tests' OutDir.
    return path.suffix.lower() in ('.dll', '.exe', '.pdb', '.ptx') or path.name.endswith(
        ('.deps.json', '.runtimeconfig.json', '.staticwebassets.runtime.json',
         '.staticwebassets.endpoints.json'))


def stage(repository, build_output, output, dotnet='dotnet', directory_build_props=None):
    repository, build_output, output = (Path(p).resolve() for p in (repository, build_output, output))
    project = repository / 'TensorSharp.Server.Host/TensorSharp.Server.Host.csproj'
    user_bin = repository / 'TensorSharp.Server.Host/bin'
    if not project.is_file() or not (build_output / 'TensorSharp.Server.Host.dll').is_file():
        raise ValueError('Require existing Host project and compiled Host output')
    if output.exists() or output == build_output or output.is_relative_to(build_output):
        raise ValueError('Require a fresh destination outside the compiled output')
    if output == user_bin or output.is_relative_to(user_bin):
        raise ValueError('Never stage into the user Host bin')
    props = Path(directory_build_props).resolve() if directory_build_props is not None else None
    if props is not None and not props.is_file():
        raise ValueError('The original build props file must exist')
    output.mkdir(parents=True, exist_ok=False)
    publish = output / 'publish'
    command = [dotnet, 'publish', str(project), '--configuration', 'Release',
        '--no-build', '--no-restore', '--verbosity', 'diagnostic',
        '-m:1', '-p:BuildProjectReferences=false',
        '-p:OutDir=' + str(build_output).replace('\\', '/') + '/',
        '-p:OutputPath=' + str(build_output).replace('\\', '/') + '/',
        '-p:PublishDir=' + str(publish).replace('\\', '/') + '/',
        '-p:TensorSharpSkipGgmlNative=true', '-p:TensorSharpSkipMlxNative=true',
        # An unexpected Build hook still cannot copy an unqualified repository
        # native binary into either compiled output or the staged product.
        '-p:GgmlNativeBinaryName=', '-p:MlxNativeBinaryName=',
        '-p:CudaArch=compute_86']
    if props is not None:
        # A private build's project-specific obj directory contains its exact
        # launch metadata and publish inputs. The default obj directory can
        # silently stage a different Host even with the correct OutDir.
        command.append('-p:DirectoryBuildPropsPath=' + str(props).replace('\\', '/'))
    source_files = [repository / 'Directory.Build.props', project,
                    repository / 'TensorSharp.Server/TensorSharp.Server.csproj',
                    repository / 'TensorSharp.Backends.GGML/TensorSharp.Backends.GGML.csproj',
                    repository / 'TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj']
    if props is not None:
        source_files.append(props)
    report = {'schema_version': 1, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
        'compiled_output': str(build_output), 'publish': str(publish),
        'tool_sha256': digest(__file__), 'model_executed': False,
        'native_injected': False, 'publish_complete': False,
        'limitations': ['No host/model/GPU execution. Static closure validation only.',
            'SDK publish may regenerate obj publish metadata; no compilation/restore is authorized.',
            'Native is deliberately suppressed; supply a separately pinned production native binary before launch.']}
    report['compiled_before'] = snapshot(build_output, compiled_input)
    report['user_bin_before'] = snapshot(user_bin)
    report['project_before'] = {str(p): digest(p) for p in source_files}
    save(output / 'receipt.json', report)
    try:
        with (output / 'publish.log').open('wb') as log:
            run = subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT, check=False)
        report['exit_code'] = run.returncode
        if run.returncode:
            raise ValueError('No-build publish failed; preserve this attempt and do not fall back to build')
        text = (output / 'publish.log').read_text(encoding='utf-8-sig', errors='replace')
        # Actual started compiler tasks, rather than target names printed for
        # false/skipped conditions. Retain the full diagnostic log for review.
        forbidden = re.findall(r'Task "(Csc|Fsc|Vbc|CompileCudaPtx)"(?:\s|\()', text)
        if forbidden or re.search(r'Building GGML native library|Compiling .*\.cu to PTX|Restoring packages for', text):
            raise ValueError('Publish unexpectedly invoked compilation or restore')
        report['compiler_tasks_started'] = forbidden
        files = snapshot(publish)
        report['published_files'] = files
        rejected = [p for p in files if any(re.search(pattern, p, re.I) for pattern in (
            r'(^|/)(InferenceWeb\.Tests|PdfProcessFixture|testhost|xunit\.|Microsoft\.TestPlatform\.|Microsoft\.VisualStudio\.TestPlatform)',
            r'(^|/)(Fixtures|PdfProcessFixture|uploads|logs)/', r'(^|/).*Probe[^/]*\.(dll|exe)$'))]
        if rejected:
            report['rejected_test_files'] = rejected
            raise ValueError('SDK publish included test/probe/mutable validation files')
        required = ('TensorSharp.Server.Host.dll', 'TensorSharp.Server.Host.exe',
                    'TensorSharp.Server.Host.deps.json', 'TensorSharp.Server.Host.runtimeconfig.json',
                    'wwwroot/index.html', 'wwwroot/mask-editor.js', 'wwwroot/mask-editor.css',
                    'wwwroot/images/assistant_logo.png', 'wwwroot/images/banner_1.png')
        if any(name not in files for name in required):
            raise ValueError('Host launcher or required static asset is absent')
        copied = {}
        static = {}
        for name, metadata in files.items():
            source = report['compiled_before'].get(name)
            if name.endswith(('.dll', '.exe')) and source is not None:
                if source != metadata:
                    raise ValueError('Published assembly differs from compiled private OutDir: ' + name)
                copied[name] = metadata['sha256']
            if name.startswith('cuda_kernels/'):
                if source != metadata:
                    raise ValueError('Published PTX differs from the compiled private OutDir: ' + name)
            if name in required[4:]:
                asset = build_output / name
                if not asset.is_file() or metadata != {'bytes': asset.stat().st_size, 'sha256': digest(asset)}:
                    raise ValueError('Published static asset differs from the validated private output: ' + name)
                static[name] = metadata['sha256']
        for name in required[:4]:
            if files[name] != report['compiled_before'].get(name):
                raise ValueError('Published Host identity differs from the private build: ' + name)
        report['exact_compiled_identity'] = copied
        report['exact_static_asset_identity'] = static
        deps = json.loads((publish / 'TensorSharp.Server.Host.deps.json').read_text(encoding='utf-8-sig'))
        missing = []
        for target in deps['targets'].values():
            for library in target.values():
                for runtime in library.get('runtime', {}):
                    if Path(runtime).name not in files:
                        missing.append(runtime)
        if missing:
            report['missing_deps_runtime_assemblies'] = missing
            raise ValueError('Host deps runtime closure is incomplete')
        report['native_present'] = sorted(name for name in files if 'GgmlOps' in Path(name).name)
        report['publish_complete'] = True
        report['status'] = 'staged-managed-closure-native-required'
    except Exception as error:
        report['status'] = 'failed-preserved'
        report['error'] = str(error)
        raise
    finally:
        report['compiled_after'] = snapshot(build_output, compiled_input)
        report['user_bin_after'] = snapshot(user_bin)
        report['project_after'] = {str(p): digest(p) for p in source_files}
        report['compiled_unchanged'] = report['compiled_before'] == report['compiled_after']
        report['user_bin_unchanged'] = report['user_bin_before'] == report['user_bin_after']
        report['project_unchanged'] = report['project_before'] == report['project_after']
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if not all(report[k] for k in ('compiled_unchanged', 'user_bin_unchanged', 'project_unchanged')):
            report['status'] = 'failed-preserved'
            report['publish_complete'] = False
            report['error'] = 'A protected compiled/user/source identity changed'
        save(output / 'receipt.json', report)
    if not report['publish_complete']:
        raise ValueError(report['error'])
    return {'path': str(publish), 'receipt': str(output / 'receipt.json'),
            'files': len(report['published_files']), 'status': report['status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--build-output', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--directory-build-props', type=Path,
                        help='Exact DirectoryBuildPropsPath used by the compiled private build')
    args = parser.parse_args()
    print(json.dumps(stage(args.repository, args.build_output, args.output,
                           directory_build_props=args.directory_build_props)))


if __name__ == '__main__':
    main()
