#!/usr/bin/env python3
"""Stage a Windows-built public GLM session probe through its own SDK project.

No build, restore, model, native injection, or Host deployment inference. Use the
same private props/OutDir as the qualified managed build and a fresh destination.
This verifies a Windows .exe apphost and portable SDK runtime assets. It does
not qualify a native Linux/macOS publish or execute any of those runtime assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def compiled_input(path, root):
    return (path.suffix.lower() in ('.dll', '.exe', '.pdb', '.ptx', '.so', '.dylib')
        or re.search(r'\.so\.[0-9.]+$', path.name) is not None
        or path.relative_to(root).parts[0].lower() == 'runtimes'
        or path.name.endswith(('.deps.json', '.runtimeconfig.json',
                               '.staticwebassets.runtime.json', '.staticwebassets.endpoints.json')))


def snapshot(root, compiled_only=False):
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and (not compiled_only or
                compiled_input(p, root))}


def runtime_closure(deps, published, compiled):
    """Bind portable SDK runtimeTargets by exact RID path, never just basename."""
    regular, targets = {}, {}
    for target in deps['targets'].values():
        for library in target.values():
            for asset in library.get('runtime', {}):
                relative = PurePosixPath(asset).name
                if relative not in published or published[relative] != compiled.get(relative):
                    raise ValueError('Missing or changed SDK runtime assembly: ' + asset)
                regular[asset] = {'published_path': relative, **published[relative]}
            for asset, metadata in library.get('runtimeTargets', {}).items():
                path = PurePosixPath(asset)
                if (path.is_absolute() or '..' in path.parts or '\\' in asset
                        or path.as_posix() != asset or path.parts[:1] != ('runtimes',)):
                    raise ValueError('Unsupported or escaping SDK runtimeTarget path: ' + asset)
                if asset not in published or published[asset] != compiled.get(asset):
                    raise ValueError('Missing or changed exact SDK runtimeTarget: ' + asset)
                targets[asset] = {'rid': metadata.get('rid'), 'asset_type': metadata.get('assetType'),
                                 **published[asset]}
    return {'runtime_assemblies': regular, 'runtime_targets': targets}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def stage(args):
    if os.name != 'nt':
        raise ValueError('This tool stages the Windows .exe apphost; native Linux/macOS publish is not qualified')
    repo, compiled, output, props = [p.resolve() for p in
        (args.repository, args.build_output, args.output, args.directory_build_props)]
    project = repo / 'eng/validation/GlmUnifiedSessionProbe/GlmUnifiedSessionProbe.csproj'
    user_bin = repo / 'TensorSharp.Server.Host/bin'
    name = 'GlmUnifiedSessionProbe'
    if not project.is_file() or not props.is_file() or not (compiled / (name + '.dll')).is_file():
        raise ValueError('Require existing SDK probe project, exact build props, and compiled probe')
    if (output.exists() or output == compiled or output.is_relative_to(compiled)
            or output == user_bin or output.is_relative_to(user_bin)):
        raise ValueError('Require a fresh stage outside the compiled output and protected Host bin')
    output.mkdir(parents=True, exist_ok=False)
    publish = output / 'publish'
    command = [args.dotnet, 'publish', str(project), '--configuration', 'Release',
        '--no-build', '--no-restore', '--verbosity', 'diagnostic', '-m:1',
        '-p:BuildProjectReferences=false', '-p:UseSharedCompilation=false',
        '-p:DirectoryBuildPropsPath=' + props.as_posix(),
        '-p:OutDir=' + compiled.as_posix() + '/', '-p:OutputPath=' + compiled.as_posix() + '/',
        '-p:PublishDir=' + publish.as_posix() + '/',
        '-p:TensorSharpSkipGgmlNative=true', '-p:TensorSharpSkipMlxNative=true',
        '-p:GgmlNativeBinaryName=', '-p:GgmlNativeBinaryNameForTests=', '-p:MlxNativeBinaryName=',
        '-p:CudaArch=compute_86']
    sources = [project, props, repo / 'Directory.Build.props', repo / 'Directory.Build.targets',
        repo / 'TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj',
        repo / 'TensorSharp.Backends.GGML/TensorSharp.Backends.GGML.csproj']
    report = {'schema': 'tensorsharp.glm-session-sdk-publish.v1', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
        'project': str(project), 'compiled_output': str(compiled), 'publish': str(publish),
        'tool_sha256': sha(__file__), 'publish_complete': False, 'native_injected': False,
        'model_or_gpu_executed': False,
        'scope': 'Windows SDK publish for this probe project and declared portable runtime assets; no Host-deployment inference, runtime-asset execution, native Linux/macOS publish, native ABI or model qualification.'}
    report['compiled_before'] = snapshot(compiled, True)
    report['user_bin_before'] = snapshot(user_bin)
    report['source_before'] = {str(p): sha(p) for p in sources}
    save(output / 'receipt.json', report)
    try:
        with (output / 'publish.log').open('wb') as log:
            completed = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False)
        report['exit_code'] = completed.returncode
        if completed.returncode:
            raise ValueError('Probe SDK publish failed; no build/restore fallback is allowed')
        text = (output / 'publish.log').read_text(encoding='utf-8-sig', errors='replace')
        started = re.findall(r'Task "(Csc|Fsc|Vbc|CompileCudaPtx)"(?:\s|\()', text)
        if started or re.search(r'Building GGML native library|Building MLX native library|Compiling .*\.cu to PTX|Restoring packages for', text):
            raise ValueError('No-build publish unexpectedly started compilation or restore')
        report['compiler_tasks_started'] = started
        files = report['published_files'] = snapshot(publish)
        rejected = [p for p in files if
            re.search(r'(^|/)(InferenceWeb\.Tests|UnifiedMemory\.Tests|PdfProcessFixture|testhost|xunit\.|Microsoft\.TestPlatform\.|Microsoft\.VisualStudio\.TestPlatform|TensorSharp\.Server\.Host)', p, re.I)
            or re.search(r'(^|/)(Fixtures|uploads|logs)/', p, re.I)
            or (re.search(r'(^|/).*Probe[^/]*\.(dll|exe)$', p, re.I)
                and p not in (name + '.dll', name + '.exe'))
            or 'GgmlOps' in Path(p).name]
        if rejected:
            report['rejected_files'] = rejected
            raise ValueError('Probe SDK publish included unrelated test/Host/probe/native/mutable output')
        required = [name + suffix for suffix in ('.dll', '.exe', '.deps.json', '.runtimeconfig.json')]
        for relative in required:
            if files.get(relative) != report['compiled_before'].get(relative) or relative not in files:
                raise ValueError('Probe launch identity missing or differs from exact compiled output: ' + relative)
        for relative, identity in files.items():
            if compiled_input(publish / relative, publish):
                if identity != report['compiled_before'].get(relative):
                    raise ValueError('Published runtime/launch input differs from exact managed build: ' + relative)
        deps = json.loads((publish / (name + '.deps.json')).read_text(encoding='utf-8-sig'))
        report['deps_runtime_closure'] = runtime_closure(deps, files, report['compiled_before'])
        report['publish_complete'] = True
        report['status'] = 'staged-sdk-probe-closure-native-required'
    except Exception as error:
        report['status'] = 'failed-preserved'
        report['error'] = repr(error)
        raise
    finally:
        report['compiled_after'] = snapshot(compiled, True)
        report['user_bin_after'] = snapshot(user_bin)
        report['source_after'] = {str(p): sha(p) for p in sources}
        report['compiled_unchanged'] = report['compiled_before'] == report['compiled_after']
        report['user_bin_unchanged'] = report['user_bin_before'] == report['user_bin_after']
        report['source_unchanged'] = report['source_before'] == report['source_after']
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if not all(report[key] for key in ('compiled_unchanged', 'user_bin_unchanged', 'source_unchanged')):
            report.update(status='failed-preserved', publish_complete=False,
                          error='A protected compiled/source/user-bin identity changed')
        save(output / 'receipt.json', report)
    if not report['publish_complete']:
        raise ValueError(report['error'])
    return {'status': report['status'], 'publish': str(publish), 'receipt': str(output / 'receipt.json'),
            'files': len(report['published_files'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--build-output', type=Path, required=True)
    parser.add_argument('--directory-build-props', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dotnet', default='dotnet')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    print(json.dumps(stage(args)))


if __name__ == '__main__':
    main()
