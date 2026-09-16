#!/usr/bin/env python3
"""Build and check a pinned source snapshot in a fresh VM tree.

This records functional evidence only. Other VM workloads can invalidate timing;
fixture passes never qualify trained-model quality or release performance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--dotnet', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--jobs', type=int, default=16)
    a = p.parse_args()
    repo, out = a.repo.resolve(), a.output.resolve()
    build = repo / 'TensorSharp.GGML.Native/build'
    if build.exists() or out.exists():
        p.error('Native build and output must be fresh; preserve previous runs')
    if sha(a.source_manifest) != a.source_manifest_sha256:
        p.error('Source manifest digest differs')
    source = json.loads(a.source_manifest.read_text())

    def source_audit():
        for name, expected in source['files'].items():
            path = (repo / name).resolve()
            if not path.is_relative_to(repo) or sha(path) != expected:
                raise ValueError('Source changed: ' + name)
        upstream = repo / 'ExternalProjects/ggml'
        revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=all'], text=True)
        if revision != source['upstream_revision'] or dirty:
            raise ValueError('Upstream revision or source differs')
        return {'files': len(source['files']), 'upstream_revision': revision, 'upstream_clean': True}

    initial = source_audit()
    out.mkdir(parents=True)
    report = {'status': 'running', 'release_qualified': False, 'steps': [],
              'source_manifest_sha256': a.source_manifest_sha256, 'source_before': initial,
              'started_at_unix': time.time(), 'gpu_selection': a.gpus,
              'limitations': ['Functional checks under shared VM load; no performance qualification.',
                              'Unavailable device counts and absent trained-model scenarios are not passes.']}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, TENSORSHARP_GGML_NO_UPDATE='1',
               DOTNET_ROOT=str(a.dotnet.parent), DOTNET_CLI_TELEMETRY_OPTOUT='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', TS_DSV4_THREADS='2', TS_DSV41_ENGRAM_THREADS='2')
    env['PATH'] = str(a.dotnet.parent) + ':/usr/local/cuda/bin:' + env['PATH']
    for name in ('GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'NVIDIA_TF32_OVERRIDE'):
        env.pop(name, None)
    report['environment_overrides'] = {k: v for k, v in env.items()
        if k in ('CUDA_VISIBLE_DEVICES', 'TENSORSHARP_GGML_NO_UPDATE', 'DOTNET_ROOT',
                 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'TS_DSV4_THREADS', 'TS_DSV41_ENGRAM_THREADS', 'PYTHONDONTWRITEBYTECODE')}

    def save():
        (out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, command, extra=None, timeout=3600):
        row = {'name': name, 'command': [str(x) for x in command], 'environment_overrides': extra or {},
               'started_at_unix': time.time(), 'status': 'running'}
        report['steps'].append(row)
        save()
        with (out / (name + '.log')).open('x') as log:
            process = subprocess.Popen(row['command'], cwd=repo, env={**env, **(extra or {})},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        row.update(exit_code=code, wall_seconds=time.time() - row['started_at_unix'],
                   status='passed' if code == 0 else 'unavailable' if code == 77 else 'failed')
        save()
        print(name, row['status'], code, flush=True)
        return code

    try:
        configure = ['cmake', '-S', repo / 'TensorSharp.GGML.Native', '-B', build, '-G', 'Unix Makefiles',
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                     '-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON', '-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF',
                     '-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON', '-DCMAKE_CUDA_ARCHITECTURES=86-real',
                     '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc', '-DTSG_CUDNN_INCLUDE_DIR=/usr/include',
                     '-DTENSORSHARP_GGML_NATIVE_ENABLE_NCCL=ON', '-DNCCL_INCLUDE_DIR=/usr/include',
                     '-DNCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so']
        if run('configure', configure) or run('native-build', ['cmake', '--build', build, '--verbose', '-j', str(a.jobs)]):
            raise RuntimeError('Fresh native build failed')
        library = build / 'libGgmlOps.so'
        report['native_sha256'] = sha(library)
        report['native_objects_sha256'] = {str(x.relative_to(build)): sha(x) for x in sorted(build.rglob('*.o'))}
        common = ['-c', 'Release', '-p:TensorSharpSkipGgmlNative=true', '-p:TensorSharpSkipMlxNative=true',
                  '-p:CudaArch=compute_86', '-m:4', '/nodeReuse:false']
        managed_codes = [run(label, [a.dotnet, 'build', project, *common]) for label, project in (
            ('managed-build', 'TensorSharp.Server.Host/TensorSharp.Server.Host.csproj'),
            ('managed-tests-build', 'InferenceWeb.Tests/InferenceWeb.Tests.csproj'),
            ('managed-benchmark-build', 'benchmarks/AgentTurnBench/AgentTurnBench.csproj'))]
        app = repo / 'TensorSharp.Server.Host/bin'
        report['managed_assemblies'] = {x.name: sha(x) for x in sorted(app.glob('TensorSharp.*.dll'))}
        (out / 'managed-build.json').write_text(json.dumps({
            'status': 'failed' if any(managed_codes) else 'passed',
            'managed_assemblies': report['managed_assemblies'],
            'source_manifest_sha256': a.source_manifest_sha256}, indent=2) + '\n')
        run('c-abi-boundary', ['python3', 'eng/tests/dsv4-execution-boundary.py', '--library', library,
                              '--report', out / 'c-abi-boundary.json'])
        run('native-ctest', ['ctest', '--test-dir', build, '-V', '--output-on-failure', '-j', '1',
                             '--output-junit', out / 'native-tests.xml'])
        junit = ET.parse(out / 'native-tests.xml')
        report['ctest_cases'] = [{'name': c.get('name'), 'status': 'failed' if c.find('failure') is not None
                                 else 'unavailable' if c.find('skipped') is not None else 'passed'}
                                for c in junit.iter('testcase')]
        fixture_env = {'PYTHONPATH': '/workspace/tensorsharp-no-patch-20260915/fixture-python-deps'}
        for backend in ('CPU', 'CUDA'):
            for capacity in (32, 512):
                for alignment in (4, 64):
                    name = f'qwen-mtp-{backend}-{capacity}-{alignment}'
                    run(name, ['python3', 'eng/tests/qwen4exp-mtp-operator.py', '--sample',
                        a.fixtures / 'qwen4exp-mtp/sample.json', '--library', library, '--backend', backend,
                        '--capacity', str(capacity), '--kv-alignment', str(alignment), '--report', out / (name + '.json')], fixture_env)
            name = 'qwen-target-' + backend
            run(name, ['python3', 'eng/tests/qwen4exp-target-snapshot.py', '--sample',
                a.fixtures / 'qwen4exp-mtp/sample.json', '--library', library, '--backend', backend,
                '--geometry', 'gdn32', '--report', out / (name + '.json')], fixture_env)
            name = 'qwen-qsa-' + backend
            run(name, ['python3', 'eng/tests/qwen4exp-qsa.py', '--library', library, '--backend', backend,
                '--expected-native-sha256', report['native_sha256'], '--output', out / name], fixture_env)
        if not any(managed_codes):
            run('managed-portable', [a.dotnet, 'test', 'InferenceWeb.Tests/InferenceWeb.Tests.csproj',
                *common, '--no-build', '--no-restore', '--filter', 'Requires!=Cuda&Requires!=Mlx&Requires!=Models&Requires!=Video',
                '--logger', 'trx;LogFileName=portable.trx', '--results-directory', out / 'portable'])
        report['source_after'] = source_audit()
        report['final_source_and_binary_identity_passed'] = sha(library) == report['native_sha256']
        report['status'] = 'failed-gates' if any(x['status'] != 'passed' for x in report['steps']) else 'native-gates-completed-full-model-unqualified'
    except Exception as error:
        report.update(status='build-or-audit-failed', error=repr(error), final_source_and_binary_identity_passed=False)
    finally:
        report['finished_at_unix'] = time.time()
        save()
    return 0 if report['status'] == 'native-gates-completed-full-model-unqualified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
