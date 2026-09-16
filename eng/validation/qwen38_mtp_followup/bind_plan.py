#!/usr/bin/env python3
"""Bind prepared Qwen-only commands to completed build/app evidence. Never executes them."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def read_pin(path, digest):
    raw = Path(path).read_bytes()
    if not re.fullmatch('[0-9a-f]{64}', digest or '') or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Manifest identity changed: ' + str(path))
    return json.loads(raw)


def qualify(build, managed, application, allow):
    if not build.get('final_source_and_binary_identity_passed'):
        raise ValueError('Final source/native/upstream audit not complete')
    if not re.fullmatch('[0-9a-f]{64}', build.get('native_sha256', '')):
        raise ValueError('Missing actual native digest')
    rows = {r['name']: r for r in build['steps']}
    required = ['managed-build', 'c-abi-boundary'] + [f'qwen-target-{b}' for b in ('CPU', 'CUDA')] + [
        f'qwen-mtp-{b}-{c}-{a}' for b in ('CPU', 'CUDA') for c in (32, 512) for a in (4, 64)]
    for name in required:
        if name not in rows or rows[name].get('exit_code') != 0:
            raise ValueError('Applicable Qwen/API/native gate not passed: ' + name)
    failed = [r['name'] for r in build['steps'] if r.get('status') == 'failed']
    if failed and not allow:
        raise ValueError('Original build has failed gates; explicit Qwen-only unqualified declaration required')
    # No completed build here establishes full-model/speculation performance.
    if build.get('status') not in ('failed-gates', 'native-gates-completed-full-model-unqualified'):
        raise ValueError('Native build is incomplete or failed outside its recorded gate set')
    expected = {k: v for k, v in build['managed_assemblies'].items() if k.startswith('TensorSharp.') and k.endswith('.dll')}
    if managed.get('status') != 'passed' or managed.get('managed_assemblies') != expected:
        raise ValueError('Separate passed managed build pin differs from native build application')
    actual = {k: v for k, v in application['non_native_files_sha256'].items() if '/' not in k and k.startswith('TensorSharp.') and k.endswith('.dll')}
    if actual != expected:
        raise ValueError('Application file set differs from built managed assemblies')
    return {'release_qualified': False, 'original_build_status': build['status'], 'original_failed_steps': failed,
            'applicable_required_passed_steps': required, 'scope': 'Qwen4Exp three-device layer placement only; does not qualify or exercise DeepSeek routed-expert TP',
            'explicit_unqualified_qwen_only': allow}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'build', 'managed-build', 'application-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--remote-package', required=True)
    p.add_argument('--remote-application-manifest', required=True)
    p.add_argument('--server-assembly', required=True)
    p.add_argument('--native', required=True)
    p.add_argument('--allow-unqualified-qwen-only', action='store_true')
    p.add_argument('--remote-output', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    package = read_pin(a.package, a.package_sha256)
    package_root = a.package.parent
    for name, expected in package['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or hashlib.sha256((package_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Prepared file changed: ' + name)
    build = read_pin(a.build, a.build_sha256)
    managed = read_pin(a.managed_build, a.managed_build_sha256)
    application = read_pin(a.application_manifest, a.application_manifest_sha256)
    if application['managed_build']['sha256'] != a.managed_build_sha256:
        raise ValueError('Application bound to a different managed build')
    if build['source_manifest_sha256'] != package['native_source_package_sha256']:
        raise ValueError('Different native source campaign; prepare a distinct reviewed package')
    qualification = qualify(build, managed, application, a.allow_unqualified_qwen_only)
    remote = a.remote_package.rstrip('/'); out = a.remote_output.rstrip('/')
    dotnet = '/workspace/tensorsharp-no-patch-20260915/dotnet/dotnet'
    model = package['model']['trunk'][0]['path']; head = package['model']['head']['path']; mmproj = package['model']['mmproj']['path']
    image = package['fixtures_root'] + '/card-0.png'
    app = str(PurePosixPath(a.server_assembly).parent)
    base_args = [dotnet, out + '/bench-app/QwenMtpFollowup.dll', '--model', model, '--backend', 'ggml_cuda', '--draft-model', head,
                 '--mmproj', mmproj, '--image', image, '--kv', 'f16', '--chunk', '512', '--max-batched', '4096', '--warmup', '0',
                 '--measure-passes', '1', '--scenarios', 'qwen-mtp']
    env = {'CUDA_VISIBLE_DEVICES': '0,1,2', 'TENSORSHARP_TP_DEGREE': '3', 'MAX_CONTEXT': '65536', 'KV_CACHE_DTYPE': 'f16',
           'TS_SPEC_DRAFT': '3', 'TS_SPEC_TYPE': 'auto', 'TS_SPEC_PMIN': '0.15', 'TS_CPU_MOE_THREADS': '16',
           'PYTHONDONTWRITEBYTECODE': '1'}
    bench = []
    for tier in ('dense', 'long-qsa-unqualified'):
        for repeat, order in enumerate((('plain', 'mtp'), ('mtp', 'plain'), ('plain', 'mtp')), 1):
            for mode in order:
                bench.append({'id': f'{tier}-r{repeat}-{mode}', 'tier': tier, 'repeat': repeat, 'mode': mode,
                    'env': {**env, 'QWEN_MTP_MODE': mode, 'QWEN_MTP_TIER': tier, 'TS_SPEC': '0' if mode == 'plain' else '1'},
                    'command': base_args + ['--out', f'{out}/{tier}-r{repeat}-{mode}.rows.json'], 'timeout_seconds': 3600,
                    'requires_exclusive_gpu_lane': True, 'expected_native_sha256': build['native_sha256']})
    a.output.mkdir(parents=True, exist_ok=False)
    profiles, commands = [], []
    template = json.loads((package_root / 'http-profile-template.json').read_bytes())
    for mode in ('plain', 'mtp'):
        profile = copy.deepcopy(template); profile['id'] = 'qwen38-mtp-v7-unqualified-http-' + mode
        profile['expected_native_sha256'] = build['native_sha256']
        profile['env'].update(env, TS_SPEC='0' if mode == 'plain' else '1', BENCH_CONFIG=remote + '/repo/benchmarks/engine_comparison/benchmark_config.json')
        profile['extra_args'] += ['--draft-model', head, '--spec-draft', '3', '--spec-type', 'auto', '--spec-pmin', '0.15', '--no-spec' if mode == 'plain' else '--spec']
        for suite in profile['suites']:
            for i, value in enumerate(suite):
                if value == '@PROFILE@': suite[i] = profile['id']
                elif value.startswith('@PACKAGE@'): suite[i] = remote + value[len('@PACKAGE@'):]
        profile['expected_harness_sha256'] = {k[len('repo/'):]: v for k, v in package['files'].items() if k.startswith('repo/')}
        path = a.output / (mode + '-http.json'); raw = (json.dumps(profile, indent=2) + '\n').encode(); path.write_bytes(raw)
        profiles.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'mode': mode})
        commands.append({'mode': mode, 'command': ['python', remote + '/repo/eng/validation/run-release-profile.py', '--repo', remote + '/repo',
            '--profile', remote + '/bound/' + path.name, '--dotnet', dotnet, '--native', a.native, '--output', out + '/http-' + mode,
            '--server-assembly', a.server_assembly, '--application-manifest', a.remote_application_manifest,
            '--application-manifest-sha256', a.application_manifest_sha256]})
    record = {'status': 'bound-not-executed', **qualification, 'input_pins': {name: getattr(a, name.replace('-', '_') + '_sha256') for name in ('package', 'build', 'managed-build', 'application-manifest')},
              'expected_native_sha256': build['native_sha256'], 'server_assembly': a.server_assembly,
              'benchmark_build': [dotnet, 'build', remote + '/benchmark/QwenMtpFollowup.csproj', '-c', 'Release', '-m:2',
                                  '-p:FrozenManagedDir=' + app, '--artifacts-path', out + '/bench-artifacts', '-o', out + '/bench-app'],
              'benchmark_runs': bench, 'http_profiles': profiles, 'http_commands': commands,
              'runtime_obligations': package['runtime_obligations'], 'timing_order': 'Three independent warmed process pairs: plain/MTP, MTP/plain, plain/MTP. Preserve order and all failures.'}
    (a.output / 'launch-plan.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Bound twelve benchmark processes and two HTTP lifecycles; no process launched')


if __name__ == '__main__':
    main()
