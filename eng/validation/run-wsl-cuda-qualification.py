#!/usr/bin/env python3
"""Qualify an already-built, frozen WSL CUDA library with serial owned probes.

No production-library build, WSL startup, download or OS memory-limit installation
occurs here. Small portable CPU fixtures are compiled in the isolated output.
The explicit plan pins supplementary tools/app files and the model identity;
the required new native pin comes from the successful isolated build report.
Existing supervisors own deadlines and cleanup. This is not a benchmark server.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_extra(root, plan):
    require(digest(root / 'frozen-inputs.json') == plan['base_manifest_sha256'], 'Base frozen manifest changed')
    for item in plan['files']:
        path = (root / item['relative']).resolve()
        require(path.is_relative_to(root) and path != root, 'Supplementary input escapes frozen root')
        require(path.is_file() and digest(path) == item['sha256'], 'Supplementary input changed: ' + str(path))


def observed_environment():
    return {key: value for key, value in os.environ.items()
            if key.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'NCCL_', 'OMP_', 'GOMP_'))
            or key in ('CUDA_VISIBLE_DEVICES', 'KV_CACHE_DTYPE', 'MAX_CONTEXT', 'NVIDIA_TF32_OVERRIDE',
                       'TMPDIR', 'TEMP', 'TMP', 'CUDA_CACHE_PATH', 'LD_LIBRARY_PATH')}


@contextlib.contextmanager
def environment(root, overrides=None):
    previous = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'NCCL_', 'OMP_', 'GOMP_')) or key in (
                    'CUDA_VISIBLE_DEVICES', 'KV_CACHE_DTYPE', 'MAX_CONTEXT', 'SPECULATIVE_DECODING', 'NVIDIA_TF32_OVERRIDE'):
                del os.environ[key]
        os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', TS_GGML_CPU_THREADS='8',
                          TS_CPU_MOE_THREADS='8', TENSORSHARP_TP_DEGREE='1', TENSORSHARP_LAYER_SPLIT_DEGREE='1',
                          KV_CACHE_DTYPE='f16', TMPDIR=str(root / 'tmp'), TEMP=str(root / 'tmp'), TMP=str(root / 'tmp'),
                          CUDA_CACHE_PATH=str(root / 'cuda-cache'), PYTHONDONTWRITEBYTECODE='1')
        os.environ.update(overrides or {})
        yield
    finally:
        os.environ.clear(); os.environ.update(previous)


def adaptive_gate(report, mode, repeats, model_hash, native_hash):
    require(report.get('Executed') is True and not report.get('Error') and report.get('NativeShutdown') is True,
            'Adaptive/raw probe did not complete cleanup')
    require(report.get('Mode') == mode and report.get('ModelSha256', '').lower() == model_hash,
            'Wrong mode or model payload identity')
    require(len(report.get('Native', [])) == 1 and report['Native'][0]['Sha256'].lower() == native_hash,
            'Probe did not observe the newly built native library')
    rows = report.get('Records', [])
    require([row.get('Run') for row in rows] == list(range(-1, repeats)), 'Missing warmup/measured requests')
    require(all(row.get('Warmup') is (i == 0) and row.get('DecodeCalls') == 63
                and len(row.get('Generated', [])) == 64 for i, row in enumerate(rows)), 'Incomplete prediction rows')
    if mode == 'adaptive':
        require(report.get('AfterDispose') and all(row['Reserved'] == 0 and row['Committed'] == 0
                for row in report['AfterDispose']), 'Adaptive owners remain after disposal')
        require(all(row.get('Before', {}).get('PhysicalPlan') is not None for row in rows),
                'Adaptive request did not report its physical capacity plan')


def upload_gate(report, native_hash):
    pools = report.get('final_budget')
    require(isinstance(pools, list) and len(pools) == 2 and
            {pool.get('Pool') for pool in pools} == {'host', 'device'} and
            all(pool.get('Reserved') == 0 and pool.get('Committed') == 0 for pool in pools),
            'Uploader final host/device ownership evidence missing or nonzero')
    require(report.get('passed') is True and report.get('full_logits_bit_exact') is True
            and report.get('logit_elements') == 512 and report.get('refusals', 0) > 0
            and len(report.get('native', [])) == 1 and report['native'][0]['sha256'] == native_hash,
            'Real uploader budget/numerical gate failed')


def teacher_gate(report, teacher, teacher_hash):
    require(report.get('Generation') == 'teacher-forced' and report.get('Teacher') == teacher
            and report.get('TeacherSha256', '').lower() == teacher_hash
            and all(row.get('Consumed') == teacher[:-1] for row in report['Records']),
            'Explicit teacher content/hash or consumed history changed')


def performance_gate(rates):
    metrics = ('PrefillTokensPerSecond', 'DecodeTokensPerSecond')
    ratios = rates.get('AdaptiveToResidentRatio', {})
    require(set(ratios) == set(metrics) and all(type(ratios[key]) in (float, int)
            and math.isfinite(ratios[key]) and ratios[key] > 0 for key in metrics),
            'Incomplete or invalid performance ratios')
    return dict(passed=all(ratios[key] >= .95 for key in metrics), minimum_ratio=.95,
                adaptive_to_resident_ratio=ratios,
                scope='Measured ABBA medians: neither prefill nor decode may be more than 5% slower. Six measured samples per arm; ranges retained; no statistical or cross-hardware guarantee.')


def semantic_gate(report, native_hash):
    require(report.get('repeated_final_logits_exact') is True and
            [value.lower() for value in report.get('native_sha256', {}).values()] == [native_hash] and
            [row['name'] for row in report.get('quality', [])] == ['math', 'extraction', 'squares'],
            'Semantic probe numeric/native identity or completed case evidence missing')
    return (report.get('passed') is True and report.get('quality_passed') is True
            and all(row.get('passed') is True and row.get('eos') is True and not row.get('reached_generation_limit')
                    for row in report['quality']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    require(sys.platform == 'linux' and str(root).startswith(('/mnt/e/', '/mnt/d/')),
            'Use the prepared D/E frozen directory from Linux/WSL')
    plan = load(root / 'qualification-plan.json')
    verify_extra(root, plan)
    output = root / 'qualification-evidence'
    output.mkdir(exist_ok=False)
    report = dict(passed=False, errors=[], phases=[], hard_limit_installed=False,
                  infrastructure_passed=False, correctness_passed=None, performance_passed=None, semantic_passed=None,
                  scope='Serial Linux/WSL CUDA correctness, sampled diagnostics and separate quiet timings; no full physical cap, TP, Apple, MTP or broad multimodal quality qualification.')
    started = time.monotonic()
    def interrupted(number, _frame):
        if 'interrupted_signal' in report:
            return
        report['interrupted_signal'] = number
        raise InterruptedError('Qualification received signal ' + str(number))
    signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    handlers = {number: signal.signal(number, interrupted) for number in signals}
    native_hash = None
    model = None
    freezer = None
    try:
        freezer = module(root / 'tools/prepare-wsl-cuda-validation.py')
        build_runner = module(root / 'tools/run-bounded-build.py')
        probe_runner = module(root / 'tools/run-bounded-probe.py')
        freezer.verify(root)
        built = load(root / 'build-evidence/execution.json')
        require(built.get('passed') is True and built.get('source_identity_unchanged') is True,
                'A successful current-source isolated native build is mandatory')
        native = root / 'build/libGgmlOps.so'
        native_hash = built['native']['sha256']
        require(digest(native) == native_hash, 'Built native hash changed')
        report['native_sha256'] = native_hash
        report['supplementary_plan_sha256'] = digest(root / 'qualification-plan.json')
        model = Path(plan['model_linux'])
        require(model.stat().st_size == plan['model_bytes'], 'Planned model size changed')
        report['model_stat_before'] = dict(bytes=model.stat().st_size, mtime_ns=model.stat().st_mtime_ns)
        for app in ('adaptive-app', 'upload-app'):
            require(digest(root / app / native.name) == native_hash, 'Wrong native deployed beside ' + app)
        target = root / 'semantic-app' / native.name
        require(not target.exists(), 'Refusing to replace an existing semantic native deployment')
        shutil.copyfile(native, target)
        require(digest(target) == native_hash, 'Semantic native copy mismatch')

        def cpu(name, command, timeout=120, overrides=None):
            phase = output / name
            with environment(root, overrides):
                launch_environment = observed_environment()
                code = build_runner.main(['--output', str(phase), '--timeout', str(timeout),
                    '--space-floor', str(root), str(20 << 30), '--space-floor', '/mnt/c', str(512 << 20), '--', *command])
            execution = load(phase / 'execution.json')
            report['phases'].append(dict(name=name, execution=execution, launch_environment=launch_environment))
            require(code == 0 and execution.get('passed') is True, 'CPU phase failed: ' + name)

        def probe(name, app, arguments, telemetry, overrides=None, timeout=300, allowed_exit_codes=(0,)):
            directory = root / app
            expected = {str(path): digest(path) for path in directory.iterdir() if path.is_file()}
            command = ['dotnet', str(directory / ('AdaptiveMemoryProbe.dll' if app == 'adaptive-app'
                        else 'GlmUploadBudgetProbe.dll' if app == 'upload-app' else 'DirectCudaModelProbe.dll')), *arguments]
            with environment(root, overrides):
                launch_environment = observed_environment()
                sampler = probe_runner.ProbeTelemetry(directory, native_hash) if telemetry else None
                execution = probe_runner.run(command, output / (name + '-owned'), timeout, sampler)
            report['phases'].append(dict(name=name, execution=execution, launch_environment=launch_environment,
                external_sampling='Enabled; overhead unquantified' if telemetry else 'Disabled for quiet/internal timing'))
            require(expected == {str(path): digest(path) for path in directory.iterdir() if path.is_file()},
                    'Application/native files changed during ' + name)
            require(execution.get('complete') is True and not execution.get('timed_out')
                    and execution.get('exit_code') in allowed_exit_codes and not execution.get('error')
                    and not execution.get('cleanup_errors'), 'Probe execution failed: ' + name)
            if telemetry:
                require(execution.get('telemetry', {}).get('passed') is True,
                        'Live native/sampled memory evidence incomplete: ' + name)

        cpu('cgroup-read-only', [sys.executable, str(root / 'qualification-tools/inspect-linux-memory-cgroup.py'),
            '--output', str(output / 'cgroup-inspection.json')], 30)
        cpu('elf-runtime-dependencies', ['ldd', str(native)], 30)
        dependencies = (output / 'elf-runtime-dependencies/process.log').read_text()
        require('not found' not in dependencies and '/stubs/' not in dependencies, 'Unresolved or stub runtime dependency')
        binaries = root / 'portable-fixtures'; binaries.mkdir(exist_ok=False)
        for name in ('glm_upload_buffer', 'glm_file_strip', 'glm_joined_workers', 'glm_host_mapping', 'glm_metadata_owner'):
            executable = binaries / name
            cpu(name + '-compile', ['g++', '-O2', '-std=c++17', '-pthread', '-I',
                str(root / 'src/ExternalProjects/ggml/include'), str(root / 'src/TensorSharp.GGML.Native/tests' / (name + '_test.cpp')),
                '-o', str(executable)])
            overrides = {'TMPDIR': '/dev/shm', 'TEMP': '/dev/shm', 'TMP': '/dev/shm'} if name == 'glm_host_mapping' else None
            cpu(name + '-run', [str(executable)], overrides=overrides)
        upload = output / 'upload'
        probe('upload', 'upload-app', ['CUDA', str(upload)], True,
              {'TS_GLM_LOAD_THREADS': '2', 'TS_GLM_LOAD_CHUNK_MB': '1', 'TS_GLM_MOE_MMAP': '0',
               'TS_CPU_MOE_THREADS': '2', 'OMP_NUM_THREADS': '2', 'TS_GGML_CPU_THREADS': '2'}, 180)
        uploaded = load(upload / 'report.json')
        upload_gate(uploaded, native_hash)
        common = ['--model', str(model), '--context', '2048', '--prompt-tokens', '640', '--steps', '64']
        seed = output / 'seed'
        probe('seed', 'adaptive-app', [*common, '--mode', 'resident', '--repeats', '1', '--capture-logits', 'false', '--output', str(seed)], False)
        seeded = load(seed / 'report.json'); adaptive_gate(seeded, 'resident', 1, plan['model_sha256'], native_hash)
        teacher_ids = seeded['Records'][1]['Generated']
        teacher = output / 'teacher.json'; teacher.write_text(json.dumps(teacher_ids))
        teacher_hash = digest(teacher)
        for mode in ('resident', 'adaptive'):
            directory = output / ('capture-' + mode)
            limits = ['--device-bytes', str(12 << 30), '--host-bytes', str(6 << 30)] if mode == 'adaptive' else []
            probe('capture-' + mode, 'adaptive-app', [*common, '--teacher', str(teacher), '--mode', mode,
                '--repeats', '1', '--capture-logits', 'true', '--output', str(directory), *limits], True)
            captured = load(directory / 'report.json')
            adaptive_gate(captured, mode, 1, plan['model_sha256'], native_hash)
            teacher_gate(captured, teacher_ids, teacher_hash)
        comparison = module(root / 'qualification-tools/compare-captures.py').compare(
            output / 'capture-resident/report.json', output / 'capture-adaptive/report.json')
        (output / 'full-row-comparison.json').write_text(json.dumps(comparison, indent=2))
        require(comparison['Passed'] and all(row['BitwiseEqual'] for row in comparison['Rows']),
                'Complete same-history resident/adaptive logits are not bit-exact; diagnose before throughput')
        quiet = []
        for index, mode in enumerate(('resident', 'adaptive', 'adaptive', 'resident')):
            name = f'quiet-{index}-{mode}'; directory = output / name
            limits = ['--device-bytes', str(12 << 30), '--host-bytes', str(6 << 30)] if mode == 'adaptive' else []
            probe(name, 'adaptive-app', [*common, '--teacher', str(teacher), '--mode', mode, '--repeats', '3',
                '--capture-logits', 'false', '--output', str(directory), *limits], False)
            measured = load(directory / 'report.json')
            adaptive_gate(measured, mode, 3, plan['model_sha256'], native_hash)
            teacher_gate(measured, teacher_ids, teacher_hash)
            quiet.append(directory / 'report.json')
        rates = module(root / 'qualification-tools/compare-runs.py').compare(quiet)
        (output / 'quiet-abba-comparison.json').write_text(json.dumps(rates, indent=2))
        require(rates['CorrectnessAndLifecyclePassed'], 'Quiet ABBA completeness/parity failed')
        report['correctness_passed'] = True
        report['performance'] = performance_gate(rates)
        report['performance_passed'] = report['performance']['passed']
        semantics = output / 'semantics.json'
        probe('semantics', 'semantic-app', ['--model', str(model), '--backend', 'ggml_cuda', '--output', str(semantics),
            '--prefill-tokens', '8', '--decode-tokens', '1', '--warmup', '0', '--iterations', '1', '--max-context', '2048',
            '--quality', 'true', '--quality-parser', 'production', '--quality-cases', 'math,extraction,squares',
            '--quality-reasoning-effort', 'low', '--max-new', '256'], True, allowed_exit_codes=(0, 1))
        semantic = load(semantics)
        report['semantic_passed'] = semantic_gate(semantic, native_hash)
        report['semantic_results'] = [dict(name=row['name'], passed=row.get('passed'), eos=row.get('eos'),
            reached_generation_limit=row.get('reached_generation_limit')) for row in semantic['quality']]
        report['semantic_scope'] = plan['semantic_scope']
        report['infrastructure_passed'] = True
        report['passed'] = all(report[key] is True for key in
            ('infrastructure_passed', 'correctness_passed', 'performance_passed', 'semantic_passed'))
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        for number in signals: signal.signal(number, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
        try:
            require(freezer is not None, 'Frozen verifier was not imported')
            freezer.verify(root); verify_extra(root, plan)
            if native_hash:
                require(digest(root / 'build/libGgmlOps.so') == native_hash, 'Native changed after qualification')
                for app in ('adaptive-app', 'upload-app', 'semantic-app'):
                    target = root / app / 'libGgmlOps.so'
                    if target.exists():
                        require(digest(target) == native_hash, 'Deployed native changed: ' + app)
            if model is not None and 'model_stat_before' in report:
                report['model_stat_after'] = dict(bytes=model.stat().st_size, mtime_ns=model.stat().st_mtime_ns)
                require(report['model_stat_after'] == report['model_stat_before'], 'Model stat changed during qualification')
            report['input_identities_unchanged'] = True
        except BaseException as error:
            report['errors'].append('Final identity: ' + repr(error)); report['passed'] = False
            report['infrastructure_passed'] = False
        report['seconds'] = time.monotonic() - started
        (output / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for number, handler in handlers.items(): signal.signal(number, handler)
    print(json.dumps({key: report[key] for key in ('passed', 'errors', 'seconds')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
