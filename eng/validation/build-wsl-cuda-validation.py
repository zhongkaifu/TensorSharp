#!/usr/bin/env python3
"""Build frozen current native CUDA sources on D/E under WSL, without model runs.

Every external tool uses the owned process-group supervisor. Source/archive
verification and the final execution report run even after a build failure.
The output must already have been prepared by prepare-wsl-cuda-validation.py.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def sibling(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_space(path, minimum):
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise OSError(f'Low disk space at {path}: {free} bytes free, require {minimum}.')
    return dict(path=str(path), free_bytes=free, minimum_bytes=minimum)


def is_wsl_output_path(path):
    return str(path).startswith(('/mnt/d/', '/mnt/e/'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--minimum-output-free-bytes', type=int, default=20 << 30)
    parser.add_argument('--minimum-system-free-bytes', type=int, default=512 << 20)
    parser.add_argument('--system-volume', type=Path, default=Path('/mnt/c'))
    args = parser.parse_args(argv)
    if min(args.minimum_output_free_bytes, args.minimum_system_free_bytes) <= 0:
        parser.error('Both disk space floors must be positive.')
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('A prepared output directory is required.')
    logs = root / 'build-evidence'
    logs.mkdir(exist_ok=False)
    report = dict(passed=False, errors=[], cleanup_errors=[], phases=[],
                  source_identity_unchanged=False, model_validation_executed=False)
    previous_handlers = {}
    freezer = None
    started = time.monotonic()
    try:
        report['phase'] = 'disk-space-preflight'
        report['space_before'] = [check_space(root, args.minimum_output_free_bytes),
                                  check_space(args.system_volume, args.minimum_system_free_bytes)]
        if sys.platform != 'linux' or not is_wsl_output_path(root):
            raise RuntimeError('Use an isolated /mnt/d or /mnt/e directory in Linux/WSL.')
        def interrupted(number, _frame):
            if 'interrupted_signal' in report:
                return
            report['interrupted_signal'] = number
            raise InterruptedError(f'Build driver received signal {number}.')
        previous_handlers = {number: signal.signal(number, interrupted)
                             for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        freezer = sibling('prepare-wsl-cuda-validation')
        supervisor = sibling('run-bounded-build')
        freezer.verify(root)
        for key, value in {'TMPDIR': 'tmp', 'TEMP': 'tmp', 'TMP': 'tmp',
                           'CUDA_CACHE_PATH': 'cuda-cache', 'CCACHE_DIR': 'ccache',
                           'XDG_CACHE_HOME': 'cache', 'DOTNET_CLI_HOME': 'dotnet-home',
                           'NUGET_PACKAGES': 'nuget', 'NUGET_HTTP_CACHE_PATH': 'nuget-http'}.items():
            directory = root / value; directory.mkdir(exist_ok=True)
            os.environ[key] = str(directory)
        os.environ.update(CCACHE_DISABLE='1', CMAKE_BUILD_PARALLEL_LEVEL='2',
                          DOTNET_NOLOGO='1', DOTNET_CLI_TELEMETRY_OPTOUT='1', PYTHONDONTWRITEBYTECODE='1')
        report['build_environment'] = {key: os.environ[key] for key in (
            'TMPDIR', 'TEMP', 'TMP', 'CUDA_CACHE_PATH', 'CCACHE_DIR', 'XDG_CACHE_HOME',
            'DOTNET_CLI_HOME', 'NUGET_PACKAGES', 'NUGET_HTTP_CACHE_PATH', 'CCACHE_DISABLE',
            'CMAKE_BUILD_PARALLEL_LEVEL', 'PYTHONDONTWRITEBYTECODE')}
        report['resolved_tools'] = {name: shutil.which(name) for name in
            ('cmake', 'gcc', 'g++', 'ninja', 'make', 'git', 'nvidia-smi', 'dotnet', 'tar', 'readelf', 'nm', 'df')}
        def run(name, command, timeout):
            report['phase'] = name
            report['space_latest'] = [check_space(root, args.minimum_output_free_bytes),
                                      check_space(args.system_volume, args.minimum_system_free_bytes)]
            output = logs / name
            code = supervisor.main(['--output', str(output), '--timeout', str(timeout),
                                    '--space-floor', str(root), str(args.minimum_output_free_bytes),
                                    '--space-floor', str(args.system_volume), str(args.minimum_system_free_bytes),
                                    '--', *command])
            execution = json.loads((output / 'execution.json').read_text())
            report['phases'].append(dict(name=name, supervisor_exit=code, execution=execution))
            if code or not execution['passed']:
                raise RuntimeError('Validation phase failed: ' + name)
        for name, command in (
            ('cmake-version', ['cmake', '--version']), ('gcc-version', ['gcc', '--version']),
            ('gxx-version', ['g++', '--version']),
            ('cuda-version', ['/usr/local/cuda-12.6/bin/nvcc', '--version']),
            ('gpu-inventory', ['nvidia-smi']), ('dotnet-runtimes', ['dotnet', '--list-runtimes']),
            ('filesystem', ['df', '-T', str(root), str(args.system_volume), '/']),
            ('python-telemetry', [sys.executable, '-c', 'import sys,psutil; print(sys.version); print(psutil.__version__)'])):
            run(name, command, 30)
        run('supervisor-cpu-tests', [sys.executable, '-m', 'unittest', 'discover', '-s',
                                    str(root / 'tools/tests'), '-p', 'test_*build*.py', '-v'], 90)
        upstream = root / 'src/ExternalProjects/ggml'
        if upstream.exists():
            raise FileExistsError('Refusing to extract upstream over an existing checkout.')
        run('extract-upstream', ['tar', '-xf', str(root / 'ggml-upstream.tar'), '-C', str(root / 'src')], 120)
        freezer.verify(root)
        generator = 'Ninja' if shutil.which('ninja') else 'Unix Makefiles'
        report['generator'] = generator
        run('generator-version', ['ninja', '--version'] if generator == 'Ninja' else ['make', '--version'], 30)
        run('configure', ['cmake', '-S', str(root / 'src/TensorSharp.GGML.Native'), '-B', str(root / 'build'),
            '-G', generator, '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc', '-DCUDAToolkit_ROOT=/usr/local/cuda-12.6',
            '-DCMAKE_CUDA_ARCHITECTURES=86-real', '-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON',
            '-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF', '-DTENSORSHARP_GGML_NATIVE_ENABLE_NCCL=OFF',
            '-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=OFF'], 300)
        run('compile-native', ['cmake', '--build', str(root / 'build'), '--target', 'GgmlOps', '--parallel', '2'], 5400)
        library = root / 'build/libGgmlOps.so'
        if not library.is_file():
            raise FileNotFoundError('Build did not produce libGgmlOps.so.')
        run('elf-dependencies', ['readelf', '-d', str(library)], 30)
        run('exported-symbols', ['nm', '-D', '--defined-only', str(library)], 30)
        symbols = (logs / 'exported-symbols/process.log').read_text(errors='replace')
        for symbol in ('TSGgml_AttachSharedCacheBudgetWithHost', 'TSGgml_DetachSharedCacheBudget'):
            if not any(row.split()[-1:] == [symbol] for row in symbols.splitlines()):
                raise ValueError('Current budget export missing: ' + symbol)
        native_hash = freezer.digest(library)
        for name in ('adaptive-app', 'upload-app'):
            target = root / name / library.name
            if target.exists():
                raise FileExistsError('Refusing to overwrite deployed native: ' + str(target))
            shutil.copyfile(library, target)
            if freezer.digest(target) != native_hash:
                raise ValueError('Native deployment copy hash mismatch.')
        cache = root / 'build/CMakeCache.txt'
        shutil.copyfile(cache, logs / 'CMakeCache.txt')
        report['native'] = dict(path=str(library), sha256=native_hash, bytes=library.stat().st_size)
        report['phase'] = 'completed-build-only'
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        cleanup_mask = signal.pthread_sigmask(signal.SIG_BLOCK, previous_handlers) \
            if previous_handlers and hasattr(signal, 'pthread_sigmask') else None
        try:
            for number in previous_handlers:
                signal.signal(number, signal.SIG_IGN)
        finally:
            if cleanup_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        try:
            if freezer is not None:
                freezer.verify(root)
                report['source_identity_unchanged'] = True
        except BaseException as error:
            report['cleanup_errors'].append(f'Final source identity: {type(error).__name__}: {error}')
        try:
            cache = root / 'build/CMakeCache.txt'
            if cache.is_file():
                shutil.copyfile(cache, logs / 'CMakeCache.txt')
        except BaseException as error:
            report['cleanup_errors'].append(f'Configuration evidence: {type(error).__name__}: {error}')
        report['seconds'] = time.monotonic() - started
        report['passed'] = (report.get('phase') == 'completed-build-only' and not report['errors']
                            and not report['cleanup_errors'] and report['source_identity_unchanged'])
        try:
            (logs / 'execution.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
    print(json.dumps({key: report[key] for key in ('passed', 'phase', 'errors', 'cleanup_errors')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
