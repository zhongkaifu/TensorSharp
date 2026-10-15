#!/usr/bin/env python3
"""Build/run bounded HostMoE component EXEs from a completed pinned native build.

Stages are explicit and serial. This never qualifies a production GgmlOps DLL:
the cache EXEs contain their own test-hook cache.cpp plus imported ggml objects.
Use a separate actual-DLL graph probe for that qualification. No model download,
real model weights, source mutation, native deployment, or upstream build occurs.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


TARGETS = ('GgmlOpsHostMoeGroupQuotaTest', 'GgmlOpsHostMoeCacheTimingTest',
           'GgmlOpsHostMoeSplitLifecycleTest', 'GgmlOpsHostMoeExpertCacheTest',
           'GgmlOpsHostMoePhaseQuotaTest', 'GgmlOpsHostMoeFileReadPlanTest',
           'GgmlOpsHostMoeDirectReadPlanTest', 'GgmlOpsExpertFileSourceDirectTest')
LIBRARIES = ('ggml.lib', 'ggml-base.lib', 'ggml-cpu.lib', 'ggml-cuda.lib')

# Fixed synthetic fixture geometry, not model tokens/vocabulary logits. These
# include only successful oracle-compared rows, never injected failures.
PHASE_ROWS = {0: 13861, 1: 19241}
PHASE_COLUMNS = 256


def analyze_component_output(case, text):
    """Require actual fixture completion; exit zero alone cannot supply coverage."""
    result = {'passed': False, 'errors': []}
    if case['marker'] not in text:
        result['errors'].append('Required completion marker missing')
    if re.search(r'(?m)^\s*(?:FAIL|SKIP)\b', text):
        result['errors'].append('Failure or unavailable-device marker present')
    if case['name'] == 'pure-phase-quotas':
        counts = re.findall(r'^PASS ([1-9][0-9]*) pure phase-plan checks; real shared tickets and CUDA remain separate qualification$', text, re.M)
        if len(counts) != 1:
            result['errors'].append('Require one complete pure phase-plan result')
        else:
            result['pure_checks'] = int(counts[0])
    if case['name'] == 'pure-file-read-plan':
        counts = re.findall(r'^PASS ([1-9][0-9]*) pure selected expert read checks; exact source bytes, bounded runs/windows, scattered slots and pre-I/O rejection; no CUDA or model loaded$', text, re.M)
        if len(counts) != 1:
            result['errors'].append('Require one complete positive pure selected-read result')
        else:
            result['pure_checks'] = int(counts[0])
    if case['name'] == 'pure-direct-read-plan':
        rows = re.findall(r'^PASS: ([1-9][0-9]*) bounded aligned direct-read planner checks; direct=([1-9][0-9]*) buffered=([1-9][0-9]*), exact logical bytes, padding/window/extent/guard/invalid-complete-plan proof$', text, re.M)
        if len(rows) != 1:
            result['errors'].append('Require one complete positive direct/fallback planner result')
        else:
            result.update(pure_checks=int(rows[0][0]), direct_pieces=int(rows[0][1]), buffered_pieces=int(rows[0][2]))
    if case['name'] == 'actual-direct-file-source':
        rows = re.findall(r'^PASS: ([1-9][0-9]*) actual direct FileSource checks; default-off, padded/exact logical bytes, unsupported/unaligned fallback, extent/identity/close proof; (Windows immutable shared source \(real truncate not available\)|POSIX actual short read, path replacement, CLOEXEC and descriptor close)$', text, re.M)
        if len(rows) != 1:
            result['errors'].append('Require actual direct FileSource completion; unsupported filesystem is not a pass')
        else:
            result.update(file_source_checks=int(rows[0][0]), platform_scope=rows[0][1])
    if case['name'].startswith('scalar-buffered-under-direct-'):
        rows = re.findall(r'^PASS: scalar buffered under direct opt-in completed_direct_reads=(0); fixed32MiB/min-1, poisoned mmap, exact output, warm reuse, interrupted ring and teardown$', text, re.M)
        if len(rows) != 1 or 'PASS: GLM scalar top8 selected payload=' not in text:
            result['errors'].append('Require zero direct reads and complete scalar buffered fixed-window/fault lifecycle result')
        else:
            result['completed_direct_reads'] = int(rows[0])
            result['scope'] = 'Experimental direct opt-in must preserve buffered scalar source path; not model throughput or physical SSD bytes'
    if case['name'].startswith('short-buffered-under-direct-'):
        rows = re.findall(r'^PASS: short-prefill buffered under direct opt-in completed_direct_reads=(0); tails1\.\.8, exact CUDA output, min-1/fault/warm/teardown proof$', text, re.M)
        if len(rows) != 1 or text.count('PASS: GLM actualdecode1 tails1..8,') != 2:
            result['errors'].append('Require zero direct short reads and both complete compact tail/clamp lifecycle results')
        else:
            result['completed_direct_reads'] = 0
    if case['name'] in ('phase-three-groups', 'phase-three-groups-direct'):
        pattern = (r'^PASS: actual phase coexist=([01]) minimum=([0-9]+) budget=([0-9]+) all-wide=([0-9]+); '
                   r'([0-9]+) complete finite CUDA operator floats byte-exact, PP128/256/384/512/1024/N3/group switch, '
                   r'min-1 nonmutation, reshape faults, read interruption, trim/resume/drop/end\. '
                   r'Not whole-model logits or asynchronous cancellation qualification\.$')
        rows = re.findall(pattern, text, re.M)
        modes = [int(row[0]) for row in rows]
        if sorted(modes) != [0, 1]:
            result['errors'].append('Require exactly one pressure and one coexistence completion')
        result['phase_modes'] = []
        for raw in rows:
            mode, minimum, budget, wide, elements = map(int, raw)
            expected_rows = PHASE_ROWS[mode]
            valid = (minimum > 0 and wide > minimum and budget >= minimum
                     and budget == (minimum if mode == 0 else wide) + 17
                     and (budget < wide if mode == 0 else budget >= wide)
                     and elements == expected_rows * PHASE_COLUMNS)
            if not valid:
                result['errors'].append('Phase budget or complete finite-output count differs from fixed fixture')
            result['phase_modes'].append({'coexist': bool(mode), 'minimum_bytes': minimum,
                'budget_bytes': budget, 'all_wide_bytes': wide, 'finite_exact_float_elements': elements,
                'operator_rows': elements // PHASE_COLUMNS if elements % PHASE_COLUMNS == 0 else None,
                'columns_per_operator_row': PHASE_COLUMNS, 'passed': valid})
        result['scope'] = 'Synthetic operator F32 rows compared to resident oracle; not model vocabulary logits, natural EOS or in-flight CUDA cancellation'
        if case['name'] == 'phase-three-groups-direct':
            direct = re.findall(r'^PASS: wide-only direct phase coexist=([01]) completed_direct_reads=([1-9][0-9]*) narrow_buffered_reads=([0-9]+) narrow_direct_reads=(0); complete phase exact output/ownership/fault proof$', text, re.M)
            if sorted(int(row[0]) for row in direct) != [0, 1] or any(row[0] == '0' and int(row[2]) == 0 for row in direct):
                result['errors'].append('Require actual direct wide reads in both modes and cold N1/N3 buffered-only reads under pressure')
            result['direct_phase_modes'] = [{'coexist': bool(int(row[0])), 'completed_direct_reads': int(row[1]),
                'narrow_buffered_reads': int(row[2]), 'narrow_direct_reads': int(row[3])} for row in direct]
    result['passed'] = not result['errors']
    return result


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def cache_value(cache, key):
    rows = [line.split('=', 1)[1] for line in cache.read_text(encoding='utf-8').splitlines()
            if line.startswith(key + ':')]
    if len(rows) != 1:
        raise ValueError('Missing/duplicate CMake cache identity: ' + key)
    return rows[0]


def metadata(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def guard():
    if os.name != 'nt':
        raise ValueError('This runner currently certifies the Windows imported-library build only')
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total', 'available', 'commit_total', 'commit_available',
                                            'virtual_total', 'virtual_available', 'extra')]
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError('GlobalMemoryStatusEx failed')
    sample = {'utc': utc(), 'available_physical': memory.available, 'available_commit': memory.commit_available}
    if memory.available < 2 << 30 or memory.commit_available < 4 << 30:
        raise MemoryError('Available RAM/commit below 2/4 GiB: ' + repr(sample))
    return sample


def device_snapshot():
    command = ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free',
               '--format=csv,noheader,nounits']
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=15)
    if run.returncode or not run.stdout.strip():
        raise RuntimeError('Actual device observation unavailable: ' + run.stdout)
    return {'utc': utc(), 'command': command, 'exit_code': run.returncode, 'raw': run.stdout,
            'scope': 'Whole-board observation; not an allocator peak or process physical ownership'}


def resolve_test_overlays(component, overlay, source, receipt_sha256):
    """A derived driver may be identical. Changed tests require explicit proof."""
    if overlay is None:
        return {}
    if component is None:
        raise ValueError('A test-only overlay requires an isolated component root')
    if overlay.get('schema') != 'tensorsharp.native-component-test-overlay.v1':
        raise ValueError('Unknown component overlay schema')
    if overlay.get('base_build_receipt_sha256') != receipt_sha256:
        raise ValueError('Test overlay is not bound to the completed native build')
    files = overlay.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Require a nonempty explicit test-only overlay')
    for relative, item in files.items():
        path = Path(relative)
        if (path.is_absolute() or '..' in path.parts or path.parts[:2] != ('TensorSharp.GGML.Native', 'tests')
                or path.suffix not in ('.cpp', '.inc', '.h') or relative not in source
                or not isinstance(item, dict) or item.get('original_sha256') != source[relative]
                or not isinstance(item.get('replacement_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', item['replacement_sha256'])
                or item['replacement_sha256'] == source[relative]):
            raise ValueError('Overlay may change only explicitly pinned native test source files: ' + relative)
    return files


class Runner:
    def __init__(self, args):
        self.args = args
        self.build = args.build_root.resolve(strict=True)
        self.output = args.output.resolve()
        self.base_hooks = self.build / 'build-hooks'
        self.component = args.component_root.resolve() if args.component_root else None
        self.hooks = self.component / 'build-hooks' if self.component else self.base_hooks
        self.receipt = self.build / 'build-summary.json'
        self.full = read(self.receipt)
        if self.full.get('passed') is not True or self.full.get('before') != self.full.get('after'):
            raise ValueError('Require completed successful unchanged-source native production/hooks build first')
        self.repo = args.repository.resolve(strict=True)
        self.upstream = self.repo / 'ExternalProjects/ggml'
        self.driver = (self.component or self.build) / 'driver'
        self.libraries = Path(cache_value(self.base_hooks / 'CMakeCache.txt', 'TENSORSHARP_VALIDATION_LIBRARIES'))
        self.source = self.full['source_snapshot']
        if not self.source or self.source != self.full['before']['current_sources']:
            raise ValueError('Native build must pin the complete owned source closure')
        if args.test_overlay and not self.component:
            raise ValueError('A test-only overlay requires an isolated component root')
        self.overlays = resolve_test_overlays(self.component,
            read(args.test_overlay) if args.test_overlay else None, self.source, sha(self.receipt))
        if self.component:
            if self.component == self.build or self.component in self.build.parents:
                raise ValueError('Component root cannot overwrite the completed native build')
        self.environment = {k: v for k, v in os.environ.items()
                            if not k.startswith(('TS_HOST_MOE_', 'TS_GLM_POOLED_'))}
        self.environment.update(CUDA_VISIBLE_DEVICES='0' if args.stage == 'cuda' else '-1',
            OMP_NUM_THREADS='8', TS_CPU_MOE_THREADS='8', TS_GGML_CPU_THREADS='8')
        if args.cuda_bin:
            self.environment['PATH'] = str(args.cuda_bin) + os.pathsep + self.environment.get('PATH', '')

    def prepare_component_driver(self):
        """Derive a fresh test driver without changing either original frozen tree."""
        if not self.component:
            return
        self.component.mkdir(parents=True, exist_ok=False)
        self.driver.mkdir()
        for relative, expected in self.source.items():
            suffix = Path(relative).relative_to('TensorSharp.GGML.Native')
            original = self.build / 'driver' / suffix
            expected_driver = self.full['generated_driver_files'].get('driver/' + suffix.as_posix(), expected)
            if sha(original) != expected_driver:
                raise ValueError('Original frozen driver changed before derivation: ' + str(original))
            destination = self.driver / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
            if relative in self.overlays:
                source = self.repo / relative
                if sha(source) != self.overlays[relative]['replacement_sha256']:
                    raise ValueError('Test overlay replacement changed before derivation: ' + relative)
                shutil.copyfile(source, destination)
        shutil.copyfile(self.build / 'import-upstream.cmake', self.component / 'import-upstream.cmake')

    def pins(self, executables=False):
        result = {'build_receipt_sha256': sha(self.receipt), 'owned_source': {}, 'frozen_source': {},
                  'driver_source': {}, 'upstream': {}, 'imported_libraries': {}, 'native_dlls': {}}
        for relative, expected in self.source.items():
            original = self.repo / relative
            frozen = self.build / 'source' / relative
            driver = self.driver / Path(relative).relative_to('TensorSharp.GGML.Native')
            for key, path in (('owned_source', original), ('frozen_source', frozen)):
                actual = sha(path)
                wanted = (self.overlays.get(relative, {}).get('replacement_sha256', expected)
                          if key == 'owned_source' else expected)
                if actual != wanted:
                    raise ValueError('Source differs from completed native build: ' + str(path))
                result[key][relative] = actual
            actual = sha(driver)
            if driver.name != 'CMakeLists.txt' and actual != self.overlays.get(relative, {}).get('replacement_sha256', expected):
                raise ValueError('Frozen driver source differs: ' + str(driver))
            result['driver_source'][relative] = actual
        for relative, expected in self.full['generated_driver_files'].items():
            if sha(self.build / relative) != expected:
                raise ValueError('Generated driver/import closure changed: ' + relative)
            if self.component and sha(self.component / relative) != expected:
                raise ValueError('Derived driver/import differs from original non-test input: ' + relative)
        if self.overlays:
            result['test_only_overlay'] = {'manifest_sha256': sha(self.args.test_overlay), 'files': self.overlays,
                                         'component_root': str(self.component)}
        elif self.component:
            result['identical_derived_driver'] = {
                'component_root': str(self.component), 'base_build_receipt_sha256': sha(self.receipt),
                'scope': 'Exact copy of every pinned source/driver/import; no source exemption or test overlay'}
        for name, expected in self.full['before']['upstream']['files'].items():
            actual = sha(self.upstream / name)
            if actual != expected:
                raise ValueError('Upstream content changed: ' + name)
            result['upstream'][name] = actual
        result['upstream_revision'] = subprocess.check_output(
            ['git', '-C', str(self.upstream), 'rev-parse', 'HEAD'], text=True).strip()
        result['upstream_status'] = subprocess.check_output(
            ['git', '-C', str(self.upstream), 'status', '--porcelain'], text=True).strip()
        if result['upstream_revision'] != self.full['before']['upstream']['revision'] or result['upstream_status']:
            raise ValueError('Upstream revision/clean status differs from build')
        for name in LIBRARIES:
            actual = sha(self.libraries / name)
            if actual != self.full['before']['libraries'][name]:
                raise ValueError('Imported ggml library changed: ' + name)
            result['imported_libraries'][name] = actual
        for mode in ('production', 'hooks'):
            path = self.build / ('build-' + mode) / 'GgmlOps.dll'
            actual = sha(path)
            if actual != self.full[mode + '_artifacts']['GgmlOps.dll']:
                raise ValueError('Native DLL differs from final build receipt: ' + mode)
            result['native_dlls'][str(path)] = actual
        if executables:
            result['executables'] = {name: sha(self.hooks / (name + '.exe')) for name in TARGETS}
        return result

    def child(self, label, command, deadline, environment):
        directory = self.output / self.args.stage / label
        directory.mkdir(parents=True, exist_ok=False)
        record = {'command': command, 'started_utc': utc(), 'deadline_seconds': deadline,
                  'complete': False, 'timed_out': False, 'pid': None, 'exit_code': None,
                  'process_exited': False, 'resource_samples': [guard()]}
        log = directory / 'process.log'
        def save():
            write(directory / 'execution.json', record)
        save()
        started = time.monotonic(); process = None
        try:
            with log.open('wb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                record['pid'] = process.pid; save()
                while True:
                    try:
                        record['exit_code'] = process.wait(timeout=15)
                        record['complete'] = True; break
                    except subprocess.TimeoutExpired:
                        record['resource_samples'].append(guard())
                        if time.monotonic() - started > deadline:
                            record['timed_out'] = True
                            raise TimeoutError(label + ' exceeded deadline')
                        save()
                record['process_exited'] = process.poll() is not None
        except BaseException as error:
            record['error'] = repr(error)
            if process is not None and process.poll() is None:
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
                record['exit_code'] = process.wait(timeout=20)
                record['process_exited'] = True
            raise
        finally:
            record['finished_utc'] = utc(); record['seconds'] = time.monotonic() - started
            record['log_sha256'] = sha(log) if log.is_file() else None
            save()
        return record, log

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        stage_path = self.output / self.args.stage
        if stage_path.exists():
            raise ValueError('Fresh stage required; preserve previous failures/skips rather than overwrite')
        # Parent's completed full-build receipt is mandatory before this lock.
        # It cannot lock unrelated Ninja invocations; root must grant one owner.
        lock = self.base_hooks / '.host-moe-component-validation.lock'
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, str(os.getpid()).encode()); os.close(handle)
        report = {'schema': 'tensorsharp.host-moe-components.v1', 'stage': self.args.stage,
                  'started_utc': utc(), 'passed': False, 'skipped': 0, 'cases': [],
                  'scope': 'Independent statically linked synthetic component EXEs; no production/hooks DLL execution, real model, throughput, default-readiness or complete physical hard-limit qualification.',
                  'tool_sha256': sha(Path(__file__)), 'environment': {k: self.environment[k] for k in
                      ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'TS_CPU_MOE_THREADS', 'TS_GGML_CPU_THREADS')}}
        summary = self.output / (self.args.stage + '-summary.json')
        try:
            stage_path.mkdir(exist_ok=False)
            if self.args.stage == 'build':
                self.prepare_component_driver()
            report['before'] = self.pins(executables=self.args.stage != 'build')
            if self.args.stage == 'build':
                if self.component:
                    command = [str(self.args.cmake), '-S', str(self.driver), '-B', str(self.hooks), '-G', 'Ninja',
                        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                        '-DCMAKE_MAKE_PROGRAM=' + cache_value(self.base_hooks / 'CMakeCache.txt', 'CMAKE_MAKE_PROGRAM'),
                        '-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON', '-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON',
                        '-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF', '-DCMAKE_CUDA_ARCHITECTURES=86-real',
                        '-DCMAKE_CUDA_FLAGS=' + cache_value(self.base_hooks / 'CMakeCache.txt', 'CMAKE_CUDA_FLAGS'),
                        '-DTENSORSHARP_VALIDATION_UPSTREAM=' + self.upstream.as_posix(),
                        '-DTENSORSHARP_VALIDATION_LIBRARIES=' + self.libraries.as_posix()]
                    record, log = self.child('configure-derived-tests', command, self.args.build_timeout, self.environment)
                    report['cases'].append(record)
                    if record['exit_code']:
                        raise RuntimeError('Derived test driver configure failed')
                command = [str(self.args.cmake), '--build', str(self.hooks), '--target', *TARGETS, '--parallel', '2']
                record, log = self.child('cmake-component-targets', command, self.args.build_timeout, self.environment)
                report['cases'].append(record)
                if record['exit_code']:
                    raise RuntimeError('Component build failed; no tests authorized from an incomplete build')
                report['after'] = self.pins(executables=True)
                if {k: v for k, v in report['after'].items() if k != 'executables'} != report['before']:
                    raise ValueError('Build changed pinned source/import/DLL inputs')
                report['passed'] = True
            else:
                built = read(self.output / 'build-summary.json')
                if built.get('passed') is not True or report['before'] != built['after']:
                    raise ValueError('Component execution requires unchanged completed component build pins')
                if self.args.stage == 'cuda':
                    cpu = read(self.output / 'cpu-summary.json')
                    if cpu.get('passed') is not True:
                        raise ValueError('CPU component stage must pass before CUDA')
                    report['device_before'] = device_snapshot()
                for case in cases(self.args.stage):
                    executable = self.hooks / (case['target'] + '.exe')
                    actual = sha(executable)
                    if actual != report['before']['executables'][case['target']]:
                        raise ValueError('Component EXE changed before case: ' + case['name'])
                    before_stat = metadata(executable)
                    env = self.environment.copy()
                    env['TEMP'] = env['TMP'] = str(self.output / 'tmp')
                    Path(env['TEMP']).mkdir(exist_ok=True)
                    record, log = self.child(case['name'], [str(executable), *case['args']], self.args.deadline, env)
                    text = log.read_text(encoding='utf-8', errors='replace')
                    record.update(name=case['name'], exe_sha256_before=actual, exe_sha256_after=sha(executable),
                        exe_metadata_before=before_stat, exe_metadata_after=metadata(executable),
                        zero_owner_contract=case['zero_owner_contract'],
                        owner_observation='Internal fixture assertions; not an external allocator sample or inferred from process exit.')
                    record['skipped'] = record['exit_code'] == 77
                    record['fixture_evidence'] = analyze_component_output(case, text)
                    record['passed'] = (record['exit_code'] == 0 and record['complete'] is True
                        and record['process_exited'] is True and not record['timed_out']
                        and record['fixture_evidence']['passed']
                        and record['exe_sha256_before'] == record['exe_sha256_after'])
                    report['cases'].append(record); report['skipped'] += int(record['skipped'])
                    write(summary, report)
                    print(json.dumps({'stage': self.args.stage, 'name': case['name'], 'pid': record['pid'],
                        'passed': record['passed'], 'skipped': record['skipped'], 'seconds': record['seconds']}), flush=True)
                    if not record['passed'] and not record['skipped']:
                        raise RuntimeError('Actual component case failed: ' + case['name'])
                report['after'] = self.pins(executables=True)
                if report['before'] != report['after']:
                    raise ValueError('Pinned inputs changed during component execution')
                if self.args.stage == 'cuda':
                    report['device_after'] = device_snapshot()
                report['passed'] = not report['skipped'] and all(x['passed'] for x in report['cases'])
        except BaseException as error:
            report['error'] = repr(error); report['traceback'] = traceback.format_exc()
            try:
                report['after_failure'] = self.pins(executables=self.args.stage != 'build')
            except BaseException as pin_error:
                report['pin_error'] = repr(pin_error)
        finally:
            report['finished_utc'] = utc()
            write(summary, report)
            lock.unlink()  # Only the lock created by this owned invocation.
        print(json.dumps({'passed': report['passed'], 'skipped': report['skipped'], 'summary': str(summary)}), flush=True)
        return 0 if report['passed'] else 77 if report['skipped'] and 'error' not in report else 1


def cases(stage):
    def case(name, target, args, marker, zero='Fixture asserts pending/committed/cache physical cleanup before successful exit'):
        return {'name': name, 'target': target, 'args': args, 'marker': marker, 'zero_owner_contract': zero}
    cache = 'GgmlOpsHostMoeExpertCacheTest'; split = 'GgmlOpsHostMoeSplitLifecycleTest'
    if stage == 'cpu':
        return [case('pure-group-quotas', TARGETS[0], [], 'pure measured-group quota checks', 'No backing allocation/backend in this pure metadata test'),
                case('pure-cache-timing', TARGETS[1], [], 'host-only cache timing checks', 'No backing allocation/backend in this pure clock policy test'),
                case('pure-phase-quotas', TARGETS[4], [], 'pure phase-plan checks', 'No backing allocation/backend in this pure metadata test'),
                case('pure-file-read-plan', TARGETS[5], [], 'pure selected expert read checks', 'Exact selected source/slot bytes in pure metadata buffers; no file/device/model'),
                case('pure-direct-read-plan', TARGETS[6], [], 'bounded aligned direct-read planner checks', 'Exact logical bytes, padding charged inside bounded caller buffers; no actual file/device/model'),
                case('actual-direct-file-source', TARGETS[7], [], 'actual direct FileSource checks', 'Actual OS direct handles/fallback/identity/descriptor cleanup; no ggml/device/model'),
                case('cpu-fallback', cache, ['--cpu'], 'PASS: disabled/non-CUDA'),
                case('cpu-scope', cache, ['--cpu', '--scope'], 'PASS: disabled/non-CUDA'),
                *[case(name, cache, ['--invalid-budget', value], 'PASS: malformed native cache budget remains disabled') for name, value in
                  [('invalid-sign', '+8192'), ('invalid-leading-space', ' 8192'), ('invalid-trailing-space', '8192 '),
                   ('invalid-overflow', '18446744073709551616')]]]
    result = [case('measured-three-groups', split, ['--group-quotas'], 'PASS: actual 3-group measured minima='),
              case('phase-three-groups', split, ['--phase-quotas'], 'PASS: actual phase coexist=')]
    for name, args in [('q8', []), ('iq1', ['--quantization', 'iq1_m']), ('iq2', ['--quantization', 'iq2_xxs'])]:
        result.append(case('split-' + name, split, ['--split-lifecycle', *args], 'PASS: Native18 split-owner'))
    for name, args in [('q8', []), ('iq1', ['--quantization', 'iq1_m']), ('iq2', ['--quantization', 'iq2_xxs'])]:
        result.append(case('compact-' + name, cache, ['--glm-compact-prefill', *args], 'PASS:'))
    legacy = [('ordinary-q8', []), ('arena-q8', ['--arena']), ('arena-iq1', ['--arena', '--quantization', 'iq1_m']),
              ('arena-iq2', ['--arena', '--quantization', 'iq2_xxs']), ('arena-file-q8', ['--arena', '--file-read']),
              ('arena-file-iq1', ['--arena', '--file-read', '--quantization', 'iq1_m']),
              ('capacity-growth-q8', ['--capacity-growth']), ('capacity-growth-iq1', ['--capacity-growth', '--quantization', 'iq1_m']),
              ('capacity-growth-iq2', ['--capacity-growth', '--quantization', 'iq2_xxs']),
              ('bulk-union-q8', ['--bulk-union']), ('bulk-union-iq1', ['--bulk-union', '--quantization', 'iq1_m']),
              ('bulk-union-iq2', ['--bulk-union', '--quantization', 'iq2_xxs']),
              ('scalar-large-file-q8', ['--glm-scalar-file-window']),
              ('scalar-large-file-iq1', ['--glm-scalar-file-window', '--quantization', 'iq1_m']),
              ('pooled-largest-quota-q8', ['--pooled-quota']),
              ('pooled-largest-quota-iq1', ['--pooled-quota', '--quantization', 'iq1_m'])]
    return result + [case(name, cache, args, 'PASS:') for name, args in legacy] + [
        case('scalar-buffered-under-direct-q8', cache, ['--glm-scalar-file-window', '--direct-reads'], 'PASS: scalar buffered under direct opt-in'),
        case('scalar-buffered-under-direct-iq1', cache, ['--glm-scalar-file-window', '--direct-reads', '--quantization', 'iq1_m'], 'PASS: scalar buffered under direct opt-in'),
        case('short-buffered-under-direct-q8', cache, ['--glm-compact-prefill', '--direct-reads'], 'PASS: short-prefill buffered under direct opt-in'),
        case('short-buffered-under-direct-iq1', cache, ['--glm-compact-prefill', '--direct-reads', '--quantization', 'iq1_m'], 'PASS: short-prefill buffered under direct opt-in'),
        case('phase-three-groups-direct', split, ['--phase-quotas', '--direct-reads'], 'PASS: wide-only direct phase coexist=')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('build', 'cpu', 'cuda'))
    parser.add_argument('--build-root', required=True, type=Path)
    parser.add_argument('--component-root', type=Path, help='Fresh isolated component driver/build root; exact pinned copy unless a test overlay is explicitly supplied')
    parser.add_argument('--test-overlay', type=Path, help='Explicit original/replacement test-source hashes bound to the native build receipt')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--cmake', type=Path, default=Path(shutil.which('cmake') or 'cmake'))
    parser.add_argument('--cuda-bin', type=Path)
    parser.add_argument('--deadline', type=int, default=180)
    parser.add_argument('--build-timeout', type=int, default=1200)
    args = parser.parse_args()
    if args.deadline <= 0 or args.build_timeout <= 0:
        parser.error('Deadlines must be positive')
    return Runner(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
