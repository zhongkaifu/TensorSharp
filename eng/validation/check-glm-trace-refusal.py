#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""CPU-only actual C ABI observer for GLM diagnostic-byte refusal and reset.

Use a tiny GGUF from GlmGraphBudgetProbe, never a real checkpoint benchmark.
An isolated child makes a native ABI crash a failed, retained test observation.
"""
from __future__ import annotations
import argparse
import ctypes as c
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

AXES = ('hc_attn_pre=1,attn_norm=1,qr=1,attn_out=1,hc_attn_post=2,ffn_norm=1,'
        'router_logits=1,moe_sel=1,moe_probs=1,moe_w=2,moe_up=2,moe_gate=2,moe_down=2,'
        'moe_out=1,ffn_out=1,l_out=2,indexer_k=1,kda_qkv=1,kda_conv=1,kda_gate=2,'
        'kda_scan_out=2,kda_normed=2,kda_out=1')
SCOPE = ('Actual C ABI CPU trace-byte refusal, unchanged publication and slot fence/reset. '
         'No CUDA, model-quality/performance, vendor ownership or physical-hard-limit acceptance.')
MAX_FIXTURE_BYTES = 16 << 20


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    if not 0 < Path(path).stat().st_size <= 16 << 20:
        raise ValueError('Receipt/report exceeds bounded JSON size')
    return json.loads(Path(path).read_text('utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', 'utf-8')


def validate_receipt(receipt, binary_name, binary_sha, source_sha):
    """Bind observations to the frozen owned-native builder's exact closure."""
    if (receipt.get('passed') is not True or receipt.get('no_upstream_compilation') is not True
            or not isinstance(receipt.get('before'), dict) or not receipt['before']
            or receipt['before'] != receipt.get('after')):
        raise ValueError('A successful unchanged-upstream owned-native build receipt is required')
    upstream = receipt['before'].get('upstream', {})
    if not isinstance(upstream, dict) or upstream.get('status') != '' or not isinstance(upstream.get('revision'), str) \
            or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', upstream['revision']):
        raise ValueError('Receipt must identify an unchanged clean upstream revision')
    if receipt.get('production_artifacts', {}).get(binary_name) != binary_sha:
        raise ValueError('Production library hash does not match its frozen build receipt')
    if receipt.get('source_snapshot', {}).get('TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp') != source_sha:
        raise ValueError('GLM implementation hash does not match its frozen build receipt')
    return upstream['revision']


def exercise(native, fixture, capture, *, context=128, prefill_rows=4, threads=2, progress=None, mode='byte-budget'):
    """Observe native state; fake implementations only test the observer itself."""
    report = {'Passed': False, 'Checks': [], 'Scope': SCOPE, 'Mode': mode}
    handle = None
    def check(name, value, **detail):
        report['Checks'].append({'Name': name, 'Passed': bool(value), **detail})
        if progress:
            progress(report)
        if not value:
            raise AssertionError(name)
    def bind(name, arguments, returns):
        fn = getattr(native, name)
        fn.argtypes = arguments
        fn.restype = returns
        return fn
    try:
        pointer, integer, i32 = c.c_void_p, c.c_int, c.c_int32
        pi32, pf32 = c.POINTER(i32), c.POINTER(c.c_float)
        model_load = bind('TSGgml_GlmLoadModel', [c.c_char_p, integer, integer, integer, integer,
                         integer, c.c_char_p, integer, integer, integer], pointer)
        free = bind('TSGgml_GlmFree', [pointer], None)
        forward = bind('TSGgml_GlmForward', [pointer, pi32, integer, pf32], integer)
        joint = bind('TSGgml_GlmForwardBatchedDecode', [pointer, integer, pi32, pi32, pi32, pf32], integer)
        npast = bind('TSGgml_GlmNPast', [pointer], integer)
        rewind = bind('TSGgml_GlmRewind', [pointer, integer], integer)
        reset = bind('TSGgml_GlmResetChecked', [pointer], integer)
        allocate = bind('TSGgml_GlmSlotAlloc', [pointer], integer)
        select = bind('TSGgml_GlmSetActiveSlot', [pointer, integer], integer)
        slot_free = bind('TSGgml_GlmSlotFree', [pointer, integer], integer)
        vocab = bind('TSGgml_GlmVocabSize', [pointer], integer)
        error = bind('TSGgml_GetLastError', [], c.c_char_p)
        backend_failed = bind('TSGgml_HasBackendFailure', [], integer)
        needs_reset = bind('TSGgml_GlmLastBatchedFailureNeedsReset', [pointer], integer)
        last_joint = bind('TSGgml_GlmLastBatchedExecution', [pointer, c.POINTER(c.c_uint64),
                          c.POINTER(integer), c.POINTER(integer), c.POINTER(integer), c.POINTER(integer),
                          pi32, pi32, pi32, integer], integer)
        def last_error():
            return (error() or b'').decode('utf-8', 'replace')
        def logits(count):
            values = (c.c_float * count)(*[731.25] * count)
            return values, c.string_at(c.addressof(values), c.sizeof(values))
        def unchanged(values, original):
            return c.string_at(c.addressof(values), c.sizeof(values)) == original
        def refusal(result, expected='byte bound'):
            text = last_error()
            return result == 0 and 'GLM trace capture failed before publication' in text and expected in text, text

        check('No initial global backend failure', backend_failed() == 0)
        check('Null checked reset is refused', reset(None) == 0)
        handle = model_load(str(fixture).encode('utf-8'), 1, context, prefill_rows, threads, 0, b'CPU', 1, 1, 0)
        check('Tiny CPU fixture loaded', bool(handle), NativeError=last_error())
        count = vocab(handle)
        check('Fixture vocabulary fits bounded probe', 44 <= count <= 65536)
        check('Initial head and reset', npast(handle) == 0 and rewind(handle, 0) == 1)
        partial_before = None
        if mode == 'partial-write':
            # The native load created this fresh directory. Block exactly the
            # second temporary binary after the first tensor was committed.
            blocker = capture / 'pass-000000.tensor-0001.bin.tmp'
            blocker.mkdir(exist_ok=False)
        prompt = (i32 * prefill_rows)(*[3 + (i * 11) % (count - 3) for i in range(prefill_rows)])
        for attempt in range(2):
            values, original = logits(count)
            partial_first = mode == 'partial-write' and attempt == 0
            passed, text = refusal(forward(handle, prompt, prefill_rows, values), 'output exists' if partial_first else 'byte bound')
            check(f'N1 refusal returns normally {attempt}', passed, NativeError=text)
            check(f'N1 publication unchanged {attempt}', unchanged(values, original) and npast(handle) == 0)
            check(f'N1 slot fenced {attempt}', rewind(handle, 0) == 0)
            check(f'N1 checked reset clears fence {attempt}', reset(handle) == 1 and npast(handle) == 0 and rewind(handle, 0) == 1)
            check(f'N1 global backend remains usable {attempt}', backend_failed() == 0)
            if partial_first:
                check('Partial pass retains its first binary and incomplete marker',
                      (capture / 'pass-000000.tensor-0000.bin').is_file()
                      and (capture / 'pass-000000.tensor-0000.bin').stat().st_size > 0
                      and (capture / 'pass-000000.incomplete').is_file()
                      and not list(capture.glob('pass-*.json')))
                blocker.rmdir()  # Owned empty blocker only; never recursive.
                partial_before = {path.name: {'Bytes': path.stat().st_size, 'Sha256': digest(path)}
                                  for path in capture.iterdir() if path.is_file()}
                report['RetainedPartialBeforeRetry'] = partial_before

        slots = [allocate(handle) for _ in range(3)]
        check('Three independent slots allocated', all(slot >= 0 for slot in slots) and len(set(slots)) == 3, Slots=slots)
        ids, tokens, positions = (i32 * 3)(*slots), (i32 * 3)(17, 31, 43), (i32 * 3)(0, 0, 0)
        for attempt in range(2):
            values, original = logits(count * 3)
            passed, text = refusal(joint(handle, 3, ids, tokens, positions, values))
            check(f'N3 refusal returns normally {attempt}', passed, NativeError=text)
            check(f'N3 complete logits sentinel unchanged {attempt}', unchanged(values, original))
            check(f'N3 requires reset {attempt}', needs_reset(handle) == 1)
            execution_id = c.c_uint64()
            rows, nodes, splits, reused = integer(), integer(), integer(), integer()
            out_slots, before, after = (i32 * 3)(), (i32 * 3)(), (i32 * 3)()
            check(f'N3 no completed publication {attempt}', last_joint(handle, c.byref(execution_id), c.byref(rows),
                  c.byref(nodes), c.byref(splits), c.byref(reused), out_slots, before, after, 3) == 0)
            for slot in slots:
                check(f'N3 head and fence {attempt} slot{slot}', select(handle, slot) == 1 and npast(handle) == 0 and rewind(handle, 0) == 0)
                check(f'N3 checked reset {attempt} slot{slot}', reset(handle) == 1 and npast(handle) == 0 and rewind(handle, 0) == 1)
            check(f'N3 global backend remains usable {attempt}', backend_failed() == 0)
        for slot in slots:
            check(f'Independent slot released {slot}', slot_free(handle, slot) == 1)
        if mode == 'partial-write':
            partial_after = {path.name: {'Bytes': path.stat().st_size, 'Sha256': digest(path)}
                             for path in capture.iterdir() if path.is_file()}
            report['RetainedPartialAfterRetries'] = partial_after
            check('Retained partial pass remains charged across N1/N3 retries', partial_before == partial_after
                  and not list(capture.glob('pass-000001*')) and not list(capture.glob('pass-*.json'))
                  and all(path.is_file() for path in capture.iterdir()))
        else:
            check('Preflight refusal emits no fake capture files', capture.is_dir() and not list(capture.iterdir()))
        free(handle)
        handle = None
        check('Model teardown returns normally', backend_failed() == 0)
        report['Passed'] = all(row['Passed'] for row in report['Checks'])
    except BaseException as error:
        report['Error'] = repr(error)
        report['Traceback'] = traceback.format_exc()
    finally:
        if handle:
            try:
                free(handle)
                report['FreedAfterFailure'] = True
            except BaseException as error:
                report['CleanupError'] = repr(error)
        if progress:
            progress(report)
    return report


def environment(arguments):
    result = {k: v for k, v in os.environ.items()
              if not k.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'CUDA_', 'OMP_', 'GOMP_'))}
    result.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS=str(arguments.threads),
                  TS_CPU_MOE_THREADS=str(arguments.threads), TS_GGML_CPU_THREADS=str(arguments.threads),
                  TS_GLM_TRACE='0', TS_GLM_TRACE_DUMP_DIR=str(arguments.output / 'capture'),
                  TS_GLM_TRACE_DUMP_ROWS=str(arguments.prefill_rows),
                  TS_GLM_TRACE_DUMP_MAX_PASSES='2' if arguments.mode == 'partial-write' else '1',
                  TS_GLM_TRACE_DUMP_MAX_BYTES=str(arguments.byte_budget), TS_GLM_TRACE_DUMP_TOKEN_AXES=AXES)
    if arguments.library_dir:
        key = 'PATH' if os.name == 'nt' else 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
        result[key] = os.pathsep.join(map(str, arguments.library_dir)) + os.pathsep + result.get(key, '')
    return result


def reference_budget(path, rows):
    reference = load(path)
    traces = reference.get('traces')
    total = reference.get('total_bytes')
    if (reference.get('schema') != 'tensorsharp.glm-trace.v1' or reference.get('kind') != 'trunk'
            or reference.get('p0') != 0 or reference.get('nt') != rows
            or type(total) is not int or not 0 < total <= 1 << 30
            or not isinstance(traces, list) or len(traces) < 2
            or any(not isinstance(trace, dict) or type(trace.get('bytes')) is not int or trace['bytes'] <= 0 for trace in traces)
            or sum(trace['bytes'] for trace in traces) != total):
        raise ValueError('Partial-write mode needs a matching completed CPU N1 p0=0 trace manifest and exact positive total bytes')
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--fixture-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-source', type=Path)
    parser.add_argument('--library-dir', type=Path, action='append', default=[])
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--prefill-rows', type=int, default=4)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--deadline-seconds', type=int, default=60)
    parser.add_argument('--mode', choices=('byte-budget', 'partial-write'), default='byte-budget')
    parser.add_argument('--reference-manifest', type=Path,
                        help='Matching completed tiny CPU first-pass manifest; required only for partial-write mode')
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    for name in ('native', 'build_receipt', 'fixture_model', 'output'):
        setattr(arguments, name, getattr(arguments, name).resolve())
    arguments.library_dir = [path.resolve(strict=True) for path in arguments.library_dir]
    if not (16 <= arguments.context <= 4096 and 1 <= arguments.prefill_rows <= min(arguments.context, 1024)
            and 1 <= arguments.threads <= 64 and 1 <= arguments.deadline_seconds <= 120):
        parser.error('Require context16..4096, prefill1..min(context,1024), threads1..64, deadline1..120')
    if arguments._child:
        # Keep Windows DLL directories alive throughout borrowed native owners.
        directories = [os.add_dll_directory(str(path)) for path in [arguments.native.parent, *arguments.library_dir]] if os.name == 'nt' else []
        native = c.CDLL(str(arguments.native))
        report = exercise(native, arguments.fixture_model, arguments.output / 'capture', context=arguments.context,
                          prefill_rows=arguments.prefill_rows, threads=arguments.threads,
                          progress=lambda report: save(arguments.output / 'child-report.json', report), mode=arguments.mode)
        print(json.dumps({'Passed': report['Passed'], 'Checks': len(report['Checks']), 'Error': report.get('Error')}), flush=True)
        return 0 if report['Passed'] else 1

    if not 0 < arguments.fixture_model.stat().st_size <= MAX_FIXTURE_BYTES:
        parser.error('Use a tiny fixture model at most16MiB; real checkpoints are outside this test')
    with arguments.fixture_model.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            parser.error('Fixture input must be a GGUF file')
    source = (arguments.native_source or arguments.build_receipt.parent / 'source/TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp').resolve(strict=True)
    if (arguments.mode == 'partial-write') != (arguments.reference_manifest is not None):
        parser.error('Specify --reference-manifest exactly when --mode partial-write is selected')
    arguments.byte_budget = 1
    if arguments.reference_manifest:
        arguments.reference_manifest = arguments.reference_manifest.resolve(strict=True)
        arguments.byte_budget = reference_budget(arguments.reference_manifest, arguments.prefill_rows)
    receipt = load(arguments.build_receipt)
    revision = validate_receipt(receipt, arguments.native.name, digest(arguments.native), digest(source))
    paths = [arguments.native, arguments.build_receipt, arguments.fixture_model, source, Path(__file__).resolve()]
    if arguments.reference_manifest:
        paths.append(arguments.reference_manifest)
    def pins():
        return {str(path): {'Bytes': path.stat().st_size, 'Sha256': digest(path)} for path in paths}
    arguments.output.mkdir(exist_ok=False)
    report = {'Passed': False, 'Scope': SCOPE, 'Mode': arguments.mode, 'ByteBudget': arguments.byte_budget,
              'Before': pins(), 'UpstreamRevision': revision,
              'StartedUtc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    summary = arguments.output / 'summary.json'
    save(summary, report)
    command = [sys.executable, str(Path(__file__).resolve()), '--_child', '--native', str(arguments.native),
               '--build-receipt', str(arguments.build_receipt), '--fixture-model', str(arguments.fixture_model),
               '--output', str(arguments.output), '--context', str(arguments.context), '--prefill-rows', str(arguments.prefill_rows),
               '--threads', str(arguments.threads), '--deadline-seconds', str(arguments.deadline_seconds)]
    command += ['--mode', arguments.mode]
    for path in arguments.library_dir:
        command += ['--library-dir', str(path)]
    env = environment(arguments)
    report.update(Command=command, Environment={k: v for k, v in env.items() if k.startswith(('TS_', 'CUDA_', 'OMP_'))})
    try:
        with (arguments.output / 'console.log').open('xb') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            report['Pid'] = process.pid
            try:
                code = process.wait(timeout=arguments.deadline_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
                raise
        report.update(ExitCode=code, OwnedProcessGone=process.poll() is not None, After=pins(),
                      ConsoleSha256=digest(arguments.output / 'console.log'))
        report['InputsUnchanged'] = report['Before'] == report['After']
        if (arguments.output / 'child-report.json').exists():
            report['Child'] = load(arguments.output / 'child-report.json')
        report['Passed'] = code == 0 and report['InputsUnchanged'] and report.get('Child', {}).get('Passed') is True
    except BaseException as error:
        report.update(Error=repr(error), Traceback=traceback.format_exc())
    finally:
        report['FinishedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(summary, report)
    print(json.dumps({'Passed': report['Passed'], 'ExitCode': report.get('ExitCode'),
                      'Checks': len(report.get('Child', {}).get('Checks', [])), 'Error': report.get('Error')}), flush=True)
    return 0 if report['Passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
