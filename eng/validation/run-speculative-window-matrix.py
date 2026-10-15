#!/usr/bin/env python3
"""Prepare/run a pinned six-process AgentTurnBench candidate-window matrix.

Each process owns one warmup and two measured plain/speculative pairs. Opposite
orders are separate processes, not four independent ABBA arms. Quality failures
are retained and do not prevent the other independent processes from running;
execution, identity and evidence failures stop the matrix. No build/download.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('speculative_matrix_validator', HERE / 'validate-speculative-rows.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
RUNS = [(1, 'plain-first'), (3, 'spec-first'), (7, 'plain-first'),
        (7, 'spec-first'), (3, 'plain-first'), (1, 'spec-first')]
PHASES = ('DraftMs', 'VerifyMs', 'SnapshotMs', 'RollbackMs', 'CatchUpMs', 'PlainMs', 'PlainMsPerToken', 'SpecMsPerToken')
CLEAR_PREFIXES = ('TS_', 'TENSORSHARP_', 'GGML_', 'OMP_', 'GOMP_', 'NCCL_')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def publish(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def managed_files(directory):
    return {str(path.resolve()): digest(path) for path in sorted(Path(directory).iterdir())
            if path.is_file() and (path.suffix == '.dll' or path.name.endswith(('.deps.json', '.runtimeconfig.json')))}


def checkpoint_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'scope': 'Size/mtime continuity only; no whole-checkpoint hash is computed by this matrix.'}


def child_environment(window):
    env = {key: value for key, value in os.environ.items() if not key.startswith(CLEAR_PREFIXES)
           and key not in ('MAX_CONTEXT', 'KV_CACHE_DTYPE', 'CUDA_VISIBLE_DEVICES')}
    env.update(CUDA_VISIBLE_DEVICES='0', MAX_CONTEXT='2048', OMP_NUM_THREADS='8',
               TS_CPU_MOE_THREADS='8', TS_GGML_CPU_THREADS='8', TS_N_CPU_MOE='8')
    if window != 7:
        env['TS_SPEC_DRAFT'] = str(window)
    return env


def qualify_rows(rows, order, window, baseline_tokens=None, baseline_prompt=None):
    """Qualification is independent of process success and never truncates arrays."""
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError('Expected exactly two scenario rows')
    labels = ['plain greedy', 'draft head (auto)']
    if order == 'spec-first':
        labels.reverse()
    if [row.get('Label') for row in rows] != labels:
        raise ValueError('Actual execution order differs from the requested order')
    if any(row.get('Scenario') != 'spec' for row in rows):
        raise ValueError('Unexpected scenario')
    validation = validator.validate(rows)
    plain = next(row for row in rows if row['Label'] == 'plain greedy')
    learned = next(row for row in rows if row['Label'] == 'draft head (auto)')
    phases = learned.get('SpeculationTimings')
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError('Missing exact speculation timing fields')
    for value in phases.values():
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid speculative timing')
    for row in rows:
        for field in ('PrefillComputeMs', 'DecodeComputeMs', 'TotalMs', 'DecodeTps'):
            value = row.get(field)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('Missing positive finite ' + field)
    actual_windows = set(map(int, re.findall(r'maxDraft=(\d+)', learned.get('Note', ''))))
    if actual_windows != {window}:
        raise ValueError('Actual armed candidate window differs: ' + repr(actual_windows))
    eos = all(row.get('Finish') == 'eos' for row in rows)
    prompt_equal = type(plain.get('Prompt')) is int and plain['Prompt'] > 0 and plain['Prompt'] == learned.get('Prompt')
    prompt_stable = baseline_prompt is None or plain.get('Prompt') == baseline_prompt
    baseline_equal = baseline_tokens is None or plain['Tokens'] == baseline_tokens
    return {'passed': validation['status'] == 'passed' and eos and prompt_equal and prompt_stable and baseline_equal,
            'validator': validation, 'both_eos': eos, 'prompt_counts_equal': prompt_equal,
            'prompt_matches_first_warmup': prompt_stable, 'prompt_tokens_count': plain.get('Prompt'),
            'plain_matches_first_warmup': baseline_equal, 'actual_armed_window': window,
            'plain_tokens': plain['Tokens'], 'speculative_tokens': learned['Tokens'],
            'plain_text': plain.get('Text'), 'speculative_text': learned.get('Text'),
            'plain_decode_ms': plain['DecodeComputeMs'], 'speculative_decode_ms': learned['DecodeComputeMs'],
            'decode_latency_change_percent': (learned['DecodeComputeMs'] / plain['DecodeComputeMs'] - 1) * 100,
            'plain_prefill_ms': plain['PrefillComputeMs'], 'speculative_prefill_ms': learned['PrefillComputeMs'],
            'plain_total_ms': plain['TotalMs'], 'speculative_total_ms': learned['TotalMs'],
            'plain_decode_tps': plain['DecodeTps'], 'speculative_decode_tps': learned['DecodeTps'],
            'phases': phases, 'speculative_note': learned.get('Note')}


def aggregate(records):
    windows = {}
    for window in (1, 3, 7):
        processes = [row for row in records if row['window'] == window]
        measurements = [pair for row in processes for pair in row.get('checks', []) if not pair['warmup']]
        if not measurements:
            continue
        med = lambda field: statistics.median(pair[field] for pair in measurements)
        eligible = (len(processes) == 2 and {row['order'] for row in processes} == {'plain-first', 'spec-first'}
                    and all(row.get('process_passed') is True and row.get('quality_passed') is True for row in processes)
                    and len(measurements) == 4)
        windows[str(window)] = {'processes': len(processes), 'measured_pairs': len(measurements),
            'all_quality_gates_passed': eligible,
            'plain_decode_ms_median': med('plain_decode_ms'), 'speculative_decode_ms_median': med('speculative_decode_ms'),
            'decode_latency_change_percent': (med('speculative_decode_ms') / med('plain_decode_ms') - 1) * 100,
            'plain_prefill_ms_median': med('plain_prefill_ms'), 'speculative_prefill_ms_median': med('speculative_prefill_ms'),
            'plain_total_ms_median': med('plain_total_ms'), 'speculative_total_ms_median': med('speculative_total_ms'),
            'individual_decode_changes_percent': [pair['decode_latency_change_percent'] for pair in measurements],
            'speculative_phase_medians': {phase: statistics.median(pair['phases'][phase] for pair in measurements) for phase in PHASES},
            'performance_qualification': 'Observed costs only; quality failures invalidate an optimization claim. Two paired processes, not independent ABBA. Telemetry/background interference is unquantified.'}
    return windows


def prepare(args):
    repository = args.repo_root.resolve()
    entry = args.entry.resolve()
    if entry.name != 'AgentTurnBench.dll' or digest(entry) != args.expected_entry_sha256:
        raise ValueError('AgentTurnBench entrypoint does not match the required pin')
    if not entry.with_suffix('.runtimeconfig.json').is_file():
        raise ValueError('Entrypoint runtimeconfig is missing')
    if args.plan.exists() or args.output.exists():
        raise FileExistsError('Plan and matrix output must be fresh paths')
    sources = [HERE / 'run-speculative-window-matrix.py', HERE / 'run-bounded-probe.py', HERE / 'validate-speculative-rows.py',
               HERE / 'multimodel-quality-bench.py', repository / 'benchmarks/AgentTurnBench/Program.cs',
               repository / 'benchmarks/AgentTurnBench/SpeculativeBenchmarkMetrics.cs',
               repository / 'benchmarks/AgentTurnBench/SpecParityDiagnostic.cs']
    plan = {'schema': 1, 'prepared_utc': utc(), 'repository': str(repository), 'entry': str(entry),
            'expected_entry_sha256': args.expected_entry_sha256, 'expected_native_sha256': args.expected_native_sha256,
            'managed_files': managed_files(entry.parent), 'source_files': {str(path.resolve()): digest(path) for path in sources},
            'checkpoints': [checkpoint_stat(args.model), checkpoint_stat(args.draft_model)],
            'output': str(args.output.resolve()), 'timeout_per_process_seconds': 360,
            'warmup_passes': 1, 'measured_passes': 2, 'runs': [],
            'scope': 'Six fresh processes, each one warmup plus two measured plain/spec pairs. Window7 means auto with TS_SPEC_DRAFT absent. CPU8 offload; original evidence directories are untouched.'}
    for index, (window, order) in enumerate(RUNS, 1):
        owned = args.output.resolve() / f'{index:02d}-window{window}-{order}'
        rows = owned / 'rows.json'
        command = ['dotnet', str(entry), '--model', str(args.model.resolve()), '--draft-model', str(args.draft_model.resolve()),
                   '--backend', 'ggml_cuda', '--kv', 'f16', '--chunk', '256', '--max-batched', '256',
                   '--scenarios', 'spec', '--spec-candidates', 'auto', '--spec-only', '--spec-minimal-system',
                   '--spec-file', '80', '--spec-new', '512', '--warmup', '1', '--measure-passes', '2',
                   '--spec-order', order, '--out', str(rows)]
        plan['runs'].append({'index': index, 'window': window, 'order': order, 'owned_output': str(owned),
                             'command': command, 'rows_base': str(rows),
                             'controlled_environment': {k: v for k, v in child_environment(window).items()
                                 if k.startswith(CLEAR_PREFIXES) or k in ('MAX_CONTEXT', 'CUDA_VISIBLE_DEVICES')}})
    publish(args.plan, plan)
    return 0


def verify_pins(plan):
    if managed_files(Path(plan['entry']).parent) != plan['managed_files']:
        raise ValueError('Top-level app DLL/deps/runtimeconfig set or identity changed')
    if {path: digest(path) for path in plan['source_files']} != plan['source_files']:
        raise ValueError('Prepared source/helper identity changed')
    if digest(plan['entry']) != plan['expected_entry_sha256']:
        raise ValueError('Entrypoint identity changed')
    if [checkpoint_stat(row['path']) for row in plan['checkpoints']] != plan['checkpoints']:
        raise ValueError('Checkpoint size/mtime continuity changed')


def execute(args):
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    if [(row['window'], row['order']) for row in plan['runs']] != RUNS:
        raise ValueError('Expected the full six-process counterbalanced matrix')
    verify_pins(plan)
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'started_utc': utc(), 'complete': False, 'passed': False, 'runs': [],
              'plan': str(args.plan.resolve()), 'plan_sha256': digest(args.plan), 'scope': plan['scope'],
              'quality_scope': 'Complete token arrays, EOS finish reason, positive draft/accept/verify engagement, fixed prompt count, cross-pass/process plain consistency; not an independent semantic oracle.',
              'identity_scope': 'Actual native module observed once by owned wrapper, disk hash checked again at exit. Top-level managed files are pinned before/after, not live managed-module enumeration.'}
    summary_path = output / 'matrix.json'
    publish(summary_path, report)
    baseline_tokens = baseline_prompt = None
    try:
        for item in plan['runs']:
            verify_pins(plan)
            owned = Path(item['owned_output'])
            command = [sys.executable, str(HERE / 'run-bounded-probe.py'), '--output', str(owned), '--timeout', str(plan['timeout_per_process_seconds']),
                       '--telemetry', '--managed-directory', str(Path(plan['entry']).parent), '--expected-native-sha256', plan['expected_native_sha256'], '--'] + item['command']
            record = {'index': item['index'], 'window': item['window'], 'order': item['order'], 'started_utc': utc(),
                      'process_passed': False, 'quality_passed': False, 'checks': [], 'wrapper_argv': command}
            report['runs'].append(record)
            publish(summary_path, report)
            print(f"START {item['index']}/6 window{item['window']} {item['order']}", flush=True)
            with (output / f"{item['index']:02d}-wrapper.log").open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=plan['repository'], env=child_environment(item['window']), stdout=log, stderr=subprocess.STDOUT)
            record['wrapper_exit'] = result.returncode
            execution = json.loads((owned / 'execution.json').read_text(encoding='utf-8-sig'))
            record.update(pid=execution.get('pid'), wall_seconds=execution.get('wall_seconds'), telemetry=execution.get('telemetry'), finished_utc=utc())
            telemetry = execution.get('telemetry', {})
            if result.returncode != 0 or execution.get('complete') is not True or execution.get('exit_code') != 0 or telemetry.get('passed') is not True:
                raise ValueError('Process or telemetry qualification failed for run ' + str(item['index']))
            for side in ('before', 'after'):
                if telemetry.get('managed_files_' + side) != plan['managed_files']:
                    raise ValueError('Managed pin mismatch: ' + side)
            metadata = json.loads(Path(item['rows_base'] + '.run.json').read_text(encoding='utf-8-sig'))
            if (metadata.get('ProcessId') != execution['pid'] or metadata.get('SpecOrder') != item['order']
                    or metadata.get('WarmupPasses') != 1 or metadata.get('MeasuredPasses') != 2
                    or metadata.get('Arguments') != item['command'][2:]):
                raise ValueError('Actual benchmark invocation differs from plan')
            record['process_metadata'] = metadata
            verify_pins(plan)
            record['process_passed'] = True
            for suffix in ('.warmup1.json', '.measure1.json', '.measure2.json'):
                path = Path(item['rows_base'] + suffix)
                rows = json.loads(path.read_text(encoding='utf-8-sig'))
                check = qualify_rows(rows, item['order'], item['window'], baseline_tokens, baseline_prompt)
                if baseline_tokens is None:
                    baseline_tokens = check['plain_tokens']
                    baseline_prompt = check['prompt_tokens_count']
                    report['plain_baseline_tokens'] = baseline_tokens
                    report['plain_baseline_prompt_tokens_count'] = baseline_prompt
                check.update(file=str(path), file_sha256=digest(path), warmup='warmup' in suffix)
                record['checks'].append(check)
            record['quality_passed'] = all(check['passed'] for check in record['checks'])
            publish(summary_path, report)
            print(f"END {item['index']}/6 pid={record['pid']} quality={record['quality_passed']}", flush=True)
            # Deliberately continue after semantic/parity failure. Execution and
            # identity failures above remain fatal, so unqualified binaries are
            # never allowed to contribute the remaining independent processes.
        verify_pins(plan)
        report['complete'] = True
        report['passed'] = len(report['runs']) == 6 and all(row['process_passed'] and row['quality_passed'] for row in report['runs'])
    except Exception as error:
        report['error'] = repr(error)
    finally:
        report['window_metrics'] = aggregate(report['runs'])
        report['finished_utc'] = utc()
        publish(summary_path, report)
        print('SUMMARY ' + str(summary_path) + ' complete=' + str(report['complete']) + ' passed=' + str(report['passed']), flush=True)
    return 0 if report['passed'] else 2 if report['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    prep.add_argument('--repo-root', type=Path, default=Path.cwd())
    for name in ('entry', 'model', 'draft-model', 'output', 'plan'):
        prep.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-entry-sha256', 'expected-native-sha256'):
        prep.add_argument('--' + name, required=True)
    run = modes.add_parser('run')
    run.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        for value in (args.expected_entry_sha256, args.expected_native_sha256):
            if not re.fullmatch('[0-9a-f]{64}', value):
                parser.error('Require lowercase SHA256 pins')
        return prepare(args)
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
