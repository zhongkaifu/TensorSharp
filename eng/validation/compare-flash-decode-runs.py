#!/usr/bin/env python3
"""Compare four isolated Flash probe runs (control/candidate/candidate/control).

Accept execution.json emitted by a process runner beside model.json. Keep cold
requests separate from measured repetitions, require every vocabulary row's
hash, and report observed memory rather than treating a cache cap as total RAM.
"""
import argparse
import json
import math
import re
from pathlib import Path, PureWindowsPath
import statistics


def distribution(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), samples=len(values))


def compare(executions, candidate_first=False, candidate_file_read=False, candidate_lfu=False):
    result = dict(passed=False, failures=[], measurements={}, cold_requests=[], memory={}, reports=[])
    result['execution_order'] = ['candidate','control','control','candidate'] if candidate_first else ['control','candidate','candidate','control']
    errors = result['failures']
    if candidate_file_read and candidate_lfu:
        errors.append('Compare source transport and eviction policy separately')
        return result
    if len(executions) != 4:
        errors.append('Exactly four ABBA executions are required')
        return result
    models = []
    previous_end = None
    for path in executions:
        path = Path(path)
        execution = json.loads(path.read_text(encoding='utf-8-sig'))
        model = json.loads((path.parent / 'model.json').read_text(encoding='utf-8-sig'))
        models.append(model)
        if execution.get('arm') is not None and execution['arm'] != result['execution_order'][len(models)-1]:
            errors.append(f'{path}: execution arm contradicts requested comparison order')
        result['reports'].append(str(path))
        if candidate_file_read or candidate_lfu:
            log_path = path.parent / 'process.log'
            log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.is_file() else ''
            reads = re.findall(r'^\[HOSTMOE-CACHE-FILE\].*?calls=(\d+) (?:file_bytes|bytes)=(\d+)', log, re.MULTILINE)
            workspaces = re.findall(r'^\[HOSTMOE-FILE-WORKSPACE\] bytes=(\d+) ceiling=(\d+)', log, re.MULTILINE)
            candidate = result['execution_order'][len(models)-1] == 'candidate'
            if candidate or candidate_lfu:
                if not reads or not any(int(calls) > 0 and int(size) > 0 for calls, size in reads):
                    errors.append(f'{path}: file-read arm has no actual source-read evidence')
                if not workspaces or any(not 0 < int(size) <= int(cap) <= 32*1024*1024 for size, cap in workspaces):
                    errors.append(f'{path}: missing or unbounded file-read workspace evidence')
            elif reads or workspaces:
                errors.append(f'{path}: mmap control unexpectedly used file staging')
            if candidate_lfu:
                policies = re.findall(r'^\[HOSTMOE-EVICTION\] layer=\d+ policy=(lru|lfu) epoch=(\d+)$', log, re.MULTILINE)
                expected = 'lfu' if candidate else 'lru'
                if not policies or any(policy != expected or (int(epoch) == 0) != (expected == 'lru') for policy, epoch in policies):
                    errors.append(f'{path}: missing or wrong native eviction policy evidence')
        if execution.get('exit_code') != 0 or execution.get('timed_out') is not False:
            errors.append(f'{path}: execution did not complete successfully')
        start, elapsed = execution.get('started_unix'), execution.get('wall_seconds')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, elapsed)) or elapsed <= 0:
            errors.append(f'{path}: invalid process interval')
        else:
            if previous_end is not None and start < previous_end:
                errors.append(f'{path}: processes overlap or are out of order')
            previous_end = start + elapsed
        if model.get('passed') is not True or model.get('run_complete') is not True:
            errors.append(f'{path}: incomplete model execution')
        cleanup = model.get('cleanup', {})
        required_cleanup = ['model_disposed', 'cache_cleared', 'reuse_released', 'native_shutdown']
        # A run without a shared device-budget scope has nothing to detach.
        if model.get('device_budget_bytes') is not None:
            required_cleanup.append('scope_detached')
        if (any(cleanup.get(k) is not True for k in required_cleanup)
                or any(cleanup.get(k) is not False for k in ('retained_model_owner', 'retained_scope_owner'))
                or cleanup.get('errors') != []):
            errors.append(f'{path}: cleanup incomplete')
        if not model.get('checkpoint_identity') or not model.get('native_sha256') or not model.get('managed_assemblies_sha256'):
            errors.append(f'{path}: missing checkpoint/native/managed identity')
        cache = model.get('cache_stats', {})
        if not (0 < cache.get('ReservedBytes', 0) <= cache.get('BudgetBytes', 0)
                and cache.get('Hits', 0) > 0 and cache.get('Misses', 0) > 0):
            errors.append(f'{path}: missing budgeted expert-cache engagement')
        runs = model.get('runs', [])
        measured = [r for r in runs if r.get('warmup') is False]
        warm = [r for r in runs if r.get('warmup') is True]
        if len(warm) != 1 or len(measured) < 2:
            errors.append(f'{path}: require one separate warmup and at least two measurements')
        for r in runs:
            if model.get('decode_mode') == 'greedy' and r.get('finish_reason') != 'eos':
                errors.append(f'{path}: greedy answer did not complete at EOS')
            calls = r.get('decode_tokens', 0)
            values = [r.get(k) for k in ('prefill_ms', 'decode_ms', 'prefill_tps', 'decode_tps')]
            if calls < 1 or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
                errors.append(f'{path}: invalid throughput row')
                continue
            if not math.isclose(r['decode_tps'], calls * 1000 / r['decode_ms'], rel_tol=1e-10):
                errors.append(f'{path}: decode denominator mismatch')
            if not math.isclose(r['prefill_tps'], r['prefill_tokens'] * 1000 / r['prefill_ms'], rel_tol=1e-10):
                errors.append(f'{path}: prefill denominator mismatch')
            if not r.get('full_logit_chain_sha256') or r.get('logit_rows') != calls + 1 or len(r.get('decode_step_ms', [])) != calls:
                errors.append(f'{path}: missing complete vocabulary history or per-step timing')
            if any(not math.isfinite(x) or x <= 0 for x in r.get('decode_step_ms', [])):
                errors.append(f'{path}: invalid per-step timing')
            before, after = r.get('cache_stats_after_prefill', {}), r.get('cache_stats_after_decode', {})
            layers = model.get('model_geometry', {}).get('layers', 0)
            if not layers or after.get('Calls', 0) - before.get('Calls', 0) != calls * layers:
                errors.append(f'{path}: not every decode expert row used the cache')
        result['cold_requests'].append([{'prefill_tps':r['prefill_tps'], 'decode_tps':r['decode_tps']} for r in warm])

    reference = models[0]
    def assemblies(m):
        return {(PureWindowsPath(k).name if '\\' in k else Path(k).name): v
                for k, v in m.get('managed_assemblies_sha256', {}).items()}
    def environment(m):
        allowed = {'TS_HOST_MOE_EXPERT_CACHE_LFU'} if candidate_lfu else {'TS_HOST_MOE_FILE_READ'} if candidate_file_read else {'TS_HOST_MOE_EXPERT_CACHE_PREFETCH'}
        return {k: v for k, v in m.get('environment', {}).items() if k not in allowed}
    def options(m):
        return {k:v for k,v in m.get('requested_options', {}).items() if k not in ('output','logits-dir')}
    for i, m in enumerate(models):
        if candidate_lfu:
            expected = '1' if result['execution_order'][i] == 'candidate' else '0'
            if m.get('environment', {}).get('TS_HOST_MOE_EXPERT_CACHE_LFU') != expected or m.get('environment', {}).get('TS_HOST_MOE_FILE_READ') != '1':
                errors.append(f'run {i}: expected file staging and the requested eviction arm')
        if candidate_file_read:
            expected = '1' if result['execution_order'][i] == 'candidate' else '0'
            if m.get('environment', {}).get('TS_HOST_MOE_FILE_READ') != expected:
                errors.append(f'run {i}: file staging does not match the requested comparison arm')
        for key in ('checkpoint_identity', 'model_geometry', 'prompt_tokens', 'forced_tokens', 'decode_mode', 'device_budget_bytes'):
            if m.get(key) != reference.get(key):
                errors.append(f'run {i}: {key} differs')
        if assemblies(m) != assemblies(reference) or environment(m) != environment(reference) or options(m) != options(reference):
            errors.append(f'run {i}: managed binaries, settings or request differ')
        if len(m.get('runs', [])) != len(reference.get('runs', [])):
            errors.append(f'run {i}: different repetition count')
        for r in m.get('runs', []):
            a = reference['runs'][0]
            for key in ('full_logit_chain_sha256', 'final_logit_sha256', 'generated_tokens', 'decode_tokens', 'logit_rows'):
                if r.get(key) != a.get(key): errors.append(f'run {i}: {key} differs, including warmup')
        if m.get('cache_stats', {}).get('ReservedBytes') != reference.get('cache_stats', {}).get('ReservedBytes'):
            errors.append(f'run {i}: expert cache reservation changed')
    if models[0].get('native_sha256') != models[3].get('native_sha256') or models[1].get('native_sha256') != models[2].get('native_sha256'):
        errors.append('Native identities do not form ABBA')
    arms = [('candidate', (0,3)), ('control', (1,2))] if candidate_first else [('control', (0,3)), ('candidate', (1,2))]
    for arm, indices in arms:
        rows = [r for i in indices for r in models[i].get('runs', []) if r.get('warmup') is False]
        if not rows: continue
        result['measurements'][arm] = {k:distribution([r[k] for r in rows]) for k in ('prefill_tps','decode_tps')}
        telemetry = [json.loads(Path(executions[i]).read_text(encoding='utf-8-sig')) for i in indices]
        rss = [x.get('windows_process_memory', {}).get('peak_rss_bytes') for x in telemetry]
        board = [p['peak_memory_used_mib'] for x in telemetry for p in x.get('gpu_sampling', {}).get('peaks', [])]
        result['memory'][arm] = {'peak_process_working_set_bytes': max(rss) if all(x is not None for x in rss) else None,
                               'sampled_whole_board_peak_mib': max(board) if board else None,
                               'expert_cache_bytes': models[indices[0]].get('cache_stats', {}).get('ReservedBytes')}
    if len(result['measurements']) == 2:
        result['candidate_to_control_ratio'] = {k:result['measurements']['candidate'][k]['median']/result['measurements']['control'][k]['median'] for k in ('prefill_tps','decode_tps')}
    result['passed'] = not errors
    result['limitations'] = ['Warm measured fixed histories are not fresh-request language quality or independent engine parity.',
                            'Cold means first request in a fresh process, not controlled cold storage; OS page cache is not flushed.',
                            'Working set and sampled whole-board usage are observations, not a hard RAM/VRAM cap or physical SSD measurement.',
                            'Passing validates comparability and output; a speedup is not required and must be read from the measurements.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executions', nargs=4, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-first', action='store_true', help='Label the outer pair as candidate (BAAB)')
    parser.add_argument('--candidate-file-read', action='store_true', help='Compare file staging against mmap with otherwise identical settings')
    parser.add_argument('--candidate-lfu', action='store_true', help='Compare frequency eviction against LRU with file staging in both arms')
    args = parser.parse_args()
    report = compare(args.executions, args.candidate_first, args.candidate_file_read, args.candidate_lfu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
