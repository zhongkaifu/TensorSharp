#!/usr/bin/env python3
"""Summarize recorded readiness evidence without treating missing runs as passes.

The input directory must be under ignored artifacts/ or docs/validation/. Model
quality, geometry parsing, synthetic soak, and physical limits remain separate.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench', HERE/'multimodel-quality-bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec = importlib.util.spec_from_file_location('fixed_task_semantic', HERE/'fixed-task-semantic.py')
fixed_tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_tasks)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def span(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return f'{min(values):.2f}–{max(values):.2f}' if values else '未测量'


def gib(value):
    return f'{value / (1 << 30):.3f}' if value is not None else '未测量'


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table>'


def recorded_display_rows(data):
    """Display explicit audit summaries; do not infer qualification from exit codes."""
    fields = ('scenario', 'status', 'quality', 'performance', 'resources', 'scope')
    if not isinstance(data, dict) or not isinstance(data.get('display_rows'), list):
        return []
    return [[row.get(field, '未记录') for field in fields] for row in data['display_rows']
            if isinstance(row, dict) and all(isinstance(row.get(field), str) for field in fields)]


def collected_audit_summaries(root):
    """Select only explicitly related audit versions; retain their prior failures.

    An optional schema-1 index records current/history paths and content hashes.
    Version filenames enable discovery, never imply that a newer passing result
    supersedes a failed one. Unindexed records remain independent displays.
    """
    root = Path(root)
    paths = {p for p in root.glob('*-summary*.json')
             if re.fullmatch(r'.+-summary(?:-v[1-9][0-9]*)?\.json', p.name)}
    paths.update(p for p in root.glob('*/summary*.json')
                 if re.fullmatch(r'summary(?:-v[1-9][0-9]*)?\.json', p.name))
    selected, history, sources, indexed = {}, [], [], set()
    index_path = root / 'readiness-audit-index.json'
    if index_path.exists():
        index = read(index_path)
        if (not isinstance(index, dict) or type(index.get('schema_version')) is not int
                or index['schema_version'] != 1 or not isinstance(index.get('groups'), list)):
            raise ValueError('Unsupported readiness audit index schema')
        sources.append(index_path)
        groups = set()

        def entry(record, expected_kind=None):
            if not isinstance(record, dict) or not isinstance(record.get('path'), str):
                raise ValueError('Audit index entry requires a relative evidence path')
            relative = Path(record['path'])
            path = root / relative
            if (relative.is_absolute() or not path.resolve().is_relative_to(root.resolve())
                    or not path.is_file() or path == index_path):
                raise ValueError('Audit index evidence is missing or outside its directory: ' + record['path'])
            identity = path.resolve()
            if identity in indexed:
                raise ValueError('Audit index repeats one evidence file: ' + record['path'])
            digest = record.get('sha256')
            if (not isinstance(digest, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', digest)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest.lower()):
                raise ValueError('Audit index evidence hash differs: ' + record['path'])
            kind = record.get('kind')
            if kind not in ('summary', 'terminal', 'malformed_summary') or (expected_kind and kind != expected_kind):
                raise ValueError('Unsupported audit index evidence kind')
            data = read(path)
            rows = recorded_display_rows(data)
            malformed_display = (isinstance(data, dict) and 'display_rows' in data
                and (not isinstance(data['display_rows'], list)
                     or len(rows) != len(data['display_rows'])))
            if (not isinstance(data, dict) or (kind == 'summary'
                    and (not rows or len(rows) != len(data['display_rows'])))
                    or (kind == 'malformed_summary' and not malformed_display)
                    or (kind == 'terminal' and (type(data.get('exit_code')) is not int
                        or not isinstance(data.get('stage'), str)))):
                raise ValueError('Audit index evidence does not match its declared schema: ' + record['path'])
            indexed.add(identity)
            sources.append(path)
            return path, data, rows

        for group in index['groups']:
            if (not isinstance(group, dict) or not isinstance(group.get('id'), str) or not group['id']
                    or group['id'] in groups or not isinstance(group.get('history'), list)):
                raise ValueError('Invalid or duplicate audit index group')
            groups.add(group['id'])
            current, _, rows = entry(group.get('current'), 'summary')
            selected[current] = rows
            for prior in group['history']:
                if not isinstance(prior, dict) or not isinstance(prior.get('reason'), str) or not prior['reason']:
                    raise ValueError('Historical audit evidence requires its explicit relationship')
                path, data, _ = entry(prior)
                # Preserve the recorded verdict/error, including an auditor
                # failure whose underlying model/soak did not run again.
                status = {k: data[k] for k in ('passed', 'status', 'exit_code') if k in data}
                detail = {k: data[k] for k in ('errors', 'error', 'stage', 'scope') if k in data}
                if prior['kind'] == 'malformed_summary':
                    detail['audit_schema_error'] = (
                        '原始 display_rows 格式错误；此历史项沿用所关联的底层运行证据，'
                        '不计为新增模型执行或模型质量失败。')
                history.append([group['id'], current.relative_to(root).as_posix(),
                                path.relative_to(root).as_posix(), json.dumps(status, ensure_ascii=False),
                                prior['reason'], json.dumps(detail, ensure_ascii=False)])
    for path in sorted(paths):
        if path.resolve() in indexed:
            continue
        data = read(path)
        rows = recorded_display_rows(data)
        if (isinstance(data, dict) and 'display_rows' in data
                and (not isinstance(data['display_rows'], list)
                     or len(rows) != len(data['display_rows']))):
            raise ValueError('Audit summary has malformed display_rows: ' + path.relative_to(root).as_posix())
        if rows:
            selected[path] = rows
            sources.append(path)
    rows = []
    for path, recorded in sorted(selected.items()):
        for row in recorded:
            rows.append(row[:-1] + [row[-1] + '; 摘要：' + path.relative_to(root).as_posix()])
    return rows, history, sources


def speculative_rows(path):
    """Require complete output/engagement gates independently of timing."""
    module = importlib.util.spec_from_file_location('speculative_gate', HERE/'validate-speculative-rows.py')
    gate = importlib.util.module_from_spec(module)
    module.loader.exec_module(gate)
    module = importlib.util.spec_from_file_location('agent_compare', HERE.parents[1]/'benchmarks/AgentTurnBench/compare.py')
    compare = importlib.util.module_from_spec(module)
    module.loader.exec_module(compare)
    try:
        indexed = compare.load_rows(path)
    except (ValueError, KeyError, TypeError) as error:
        return [[path.stem, '无法验收', str(error), '', '', '', '', '', '']]
    result = []
    for scenario, plain_label, label in [('spec', 'plain greedy', 'draft head (auto)'),
                                          ('image', 'plain + image', 'draft head (auto) + image')]:
        if (scenario, label) not in indexed:
            continue
        check = gate.validate(list(indexed.values()), scenario=scenario, plain_label=plain_label, label=label)
        plain, candidate = indexed.get((scenario, plain_label), {}), indexed[(scenario, label)]
        eos = plain.get('Finish') == candidate.get('Finish') == 'eos'
        pair = lambda key: ' → '.join(f'{r[key]:.2f}' if isinstance(r.get(key), (int, float)) else '未测量'
                                      for r in (plain, candidate))
        result.append([path.parent.name + '/' + path.name, scenario,
                       check['status'] == 'passed' and eos,
                       f"{len(plain.get('Tokens', []))}/{len(candidate['Tokens'])}; EOS={eos}",
                       '/'.join(str(candidate.get(key)) for key in ('Drafted', 'Accepted', 'VerifySteps')),
                       pair('PrefillComputeTps'), pair('DecodeComputeTps'), pair('TotalMs'),
                       '; '.join(check['failures']) or ('完整 EOS 输出一致' if eos else '未完成 EOS')])
    return result


def maya_summary(data):
    """Recheck fixed-task semantics; preserve the original runner's decision."""
    rows = data.get('rows', [])
    expected = data.get('plan', {}).get('repetitions', 0)
    complete = (data.get('completed') is True and data.get('process_success') is True
                and data.get('exit_code') == 0 and data.get('identities_unchanged') is True
                and not data.get('errors') and data.get('timed_out') is False
                and expected > 0 and len(rows) == expected and data.get('prompt_tokenizer_exact') is True)
    reference = data.get('reference', {})
    case = data.get('plan', {}).get('case')
    task_bound = isinstance(case, str) and case == reference.get('name') and case in fixed_tasks.PROMPTS
    prompt = reference.get('prompt') if task_bound else None
    reference_answer = fixed_tasks.evaluate(case, prompt, reference.get('parsed_text'))
    original_policy = data.get('semantic_policy', fixed_tasks.LEGACY_MAYA_POLICY)
    eos_ids = data.get('plan', {}).get('eos_ids', [])
    reference_tokens = reference.get('generated_tokens', [])
    reference_prompt = reference.get('prompt_tokens', [])
    reference_structure = (reference.get('eos') is True and isinstance(reference_prompt, list)
        and bool(reference_prompt) and all(type(t) is int and t >= 0 for t in reference_prompt)
        and isinstance(reference_tokens, list) and bool(reference_tokens)
        and all(type(t) is int and t >= 0 for t in reference_tokens)
        and reference_tokens[-1] in eos_ids and not any(t in eos_ids for t in reference_tokens[:-1]))
    qualified = []
    for row in rows:
        tokens = row.get('tokens', [])
        eos = bool(tokens) and tokens[-1] in eos_ids and not any(t in eos_ids for t in tokens[:-1])
        wire = (row.get('generated') == len(tokens) and row.get('prompt_tokens') == len(reference.get('prompt_tokens', []))
                and row.get('reused') == row.get('drafts_offered') == row.get('drafts_accepted') == 0)
        semantic = row.get('semantic', {})
        rendered_prompt, raw_text = reference.get('rendered_prompt'), semantic.get('raw_text')
        closure_available = isinstance(rendered_prompt, str) and isinstance(raw_text, str)
        current_closed = fixed_tasks.thinking_closed(rendered_prompt, raw_text) if closure_available else None
        closure = dict(status='rechecked' if closure_available else 'structure_recheck_unavailable',
            policy=fixed_tasks.THINKING_POLICY, saved_thinking_closed=semantic.get('thinking_closed'),
            current_thinking_closed=current_closed,
            scope='Rechecks GLM/Qwen angle-bracket closure from recorded rendered prompt and raw response; content/tool-call fields remain recorded production-parser output, not token re-decoding.')
        independent = fixed_tasks.parsed_answer(case, prompt, semantic.get('content'),
            current_closed, semantic.get('tool_calls'))
        passed = (complete and reference_structure
                  and wire and eos and row.get('finish') == 'stop' and independent['passed'])
        rate = lambda n, ms: n * 1000 / ms if isinstance(ms, (float, int)) and math.isfinite(ms) and ms > 0 and n > 0 else None
        qualified.append(dict(passed=passed, eos=eos, exact=bool(tokens) and tokens == reference.get('generated_tokens'),
            independent_semantic=independent, recorded_runner_passed=row.get('quality', {}).get('passed'),
            thinking_closure_recheck=closure,
            recorded_semantic_passed=semantic.get('passed'),
            pp=rate(row.get('prompt_tokens', 0), row.get('prompt_ms')),
            tg=rate(row.get('generated', 0) - 1, row.get('decode_ms'))))
    return dict(complete=complete, expected=expected, rows=qualified,
                recorded_runner_quality_passed=data.get('quality_passed'), recorded_runner_policy=original_policy,
                independent_policy=fixed_tasks.POLICY, reference_independent_semantic=reference_answer,
                reference_structure_valid=reference_structure, reference_recorded_passed=reference.get('passed'),
                independent_quality_passed=complete and all(row['passed'] for row in qualified),
                quality_passed=complete and all(row['passed'] for row in qualified),
                performance_complete=complete and data.get('performance_observation_complete') is True
                    and data.get('telemetry_samples', 0) > 0 and not data.get('sampling_errors'))


def maya_rows(path):
    data = read(path)
    summary = maya_summary(data)
    telemetry = path.parent / 'telemetry.jsonl'
    samples = [json.loads(line) for line in telemetry.read_text(encoding='utf-8').splitlines() if line.strip()] if telemetry.exists() else []
    peak = lambda key: max((s['memory'][key] for s in samples if s.get('memory', {}).get(key) is not None), default=None)
    board = max((g['used_mib'] for s in samples for g in s.get('gpu', []) if g.get('used_mib') is not None), default=None)
    memory = [gib(peak('rss')), gib(peak('private')), gib(board * (1 << 20)) if board is not None else '未测量']
    result = []
    for index, (row, check) in enumerate(zip(data.get('rows', []), summary['rows'])):
        result.append([path.parent.name + f' / {index + 1}', summary['complete'], check['passed'],
            check['exact'], f"{row.get('generated')}; EOS={check['eos']}", row.get('first_token_ms'),
            span([check['pp']]), span([check['tg']]), row.get('wall_ms'), summary['performance_complete'], *memory,
            'prefill=' + str(data.get('plan', {}).get('prefill', 'auto')) + '; ' + data.get('plan', {}).get('controls_scope', '')
                + '; closure=' + check['thinking_closure_recheck']['status']
                + '; saved/current=' + str(check['thinking_closure_recheck']['saved_thinking_closed'])
                + '/' + str(check['thinking_closure_recheck']['current_thinking_closed']),
            check['recorded_runner_passed'], summary['recorded_runner_policy'], summary['independent_policy'],
            summary['reference_independent_semantic']['passed']])
    if not result:
        result.append([path.parent.name, False, False, '未测量', '无完整请求', '未测量', '未测量', '未测量',
                       '未测量', False, *memory, '; '.join(data.get('errors', [])),
                       '无完整请求', summary['recorded_runner_policy'], summary['independent_policy'],
                       summary['reference_independent_semantic']['passed']])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    root = bench.evidence_directory(args.directory)
    quality, reference, image, context, soak, geometry, direct, numerical, tests, limits = [], [], [], [], [], [], [], [], [], []
    speculative, context_comparisons, processes, planned_chats, maya = [], [], [], [], []
    audited, audit_history, sources = collected_audit_summaries(root)
    for path in sorted(root.glob('*/report.json')):
        data = read(path)
        if 'source_revision' in data.get('plan', {}) and 'pack_evidence' in data.get('plan', {}):
            maya.extend(maya_rows(path))
            sources.append(path)
            for name in ('stdout.log', 'telemetry.jsonl'):
                if (path.parent / name).exists():
                    sources.append(path.parent / name)
        if 'records' not in data or 'execution_plan' not in data:
            continue
        rows, mem = data['records'], data.get('memory', {})
        quality.append([path.parent.name, f"{sum(r.get('quality_passed') is True for r in rows)}/{data['execution_plan']['expected_requests']}",
                        '完成' if data.get('run_complete') else '未完成',
                        span([r.get('metrics', {}).get('prefill_tps') for r in rows]),
                        span([r.get('metrics', {}).get('decode_tps') for r in rows]),
                        gib(mem.get('peak_rss_bytes')), gib(mem.get('peak_private_bytes')),
                        gib(mem['peak_board_mib'] * (1 << 20)) if mem.get('peak_board_mib') is not None else '未测量',
                        str(data.get('json_schema', {}).get('passed', '未测量')),
                        '/'.join(str(r.get('roundtrip_ok', r.get('passed', r.get('ok', '未测量')))) for r in data.get('tool_roundtrips', [])) or '未测量'])
        sources.append(path)
    for path in sorted(root.glob('*/quality/report.json')):
        data = read(path)
        rows = data.get('records', [])
        reference.append([path.parent.parent.name, data.get('run_complete'),
                          f"{sum(r.get('quality_passed') is True for r in rows)}/{len(rows)}",
                          span([r.get('metrics', {}).get('prefill_tps') for r in rows]),
                          span([r.get('metrics', {}).get('decode_tps') for r in rows])])
        sources.append(path)
    for path in sorted(root.glob('*/benchmark.json')):
        data = read(path)
        if 'runs' not in data:
            continue
        for pair in data['runs']:
            for engine, result in pair['engines'].items():
                image.append([path.parent.name, engine, result.get('status'), result.get('wall_seconds'),
                              json.dumps(result.get('phases_seconds', {}), ensure_ascii=False), gib(result.get('max_rss_bytes'))])
        sources.append(path)
    for path in sorted(set(root.glob('*.json')) | set(root.glob('*/rows.json'))
                       | set(root.glob('*/rows.json.*.json'))):
        data = read(path)
        if isinstance(data, list) and data and isinstance(data[0], dict) and {'Scenario', 'Label', 'Tokens'} <= data[0].keys():
            speculative.extend(speculative_rows(path))
            sources.append(path)
            for evidence in (path.with_name(path.stem + '-validation.json'), path.parent/'validation.json',
                             path.parent/'execution.json', path.parent/'rows.json.run.json',
                             path.with_name(path.stem + '-identity.json')):
                if evidence.is_file() and evidence not in sources:
                    sources.append(evidence)
        if isinstance(data, dict) and {'exact_tokens', 'actual_concurrent_width', 'process_pairs'} <= data.keys():
            changes = data.get('metrics', {})
            context_comparisons.append([path.stem, data.get('passed'), data['exact_tokens'],
                data['actual_concurrent_width'], data['process_pairs'],
                *[changes.get(key, {}).get('latency_change_percent', '未测量')
                  for key in ('wall_seconds', 'prefill_compute_seconds', 'decode_compute_seconds')]])
            sources.append(path)
        if isinstance(data, dict) and 'exact' in data and 'elements' in data and 'max_abs' in data:
            numerical.append([path.stem, data['exact'], data['elements'], data['max_abs'],
                              data.get('rms'), data.get('scope', '')])
            sources.append(path)
        if isinstance(data, dict) and 'exact_full_logits' in data:
            numerical.append([path.stem, data['exact_full_logits'], '完整 capture / teacher tokens / binary identity',
                              0 if data['exact_full_logits'] else '见记录', 0 if data['exact_full_logits'] else '见记录',
                              f"policy={data.get('policy')}; 性能门槛通过={data.get('passed')}; "
                              f"语义另测={data.get('quality_checked')}; 独立进程对数={data.get('process_pairs')}"])
            sources.append(path)
        if 'logits' in data and 'runs' in data and 'model_path' in data:
            runs = [r for r in data['runs'] if not r.get('warmup')]
            memory = data.get('memory', [])
            peak = lambda key: max((m[key] for m in memory if m.get(key) is not None), default=None)
            direct.append([path.stem, data.get('passed'),
                           str(data.get('quality_passed')) if data.get('quality_checked') else '未测语义',
                           span([r.get('prefill_tps') for r in runs]), span([r.get('decode_tps') for r in runs]),
                           gib(peak('process_peak_working_set_bytes')), gib(peak('process_private_bytes')),
                           data.get('gptoss_per_expert_prefill', {}).get('requested', False)])
            sources.append(path)
        if 'TokenParity' in data and 'ContextTokens' in data:
            context.append([path.stem, data.get('Passed'), data.get('ContextTokens'),
                            max((r['Running'] for r in data.get('RunningSamples', [])), default=0),
                            data.get('TokenParity'), data.get('LoadSeconds'), data.get('WallSeconds'),
                            span([r.get('PrefillTokensPerSecond') for r in data.get('Requests', [])]),
                            span([r.get('DecodeTokensPerSecond') for r in data.get('Requests', [])])])
            sources.append(path)
    for path in sorted(root.glob('gpt-expert-oracle-*/report.json')):
        data = read(path)
        numerical.append([path.parent.name, data.get('all_fused_split_exact'),
                          '单专家、逐阶段完整数组及抽样 FP64', '见逐阶段记录', '见逐阶段记录',
                          f"执行完成={data.get('completed')}, 释放完成={data.get('cleanup')}; "
                          + data.get('input', '')])
        sources.append(path)
        if 'all_fused_temporary_exact' in data:
            numerical.append([path.parent.name + ' / temporary', data['all_fused_temporary_exact'],
                              '两个专家交替后的完整输出', '见逐阶段记录', '见逐阶段记录',
                              f"设备 lazy cache quota={data.get('device_cache_bytes')}; 释放完成={data.get('cleanup')}"])
    for path in sorted(root.rglob('*.trx')):
        tree = ET.parse(path)
        counters = next((node.attrib for node in tree.iter() if node.tag.endswith('}Counters')), None)
        if counters:
            tests.append([path.relative_to(root).as_posix(), counters.get('passed'), counters.get('failed'),
                          int(counters.get('total', 0)) - int(counters.get('executed', 0)), counters.get('total')])
            sources.append(path)
    for path in sorted(root.glob('standalone-memory-final-*.json')):
        data = read(path)
        if 'tests' not in data or 'hardwareInference' not in data:
            continue
        rows = data['tests']
        tests.append([path.stem, sum(r.get('passed') is True and not r.get('skipped') for r in rows),
                      sum(r.get('passed') is False and not r.get('skipped') for r in rows),
                      sum(bool(r.get('skipped')) for r in rows), len(rows)])
        sources.append(path)
    for path in sorted(root.glob('*/execution.json')):
        data = read(path)
        if isinstance(data.get('plan'), dict) and 'futureRequest' in data['plan']:
            response_path = path.parent/'response.json'
            response = read(response_path) if response_path.exists() else {}
            timings = response.get('timings') or {}
            parity = data.get('prompt_comparability') or {}
            planned_chats.append([path.parent.name, data.get('complete'), data.get('quality_passed'),
                data.get('timed_out'), parity.get('same_input_control', '未比较'),
                data.get('load_ready_seconds', '未完成'), data.get('request_wall_seconds', '未完成'),
                timings.get('prompt_per_second', '未测量'), timings.get('predicted_per_second', '未测量'),
                data.get('deployment_unchanged', '未完成'),
                data.get('error') or '; '.join(data.get('cleanup_errors', []))])
            for evidence in (path, response_path, path.parent/'prompt.json', path.parent/'memory.jsonl'):
                if evidence.exists() and evidence not in sources:
                    sources.append(evidence)
        if 'command' in data and 'timeout_seconds' in data:
            telemetry = data.get('telemetry') or {}
            memory = telemetry.get('memory') or {}
            processes.append([path.parent.name, data.get('complete'), data.get('timed_out'),
                data.get('exit_code'), data.get('wall_seconds'), telemetry.get('passed', '未测量'),
                gib(memory.get('peak_rss_bytes')), gib(memory.get('peak_private_bytes')),
                gib(memory['peak_board_mib'] * (1 << 20)) if memory.get('peak_board_mib') is not None else '未测量',
                data.get('error', '')])
            if path not in sources:
                sources.append(path)
        if data.get('enforcement', '').startswith('windows-job'):
            limits.append([path.parent.name, data.get('passed'), data.get('enforcement'),
                           gib(data.get('effective_commit_bytes')), data.get('scope')])
            sources.append(path)
    path = root/'wsl/canary-v3/report.json'
    if path.exists():
        data = read(path)
        limits.append(['WSL canary-v3', data.get('passed'), 'Linux cgroup v2 parent / child',
                       gib(data.get('parentLimit')), '可见父组 RAM/file/kernel 计账限制；不含完整 CUDA/driver，不证明 Windows/Apple 等价'])
        sources.append(path)
    for directory in sorted(root.glob('cpu-soak-*')):
        final = directory/'report.json'
        progress = directory/'progress.jsonl'
        if final.exists():
            data = read(final)
            sources.append(final)
        elif progress.exists() and progress.stat().st_size:
            data = json.loads(progress.read_text(encoding='utf-8-sig').splitlines()[-1])
            sources.append(progress)
        else:
            continue
        if (directory/'interruption.json').exists() and not final.exists():
            data['status'] = 'interrupted'
            sources.append(directory/'interruption.json')
        soak.append([directory.name, data.get('status'), data.get('elapsedSeconds'), data.get('submitted'),
                     data.get('cancelled'), data.get('completed'), data.get('livePhysicalBytes'), data.get('failure')])
    path = root/'metadata-abba.json'
    if path.exists():
        for model in read(path)['Results']:
            arms = {name: [r for r in model['Runs'] if r['Arm'] == name and not r['Warmup']] for name in ('full', 'geometry')}
            mean = lambda name, key: statistics.mean(r[key] for r in arms[name])
            geometry.append([Path(model['Path']).name, f"{mean('full', 'Milliseconds'):.2f}",
                             f"{mean('geometry', 'Milliseconds'):.2f}",
                             f"{mean('full', 'AllocatedBytes') / (1 << 20):.2f}",
                             f"{mean('geometry', 'AllocatedBytes') / (1 << 20):.2f}",
                             len({r['GeometrySha256'] for r in model['Runs']}) == 1])
        sources.append(path)
    sections = [
        ('最新专项审计', '展示各摘要明确记录的数值、语义、性能和资源边界；不是重新执行验收。显式索引按路径和内容身份指定当前审计，不按通过/失败排序；未声明关系的记录不自动合并。摘要与后续原运行记录可能重叠，不能相加为独立测试次数。性能观测、性能门槛通过与稳定提速不同；账本归零不等于全物理归属。',
         ['场景', '状态', '质量/正确性', '性能', '资源', '边界'], audited),
        ('历史审计与状态快照', '保留索引明确关联的原结论、失败及修订原因；这些是同一底层执行的审计版本或状态快照，不计作新增模型运行。修复时间解析或终端编码不表示重新运行了模型或长期测试。未被索引声明的历史证据仍按原规则展示。',
         ['审计族', '当前摘要', '历史记录', '原状态', '与当前版本的关系', '原错误/边界'], audit_history),
        ('模型文本质量与速度', '严格任务通过数；没有删改输出。真实工具调用及 JSON Schema 单列。数字不是等工作量的 llama.cpp 性能对照。',
         ['场景', '严格通过', '执行状态', 'prefill tok/s', 'decode tok/s', 'RSS GiB', 'private GiB', '整卡 VRAM GiB', 'Schema', '工具往返'], quality),
        ('图像生成', '完成执行不等于语义质量通过；需检查输出图像。两引擎阶段计时边界不同。',
         ['场景', '引擎', '执行状态', 'wall s', '阶段 s', 'RSS GiB'], image),
        ('独立 llama.cpp 对照', '分别保留默认思考模板和显式 reasoning_budget_tokens=0 的结果；前者可能在 256 tokens 内未生成最终答案。须对齐模式再分析质量差异，不能混合计时分母直接计算速度比。',
         ['场景', '完整执行', '严格通过/已记录请求', 'prefill tok/s', 'decode tok/s'], reference),
        ('有截止时间的 llama.cpp 单请求', '模型加载和全部 HTTP 请求共用截止时间。low reasoning effort 与 reasoning_budget_tokens=0 不等价。语义通过、模板/token 完全一致、部署身份和执行完成分别记录；单次 HTTP 速度不能与固定 token 探针的计算吞吐直接相除。stop 不证明具体 EOS ID。',
         ['场景', '完成', '语义通过', '超时', '同输入', '加载就绪 s', '请求 wall s', 'prefill tok/s', 'decode tok/s', '部署不变', '错误'], planned_chats),
        ('Maya 同输入 GLM 对照', '独立复核以固定任务及其原始问题为准，使用记录的生产解析器 content；不重新解码 token，也不以 reference 答案为 ground truth。原运行器结论与策略另列，历史结果未覆盖；缺少任务或问题不能通过。自然 EOS、完整执行、部署身份和无工具调用仍须满足。decode 分母为 generated−1 次模型 forward，DONE 时间还含采样及输出。compat-bf16 改变部分权重运算，同硬件/源检查点不等于相同物理硬预算。进程峰值在请求行重复展示，RSS/private/整卡 VRAM 不可相加。采样会漏峰且有未量化开销；顺序和缓存未平衡，不证明稳定胜负。',
         ['场景', '进程完整', '独立完整语义复核', '完整 token exact', '输出 / EOS', 'TTFT ms', 'prefill tok/s', 'decode forward tok/s', 'wall ms', '观测完整', '采样 RSS GiB', '采样 private GiB', '整卡 VRAM GiB', '配置与预算边界', '原运行器结论', '原评分策略', '独立评分策略', 'reference 独立答案复核'], maya),
        ('长上下文', 'reference 是无共享预算 TensorSharp。running=2 是两请求同时准入、交替执行，不是双序列 CUDA 内核。',
         ['场景', '通过', '容量', '最大 running', 'token 一致', '加载 s', 'wall s', 'prefill tok/s', 'decode tok/s'], context),
        ('长上下文同构建对照', '只接收比较器记录；正数是候选延迟增加，负数是减少。完整 token、真实并发宽度、相同程序集和清理另由比较器验收。有限进程对及后台负载仍不能证明稳态性能。',
         ['场景', '综合通过', '完整 token 一致', '实际并发', '进程对数', 'wall 变化 %', 'prefill compute 变化 %', 'decode compute 变化 %'], context_comparisons),
        ('学习式 MTP 完整输出与计时', '每行重新要求完整 token/finish 一致、两臂 EOS 及 Drafted/Accepted/VerifySteps 均大于零；不能由进程退出零代替。表格按 plain → MTP 展示，实际执行顺序以对应 run.json 的 SpecOrder 为准；warmup 文件保留但不计入性能均值。同进程先后顺序存在冷图/JIT/缓存差别，prefill 和 wall 差值不是 MTP 提速证明。图像 Agent 计时排除媒体准备/编码，内存与实际加载 native 身份以单独 execution 记录为准。',
         ['记录', '场景', '严格通过', '输出数 / EOS', 'draft/accept/verify', 'prefill compute tok/s', 'decode compute tok/s', 'engine wall ms', '检查'], speculative),
        ('固定 token 与数值探针', '单次通过仅说明有限值、重复性及已请求的语义检查；不表示与另一实现数值一致。强制专家后备路径、1-byte lazy cache 等诊断配置单独保留，不是正常部署速度。RSS/private 不能相加。',
         ['场景', '单次通过', '语义', 'prefill tok/s', 'decode tok/s', '峰值 RSS GiB', '峰值 private GiB', '强制专家后备'], direct),
        ('有截止时间的进程与采样', '进程退出零、观测完整和模型质量是独立条件。超时或未完成进程保留在此，不能因没有最终模型报告而被当作通过。RSS/private/整卡 VRAM 为采样值，不能相加，也不是物理硬限。',
         ['场景', '进程完成', '超时', '退出码', 'wall s', '观测通过', 'RSS GiB', 'private GiB', '整卡 VRAM GiB', '执行错误'], processes),
        ('数值资格与独立诊断', 'False 表示完整字节一致性没有通过，不能由执行成功、相同 argmax 或放宽误差门槛替代。FP64 使用原量化权重的独立解码与合成输入，不代表完整模型语义质量。',
         ['场景', '完整字节一致', '比较范围', '最大绝对差', 'RMS', '边界'], numerical),
        ('测试执行记录', '各 filter 有重叠，不可相加为独立通过总数。历史失败和未执行记录保留；只有具体实机测试可以证明对应 GPU/模型覆盖。',
         ['记录', '通过', '失败', '未执行', '总数'], tests),
        ('OS 限额验证', '执行器和计账维度分别列出；Windows private commit、Linux cgroup、进程 RSS、整卡 VRAM 不是可互换的同一个限额。',
         ['场景', '通过', '执行器', '限额 GiB', '覆盖范围'], limits),
        ('长期取消与公平性', 'CPU worker、真实有界原生主机分配及确定性状态模型；不代表训练模型/GPU/SSD soak。running 不算验收通过。',
         ['场景', '状态', '已运行 s', '提交', '取消', '完成', '存活物理 bytes', '失败'], soak),
        ('规划元数据读取', '8 次 ABBA 顺序中排除首对后的各 3 次均值；只测读取调用线程分配量。不是整模型加载、保留 RAM 或受控磁盘冷启动。',
         ['模型', '完整 ms', 'geometry ms', '完整分配 MiB', 'geometry 分配 MiB', '几何 hash 一致'], geometry)]
    body = ['<!doctype html><meta charset="utf-8"><title>TensorSharp 默认就绪验证</title>',
            '<style>body{font:16px system-ui;margin:32px;background:#f5f7fa;color:#172436}h1,h2{color:#123757}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d6dce5;text-align:left}th{background:#e9eff7}section{margin:32px 0;overflow-x:auto}p{max-width:1050px;line-height:1.65}code{overflow-wrap:anywhere}</style>',
            '<h1>TensorSharp 默认就绪验证</h1><p>共享预算、AdaptiveModelSession 与请求准入仍为显式启用。账本、物理观测、OS 硬限有不同的覆盖范围。此页仅汇总已经写出的证据；缺失、跳过、未完成不算通过。模型测试期间可能与下载及 CPU soak 重叠，不能作为独占机器的稳态性能结论。</p>']
    for title, note, headers, rows in sections:
        body += [f'<section><h2>{title}</h2><p>{note}</p>', table(headers, rows) if rows else '<p>尚无此类完成记录。</p>', '</section>']
    for path in sorted(root.glob('*/visual-review.json')):
        review = read(path)
        picture = path.parent / review['image']
        if picture.is_file() and picture.parent == path.parent:
            relative = html.escape(picture.relative_to(root).as_posix(), quote=True)
            body += ['<section><h2>已检查的图像输出</h2>',
                     f'<img alt="{html.escape(path.parent.name, quote=True)}" src="{relative}" width="512" style="max-width:100%;height:auto">',
                     f'<p>{html.escape(str(review.get("checks", {})))}</p></section>']
        sources.append(path)
    body.append('<h2>原始证据</h2><ul>' + ''.join(f'<li><a href="{html.escape(p.relative_to(root).as_posix(), quote=True)}">{html.escape(p.relative_to(root).as_posix())}</a></li>' for p in sorted(set(sources))) + '</ul>')
    (root/'report.html').write_text('\n'.join(body), encoding='utf-8')
    print(root/'report.html')


if __name__ == '__main__':
    main()
