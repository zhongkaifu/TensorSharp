#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""Extract actual completed Server token traces and prepare raw llama plans.

Never starts a model. Requires a finished owned acceptance run, its JSONL copy,
binary snapshots and a pinned existing llama deployment/checkpoint plan. Only
uncompacted, nonthinking, single-generation text requests are supported. The
export is conditioning evidence, not semantic acceptance or a fresh model hash.
The AR commit check accepts exactly two counted conventions: a terminal sample
left unforwarded (P+O-1), or every output sample forwarded (P+O). It records
which relation the trace establishes; this is not a count tolerance or an
independent assertion of the executor path.
The capture schema is explicit: concurrency/report.json uses its fused JSONL
pin; serial-three and three-chat use direct JSONL and checker pins. Three-chat
can select chat-1 or chat-2 only after all original identities, first-chat
cancellation and actual N3/N2 graphs are checked. Its cancelled chat-0 is never
exported as a completed answer. Free-text trajectory agreement is not factual
quality or a matched-budget performance comparison.
"""
from __future__ import annotations
import argparse
import calendar
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import subprocess

MAX_BYTES = 128 << 20
SHA = re.compile(r'[0-9a-fA-F]{64}\Z')
EVENTS = ('generation.validation.input', 'generation.validation.output', 'generation.timings')
THREE_CHAT_CHECKER = Path(__file__).with_name('unified-three-chat-regression.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0, maximum=2147483647):
    return type(value) is int and minimum <= value <= maximum


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON member: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    def floating(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON number: ' + value)
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=floating)


def read(path):
    path = Path(path)
    require(0 < path.stat().st_size <= MAX_BYTES, 'Input size outside bounded evidence range')
    raw = path.read_bytes()
    require(len(raw) <= MAX_BYTES, 'Input grew beyond evidence bound')
    return raw.decode('utf-8-sig'), {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def load(path):
    raw, pin = read(path)
    return strict_json(raw), pin


def token_hash(values):
    require(isinstance(values, list) and values and all(integer(v) for v in values), 'Invalid int32 token list')
    return hashlib.sha256(struct.pack('<' + 'i' * len(values), *values)).hexdigest()


def token_csv(value):
    require(isinstance(value, str) and re.fullmatch(r'(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*', value),
            'Token CSV must contain canonical nonnegative decimal IDs')
    parts = value.split(',')
    require(len(parts) <= 1048576 and all(len(v) <= 10 for v in parts), 'Token trace exceeds bounded int32 range')
    result = [int(v) for v in parts]
    token_hash(result)
    return result


def instant(value):
    require(isinstance(value, str), 'Timestamp must be a string')
    match = re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(?:Z|\+00:00)', value)
    require(match is not None, 'Require explicit UTC evidence timestamp')
    parsed = datetime.strptime(match[1], '%Y-%m-%dT%H:%M:%S')
    return calendar.timegm(parsed.timetuple()) * 1000000000 + int((match[2] or '').ljust(9, '0'))


def path_key(value):
    require(isinstance(value, str) and value, 'Missing file path')
    return value.replace('\\', '/').casefold()


def event_is(row, prefix):
    text = row.get('template', row.get('message', ''))
    return isinstance(text, str) and (text == prefix or text.startswith(prefix + ' '))


def ar_commit_convention(prompt_count, output_count, computed_tokens):
    """Classify exact plain-AR commit counts, including an engine EOS token.

    BatchExecutor's fused decode first samples the token used as graph input,
    then appends that token and advances NumComputedTokens after graph success
    (Scheduling/BatchExecutor.cs, sampled/logits batched decode). Its solo
    fused fallback likewise samples/appends before Forward, then advances.
    These completed paths have forwarded the terminal output: P+O. A
    forward-then-sample path leaves only the terminal sample unforwarded:
    P+O-1. Neither permits an extra forwarded token or two missing outputs.
    Counts establish the commit relation, not which executor implementation
    ran; actual fused execution is independently checked by the capture gates.
    """
    require(integer(prompt_count, 1) and integer(output_count, 1)
            and integer(computed_tokens), 'Invalid AR commit counts')
    total = prompt_count + output_count
    if computed_tokens == total:
        return {'name': 'terminal_output_forwarded',
                'computed_tokens_formula': 'prompt_count + output_count',
                'terminal_output_forwarded': True,
                'unforwarded_output_tokens': 0}
    if computed_tokens == total - 1:
        return {'name': 'terminal_output_not_forwarded',
                'computed_tokens_formula': 'prompt_count + output_count - 1',
                'terminal_output_forwarded': False,
                'unforwarded_output_tokens': 1}
    raise ValueError('Computed trajectory differs from both exact nonspeculative AR commit conventions')


def extract_requests(report, records, mode, eos):
    require(isinstance(records, list) and all(isinstance(r, dict) for r in records), 'Invalid JSONL record')
    rows = report.get(mode)
    require(isinstance(rows, list) and rows, 'Missing actual HTTP request rows')
    require(isinstance(eos, list) and eos and all(integer(v) for v in eos), 'Missing checkpoint EOS IDs')
    seen_http, seen_internal, exports = set(), set(), []
    for row in rows:
        http_id = row.get('request_id')
        session = row.get('session', {}).get('sessionId')
        require(isinstance(http_id, str) and http_id and http_id not in seen_http, 'Missing/duplicate HTTP identity')
        require(isinstance(session, str) and session, 'Missing session identity')
        seen_http.add(http_id)
        require(row.get('completed') is True and row.get('status') == 200 and not row.get('error')
                and not row.get('cleanup_error'), 'HTTP request did not complete cleanly')
        body, terminal = row.get('request', {}), row.get('terminal', {})
        require(body.get('sessionId') == session and terminal.get('sessionId') == session, 'Request/session mapping differs')
        require(terminal.get('done') is True and terminal.get('aborted') is False and not terminal.get('error'), 'Invalid terminal HTTP state')
        require(body.get('think') is False and all(type(body.get(key)) in (int, float) and body[key] == value
                for key, value in (('temperature', 0), ('topP', 1), ('topK', 0), ('repeatPenalty', 1))),
                'Only explicit greedy nonthinking capture is supported')
        require(integer(body.get('maxTokens'), 1) and integer(body.get('seed'), -2147483648), 'Invalid generation limits/seed')
        messages = body.get('messages')
        require(isinstance(messages, list) and messages and all(isinstance(m, dict)
            and set(m) <= {'role', 'content'} and m.get('role') in ('user', 'system', 'assistant', 'developer')
            and isinstance(m.get('content'), str) for m in messages), 'Only text request history can be an independent raw-token oracle')
        matches = [(i, r) for i, r in enumerate(records, 1) if r.get('scope', {}).get('RequestId') == http_id]
        starts = [(i, r) for i, r in matches if r.get('event', {}).get('name') == 'HttpRequestStarted']
        ends = [(i, r) for i, r in matches if r.get('event', {}).get('name') == 'HttpRequestCompleted']
        require(len(starts) == len(ends) == 1, 'Missing/duplicate HTTP boundaries')
        for _, boundary in starts + ends:
            props = boundary.get('props', {})
            require(props.get('Method') == 'POST' and props.get('Path') == '/api/chat', 'Unexpected HTTP boundary')
        require(ends[0][1]['props'].get('StatusCode') == 200, 'Unsuccessful HTTP boundary')
        selected = {}
        for name in EVENTS:
            events = [(i, r) for i, r in matches if event_is(r, name)]
            # Fail closed for child/tool rounds, rather than choosing the first
            # generation or silently treating inherited HTTP scope as the root.
            require(len(events) == 1, 'Missing/duplicate/multiple generation event: ' + name)
            selected[name] = events[0]
        input_line, input_record = selected[EVENTS[0]]
        output_line, output_record = selected[EVENTS[1]]
        timing_line, timing_record = selected[EVENTS[2]]
        require(starts[0][0] < input_line < timing_line < output_line < ends[0][0], 'Generation event ordering differs')
        ordered = [starts[0][1], input_record, timing_record, output_record, ends[0][1]]
        times = [instant(r['ts']) for r in ordered]
        require(times == sorted(times), 'Reversed generation timestamps')
        ip, op, tp = (r.get('props', {}) for r in (input_record, output_record, timing_record))
        internal = ip.get('InternalRequestId')
        require(isinstance(internal, str) and internal and internal not in seen_internal, 'Missing/duplicate internal identity')
        seen_internal.add(internal)
        require(op.get('InternalRequestId') == tp.get('InternalRequestId') == internal
                and ip.get('SessionId') == tp.get('SessionId') == session, 'Internal/session mapping differs')
        require(ip.get('Architecture') == 'qwen4exp' and ip.get('HistoryCompacted') is False
                and ip.get('Thinking') is False, 'Unsupported architecture, compacted input, or thinking')
        require(integer(ip.get('MaxNewTokens'), 1) and ip['MaxNewTokens'] == body['maxTokens'], 'Effective max token count differs')
        prompt, output = token_csv(ip.get('Tokens')), token_csv(op.get('Tokens'))
        require(integer(terminal.get('promptTokens')) and terminal['promptTokens'] == len(prompt)
                and integer(tp.get('PromptTokens')) and tp['PromptTokens'] == len(prompt), 'Prompt count differs')
        require(tp.get('EngineCompletionAvailable') is True and integer(tp.get('EngineOutputTokens'))
                and tp['EngineOutputTokens'] == len(output), 'Engine output count differs or is unavailable')
        reason = op.get('FinishReason')
        require(reason == tp.get('FinishReason') and reason in ('eos', 'max_tokens'), 'Unexpected/mismatched finish reason')
        require(len(output) <= ip['MaxNewTokens'] and not any(t in eos for t in output[:-1]), 'Output count/early EOS differs')
        natural = reason == 'eos'
        require((natural and output[-1] in eos) or (not natural and output[-1] not in eos and len(output) == ip['MaxNewTokens']),
                'Terminal EOS/length does not agree with actual IDs')
        published = len(output) - int(natural)
        require(integer(tp.get('GeneratedTokens')) and tp['GeneratedTokens'] == published
                and integer(terminal.get('tokenCount')) and terminal['tokenCount'] == published
                and terminal.get('truncated') is (not natural), 'Published count/stop state differs')
        commit = ar_commit_convention(len(prompt), len(output), op.get('ComputedTokens'))
        exports.append({'name': row.get('name'), 'http_request_id': http_id, 'session_id': session,
            'internal_request_id': internal, 'architecture': 'qwen4exp', 'history_compacted': False,
            'thinking': False, 'max_new_tokens': ip['MaxNewTokens'], 'request': body,
            'prompt': {'tokens': prompt, 'count': len(prompt), 'i32le_sha256': token_hash(prompt)},
            'output': {'tokens': output, 'count': len(output), 'i32le_sha256': token_hash(output),
                       'finish_reason': reason, 'natural_eos': natural, 'computed_tokens': op['ComputedTokens'],
                       'ar_commit_convention': commit},
            'original_visible_text': row.get('text'), 'log_lines': {'input': input_line, 'output': output_line, 'timing': timing_line}})
    return exports


def identity_gates(audit, model, baseline):
    require(audit.get('cleanupComplete') is True, 'Owned server cleanup is not complete')
    before, after = audit.get('binariesBefore'), audit.get('binariesAfter')
    required = {'GgmlOps.dll', 'TensorSharp.Chat.dll', 'TensorSharp.Runtime.dll', 'TensorSharp.Models.dll',
                'TensorSharp.AgentHost.dll', 'TensorSharp.Server.Host.dll'}
    require(isinstance(before, dict) and required <= set(before) and before == after
            and all(isinstance(v, str) and SHA.fullmatch(v) for v in before.values()), 'Missing/changed binary snapshots')
    require(audit.get('childEnvironment', {}).get('TS_VALIDATION_TOKEN_TRACE') == '1', 'Actual token trace was not enabled')
    command, owner = audit.get('command'), audit.get('owned', {})
    require(isinstance(command, list) and command and path_key(command[0]) == path_key(owner.get('executable'))
            and integer(owner.get('pid'), 1) and type(owner.get('startTicks')) is int and owner['startTicks'] > 0, 'Missing owned executable/process identity')
    require(path_key(audit.get('loadedNative')) == path_key(str(Path(command[0].replace('\\', '/')).parent / 'GgmlOps.dll')), 'Loaded native path differs')
    require(command.count('--model') == 1 and command.index('--model') + 1 < len(command), 'Missing model entrypoint')
    shards = baseline['checkpoint']['shards']
    require(shards and path_key(command[command.index('--model') + 1]) == path_key(shards[0]['path']), 'Model entrypoint differs from baseline checkpoint')
    require(model.get('loaded') == Path(shards[0]['path'].replace('\\', '/')).name
            and model.get('architecture') == 'qwen4exp' and model.get('loadedBackend') == 'ggml_cuda', 'Loaded model identity differs')
    files = audit.get('modelFiles', [])
    require(isinstance(files, list) and len({path_key(f.get('FullName')) for f in files}) == len(files), 'Duplicate model file metadata')
    by_path = {path_key(f['FullName']): f for f in files}
    for shard in shards:
        row = by_path.get(path_key(shard['path']), {})
        require(isinstance(shard.get('sha256'), str) and SHA.fullmatch(shard['sha256'])
            and integer(shard.get('bytes'), 1, 1 << 63) and type(shard.get('mtime_ns')) is int,
            'Missing pinned checkpoint identity')
        require(row.get('Length') == shard['bytes'] and instant(row.get('LastWriteTimeUtc')) == shard['mtime_ns'],
                'Model size/mtime differs from historical full-hash checkpoint')
    return {'binaries': before, 'loaded_native': audit['loadedNative'], 'owned': owner,
            'checkpoint': baseline['checkpoint'], 'wrapper_passed': audit.get('passed'),
            'wrapper_errors': audit.get('errors'), 'scope': 'Wrapper before/after binary hashes and actual loaded native path; model sizes/mtimes linked to historical checkpoint SHA, not fresh checkpoint content hashing.'}


def report_log_pin(report, case_directory):
    if case_directory == 'concurrency':
        return report.get('fused_evidence', {}).get('server_log', {})
    require(case_directory in ('serial-three', 'three-chat'), 'Unsupported capture case directory')
    return report.get('server_log', {})


def three_chat_capture_gates(audit, report, records):
    """Check the complete cancelled-parent capture before selecting any sibling."""
    rows = report.get('concurrent')
    require(isinstance(rows, list) and len(rows) == 3 and all(isinstance(r, dict) for r in rows)
            and [r.get('name') for r in rows] == ['chat-0', 'chat-1', 'chat-2'],
            'Three-chat capture requires the three original ordered request rows')
    ids, sessions = [], []
    for row in rows:
        http_id, session = row.get('request_id'), row.get('session', {}).get('sessionId')
        require(isinstance(http_id, str) and http_id and isinstance(session, str) and session
                and row.get('request', {}).get('sessionId') == session,
                'Missing/mismatched original three-chat HTTP/session identity')
        ids.append(http_id); sessions.append(session)
    require(len(set(ids)) == len(set(sessions)) == 3, 'Duplicate original three-chat identities')
    require(report.get('passed') is True and report.get('completed') is True and not report.get('error')
            and not report.get('cleanup_errors') and report.get('args', {}).get('expect_stall') is False,
            'Three-chat acceptance did not complete successfully')
    checker_sha = hashlib.sha256(THREE_CHAT_CHECKER.read_bytes()).hexdigest()
    require(report.get('tool_sha256') == audit.get('threeChatCheckerSha') == checker_sha,
            'Three-chat checker differs from the owned wrapper or current validator')
    require(audit.get('childEnvironment', {}).get('TS_BATCHED_FUSED_TRACE') == '1',
            'Three-chat native graph trace was not enabled')
    queue = report.get('final_queue', {})
    require(queue.get('busy') is False and queue.get('pending_requests') == 0
            and type(queue.get('pending_requests')) is int, 'Three-chat queue did not drain')
    cancelled = report.get('cancel_at')
    require(type(cancelled) in (int, float) and math.isfinite(cancelled)
            and rows[0].get('client_cancelled_at') == cancelled and rows[0].get('completed') is False,
            'First-chat cancellation identity/state is missing')
    # Recompute engagement rather than trusting a copied passed/count field.
    spec = importlib.util.spec_from_file_location('token_oracle_three_chat_checker', THREE_CHAT_CHECKER)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    native = checker.native_proof(rows, records)
    require(native.get('passed') is True and native == report.get('native')
            and len(native.get('bindings', {})) == 3,
            'Three-chat native N3/N2 proof differs from the actual captured log')
    evidence = [r for r in records if event_is(r, 'generation.timings')
        and r.get('scope', {}).get('RequestId') == ids[0]
        and r.get('props', {}).get('FinishReason') == 'cancelled']
    require(len(evidence) == 1 and evidence == report.get('cancel_server_evidence'),
            'First-chat server cancellation proof differs from the captured log')
    for row in rows:
        matching = [r for r in records if r.get('scope', {}).get('RequestId') == row['request_id']]
        selected = [[r for r in matching if event_is(r, name)] for name in EVENTS]
        require(all(len(group) == 1 for group in selected),
                'Missing/duplicate/multiple original three-chat root generation')
        ip, op, tp = (group[0].get('props', {}) for group in selected)
        internal = ip.get('InternalRequestId')
        require(native['bindings'].get(internal) == row['request_id']
                and internal == op.get('InternalRequestId') == tp.get('InternalRequestId')
                and row['session']['sessionId'] == ip.get('SessionId') == tp.get('SessionId'),
                'Original three-chat root/internal/session binding differs')
    first_matches = [(n, r) for n, r in enumerate(records)
                     if r.get('scope', {}).get('RequestId') == ids[0]]
    boundaries = [[(n, r) for n, r in first_matches if r.get('event', {}).get('name') == name]
                  for name in ('HttpRequestStarted', 'HttpRequestCompleted')]
    require(all(len(group) == 1 for group in boundaries), 'Missing/duplicate cancelled HTTP boundaries')
    first_generation = [[(n, r) for n, r in first_matches if event_is(r, name)] for name in EVENTS]
    # EVENTS stores input/output/timings; the actual stream order is input,
    # timings, output, including cancellation's final trace publication.
    ordered = [boundaries[0][0], first_generation[0][0], first_generation[2][0],
               first_generation[1][0], boundaries[1][0]]
    require([n for n, _ in ordered] == sorted(n for n, _ in ordered)
            and [instant(r['ts']) for _, r in ordered] == sorted(instant(r['ts']) for _, r in ordered),
            'Cancelled first-chat event ordering differs')
    require(all(r.get('props', {}).get('Method') == 'POST'
                and r.get('props', {}).get('Path') == '/api/chat' for group in boundaries for _, r in group)
            and boundaries[1][0][1]['props'].get('StatusCode') == 200,
            'Cancelled first-chat HTTP boundary differs')
    require(first_generation[1][0][1]['props'].get('FinishReason') == 'cancelled',
            'Cancelled first-chat output trace has a different finish reason')
    return {'checker_sha256': checker_sha, 'n3_steps': native['n3_steps'],
        'n2_sibling_steps': native['n2_sibling_steps'], 'cancellation_line': native['cancellation_line'],
        'first_chat_cancellation': evidence[0],
        'original_requests': [{'name': r['name'], 'http_request_id': r['request_id'],
                               'session_id': r['session']['sessionId']} for r in rows]}


def extract_capture_requests(report, records, mode, eos, case_directory, request_name=None):
    require(request_name is None or case_directory == 'three-chat',
            'Request selection is supported only for the explicit three-chat schema')
    if case_directory != 'three-chat' or request_name is None:
        # The normal completion gate deliberately rejects the cancelled first
        # row if a caller attempts to export the entire three-chat capture.
        return extract_requests(report, records, mode, eos)
    require(request_name in ('chat-1', 'chat-2'), 'Only completed three-chat siblings may be selected')
    siblings = dict(report)
    siblings[mode] = report[mode][1:]
    exported = extract_requests(siblings, records, mode, eos)
    # Both surviving roots must be clean; filtering cannot hide a sibling's
    # duplicate tool round, missing trace, media, compaction or bad EOS count.
    return [row for row in exported if row['name'] == request_name]


def bind_run_evidence(audit, model, report, records, mode, case_directory='concurrency'):
    """Bind the captured log to this deployment and this owned lifetime."""
    command, owner = audit['command'], audit['owned']
    server = Path(command[0].replace('\\', '/')).parent
    log = report_log_pin(report, case_directory)
    three_chat = None
    if case_directory == 'three-chat':
        require(mode == 'concurrent', 'Three-chat capture requires concurrent mode')
        three_chat = three_chat_capture_gates(audit, report, records)
    if case_directory == 'serial-three':
        require(mode == 'serial' and isinstance(report.get('serial'), list) and len(report['serial']) == 3,
                'Serial three-chat capture requires exactly three serial requests')
        require(isinstance(report.get('tool_sha256'), str) and SHA.fullmatch(report['tool_sha256'])
                and report['tool_sha256'] == audit.get('serialThreeCheckerSha'),
                'Serial three-chat checker identity differs from owned wrapper')
        require(isinstance(report.get('concurrent_report_sha256'), str)
                and SHA.fullmatch(report['concurrent_report_sha256'])
                and report['concurrent_report_sha256'] == audit.get('concurrentReportSha'),
                'Serial control reference differs from owned wrapper')
    log_path = log.get('path')
    require(isinstance(log_path, str) and log_path, 'Missing original deployment log path')
    log_file = Path(log_path.replace('\\', '/'))
    require(path_key(str(log_file.parent)) == path_key(str(server / 'logs'))
            and log_file.name.startswith('tensorsharp-server-') and log_file.suffix == '.jsonl',
            'Captured JSONL path is not the owned deployment log')
    require(path_key(report.get('args', {}).get('server_log')) == path_key(log_path),
            'HTTP checker log path differs from captured log')
    for key in ('loaded', 'architecture', 'loadedBackend'):
        require(report.get('model', {}).get(key) == model.get(key), 'HTTP served model identity differs: ' + key)
    require(command.count('--port') == 1 and command.index('--port') + 1 < len(command), 'Missing owned HTTP port')
    require(report.get('args', {}).get('url') == 'http://127.0.0.1:' + command[command.index('--port') + 1],
            'HTTP checker endpoint differs from owned deployment')
    launch = instant(audit.get('startedUtc'))
    created = (owner['startTicks'] - 621355968000000000) * 100
    finished = instant(audit.get('finishedUtc'))
    require(launch <= created < finished, 'Invalid owned process lifetime')
    require(created <= instant(report.get('started_utc')) <= finished, 'HTTP report predates owned process or follows cleanup')
    targets = {row['request_id'] for row in report[mode]}
    for record in records:
        if record.get('scope', {}).get('RequestId') in targets:
            require(created <= instant(record.get('ts')) <= finished,
                    'HTTP/trace event is outside owned process lifetime')
    result = {'log_path': log_path, 'process_created_unix_ns': created,
            'wrapper_finished_unix_ns': finished, 'http_endpoint': report['args']['url'],
            'scope': 'Captured original log path and all selected HTTP-scope timestamps are bound to the wrapper deployment, PID/startTicks, endpoint and lifetime.'}
    if three_chat is not None:
        result['three_chat_capture'] = three_chat
    return result


def make_plan(baseline, exported, evidence_pin, index, max_new, port):
    require(integer(max_new, 1, 2048) and integer(port, 1024, 65535), 'Invalid baseline output/port bounds')
    plan = copy.deepcopy(baseline)
    plan.update(schema_version=2, status='prepared-not-executed', case='free_text', record_only=True,
                semantic_pass=None, generated_utc=datetime.now(timezone.utc).isoformat())
    plan['prompt'] = copy.deepcopy(exported['prompt'])
    body = dict(prompt=plan['prompt']['tokens'], n_predict=max_new, temperature=0.0, samplers=['temperature'],
        top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, repeat_last_n=0, presence_penalty=0.0,
        frequency_penalty=0.0, dry_multiplier=0.0, cache_prompt=False, return_tokens=True,
        n_probs=0, stream=False, ignore_eos=False, stop=[], seed=exported['request']['seed'])
    plan['futureRequest'] = {'endpoint': f'http://127.0.0.1:{port}/completion', 'body': body}
    command = plan['futureCommand']
    for key, value in (('--port', port), ('--ctx-size', max(2048, ((len(body['prompt']) + max_new + 511) // 512) * 512))):
        require(command.count(key) == 1 and command.index(key) + 1 < len(command), 'Missing baseline option ' + key)
        command[command.index(key) + 1] = str(value)
    # Do not carry the old task's semantic claims, raw-request paths or gates.
    keep = {'schema_version', 'status', 'case', 'record_only', 'semantic_pass', 'generated_utc', 'source',
            'deployment', 'checkpoint', 'prompt', 'futureRequest', 'futureCommand', 'environment', 'expected_checkpoint_eos_ids'}
    plan = {k: v for k, v in plan.items() if k in keep}
    plan['deployment']['all_files_rehashed_now'] = False
    plan['conditioning_evidence'] = dict(evidence_pin, request_index=index, http_request_id=exported['http_request_id'])
    plan['reference_output'] = copy.deepcopy(exported['output'])
    plan['limitations'] = ['Actual Server input IDs, not independently rerendered visible text.',
        'Free text semantics/facts remain unassessed even if execution and trajectory pass.',
        'Capped output is a prefix observation, not a completed answer.',
        'CPU experts mmap and CUDA dense/KV placement is a quality control, not a matched-budget speed comparison.',
        'Source/deployment/model validation is repeated before launch; historical complete checkpoint hashes require unchanged size/mtime.']
    return plan


def prepare(run, baseline_path, mode, output, max_new=256, port=52178, case_directory='concurrency', request_name=None):
    run, output = Path(run), Path(output)
    require(not output.exists(), 'Fresh output directory required')
    require(case_directory in ('concurrency', 'serial-three', 'three-chat'), 'Unsupported capture case directory')
    pins = []
    def pinned(path):
        data, pin = load(path); pins.append(pin); return data
    audit, model = pinned(run / 'launch-audit.json'), pinned(run / 'initial-model.json')
    report = pinned(run / case_directory / 'report.json')
    log_name = 'fused-server-log.jsonl' if case_directory == 'concurrency' else 'server-log.jsonl'
    raw, log_pin = read(run / case_directory / log_name); pins.append(log_pin)
    require(report_log_pin(report, case_directory).get('sha256') == log_pin['sha256'],
            'Captured JSONL digest differs from HTTP report')
    records = [strict_json(line) for line in raw.splitlines() if line.strip()]
    if case_directory == 'three-chat':
        for name in ('unified-three-chat-regression.py', 'unified-qwen-concurrency-bench.py',
                     'verify-fused-batch-evidence.py'):
            pins.append(read(Path(__file__).with_name(name))[1])
    baseline = pinned(baseline_path)
    source = baseline['source']
    require(subprocess.check_output(['git', '-C', source['path'], 'rev-parse', 'HEAD'], text=True).strip() == source['revision']
            and not subprocess.check_output(['git', '-C', source['path'], 'status', '--porcelain'], text=True).strip(),
            'llama source is not the clean pinned revision')
    identity = identity_gates(audit, model, baseline)
    identity['log_lifetime_binding'] = bind_run_evidence(audit, model, report, records, mode, case_directory)
    exports = extract_capture_requests(report, records, mode, baseline['expected_checkpoint_eos_ids'],
                                       case_directory, request_name)
    for pin in pins:
        require(read(pin['path'])[1]['sha256'] == pin['sha256'], 'Evidence changed during extraction')
    result = {'schema_version': 1, 'status': 'extracted-not-baseline-executed', 'semantic_pass': None,
        'mode': mode, 'case_directory': case_directory, 'request_name': request_name,
        'source_pins': pins, 'identity': identity, 'requests': exports,
        'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Exact raw Server SequenceState prompt/output IDs; one completed root generation per HTTP request. No factual judgment or independent model execution.'}
    output.mkdir(parents=True, exist_ok=False)
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    evidence_path = output / 'tokens.json'; save(evidence_path, result)
    evidence_pin = read(evidence_path)[1]
    for index, exported in enumerate(exports):
        plan = make_plan(baseline, exported, evidence_pin, index, max_new, port)
        save(output / f'plan-{index}.json', plan)
        save(output / f'request-{index}.json', plan['futureRequest']['body'])
    return {'path': str(evidence_path.resolve()), 'sha256': evidence_pin['sha256'], 'requests': len(exports), 'semantic_pass': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--baseline-plan', required=True, type=Path)
    parser.add_argument('--mode', choices=('serial', 'concurrent'), default='serial')
    parser.add_argument('--case-directory', choices=('concurrency', 'serial-three', 'three-chat'), default='concurrency',
                        help='Owned capture schema; three-chat requires complete cancelled-parent and N3/N2 evidence')
    parser.add_argument('--request-name', choices=('chat-1', 'chat-2'),
                        help='Explicit completed sibling from three-chat; chat-0 is never a clean completion')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-new', type=int, default=256)
    parser.add_argument('--port', type=int, default=52178)
    args = parser.parse_args()
    print(json.dumps(prepare(args.run, args.baseline_plan, args.mode, args.output, args.max_new, args.port,
                             args.case_directory, args.request_name)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
