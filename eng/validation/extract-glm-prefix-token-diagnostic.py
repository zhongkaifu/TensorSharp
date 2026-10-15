#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""Read-only exact-ID diagnostic for a completed owned GLM prefix Host run.

This tool does not execute models or rehash DLL/GGUF payloads.
It preserves the original failed visible-output gate. Successful extraction
establishes event linkage and token/count observations, not state equality,
quality, performance, cross-engine parity, or default readiness.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

ORACLE = Path(__file__).resolve().with_name('prepare-server-token-oracle.py')
MAX_BYTES = 128 << 20
NAMES = ('ff7-first', 'ff7-repeat', 'ff7-repeat-client-followup', 'ff7-first-client-followup')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    size = path.stat().st_size
    require(0 < size <= MAX_BYTES, 'Evidence input exceeds bounded size: ' + str(path))
    raw = path.read_bytes()
    require(len(raw) == size, 'Evidence input changed while reading: ' + str(path))
    return raw, {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


CONTROLLER_BEFORE = pin(__file__)[1]
ORACLE_BEFORE = pin(ORACLE)[1]
spec = importlib.util.spec_from_file_location('token_oracle_diagnostic_helpers', ORACLE)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
require(pin(ORACLE)[1] == ORACLE_BEFORE, 'Oracle helper changed while importing')


class Evidence:
    def __init__(self):
        self.pins = {}

    def raw(self, path, expected=None):
        raw, actual = pin(path)
        if expected is not None:
            require(helpers.path_key(expected.get('path')) == helpers.path_key(actual['path']), 'Evidence pin path differs')
            require(expected.get('sha256', '').lower() == actual['sha256'], 'Evidence SHA differs: ' + actual['path'])
            require(expected.get('bytes', actual['bytes']) == actual['bytes'], 'Evidence byte count differs')
        key = helpers.path_key(actual['path'])
        require(key not in self.pins or self.pins[key] == actual, 'Evidence changed between reads')
        self.pins[key] = actual
        return raw

    def load(self, path, expected=None):
        return helpers.strict_json(self.raw(path, expected).decode('utf-8-sig'))

    def bookend(self):
        after = {key: pin(value['path'])[1] for key, value in self.pins.items()}
        require(after == self.pins, 'Bounded evidence/source bookends changed')
        require(pin(__file__)[1] == CONTROLLER_BEFORE and pin(ORACLE)[1] == ORACLE_BEFORE,
                'Diagnostic controller/helper changed')
        return after


def value_after(arguments, flag):
    require(isinstance(arguments, list) and arguments.count(flag) == 1, 'Missing/duplicate argument ' + flag)
    offset = arguments.index(flag) + 1
    require(offset < len(arguments) and isinstance(arguments[offset], str), 'Missing argument value ' + flag)
    return arguments[offset]


def absolute_capture_path(value):
    """Validate archived paths lexically without touching later-growing logs."""
    require(isinstance(value, str) and value and '\x00' not in value, 'Missing canonical capture path')
    normalized = value.replace('\\', '/')
    path = (PureWindowsPath(value) if re.match(r'^[A-Za-z]:/', normalized)
            or normalized.startswith('//') else PurePosixPath(normalized))
    require(path.is_absolute() and '..' not in path.parts, 'Capture path must be absolute without parent traversal')
    return path


def validate_log_capture(identity, launch_source, http_source, snapshot, expected_snapshot, raw):
    """Bind an immutable single/rotated snapshot, never re-read live sources.

    Canonical family constraints match prefix-warmup-http.logger_paths. Each
    source SHA is independently verified against its exact byte range in the
    combined snapshot, whose producer concatenates sources without separators.
    """
    require(isinstance(identity, dict), 'Missing HTTP structured-log capture')
    source = absolute_capture_path(launch_source)
    require(helpers.path_key(str(source)) == helpers.path_key(str(absolute_capture_path(http_source))),
            'Host/HTTP explicit log input differs')
    actual_snapshot = absolute_capture_path(str(snapshot))
    require(helpers.path_key(str(actual_snapshot)) == helpers.path_key(str(absolute_capture_path(str(expected_snapshot)))),
            'Require exact HTTP output/server-log.jsonl snapshot')
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_BYTES, 'Invalid bounded snapshot payload')

    def checked_pin(value):
        require(isinstance(value, dict) and helpers.integer(value.get('bytes'), 0, MAX_BYTES)
                and isinstance(value.get('sha256'), str) and helpers.SHA.fullmatch(value['sha256']),
                'Invalid bounded source/snapshot pin')
        return absolute_capture_path(value.get('path'))

    if 'schema' not in identity:
        original = checked_pin(identity)
        require(helpers.path_key(str(original)) == helpers.path_key(str(source)), 'Single source log identity differs')
        require(identity['bytes'] == len(raw) and identity['sha256'].lower() == hashlib.sha256(raw).hexdigest(),
                'Single source snapshot payload differs')
        return {'kind': 'single-file', 'source_count': 1, 'source_bytes': len(raw)}

    require(identity.get('schema') == 'tensorsharp.rotated-log-capture.v1', 'Unknown structured-log capture schema')
    require(helpers.path_key(str(absolute_capture_path(identity.get('explicit_input')))) == helpers.path_key(str(source)),
            'Rotated capture explicit input differs')
    combined = identity.get('combined_snapshot')
    combined_path = checked_pin(combined)
    require(helpers.path_key(str(combined_path)) == helpers.path_key(str(actual_snapshot))
            and combined['bytes'] == len(raw) and combined['sha256'].lower() == hashlib.sha256(raw).hexdigest(),
            'Combined snapshot pin differs')
    sources = identity.get('sources')
    require(isinstance(sources, list) and 2 <= len(sources) <= 32, 'Rotated capture requires bounded2..32 sources')
    pattern = r'(?P<prefix>.+)-(?P<day>[0-9]{8})(?:-(?P<index>[0-9]{4}))?\.jsonl'
    anchor = re.fullmatch(pattern, source.name)
    require(anchor is not None, 'Rotated capture requires a canonical logger anchor')
    try:
        start = datetime.strptime(anchor['day'], '%Y%m%d').date()
    except ValueError:
        raise ValueError('Invalid canonical logger UTC date') from None
    ordered, seen_paths, indices = [], set(), {}
    offset = 0
    for ordinal, original in enumerate(sources):
        path = checked_pin(original)
        key = helpers.path_key(str(path))
        require(key not in seen_paths and helpers.path_key(str(path.parent)) == helpers.path_key(str(source.parent)),
                'Duplicate source or canonical logger directory differs')
        seen_paths.add(key)
        match = re.fullmatch(pattern, path.name)
        require(match is not None and match['prefix'] == anchor['prefix'], 'Malformed canonical logger family member')
        try:
            day = datetime.strptime(match['day'], '%Y%m%d').date()
        except ValueError:
            raise ValueError('Invalid canonical logger UTC date') from None
        require(day in (start, start + timedelta(days=1)), 'Logger family exceeds adjacent2 UTC days')
        index = int(match['index'] or 0)
        require(match['index'] is None or index > 0, 'Canonical logger index0 must have no suffix')
        ordered.append((day, index))
        indices.setdefault(day, []).append(index)
        length = original['bytes']
        end = offset + length
        require(end <= len(raw), 'Source byte ranges exceed combined snapshot')
        require(hashlib.sha256(memoryview(raw)[offset:end]).hexdigest() == original['sha256'].lower(),
                'Source pin differs from its exact combined-snapshot byte range')
        require(ordinal == len(sources) - 1 or length == 0 or raw[end - 1] == 10,
                'Rotated prior file lacks complete trailing LF')
        offset = end
    require(offset == len(raw), 'Source byte sum differs from combined snapshot')
    require(ordered == sorted(ordered)
            and all(values == list(range(len(values))) for values in indices.values()),
            'Canonical logger ordering/rotation indices have gaps or duplicates')
    require(helpers.path_key(str(source)) in seen_paths, 'Explicit logger anchor is absent from source pins')
    return {'kind': 'rotated-files', 'source_count': len(sources), 'source_bytes': offset,
            'source_hashes_verified_against_snapshot_ranges': True,
            'live_source_files_reopened': False}


def compare_ids(left, right):
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    equal = left == right
    return {'equal': equal, 'left_count': len(left), 'right_count': len(right),
            'longest_common_prefix_tokens': common,
            'first_difference': None if equal else {'index': common,
                'left': left[common] if common < len(left) else None,
                'right': right[common] if common < len(right) else None},
            'left_i32le_sha256': helpers.token_hash(left), 'right_i32le_sha256': helpers.token_hash(right)}


def wire_without_session(row):
    return {key: value for key, value in row['request'].items() if key != 'sessionId'}


def extract(row, records, architecture, run_start, run_end, seen):
    name, http_id = row.get('name'), row.get('request_id')
    session = row.get('session', {}).get('sessionId')
    require(isinstance(http_id, str) and http_id and http_id not in seen['http'], 'Missing/duplicate HTTP identity')
    require(isinstance(session, str) and session, 'Missing session identity')
    seen['http'].add(http_id)
    require(row.get('completed') is True and row.get('status') == 200 and not row.get('error')
            and not row.get('cleanup_error') and not row.get('tool_frames_observed'), 'HTTP request is incomplete or includes tools')
    body, terminal = row.get('request', {}), row.get('terminal', {})
    require(body.get('sessionId') == terminal.get('sessionId') == session, 'HTTP session mapping differs')
    require(terminal.get('done') is True and terminal.get('aborted') is False and not terminal.get('error'), 'Invalid HTTP terminal state')
    require(body.get('think') is False and all(type(body.get(key)) in (int, float) and body[key] == value
        for key, value in (('temperature', 0), ('topP', 1), ('topK', 0), ('repeatPenalty', 1),
                           ('presencePenalty', 0), ('frequencyPenalty', 0))), 'Require explicit nonthinking greedy sampling')
    require(helpers.integer(body.get('seed'), -2147483648), 'Invalid sampling seed')
    require(helpers.integer(body.get('maxTokens'), 1), 'Invalid maxTokens')
    messages = body.get('messages')
    require(isinstance(messages, list) and messages and all(isinstance(m, dict)
            and set(m) <= {'role', 'content', 'thinking'} and m.get('role') in ('user', 'system', 'assistant', 'developer')
            and isinstance(m.get('content'), str) and isinstance(m.get('thinking', ''), str)
            for m in messages), 'Require plain production text/thinking history DTOs')
    matches = [(i, record) for i, record in enumerate(records, 1)
               if record.get('scope', {}).get('RequestId') == http_id]
    starts = [(i, r) for i, r in matches if r.get('event', {}).get('name') == 'HttpRequestStarted']
    ends = [(i, r) for i, r in matches if r.get('event', {}).get('name') == 'HttpRequestCompleted']
    require(len(starts) == len(ends) == 1, 'Missing/duplicate HTTP boundaries: ' + name)
    for _, boundary in starts + ends:
        require(boundary.get('props', {}).get('Method') == 'POST'
                and boundary.get('props', {}).get('Path') == '/api/chat', 'Unexpected HTTP route')
    require(ends[0][1]['props'].get('StatusCode') == 200, 'HTTP boundary status differs')
    selected = {}
    for label in helpers.EVENTS:
        hits = [(i, r) for i, r in matches if helpers.event_is(r, label)]
        require(len(hits) == 1, 'Missing/duplicate/multiple generation events: ' + name + '/' + label)
        selected[label] = hits[0]
    il, ir = selected['generation.validation.input']
    ol, orec = selected['generation.validation.output']
    tl, tr = selected['generation.timings']
    require(starts[0][0] < il < tl < ol < ends[0][0], 'Generation event order differs')
    times = [helpers.instant(r['ts']) for r in (starts[0][1], ir, tr, orec, ends[0][1])]
    require(times == sorted(times) and run_start <= times[0] <= times[-1] <= run_end, 'Generation timestamps differ')
    ip, op, tp = (r.get('props', {}) for r in (ir, orec, tr))
    internal = ip.get('InternalRequestId')
    require(isinstance(internal, str) and internal and internal not in seen['internal'], 'Missing/duplicate internal generation identity')
    seen['internal'].add(internal)
    require(internal == op.get('InternalRequestId') == tp.get('InternalRequestId')
            and ip.get('SessionId') == tp.get('SessionId') == session, 'Internal/HTTP/session linkage differs')
    require(ip.get('Architecture') == architecture and ip.get('HistoryCompacted') is False
            and ip.get('Thinking') is False, 'Architecture/compaction/effective thinking differs')
    require(ip.get('MaxNewTokens') == body['maxTokens'], 'Effective maxTokens differs')
    prompt, output = helpers.token_csv(ip.get('Tokens')), helpers.token_csv(op.get('Tokens'))
    require(terminal.get('promptTokens') == tp.get('PromptTokens') == len(prompt), 'Prompt token count differs')
    require(tp.get('EngineCompletionAvailable') is True and tp.get('EngineOutputTokens') == len(output), 'Engine output token count differs')
    # These are bounded length-stop diagnostics, not EOS trajectory exports.
    require(op.get('FinishReason') == tp.get('FinishReason') == 'max_tokens'
            and len(output) == body['maxTokens'] and terminal.get('truncated') is True,
            'Require the predeclared complete max_tokens trajectory')
    require(tp.get('GeneratedTokens') == terminal.get('tokenCount') == len(output), 'Published output count differs')
    cached = terminal.get('kvReusedTokens')
    require(helpers.integer(cached) and cached <= len(prompt) and tp.get('KvReusedTokens') == cached, 'Cached token linkage differs')
    usage = row.get('usage', {})
    require(usage.get('prompt_tokens') == len(prompt) and usage.get('completion_tokens') == len(output)
            and usage.get('prompt_tokens_details', {}).get('cached_tokens') == cached, 'HTTP usage linkage differs')
    commit = helpers.ar_commit_convention(len(prompt), len(output), op.get('ComputedTokens'))
    return {'name': name, 'http_request_id': http_id, 'internal_request_id': internal,
            'session_id': session, 'request': body, 'cached_tokens': cached,
            'prompt': {'tokens': prompt, 'count': len(prompt), 'i32le_sha256': helpers.token_hash(prompt)},
            'output': {'tokens': output, 'count': len(output), 'i32le_sha256': helpers.token_hash(output),
                       'finish_reason': op['FinishReason'], 'computed_tokens': op['ComputedTokens'],
                       'ar_commit_convention': commit},
            'text': row.get('text'), 'thinking': row.get('thinking'),
            'log_lines': {'http_start': starts[0][0], 'input': il, 'timing': tl, 'output': ol, 'http_end': ends[0][0]}}


def trajectory_check(opening, followup):
    require(opening['session_id'] == followup['session_id'], 'Followup conversation identity differs')
    require(opening['log_lines']['http_end'] < followup['log_lines']['http_start'], 'Followup precedes its opening')
    p, o = opening['prompt']['tokens'], opening['output']['tokens']
    computed = opening['output']['computed_tokens']
    consumed_output = computed - len(p)
    require(0 <= consumed_output <= len(o), 'Invalid consumed output count')
    history = p + o[:consumed_output]
    follow = followup['prompt']['tokens']
    cached = followup['cached_tokens']
    shared = min(len(history), len(follow))
    return {'opening': opening['name'], 'followup': followup['name'],
            'opening_actual_computed_head': computed, 'opening_output_consumed_count': consumed_output,
            'followup_cached_head': cached, 'cached_head_equals_opening_computed_head': cached == computed,
            'cached_head_beyond_opening_prompt': cached > len(p),
            'cached_head_within_observed_opening_history': cached <= len(history),
            'cached_prefix_matches_observed_consumed_history': history[:cached] == follow[:cached] if cached <= len(history) else None,
            'entire_observed_consumed_history_matches_followup_prefix': history == follow[:len(history)],
            'consumed_history_vs_followup_common_extent': compare_ids(history[:shared], follow[:shared]),
            'observed_consumed_history_i32le_sha256': helpers.token_hash(history),
            'uncomputed_followup_tail_tokens': len(follow) - cached,
            'scope': 'Counts and token-history linkage only. Cached-token telemetry is not independent proof of hidden KV tensor identity.'}


def parse_arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch-audit', required=True, type=Path)
    ap.add_argument('--http-report', required=True, type=Path)
    ap.add_argument('--server-log', required=True, type=Path, help='Original HTTP output/server-log.jsonl snapshot, not the later-growing live source log')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--expected-sdk-version', type=int, default=26)
    ap.add_argument('--expected-native-sha256', required=True)
    args = ap.parse_args(argv)
    # Receipt paths are absolute. Normalize the user-supplied paths before any
    # parent/path comparison so relative and absolute invocations are identical.
    for name in ('launch_audit', 'http_report', 'server_log', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv=None):
    args = parse_arguments(argv)
    require(not args.output.exists(), 'Output must be fresh; never overwrite diagnostic evidence')
    evidence = Evidence()
    audit = evidence.load(args.launch_audit)
    http = evidence.load(args.http_report)
    require(audit.get('schema') == 'tensorsharp.prefix-warmup-host.v1'
            and http.get('schema') == 'tensorsharp.prefix-warmup-http.v1', 'Unexpected capture schemas')
    require(audit.get('owned_server_exited') is True and audit.get('sampler_stopped') is True
            and audit.get('resource_observation_complete') is True and audit.get('watchdog_expired') is False
            and audit.get('cleanup_errors') == [] and audit.get('sampling_errors') == [], 'Incomplete owned process/resources/cleanup')
    descendants = audit.get('owned_descendant_cleanup', {})
    require(descendants.get('passed') is True and descendants.get('live_after') == [] and descendants.get('errors') == [],
            'Owned descendant cleanup did not pass')
    require(http.get('completed') is True and not http.get('error'), 'HTTP diagnostic did not finish')
    require(audit.get('passed') is True or (audit.get('passed') is False
            and audit.get('errors') == ['ValueError: HTTP prefix/visible consistency gate failed']),
            'Host failure includes infrastructure/unknown failure')
    require(audit.get('http_exit_code') == (0 if http.get('passed') is True else 1), 'Host/HTTP terminal exit differs')
    require(audit.get('qualification_before') == audit.get('qualification_after')
            and isinstance(audit.get('qualification_before'), dict), 'Staged closure/qualified source bookends differ')
    require(audit.get('model_before') == audit.get('model_after') and audit.get('model_before'), 'Model metadata bookends differ')
    require(audit.get('plan_pin') == audit.get('plan_pin_after'), 'Plan bookends differ')
    plan = evidence.load(audit['plan_pin']['path'], audit['plan_pin'])
    require(plan.get('sdk_version') == args.expected_sdk_version and plan.get('architecture') == 'glm5-next'
            and plan.get('validation_token_trace') is True, 'Wrong SDK/architecture or no explicit trace plan')
    require(audit.get('child_environment', {}).get('TS_VALIDATION_TOKEN_TRACE') == '1'
            and audit.get('token_trace_control', {}).get('enabled') is True, 'Actual token tracing not enabled')
    require(http.get('source_pins_before') == http.get('source_pins_after') and http.get('source_pins_before'), 'HTTP source bookends differ')
    for original in http['source_pins_before'].values():
        evidence.raw(original['path'], original)
    for original in plan.get('source_pins', {}).values():
        evidence.raw(original['path'], original)
    qualification = audit['qualification_before']
    native = audit.get('loaded_native', {})
    expected_native = args.expected_native_sha256.lower()
    require(helpers.SHA.fullmatch(expected_native) and qualification.get('native_sha256') == expected_native
            and native.get('sha256') == expected_native, 'Native binary receipt identity differs')
    owner = audit.get('owned_process', {})
    require(helpers.integer(owner.get('pid'), 1) and type(owner.get('birth')) in (int, float) and owner['birth'] > 0
            and audit.get('command') == owner.get('cmdline') and owner['cmdline']
            and helpers.path_key(owner['executable']) == helpers.path_key(owner['cmdline'][0]), 'Owned Host identity differs')
    require(helpers.path_key(str(Path(owner['executable']).parent / 'GgmlOps.dll')) == helpers.path_key(native.get('path')),
            'Loaded native is not the owned staged native path')
    require(audit.get('initial_model', {}).get('architecture') == 'glm5-next'
            and audit['initial_model'].get('loadedBackend') == 'ggml_cuda'
            and audit['initial_model'].get('unifiedMemory') is True, 'Actual loaded execution identity differs')
    receipt_pins = qualification.get('receipt_and_tool_pins', {})
    require(all(key in receipt_pins for key in ('delivery', 'native_summary', 'stage_summary', 'deployment', 'self_test')),
            'Missing completed deployment/source receipt linkage')
    # Metadata is bounded and read directly; DLL/GGUF payload integrity remains
    # the original completed launcher's snapshot, never an invented fresh hash.
    for key, original in receipt_pins.items():
        if Path(original['path']).suffix.lower() in ('.json', '.py', '.ps1'):
            evidence.raw(original['path'], original)
    delivery = evidence.load(receipt_pins['delivery']['path'], receipt_pins['delivery'])
    require(delivery.get('passed') is True and delivery.get('publication_complete') is True
            and delivery.get('cpu_version') == args.expected_sdk_version, 'SDK full publication not qualified')
    require(helpers.path_key(str(args.http_report.parent)) == helpers.path_key(value_after(audit['http_arguments'], '--output'))
            and helpers.path_key(str(args.http_report)) == helpers.path_key(str(args.launch_audit.parent / 'http/report.json'))
            and helpers.path_key(str(args.launch_audit.parent)) == helpers.path_key(plan.get('output_directory')),
            'Explicit HTTP/Host files are not the actual launch command targets')
    if audit.get('http_report_pin') is not None:
        evidence.raw(args.http_report, audit['http_report_pin'])
        http_binding = 'original Host receipt report pin'
    else:
        http_binding = 'retrospective bounded SHA; exact actual Host --output path and HTTP source/log/identity receipts; original failed Host has no report pin'
    log_pin = http.get('server_log')
    raw_snapshot = evidence.raw(args.server_log)
    log_binding = validate_log_capture(log_pin, value_after(audit['http_arguments'], '--server-log'),
        http.get('arguments', {}).get('server_log'), args.server_log,
        args.http_report.parent / 'server-log.jsonl', raw_snapshot)
    raw_log = raw_snapshot.decode('utf-8-sig')
    records = [helpers.strict_json(line) for line in raw_log.splitlines() if line.strip()]
    require(records and all(isinstance(record, dict) for record in records), 'Invalid JSONL records')
    start, end = helpers.instant(audit['started_utc']), helpers.instant(audit['finished_utc'])
    require(start <= helpers.instant(http['started_utc']) <= helpers.instant(http['finished_utc']) <= end, 'Capture timestamps differ')
    serial = http.get('serial')
    require(isinstance(serial, list) and all(isinstance(row, dict) for row in serial), 'Missing serial HTTP observations')
    selected = {}
    for name in NAMES:
        matches = [row for row in serial if row.get('name') == name]
        require(len(matches) == 1, 'Missing/duplicate predeclared request: ' + name)
        selected[name] = matches[0]
        actual = evidence.load(args.http_report.parent / name / 'result.json')
        require(actual == matches[0], 'Original per-request receipt differs from HTTP report: ' + name)
        actual_request = evidence.load(args.http_report.parent / name / 'request.json')
        require(actual_request.get('request') == actual['request']
                and actual_request.get('request_id') == actual['request_id']
                and actual_request.get('name') == name
                and actual_request.get('session') == actual['session'], 'Original HTTP request payload/identity differs: ' + name)
    require(wire_without_session(selected[NAMES[0]]) == wire_without_session(selected[NAMES[1]]), 'Opening wire request configuration differs')
    require(wire_without_session(selected[NAMES[2]]) == wire_without_session(selected[NAMES[3]]), 'Followup wire request configuration differs')
    seen = {'http': set(), 'internal': set()}
    rows = {name: extract(selected[name], records, 'glm5-next', start, end, seen) for name in NAMES}
    require(rows[NAMES[0]]['session_id'] != rows[NAMES[1]]['session_id'], 'Distinct opening chats share one session')
    checks = {'opening_prompt_ids': compare_ids(rows[NAMES[0]]['prompt']['tokens'], rows[NAMES[1]]['prompt']['tokens']),
              'opening_output_ids': compare_ids(rows[NAMES[0]]['output']['tokens'], rows[NAMES[1]]['output']['tokens']),
              'followup_prompt_ids': compare_ids(rows[NAMES[2]]['prompt']['tokens'], rows[NAMES[3]]['prompt']['tokens']),
              'followup_output_ids': compare_ids(rows[NAMES[2]]['output']['tokens'], rows[NAMES[3]]['output']['tokens'])}
    histories = [trajectory_check(rows[NAMES[1]], rows[NAMES[2]]), trajectory_check(rows[NAMES[0]], rows[NAMES[3]])]
    evidence.bookend()
    result = {'schema': 'tensorsharp.glm-prefix-raw-token-diagnostic.v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'extraction_completed': True,
              'quality_accepted': False, 'performance_qualified': False, 'default_readiness_claim': False,
              'scope': 'Raw prompt/output ID, exact AR commit-count and cached-boundary observations only; no hidden-state tensor equality or cause proven.',
              'original_host_passed': audit['passed'], 'original_host_errors': audit.get('errors'),
              'original_http_passed': http['passed'],
              'original_followup_gate': http.get('client_visible_follow_up'),
              'native_sha256_from_original_launch_snapshot': expected_native,
              'sdk_version': args.expected_sdk_version,
              'http_report_binding_scope': http_binding,
              'original_http_source_log_pin': log_pin,
              'log_snapshot_binding': log_binding,
              'log_snapshot_scope': 'Exact HTTP snapshot bytes/SHA from original source pin; live source may have subsequent shutdown records and is not substituted.',
              'model_metadata_from_original_launch_bookends': audit['model_before'],
              'controller_pin': CONTROLLER_BEFORE, 'helper_pin': ORACLE_BEFORE,
              'inputs_bookended': list(evidence.pins.values()), 'requests': list(rows.values()),
              'comparisons': checks, 'consumed_history_checks': histories}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'extraction_completed': True,
                      'original_quality_passed': http['passed'], 'comparisons': checks,
                      'consumed_history_checks': histories}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
