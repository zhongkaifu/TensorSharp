#!/usr/bin/env python3
"""Observe public-prefix reuse and bounded output consistency on a running Host.

Never starts/stops a host or changes its cache/think policy. Default catalogs
remain enabled. HTTP frames do not prove batched GPU execution or model quality.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

common = module('prefix_http_common', HERE / 'unified-qwen-concurrency-bench.py')
checker = module('prefix_http_timings', common.FUSED_CHECKER)
PROMPTS = ('请详细介绍最终幻想7', '请详细介绍时间简史', '请详细介绍最终幻想7')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def body_for(session_id, messages, think, max_tokens):
    # Omitted catalog/tool fields are intentional: this is the real new-chat
    # renderer, rather than the smaller prose-only benchmark request.
    return dict(sessionId=session_id, messages=copy.deepcopy(messages), newChat=False,
                maxTokens=max_tokens, think=think, temperature=0, topP=1, topK=0,
                repeatPenalty=1, presencePenalty=0, frequencyPenalty=0, seed=17)

def usage_of(terminal):
    require(isinstance(terminal, dict) and terminal.get('done') is True, 'Missing terminal frame')
    require(terminal.get('aborted') is False and not terminal.get('error'), 'Terminal abort/error')
    for name in ('promptTokens', 'tokenCount', 'kvReusedTokens'):
        require(type(terminal.get(name)) is int and terminal[name] >= (0 if name == 'kvReusedTokens' else 1), 'Invalid terminal ' + name)
    require(terminal['kvReusedTokens'] <= terminal['promptTokens'], 'Cached tokens exceed prompt tokens')
    require(type(terminal.get('truncated')) is bool, 'Missing explicit truncation state')
    return dict(prompt_tokens=terminal['promptTokens'], completion_tokens=terminal['tokenCount'],
                prompt_tokens_details=dict(cached_tokens=terminal['kvReusedTokens']),
                source='WebUI terminal fields; not tokenizer counts of SSE chunks')

def consume_frame(row, frame, now):
    require(isinstance(frame, dict) and not frame.get('error'), 'Invalid/error SSE frame')
    row['frames'].append([now, frame])
    raw = False
    for key in ('token', 'thinking'):
        value = frame.get(key)
        if value is not None:
            require(isinstance(value, str), 'Non-string ' + key)
        if value:
            raw = True
            if key == 'thinking':
                row['thinking'] += value
                if row.get('first_thinking_at') is None: row['first_thinking_at'] = now
            else:
                row['visible_frames'].append({'token': value})
                row['text'] += value
                if row['first_content_at'] is None: row['first_content_at'] = now
    if 'replace' in frame:
        require(isinstance(frame['replace'], str), 'Invalid content replacement')
        row['visible_frames'].append({'replace': frame['replace']})
        row['text'] = frame['replace']
    if raw:
        row['raw_frames'] += 1
        if row['first_raw_at'] is None: row['first_raw_at'] = now
    if any(key in frame for key in ('tool_calls', 'tool_progress', 'skill_step')):
        row['tool_frames_observed'] = True
    if frame.get('done') is True:
        require(not row['completed'], 'Duplicate terminal frame')
        row.update(completed=True, terminal=frame, terminal_at=now)
    return raw

def stream(client, output, name, session, body, cancel_after=None):
    folder = output / name
    folder.mkdir()
    row = dict(name=name, session=session, request=body, request_id='prefix-warmup-' + uuid.uuid4().hex,
               frames=[], visible_frames=[], raw_frames=0, text='', thinking='', completed=False,
               first_raw_at=None, first_content_at=None, first_thinking_at=None, tool_frames_observed=False, cancelled_as_requested=False)
    save(folder / 'request.json', row)
    conn, response, timer, sock = client.connect(), None, None, None
    expired = threading.Event()
    def expire():
        expired.set()
        if sock:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
    try:
        row['started'] = time.perf_counter()
        conn.connect(); sock = conn.sock
        timer = threading.Timer(client.timeout, expire); timer.daemon = True; timer.start()
        conn.request('POST', client.url.path.rstrip('/') + '/api/chat', json.dumps(body, ensure_ascii=False).encode('utf-8'),
                     {'Content-Type': 'application/json', 'Connection': 'close', 'X-TensorSharp-Request-Id': row['request_id']})
        response = conn.getresponse()
        row.update(status=response.status, headers_at=time.perf_counter())
        require(response.status == 200, 'Chat HTTP ' + str(response.status))
        require('text/event-stream' in response.getheader('Content-Type', ''), 'Chat did not return SSE')
        with (folder / 'response.sse').open('xb') as wire:
            total = 0
            while True:
                line = response.readline((1 << 20) + 1)
                if not line: break
                total += len(line)
                require(len(line) <= 1 << 20 and total <= 16 << 20, 'SSE evidence bound exceeded')
                wire.write(line)
                if not line.startswith(b'data:'): continue
                frame = json.loads(line[5:].strip().decode('utf-8'))
                raw = consume_frame(row, frame, time.perf_counter())
                if cancel_after is not None and raw and row['raw_frames'] >= cancel_after:
                    require(not row['completed'], 'Request finished before cancellation')
                    row['cancelled_as_requested'] = True
                    break
        require(not expired.is_set(), 'Absolute request deadline expired')
        if cancel_after is None:
            require(row['completed'] and row['raw_frames'] > 0, 'Missing terminal or nonempty model frames')
            require(row['terminal'].get('sessionId') == session['sessionId'], 'Terminal session mismatch')
            row['usage'] = usage_of(row['terminal'])
        else:
            require(row['cancelled_as_requested'], 'Cancellation was not exercised')
    except Exception as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if timer: timer.cancel()
        if response: response.close()
        conn.close()
        row['ended'] = time.perf_counter()
        row['client_first_raw_ttft_ms'] = None if row['first_raw_at'] is None else (row['first_raw_at'] - row['started']) * 1000
        row['client_first_content_ttft_ms'] = None if row['first_content_at'] is None else (row['first_content_at'] - row['started']) * 1000
        row['client_first_thinking_ttft_ms'] = None if row['first_thinking_at'] is None else (row['first_thinking_at'] - row['started']) * 1000
        row['client_wall_ms'] = (row['ended'] - row.get('started', row['ended'])) * 1000
        row['text_utf8_sha256'] = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
        save(folder / 'result.json', row)
    return row

def repeat_gate(first, repeated, timings):
    def reasons(row):
        return [entry['finish_reason'] for entry in timings.get('rows', []) if entry['http_request_id'] == row['request_id']]
    observed = bool(first['visible_frames'] and repeated['visible_frames'])
    checks = dict(nonempty_visible_observation=observed,
                  exact_visible_frames=first['visible_frames'] == repeated['visible_frames'],
                  exact_visible_text=first['text'] == repeated['text'],
                  same_prompt_tokens=first['usage']['prompt_tokens'] == repeated['usage']['prompt_tokens'],
                  same_completion_tokens=first['usage']['completion_tokens'] == repeated['usage']['completion_tokens'],
                  same_http_finish=all(first['terminal'].get(key) == repeated['terminal'].get(key) for key in ('truncated', 'aborted', 'error')),
                  same_measured_finish=bool(reasons(first)) and reasons(first) == reasons(repeated))
    return dict(passed=all(checks.values()), checks=checks,
                status='inconclusive_no_visible_output' if not observed else 'passed' if all(checks.values()) else 'failed',
                scope='Literal bounded visible frames/text, finish and usage; no raw token-ID/logit or full-answer semantic parity. Cached token counts may differ and are evaluated separately.')

def stream_repeat_gate(first, repeated, timings):
    """A distinct bounded thinking/content gate, never a visible-quality pass."""
    visible = repeat_gate(first, repeated, timings)
    def emitted(row):
        return [{key: frame[key] for key in ('token', 'thinking', 'replace')
                 if key in frame and (frame[key] or key == 'replace')}
                for _, frame in row['frames']
                if any(key in frame and (frame[key] or key == 'replace') for key in ('token', 'thinking', 'replace'))]
    checks = {key: value for key, value in visible['checks'].items() if key != 'nonempty_visible_observation'}
    checks.update(nonempty_model_observation=bool((first['text'] or first['thinking']) and (repeated['text'] or repeated['thinking'])),
                  exact_emitted_model_frames=emitted(first) == emitted(repeated),
                  exact_thinking=first['thinking'] == repeated['thinking'],
                  no_tool_frames=not first['tool_frames_observed'] and not repeated['tool_frames_observed'])
    return dict(passed=all(checks.values()), checks=checks, visible_quality_passed=visible['passed'],
        visible_quality_status=visible['status'],
        scope='Literal bounded actual emitted thinking/content frames and text, finish and usage. Thinking-only parity does not pass visible-output quality, semantic quality, raw token-ID or native-state identity.')

def production_history(history, session, prior):
    require(isinstance(history, dict) and history.get('sessionId') == session['sessionId']
            and history.get('source') == 'production-wire', 'History lacks exact production-wire session identity')
    messages = history.get('messages')
    require(isinstance(messages, list) and len(messages) >= 2 and all(isinstance(m, dict) for m in messages), 'Missing complete production wire messages')
    require(messages[-1].get('role') == 'assistant' and messages[-1].get('content') == prior['text'], 'Returned assistant content differs from actual stream')
    require(messages[-1].get('thinking', '') == prior['thinking'], 'Returned assistant thinking differs from actual stream')
    return copy.deepcopy(messages)

def client_visible_history(prior, timings):
    """Replay the actual emitted WebUI DTO; never manufacture raw output IDs.

    ParseWebUi consumes assistant role/content/thinking. It does not parse tool
    history, so this mode refuses all tool or unsupported metadata frames.
    ConversationTranscriptStore may restore recorded raw IDs internally; actual
    cached usage remains the only reuse observation made by this controller.
    """
    require(prior.get('completed') is True and not prior.get('error') and bool(prior.get('text')),
            'Client-visible replay requires a completed nonempty actual stream')
    usage_of(prior.get('terminal'))
    require(prior['terminal'].get('sessionId') == prior['session']['sessionId'], 'Client-visible replay session differs')
    require(not prior.get('tool_frames_observed'), 'Client-visible replay does not support tool history')
    original = prior['request'].get('messages')
    require(isinstance(original, list) and len(original) == 1 and set(original[0]) == {'role', 'content'}
            and original[0].get('role') == 'user' and isinstance(original[0].get('content'), str),
            'Client-visible replay supports one original text-only user message')
    terminal_keys = {'done', 'tokenCount', 'elapsed', 'tokPerSec', 'aborted', 'truncated', 'error',
                     'sessionId', 'promptTokens', 'kvReusedTokens', 'kvReusePercent'}
    frames = prior.get('frames', [])
    require(frames and frames[-1][1] == prior['terminal'], 'Client-visible replay requires the actual terminal to be last')
    for _, frame in frames:
        keys = set(frame)
        supported = keys in ({'token'}, {'thinking'}, {'replace'}, {'status'}) \
            or frame.get('done') is True and keys <= terminal_keys
        if keys == {'status'}: supported = isinstance(frame['status'], str)
        require(supported, 'Client-visible replay refuses unsupported frame metadata/tool history')
    linked = [entry for entry in timings.get('rows', []) if entry['http_request_id'] == prior['request_id']]
    require(len(linked) == 1 and linked[0]['engine_completion_available'], 'Client-visible replay requires exactly one measured engine round')
    messages = copy.deepcopy(original)
    messages.append(dict(role='assistant', content=prior['text'], thinking=prior['thinking']))
    return messages

def timing_linkage(rows, records):
    measured = checker.generation_timing_evidence({'serial': rows, 'concurrent': []}, records)
    matches = []
    for row in rows:
        linked = [entry for entry in measured['rows'] if entry['http_request_id'] == row['request_id']]
        usage = row['usage']
        # Multi-round tool work is retained. The sum is the WebUI whole-turn
        # usage, whereas each individual engine completion belongs to one round.
        matches.append(dict(name=row['name'], request_id=row['request_id'], rounds=len(linked),
            prompt_matches=sum(x['prompt_tokens'] for x in linked) == usage['prompt_tokens'],
            cached_matches=sum(x['kv_reused_tokens'] for x in linked) == usage['prompt_tokens_details']['cached_tokens'],
            completion_matches=sum(x['recorded_pipeline_tokens'] for x in linked) == usage['completion_tokens'],
            all_engine_completions_available=bool(linked) and all(x['engine_completion_available'] for x in linked)))
    passed = measured['all_current_requests_measured'] and all(all(row[k] for k in ('prompt_matches', 'cached_matches', 'completion_matches', 'all_engine_completions_available')) for row in matches)
    return dict(passed=passed, measurements=measured, usage_links=matches)

def logger_paths(path):
    """A bounded fresh logger family, covering UTC and numeric size rotation."""
    path = Path(path).resolve()
    pattern = r'(?P<prefix>.+)-(?P<day>[0-9]{8})(?:-(?P<index>[0-9]{4}))?\.jsonl'
    first = re.fullmatch(pattern, path.name)
    require(path.is_file(), 'Missing explicit structured log file')
    if first is None:
        require(not path.name.startswith('tensorsharp-server-'), 'Malformed canonical logger name')
        return [path]
    try: start = datetime.strptime(first['day'], '%Y%m%d').date()
    except ValueError: raise ValueError('Invalid canonical logger UTC date') from None
    candidates = list(path.parent.glob(first['prefix'] + '-*.jsonl'))
    require(1 <= len(candidates) <= 32, 'Canonical logger family exceeds bounded32 files')
    ordered, indices = [], {}
    for candidate in candidates:
        match = re.fullmatch(pattern, candidate.name)
        require(match is not None and match['prefix'] == first['prefix'], 'Malformed canonical logger family member')
        try: day = datetime.strptime(match['day'], '%Y%m%d').date()
        except ValueError: raise ValueError('Invalid canonical logger UTC date') from None
        require(day in (start, start + timedelta(days=1)), 'Logger family exceeds adjacent2 UTC days')
        resolved = candidate.resolve()
        require(resolved.parent == path.parent and resolved.is_file(), 'Canonical logger member escapes its directory')
        index = int(match['index'] or 0)
        require(match['index'] is None or index > 0, 'Canonical logger index0 must have no suffix')
        indices.setdefault(day, []).append(index)
        ordered.append((day, index, resolved))
    require(all(sorted(values) == list(range(len(values))) for values in indices.values()),
            'Canonical logger size-rotation indices have gaps or duplicates')
    result = [row[2] for row in sorted(ordered)]
    require(path in result, 'Explicit logger file disappeared from its family')
    return result


def read_records(path, output, filename='server-log.jsonl'):
    for attempt in range(10):
        paths = logger_paths(path)
        chunks, pins = [], []
        for source in paths:
            chunk, source_pin = checker.read_input(source, checker.MAX_LOG_BYTES)
            chunks.append(chunk); pins.append(source_pin)
        require(sum(map(len, chunks)) <= checker.MAX_LOG_BYTES, 'Combined structured log exceeds aggregate128MiB bound')
        require(all(not chunk or chunk.endswith(b'\n') for chunk in chunks[:-1]), 'Rotated structured log has an incomplete prior file')
        if logger_paths(path) != paths:
            require(attempt != 9, 'Logger family changed during every capture')
            time.sleep(.1)
            continue
        raw = b''.join(chunks)
        try:
            rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
            require(all(isinstance(row, dict) for row in rows), 'Non-object structured log record')
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            if attempt == 9: raise
            time.sleep(.1)
    snapshot = output / filename
    snapshot.write_bytes(raw)
    identity = pins[0] if len(pins) == 1 else dict(schema='tensorsharp.rotated-log-capture.v1',
        explicit_input=str(Path(path).resolve()), sources=pins, combined_snapshot=pin(snapshot),
        scope='Exact ordered canonical files, at most32 files/128MiB/adjacent2 UTC dates within the bounded controller lifetime. Request/session timing linkage is evaluated independently; no single original file or model acceptance is invented.')
    return rows, identity

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--provenance', type=Path)
    parser.add_argument('--think', choices=('on', 'off'), default='off')
    parser.add_argument('--output-gate', choices=('visible', 'stream'), default='visible',
                        help='Visible stays strict. Stream is a separate bounded thinking-on observation; absent visible output remains inconclusive.')
    parser.add_argument('--max-tokens', type=int, choices=(8, 16), default=16)
    parser.add_argument('--follow-up-max-tokens', type=int, choices=(8, 16, 64, 128),
                        help='Separate bounded continuation cap; initial new-chat max8/16 is unchanged.')
    parser.add_argument('--long-new-chat-max-tokens', type=int, choices=(256,),
                        help='Optional separate thinking-on FF7 new session, fully drained; original three requests stay unchanged.')
    parser.add_argument('--long-new-chat-expected-cached-tokens', type=int,
                        help='Exact positive public-prefix usage required for the optional new session; not private conversation reuse.')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--cache-expectation', choices=('enabled', 'disabled', 'observe'), default='enabled')
    followup_modes = parser.add_mutually_exclusive_group()
    followup_modes.add_argument('--follow-up', action='store_true')
    followup_modes.add_argument('--client-visible-follow-up', action='store_true')
    parser.add_argument('--history-route', help='Same-host route containing {sessionId}, returning {sessionId,source:"production-wire",messages:[...]}')
    parser.add_argument('--cancellation', action='store_true')
    parser.add_argument('--label', default='unspecified')
    args = parser.parse_args(argv)
    require(args.output_gate != 'stream' or args.think == 'on', 'Stream gate requires explicit thinking on')
    require(args.output_gate != 'stream' or not (args.follow_up or args.client_visible_follow_up),
            'Thinking stream scope does not qualify production-history continuation')
    require(args.follow_up_max_tokens is None or args.follow_up or args.client_visible_follow_up,
            'Followup token cap requires an explicitly requested followup')
    require((args.long_new_chat_max_tokens is None and args.long_new_chat_expected_cached_tokens is None)
            or args.long_new_chat_max_tokens == 256 and type(args.long_new_chat_expected_cached_tokens) is int
                and args.long_new_chat_expected_cached_tokens > 0,
            'Long new-chat probe requires bounded256 cap and an exact positive public cached-token expectation')
    require(args.long_new_chat_max_tokens is None or args.think == 'on' and args.output_gate == 'stream'
            and not (args.follow_up or args.client_visible_follow_up or args.cancellation),
            'Long new-chat probe is a separate thinking-on stream observation without followup/cancellation')
    follow_up_max_tokens = args.follow_up_max_tokens if args.follow_up_max_tokens is not None else args.max_tokens
    url = urlsplit(args.base_url)
    require(url.scheme in ('http', 'https') and url.hostname and not url.username and not url.query and not url.fragment, 'Invalid base URL')
    require(math.isfinite(args.timeout) and 1 <= args.timeout <= 1200, 'Invalid bounded timeout')
    output = args.output.resolve()
    require(not output.exists() and any(output.is_relative_to(ROOT / name) for name in ('artifacts', 'docs/validation')), 'Fresh ignored output directory required')
    if args.history_route:
        require(args.history_route.startswith('/') and '{sessionId}' in args.history_route, 'History route must be relative to the same host and contain {sessionId}')
    output.mkdir(parents=True)
    sources = [Path(__file__), HERE / 'unified-qwen-concurrency-bench.py', common.FUSED_CHECKER]
    source_pins = {str(path): pin(path) for path in sources}
    if args.provenance: source_pins[str(args.provenance)] = pin(args.provenance)
    client = common.Client(args.base_url, output, args.timeout)
    report = dict(schema='tensorsharp.prefix-warmup-http.v1', started_utc=datetime.now(timezone.utc).isoformat(),
        label=args.label, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_pins_before=source_pins, passed=False, completed=False, serial=[], sessions_cleanup=[],
        follow_up=dict(status='not_requested', passed=None), cancellation=dict(status='not_requested', passed=None),
        client_visible_follow_up=dict(status='not_requested', passed=None),
        long_new_chat=dict(status='not_requested', passed=None),
        gpu_parallelism_claim=False, model_semantic_quality_claim=False,
        scope='One live Host and dedicated owned sessions. Bounded max8/16 outputs; tools/catalog unchanged. Separate Host invocations supply think/cache controls and loaded-runtime provenance. No model reload/media, full semantic, token-ID, physical-hard-limit or shared-budget default-readiness acceptance.')
    sessions = []
    def new_session():
        value = client.json('POST', '/api/sessions')
        require(isinstance(value.get('sessionId'), str) and value['sessionId'] and value['sessionId'] not in [s['sessionId'] for s in sessions], 'Invalid/duplicate newly created session')
        sessions.append(value)
        return value
    try:
        report['model_before'] = client.json('GET', '/api/models')
        for index, prompt in enumerate(PROMPTS):
            session = new_session()
            body = body_for(session['sessionId'], [{'role': 'user', 'content': prompt}], args.think == 'on', args.max_tokens)
            row = stream(client, output, ('ff7-first', 'time-history', 'ff7-repeat')[index], session, body)
            report['serial'].append(row)
            require(not row.get('error') and row['completed'], 'Bounded new-chat request failed')
        if args.follow_up:
            report['follow_up'] = dict(status='inconclusive_failure', passed=False,
                reason='Exact production wire history is required; no transcript is reconstructed from partial model output')
            require(args.history_route is not None, 'Followup requires an actual production history route')
            follow_rows = []
            for prior in (report['serial'][0], report['serial'][2]):
                history = client.json('GET', args.history_route.replace('{sessionId}', prior['session']['sessionId']))
                messages = production_history(history, prior['session'], prior)
                messages.append({'role': 'user', 'content': '继续介绍。'})
                row = stream(client, output, prior['name'] + '-followup', prior['session'], body_for(prior['session']['sessionId'], messages, args.think == 'on', follow_up_max_tokens))
                row['production_history'] = history
                report['serial'].append(row); follow_rows.append(row)
                require(not row.get('error') and row['completed'], 'Followup failed')
            threshold = max(report['serial'][0]['usage']['prompt_tokens'], report['serial'][2]['usage']['prompt_tokens'])
            reused = [row['usage']['prompt_tokens_details']['cached_tokens'] for row in follow_rows]
            report['follow_up'] = dict(status='observed', passed=all(n > threshold for n in reused),
                reused_tokens=reused, beyond_prior_whole_prompt_threshold=threshold,
                scope='Production wire messages forwarded unchanged, including thinking/tools; actual cached tokens must exceed the prior whole prompt, not just a public prefix. HTTP does not expose full native state or token-ID identity.')
        if args.client_visible_follow_up:
            report['client_visible_follow_up'] = dict(status='inconclusive_failure', passed=False,
                scope='Actual client-visible WebUI replay only; no hidden-token/state identity claim')
            pre_records, identity = read_records(args.server_log, output, 'server-log-before-followup.jsonl')
            measured_before = timing_linkage(report['serial'], pre_records)
            report['client_visible_follow_up']['initial_timing_linkage'] = measured_before
            require(measured_before['passed'], 'Client-visible replay requires qualified initial round timings/usage')
            histories = [(prior, client_visible_history(prior, measured_before['measurements']))
                         for prior in (report['serial'][2], report['serial'][0])]
            follow_rows = []
            for prior, messages in histories:
                messages.append(dict(role='user', content='继续介绍。'))
                row = stream(client, output, prior['name'] + '-client-followup', prior['session'],
                             body_for(prior['session']['sessionId'], messages, args.think == 'on', follow_up_max_tokens))
                report['serial'].append(row); follow_rows.append(row)
                require(not row.get('error') and row['completed'], 'Client-visible followup failed')
            reused = [row['usage']['prompt_tokens_details']['cached_tokens'] for row in follow_rows]
            thresholds = [prior['usage']['prompt_tokens'] for prior, _ in histories]
            report['client_visible_follow_up'].update(status='observed', passed=reused[0] > thresholds[0],
                reused_tokens=reused, per_prior_whole_prompt_thresholds=thresholds,
                reuse_candidate=dict(name=follow_rows[0]['name'], session_id=follow_rows[0]['session']['sessionId'],
                    cached_tokens=reused[0], prior_prompt_tokens=thresholds[0], passed=reused[0] > thresholds[0]),
                reference=dict(name=follow_rows[1]['name'], session_id=follow_rows[1]['session']['sessionId'],
                    cached_tokens=reused[1], prior_prompt_tokens=thresholds[1], beyond_prior_prompt=reused[1] > thresholds[1],
                    scope='Older conversation may legitimately be evicted; recomputation or public-only reuse is allowed as a literal-output reference.'),
                scope='Replay the most recent third FF7 first, then the older first FF7 as reference. Exact actual emitted content/thinking and original user are production WebUI DTOs, without tools/unsupported metadata or reconstructed raw IDs. Require only the recent candidate actual cached tokens above its own prior whole prompt; no token/state-ID equality claim.')
        if args.cancellation:
            session = new_session()
            row = stream(client, output, 'cancel-owned', session, body_for(session['sessionId'], [{'role': 'user', 'content': PROMPTS[0]}], args.think == 'on', 128), cancel_after=2)
            report['cancelled_request'] = row
            require(row['cancelled_as_requested'] and not row.get('error'), 'Partial cancellation was not exercised')
            deadline = time.monotonic() + min(args.timeout, 30)
            queue_client = common.Client(args.base_url, output, min(args.timeout, 5))
            while True:
                queue = queue_client.json('GET', '/api/queue/status')
                report['queue_after_cancel'] = queue
                if queue.get('busy') is False and queue.get('pending_requests') == 0: break
                require(time.monotonic() < deadline, 'Owned cancellation queue did not recover')
                time.sleep(.1)
            recovery = new_session()
            row = stream(client, output, 'recovery-new-chat', recovery, body_for(recovery['sessionId'], [{'role': 'user', 'content': PROMPTS[1]}], args.think == 'on', args.max_tokens))
            report['serial'].append(row)
            require(not row.get('error') and row['completed'], 'Fresh request after cancellation failed')
        if args.long_new_chat_max_tokens is not None:
            session = new_session()
            row = stream(client, output, 'ff7-long-new-chat', session,
                         body_for(session['sessionId'], [{'role': 'user', 'content': PROMPTS[0]}], True, args.long_new_chat_max_tokens))
            report['serial'].append(row)
            report['long_new_chat'] = dict(status='inconclusive_failure', passed=False, request=row,
                semantic_quality_passed=None, repeat_parity_passed=None,
                scope='One additional independent FF7 new session, greedy max256, fully drained. Exact public-prefix usage and measured completion qualify only bounded latency observation, not semantic or repeated-output quality. No automatic cap escalation.')
            require(not row.get('error') and row['completed'], 'Long new-chat request did not complete')
        records, identity = read_records(args.server_log, output)
        report['server_log'] = identity
        timings = timing_linkage(report['serial'], records)
        report['server_timing_linkage'] = timings
        if args.long_new_chat_max_tokens is not None:
            probe = report['long_new_chat']
            row = probe['request']
            cached = row['usage']['prompt_tokens_details']['cached_tokens']
            links = [entry for entry in timings['usage_links'] if entry['request_id'] == row['request_id']]
            checks = dict(exact_public_cached_tokens=cached == args.long_new_chat_expected_cached_tokens,
                measured_completion=len(links) == 1 and all(links[0][key] for key in
                    ('prompt_matches', 'cached_matches', 'completion_matches', 'all_engine_completions_available')),
                no_tool_frames=not row['tool_frames_observed'])
            probe.update(status='observed', passed=all(checks.values()), checks=checks,
                expected_public_cached_tokens=args.long_new_chat_expected_cached_tokens, actual_cached_tokens=cached,
                visible_observation_status='observed' if row['text'] else 'inconclusive_no_visible_output',
                client_first_raw_ttft_ms=row['client_first_raw_ttft_ms'],
                client_first_thinking_ttft_ms=row['client_first_thinking_ttft_ms'],
                client_first_content_ttft_ms=row['client_first_content_ttft_ms'])
        report['repeat_parity'] = repeat_gate(report['serial'][0], report['serial'][2], timings['measurements'])
        report['stream_repeat_parity'] = stream_repeat_gate(report['serial'][0], report['serial'][2], timings['measurements'])
        report['output_acceptance_gate'] = dict(mode=args.output_gate,
            passed=report['repeat_parity']['passed'] if args.output_gate == 'visible' else report['stream_repeat_parity']['passed'],
            visible_quality_passed=report['repeat_parity']['passed'], visible_quality_status=report['repeat_parity']['status'],
            scope='Controller acceptance mode is explicit. A passing thinking stream gate never converts absent visible content into a visible-quality pass.')
        if args.follow_up and report['follow_up']['status'] == 'observed':
            follow_parity = repeat_gate(report['serial'][3], report['serial'][4], timings['measurements'])
            report['follow_up']['repeat_parity'] = follow_parity
            report['follow_up']['passed'] = report['follow_up']['passed'] and follow_parity['passed']
        if args.client_visible_follow_up and report['client_visible_follow_up']['status'] == 'observed':
            follow_parity = repeat_gate(report['serial'][3], report['serial'][4], timings['measurements'])
            report['client_visible_follow_up']['repeat_parity'] = follow_parity
            report['client_visible_follow_up']['passed'] = report['client_visible_follow_up']['passed'] and follow_parity['passed']
        cached = [row['usage']['prompt_tokens_details']['cached_tokens'] for row in report['serial'][:3]]
        cache_passed = cached[2] > 0 if args.cache_expectation == 'enabled' else all(n == 0 for n in cached) if args.cache_expectation == 'disabled' else None
        report['public_prefix_observation'] = dict(expectation=args.cache_expectation, cached_tokens=cached,
            passed=cache_passed, scope='New-session cached usage linked to actual generation timing. Positive usage alone does not identify a specific checkpoint payload or prove an optimal warmup.')
        if args.cancellation:
            cancelled = report['cancelled_request']
            matches = [r for r in records if checker.event_is(r, 'generation.timings')
                and checker.request_scope(r).get('RequestId') == cancelled['request_id']
                and r.get('props', {}).get('SessionId') == cancelled['session']['sessionId']
                and r.get('props', {}).get('FinishReason') == 'cancelled']
            report['cancellation'] = dict(status='observed', passed=len(matches) == 1,
                server_cancel_records=matches, queue_recovered=True,
                scope='Actual server cancellation, idle queue and a completed fresh request. No native-slot identity or GPU-parallel claim.')
        report['model_after'] = client.json('GET', '/api/models')
        report['completed'] = True
        report['passed'] = timings['passed'] and report['output_acceptance_gate']['passed'] and cache_passed is not False \
            and report['follow_up']['passed'] is not False and report['cancellation']['passed'] is not False
        report['passed'] = report['passed'] and report['client_visible_follow_up']['passed'] is not False
        report['passed'] = report['passed'] and report['long_new_chat']['passed'] is not False
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for session in sessions:
            try:
                result = client.json('DELETE', '/api/sessions/' + session['sessionId'])
                require(result.get('ok') is True and result.get('sessionId') == session['sessionId'], 'Session cleanup was not acknowledged')
                report['sessions_cleanup'].append(dict(sessionId=session['sessionId'], passed=True, response=result))
            except Exception as error:
                report['sessions_cleanup'].append(dict(sessionId=session['sessionId'], passed=False, error=str(error)))
                report['passed'] = False
        report['source_pins_after'] = {path: pin(path) for path in source_pins}
        if report['source_pins_after'] != source_pins:
            report.update(passed=False, source_error='Tool/provenance changed during execution')
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output / 'report.json', report)
    print(json.dumps({k: report.get(k) for k in ('passed', 'completed', 'error', 'public_prefix_observation', 'follow_up', 'client_visible_follow_up', 'cancellation')}, ensure_ascii=False))
    return int(not report['passed'])

if __name__ == '__main__':
    raise SystemExit(main())
