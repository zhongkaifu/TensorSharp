#!/usr/bin/env python3
"""Separate 32 real OpenAI HTTP tool-contract probes; stores complete wire bodies and responses."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


TOOL = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather for a city.',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city'], 'additionalProperties': False}}}
NAMED = {'type': 'function', 'function': {'name': 'get_weather'}}


def catalog(model):
    wrong = {'type': 'function', 'function': {'name': 'undeclared_weather_tool'}}
    case_wrong = {'type': 'function', 'function': {'name': 'Get_Weather'}}
    configs = [
        ('required-absent', {'tool_choice': 'required'}, 400, 'at least one'),
        ('required-empty', {'tools': [], 'tool_choice': 'required'}, 400, 'at least one'),
        ('named-absent', {'tool_choice': NAMED}, 400, 'at least one'),
        ('named-empty', {'tools': [], 'tool_choice': NAMED}, 400, 'at least one'),
        ('named-undeclared', {'tools': [TOOL], 'tool_choice': wrong}, 400, 'undeclared_weather_tool'),
        ('named-case-mismatch', {'tools': [TOOL], 'tool_choice': case_wrong}, 400, 'Get_Weather'),
        ('required-empty-serial', {'tools': [], 'tool_choice': 'required', 'parallel_tool_calls': False}, 400, 'at least one'),
        ('named-undeclared-serial', {'tools': [TOOL], 'tool_choice': wrong, 'parallel_tool_calls': False}, 400, 'undeclared_weather_tool'),
        ('required-offered', {'tools': [TOOL], 'tool_choice': 'required'}, 200, 'tool'),
        ('named-offered', {'tools': [TOOL], 'tool_choice': NAMED}, 200, 'tool'),
        ('none-offered', {'tools': [TOOL], 'tool_choice': 'none'}, 200, 'text'),
        ('auto-offered', {'tools': [TOOL], 'tool_choice': 'auto'}, 200, 'tool'),
        ('choice-absent-offered', {'tools': [TOOL]}, 200, 'tool'),
        ('none-no-tools', {'tool_choice': 'none'}, 200, 'text'),
        ('auto-no-tools', {'tool_choice': 'auto'}, 200, 'text'),
        ('required-offered-serial', {'tools': [TOOL], 'tool_choice': 'required', 'parallel_tool_calls': False}, 200, 'tool'),
    ]
    rows = []
    for name, options, status, oracle in configs:
        for stream in (False, True):
            message = 'Call get_weather for Paris. Do not invent its result.' if oracle != 'text' else 'Reply with only the single word apple. Do not use any tool.'
            body = {'model': model, 'messages': [{'role': 'user', 'content': message}], 'stream': stream,
                    'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 42, 'think': False, 'max_tokens': 256,
                    **copy.deepcopy(options)}
            if stream:
                body['stream_options'] = {'include_usage': True}
            rows.append({'id': name + ('-stream' if stream else '-blocking'), 'body': body,
                         'expected_status': status, 'oracle': oracle})
    return rows


def parse_success(raw, streaming):
    if not streaming:
        body = json.loads(raw)
        choice = body['choices'][0]
        return {'message': choice['message'], 'finish': choice.get('finish_reason'), 'usage': body.get('usage'), 'done': True}
    message = {'role': 'assistant', 'content': ''}
    tool_calls, finish, usage, done = {}, None, None, False
    for line in raw.splitlines():
        if not line.startswith('data:'):
            continue
        text = line[5:].strip()
        if text == '[DONE]':
            done = True
            continue
        event = json.loads(text)
        if 'error' in event:
            raise ValueError('SSE error: ' + json.dumps(event['error']))
        if event.get('usage') is not None:
            usage = event['usage']
        for choice in event.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('Unexpected extra choice')
            delta = choice.get('delta', {})
            message['content'] += delta.get('content') or ''
            for part in delta.get('tool_calls', []):
                if type(part.get('index')) is not int or part['index'] < 0:
                    raise ValueError('Tool delta missing nonnegative index')
                call = tool_calls.setdefault(part['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    if call['id'] and call['id'] != part['id']:
                        raise ValueError('Tool ID changed across deltas')
                    call['id'] = part['id']
                for field in ('name', 'arguments'):
                    call['function'][field] += part.get('function', {}).get(field) or ''
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    if tool_calls:
        if set(tool_calls) != set(range(len(tool_calls))):
            raise ValueError('Missing tool-call index')
        message['tool_calls'] = [tool_calls[i] for i in sorted(tool_calls)]
    return {'message': message, 'finish': finish, 'usage': usage, 'done': done}


def judge(case, status, content_type, raw):
    if status != case['expected_status']:
        raise ValueError(f'HTTP {status}, expected {case["expected_status"]}')
    if status == 400:
        if 'application/json' not in content_type.lower() or raw.lstrip().startswith('data:'):
            raise ValueError('Invalid contract must reject before SSE begins')
        data = json.loads(raw)
        if data.get('error', {}).get('type') != 'invalid_request_error' or case['oracle'] not in data['error'].get('message', ''):
            raise ValueError('Wrong API validation error')
        if data.get('choices') or data.get('usage'):
            raise ValueError('Rejected body contains generation or usage')
        return {'api_rejected_before_stream': True,
                'zero_native_inference_proven': False, 'qualification': 'HTTP proves rejection/no emitted completion; independent adapter unit tests prove zero queue admissions'}
    parsed = parse_success(raw, case['body']['stream'])
    if not parsed['done'] or not isinstance(parsed['usage'], dict) or parsed['usage'].get('completion_tokens', 0) <= 0:
        raise ValueError('Missing completed delivery and generation usage')
    calls = parsed['message'].get('tool_calls', [])
    content = parsed['message'].get('content') or ''
    if case['oracle'] == 'tool':
        if parsed['finish'] != 'tool_calls' or len(calls) != 1 or not calls[0].get('id'):
            raise ValueError('Actual parsed one-call contract/finish/ID failed (raw XML is not an API call)')
        if calls[0]['function']['name'] != 'get_weather' or json.loads(calls[0]['function']['arguments']) != {'city': 'Paris'}:
            raise ValueError('Wrong function or exact arguments')
    elif parsed['finish'] != 'stop' or calls or content.strip() != 'apple':
        raise ValueError('Text-only exact answer/no-tool contract failed')
    return parsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:5120')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Only the identity-bound local benchmark server is supported')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = catalog(args.model)
    (args.output / 'catalog.json').write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')
    if not args.execute:
        print('Prepared 32 requests; no HTTP requests sent')
        return 0
    results = []
    for case in cases:
        wire = json.dumps(case['body'], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        prefix = args.output / case['id']
        prefix.with_suffix('.request.json').write_bytes(wire)
        started = time.time()
        row = {'id': case['id'], 'status': 'failed', 'started_unix': started, 'body_sha256': hashlib.sha256(wire).hexdigest()}
        try:
            request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions', wire, {'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(request, timeout=1200)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read()
                prefix.with_suffix('.response.raw').write_bytes(raw)
                row.update(http_status=response.status, content_type=response.headers.get('Content-Type', ''), response_sha256=hashlib.sha256(raw).hexdigest())
            row['parsed'] = judge(case, row['http_status'], row['content_type'], raw.decode('utf-8'))
            row['status'] = 'passed'
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        row['wall_seconds'] = time.time() - started
        results.append(row)
        (args.output / 'result.json').write_text(json.dumps({'status': 'running', 'release_qualified': False, 'cases': results}, indent=2) + '\n', encoding='utf-8')
    passed = sum(r['status'] == 'passed' for r in results)
    (args.output / 'result.json').write_text(json.dumps({'status': 'passed' if passed == 32 else 'failed', 'release_qualified': False,
        'passed': passed, 'total': 32, 'cases': results, 'scope': 'Actual OpenAI parser/tool-choice contract. No tools are executed; sandbox/skills are a separate 46-case campaign.'}, indent=2) + '\n', encoding='utf-8')
    print(f'{passed}/32 HTTP contracts passed')
    return int(passed != 32)


if __name__ == '__main__':
    raise SystemExit(main())
