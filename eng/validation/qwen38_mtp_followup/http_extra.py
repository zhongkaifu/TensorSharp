#!/usr/bin/env python3
"""Separate counted long-context or actual video_url/history checks; never replaces original cases."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def load_decoder():
    p = Path(__file__).with_name('http_contract.py')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('qwen_http_contract', p))
    exec(compile(p.read_bytes(), str(p), 'exec'), module.__dict__)
    return module


def video_body(model, path, stream):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '7bd626261836db05c66167db95c02cb059b77a5019c2ff981be2c8c8a5789c67':
        raise ValueError('Pinned video fixture changed')
    body = {'model': model, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Read the numbers on both sampled video frames in chronological order. Return only JSON with codes as an array of two strings.'},
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + base64.b64encode(raw).decode(), 'fps': 1, 'max_frames': 2}}]}],
        'stream': stream, 'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 42, 'think': False,
        'max_tokens': 256, 'response_format': {'type': 'json_object'}}
    if stream:
        body['stream_options'] = {'include_usage': True}
    # The original fixture is three 1-fps frames. The production sampler's
    # evenly-spaced cap selects frame0 and frame2, not the first two frames.
    return body, [{'codes': ['17', '86']}, {'second': 2}], {'indices': [0, 2], 'seconds': [0, 2]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['video2-history', 'counted-long62k'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--video', type=Path)
    p.add_argument('--request', type=Path)
    p.add_argument('--request-sha256')
    p.add_argument('--count', type=Path)
    p.add_argument('--count-sha256')
    p.add_argument('--url', default='http://127.0.0.1:5120')
    p.add_argument('--blocking', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if urllib.parse.urlparse(a.url).hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Local identity-bound server only')
    a.output.mkdir(parents=True, exist_ok=False)
    if a.case == 'video2-history':
        body, expected, sampling = video_body(a.model, a.video, not a.blocking)
    else:
        raw = a.request.read_bytes()
        count_raw = a.count.read_bytes()
        if hashlib.sha256(raw).hexdigest() != a.request_sha256 or hashlib.sha256(count_raw).hexdigest() != a.count_sha256:
            raise ValueError('Counted input identity mismatch')
        body, count = json.loads(raw), json.loads(count_raw)
        if count['body_sha256'] != a.request_sha256 or count['prompt_tokens'] > 65535 or count['maximum_position_exclusive'] > 65536:
            raise ValueError('Input not admissible')
        if body['model'] != a.model or a.blocking:
            raise ValueError('Counted original wire body must stay exact, including model and stream')
        expected = [{'ALPHA': 'silver-4821', 'BETA': 'amber-7392', 'GAMMA': 'violet-1605'}]
        sampling = {'actual_prompt_tokens': count['prompt_tokens'], 'qsa_qualified': False}
    report = {'status': 'prepared-not-run', 'release_qualified': False, 'case': a.case, 'sampling': sampling, 'expected': expected, 'turns': [],
              'scope': 'Actual OpenAI request. Video rejection is a runtime gap, not passed media coverage; long case is new and QSA-unqualified.'}
    (a.output / 'first-request.json').write_text(json.dumps(body, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    if not a.execute:
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Prepared extra case; no HTTP request sent'); return 0
    decoder = load_decoder()
    report['status'] = 'running'
    for turn, oracle in enumerate(expected):
        wire = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        row = {'turn': turn, 'status': 'failed', 'request_sha256': hashlib.sha256(wire).hexdigest(), 'expected': oracle, 'started_unix': time.time()}
        (a.output / f'turn-{turn}.request.json').write_bytes(wire)
        try:
            request = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', wire, {'Content-Type': 'application/json'})
            try: response = urllib.request.urlopen(request, timeout=1200)
            except urllib.error.HTTPError as error: response = error
            with response:
                raw = response.read(); row['http_status'] = response.status
            (a.output / f'turn-{turn}.response.raw').write_bytes(raw)
            row['response_sha256'] = hashlib.sha256(raw).hexdigest()
            if row['http_status'] != 200: raise ValueError('HTTP ' + str(row['http_status']))
            parsed = decoder.parse_success(raw.decode('utf-8'), body['stream']); row['parsed'] = parsed
            if not parsed['done'] or parsed['finish'] != 'stop' or not parsed['usage'] or parsed['message'].get('tool_calls'):
                raise ValueError('Missing completed exact JSON response')
            actual = json.loads(parsed['message']['content'])
            # Exact JSON shape/types: bool is not an acceptable numeric timestamp.
            if actual != oracle or any(type(actual[k]) is not type(v) for k, v in oracle.items()):
                raise ValueError('Exact original content/type oracle failed')
            if a.case == 'counted-long62k' and parsed['usage'].get('prompt_tokens') != sampling['actual_prompt_tokens']:
                raise ValueError('Actual HTTP token count differs from pinned production count')
            row['status'] = 'passed'
            if a.case == 'video2-history':
                body['messages'] += [parsed['message'], {'role': 'user', 'content': 'Using the same video history and its frame timestamps, at what second does 86 appear? Return only JSON with second as a number.'}]
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        row['wall_seconds'] = time.time() - row['started_unix']; report['turns'].append(row)
        (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if row['status'] != 'passed': break
    report['status'] = 'passed' if len(report['turns']) == len(expected) and all(r['status'] == 'passed' for r in report['turns']) else 'failed'
    report['unrun_turns'] = len(expected) - len(report['turns'])
    (a.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['status']); return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
