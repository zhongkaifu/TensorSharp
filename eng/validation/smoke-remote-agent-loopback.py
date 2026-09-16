#!/usr/bin/env python3
"""Exercise real local tools/artifacts with scripted HTTP responses, without a model.

This is a plumbing fixture, never model-quality or native-binding evidence. The
two original code scenarios are copied verbatim from the exported 46-case file.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


SUM_SOURCE = """import json

def sum_numbers(n):
    return n * (n + 1) // 2

if __name__ == '__main__':
    assert sum_numbers(0) == 0
    assert sum_numbers(1) == 1
    assert sum_numbers(37) == 703
    with open('result.json', 'w') as output:
        json.dump({'n': 37, 'sum': 703, 'tests_passed': True}, output)
"""
PARITY_SOURCE = """import json

def parity(n): return 'even'

if __name__ == '__main__':
    outputs = [parity(n) for n in [-3, 0, 4, 7]]
    assert outputs == ['odd', 'even', 'even', 'odd']
    with open('result.json', 'w') as output:
        json.dump({'outputs': outputs, 'tests_passed': True}, output)
"""
STEPS = {
    'code_generation_run': [('write_file', {'path': 'sum_numbers.py', 'content': SUM_SOURCE}),
                            ('shell', {'command': 'python3 sum_numbers.py'})],
    'code_edit_run': [('write_file', {'path': 'parity.py', 'content': PARITY_SOURCE}),
                      ('read_file', {'path': 'parity.py'}),
                      ('edit_file', {'path': 'parity.py', 'old_string': "def parity(n): return 'even'",
                                     'new_string': "def parity(n): return 'even' if n % 2 == 0 else 'odd'"}),
                      ('shell', {'command': 'python3 parity.py'})],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assembly', 'fixtures', 'skills-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dotnet', default='/usr/bin/dotnet')
    parser.add_argument('--artifact-port', type=int, default=18482)
    parser.add_argument('--bwrap', default='/usr/local/bin/bwrap')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh scripted-smoke output directory')
    args.output.mkdir(parents=True)
    original = json.loads(args.fixtures.read_text())
    selected = [case for case in original['cases'] if case['variant'] == 'original'
                and case['trial'] == 'c1-i0' and case['scenario'] in STEPS]
    if len(selected) != 2:
        raise ValueError('Expected the two original serial code cases')
    fixture_path = args.output / 'scripted-fixtures.json'
    fixture_path.write_text(json.dumps({'scope': 'Scripted HTTP plumbing only, no model', 'cases': selected}, indent=2) + '\n')
    identity = args.output / 'synthetic-identity.json'
    identity.write_text(json.dumps({'scripted_fixture': True, 'scope': 'Synthetic fixture; no remote process or native mapping exists'}, indent=2) + '\n')
    errors, requests = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                    chunks = []
                    size = 0
                    while True:
                        count = int(self.rfile.readline().split(b';', 1)[0], 16)
                        if count == 0:
                            while self.rfile.readline() not in (b'\r\n', b'\n', b''):
                                pass
                            break
                        size += count
                        if size > 1024 * 1024:
                            raise ValueError('Scripted fixture request exceeds 1 MiB')
                        chunks.append(self.rfile.read(count))
                        if self.rfile.read(2) != b'\r\n':
                            raise ValueError('Invalid HTTP chunk boundary')
                    body = b''.join(chunks)
                else:
                    size = int(self.headers['Content-Length'])
                    if size > 1024 * 1024:
                        raise ValueError('Scripted fixture request exceeds 1 MiB')
                    body = self.rfile.read(size)
                payload = json.loads(body)
                if self.path != '/v1/chat/completions' or payload['model'] != 'scripted-loopback-fixture':
                    raise ValueError('Unexpected endpoint/model')
                if payload.get('skills') != [] or payload.get('skills_discovery') is not False:
                    raise ValueError('Remote skill execution was not explicitly disabled')
                messages = payload['messages']
                user = next(message['content'] for message in messages if message['role'] == 'user')
                scenario = next(case['scenario'] for case in selected if case['spec']['prompt'] == user)
                calls = [call for message in messages for call in message.get('tool_calls', [])]
                results = [message for message in messages if message['role'] == 'tool']
                if [call['id'] for call in calls] != [message.get('tool_call_id') for message in results]:
                    raise ValueError('Assistant/result ID association was not preserved')
                step = len(results)
                if step < len(STEPS[scenario]):
                    name, arguments = STEPS[scenario][step]
                    declared = {item['function']['name'] for item in payload['tools']}
                    if name not in declared:
                        raise ValueError('Local tool was not declared: ' + name)
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': f'fixture::{scenario}:{step}', 'type': 'function',
                         'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
                    finish = 'tool_calls'
                elif step == len(STEPS[scenario]):
                    links = re.findall(r'http://127\.0\.0\.1:[0-9]+/api/code/artifacts/[^\s"<>\\]+/result\.json', results[-1]['content'])
                    if not links:
                        raise ValueError('Actual shell result did not advertise result.json')
                    message = {'role': 'assistant', 'content': links[-1]}
                    finish = 'stop'
                else:
                    raise ValueError('Unexpected extra generation round')
                response = {'id': f'scripted-{scenario}-{step}', 'object': 'chat.completion',
                            'model': 'scripted-loopback-fixture',
                            'choices': [{'index': 0, 'finish_reason': finish, 'message': message}]}
                requests.append({'scenario': scenario, 'step': step, 'at_unix': time.time(),
                                 'request': payload, 'response': response})
                body = json.dumps(response).encode()
                self.send_response(200)
            except Exception as error:
                errors.append(repr(error))
                body = json.dumps({'error': repr(error)}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [sys.executable, str(Path(__file__).with_name('run-remote-agent-profile.py')),
               '--dotnet', args.dotnet, '--assembly', str(args.assembly), '--fixtures', str(fixture_path),
               '--identity', str(identity), '--skills-root', str(args.skills_root),
               '--artifact-root', str(args.output / 'artifacts'), '--output', str(args.output / 'profile'),
               '--endpoint', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
               '--model', 'scripted-loopback-fixture', '--expected-native-sha256', '0' * 64,
               '--artifact-port', str(args.artifact_port), '--bwrap', args.bwrap, '--scripted-fixture']
    try:
        result = subprocess.run(command, timeout=240)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (args.output / 'scripted-http-trace.json').write_text(json.dumps(requests, indent=2) + '\n')
    passed = result.returncode == 0 and not errors and len(requests) == sum(len(value) + 1 for value in STEPS.values())
    report = {'status': 'passed' if passed else 'failed', 'scripted_fixture': True, 'release_qualified': False,
              'scope': 'Real local file/read/edit/shell/sandbox/artifact/independent source checks; scripted responses, no model quality or native proof',
              'assembly_sha256': hashlib.sha256(args.assembly.read_bytes()).hexdigest(),
              'source_fixtures_sha256': hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              'request_count': len(requests), 'errors': errors, 'profile_exit_code': result.returncode, 'command': command}
    (args.output / 'smoke-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
