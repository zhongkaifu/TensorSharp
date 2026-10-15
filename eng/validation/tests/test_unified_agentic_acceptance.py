import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('agentic_acceptance', Path(__file__).resolve().parents[1] / 'unified-agentic-acceptance.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
NONCE = '0123456789abcdef' * 2


def good(events=()):
    return {'transport_complete': True, 'request_id_echo': True, 'cleanup_errors': [],
            'events': list(events) + [{'done': True, 'aborted': False, 'truncated': False, 'error': None,
                                    'promptTokens': 123, 'tokenCount': 5}]}


def step(tool, **extra):
    return {'skill_step': tool, 'ok': True, 'agent_id': '/root', **extra}


def artifact(name, kind, **extra):
    return {'name': name, 'json': {'kind': kind, 'nonce': NONCE, 'pid': 54321, 'created_unix_ns': time.time_ns(), **extra}}


def agent_row():
    def snapshot(result):
        return {'agents': [{'agent_id': '/root/fact_reader', 'workspace_id': '/root/fact_reader',
                            'permissions': 'read-only', 'status': 'completed', 'result': result, 'error': None}]}
    events = [step('spawn_agent'), step('skills_read', agent_id='/root/fact_reader', skill=m.SKILL, detail='references/alpha.txt'),
              step('wait_agent'), snapshot(f'FACT ALPHA {NONCE} = 19'), step('send_input'),
              step('skills_read', agent_id='/root/fact_reader', skill=m.SKILL, detail='references/beta.txt'),
              step('wait_agent'), snapshot(f'FACT BETA {NONCE} = 23'), step('list_agents'), step('close_agent'),
              step('wait_agent'), snapshot(f'FACT BETA {NONCE} = 23')]
    return good(events)


class AgenticGateTests(unittest.TestCase):
    def test_terminal_requires_complete_done_without_truncation_or_escaped_calls(self):
        self.assertTrue(m.terminal_gate(good())['passed'])
        variants = []
        for key, value in [('aborted', True), ('truncated', True), ('error', 'bad'), ('promptTokens', 0)]:
            row = good(); row['events'][-1][key] = value; variants.append(row)
        row = good(); row['events'].append({'token': 'late'}); variants.append(row)
        row = good(); row['events'].insert(0, {'tool_calls': [{'name': 'shell'}]}); variants.append(row)
        row = good(); row['events'].append(dict(row['events'][-1])); variants.append(row)
        row = good(); row['transport_complete'] = False; variants.append(row)
        for row in variants: self.assertFalse(m.terminal_gate(row)['passed'])

    def test_prose_claims_are_not_skill_execution(self):
        self.assertFalse(m.case_gate('skill', good([{'token': 'I ran the script: 16'}]), [], NONCE)['passed'])
        row = good([step('skills_list'), step('skills_read'), step('skills_run', skill=m.SKILL, detail='scripts/run_probe.py')])
        artifacts = [artifact('skill-receipt.json', 'skill', checksum=16)]
        self.assertTrue(m.case_gate('skill', row, artifacts, NONCE)['passed'])
        artifacts[0]['json']['nonce'] = 'forged'
        self.assertFalse(m.case_gate('skill', row, artifacts, NONCE)['passed'])
        artifacts[0]['json']['nonce'] = NONCE; row['events'][2]['ok'] = False
        self.assertFalse(m.case_gate('skill', row, artifacts, NONCE)['passed'])

    def test_code_requires_both_actual_sources_and_two_executions(self):
        row = good([step(n) for n in ('write_file', 'shell', 'read_file', 'apply_patch', 'shell')])
        files = [artifact('code-before.json', 'code-before', result=8), artifact('code-after.json', 'code-after', result=49),
                 {'name': 'solution.py', 'text': 'def square(x): return x + 1'},
                 {'name': 'solution.py', 'text': 'def square(x):\n    return(x*x)'}]
        self.assertTrue(m.case_gate('code', row, files, NONCE)['passed'])
        self.assertFalse(m.case_gate('code', row, files[1:], NONCE)['passed'])
        self.assertFalse(m.case_gate('code', row, files[:-1], NONCE)['passed'])
        bad = copy.deepcopy(row); bad['events'].pop(1)
        self.assertFalse(m.case_gate('code', bad, files, NONCE)['passed'])

    def test_code_static_checker_never_runs_imports_or_accepts_keywords(self):
        for text in ('# def square(x): return x*x', 'import os\ndef square(x): return x*x',
                     'def square(x): return 49', 'def square(x):\n return x*x\nprint(square(7))'):
            self.assertFalse(m.validate_function(text, '* x'))

    def test_same_child_followup_read_and_close_are_required(self):
        row = agent_row(); self.assertTrue(m.case_gate('agents', row, [], NONCE)['passed'])
        for tool in ('spawn_agent', 'send_input', 'close_agent', 'list_agents'):
            bad = copy.deepcopy(row); bad['events'] = [e for e in bad['events'] if e.get('skill_step') != tool]
            self.assertFalse(m.case_gate('agents', bad, [], NONCE)['passed'])
        bad = copy.deepcopy(row); bad['events'][5]['agent_id'] = '/root'
        self.assertFalse(m.case_gate('agents', bad, [], NONCE)['passed'])
        bad = copy.deepcopy(row); bad['events'][-2]['agents'][0]['status'] = 'running'
        self.assertFalse(m.case_gate('agents', bad, [], NONCE)['passed'])
        bad = copy.deepcopy(row); bad['events'].insert(0, step('spawn_agent'))
        self.assertFalse(m.case_gate('agents', bad, [], NONCE)['passed'])

    def test_last_terminal_and_exact_requestid_are_required(self):
        def log(message, request='r', scope='scope'):
            return json.dumps({scope: {'RequestId': request}, 'message': message})
        first = log('chat.complete finishReason=eos tokens=8')
        self.assertTrue(m.request_log_evidence(first, 'r')['last_terminal_natural_eos'])
        self.assertFalse(m.request_log_evidence(first + '\n' + log('chat.complete finishReason=length tokens=2'), 'r')['last_terminal_natural_eos'])
        self.assertFalse(m.request_log_evidence(first + '\n' + log('chat.cancelled tokens=2'), 'r')['last_terminal_natural_eos'])
        self.assertFalse(m.request_log_evidence(first, 'wrong')['last_terminal_natural_eos'])
        dual = json.dumps({'scope': {'RequestId': 'other'}, 'scopes': {'RequestId': 'r'}, 'message': 'chat.complete finishReason=eos'})
        self.assertFalse(m.request_log_evidence(dual, 'r')['last_terminal_natural_eos'])

    def test_process_exit_logs_reject_nonzero_timeout_and_other_skill(self):
        messages = ['codeexec.ran exit=0 timedOut=False ms=1', 'codeexec.ran exit=1 timedOut=False ms=1',
                    'codeexec.ran exit=0 timedOut=True ms=1',
                    f'skills.script.ran skill={m.SKILL} script=scripts/run_probe.py sandbox=none exit=0 ms=1',
                    'skills.script.ran skill=other script=scripts/run_probe.py exit=0 ms=1']
        text = '\n'.join(json.dumps({'scope': {'RequestId': 'r'}, 'message': x}) for x in messages)
        got = m.request_log_evidence(text, 'r')
        self.assertEqual(1, got['successful_shell_exits']); self.assertEqual(1, got['successful_fixture_script_exits'])

    def test_artifact_route_rejects_external_and_traversal(self):
        self.assertEqual('/api/code/artifacts/abc/solution.py', m.artifact_route('/api/code/artifacts/abc/solution.py'))
        for url in ('https://example.com/x', '//127.0.0.1/a', '/api/code/artifacts/a/../file',
                    '/api/code/artifacts/a/%2e%2e/file', '/api/code/artifacts/a/%252e%252e/file',
                    '/api/code/artifacts/a/x?query=1', '/api/code/artifacts/a', '/api/code/artifacts/a/%5cx'):
            with self.subTest(url=url), self.assertRaises(ValueError): m.artifact_route(url)

    def test_sse_requires_object_json_and_retains_tool_and_done(self):
        raw = b': tick\n\ndata: {"skill_step":"shell",\ndata: "ok":true}\n\ndata: {"done":true}\n\n'
        self.assertEqual(2, len(m.parse_sse(raw)))
        for raw in (b'data: [1]\n\n', b'data: {\n\n', b'garbage\n', b'data: \xff\n'):
            with self.assertRaises((ValueError, UnicodeDecodeError)): m.parse_sse(raw)

    def test_deadlines_and_remote_urls_rejected(self):
        for value in ('nan', 'inf', '0', '-1', '3601'):
            with self.assertRaises(ValueError): m.timeout(value)
        for url in ('https://localhost', 'http://other', 'http://user@localhost', 'http://localhost/path'):
            with self.assertRaises(ValueError): m.local_url(url)

    def test_receipt_pid_is_not_assumed_gone_and_pid_reuse_is_distinguished(self):
        files = [artifact('code-after.json', 'code-after', result=49)]
        created = files[0]['json']['created_unix_ns'] / 1e9
        with patch.object(m.psutil, 'Process') as process:
            process.return_value.create_time.return_value = created - 1
            process.return_value.is_running.return_value = True
            self.assertFalse(m.receipt_processes_exited(files, created - 2, created + 1)['passed'])
            process.return_value.create_time.return_value = created + 1
            self.assertTrue(m.receipt_processes_exited(files, created - 2, created + 1)['passed'])
        self.assertFalse(m.receipt_processes_exited(files, created + 1, created + 2)['passed'])
        self.assertFalse(m.receipt_processes_exited([], created - 1, created + 1)['passed'])

    def test_workspace_matches_only_owned_sanitized_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('ts-session-rootchild-123', 'ts-session-other-456', 'unrelated'): (root / name).mkdir()
            self.assertEqual([str(root / 'ts-session-rootchild-123')], m.matching_workspace_paths(root, ['/root/child']))

    def test_workspace_cleanup_requires_actual_creation_path_not_an_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); actual = root / 'ts-session-rootchild-123'
            records = [{'session_id':'/root/child', 'root':str(actual)}]
            self.assertTrue(m.workspace_evidence(root, ['/root/child'], records)['passed'])
            self.assertFalse(m.workspace_evidence(root, ['/root/child'], [])['passed'])
            actual.mkdir()
            self.assertFalse(m.workspace_evidence(root, ['/root/child'], records)['passed'])
            with self.assertRaises(ValueError): m.workspace_evidence(root / 'wrong', ['/root/child'], records)

    def test_known_skill_executes_real_standard_library_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); skill = root / 'skill'
            shutil.copytree(m.HERE / 'fixtures/unified-agentic/skills' / m.SKILL, skill)
            (skill / 'references').mkdir()
            (skill / 'references/input.json').write_text(json.dumps({'nonce': NONCE, 'values': [3,5,8]}), encoding='utf-8')
            started = time.time()
            result = subprocess.run([sys.executable, str(skill / 'scripts/run_probe.py'), NONCE], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(0, result.returncode, result.stderr)
            receipt = json.loads((root / 'skill-receipt.json').read_text(encoding='utf-8'))
            self.assertTrue(m.receipt_gate('skill', receipt, NONCE))
            self.assertTrue(m.receipt_processes_exited([{'name':'skill-receipt.json','json':receipt}], started, time.time())['passed'])
            bad = subprocess.run([sys.executable, str(skill / 'scripts/run_probe.py'), 'wrong'], cwd=root, capture_output=True, timeout=10)
            self.assertNotEqual(0, bad.returncode)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.release = threading.Event()
        release = self.release
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_DELETE(self):
                body = b'{"ok":true}'
                self.send_response(200); self.send_header(m.http_checks.HEADER, self.headers[m.http_checks.HEADER])
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                body = b'def square(x): return x*x\n'
                self.send_response(200); self.send_header(m.http_checks.HEADER, self.headers[m.http_checks.HEADER])
                self.send_header('Content-Length', str(len(body))); self.end_headers()
                if self.path == '/slow': release.wait(2)
                try: self.wfile.write(body)
                except (OSError, ConnectionError): pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.release.set(); self.server.shutdown(); self.server.server_close(); self.thread.join(2); self.tmp.cleanup()

    def test_raw_download_does_not_require_json_and_delete_reaches_real_method(self):
        raw = m.extra_call(self.url, self.root / 'get', 'GET', '/source', raw=True)
        self.assertTrue(raw['transport_complete']); self.assertNotIn('error', raw)
        self.assertTrue((self.root / 'get/response.raw').read_bytes().startswith(b'def square'))
        deleted = m.extra_call(self.url, self.root / 'delete', 'DELETE', '/session')
        self.assertEqual({'ok':True}, deleted['json']); self.assertTrue(deleted['request_id_echo'])

    def test_binary_side_artifact_is_preserved_without_decoding_or_executing(self):
        rawdir = self.root / 'wire'; rawdir.mkdir(); raw = b'\xa7\r\r\n\x00\xff'
        (rawdir / 'response.raw').write_bytes(raw)
        row = good([{'files':[{'name':'__pycache__/solution.pyc','bytes':len(raw),'url':'/api/code/artifacts/x/solution.pyc'}]}])
        response = {'status':200,'transport_complete':True,'request_id_echo':True,'directory':str(rawdir)}
        with patch.object(m, 'extra_call', return_value=response):
            files = m.download_artifacts(m.http_checks.Client(self.url, self.root), row, self.root)
        self.assertTrue(files[0]['binary']); self.assertNotIn('text', files[0])
        self.assertEqual(m.hashlib.sha256(raw).hexdigest(), files[0]['sha256'])

    def test_absolute_deadline_releases_stalled_owned_socket(self):
        started = time.monotonic()
        row = m.extra_call(self.url, self.root / 'slow', 'GET', '/slow', raw=True, duration=.1)
        self.assertFalse(row['transport_complete']); self.assertIn('error', row)
        self.assertLess(time.monotonic()-started, 1)


if __name__ == '__main__': unittest.main()
