#!/usr/bin/env python3
"""Real local Server agentic acceptance. Prepare fixtures, then test an owned server.

This runner never starts a model, executes generated code itself, or fabricates
tool replies. All execution is requested through /api/chat and serviced by the
real host. Downloaded Python is inspected with AST, never executed here.
Windows unconfined shell is not filesystem/network isolation. Child workspace
cleanup and process exit are separate gates, not inferred from a final answer.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import threading
import time
import uuid
import psutil
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit

HERE = Path(__file__).resolve().parent
CHECKER = HERE / 'unified-server-http-checks.py'
spec = importlib.util.spec_from_file_location('unified_http_transport', CHECKER)
http_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http_checks)
SKILL = 'unified-agentic-probe'
MAX_WIRE = 16 << 20
MAX_ARTIFACT = 1 << 20


def require(value, reason):
    if not value:
        raise ValueError(reason)


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def timeout(value):
    value = float(value)
    require(math.isfinite(value) and 0 < value <= 3600, 'timeout must be finite and in (0,3600]')
    return value


def fixture_pins(root):
    paths = sorted((root / 'skills').rglob('*'))
    require(all(not p.is_symlink() for p in paths), 'Fixture links are forbidden')
    return {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}


def prepare(output):
    output = http_checks.evidence_output_path(output)
    require(not output.exists(), 'Prepare requires a fresh ignored directory')
    output.mkdir(parents=True)
    shutil.copytree(HERE / 'fixtures/unified-agentic/skills', output / 'skills')
    nonce = uuid.uuid4().hex
    refs = output / 'skills' / SKILL / 'references'
    refs.mkdir()
    save(refs / 'input.json', {'nonce': nonce, 'values': [3, 5, 8]})
    (refs / 'alpha.txt').write_text(f'FACT ALPHA {nonce} = 19\n', encoding='utf-8')
    (refs / 'beta.txt').write_text(f'FACT BETA {nonce} = 23\n', encoding='utf-8')
    plan = {'schema_version': 1, 'created_utc': now(), 'nonce': nonce,
            'fixture_root': str(output), 'files': fixture_pins(output),
            'server_flags': ['--code-exec', '--code-exec-timeout', '30', '--skills-dir', str(output / 'skills'),
                '--skills-allow-exec', '--skills-max-rounds', '16', '--agents-max-concurrent', '2',
                '--agents-max-count', '4', '--agents-max-depth', '1', '--agents-max-rounds', '8',
                '--agents-max-generations', '12', '--agents-timeout', '600', '--sampling-precedence', 'request'],
            'windows_explicit_flags': ['--code-exec-unconfined', '--skills-sandbox', 'preferred'],
            'limits': 'No package/network flags. Windows root shell is not file/network confined; read-only child agents do not require child shell. Model and server launch/identity remain owned by the outer runner.'}
    save(output / 'plan.json', plan)
    print(output / 'plan.json')


def local_url(value):
    parsed = urlsplit(value)
    require(parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')
            and not parsed.username and not parsed.password and parsed.path in ('', '/')
            and not parsed.query and not parsed.fragment, 'Only an explicitly owned local HTTP server is supported')
    return parsed


def artifact_route(value):
    parsed = urlsplit(value)
    require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
            'Artifact URL must be a relative host-owned route')
    path = unquote(parsed.path)
    require(path.startswith('/api/code/artifacts/') and '\\' not in path and '\x00' not in path and '%' not in path,
            'Unexpected artifact route')
    parts = path.split('/')[4:]
    require(len(parts) >= 2 and all(x and x not in ('.', '..') for x in parts), 'Invalid artifact path')
    return parsed.path


def extra_call(url, output, method, route, *, raw=False, duration=30):
    """Bounded raw GET/JSON DELETE; do not alter the shared HTTP checker's API."""
    endpoint = local_url(url)
    require(method in ('GET', 'DELETE'), 'Unsupported auxiliary method')
    output.mkdir()
    request_id = 'unified-agentic-' + uuid.uuid4().hex
    row = {'request_id': request_id, 'directory': str(output), 'method': method,
           'route': route, 'transport_complete': False, 'cleanup_errors': []}
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=duration)
    response, sock = None, None
    wire = bytearray()
    expired = threading.Event()
    deadline = time.monotonic() + duration
    def expire():
        expired.set()
        current = sock or conn.sock
        if current is not None:
            try: current.shutdown(socket.SHUT_RDWR)
            except OSError: pass
            try: current.close()
            except OSError: pass
    timer = threading.Timer(duration, expire); timer.daemon = True
    try:
        timer.start()
        conn.connect(); sock = conn.sock
        require(time.monotonic() < deadline and not expired.is_set(), 'Auxiliary connect deadline')
        conn.request(method, route, headers={http_checks.HEADER: request_id, 'Connection': 'close'})
        response = conn.getresponse()
        row.update(status=response.status, request_id_echo=response.getheader(http_checks.HEADER) == request_id)
        while True:
            chunk = response.read(min(65536, MAX_ARTIFACT + 1 - len(wire)))
            require(time.monotonic() < deadline and not expired.is_set(), 'Auxiliary response deadline')
            if not chunk: break
            wire.extend(chunk)
            require(len(wire) <= MAX_ARTIFACT, 'Auxiliary response exceeds bound')
        if not raw: row['json'] = json.loads(wire.decode('utf-8-sig'))
        row['transport_complete'] = True
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    finally:
        for name, action in [('timer-cancel', timer.cancel), ('response-close', lambda: response.close() if response else None),
                             ('connection-close', conn.close), ('timer-join', lambda: timer.join(2) if timer.ident else None)]:
            try: action()
            except BaseException as error: row['cleanup_errors'].append(f'{name}: {error}')
        row['timed_out'] = expired.is_set()
        (output / 'response.raw').write_bytes(wire)
        save(output / 'response.json', row)
    return row


def request_log_evidence(text, request_id):
    records, terminals, shells, scripts, workspaces = [], [], [], [], []
    for line in text.splitlines():
        try: record = json.loads(line)
        except ValueError: continue
        if not isinstance(record, dict): continue
        scope = record['scope'] if 'scope' in record else record.get('scopes')
        if not http_checks._contains_request(scope, request_id): continue
        message = record.get('message')
        if not isinstance(message, str): continue
        if message.startswith(('chat.complete ', 'chat.cancelled ', 'generate.complete ', 'generate.cancelled ')):
            terminals.append(message); records.append(record)
        if message.startswith('codeexec.ran '):
            shells.append(message); records.append(record)
        if message.startswith('skills.script.ran '):
            scripts.append(message); records.append(record)
        if message.startswith('workspace.created '):
            match = re.fullmatch(r'workspace\.created session=(\S+) root=(.+)', message)
            if match:
                workspaces.append({'session_id': match[1], 'root': match[2]})
                records.append(record)
    return {'last_terminal_natural_eos': bool(terminals) and terminals[-1].startswith(('chat.complete ', 'generate.complete '))
                and re.search(r'\bfinishReason=eos(?:\s|$)', terminals[-1]) is not None,
            'successful_shell_exits': sum(bool(re.search(r'\bexit=0(?:\s|$)', m))
                and bool(re.search(r'\btimedOut=False(?:\s|$)', m, re.I)) for m in shells),
            'successful_fixture_script_exits': sum(bool(re.search(r'\bexit=0(?:\s|$)', m))
                and ('skill=' + SKILL + ' ') in m and 'script=scripts/run_probe.py ' in m for m in scripts),
            'terminal_messages': terminals, 'workspace_creations': workspaces, 'records': records,
            'scope': 'Last correlated generation termination plus non-truncated WebUI done; nested tool/child generations share the outer RequestId.'}


def read_log(path, offset):
    with Path(path).open('rb') as source:
        source.seek(offset)
        raw = source.read((32 << 20) + 1)
    require(len(raw) <= 32 << 20, 'Log capture exceeded bound')
    return raw


class ProcessObservation:
    """Observe descendants, never terminate arbitrary PIDs. Outer launcher owns server."""
    def __init__(self, pid):
        self.server = psutil.Process(pid)
        self.creation = self.server.create_time()
        self.baseline = {(p.pid, p.create_time()) for p in self.server.children(recursive=True)}
        self.seen, self.errors = {}, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def identity(self):
        return {'pid': self.server.pid, 'create_time': self.creation, 'executable': self.server.exe()}

    def observe(self):
        while not self.stop.is_set():
            try:
                require(self.server.create_time() == self.creation and self.server.is_running(), 'Server identity changed')
                for p in self.server.children(recursive=True):
                    try:
                        key = (p.pid, p.create_time())
                        if key not in self.baseline:
                            self.seen[key] = {'pid': p.pid, 'create_time': key[1], 'name': p.name()}
                    except psutil.NoSuchProcess: pass
            except Exception as error:
                self.errors.append(f'{type(error).__name__}: {error}')
                return
            self.stop.wait(.1)

    def finish(self):
        self.stop.set()
        if self.thread.ident: self.thread.join(3)
        remaining = []
        for (pid, created), record in self.seen.items():
            try:
                p = psutil.Process(pid)
                if p.create_time() == created and p.is_running(): remaining.append(record)
            except psutil.NoSuchProcess: pass
            except Exception as error: self.errors.append(f'{type(error).__name__}: {error}')
        return {'observed': list(self.seen.values()), 'remaining': remaining, 'errors': self.errors,
                'passed': not remaining and not self.errors and not self.thread.is_alive(),
                'scope': '100ms sampled server descendants with PID+creation identity; fast processes may be missed. Receipts and successful host invocation/exit logs are independent proof.'}


def receipt_processes_exited(artifacts, started_unix, ended_unix):
    checks = []
    for item in artifacts:
        value = item.get('json')
        if not isinstance(value, dict) or value.get('kind') not in ('skill', 'code-before', 'code-after'): continue
        created = value.get('created_unix_ns')
        valid = type(value.get('pid')) is int and value['pid'] > 0 and type(created) is int and started_unix <= created / 1e9 <= ended_unix
        record = {'receipt': item['name'], 'pid': value.get('pid'), 'timestamp_in_request': valid, 'same_process_alive': None}
        if valid:
            try:
                process = psutil.Process(value['pid'])
                # A creation time after the receipt is a reused PID, not this command.
                record['same_process_alive'] = process.is_running() and process.create_time() <= created / 1e9
            except psutil.NoSuchProcess: record['same_process_alive'] = False
            except Exception as error: record['error'] = str(error)
        checks.append(record)
    return {'passed': bool(checks) and all(c['timestamp_in_request'] and c['same_process_alive'] is False for c in checks), 'receipts': checks}


def parse_sse(raw):
    events, data = [], []
    for line in raw.decode('utf-8', errors='strict').splitlines() + ['']:
        if not line:
            if data:
                value = json.loads('\n'.join(data))
                require(isinstance(value, dict), 'Expected an object SSE frame')
                events.append(value)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif line.startswith(':'):
            continue
        else:
            require(line.startswith(('event:', 'id:', 'retry:')), 'Unknown SSE line')
    return events


def stream_chat(url, body, output, deadline_seconds):
    """One absolute deadline covers connect, headers, streaming and cleanup."""
    endpoint = local_url(url)
    request_id = 'unified-agentic-' + uuid.uuid4().hex
    row = {'request_id': request_id, 'request': body, 'started_utc': now(), 'events': [],
           'transport_complete': False, 'cleanup_errors': [], 'timed_out': False}
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=deadline_seconds)
    response = None
    sock = None
    wire = bytearray()
    deadline = time.monotonic() + deadline_seconds
    def expire():
        row['timed_out'] = True
        if sock is not None:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass
        try: conn.close()
        except OSError: pass
    timer = threading.Timer(deadline_seconds, expire)
    timer.daemon = True
    try:
        timer.start()
        conn.connect()
        sock = conn.sock
        require(time.monotonic() < deadline and not row['timed_out'], 'Chat deadline during connect')
        conn.request('POST', '/api/chat', json.dumps(body).encode('utf-8'),
                     {'Content-Type': 'application/json', 'Connection': 'close', http_checks.HEADER: request_id})
        response = conn.getresponse()
        row['status'] = response.status
        row['request_id_echo'] = response.getheader(http_checks.HEADER) == request_id
        row['content_type'] = response.getheader('Content-Type')
        require(response.status == 200 and 'text/event-stream' in (row['content_type'] or ''), 'Chat did not return SSE')
        while True:
            require(time.monotonic() < deadline and not row['timed_out'], 'Chat deadline exceeded')
            chunk = response.readline(min(65536, MAX_WIRE + 1 - len(wire)))
            if not chunk: break
            wire.extend(chunk)
            require(len(wire) <= MAX_WIRE, 'Chat wire evidence exceeded 16MiB')
        require(time.monotonic() < deadline and not row['timed_out'], 'Late chat completion')
        row['events'] = parse_sse(bytes(wire))
        row['transport_complete'] = True
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    finally:
        for name, action in [('timer-cancel', timer.cancel), ('response-close', lambda: response.close() if response else None),
                             ('connection-close', conn.close), ('timer-join', lambda: timer.join(2) if timer.ident else None)]:
            try: action()
            except BaseException as error: row['cleanup_errors'].append(f'{name}: {error}')
        row['elapsed_seconds'] = deadline_seconds - max(0., deadline-time.monotonic())
        row['ended_utc'] = now()
        (output / 'response.sse').write_bytes(wire)
        row['wire_sha256'] = hashlib.sha256(wire).hexdigest()
        save(output / 'response.json', row)
    return row


def terminal_gate(row):
    terminal = [e for e in row['events'] if e.get('done') is True]
    checks = {'transport_complete': row.get('transport_complete') is True,
              'request_id_echo': row.get('request_id_echo') is True,
              'no_transport_errors': not row.get('error') and not row.get('cleanup_errors') and not row.get('timed_out'),
              'one_done': len(terminal) == 1,
              'successful_done': len(terminal) == 1 and terminal[0].get('aborted') is False
                  and terminal[0].get('truncated') is False and not terminal[0].get('error')
                  and type(terminal[0].get('promptTokens')) is int and terminal[0]['promptTokens'] > 0
                  and row['events'][-1] is terminal[0],
              'no_escaped_client_tool_calls': not any(e.get('tool_calls') for e in row['events'])}
    return {'passed': all(checks.values()), 'checks': checks}


def invocations(row, tool):
    return [e for e in row['events'] if e.get('skill_step') == tool and e.get('ok') is True]


def validate_function(source, operation):
    try:
        node = ast.parse(source)
        expected = ast.parse('def square(x):\n    return x ' + operation + '\n')
        return ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)
    except (SyntaxError, TypeError, ValueError):
        return False


def receipt_gate(kind, value, nonce):
    if not isinstance(value, dict): return False
    common = value.get('nonce') == nonce and value.get('kind') == kind and type(value.get('pid')) is int and value['pid'] > 0
    if kind == 'skill': return common and value.get('checksum') == 16
    if kind == 'code-before': return common and value.get('result') == 8
    if kind == 'code-after': return common and value.get('result') == 49
    return False


def case_gate(case, row, artifacts, nonce):
    terminal = terminal_gate(row)
    checks = dict(terminal['checks'])
    steps = [e for e in row['events'] if 'skill_step' in e]
    checks['all_invocations_successful'] = bool(steps) and all(e.get('ok') is True for e in steps)
    def receipts(name, kind):
        matches = [a for a in artifacts if a.get('name') == name]
        return bool(matches) and any(receipt_gate(kind, a.get('json'), nonce) for a in matches)
    if case == 'skill':
        for tool in ('skills_list', 'skills_read', 'skills_run'):
            checks[tool + '_executed'] = bool(invocations(row, tool))
        checks['actual_skill_receipt'] = receipts('skill-receipt.json', 'skill')
        checks['matching_skill_run'] = any(e.get('skill') == SKILL and e.get('detail') == 'scripts/run_probe.py'
                                             for e in invocations(row, 'skills_run'))
    elif case == 'code':
        checks['write_executed'] = bool(invocations(row, 'write_file') or invocations(row, 'apply_patch'))
        checks['edit_executed'] = bool(invocations(row, 'apply_patch') or invocations(row, 'edit_file'))
        checks['read_executed'] = bool(invocations(row, 'read_file'))
        checks['two_shell_executions'] = len(invocations(row, 'shell')) >= 2
        checks['before_result'] = receipts('code-before.json', 'code-before')
        checks['after_result'] = receipts('code-after.json', 'code-after')
        sources = [a.get('text') for a in artifacts if a.get('name') == 'solution.py']
        checks['initial_source_matches'] = any(validate_function(s, '+ 1') for s in sources)
        checks['edited_source_matches'] = any(validate_function(s, '* x') for s in sources)
    elif case == 'agents':
        for tool in ('spawn_agent', 'wait_agent', 'send_input', 'list_agents', 'close_agent'):
            checks[tool + '_executed'] = bool(invocations(row, tool))
        snapshots = [a for e in row['events'] for a in (e.get('agents') or [])]
        completed = [a for a in snapshots if a.get('status') == 'completed' and not a.get('error')]
        checks['real_child_first_generation'] = any(f'FACT ALPHA {nonce} = 19' in (a.get('result') or '') for a in completed)
        checks['real_child_followup_generation'] = any(f'FACT BETA {nonce} = 23' in (a.get('result') or '') for a in completed)
        checks['child_read_tool_executed'] = any(e.get('agent_id') not in (None, '', '/root') for e in invocations(row, 'skills_read'))
        for name in ('alpha', 'beta'):
            checks['child_read_' + name] = any(e.get('agent_id') not in (None, '', '/root') and e.get('skill') == SKILL
                and e.get('detail') == 'references/' + name + '.txt' for e in invocations(row, 'skills_read'))
        by_id = {}
        for a in snapshots: by_id[a['agent_id']] = a
        checks['observed_one_child_terminal'] = len(by_id) == 1 and all(a.get('status') == 'completed' and a.get('permissions') == 'read-only'
            and a.get('workspace_id') for a in by_id.values())
        checks['exact_one_spawn'] = len(invocations(row, 'spawn_agent')) == 1
        checks['three_actual_waits'] = len(invocations(row, 'wait_agent')) >= 3
    else:
        raise ValueError('Unknown case')
    return {'passed': all(checks.values()), 'checks': checks,
            'scope': 'Actual host-emitted invocation trace, local artifact bytes and child progress. Final prose is not execution proof.'}


def prompt(case, nonce):
    if case == 'skill':
        return (f'This is a real local tool execution test. Nonce {nonce}. First call skills_list, then skills_read for '
                f'{SKILL}/SKILL.md. Follow that skill and run scripts/run_probe.py using skills_run with the nonce. '
                'Do not replace execution with an explanation, shell command, or forged receipt. Finally report the checksum and generated receipt link.')
    if case == 'code':
        return (f'Perform this small code editing and execution test in your private workspace. Nonce {nonce}. '
                'Do not use any skill. 1. Create solution.py containing only def square(x): return x + 1. '
                '2. Use shell to run Python that imports solution and calls square(7), writes code-before.json containing '
                f'kind="code-before", nonce="{nonce}", result=the actual result, pid=os.getpid(), and created_unix_ns=time.time_ns(). '
                '3. Read solution.py with read_file. Edit its function using apply_patch to return x * x. '
                '4. Use a new shell invocation to import the edited function, call square(7), and write code-after.json with '
                'kind="code-after" and the same nonce, result and process/time fields. Print each receipt. '
                'Use only Python standard library, install nothing, no network, no background commands. '
                'Retain both receipts and solution.py as downloadable files. Finally give their links and actual results.')
    return (f'Test real child-agent lifecycle. Use read-only agents, not simulated responses. Nonce {nonce}. '
            f'Spawn exactly one explorer named fact_reader. Ask it to skills_read {SKILL}/references/alpha.txt and return '
            'its exact FACT line. wait_agent until completed. Then send_input to that SAME child asking it to read '
            f'{SKILL}/references/beta.txt and return its exact FACT line; wait_agent until completed again. '
            'Call list_agents, close_agent on that child, then wait_agent once more so final progress is observed. '
            'Do not create other children or execute shell. Report both FACT lines and that close_agent returned successfully.')


def download_artifacts(client, row, output):
    artifacts = []
    seen = set()
    for event in row['events']:
        for item in event.get('files') or []:
            url = item.get('url')
            if not url or url in seen: continue
            seen.add(url)
            route = artifact_route(url)
            require(type(item.get('bytes')) is int and 0 <= item['bytes'] <= MAX_ARTIFACT, 'Artifact exceeds small fixture bound')
            # Client preserves raw HTTP response; downloaded code is never executed.
            response = extra_call(client.url.geturl(), output / ('artifact-' + str(len(artifacts))), 'GET', route, raw=True)
            require(response.get('status') == 200 and response.get('transport_complete') is True and not response.get('error')
                    and not response.get('cleanup_errors') and response.get('request_id_echo') is True, 'Artifact download failed')
            raw = (Path(response['directory']) / 'response.raw').read_bytes()
            require(len(raw) == item['bytes'] and len(raw) <= MAX_ARTIFACT, 'Artifact byte size mismatch')
            record = {'name': Path(item['name']).name, 'url': route, 'bytes': len(raw),
                      'sha256': hashlib.sha256(raw).hexdigest(), 'raw_directory': response['directory']}
            try:
                text = raw.decode('utf-8-sig')
                record['text'] = text
                try: record['json'] = json.loads(text)
                except ValueError: pass
            except UnicodeDecodeError:
                record['binary'] = True
            artifacts.append(record)
    save(output / 'artifacts.json', artifacts)
    return artifacts


def matching_workspace_paths(root, ids):
    if root is None or not root.exists(): return []
    prefixes = ['ts-session-' + ''.join(c for c in ident if c.isascii() and c.isalnum())[:64] + '-' for ident in ids]
    return [str(p) for p in root.iterdir() if p.is_dir() and any(p.name.startswith(prefix) for prefix in prefixes)]


def workspace_evidence(root, ids, creations):
    """Absence in an arbitrary empty directory is not workspace removal proof."""
    root = Path(root).resolve()
    expected = set(ids)
    observed = []
    for record in creations:
        if record['session_id'] not in expected: continue
        actual = Path(record['root']).resolve()
        prefix = 'ts-session-' + ''.join(c for c in record['session_id'] if c.isascii() and c.isalnum())[:64] + '-'
        require(actual.parent == root and actual.name.startswith(prefix), 'Observed workspace disagrees with supplied root')
        observed.append({'session_id': record['session_id'], 'root': str(actual), 'still_exists': actual.exists()})
    return {'passed': bool(expected) and expected == {r['session_id'] for r in observed}
                and all(not r['still_exists'] for r in observed), 'observed': observed,
            'scope': 'Same-request host creation log establishes exact test-owned path; absence checked after host disposal.'}


def run(args):
    endpoint = local_url(args.url)
    cases = args.cases.split(',')
    require(cases and len(cases) == len(set(cases)) and set(cases) <= {'skill', 'code', 'agents'}, 'Unique known cases required')
    output = http_checks.evidence_output_path(args.output)
    require(not output.exists(), 'Run requires fresh output evidence')
    output.mkdir(parents=True)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    fixture_root = Path(plan['fixture_root'])
    require(fixture_pins(fixture_root) == plan['files'], 'Fixture identity changed')
    report = {'schema_version': 1, 'started_utc': now(), 'completed': False, 'passed': False, 'errors': [],
              'plan': str(plan_path), 'plan_sha256': sha(plan_path), 'tool_sha256': sha(__file__),
              'http_helper_sha256': sha(CHECKER), 'requested_cases': cases, 'cases': [], 'scope': 'Single local server. Real tools and child generations; no equal-resource speed claim. Tool processes, GC, artifacts and child metadata are outside full model payload accounting. Child agents are logical model tasks, not operating-system child processes.'}
    client = http_checks.Client(args.url, output, min(args.timeout, 300))
    server_log = Path(args.server_log)
    initial_offset = server_log.stat().st_size
    session_id = None
    observer = None
    try:
        observer = ProcessObservation(args.server_pid)
        require(any(c.status == psutil.CONN_LISTEN and c.laddr.port == (endpoint.port or 80)
                    for c in observer.server.net_connections(kind='tcp')), 'Specified server PID does not own the listening port')
        report['server_identity_before'] = observer.identity()
        observer.thread.start()
        state = client.call('models', '/api/models')
        model = http_checks.expect_json(state)
        require(model.get('unifiedMemory') is True and model.get('architecture') == 'qwen4exp', 'Unified Qwen session required')
        report['model'] = state
        skills = client.call('skills', '/api/skills')
        catalog = http_checks.expect_json(skills)
        require(catalog.get('allowScripts') is True, 'Skill script execution unavailable')
        discovery = client.call('skill-discovery', '/v1/skills')
        require(SKILL in json.dumps(http_checks.expect_json(discovery)), 'Fixture skill not discovered')
        report['skills'] = skills
        for case in cases:
            case_dir = output / case
            case_dir.mkdir()
            created = client.call('create-' + case, '/api/sessions', {})
            session_id = http_checks.expect_json(created)['sessionId']
            require(re.fullmatch(r'[A-Za-z0-9_-]+', session_id) is not None, 'Invalid fresh session ID')
            body = {'sessionId': session_id, 'messages': [{'role': 'user', 'content': prompt(case, plan['nonce'])}],
                    'maxTokens': args.max_tokens, 'think': False, 'temperature': 0, 'top_p': 1,
                    'skills_discovery': case != 'code', 'multi_agent': case == 'agents'}
            if case == 'code': body['skills'] = []
            started_unix = time.time()
            row = stream_chat(args.url, body, case_dir, args.timeout)
            ended_unix = time.time()
            row['case'] = case
            report['cases'].append(row)
            artifacts = download_artifacts(client, row, case_dir)
            row['quality'] = case_gate(case, row, artifacts, plan['nonce'])
            idle, samples = http_checks.wait_idle(client, 'idle-before-dispose-' + case)
            row['queue_before_dispose'] = samples
            require(idle, 'Request/children have not drained; do not start another test')
            log_bytes = read_log(server_log, initial_offset)
            text = log_bytes.decode('utf-8', errors='replace')
            row['server_evidence'] = request_log_evidence(text, row['request_id'])
            row['quality']['checks']['last_same_request_eos'] = row['server_evidence']['last_terminal_natural_eos']
            if case == 'code':
                row['quality']['checks']['two_successful_process_exit_logs'] = row['server_evidence']['successful_shell_exits'] >= 2
            if case == 'skill':
                row['quality']['checks']['successful_fixture_process_exit_log'] = row['server_evidence']['successful_fixture_script_exits'] >= 1
            if case in ('skill', 'code'):
                row['receipt_processes'] = receipt_processes_exited(artifacts, started_unix, ended_unix)
                row['quality']['checks']['receipt_processes_exited'] = row['receipt_processes']['passed']
            child_ids = {a['workspace_id'] for e in row['events'] for a in (e.get('agents') or []) if a.get('workspace_id')}
            row['observed_child_workspace_ids'] = sorted(child_ids)
            row['remaining_child_workspaces'] = matching_workspace_paths(Path(args.agent_workspaces), child_ids)
            row['quality']['checks']['child_workspaces_removed'] = not row['remaining_child_workspaces'] and (case != 'agents' or
                bool(child_ids) and Path(args.agent_workspaces).is_dir())
            if case == 'agents':
                row['child_workspace_evidence'] = workspace_evidence(args.agent_workspaces, child_ids, row['server_evidence']['workspace_creations'])
                row['quality']['checks']['child_workspace_lifecycle_observed'] = row['child_workspace_evidence']['passed']
            # Dispose the real persistent session and inspect only its own scratch namespace.
            removed = extra_call(args.url, output / ('delete-' + case), 'DELETE', '/api/sessions/' + quote(session_id))
            require(http_checks.expect_json(removed).get('ok') is True and not removed['cleanup_errors']
                    and removed.get('request_id_echo') is True, 'Session deletion failed')
            row['session_delete'] = removed
            row['remaining_session_workspaces'] = matching_workspace_paths(Path(args.server_scratch), [session_id])
            row['quality']['checks']['session_workspace_removed'] = not row['remaining_session_workspaces'] and Path(args.server_scratch).is_dir()
            row['session_workspace_evidence'] = workspace_evidence(args.server_scratch, [session_id], row['server_evidence']['workspace_creations'])
            row['quality']['checks']['session_workspace_lifecycle_observed'] = row['session_workspace_evidence']['passed']
            session_id = None
            idle, samples = http_checks.wait_idle(client, 'idle-' + case)
            row['queue_after'] = samples
            row['quality']['checks']['queue_drained'] = idle
            row['quality']['passed'] = all(row['quality']['checks'].values())
            save(output / 'report.json', report)
            require(row.get('transport_complete') is True, 'Incomplete request; remaining cases not executed')
        report['completed'] = True
    except BaseException as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        if session_id:
            try:
                cleanup = extra_call(args.url, output / 'cleanup-session', 'DELETE', '/api/sessions/' + quote(session_id))
                report['cleanup_session'] = cleanup
                require(http_checks.expect_json(cleanup).get('ok') is True, 'Final session deletion failed')
            except BaseException as error: report['errors'].append('Session cleanup: ' + str(error))
        try:
            require(fixture_pins(fixture_root) == plan['files'] and sha(plan_path) == report['plan_sha256'], 'Fixture/plan changed')
            require(sha(__file__) == report['tool_sha256'] and sha(CHECKER) == report['http_helper_sha256'], 'Runner/helper identity changed')
            raw = read_log(server_log, initial_offset)
            (output / 'server-new.log').write_bytes(raw)
            report['server_log_sha256'] = hashlib.sha256(raw).hexdigest()
        except BaseException as error: report['errors'].append('Final identity/evidence: ' + str(error))
        if observer is not None:
            try:
                report['server_identity_after'] = observer.identity()
                require(report['server_identity_after'] == report['server_identity_before'], 'Server identity changed')
            except BaseException as error: report['errors'].append('Server identity: ' + str(error))
            try:
                report['process_observation'] = observer.finish()
                require(report['process_observation']['passed'], 'Observed process cleanup incomplete')
            except BaseException as error: report['errors'].append('Process cleanup observation: ' + str(error))
        report['finished_utc'] = now()
        report['passed'] = report['completed'] and not report['errors'] and len(report['cases']) == len(cases) and all(r.get('quality', {}).get('passed') is True for r in report['cases'])
        save(output / 'report.json', report)
    print(json.dumps({'report': str(output / 'report.json'), 'passed': report['passed'], 'errors': report['errors']}))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--output', required=True)
    exe = sub.add_parser('run')
    for flag in ('url', 'plan', 'output', 'server-log', 'server-scratch', 'agent-workspaces'):
        exe.add_argument('--' + flag, required=True)
    exe.add_argument('--cases', default='skill,code,agents')
    exe.add_argument('--timeout', type=timeout, default=1800)
    exe.add_argument('--max-tokens', type=int, default=1024)
    exe.add_argument('--server-pid', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': return prepare(args.output)
    require(1 <= args.max_tokens <= 4096, 'max-tokens must be in [1,4096]')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
