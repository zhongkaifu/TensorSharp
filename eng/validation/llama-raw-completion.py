#!/usr/bin/env python3
"""Run one pinned raw-token llama.cpp completion in an owned, bounded server.

No chat template, grammar, schema, output repair, retry or throughput comparison
is implicit. The reviewed plan specifies exact integer prompt IDs and deployment
pins. A completed HTTP response, natural EOS, strict semantics and process cleanup
are separate evidence. Source and checkpoint files are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import struct
import sys
import threading
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


owned = load_module('raw_owned_server', 'llama-planned-chat.py')
bench = owned.bench
semantic = load_module('raw_semantic_check', 'qwen38-strata-compare.py')


def token_hash(tokens):
    """Canonical checkpoint IDs, never a hash of retokenized response text."""
    if not isinstance(tokens, list) or not tokens or any(type(x) is not int or not 0 <= x <= 2147483647 for x in tokens):
        raise ValueError('Require nonempty signed-int32 nonnegative token IDs')
    return hashlib.sha256(struct.pack('<' + 'i' * len(tokens), *tokens)).hexdigest()


def validate_plan(plan, body):
    endpoint = urlsplit(plan['futureRequest']['endpoint'])
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or not endpoint.port
            or endpoint.path != '/completion' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password):
        raise ValueError('Require an explicit loopback raw /completion endpoint')
    command = plan['futureCommand']
    if not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError('Require a nonempty command argument array')
    for option, value in (('--host', '127.0.0.1'), ('--port', str(endpoint.port)),
                          ('--parallel', '1'), ('--spec-type', 'none')):
        if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != value:
            raise ValueError('Server arguments differ from the owned scalar endpoint')
    prompt = body.get('prompt')
    if (not isinstance(prompt, list) or not prompt or any(type(x) is not int or not 0 <= x <= 2147483647 for x in prompt)
            or prompt != plan['prompt']['tokens'] or len(prompt) != plan['prompt']['count']
            or body != plan['futureRequest']['body']):
        raise ValueError('Raw integer prompt/request differs from the reviewed plan')
    if 'i32le_sha256' in plan['prompt'] and token_hash(prompt) != plan['prompt']['i32le_sha256']:
        raise ValueError('Raw prompt int32LE digest differs')
    expected = dict(temperature=0.0, samplers=['temperature'], top_k=0, top_p=1.0, min_p=0.0,
        repeat_penalty=1.0, repeat_last_n=0, presence_penalty=0.0, frequency_penalty=0.0,
        dry_multiplier=0.0, cache_prompt=False, return_tokens=True, n_probs=0, stream=False,
        ignore_eos=False, stop=[])
    if set(body) - set(expected) - {'prompt', 'n_predict', 'seed'}:
        raise ValueError('Unreviewed output-changing request field')
    if any(body.get(k) != value or (isinstance(value, bool) and body.get(k) is not value)
           or (type(value) in (int, float) and type(body.get(k)) not in (int, float))
           for k, value in expected.items()):
        raise ValueError('Require unmodified greedy sampling, EOS and no output constraints')
    if type(body.get('n_predict')) is not int or body['n_predict'] <= 0:
        raise ValueError('Require a positive integer output bound')
    if command.count('--ctx-size') != 1 or command.index('--ctx-size') + 1 >= len(command):
        raise ValueError('Require one explicit context capacity')
    if int(command[command.index('--ctx-size') + 1]) < len(prompt) + body['n_predict']:
        raise ValueError('Context must contain the complete requested trajectory')
    eos = plan['expected_checkpoint_eos_ids']
    if not isinstance(eos, list) or not eos or any(type(x) is not int or not 0 <= x <= 2147483647 for x in eos):
        raise ValueError('Require actual checkpoint EOS IDs')
    return endpoint


def quality_result(response, body, eos, case):
    tokens = response.get('tokens')
    valid_tokens = (isinstance(tokens, list) and 0 < len(tokens) <= body['n_predict']
        and all(type(x) is int and 0 <= x <= 2147483647 for x in tokens))
    prompt_matches = (type(response.get('tokens_evaluated')) is int
        and response['tokens_evaluated'] == len(body['prompt']))
    output_count_matches = (valid_tokens and type(response.get('tokens_predicted')) is int
        and response['tokens_predicted'] == len(tokens))
    natural_eos = (valid_tokens and tokens[-1] in eos and not any(t in eos for t in tokens[:-1])
        and response.get('stop_type') == 'eos' and response.get('truncated') is False)
    content = response.get('content')
    if case == 'free_text':
        # A bounded prefix is legitimate observation, not semantic completion.
        # llama's `truncated` denotes context truncation, not n_predict exhaustion.
        bounded_limit = (valid_tokens and len(tokens) == body['n_predict']
            and not any(t in eos for t in tokens) and response.get('stop_type') == 'limit'
            and response.get('truncated') is False)
        trajectory = bool(prompt_matches and output_count_matches and isinstance(content, str)
                          and (natural_eos or bounded_limit))
        return dict(passed=trajectory, trajectory_passed=trajectory, record_only=True,
            semantic_pass=None, semantic_check=dict(passed=None, reason='Free text is unassessed; no factual oracle applied'),
            natural_eos=bool(natural_eos), bounded_limit=bool(bounded_limit),
            prompt_token_count_matches=prompt_matches, returned_token_count_matches=bool(output_count_matches),
            returned_tokens_valid=valid_tokens, generated_tokens_i32le_sha256=token_hash(tokens) if valid_tokens else None,
            content=content, generated_tokens=tokens, stop_type=response.get('stop_type'), truncated=response.get('truncated'),
            scope='Record-only exact raw-ID conditioning/count/stop validation. PASS is not semantic or factual acceptance; capped output is not a completed answer.')
    check = semantic.semantic_check(case, content, natural_eos and prompt_matches) if isinstance(content, str) else {
        'passed': False, 'reason': 'Missing original string content'}
    return dict(passed=bool(check['passed'] and natural_eos and prompt_matches and output_count_matches),
        natural_eos=bool(natural_eos), prompt_token_count_matches=prompt_matches,
        returned_token_count_matches=bool(output_count_matches),
        returned_tokens_valid=valid_tokens, semantic_check=check, content=content,
        generated_tokens=tokens, stop_type=response.get('stop_type'), truncated=response.get('truncated'),
        scope='Exact submitted raw IDs and returned count; original response text is not repaired.')


def conditioning_evidence(plan):
    """Bind record-only plans to the separately validated actual Server IDs."""
    if plan.get('case') != 'free_text':
        return None
    if plan.get('record_only') is not True or plan.get('semantic_pass') is not None:
        raise ValueError('Free text must be explicit record-only, semantically unassessed')
    pin = plan['conditioning_evidence']
    raw = Path(pin['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Server token export changed')
    export = json.loads(raw)
    index = pin['request_index']
    if type(index) is not int or index < 0 or index >= len(export['requests']):
        raise ValueError('Invalid conditioning request index')
    row = export['requests'][index]
    if (export.get('status') != 'extracted-not-baseline-executed'
            or row['http_request_id'] != pin['http_request_id']
            or row['prompt'] != plan['prompt'] or row['output'] != plan['reference_output']
            or export['identity']['checkpoint']['shards'] != plan['checkpoint']['shards']
            or token_hash(row['prompt']['tokens']) != row['prompt']['i32le_sha256']
            or token_hash(row['output']['tokens']) != row['output']['i32le_sha256']):
        raise ValueError('Server conditioning/reference identity differs from plan')
    return dict(path=pin['path'], sha256=pin['sha256'], http_request_id=row['http_request_id'],
                prompt_i32le_sha256=row['prompt']['i32le_sha256'])


def compare_trajectory(tokens, reference):
    """Descriptive only: a divergence is not an execution/semantic verdict."""
    expected = reference['tokens']
    common = min(len(tokens), len(expected))
    mismatch = next((i for i in range(common) if tokens[i] != expected[i]), None)
    return dict(compared_token_count=common, compared_prefix_exact=mismatch is None,
        full_trajectory_exact=tokens == expected, first_mismatch_index=mismatch,
        reference_token=expected[mismatch] if mismatch is not None else None,
        observed_token=tokens[mismatch] if mismatch is not None else None,
        reference_count=len(expected), observed_count=len(tokens),
        scope='Recorded raw-ID agreement only, including any EOS. Matching capped prefixes do not establish complete-answer or factual correctness.')


def start_watchdog(server, timeout, timed_out, errors):
    def expire():
        if server.poll() is None:
            timed_out.set()
            try:
                server.kill()
            except Exception as error:
                errors.append(repr(error))
    timer = threading.Timer(timeout, expire)
    timer.daemon = True
    timer.start()
    return timer


def deployment_hashes(plan):
    entries = plan['deployment']['files']
    if not entries or any(Path(row['path']).name != name for name, row in entries.items()):
        raise ValueError('Invalid deployment names/paths')
    hashes = {name: bench.sha(row['path']) for name, row in entries.items()}
    if hashes != {name: row['sha256'] for name, row in entries.items()}:
        raise ValueError('Deployment changed from reviewed pins')
    return hashes


def observe_modules(pid, plan, timeout):
    if sys.platform == 'win32':
        command = ['powershell', '-NoProfile', '-Command',
                   f'(Get-Process -Id {int(pid)} -ErrorAction Stop).Modules | Select-Object -ExpandProperty FileName']
        result = subprocess.run(command, capture_output=True, text=True, check=True,
                                timeout=min(30, timeout), creationflags=bench.PROCESS_FLAGS)
        paths = result.stdout.splitlines()
    else:
        paths = [row.path for row in bench.psutil.Process(pid).memory_maps(grouped=True)]
    expected = plan['deployment']['files']
    seen = {}
    for value in paths:
        path = Path(value.strip())
        name = path.name
        if name not in expected:
            continue
        if name in seen or path.resolve() != Path(expected[name]['path']).resolve():
            raise ValueError('Loaded deployment path differs: ' + str(path))
        digest = bench.sha(path)
        if digest != expected[name]['sha256']:
            raise ValueError('Loaded deployment digest differs: ' + str(path))
        seen[name] = dict(path=str(path), sha256=digest)
    required = {Path(plan['futureCommand'][0]).name, 'llama.dll', 'ggml-cuda.dll'} if sys.platform == 'win32' else {
        Path(plan['futureCommand'][0]).name, 'libllama.so', 'libggml-cuda.so'}
    if not required.issubset(seen):
        raise ValueError('Required loaded executable/llama/CUDA identities unavailable: ' + repr(required - set(seen)))
    return seen


def run(plan, body, request_bytes, output, timeout, case):
    endpoint = validate_plan(plan, body)
    # Earlier reviewed plans were exclusively tool_json. Other semantic tasks
    # must explicitly bind their checker in the plan instead of relabelling it.
    if case != plan.get('case', 'tool_json'):
        raise ValueError('Semantic case differs from the reviewed plan')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, passed=False, quality_passed=None if case == 'free_text' else False,
        semantic_pass=None if case == 'free_text' else False, trajectory_passed=False,
        record_only=case == 'free_text',
        passed_scope='Owned execution and valid raw-token trajectory only; semantics unassessed' if case == 'free_text' else 'Owned execution and fixed-task semantic check',
        timed_out=False,
        timeout_seconds=timeout, timeout_scope='One server lifetime including load and HTTP; preflight hashes excluded',
        command=plan['futureCommand'], plan=plan, case=case, started_unix=time.time(),
        harness_sha256=bench.sha(__file__), semantic_checker_sha256=bench.sha(semantic.__file__),
        helper_sha256=bench.sha(owned.__file__), limitations=[
            'One raw prompt only; no matched-budget or quiet-throughput claim.',
            'Checkpoint hashes reused with current size/mtime checks; OS page cache uncontrolled.',
            'Board samples include desktop; process termination is not native allocator cleanup validation.',
            'The owned server is terminated after the response; its teardown exit code is not a natural model exit.'])
    def save(name, value):
        tmp = output / (name + '.tmp')
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        tmp.replace(output / name)
    save('execution.json', report)
    (output / 'request.json').write_bytes(request_bytes)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    server = sampler = timer = None
    deadline = None
    timed_out = threading.Event()
    watchdog_errors = []
    try:
        report['conditioning_evidence'] = conditioning_evidence(plan)
        with socket.socket() as check:
            check.bind(('127.0.0.1', endpoint.port))
        source = plan['source']
        revision = subprocess.check_output(['git', '-C', source['path'], 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', source['path'], 'status', '--porcelain'], text=True).strip()
        if dirty or revision != source['revision']:
            raise ValueError('Upstream source revision/clean state changed')
        report['deployment_before'] = deployment_hashes(plan)
        command = plan['futureCommand']
        if Path(command[0]).resolve() != Path(plan['deployment']['files'][Path(command[0]).name]['path']).resolve():
            raise ValueError('Executable is not the pinned deployment')
        if command.count('--model') != 1 or command.index('--model') + 1 >= len(command):
            raise ValueError('Require one pinned model entrypoint')
        shards = plan['checkpoint']['shards']
        if not shards or Path(command[command.index('--model') + 1]).resolve() != Path(shards[0]['path']).resolve():
            raise ValueError('Model differs from reviewed checkpoint')
        def checkpoint_state():
            result = []
            for row in shards:
                stat = Path(row['path']).stat()
                if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                    raise ValueError('Checkpoint metadata changed: ' + row['path'])
                result.append(dict(row))
            return result
        report['checkpoint_before'] = checkpoint_state()
        integrity = plan['checkpoint']
        if bench.sha(integrity['integrity_manifest']) != integrity['integrity_manifest_sha256']:
            raise ValueError('Prior checkpoint integrity manifest changed')
        controls = plan['environment']
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(tuple(controls['clear_prefixes']))}
        env.update(controls['set'])
        report['environment'] = {k: v for k, v in env.items() if k.upper().startswith(
            ('LLAMA_', 'GGML_', 'TS_', 'TENSORSHARP_', 'STRATA_', 'OMP_', 'GOMP_', 'CUDA_', 'KMP_', 'MKL_', 'OPENBLAS_'))}
        with (output / 'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command, cwd=Path(command[0]).parent, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, creationflags=bench.PROCESS_FLAGS)
            started = time.monotonic()
            deadline = started + timeout
            report['server_pid'] = server.pid
            timer = start_watchdog(server, timeout, timed_out, watchdog_errors)
            sampler = bench.Sampler(output / 'memory.jsonl')
            sampler.process = bench.psutil.Process(server.pid)
            sampler.thread.start()
            save('execution.json', report)
            def remaining():
                left = deadline - time.monotonic()
                if timed_out.is_set() or left <= 0:
                    timed_out.set()
                    raise TimeoutError('Total raw-completion lifetime exceeded')
                if server.poll() is not None:
                    raise RuntimeError(f'Owned server exited early: {server.returncode}')
                return left
            while True:
                try:
                    owned.request(endpoint.port, '/health', None, min(2, remaining()))
                    break
                except (OSError, TimeoutError, RuntimeError):
                    time.sleep(min(.5, remaining()))
            report['load_ready_seconds'] = time.monotonic() - started
            listeners = [c for c in sampler.process.net_connections(kind='tcp') if c.status == bench.psutil.CONN_LISTEN
                and c.laddr.port == endpoint.port and c.laddr.ip == '127.0.0.1']
            if len(listeners) != 1:
                raise ValueError('Healthy listener does not belong to the owned server')
            report['owned_listener_observed'] = dict(pid=server.pid, host='127.0.0.1', port=endpoint.port)
            report['loaded_modules'] = observe_modules(server.pid, plan, remaining())
            save('execution.json', report)
            connection = http.client.HTTPConnection('127.0.0.1', endpoint.port, timeout=remaining())
            http_started = time.monotonic()
            try:
                connection.request('POST', '/completion', request_bytes, {'Content-Type': 'application/json'})
                response = connection.getresponse()
                report['http_status'] = response.status
                raw = response.read()
                (output / 'response.raw.json').write_bytes(raw)
                report['http_wall_seconds'] = time.monotonic() - http_started
                if response.status != 200:
                    raise RuntimeError('Raw completion HTTP ' + str(response.status))
                response_data = json.loads(raw)
                save('response.json', response_data)
                report['quality'] = quality_result(response_data, body, plan['expected_checkpoint_eos_ids'], case)
                report['trajectory_passed'] = report['quality']['passed']
                report['quality_passed'] = None if case == 'free_text' else report['quality']['passed']
                report['semantic_pass'] = None if case == 'free_text' else report['quality']['passed']
                if case == 'free_text' and report['quality']['returned_tokens_valid']:
                    report['reference_comparison'] = compare_trajectory(response_data['tokens'], plan['reference_output'])
                report['timings'] = response_data.get('timings')
                report['complete'] = True
            finally:
                connection.close()
            remaining()
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if not isinstance(error, Exception):
            raise
    finally:
        report['timed_out'] = timed_out.is_set() or (deadline is not None and time.monotonic() >= deadline)
        report.update(owned.cleanup_owned(server, timer, sampler))
        report['watchdog_errors'] = watchdog_errors
        report['resource_observation_valid'] = bool(sampler is not None and sampler.rows and not sampler.errors)
        try:
            report['deployment_after'] = deployment_hashes(plan)
            if case == 'free_text' and 'conditioning_evidence' in report:
                report['conditioning_evidence_after'] = conditioning_evidence(plan)
            if 'checkpoint_before' in report:
                report['checkpoint_after'] = checkpoint_state()
        except Exception as error:
            report['identity_after_error'] = repr(error)
        report['finished_unix'] = time.time()
        report['passed'] = (report['complete'] and report['trajectory_passed'] and report['owned_server_exited']
            and not report['timed_out'] and not report.get('error') and not report['cleanup_errors']
            and not watchdog_errors and not report.get('identity_after_error'))
        save('execution.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(semantic.CASES) + ('free_text',), default='tool_json')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Require a positive finite timeout')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    request_bytes = args.request.read_bytes()
    body = json.loads(request_bytes)
    result = run(plan, body, request_bytes, bench.evidence_directory(args.output), args.timeout, args.case)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
