#!/usr/bin/env python3
"""Execute one reviewed llama.cpp chat plan in an owned, bounded server.

The JSON plan records futureCommand, futureRequest, clean source revision,
checkpoint manifest and a prior deployment manifest. This harness never builds,
patches or downloads. Normal HTTP stop is recorded separately from exact EOS IDs.
No HTTP latency is advertised as matched teacher-forced throughput.
"""
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
import threading
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('planned_chat_bench', HERE / 'multimodel-quality-bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def quality_result(response, expected):
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        return {'passed': False, 'reason': 'Expected exactly one completed choice'}
    choice = choices[0]
    message = choice.get('message') or {}
    content = message.get('content')
    normal_stop = choice.get('finish_reason') == 'stop'
    return {'passed': normal_stop and isinstance(content, str) and content.strip() == expected
            and not message.get('tool_calls') and not message.get('function_call'),
            'normal_stop': normal_stop, 'checkpoint_eos_id_observed': False,
            'finish_reason': choice.get('finish_reason'), 'content': content,
            'reasoning': message.get('reasoning_content', message.get('reasoning')),
            'tool_calls': message.get('tool_calls'),
            'scope': 'One answer with normal HTTP stop; response does not expose exact EOS token ID.'}


def plan_endpoint(plan):
    endpoint = urlsplit(plan['futureRequest']['endpoint'])
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or not endpoint.port
            or endpoint.path != '/v1/chat/completions' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password):
        raise ValueError('Require an explicit loopback chat endpoint')
    command = plan['futureCommand']
    if not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError('Require a nonempty argument array')
    for option, value in (('--host', '127.0.0.1'), ('--port', str(endpoint.port)), ('--parallel', '1')):
        if (command.count(option) != 1 or command.index(option) + 1 >= len(command)
                or command[command.index(option) + 1] != value):
            raise ValueError('Server arguments do not match the owned single-slot endpoint')
    body = plan['futureRequest']['body']
    if body.get('stream') is not False or body.get('stop') or body.get('tools') or body.get('n', 1) != 1:
        raise ValueError('Require a single nonstreaming answer without custom stop or tools')
    return endpoint


def request(port, path, body, timeout):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        connection.request('GET' if body is None else 'POST', path,
                           None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8'),
                           {} if body is None else {'Content-Type': 'application/json'})
        result = connection.getresponse()
        text = result.read().decode('utf-8')
        if result.status != 200:
            raise RuntimeError(f'HTTP {result.status}: {text[:1000]}')
        return json.loads(text)
    finally:
        connection.close()


def cleanup_owned(server, timer, sampler):
    """Best effort for each owned resource, preserving every failure for the report."""
    result = {'cleanup_errors': [], 'owned_server_exited': server is None}
    def attempt(label, action):
        try:
            return action()
        except Exception as error:
            result['cleanup_errors'].append(f'{label}: {error!r}')
            return None
    if timer is not None:
        attempt('timer.cancel', timer.cancel)
        attempt('timer.join', lambda: timer.join(timeout=15))
        if attempt('timer.is_alive', timer.is_alive) is not False:
            result['cleanup_errors'].append('watchdog did not stop')
    if server is not None:
        if attempt('server.poll', server.poll) is None:
            attempt('server.terminate', server.terminate)
            attempt('server.wait after terminate', lambda: server.wait(timeout=10))
        if attempt('server.poll after terminate', server.poll) is None:
            attempt('server.kill', server.kill)
            attempt('server.wait after kill', lambda: server.wait(timeout=10))
        result['server_exit'] = attempt('server.final_poll', server.poll)
        result['owned_server_exited'] = result['server_exit'] is not None
        if not result['owned_server_exited']:
            result['cleanup_errors'].append('owned server did not exit')
    if sampler is not None:
        attempt('sampler.stop', sampler.stop.set)
        attempt('sampler.join', lambda: sampler.thread.join(timeout=15))
        result['sampling_errors'] = sampler.errors
        result['sampler_stopped'] = attempt('sampler.is_alive', sampler.thread.is_alive) is False
        if not result['sampler_stopped']:
            result['cleanup_errors'].append('sampler did not stop')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-answer', required=True)
    parser.add_argument('--direct-report', type=Path, help='Optional completed DirectCudaModelProbe JSON for exact prompt/token comparison')
    parser.add_argument('--case', default='math')
    parser.add_argument('--timeout', type=float, default=2700, help='Total server lifetime, including model loading and all HTTP calls')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Require a positive finite timeout')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    endpoint = plan_endpoint(plan)
    out = bench.evidence_directory(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, quality_passed=False, timed_out=False, plan=plan,
                  timeout_seconds=args.timeout, harness_sha256=bench.sha(__file__),
                  helper_sha256=bench.sha(bench.__file__),
                  limitations=['No equal physical hard-budget assertion.',
                    'File cache is uncontrolled; checkpoint hashes are reused from prior whole-file verification with current size checks.',
                    'RSS and board VRAM are samples, not allocator peaks; board VRAM includes other processes.',
                    'On-disk deployment identity is not live module identity. Server termination is not native cleanup validation.',
                    'Normal HTTP stop is not evidence of a particular EOS token ID.'])
    def save(name, value):
        temporary = out / (name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(out / name)
    save('execution.json', report)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    server = sampler = timer = None
    deadline = None
    timed_out = threading.Event()
    started = time.monotonic()
    try:
        direct_case = None
        if args.direct_report:
            direct = json.loads(args.direct_report.read_text(encoding='utf-8-sig'))
            cases = [row for row in direct['quality'] if row['name'] == args.case]
            if len(cases) != 1 or not isinstance(cases[0]['rendered_prompt'], str) or not isinstance(cases[0]['prompt_tokens'], list):
                raise ValueError('Direct report must contain one selected quality case with prompt/token evidence')
            direct_case = cases[0]
            report['direct_report_sha256'] = bench.sha(args.direct_report)
        with socket.socket() as port_check:
            port_check.bind(('127.0.0.1', endpoint.port))
        source = plan['source']
        revision = subprocess.check_output(['git', '-C', source['path'], 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', source['path'], 'status', '--porcelain'], text=True).strip()
        if dirty or revision != source['revision']:
            raise ValueError('Reference source revision/clean state differs from the reviewed plan')
        deployment_manifest = args.plan.parent / plan['deployment']['priorCompleteDeploymentManifest']
        deployment = json.loads(deployment_manifest.read_text(encoding='utf-8-sig'))['deployment']
        def deployment_hashes():
            return {entry['path']: bench.sha(entry['path']) for entry in deployment}
        report['deployment_before'] = deployment_hashes()
        if report['deployment_before'] != {entry['path']: entry['sha256'] for entry in deployment}:
            raise ValueError('Deployment differs from the pinned complete deployment manifest')
        command = plan['futureCommand']
        model = plan['checkpoint']['entrypoint']
        if (command.count('--model') != 1 or command.index('--model') + 1 >= len(command)
                or Path(command[command.index('--model') + 1]).resolve() != Path(model).resolve()):
            raise ValueError('Command checkpoint differs from the plan')
        if Path(command[0]).resolve() not in {Path(x['path']).resolve() for x in deployment}:
            raise ValueError('Executable is not included in the pinned deployment')
        report['checkpoint_before'] = bench.verified_identities(bench.checkpoint_files(model), plan['checkpoint']['verifiedManifest'])
        body = dict(plan['futureRequest']['body'], timings=True)
        save('request.json', body)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'TS_', 'TENSORSHARP_', 'OMP_', 'GOMP_', 'NCCL_'))}
        report['environment'] = {'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '8'}
        env.update(report['environment'])
        with (out / 'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command, cwd=Path(command[0]).parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=bench.PROCESS_FLAGS)
            report['server_pid'] = server.pid
            started = time.monotonic()
            deadline = started + args.timeout
            # Independent watchdog: a socket timeout is an idle timeout, not a total deadline.
            def expire():
                if server.poll() is None:
                    timed_out.set()
                    server.kill()
            timer = threading.Timer(args.timeout, expire)
            timer.daemon = True
            timer.start()
            sampler = bench.Sampler(out / 'memory.jsonl')
            sampler.process = bench.psutil.Process(server.pid)
            sampler.thread.start()
            save('execution.json', report)
            def remaining():
                if timed_out.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError('Total server lifetime exceeded')
                if server.poll() is not None:
                    raise RuntimeError(f'Owned server exited: {server.returncode}')
                return deadline - time.monotonic()
            while True:
                left = remaining()
                try:
                    request(endpoint.port, '/health', None, min(2, left))
                    break
                except (OSError, TimeoutError, RuntimeError):
                    time.sleep(min(.5, max(0, deadline - time.monotonic())))
            report['load_ready_seconds'] = time.monotonic() - started
            listeners = [connection for connection in sampler.process.net_connections(kind='tcp')
                         if connection.status == bench.psutil.CONN_LISTEN and connection.laddr.port == endpoint.port
                         and connection.laddr.ip == '127.0.0.1']
            if len(listeners) != 1:
                raise RuntimeError('Healthy endpoint is not confirmed as the owned process listener')
            report['owned_listener_observed'] = {'pid': server.pid, 'host': '127.0.0.1', 'port': endpoint.port}
            rendered = request(endpoint.port, '/apply-template', body, remaining())
            tokens = request(endpoint.port, '/tokenize', {'content': rendered['prompt'], 'add_special': True, 'parse_special': True}, remaining())
            prompt = {'rendered': rendered, 'tokenized': tokens,
                      'rendered_utf8_sha256': hashlib.sha256(rendered['prompt'].encode('utf-8')).hexdigest()}
            if direct_case is not None:
                prompt.update(direct_report_sha256=report['direct_report_sha256'],
                              rendered_text_exact=rendered['prompt'] == direct_case['rendered_prompt'],
                              prompt_tokens_exact=tokens['tokens'] == direct_case['prompt_tokens'])
            report['prompt_comparability'] = {
                'direct_report_provided': direct_case is not None,
                'rendered_text_exact': prompt.get('rendered_text_exact'),
                'prompt_tokens_exact': prompt.get('prompt_tokens_exact'),
                'same_input_control': (prompt['rendered_text_exact'] and prompt['prompt_tokens_exact']) if direct_case is not None else None,
                'scope': 'Semantic correctness is independent of prompt parity. A mismatch cannot qualify as a same-input engine comparison.'}
            save('prompt.json', prompt)
            request_start = time.monotonic()
            response = request(endpoint.port, endpoint.path, body, remaining())
            if time.monotonic() >= deadline:
                timed_out.set()  # Watchdog scheduling delay cannot turn a late response into PASS.
            report['request_wall_seconds'] = time.monotonic() - request_start
            save('response.json', response)
            report['quality'] = quality_result(response, args.expected_answer)
            report['complete'] = True
    except Exception as error:
        report['error'] = repr(error)
    finally:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out.set()
        report.update(cleanup_owned(server, timer, sampler))
        try:
            if 'deployment_before' in report:
                report['deployment_after'] = deployment_hashes()
                report['deployment_unchanged'] = report['deployment_before'] == report['deployment_after']
            if 'checkpoint_before' in report:
                report['checkpoint_after'] = bench.verified_identities(bench.checkpoint_files(model), plan['checkpoint']['verifiedManifest'])
                report['checkpoint_metadata_unchanged'] = report['checkpoint_before'] == report['checkpoint_after']
        except Exception as error:
            report['identity_error'] = repr(error)
        report['timed_out'] = timed_out.is_set()
        report['wall_seconds'] = time.monotonic() - started
        report['quality_passed'] = (report['complete'] and not report['timed_out'] and not report.get('error')
            and not report['cleanup_errors']
            and report.get('quality', {}).get('passed') is True and report.get('deployment_unchanged') is True
            and report.get('checkpoint_metadata_unchanged') is True and report.get('owned_server_exited') is True)
        save('execution.json', report)
    return 0 if report['quality_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
