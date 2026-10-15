#!/usr/bin/env python3
"""Own a local llama.cpp reference process and bind the strict quality client to it.

The quality checker remains independent of TensorSharp's output. Reported engine
timings have different token denominators; this is not a matched-throughput oracle.
"""
from __future__ import annotations
import argparse
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import psutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench', HERE/'multimodel-quality-bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
cleanup_spec = importlib.util.spec_from_file_location('llama_owned_cleanup', HERE/'llama-planned-chat.py')
cleanup = importlib.util.module_from_spec(cleanup_spec)
cleanup_spec.loader.exec_module(cleanup)


def run_client(command, timeout, log, report):
    """Bound and reap the separately owned quality client, including interrupts."""
    client = None
    try:
        client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=bench.PROCESS_FLAGS)
        report['client_pid'] = client.pid
        report['client_exit'] = client.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        report['timed_out'] = True
        raise TimeoutError('Total reference lifetime exceeded during quality requests')
    finally:
        report['client_cleanup'] = cleanup.cleanup_owned(client, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-manifest', type=Path)
    parser.add_argument('--port', type=int, default=52176)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--cases', default='squares,tool_json,code,long_extract')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600,
                        help='Total owned server lifetime, including load and all quality requests')
    parser.add_argument('--reasoning-budget', type=int)
    parser.add_argument('--server-arg', action='append', default=[])
    args = parser.parse_args()
    out = bench.evidence_directory(args.output)
    if min(args.context, args.max_tokens, args.repetitions, args.timeout) <= 0 or not math.isfinite(args.timeout):
        parser.error('Require positive request limits')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            parser.error('Reference port is occupied')
    out.mkdir(parents=True, exist_ok=False)
    command = [str(args.server.resolve()), '--model', str(args.model.resolve()), '--host', '127.0.0.1',
               '--port', str(args.port), '--ctx-size', str(args.context), '--parallel', '1', '--threads', '8',
               '--gpu-layers', '999', '--jinja', '--cache-ram', '0', *args.server_arg]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'TS_', 'TENSORSHARP_'))}
    env['CUDA_VISIBLE_DEVICES'] = '0'
    report = dict(started_unix=time.time(), command=command, timed_out=False, complete=False, passed=False,
                  timeout_seconds=args.timeout, timeout_scope='Total owned server lifetime; pre-launch hashing excluded',
                  limitations=['One GPU; default llama placement may differ from TensorSharp.',
                               'File cache uncontrolled; fresh hashing warms pages. Download and CPU soak may overlap.',
                               'Board usage includes desktop; samples miss short peaks. Process termination is not native cleanup validation.'])
    def save():
        temporary = out/'launch.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(out/'launch.json')
    save()
    server = sampler = timer = None
    deadline = None
    timed_out = threading.Event()
    watchdog_errors = []
    try:
        report['revision'] = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
        report['dirty'] = subprocess.check_output(['git', '-C', str(args.repo), 'status', '--porcelain'], text=True).strip()
        if report['dirty']:
            raise RuntimeError('Reference checkout must be unchanged')
        report['checkpoint'] = bench.verified_identities(bench.checkpoint_files(args.model), args.checkpoint_manifest)
        report['deployment'] = [bench.identity(p) for p in args.server.parent.iterdir() if p.is_file() and p.suffix in ('.exe', '.dll')]
        with (out/'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command, cwd=args.server.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      creationflags=bench.PROCESS_FLAGS)
            report['server_pid'] = server.pid
            deadline = time.monotonic() + args.timeout
            def expire():
                if server.poll() is None:
                    timed_out.set()
                    try:
                        server.kill()
                    except Exception as error:
                        watchdog_errors.append(repr(error))
            timer = threading.Timer(args.timeout, expire)
            timer.daemon = True
            timer.start()
            sampler = bench.Sampler(out/'memory.jsonl')
            sampler.process = psutil.Process(server.pid)
            sampler.thread.start()
            save()
            def remaining():
                left = deadline - time.monotonic()
                if timed_out.is_set() or left <= 0:
                    timed_out.set()
                    raise TimeoutError('Total reference lifetime exceeded')
                return left
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Reference exited during load ({server.returncode})')
                connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=min(2, remaining()))
                try:
                    connection.request('GET', '/health')
                    response = connection.getresponse()
                    response.read()
                    if response.status == 200:
                        break
                except (OSError, TimeoutError):
                    pass
                finally:
                    connection.close()
                time.sleep(min(0.5, remaining()))
            listeners = [connection for connection in sampler.process.net_connections(kind='tcp')
                         if connection.status == psutil.CONN_LISTEN and connection.laddr.port == args.port
                         and connection.laddr.ip == '127.0.0.1']
            if len(listeners) != 1:
                raise RuntimeError('Healthy reference endpoint is not the owned process listener')
            report['owned_listener_observed'] = {'pid': server.pid, 'host': '127.0.0.1', 'port': args.port}
            client = [sys.executable, str(HERE/'llama-quality-control.py'), '--port', str(args.port),
                      '--output', str(out/'quality'), '--cases', args.cases, '--max-tokens', str(args.max_tokens),
                      '--repetitions', str(args.repetitions), '--timeout', str(remaining())]
            if args.reasoning_budget is not None:
                client += ['--reasoning-budget', str(args.reasoning_budget)]
            with (out/'quality-client.log').open('w', encoding='utf-8') as client_log:
                run_client(client, remaining(), client_log, report)
            report['complete'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if not isinstance(error, Exception):
            raise
    finally:
        if deadline is not None and time.monotonic() >= deadline:
            timed_out.set()
        report.update(cleanup.cleanup_owned(server, timer, sampler))
        report['timed_out'] = report['timed_out'] or timed_out.is_set()
        report['watchdog_errors'] = watchdog_errors
        report['finished_unix'] = time.time()
        client_cleanup = report.get('client_cleanup', {})
        report['passed'] = (report['complete'] and report.get('client_exit') == 0 and 'error' not in report
            and not report['timed_out'] and not report['cleanup_errors'] and not watchdog_errors
            and client_cleanup.get('owned_server_exited') is True and not client_cleanup.get('cleanup_errors'))
        save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
