#!/usr/bin/env python3
"""Replay the strict multimodel tasks against an existing independent llama server.

This tool owns requests, not the server. Preserve its launch log, exact clean
revision, checkpoint hashes and memory telemetry separately. Timings use llama's
reported prompt/predicted counts; no cross-engine token-denominator parity claim.
"""
import argparse
import base64
import http.client
import importlib.util
import json
import math
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('quality_bench', HERE / 'multimodel-quality-bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def llama_metrics(response):
    timing = response['timings']
    if timing.get('cache_n') != 0:
        raise ValueError('Expected an uncached llama prompt')
    result = {}
    for label, prefix in (('prefill', 'prompt'), ('decode', 'predicted')):
        count, milliseconds = timing[prefix + '_n'], timing[prefix + '_ms']
        if type(count) is not int or count < 1 or type(milliseconds) not in (int, float) or not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError('Missing real llama compute timing')
        # llama samples its first predicted token from the prefill logits, so
        # predicted_ms times n-1 forward steps. Preserve its reported numerator
        # separately rather than inflating short-answer throughput by n/(n-1).
        steps = count - 1 if prefix == 'predicted' else count
        rate = steps * 1000 / milliseconds
        reported = timing.get(prefix + '_per_second')
        if steps <= 0 or type(reported) not in (int, float) or not math.isfinite(reported) or not math.isclose(rate, reported, rel_tol=1e-9):
            raise ValueError('llama timing denominator differs from its reported rate')
        result[label + '_tps'] = rate
        result[label + '_tokens'] = steps
        result[label + '_reported_tokens'] = count
        result[label + '_ms'] = milliseconds
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default='squares,tool_json,code,long_extract')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--reasoning-budget', type=int, help='Explicit llama reasoning_budget_tokens override; 0 forces immediate reasoning closure, -1 is unrestricted')
    args = parser.parse_args()
    cases = args.cases.split(',')
    if (args.repetitions < 1 or args.max_tokens < 1 or args.timeout <= 0 or
            (args.reasoning_budget is not None and args.reasoning_budget < -1) or
            len(cases) != len(set(cases)) or any(case not in bench.PROMPTS for case in cases)):
        parser.error('Require positive limits and unique known cases')
    if 'image_ocr' in cases and args.image is None:
        parser.error('Image case needs --image')
    out = args.output.resolve()
    if not any(out.is_relative_to(bench.ROOT / base) for base in ('artifacts', 'docs/validation')):
        parser.error('Evidence must stay in ignored artifact directories')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(engine='llama.cpp', started_unix=time.time(), records=[], run_complete=False,
                  image=bench.identity(args.image) if args.image else None,
                  harness_sha256=bench.sha(__file__), checker_sha256=bench.sha(bench.__file__),
                  limitations=['Server identity/resources must be bound by separate launch evidence.',
                               'Native engine timings and chat-template token counts can differ from TensorSharp.',
                               'Prior independent quality failures remain failures; successful timings do not repair answers.'])
    save = lambda: (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    try:
        for case in cases:
            for repetition in range(args.repetitions):
                content = bench.PROMPTS[case]
                if case == 'image_ocr':
                    content = [dict(type='text', text=content), dict(type='image_url', image_url=dict(
                        url='data:image/png;base64,' + base64.b64encode(args.image.read_bytes()).decode()))]
                body = dict(messages=[dict(role='user', content=content)], temperature=0, top_k=0, top_p=1,
                            min_p=0, repeat_penalty=1, presence_penalty=0, frequency_penalty=0, seed=17,
                            max_tokens=args.max_tokens, stream=False, cache_prompt=False,
                            chat_template_kwargs=dict(enable_thinking=False), timings=True)
                if args.reasoning_budget is not None:
                    body['reasoning_budget_tokens'] = args.reasoning_budget
                row = dict(case=case, repetition=repetition, request=body, quality_passed=False)
                report['records'].append(row)
                save()
                connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=args.timeout)
                started = time.monotonic()
                try:
                    connection.request('POST', '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    row['http_status'] = response.status
                    row['raw_response'] = response.read().decode()
                    row['response'] = data = json.loads(row['raw_response'])
                    if response.status != 200:
                        raise ValueError(f'HTTP {response.status}')
                    choice = data['choices'][0]
                    row['complete'] = choice['finish_reason'] == 'stop'
                    row['quality_passed'] = bench.quality(case, choice['message'].get('content'), row['complete'])
                    row['metrics'] = llama_metrics(data)
                except (TimeoutError, OSError):
                    raise  # A timed-out request might still own the GPU.
                except Exception as error:
                    row['error'] = str(error)
                finally:
                    row['http_wall_ms'] = (time.monotonic() - started) * 1000
                    connection.close()
                    save()
                print(json.dumps({key: row.get(key) for key in ('case', 'repetition', 'quality_passed', 'metrics', 'error')}), flush=True)
        report['run_complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['finished_unix'] = time.time()
        report['passed'] = report['run_complete'] and all(row['quality_passed'] and 'metrics' in row and 'error' not in row for row in report['records'])
        save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
