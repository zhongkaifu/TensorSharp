#!/usr/bin/env python3
"""Retain vectors and timings when an embedding consistency gate fails."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks/EmbeddingBench'))
import embedding_bench as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--minimum-measure-seconds', type=float, default=1)
    parser.add_argument('--prewarm-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.minimum_measure_seconds < 0 or args.prewarm_seconds < 0:
        parser.error('Invalid warmup/repeat/measurement duration')
    bench.KEEP_ALIVE = True
    report = {'model': args.model, 'started_at_unix': time.time(), 'status': 'running',
              'scope': 'Diagnostic measurements; consistency failure does not qualify as a passing benchmark.',
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs': bench.TEXTS, 'rows': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        batch, usage = bench.embedding(args.url, args.model, bench.TEXTS)
        report.update(batch_vectors=batch, usage=usage)
        save()
        for index, text in enumerate(bench.TEXTS):
            single, tokens = bench.embedding(args.url, args.model, text)
            row = {'index': index, 'single': single[0], 'usage': tokens,
                   'batch_cosine': bench.cosine(single[0], batch[index]),
                   'batch_max_component_error': max(abs(a-b) for a,b in zip(single[0], batch[index])),
                   'equal_length_batches': []}
            for degree in (2, 4):
                copies, copy_tokens = bench.embedding(args.url, args.model, [text] * degree)
                row['equal_length_batches'].append({'degree': degree, 'vectors': copies, 'usage': copy_tokens,
                    'single_cosines': [bench.cosine(single[0], vector) for vector in copies]})
            report['rows'].append(row)
            save()
        report['consistency_gate'] = {'minimum_cosine': .9999,
            'passed': all(row['batch_cosine'] > .9999 for row in report['rows'])}
        report['measurement_settings'] = {key: getattr(args, key) for key in
            ('warmup', 'repeats', 'minimum_measure_seconds', 'prewarm_seconds')}
        started = time.monotonic()
        calls = 0
        while time.monotonic() - started < args.prewarm_seconds:
            bench.embedding(args.url, args.model, bench.scenarios()['single_short'])
            calls += 1
        report['prewarm'] = {'calls': calls, 'seconds': time.monotonic() - started}
        report['benchmarks'] = bench.benchmark(args.url, args.model, args.warmup, args.repeats, args.minimum_measure_seconds)
        report['status'] = 'diagnostic-complete'
    except Exception as error:
        report.update(status='diagnostic-failed', error=str(error))
    report['finished_at_unix'] = time.time()
    save()
    print(report['status'], report.get('consistency_gate'), flush=True)
    return int(report['status'] != 'diagnostic-complete')


if __name__ == '__main__':
    raise SystemExit(main())
