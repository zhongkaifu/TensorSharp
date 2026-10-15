#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
"""Check diagnostic stream/read intervals, separately from quiet throughput."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HEADER = '[HOSTMOE-FILE-PIPELINE] '
SPAN = '[HOSTMOE-FILE-PIPELINE-SPAN] '


def require(value, message):
    if not value:
        raise ValueError(message)


def fields(text):
    result = {}
    for pair in text.split():
        key, value = pair.split('=', 1)
        require(key not in result, 'Duplicate timeline field')
        result[key] = value
    return result


def duration_union(intervals):
    end = None
    total = 0
    for first, last in sorted(intervals):
        if first >= last:
            continue
        total += last - max(first, end if end is not None else first) if end is None or last > end else 0
        end = max(last, end if end is not None else last)
    return total


def analyze(text):
    calls = []
    current = None
    for line in text.splitlines():
        if line.startswith(HEADER):
            current = {'header': fields(line[len(HEADER):]), 'spans': []}
            calls.append(current)
        elif line.startswith(SPAN):
            require(current is not None, 'Span without call header')
            current['spans'].append(fields(line[len(SPAN):]))
    require(calls, 'No pipeline intervals were recorded')
    rows = []
    for call in calls:
        h, spans = call['header'], call['spans']
        count, seq, pinned = int(h['chunks']), int(h['seq']), int(h['pinned'])
        a, b = int(h['gate_stream_start_ns']), int(h['gate_stream_end_ns'])
        require(count == len(spans) and count > 0 and seq > 0, 'Incomplete call/span set')
        require(0 < pinned <= 16 << 20 and 0 < a < b, 'Invalid pinned bound or gate interval')
        total_ms = float(h['total_ms'])
        require(math.isfinite(total_ms) and total_ms > 0, 'Invalid total time')
        require(('read_jobs' in h) == ('max_read_bytes' in h), 'Incomplete parallel-read geometry')
        read_jobs = int(h['read_jobs']) if 'read_jobs' in h else None
        max_read_bytes = int(h['max_read_bytes']) if 'max_read_bytes' in h else None
        if read_jobs is not None:
            require(read_jobs >= count and 0 < max_read_bytes <= 16 << 20,
                    'Invalid parallel-read geometry')
        reads, copies = [], []
        parts = set()
        for index, span in enumerate(spans):
            require(int(span['chunk']) == index and int(span['seq']) == seq, 'Misordered/mismatched chunk')
            part = int(span['part'])
            require(part in (0, 1, 2) and int(span['bytes']) > 0, 'Invalid projection/span bytes')
            parts.add(part)
            r0, r1 = int(span['read_start_ns']), int(span['read_end_ns'])
            c0, c1 = int(span['copy_start_ns']), int(span['copy_end_ns'])
            require(0 < r0 <= r1 <= c0 <= c1, 'Read/copy lifetime order violated')
            if part == 2:
                reads.append((max(a, r0), min(b, r1)))
                copies.append((max(a, c0), min(b, c1)))
        require(parts == {0, 1, 2}, 'Missing projection')
        read_sum = sum(max(0, y - x) for x, y in reads)
        copy_sum = sum(max(0, y - x) for x, y in copies)
        for key, actual in [('down_read_overlap_sum_ms', read_sum), ('down_copy_overlap_sum_ms', copy_sum)]:
            value = float(h[key])
            require(math.isfinite(value) and abs(value - actual / 1e6) <= .000501, 'Logged overlap does not match raw intervals')
        rows.append({'seq': seq, 'chunks': count, 'pinned_bytes': pinned, 'total_ms': total_ms,
                     'read_jobs': read_jobs, 'max_read_bytes': max_read_bytes,
                     'parallel_read_geometry_observed': read_jobs is not None and read_jobs > count,
                     'gate_stream_interval_ns': [a, b],
                     'down_read_overlap_union_ns': duration_union(reads),
                     'down_copy_overlap_union_ns': duration_union(copies),
                     'down_read_overlap_sum_ns': read_sum, 'down_copy_overlap_sum_ns': copy_sum})
    return {'valid_intervals': True, 'calls': rows,
            'parallel_read_geometry_observed': any(r['parallel_read_geometry_observed'] for r in rows),
            'read_compute_overlap_observed': any(r['down_read_overlap_union_ns'] > 0 for r in rows),
            'transfer_compute_overlap_observed': any(r['down_copy_overlap_union_ns'] > 0 for r in rows),
            'scope': 'Common steady-clock host callbacks delimit CUDA stream intervals including host submission gaps and callback overhead. Intersection proves concurrency with those intervals, not pure kernel activity or a quiet performance improvement. Read sums may overlap each other; union is reported separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-overlap', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be fresh')
    require(args.log.stat().st_size <= 256 << 20, 'Diagnostic log is too large')
    raw = args.log.read_bytes()
    result = {'log': str(args.log.resolve()), 'log_sha256': hashlib.sha256(raw).hexdigest(),
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'passed': False}
    try:
        result.update(analyze(raw.decode('utf-8')))
        result['passed'] = not args.require_overlap or (result['read_compute_overlap_observed'] and result['transfer_compute_overlap_observed'])
    except (ValueError, KeyError, UnicodeError) as error:
        result['error'] = str(error)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
