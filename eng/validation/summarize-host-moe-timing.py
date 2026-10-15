#!/usr/bin/env python3
"""Summarize opt-in HOSTMOE cache host-wall records; never launch a model.

Stages can nest and backend submission may wait for earlier work. Their sums are
not an additive latency breakdown, CUDA kernel timings, or physical SSD I/O.
Scalar callbacks are deliberately not labeled prefill/decode without a separate
request trace. This tool rejects malformed, partial, and non-finite records.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


MARKER = '[HOSTMOE-CACHE-TIMING]'
SCOPE = 'nested_host_wall_not_cuda_kernel_or_physical_io'
INTEGER_KEYS = ('layer', 'rows', 'joint', 'bulk', 'plan_regions', 'wide_replaces',
                'file_windows', 'file_bytes', 'graph_submissions')
MILLISECOND_KEYS = ('total_ms', 'plan_ms', 'allocation_ms', 'wide_replace_ms',
                    'file_pipeline_ms', 'upload_enqueue_ms', 'upload_wait_ms',
                    'input_bridge_ms', 'compute_ms', 'output_bridge_ms')
KEYS = set(INTEGER_KEYS + MILLISECOND_KEYS + ('outcome', 'scope'))


def parse_record(line, line_number):
    if MARKER not in line:
        return None
    _, payload = line.split(MARKER, 1)
    fields = {}
    for field in payload.strip().split():
        key, separator, value = field.partition('=')
        if not separator or key in fields:
            raise ValueError(f'Line {line_number}: invalid or duplicate timing field')
        fields[key] = value
    if set(fields) != KEYS or fields['scope'] != SCOPE:
        raise ValueError(f'Line {line_number}: incomplete or unknown timing schema/scope')
    for key in INTEGER_KEYS:
        value = fields[key]
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f'Line {line_number}: invalid nonnegative integer {key}')
        fields[key] = int(value)
    for key in MILLISECOND_KEYS:
        fields[key] = float(fields[key])
        if not math.isfinite(fields[key]) or fields[key] < 0:
            raise ValueError(f'Line {line_number}: invalid finite interval {key}')
    if fields['rows'] <= 0 or fields['joint'] not in (0, 1) or fields['bulk'] not in (0, 1):
        raise ValueError(f'Line {line_number}: invalid execution shape')
    if fields['joint'] and fields['bulk']:
        raise ValueError(f'Line {line_number}: bulk prefill cannot be joint decode')
    if fields['outcome'] not in ('completed', 'exception', 'not_completed'):
        raise ValueError(f'Line {line_number}: invalid outcome')
    fields['line'] = line_number
    return fields


def execution_kind(record):
    if record['bulk']:
        return 'genuine-wide-prefill'
    if record['joint']:
        return 'joint-decode'
    if record['rows'] > 1:
        return 'nonjoint-narrow-prefill'
    return 'scalar-phase-unresolved'


def aggregate(records):
    result = dict(Callbacks=len(records), Rows=sum(r['rows'] for r in records),
                  Layers=sorted({r['layer'] for r in records}),
                  Outcomes={value: sum(r['outcome'] == value for r in records)
                            for value in ('completed', 'exception', 'not_completed')},
                  SumHostIntervalsMs={key: math.fsum(r[key] for r in records)
                                      for key in MILLISECOND_KEYS},
                  Counters={key: sum(r[key] for r in records)
                            for key in ('plan_regions', 'wide_replaces', 'file_windows',
                                        'file_bytes', 'graph_submissions')})
    if records:
        slowest = max(records, key=lambda r: r['total_ms'])
        result['LargestCallback'] = slowest
    return result


def summarize(text):
    records = [record for number, line in enumerate(text.splitlines(), 1)
               if (record := parse_record(line, number)) is not None]
    if not records:
        raise ValueError('No HOSTMOE-CACHE-TIMING records; timing coverage is unavailable')
    grouped = {}
    for record in records:
        key = (execution_kind(record), record['rows'])
        grouped.setdefault(key, []).append(record)
    groups = [dict(ExecutionKind=kind, CallbackRows=rows, **aggregate(values))
              for (kind, rows), values in sorted(grouped.items())]
    # A same-shape run may contain several chunks or requests. Preserve ordinal
    # blocks without inventing a request ID or assuming each is a whole layer set.
    blocks = []
    previous_shape = None
    for record in records:
        shape = (record['rows'], record['joint'], record['bulk'])
        if shape != previous_shape:
            blocks.append(dict(FirstLine=record['line'], LastLine=record['line'],
                               ExecutionKind=execution_kind(record), CallbackRows=record['rows'],
                               Records=[]))
        blocks[-1]['Records'].append(record)
        blocks[-1]['LastLine'] = record['line']
        previous_shape = shape
    for block in blocks:
        block.update(aggregate(block.pop('Records')))
    return dict(Schema='host-moe-cache-timing-summary.v1', Coverage='diagnostic host callbacks only',
                AllCallbacksCompleted=all(r['outcome'] == 'completed' for r in records),
                Groups=groups, OrdinalShapeBlocks=blocks,
                LayerGroups=[dict(ExecutionKind=kind, CallbackRows=rows, Layer=layer,
                                  **aggregate([r for r in values if r['layer'] == layer]))
                             for (kind, rows), values in sorted(grouped.items())
                             for layer in sorted({r['layer'] for r in values})],
                Limits=['Timing instrumentation and stderr writes change the diagnostic run; qualify performance separately without timing.',
                        'Host stage intervals can nest and must not be added into a latency waterfall.',
                        'FilePipelineMs includes worker reads and upload enqueue callbacks; FileBytes are requested source bytes, not physical SSD reads.',
                        'Upload enqueue intervals are not H2D completion; backend compute wall can include queued work or waits.',
                        'Rows sum is callback rows across layers, not unique prompt/output tokens.',
                        'Scalar callbacks and same-shape blocks need an independent request/forward trace to identify phase and boundaries.',
                        'Outer dense/attention/indexer/PLE/graph-admission time and total physical resource ownership are not covered.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.log.stat().st_size > 64 << 20:
        raise ValueError('Expected a bounded diagnostic log <=64 MiB')
    raw = args.log.read_bytes()
    result = summarize(raw.decode('utf-8-sig', errors='strict'))
    result['Input'] = dict(Path=str(args.log.resolve()), Bytes=len(raw), Sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(Completed=True, Callbacks=sum(r['Callbacks'] for r in result['Groups']),
                         AllCallbacksCompleted=result['AllCallbacksCompleted'], Output=str(args.output))))


if __name__ == '__main__':
    main()
