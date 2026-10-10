#!/usr/bin/env python3
"""Summarize diagnostic CUPTI JSONL inside completed decode markers (not a benchmark)."""
import argparse
import bisect
import collections
import json
import pathlib


def summarize(records, iteration=1, skip_steps=2):
    summaries = [r for r in records if r['kind'] == 'summary']
    if len(summaries) != 1 or summaries[0]['dropped'] or summaries[0]['errors'] or summaries[0]['records'] <= 0:
        raise ValueError('Missing, partial or dropped CUPTI recording')
    marks = {}
    for row in records:
        if row['kind'] != 'mark':
            continue
        label = row['label'].split('/')
        if len(label) != 4 or label[0] != 'decode' or int(label[1]) != iteration or int(label[2]) < skip_steps:
            continue
        step, edge = int(label[2]), label[3]
        if edge not in ('begin', 'end') or edge in marks.setdefault(step, {}):
            raise ValueError('Duplicate/unknown decode marker')
        marks[step][edge] = row['timestamp']
    windows = []
    for step, edges in sorted(marks.items()):
        if set(edges) != {'begin', 'end'} or edges['end'] <= edges['begin']:
            raise ValueError('Incomplete decode marker')
        windows.append((edges['begin'], edges['end'], step))
    if not windows or any(left[1] > right[0] for left, right in zip(windows, windows[1:])):
        raise ValueError('Missing or overlapping decode windows')
    if [w[2] for w in windows] != list(range(skip_steps, windows[-1][2] + 1)):
        raise ValueError('Missing selected decode step')
    starts = [w[0] for w in windows]
    grouped = {kind: collections.defaultdict(lambda: [0, 0, 0]) for kind in ('kernel', 'api', 'copy')}
    counts = collections.Counter()
    for row in records:
        kind = row['kind']
        if kind not in grouped:
            continue
        if row['end'] <= row['start']:
            raise ValueError('Incomplete activity timestamps')
        index = bisect.bisect_right(starts, row['start']) - 1
        if index < 0 or row['start'] >= windows[index][1]:
            continue
        if row['end'] > windows[index][1]:
            raise ValueError('Activity crosses a completed forward boundary')
        name = str(row.get('name', row.get('copy_kind', 'unknown')))
        bucket = grouped[kind][name]
        bucket[0] += 1
        bucket[1] += row['end'] - row['start']
        bucket[2] += row.get('bytes', 0)
        counts[(windows[index][2], kind)] += 1
    if any(counts[(w[2], 'kernel')] == 0 for w in windows):
        raise ValueError('Selected decode step has no kernels')
    steps = len(windows)
    return dict(iteration=iteration, skipped_initial_steps=skip_steps, steps=steps,
        mean_marked_wall_ms=sum(end-start for start,end,_ in windows)/steps/1e6,
        **{kind: [dict(name=name, count=count, count_per_step=count/steps,
                      ms_per_step=ns/steps/1e6, bytes_per_step=size/steps)
                  for name, (count, ns, size) in sorted(data.items(), key=lambda x: -x[1][1])]
           for kind,data in grouped.items()},
        limitations=['Diagnostic instrumentation can alter CPU/GPU overlap and timings.',
                     'API and GPU durations overlap; their sums are not additive wall time.',
                     'This is a within-run attribution, not an uninstrumented throughput result.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=pathlib.Path)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--skip-steps', type=int, default=2)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(line) for line in args.trace.read_text().splitlines()], args.iteration, args.skip_steps)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('steps', 'mean_marked_wall_ms')}))
    for kind in ('kernel', 'api', 'copy'):
        print(kind, 'total ms/step:', sum(x['ms_per_step'] for x in result[kind]))
        for row in result[kind][:12]:
            print(f"  {row['ms_per_step']:.4f} ms {row['count_per_step']:.1f}/step {row['name']}")


if __name__ == '__main__':
    main()
