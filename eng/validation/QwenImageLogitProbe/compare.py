#!/usr/bin/env python3
"""Compare complete F32 distributions at identical teacher-forced image histories.

This is a correctness diagnostic, not a throughput benchmark. State hash differences
locate the first visible seam; they do not independently prove a numerical error.
"""
import argparse
import array
import json
import math
import pathlib
import sys


def load(path):
    root = pathlib.Path(path)
    report = json.loads((root / 'report.json').read_text(encoding='utf8'))
    if not (report.get('Completed') and report.get('ExitCode') == 0
            and report.get('SessionDisposed') and report.get('RequestedShapeEngaged')
            and report.get('TeacherTokensActual') == [271, 101878]):
        raise ValueError(f'{root}: incomplete or unqualified diagnostic')
    if any(p['Reserved'] or p['Committed'] for p in report['FinalBudget']):
        raise ValueError('Nonzero final ledger')
    if len(report['Captures']) != 3:
        raise ValueError('Require P, P+1 and P+2 captures')
    return root, report


def differing(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(differing(a.get(key), b.get(key), prefix + '/' + key))
        return result
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [prefix + '/length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differing(x, y, prefix + '/' + str(i))]
    return [] if a == b else [prefix]


def compare(left, right):
    lp, a = load(left)
    rp, b = load(right)
    for key in ('InputFileSha256', 'NativeSha256', 'ManagedCore', 'HostRoutedLayers'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched {key}')
    steps = []
    for ca, cb in zip(a['Captures'], b['Captures']):
        if ca['Step'] != cb['Step'] or ca['Computed'] != cb['Computed']:
            raise ValueError('Different teacher history position')
        values = []
        for root, capture in ((lp, ca), (rp, cb)):
            value = array.array('f')
            value.frombytes((root / capture['LogitsFile']).read_bytes())
            if sys.byteorder != 'little':
                value.byteswap()
            if len(value) != capture['Vocabulary'] or not all(map(math.isfinite, value)):
                raise ValueError('Incomplete/nonfinite full-vocabulary logits')
            values.append(value)
        x, y = values
        if len(x) != len(y):
            raise ValueError('Vocabulary mismatch')
        delta = [float(u) - v for u, v in zip(x, y)]
        mean = math.fsum(delta) / len(delta)
        steps.append({'Step': ca['Step'], 'Computed': ca['Computed'], 'Vocabulary': len(x),
                      'MaxAbsoluteDifference': max(map(abs, delta)),
                      'RmsDifference': math.sqrt(math.fsum(v*v for v in delta)/len(delta)),
                      'MeanOffset': mean,
                      'OffsetRemovedRms': math.sqrt(math.fsum((v-mean)**2 for v in delta)/len(delta)),
                      'ArgmaxLeft': ca['Argmax'], 'ArgmaxRight': cb['Argmax'],
                      'MarginLeft': ca['Margin'], 'MarginRight': cb['Margin'],
                      'StateDifferences': differing(ca['State'], cb['State'])})
    return {'Schema': 'qwen-image-logit-comparison.v1', 'Left': str(lp), 'Right': str(rp),
            'ChunkLeft': a['Chunk'], 'ChunkRight': b['Chunk'], 'MergerLeft': a['Merger'], 'MergerRight': b['Merger'],
            'PreparedMediaEqual': a['PreparedMedia'] == b['PreparedMedia'], 'Steps': steps,
            'Limitations': ['Fixed teacher inputs, not unconstrained answer quality.',
                            'Cache0 and device_min_batch0 are diagnostic controls, not a performance configuration.',
                            'Different state hashes or small logits do not alone prove incorrect chunk semantics.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pathlib.Path(args.output).write_text(json.dumps(compare(args.left, args.right), indent=2), encoding='utf8')
