#!/usr/bin/env python3
"""Offline independent FP64 and F16-Q sensitivity checks for shape-probe dumps.

No GPU or model access. Rounding Q to F16 is a diagnostic counterfactual, not a
replacement for the original API-input oracle or a new acceptance tolerance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_file(item, dtype):
    path = Path(item['Path'])
    data = path.read_bytes()
    if len(data) != item['Bytes'] or hashlib.sha256(data).hexdigest() != item['Sha256']:
        raise ValueError(f'Dump identity mismatch: {path}')
    return np.frombuffer(data, dtype=dtype)


def evaluate(q, k, v, prefix):
    heads, rows, dim = q.shape
    kv_heads = k.shape[0]
    result = np.empty((rows, heads, dim), dtype=np.float64)
    for h in range(heads):
        kh = h // (heads // kv_heads)
        for row in range(rows):
            keys = prefix + row + 1
            scores = k[kh, :keys] @ q[h, row]
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            result[row, h] = probabilities @ v[kh, :keys]
    return result.ravel()


def metrics(reference, actual):
    delta = actual - reference
    return dict(all_finite=bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
                max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta * delta))),
                relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    result = dict(report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_executed=False, original_threshold_unchanged=True, cases=[])
    for case in report['cases']:
        g = case['geometry']; rows = case['query_rows']; prefix = case['prefix']
        q = load_file(case['inputs'][0], '<f4').reshape(g['Heads'], 8, g['Dim'])[:, :rows].astype(np.float64)
        k, v = [load_file(x, '<f2').reshape(g['KvHeads'], g['CacheStride'], g['Dim']).astype(np.float64)
                for x in case['inputs'][1:]]
        reference = load_file(case['reference'], '<f8')
        independent = evaluate(q, k, v, prefix)
        q_rounded = evaluate(q.astype(np.float16).astype(np.float64), k, v, prefix)
        result['cases'].append(dict(geometry=g['Name'], profile=case['profile'], rows=rows,
            independent_fp64_vs_csharp=metrics(reference, independent),
            rounded_q_counterfactual_vs_api_reference=metrics(reference, q_rounded),
            phases=[dict(phase=a['phase'], native_vs_api_reference=metrics(reference, load_file(a['output'], '<f4')),
                         native_vs_f16_q_counterfactual=metrics(q_rounded, load_file(a['output'], '<f4')))
                    for a in case['arms']]))
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(cases=len(result['cases']), max_fp64_reimplementation_difference=max(
        c['independent_fp64_vs_csharp']['max_abs'] for c in result['cases']))))


if __name__ == '__main__':
    main()
