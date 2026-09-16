#!/usr/bin/env python3
"""Require actual learned-drafter execution; preserve strict greedy divergence."""
import argparse
import importlib.util
import json
from pathlib import Path


def validate(rows, require_parity=True):
    indexed = {}
    for row in rows:
        key = (row['Scenario'], row['Label'])
        if key in indexed:
            raise ValueError('Duplicate speculative scenario/label: ' + str(key))
        indexed[key] = row
    plain = indexed.get(('spec', 'plain greedy'))
    learned = indexed.get(('spec', 'draft head (auto)'))
    result = {'status': 'failed', 'failures': [], 'greedy_parity_required': require_parity,
              'scope': 'Learned drafter solo execution; separate ngram/tool/concurrency rows are not learned-drafter coverage.'}
    if plain is None or learned is None:
        result['failures'].append('Plain or learned-drafter row is missing; attachment/fallback cannot count as coverage')
        return result
    result['counters'] = {name: learned.get(name) for name in ('Drafted', 'Accepted', 'VerifySteps', 'PlainSteps', 'Rollbacks')}
    for name, value in result['counters'].items():
        if type(value) is not int or value < 0:
            result['failures'].append(f'Missing or invalid nonnegative integer {name}: {value!r}')
    for name in ('Drafted', 'Accepted', 'VerifySteps'):
        if type(result['counters'][name]) is not int or result['counters'][name] <= 0:
            result['failures'].append(f'Learned drafter did not exercise {name}')
    if (type(result['counters']['Accepted']) is int and type(result['counters']['Drafted']) is int
            and result['counters']['Accepted'] > result['counters']['Drafted']):
        result['failures'].append('Accepted exceeds Drafted')
    left, right = plain['Tokens'], learned['Tokens']
    for label, tokens in (('plain', left), ('learned', right)):
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            result['failures'].append(f'{label} tokens must be a nonempty list of token IDs')
    if not isinstance(left, list) or not isinstance(right, list):
        return result
    first = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
    equal = left == right and plain['Finish'] == learned['Finish']
    result.update(greedy_tokens_equal=left == right, finish_equal=plain['Finish'] == learned['Finish'],
                  first_divergence=None if left == right else first,
                  plain_tokens=len(left), learned_tokens=len(right))
    if require_parity and not equal:
        result['failures'].append('Learned-drafter greedy output differs from sequential plain output; no numerical tolerance or truncation applied')
    result['status'] = 'failed' if result['failures'] else 'passed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('bench_compare', Path(__file__).resolve().parents[2] / 'benchmarks/AgentTurnBench/compare.py')
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    rows = comparator.load_rows(args.rows)
    report = validate(list(rows.values()))
    report['rows'] = str(args.rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(report['status'], '; '.join(report['failures']))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
