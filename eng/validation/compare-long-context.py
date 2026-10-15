#!/usr/bin/env python3
"""Compare completed, equally concurrent LongContextProbe runs of one build.

Token equality is execution parity, not semantic quality. Timing includes the
recorded machine interference; this tool cannot establish statistical confidence.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def compare(before, after, max_regression=5):
    if not before or len(before) != len(after):
        raise ValueError('Require equal nonempty reference/candidate process counts')
    reference = before[0]
    fields = ('ModelSha256', 'ContextTokens', 'MinimumPromptTokens', 'Steps', 'Width',
              'ChunkTokens', 'MaximumRunningRequests', 'PrefillChunkLimitEnforced',
              'BulkRestoreEnabled', 'GgmlRevision')
    if not reference.get('ModelSha256') or not reference.get('GgmlRevision'):
        raise ValueError('Missing checkpoint or upstream identity')
    identity_keys = ('ProcessId', 'StartedUtc', 'ExecutionEnvironment', 'ProbeSha256')
    identified = [all(k in r for k in identity_keys) for r in before + after]
    if any(any(k in r for k in identity_keys) for r in before + after) and not all(identified):
        raise ValueError('Do not mix legacy and process/environment-identified reports')
    if all(identified):
        if len({(r['ProcessId'], r['StartedUtc']) for r in before + after}) != len(before + after):
            raise ValueError('Repeated process cannot count as independent evidence')
        if any(r['ExecutionEnvironment'] != reference['ExecutionEnvironment']
               or r['ProbeSha256'] != reference['ProbeSha256'] for r in before + after):
            raise ValueError('Probe or execution environment differs')
    budgets = [r.get('RequestedBudget') for r in after]
    budget_checked = any(b is not None for b in budgets)
    if budget_checked:
        for budget in budgets:
            if not isinstance(budget, list) or not budget or any(not isinstance(x, dict)
                                 or not isinstance(x.get('Pool'), str) or not x['Pool']
                                 or type(x.get('Bytes')) is not int or x['Bytes'] < 0 for x in budget):
                raise ValueError('Missing or invalid candidate budget identity')
            if len({x['Pool'] for x in budget}) != len(budget):
                raise ValueError('Duplicate candidate budget pool')
        normalized = lambda b: sorted((x['Pool'], x['Bytes']) for x in b)
        if any(normalized(b) != normalized(budgets[0]) for b in budgets):
            raise ValueError('Candidate process budgets differ')
    identities = lambda r, key: sorted((x.get('Path', x.get('FileName')).replace('\\', '/').rsplit('/', 1)[-1],
                                       x['Sha256'].upper()) for x in r[key])
    def outputs(r):
        rows = r['Requests']
        if len(rows) != r['Width'] or len({x['RequestId'] for x in rows}) != len(rows):
            raise ValueError('Missing or duplicate request')
        for x in rows:
            if len(x['Tokens']) != r['Steps'] or not x['PromptHash']:
                raise ValueError('Incomplete token or prompt evidence')
        return sorted((x['RequestId'], x['PromptTokens'], x['PromptHash'], x['Tokens'],
                       x['Status'], x['FinishReason']) for x in rows)
    expected = outputs(reference)
    samples = {'reference': [], 'candidate': []}
    for arm, reports in (('reference', before), ('candidate', after)):
        for report in reports:
            if report.get('Arm') != arm or report.get('Passed') is not True or report.get('Error'):
                raise ValueError('Incomplete or failed process')
            if any(report.get(k) != reference.get(k) for k in fields):
                raise ValueError('Workload or upstream revision differs')
            for key in ('Native', 'Managed'):
                if not reference.get(key) or identities(report, key) != identities(reference, key):
                    raise ValueError('Loaded build identity differs: ' + key)
            if outputs(report) != expected:
                raise ValueError('Complete generated token streams differ')
            if arm == 'candidate' and report.get('TokenParity') is not True:
                raise ValueError('Candidate did not verify its reference')
            if max((x['Running'] for x in report['RunningSamples']), default=0) < report['Width']:
                raise ValueError('Submitted width was not actually admitted concurrently')
            if any(x['Reserved'] or x['Committed'] for x in (report['AfterDispose'] or [])):
                raise ValueError('Live ledger ownership remained after disposal')
            if arm == 'candidate' and not report['AfterDispose']:
                raise ValueError('Candidate is missing disposal evidence')
            times = {'wall_seconds': report['WallSeconds'],
                     'prefill_compute_seconds': sum(x['PrefillSeconds'] for x in report['Requests']),
                     'decode_compute_seconds': sum(x['DecodeSeconds'] for x in report['Requests'])}
            if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in times.values()):
                raise ValueError('Invalid or absent latency')
            samples[arm].append(times)
    metrics = {}
    for key in samples['reference'][0]:
        b = [x[key] for x in samples['reference']]
        a = [x[key] for x in samples['candidate']]
        bm, am = statistics.median(b), statistics.median(a)
        metrics[key] = dict(reference=b, candidate=a, reference_median=bm,
                            candidate_median=am, latency_change_percent=(am / bm - 1) * 100)
    return dict(passed=all(m['latency_change_percent'] <= max_regression for m in metrics.values()),
                exact_tokens=True, actual_concurrent_width=reference['Width'], process_pairs=len(before),
                process_and_environment_identity_checked=all(identified),
                candidate_budget_identity_checked=budget_checked,
                metrics=metrics, max_regression_percent=max_regression,
                scope='Same-build TensorSharp budget comparison; capped token parity, not semantic quality. '
                      'Concurrent means simultaneous admission and interleaved per-sequence execution, not batched CUDA kernels. '
                      'Process medians do not remove background I/O, CPU, temperature or cache variation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', nargs='+', type=Path, required=True)
    p.add_argument('--after', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-regression-percent', type=float, default=5)
    a = p.parse_args()
    try:
        if not math.isfinite(a.max_regression_percent) or a.max_regression_percent < 0:
            raise ValueError('Regression threshold must be finite and nonnegative')
        paths = a.before + a.after
        if len({x.resolve() for x in paths}) != len(paths):
            raise ValueError('Repeated evidence file cannot count as another process')
        load = lambda paths: [json.loads(x.read_text(encoding='utf-8-sig')) for x in paths]
        result = compare(load(a.before), load(a.after), a.max_regression_percent)
    except (ValueError, TypeError, KeyError, OSError) as e:
        result = dict(passed=False, error=str(e))
    result['sources'] = dict(reference=[str(x) for x in a.before], candidate=[str(x) for x in a.after])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
