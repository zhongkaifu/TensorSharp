#!/usr/bin/env python3
"""Strict learned-head parity, engagement, reuse and unchanged 5% performance/memory gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import re


DENSE = ['warmup', 'short', 'copy', 'json', 'retained-A', 'retained-B', 'retained-A-followup',
         'parallel4-i0', 'parallel4-i1', 'parallel4-i2', 'parallel4-i3', 'after-parallel4', 'image', 'image-followup',
         'ordered-frames', 'ordered-frames-followup']
COUNTERS = ('Drafted', 'Accepted', 'VerifySteps', 'PlainSteps', 'Rollbacks')
METRICS = {'TtftMs': 1, 'TotalMs': 1, 'DecodeTps': -1, 'PrefillTps': -1}


def index(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError('Duplicate row ' + row[key])
        result[row[key]] = row
    return result


def token_sha(tokens):
    return hashlib.sha256(b''.join(t.to_bytes(4, 'little', signed=True) for t in tokens)).hexdigest()


def pair(plain, mtp, plain_inputs, mtp_inputs):
    failures, details = [], []
    if plain_inputs['mode'] != 'plain' or mtp_inputs['mode'] != 'mtp' or plain_inputs['tier'] != mtp_inputs['tier']:
        raise ValueError('Mode/tier identity mismatch')
    tier = plain_inputs['tier']
    if tier not in ('dense', 'long-qsa-unqualified'):
        raise ValueError('Unknown tier')
    for metadata in (plain_inputs, mtp_inputs):
        if metadata['actual_head'].lower() != 'mtp':
            failures.append('Learned head is not actually attached')
    expected = set(DENSE if tier == 'dense' else DENSE[:-4])
    a, b = index(plain, 'Label'), index(mtp, 'Label')
    ia, ib = index(plain_inputs['inputs'], 'id'), index(mtp_inputs['inputs'], 'id')
    if any(set(rows) != expected for rows in (a, b, ia, ib)):
        failures.append('Exact complete row/input set missing or unexpected')
    for key in sorted(expected & set(a) & set(b) & set(ia) & set(ib)):
        left, right = a[key], b[key]
        issues = []
        for mode, row, inp in [('plain', left, ia[key]), ('mtp', right, ib[key])]:
            tokens = row.get('Tokens')
            valid = isinstance(tokens, list) and bool(tokens) and all(type(t) is int and 0 <= t < 2**31 for t in tokens)
            if not valid or row.get('OutTokens') != len(tokens or []):
                issues.append(mode + ' invalid or truncated raw output token list')
            if row.get('Finish') not in ('eos', 'length'):
                issues.append(mode + ' invalid finish')
            if not isinstance(inp.get('tokens'), list) or not inp['tokens']:
                issues.append(mode + ' missing prompt tokens')
            elif inp['tokens_i32_sha256'] != token_sha(inp['tokens']) or inp['prompt_tokens'] != len(inp['tokens']) or row['Prompt'] != len(inp['tokens']):
                issues.append(mode + ' prompt digest/count mismatch')
            upper = inp['prompt_tokens'] + inp['max_new_tokens'] + inp['verify_reserve']
            if inp['verify_reserve'] != 4 or upper != inp['maximum_position_exclusive'] or upper > (2051 if tier == 'dense' else 65536):
                issues.append(mode + ' tier budget exceeded')
            if any(type(row.get(c)) is not int or row[c] < 0 for c in COUNTERS):
                issues.append(mode + ' invalid speculation counters')
            elif row['Accepted'] > row['Drafted']:
                issues.append(mode + ' accepted exceeds drafted')
            if mode == 'plain' and any(row.get(c) != 0 for c in ('Drafted', 'Accepted', 'VerifySteps', 'Rollbacks')):
                issues.append('plain run unexpectedly speculated')
            timelines = row.get('RequestTimelines')
            if not isinstance(timelines, list) or len(timelines) != 1 or timelines[0].get('Finish') != row.get('Finish') or timelines[0].get('Error'):
                issues.append(mode + ' missing/error per-request completion timeline')
            elif timelines[0].get('OutTokens') != row.get('OutTokens') or any(timelines[0].get('Speculation', {}).get(c) != row.get(c) for c in COUNTERS):
                issues.append(mode + ' per-request counters/token count disagree with row')
        same_input = all(ia[key].get(k) == ib[key].get(k) for k in ('tokens', 'max_new_tokens', 'verify_reserve', 'media'))
        if not same_input:
            issues.append('Rendered/injected prompt or sampling budget differs; no matched workload claim')
        equal = left.get('Tokens') == right.get('Tokens')
        if not equal or left.get('Finish') != right.get('Finish'):
            issues.append('Full greedy token/finish parity failed; no truncation or tolerance')
        if key in ('retained-B', 'retained-A-followup') and (left['Reused'] <= 0 or right['Reused'] <= 0):
            issues.append('Required retained/shared prefix did not actually reuse tokens')
        if key != 'warmup' and key != 'short' and (right['Drafted'] <= 0 or right['VerifySteps'] <= 0):
            issues.append('Learned speculation never engaged for this request')
        details.append({'id': key, 'matched_input': same_input, 'tokens_equal': equal, 'finish_equal': left.get('Finish') == right.get('Finish'),
                        'first_divergence': None if equal else next((i for i, (x, y) in enumerate(zip(left.get('Tokens', []), right.get('Tokens', []))) if x != y), min(len(left.get('Tokens', [])), len(right.get('Tokens', [])))),
                        'mtp_counters': {c: right.get(c) for c in COUNTERS}, 'failures': issues})
        failures.extend(key + ': ' + issue for issue in issues)
    if not any(b[k].get('Accepted', 0) > 0 for k in expected & set(b) if k != 'warmup'):
        failures.append('No measured proposal was accepted')
    if not any(b[k].get('Rollbacks', 0) > 0 for k in expected & set(b) if k != 'warmup'):
        failures.append('Rejected-tail rollback not exercised; extra fixed diagnostic required')
    return {'status': 'failed' if failures else 'passed', 'release_qualified': False, 'tier': tier, 'failures': failures, 'rows': details}


def compare_repeats(repeats):
    if len(repeats) != 3:
        raise ValueError('Exactly three warmed independent process pairs required')
    reports = [pair(r['plain'], r['mtp'], r['plain_inputs'], r['mtp_inputs']) for r in repeats]
    if len({r['tier'] for r in reports}) != 1:
        raise ValueError('Mixed tiers')
    failures = [f'repeat {i + 1}: {f}' for i, report in enumerate(reports) for f in report['failures']]
    identities = []
    for i, repeat in enumerate(repeats):
        for mode in ('plain', 'mtp'):
            owner = repeat.get(mode + '_owner', {})
            if owner.get('exit_code') != 0:
                failures.append(f'repeat {i + 1}/{mode}: benchmark process failed or exit status missing')
            for flag in ('before_identity_passed', 'after_identity_passed', 'process_gone', 'exclusive_timing'):
                if owner.get(flag) is not True:
                    failures.append(f'repeat {i + 1}/{mode}: owner did not establish {flag}')
            fields = ('native_sha256', 'managed_manifest_sha256', 'model_inventory_sha256', 'head_sha256')
            identity = tuple(owner.get(field) for field in fields)
            if not all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in identity):
                failures.append(f'repeat {i + 1}/{mode}: actual process/application/model identity missing')
            identities.append(identity)
    if len(set(identities)) != 1:
        failures.append('Processes do not share the same actual native/managed/model/head identity')
    timing = []
    for key in sorted(set(DENSE if reports[0]['tier'] == 'dense' else DENSE[:-4]) - {'warmup'}):
        for metric, sign in METRICS.items():
            try:
                p = [index(r['plain'], 'Label')[key][metric] for r in repeats]
                m = [index(r['mtp'], 'Label')[key][metric] for r in repeats]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in p + m):
                    raise ValueError('Missing positive finite timing')
                pm, mm = statistics.median(p), statistics.median(m)
                regression = sign * (mm / pm - 1)
                failed = regression > 0.05
                timing.append({'id': key, 'metric': metric, 'plain_samples': p, 'mtp_samples': m, 'plain_median': pm,
                               'mtp_median': mm, 'regression_fraction': regression, 'gate': 0.05, 'passed': not failed})
                if failed:
                    failures.append(f'{key}/{metric}: original 5% gate failed ({regression:.3%})')
            except (KeyError, ValueError) as exc:
                failures.append(f'{key}/{metric}: {exc}')
    memory = []
    for metric in ('host_rss_peak_bytes', 'cgroup_current_peak_bytes', 'gpu_used_peak_bytes'):
        # Memory is an independent strict gate, not silently waived because the
        # learned head owns extra scratch. Both modes attach the same head.
        try:
            p = [r['plain_memory'][metric] for r in repeats]
            m = [r['mtp_memory'][metric] for r in repeats]
            if not all(type(v) is int and v > 0 for v in p + m):
                raise ValueError('Missing measured memory peak')
            regression = statistics.median(m) / statistics.median(p) - 1
            passed = regression <= 0.05
            memory.append({'metric': metric, 'plain_samples': p, 'mtp_samples': m, 'regression_fraction': regression, 'gate': 0.05, 'passed': passed})
            if not passed:
                failures.append(metric + ': original 5% memory gate failed')
        except (KeyError, ValueError) as exc:
            failures.append(metric + ': ' + str(exc))
    return {'status': 'failed' if failures else 'passed', 'release_qualified': False, 'failures': failures,
            'pairs': reports, 'timing': timing, 'memory': memory,
            'limitations': ['Observational until owner verifies quiet processes, source/native/app/model identity before/after and identical settings.',
                            'The original native TP gate and long-context QSA limitation remain unchanged.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats', type=Path, required=True, help='JSON list of three resolved row/input/memory objects')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = compare_repeats(json.loads(a.repeats.read_bytes()))
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'failures': len(report['failures'])}))
    raise SystemExit(report['status'] != 'passed')
