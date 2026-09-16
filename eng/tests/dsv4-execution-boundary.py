#!/usr/bin/env python3
"""Exercise actual V4/V4.1 C ABI failure statuses without a model or GPU.

Requires a test-enabled native library. The scoped native fixture injects
bad_alloc, a non-standard exception, or allocation decline before any tensor
access. This proves error containment/reset eligibility, not model recovery.
Use dsv41-failure-state.py for real cache mutation and independent logits.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    path = args.library.resolve()
    directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
    try:
        lib = ctypes.CDLL(str(path))
        test = lib.TSGgml_Dsv4TestExecutionBoundary
        test.argtypes = [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        test.restype = ctypes.c_int
        reset_test = lib.TSGgml_Dsv4TestResetTruncateBoundary
        reset_test.argtypes = [ctypes.c_int] * 2 + [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        reset_test.restype = ctypes.c_int
        checks = []
        for v41 in (0, 1):
            for api, name in enumerate(('forward', 'speculative', 'batched_decode', 'dspark_draft')):
                if v41 and api in (1, 3):
                    continue
                for failure, kind in enumerate(('bad_alloc', 'unknown_exception', 'allocation_decline'), 1):
                    actual = (ctypes.c_int * 11)()
                    status = test(v41, api, failure, actual, len(actual))
                    latch = api in (0, 1)
                    first = -3 if api in (0, 2) else 0
                    retry = -4 if api == 0 else first
                    expected = [first, int(latch), 5, 0, 5, retry, 1 if latch else 2, 0, 0,
                                1 if latch else 3, 2 if latch else 7]
                    checks.append({'architecture': 'V4.1' if v41 else 'V4', 'api': name,
                                   'injection': kind, 'fixture_status': status, 'expected': expected,
                                   'observed': list(actual), 'passed': status == 0 and list(actual) == expected})
        for v41 in (0, 1):
            # The ordinary batched fault rows above require a headless model:
            # a DSpark head deliberately refuses that graph before it can write
            # target features. Exercise that refusal separately, with injection
            # armed and explicit pre-reset state/owner/output invariants.
            actual = (ctypes.c_int * 14)()
            status = test(v41, 4, 1, actual, len(actual))
            expected = [-2, 0, 5, 0, 5, -2, 0, 0, 0, 3, 7, 1, 1, 1]
            checks.append({'architecture': 'V4.1' if v41 else 'V4',
                           'api': 'dspark_head_batched_refusal',
                           'injection': 'bad_alloc armed but never reached',
                           'fixture_status': status, 'expected': expected,
                           'observed': list(actual), 'passed': status == 0 and list(actual) == expected,
                           'pre_reset_invariants': {
                               'head_flags_slot_latches_positions_history_unchanged': bool(actual[11]),
                               'active_slot_and_graph_owners_unchanged': bool(actual[12]),
                               'scratch_and_drafted_canaries_unchanged': bool(actual[13])},
                           'scope': 'Head-present multi-slot refusal before execution; no silent state mutation'})
        for api, name in enumerate(('reset', 'truncate_reset', 'truncate_checkpoint_restore',
                                     'reset_checked_v41', 'reset_checked_v4', 'reset_checked_dspark')):
            for failure, kind in enumerate(('bad_alloc', 'unknown_exception'), 1):
                actual = (ctypes.c_int * 16)()
                status = reset_test(api, failure, actual, len(actual))
                expected = [0, 1, 20 if api == 2 else 0, 77 if api == 2 else 0,
                            77 if api == 2 else 0, -4, 0, 1, 0, 0, 0, 0, 0, 1, 1, 2]
                checks.append({'architecture': 'V4' if api >= 4 else 'V4.1', 'api': name, 'injection': kind,
                               'fixture_status': status, 'expected': expected, 'observed': list(actual),
                               'passed': status == 0 and list(actual) == expected,
                               'scope': 'Real CPU tensor clear/restore; ordinary refusal and failed-slot reset'})
        result = {'scope': 'Actual C ABI containment and reset eligibility; no model inference',
                  'library_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'test_hooks_required': True, 'checks': checks,
                  'passed': all(check['passed'] for check in checks)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
        print(f"Passed {sum(c['passed'] for c in checks)}/{len(checks)} actual C ABI boundary cases")
        if not result['passed']:
            raise SystemExit(1)
    finally:
        if directory is not None:
            directory.close()


if __name__ == '__main__':
    main()
