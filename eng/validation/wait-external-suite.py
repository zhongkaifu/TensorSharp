#!/usr/bin/env python3
"""Keep a release server alive while its owning external client runs a suite.

The owner writes a fresh, nonce-bound completion file only after preserving
the client report and remote process postcheck. This helper performs no
inference and does not itself establish correctness or release qualification.
"""
import argparse
import json
from pathlib import Path
import re
import time


def identity_timestamp(path, nonce):
    identity = json.loads(path.read_text())
    if identity.get('nonce') != nonce:
        raise ValueError('Ready identity belongs to another campaign')
    return path.stat().st_mtime_ns


def completion_result(path, nonce, ready_timestamp=None):
    # Identity and completion live on the same VM filesystem. A client may
    # finish between identity publication and waiter startup; that completion
    # is valid only when it is at least as new as the nonce-bound identity.
    if ready_timestamp is not None and path.stat().st_mtime_ns < ready_timestamp:
        raise ValueError('External completion predates the ready identity')
    result = json.loads(path.read_text())
    if result.get('nonce') != nonce:
        raise ValueError('External completion belongs to another campaign')
    passed = (result.get('client_exit_code') == 0
              and result.get('independent_code_exit_code') == 0
              and result.get('remote_binding_postcheck_passed') is True
              and re.fullmatch('[0-9a-f]{64}', result.get('client_report_sha256') or '') is not None)
    return {'external_client_passed': passed,
            'client_report_sha256': result.get('client_report_sha256'),
            'scope': 'External client result; full release matrix remains separate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--ready-identity', type=Path,
                        help='Fresh nonce-bound identity on the same filesystem; permits a later already-published completion')
    parser.add_argument('--timeout', type=float, default=10800)
    args = parser.parse_args()
    ready_timestamp = identity_timestamp(args.ready_identity, args.nonce) if args.ready_identity else None
    if args.completion.exists() and ready_timestamp is None:
        parser.error('Completion file already exists; use a fresh campaign')
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    deadline = time.monotonic() + args.timeout
    print('Waiting for the external client completion and binding postcheck.', flush=True)
    while time.monotonic() < deadline:
        if args.completion.exists():
            # The owner writes a temporary sibling then atomically renames it.
            result = completion_result(args.completion, args.nonce, ready_timestamp)
            print(json.dumps(result), flush=True)
            return int(not result['external_client_passed'])
        time.sleep(2)
    print('External client completion timed out.', flush=True)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
