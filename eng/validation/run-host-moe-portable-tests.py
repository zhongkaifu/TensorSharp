#!/usr/bin/env python3
"""Compile/run pure HostMoE policy tests with a POSIX C++ compiler.

These tests use standard C++ headers only: no ggml, device, model, allocator,
vendor library or native TensorSharp DLL is loaded. They do not certify GPU
execution, physical memory limits or Apple-device compatibility.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'TensorSharp.GGML.Native'
CASES = ('host_moe_group_quota_test', 'host_moe_phase_quota_test', 'host_moe_cache_timing_test',
         'host_moe_cache_policy_test', 'host_moe_file_read_plan_test', 'host_moe_direct_read_plan_test')
HEADERS = ('ggml_ops_host_moe_group_quota.h', 'ggml_ops_host_moe_phase_quota.h',
           'ggml_ops_host_moe_cache_timing.h', 'ggml_ops_host_moe_cache_policy.h',
           'ggml_ops_precision_policy.h', 'ggml_ops_host_moe_file_read_plan.h', 'ggml_ops_file_source.h')


def pin(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Use a POSIX host or WSL; Windows MSVC has a separate component qualification.')
    output = args.output.absolute()
    if '..' in args.output.parts or not any(output.is_relative_to((ROOT / name).absolute())
                                           for name in ('artifacts', 'docs/validation')):
        parser.error('Use a fresh ignored evidence directory in this checkout.')
    output.mkdir(parents=True, exist_ok=False)
    files = [NATIVE / 'tests' / (case + '.cpp') for case in CASES] + [NATIVE / name for name in HEADERS]
    before = {str(path.relative_to(ROOT)): pin(path) for path in files}
    compiler = shutil.which(args.compiler)
    report = dict(passed=False, scope=__doc__, started_utc=datetime.now(timezone.utc).isoformat(),
                  platform=sys.platform, compiler=compiler, sources_before=before, cases=[])
    try:
        if not compiler:
            raise FileNotFoundError('C++ compiler unavailable: ' + args.compiler)
        report['compiler_version'] = subprocess.check_output([compiler, '--version'], text=True, timeout=15)
        for case in CASES:
            executable = output / case
            command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       str(NATIVE / 'tests' / (case + '.cpp')), '-o', str(executable)]
            row = dict(name=case, compile_command=command, passed=False)
            report['cases'].append(row)
            with (output / (case + '-build.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
            (output / (case + '-run.log')).write_text(run.stdout + run.stderr)
            row.update(exit_code=run.returncode, executable=pin(executable), stdout=run.stdout, stderr=run.stderr)
            if run.returncode or not run.stdout.startswith('PASS'):
                raise RuntimeError('Pure component test failed: ' + case)
            row['passed'] = True
        report['sources_after'] = {str(path.relative_to(ROOT)): pin(path) for path in files}
        if report['sources_after'] != before:
            raise RuntimeError('Source changed during qualification.')
        report['passed'] = True
    except Exception as error:
        report['error'] = repr(error)
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], output=str(output), error=report.get('error'))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
