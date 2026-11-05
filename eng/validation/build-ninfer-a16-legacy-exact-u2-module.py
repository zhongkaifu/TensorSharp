#!/usr/bin/env python3
"""Build one separately identified bounded-phase exact-A16 scheduling candidate."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='sm_120a')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    source = root / 'TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_q35_ninfer_a16_legacy_exact_u2.cu'
    if args.output.exists():
        raise FileExistsError('Use a fresh module path; do not overwrite an earlier candidate image.')
    contract = subprocess.run([sys.executable, str(root / 'eng/validation/check-ninfer-a16-exact-u2-contract.py')],
                              capture_output=True, text=True, check=True)
    verified = json.loads(contract.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.nvcc, '-std=c++17', '-O3', '--fmad=true', '--ftz=false',
               '-arch=' + args.arch, '-fatbin', str(source), '-o', str(args.output)]
    compiler = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    result = subprocess.run(command, capture_output=True, text=True)
    args.output.with_suffix(args.output.suffix + '.build.log').write_text(result.stdout + result.stderr)
    info = {'schema': 'ninfer-a16-legacy-exact-u2-module-build/1',
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'static_source_contract': verified,
            'scope': 'T1-two-phase-unroll-only; unchanged-arithmetic scheduling hypothesis',
            'compiler': compiler.stdout + compiler.stderr, 'command': command,
            'returncode': result.returncode, 'module': str(args.output),
            'module_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest() if result.returncode == 0 else None}
    args.output.with_suffix(args.output.suffix + '.build.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
