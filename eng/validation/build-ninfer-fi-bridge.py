#!/usr/bin/env python3
"""Compile the TensorSharp-owned FI placement bridge. No model/GPU validation is implied."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', choices=('sm_120a', 'sm_121a'), default='sm_120a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / 'TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_ninfer_fi_bridge.cu'
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError('A fresh absolute output filename is required')
    for suffix in ('.build.json', '.build.log'):
        if args.output.with_suffix(suffix).exists():
            raise ValueError('Build evidence must also be fresh')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    command = [args.nvcc, '-std=c++17', '-O3', '--ftz=false', '--fmad=true', '-arch=' + args.arch,
               '-fatbin', str(source), '-o', str(args.output)]
    result = subprocess.run(command, capture_output=True, text=True)
    args.output.with_suffix('.build.log').write_text(result.stdout + result.stderr)
    evidence = dict(schema='ninfer-fi-bridge-build/1', command=command, returncode=result.returncode,
                    sourceSha256=sha(source), compiler=version.stdout + version.stderr, module=str(args.output),
                    moduleSha256=sha(args.output) if result.returncode == 0 else None,
                    gpuCorrectness='NOT_RUN', modelQuality='NOT_RUN', performance='NOT_RUN')
    args.output.with_suffix('.build.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
