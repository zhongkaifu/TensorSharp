#!/usr/bin/env python3
"""Compile one isolated T8/5120 norm-cache candidate; no production hook or GPU qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', choices=('sm_120a',), default='sm_120a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    native = root / 'TensorSharp.Backends.Cuda/native/ninfer'
    source = root / 'eng/validation/NinferNormCacheProbe/cached-norm.cu'
    log = args.output.with_suffix(args.output.suffix + '.build.log')
    report = args.output.with_suffix(args.output.suffix + '.build.json')
    if not args.output.is_absolute() or any(p.exists() for p in (args.output, log, report)):
        raise ValueError('A fresh absolute module filename and fresh build evidence are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    command = [args.nvcc, '-std=c++17', '-O3', '--ftz=false', '--fmad=true', '-Xptxas=-v',
               '-arch=' + args.arch, '-fatbin', str(source), '-o', str(args.output)]
    result = subprocess.run(command, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    evidence = dict(schema='ninfer-norm-cache-build/1', command=command, returncode=result.returncode,
                    sourceSha256=sha(source), referenceSourceSha256=sha(native / 'tensorsharp_q35_ninfer.cu'),
                    compiler=version.stdout + version.stderr, module=str(args.output),
                    moduleSha256=sha(args.output) if result.returncode == 0 else None,
                    entry='ts_ni_add_norm_t8_cached', grid=[8, 1, 1], block=[256, 1, 1], width=5120,
                    sourceLicense='BSD-3-Clause; independent register-reuse idea reference is Ignis e600a7d9 (Apache-2.0)',
                    gpuCorrectness='NOT_RUN', modelQuality='NOT_RUN', performance='NOT_RUN')
    report.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
