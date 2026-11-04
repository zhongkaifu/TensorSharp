#!/usr/bin/env python3
"""Compile the original-layout DFlash CUDA adapter; this does not validate inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='sm_120a')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = root / 'TensorSharp.Backends.Cuda/native/ninfer'
    source = directory / 'tensorsharp_q35_ninfer_draft.cu'
    upstream = directory / 'draft/upstream'
    manifest = json.loads((upstream / 'manifest.json').read_text())
    for item in manifest['sources']:
        data = (upstream / item['path']).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], item['path']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['blob'], item['path']
    if not args.output.is_absolute() or args.output.exists():
        raise ValueError('A fresh absolute output filename is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    command = [args.nvcc, '-std=c++17', '-O3', '--ftz=false', '--fmad=true', '-arch=' + args.arch,
               '-I' + str(upstream / 'kernel/vendor/src'), '-fatbin', str(source), '-o', str(args.output)]
    result = subprocess.run(command, capture_output=True, text=True)
    args.output.with_suffix('.build.log').write_text(result.stdout + result.stderr)
    evidence = dict(schema='ninfer-dflash-module-build/1', command=command, returncode=result.returncode,
                    sourceSha256=sha(source.read_bytes()), upstreamManifestSha256=sha((upstream / 'manifest.json').read_bytes()),
                    upstreamRevision=manifest['revision'], unchangedHeadersVerified=len(manifest['sources']),
                    compiler=version.stdout + version.stderr, module=str(args.output),
                    moduleSha256=sha(args.output.read_bytes()) if result.returncode == 0 else None,
                    gpuCorrectness='NOT_RUN', modelQuality='NOT_RUN', performance='NOT_RUN')
    args.output.with_suffix('.build.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
