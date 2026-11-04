#!/usr/bin/env python3
"""Verify unchanged pinned headers and build the standalone original shortlist head."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def verify(upstream):
    manifest = json.loads((upstream / 'manifest.json').read_text())
    if manifest['revision'] != 'e600a7d9e0d5921d767a6cb464373583b3cdcaf4' or manifest['modified_upstream_files']:
        raise ValueError('Unexpected shortlist dependency revision/modifications')
    for item in manifest['sources']:
        data = (upstream / item['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256'] or blob != item['blob']:
            raise ValueError('Pinned original file differs: ' + item['path'])
    for item in manifest['attribution_files']:
        data = (upstream / item['path']).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Original license/notice differs: ' + item['path'])
    return {'revision': manifest['revision'], 'originalFilesVerified': len(manifest['sources']),
            'headerFilesVerified': sum(x['path'].endswith(('.h', '.cuh')) for x in manifest['sources']),
            'attributionFilesVerified': len(manifest['attribution_files']),
            'manifestSha256': hashlib.sha256((upstream / 'manifest.json').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', default='sm_120a')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    folder = root / 'TensorSharp.Backends.Cuda/native/ninfer/shortlist'
    contract = verify(folder / 'upstream')
    if args.verify_only:
        print(json.dumps({'verification': contract, 'nativeCompilation': 'NOT_RUN', 'GPU': 'NOT_RUN'}, indent=2))
        return
    if args.output is None:
        parser.error('--output is required unless --verify-only')
    if any(p.exists() for p in (args.output, args.output.with_suffix(args.output.suffix + ".build.log"), args.output.with_suffix(args.output.suffix + ".build.json"))):
        raise FileExistsError('Use a fresh module path; do not replace previous evidence.')
    source = folder / 'tensorsharp_ninfer_shortlist.cu'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [args.nvcc, '-std=c++17', '-O3', '--fmad=true', '--ftz=false', '-arch=' + args.arch,
               '-I' + str(folder / 'upstream/kernel/vendor/src'), '-fatbin', str(source), '-o', str(args.output)]
    compiler = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    result = subprocess.run(command, capture_output=True, text=True)
    args.output.with_suffix(args.output.suffix + '.build.log').write_text(result.stdout + result.stderr)
    info = {'schema': 'ninfer-original-shortlist-module-build/1', 'originalSourceVerification': contract,
            'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'compiler': compiler.stdout + compiler.stderr, 'command': command, 'returncode': result.returncode,
            'module': str(args.output), 'moduleSha256': hashlib.sha256(args.output.read_bytes()).hexdigest() if result.returncode == 0 else None,
            'scope': 'original global Q4 templates plus range-preserving row-id remap; no serving activation'}
    args.output.with_suffix(args.output.suffix + '.build.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
