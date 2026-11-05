#!/usr/bin/env python3
"""Build the explicit original-source GDN chunk candidate; no model default changes."""
import argparse, hashlib, json, pathlib, subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--arch', default='sm_120a', choices=['sm_120a', 'sm_121a'])
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / 'TensorSharp.Backends.Cuda/native/ninfer/chunk'
    provenance = json.loads((root / 'official-source-manifest.json').read_text())
    if provenance['revision'] != 'e600a7d9e0d5921d767a6cb464373583b3cdcaf4':
        raise RuntimeError('Unexpected original source revision')
    for item in provenance['files']:
        path = root / 'upstream' / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Original source changed: ' + item['path'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = root / 'tensorsharp_ninfer_gdn_chunk.cu'
    command = [args.nvcc, '-std=c++20', '-O3', '--fmad=true', '--ftz=false',
               '--prec-div=true', '--prec-sqrt=true', '-arch=' + args.arch,
               '-I' + str(root / 'upstream/kernel/vendor/src'),
               '-Xptxas=-v', '-fatbin', str(source), '-o', str(args.output)]
    compiler = subprocess.run([args.nvcc, '--version'], text=True, capture_output=True, check=True)
    result = subprocess.run(command, text=True, capture_output=True)
    log_path = args.output.with_suffix(args.output.suffix + '.build.log')
    log_path.write_text(result.stdout + result.stderr)
    evidence = {'schema': 'ninfer-original-gdn-chunk-build/1', 'scope': 'candidate-component-only',
                'original_revision': provenance['revision'], 'original_files_verified': len(provenance['files']),
                'source_sha256': sha(source), 'provenance_sha256': sha(root / 'official-source-manifest.json'),
                'command': command, 'compiler': compiler.stdout + compiler.stderr,
                'returncode': result.returncode, 'build_log': str(log_path),
                'module': str(args.output), 'module_sha256': sha(args.output) if result.returncode == 0 else None}
    args.output.with_suffix(args.output.suffix + '.build.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
