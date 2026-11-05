#!/usr/bin/env python3
"""Build one TS-owned BN8/stages2 experiment, with unchanged included BN32 control."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def abi(ptx, symbol):
    match = re.search(r'\.entry\s+' + re.escape(symbol) + r'\s*\((.*?)\)', ptx, re.S)
    if match is None:
        raise ValueError('Missing actual PTX entry: ' + symbol)
    arguments = match[1].split(',')
    expected = ['u64', 'u64', 'u32', 'u32', 'u64', 'u64', 'u64', 'u32', 'f32', 'u64', 'u64', 'u64']
    types = []
    for argument in arguments:
        kind = re.search(r'\.(u64|u32|f32)\b', argument)
        if kind is None or '[' in argument:
            raise ValueError('Unexpected PTX argument: ' + argument)
        types.append(kind[1])
    if types != expected:
        raise ValueError('Actual GEMM parameter types differ: ' + repr(types))
    offset = 0
    result = []
    for kind in types:
        size = 8 if kind == 'u64' else 4
        offset = (offset + size - 1) // size * size
        result.append(dict(type=kind, offset=offset, bytes=size))
        offset += size
    return dict(symbol=symbol, parameters=result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    folder = Path(__file__).resolve().parent
    root = folder.parents[3]
    include = root / 'TensorSharp.Backends.Cuda/native/kernels'
    source = folder / 'tensorsharp_a4_gdn_t8.cu'
    ptx = args.output.with_suffix('.ptx')
    log = args.output.with_suffix(args.output.suffix + '.build.log')
    report = args.output.with_suffix(args.output.suffix + '.build.json')
    if not args.output.is_absolute() or any(p.exists() for p in (args.output, ptx, log, report)):
        raise ValueError('Fresh absolute module/PTX/build-evidence paths are required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    flags = ['-std=c++17', '-O3', '--ftz=false', '--fmad=true', '-Xptxas=-v', '-arch=sm_120a', '-I' + str(include)]
    commands, output, code = [], [], 0
    for kind, target in [('-ptx', ptx), ('-fatbin', args.output)]:
        command = [args.nvcc] + flags + [kind, str(source), '-o', str(target)]
        commands.append(command)
        result = subprocess.run(command, capture_output=True, text=True)
        output.append(result.stdout + result.stderr)
        code = result.returncode
        if code:
            break
    log.write_text('\n'.join(output))
    entries = [abi(ptx.read_text(), s) for s in ('ts_q35_gemm_w4a4_8', 'ts_q35_gemm_w4a4_32')] if code == 0 else []
    inputs = [source] + [include / name for name in ('tensorsharp_q35_kernels.cu', 'tensorsharp_q35_gdn_prepared.cuh',
                                                       'tensorsharp_q35_dflash.cuh', 'tensorsharp_q35_topk.cuh')]
    evidence = dict(schema='ninfer-a4-gdn-bn8-build/1', returncode=code, commands=commands,
                    compiler=version.stdout + version.stderr,
                    sources=[dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in inputs],
                    module=str(args.output), moduleSha256=sha(args.output) if code == 0 else None,
                    ptxSha256=sha(ptx) if ptx.exists() else None, entries=entries,
                    fixedTrial=dict(rows=8, outDim=16384, columns=5120, splits=2, grid=[128, 1, 2], block=[256, 1, 1],
                                    candidateDynamicSharedBytes=43520, controlDynamicSharedBytes=76800),
                    scope='One BN8/stages2 specialization; same-source BN32 plus separate actual-old-asset comparison required',
                    gpuCorrectness='NOT_RUN', performance='NOT_RUN', productionIntegration=False)
    report.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
