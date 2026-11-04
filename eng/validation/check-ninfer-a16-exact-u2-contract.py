#!/usr/bin/env python3
"""Static source/order checks only; does not qualify native or GPU execution."""
import hashlib
import json
import pathlib
import re


BASELINE_SHA256 = '159f348e3cb6427aa8f21ee9c78f75ffe20c68b5928bfcceade205499337ecf4'


def code(text):
    # These pinned files contain no comment markers within string literals.
    # Preserve every non-comment token, including all launch declarations.
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return re.sub(r'\s+', '', text)


def check(root):
    folder = root / 'TensorSharp.Backends.Cuda/native/ninfer'
    baseline = folder / 'tensorsharp_q35_ninfer_a16_legacy_exact_vector.cu'
    candidate = folder / 'tensorsharp_q35_ninfer_a16_legacy_exact_u2.cu'
    before, after = baseline.read_bytes(), candidate.read_bytes()
    if hashlib.sha256(before).hexdigest() != BASELINE_SHA256:
        raise ValueError('The qualified vector source changed; do not silently rebase this contract.')
    old = 'constexprintphase_unroll=Tokens==1?Columns/kPhaseValues:1;'
    new = 'constexprintphase_unroll=Tokens==1?2:1;'
    original, proposed = code(before.decode()), code(after.decode())
    if original.count(old) != 1 or proposed != original.replace(old, new, 1):
        raise ValueError('A non-comment token changed outside the single phase-unroll policy.')
    cases = []
    for columns in (5120, 6144, 17408):
        phases = columns // 512
        if columns % 512 or phases % 2:
            raise ValueError('Supported phase geometry has an unexpected remainder.')
        bounded = [outer * 2 + inner for outer in range(phases // 2) for inner in range(2)]
        if bounded != list(range(phases)):
            raise ValueError('Bounded unroll changed logical phase order.')
        for lane in range(32):
            def sequence(order):
                return [(phase * 32 + lane) * 16 + j + half
                        for phase in order for j in range(8) for half in (0, 8)]
            if sequence(bounded) != sequence(range(phases)):
                raise ValueError('Low/high per-lane FMA column sequence changed.')
        cases.append({'columns': columns, 'phases': phases, 'oldT1Unroll': phases,
                      'newT1Unroll': 2, 'T2T3UnrollUnchanged': 1,
                      'oneAccumulatorFmaOrderUnchangedForAll32Lanes': True})
    return {'schema': 'ninfer-a16-exact-u2-static-contract/1',
            'baselineSha256': BASELINE_SHA256,
            'candidateSha256': hashlib.sha256(after).hexdigest(),
            'nonCommentTokenChanges': 'one constexpr T1 phase-unroll expression only',
            'decoderArithmeticXorSymbolsAbiGridUnchanged': True, 'cases': cases,
            'nativeCompilation': 'NOT_RUN', 'GPU': 'NOT_RUN',
            'modelQuality': 'NOT_RUN', 'performance': 'NOT_RUN',
            'limits': 'Source/order proof only. Compiler code generation, actual device bits and timing require the unchanged GPU gates.'}


if __name__ == '__main__':
    print(json.dumps(check(pathlib.Path(__file__).resolve().parents[2]), indent=2))
