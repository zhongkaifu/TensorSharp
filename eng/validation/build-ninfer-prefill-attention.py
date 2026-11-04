#!/usr/bin/env python3
"""Build the profile-directed BF16 prefill adapter and its unchanged original oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISION = 'e600a7d9e0d5921d767a6cb464373583b3cdcaf4'
ORIGINAL = 'kernel/vendor/src/ops/kernel/gqa_attention_prefill_bf16.cuh'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def owned_header(source):
    """Only remove unused append, rename symbols, and replace the K/V address accessor."""
    begin = source.index('template <typename Geometry, typename Metadata>\n__global__ void gqa_attention_prefill_fill_bf16_kernel(')
    end = source.index('// Stage one [Bc, D] K or V tile', begin)
    source = source[:begin] + source[end:]
    namespace = 'namespace ninfer::ops {\n'
    plane = '''namespace ninfer::ops {

// TensorSharp address policy only. K/V were already appended by the existing
// BF16 preparation kernel. Each owned block holds256 rows per KV head.
struct TsNinferPrefillPlane {
    const std::uint8_t* const* blocks;
    std::int64_t offset;
};
static_assert(sizeof(TsNinferPrefillPlane) == 16);
'''
    if source.count(namespace) != 1:
        raise ValueError('Original namespace contract changed')
    source = source.replace(namespace, plane)
    old = '''    const __nv_bfloat16* cache_block =
        Rotated ? cache + (static_cast<std::int64_t>(kv_head) * scratch_span + k0 - key_base) * D
                : cache + paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
                              physical_page, kv_head, k0 & kPagedKVPageMask, 0);'''
    new = '''    static_assert(!Rotated, "The TensorSharp adapter admits only original-frame BF16 KV");
    (void)physical_page;
    (void)scratch_span;
    const auto* plane = reinterpret_cast<const TsNinferPrefillPlane*>(cache);
    const auto* block = plane->blocks[k0 / 256];
    const __nv_bfloat16* cache_block = reinterpret_cast<const __nv_bfloat16*>(
        block + plane->offset +
        (static_cast<std::int64_t>(kv_head) * 256 + k0 % 256) * D * sizeof(__nv_bfloat16));'''
    if source.count(old) != 1:
        raise ValueError('Original cache address contract changed')
    source = source.replace(old, new)
    source = source.replace('gqa_prefill_stage_kv', 'ts_ni_prefill_stage_kv')
    source = source.replace('gqa_attention_prefill_bf16_kernel', 'ts_ni_prefill_bf16_kernel')
    source = source.replace('kGqaPrefillRotatedSmemBytes', 'kTsNinferPrefillRotatedSmemBytes')
    return ('// Copyright (c) Zhongkai Fu. TensorSharp address adapter under Apache-2.0.\n'
            '// Derived from gpillon/ignis ' + REVISION + '\n'
            '// ' + ORIGINAL + '\n'
            '// Original arithmetic, scheduling and rounding are unchanged; upstream/ is pristine.\n'
            '// See official-source-manifest.json and upstream/{LICENSE,NOTICE,kernel/vendor/LICENSE}.\n' + source)


def verify(folder):
    raw = (folder / 'official-source-manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest['revision'] != REVISION or manifest['modifiedUpstreamFiles']:
        raise ValueError('Unexpected original source provenance')
    for item in manifest['files']:
        data = (folder / 'upstream' / item['path']).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != item['bytes'] or digest(data) != item['sha256'] or blob != item['gitBlobSha1']:
            raise ValueError('Original dependency changed: ' + item['path'])
    original = (folder / 'upstream' / ORIGINAL).read_text()
    owned = (folder / 'tensorsharp_ninfer_prefill_bf16.cuh').read_text()
    if owned != owned_header(original):
        raise ValueError('Owned core differs beyond its declared paged-address adaptation')
    return dict(revision=REVISION, originalFilesVerified=len(manifest['files']), modifiedUpstreamFiles=[],
                sourceManifestSha256=digest(raw), ownedArithmeticMatchesOriginal=True,
                ownedHeaderSha256=digest(owned.encode()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', choices=('sm_120a', 'sm_121a'), default='sm_120a')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = root / 'TensorSharp.Backends.Cuda/native/ninfer/prefill_attention'
    verification = verify(folder)
    if args.verify_only:
        print(json.dumps(dict(sourceVerification=verification, nativeCompilation='NOT_RUN', GPU='NOT_RUN'), indent=2))
        return 0
    if args.output is None or not args.output.is_absolute() or args.output.exists():
        parser.error('--output must name a fresh absolute module file')
    for suffix in ('.build.json', '.build.log'):
        if args.output.with_suffix(suffix).exists():
            raise ValueError('Fresh build evidence required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = folder / 'tensorsharp_ninfer_prefill_attention.cu'
    compiler = subprocess.run([args.nvcc, '--version'], capture_output=True, text=True, check=True)
    command = [args.nvcc, '-std=c++20', '-O3', '--ftz=false', '--fmad=true', '-arch=' + args.arch,
               '-I' + str(folder / 'upstream/kernel/vendor/src'), '-I' + str(folder / 'upstream/kernel/vendor/include'),
               '-fatbin', str(source), '-o', str(args.output)]
    result = subprocess.run(command, capture_output=True, text=True)
    args.output.with_suffix('.build.log').write_text(result.stdout + result.stderr)
    evidence = dict(schema='ninfer-prefill-attention-build/1', sourceVerification=verification,
                    sourceSha256=digest(source.read_bytes()), command=command, returncode=result.returncode,
                    compiler=compiler.stdout + compiler.stderr,
                    moduleSha256=digest(args.output.read_bytes()) if result.returncode == 0 else None,
                    gpuCorrectness='NOT_RUN', modelQuality='NOT_RUN', performance='NOT_RUN')
    args.output.with_suffix('.build.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
