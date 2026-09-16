#!/usr/bin/env python3
"""Freeze the prepared Qwen follow-up sources, exact old workloads and new separate inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--benchmark-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); root = a.workspace.resolve(); prep = a.prepared.resolve(); out = a.output.resolve()
    old = prep.parent / 'qwen38-layer3-v5-prepared'; pin = json.loads((old / 'package.json').read_bytes())
    if sha(old / 'package.json') != 'f99afed50db9a859320fa610367d9614ba74863a027dd9bf1bbf21b58df23378': raise ValueError('Original package changed')
    out.mkdir(parents=True, exist_ok=False)
    def put(path, source):
        p = out / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(source.read_bytes())
    bench = a.benchmark_directory.resolve(); b = json.loads((bench / 'prepared.json').read_bytes())
    for name, digest in b['files'].items():
        if sha(bench / name) != digest: raise ValueError('Derived benchmark changed')
        put('benchmark/' + name, bench / name)
    put('benchmark/prepared.json', bench / 'prepared.json')
    for p in sorted((root / 'eng/validation/qwen38_mtp_followup').iterdir()):
        if p.is_file(): put('repo/eng/validation/qwen38_mtp_followup/' + p.name, p)
    for name in ('run-release-profile.py', 'release_application_identity.py', 'sample-release-telemetry.py'):
        put('repo/eng/validation/' + name, root / 'eng/validation' / name)
    for name, digest in pin['files'].items():
        if name.startswith('harness/benchmarks/engine_comparison/'):
            if sha(old / name) != digest: raise ValueError('Original HTTP helper changed: ' + name)
            put('repo/' + name[len('harness/'):], old / name)
    for name in ('request.json', 'input.json'): put('new-long62k/' + name, prep / 'new-long62k-r1' / name)
    for name in ('count.json', 'tokens.i32', 'rendered.utf8'): put('new-long62k/' + name, prep / 'new-long62k-count-r1' / name)
    for name in ('local-checks.json', 'count-managed-pins.json', 'count-build.log', 'count-run.log', 'benchmark-build.log', 'offline-tests.log', 'build-local-checks.py'):
        put('local-verification/' + name, prep / name)
    for name in ('final-local-checks.json', 'final-local-checks.py', 'benchmark-final-prepare.log', 'benchmark-final-build.log', 'final-offline-tests.log'):
        put('local-verification/' + name, prep / name)
    put('http32-catalog.json', prep / 'http32-plan/catalog.json')
    template = json.loads((old / 'profiles/candidate.json').read_bytes())
    original_id = template['id']; template['id'] = '@PROFILE@'; template['expected_native_sha256'] = None
    template['expected_harness_sha256'] = {}
    template['env']['BENCH_CONFIG'] = '@PACKAGE@/repo/benchmarks/engine_comparison/benchmark_config.json'
    for suite in template['suites']:
        for i, value in enumerate(suite):
            if value == original_id: suite[i] = '@PROFILE@'
    head = {'path': '/workspace/models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf', 'bytes': 2786568256,
            'sha256': '5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6',
            'repository': 'unsloth/Qwen3.8-Flash-Next-GGUF', 'revision': '38bb39ee97821de2c9009abb7e93950eec396e66'}
    template['artifact_inventory']['companions'].append({**head, 'download_verified_sha256': head['sha256'], 'publisher_digest_kind': 'lfs-sha256'})
    extra = '{repo}/eng/validation/qwen38_mtp_followup/'
    template['suites'] += [
        ['{python}', extra + 'http_contract.py', '--model', '{model_id}', '--url', '{url}', '--execute', '--output', '{output}/tool-contract32'],
        ['{python}', extra + 'http_extra.py', '--case', 'counted-long62k', '--model', '{model_id}', '--url', '{url}', '--execute',
         '--request', '@PACKAGE@/new-long62k/request.json', '--request-sha256', sha(out / 'new-long62k/request.json'),
         '--count', '@PACKAGE@/new-long62k/count.json', '--count-sha256', sha(out / 'new-long62k/count.json'), '--output', '{output}/new-counted-long62k'],
    ]
    for blocking in (False, True):
        template['suites'].append(['{python}', extra + 'http_extra.py', '--case', 'video2-history', '--model', '{model_id}', '--url', '{url}', '--execute',
            '--video', pin['media_root'] + '/ordered-cards.mp4', '--output', '{output}/video2-history-' + ('blocking' if blocking else 'stream')]
            + (['--blocking'] if blocking else []))
    template['scope_note'] = 'UNBOUND TEMPLATE. Original six HTTP suites and every original oracle retained, including oversized64K failure. Four additional suites are separate: 32 tool contracts, counted62K, video2/history stream and blocking. No actual sandbox/tool execution here.'
    (out / 'http-profile-template.json').write_text(json.dumps(template, indent=2) + '\n', encoding='utf-8')
    record = {'schema': 1, 'status': 'prepared-not-bound-not-executed', 'release_qualified': False,
        'native_source_package_sha256': b['source_package_manifest_sha256'], 'native_source_archive_sha256': b['source_archive_sha256'],
        'benchmark_prepared_sha256': sha(bench / 'prepared.json'), 'original_http_workloads_package_sha256': sha(old / 'package.json'),
        'model': {'trunk': pin['models'][:3] if 'models' in pin else copy.deepcopy(template['artifact_inventory']['weights']),
                  'mmproj': copy.deepcopy(template['artifact_inventory']['companions'][0]), 'head': head},
        'inventory_sha256': pin['inventory_sha256'], 'fixtures_root': pin['media_root'], 'fixtures_sha256': pin['media_sha256'],
        'new_long_input': json.loads((out / 'new-long62k/input.json').read_bytes()),
        'new_long_count': json.loads((out / 'new-long62k/count.json').read_bytes()),
        'scheduler_scope': {'dense_rows_per_process': 16, 'long_rows_per_process': 12, 'warmup_rows_per_process': 1,
                            'learned_draft_window': 3, 'minimum_proposal_probability': 0.15, 'probability_source': 'Pinned production DraftHeadSpeculator.DefaultGate, fixed before full-model outcomes',
                            'three_gpu_mode': 'Contiguous layer split, not routed-expert tensor parallelism',
                            'tiers': ['dense: actual injected/rendered prompt + max output + four verifier positions <=2051', 'long-qsa-unqualified: longer original pre-QSA runtime, no equivalence/release claim'],
                            'ordering': ['plain,MTP', 'MTP,plain', 'plain,MTP']},
        'runtime_obligations': [
            'Root owns exclusive VM authorization; do not overlap server/benchmark, download, conversion, compiler, sampler duplication or other GPU workload.',
            'Bind only after all applicable Qwen CPU/CUDA/head/target and C ABI gates pass, final source/native/upstream audits finish. Preserve original failed DS/TP gates and release_qualified=false.',
            'Both plain and MTP attach the same exact trained head, trunk shards and mmproj. Check immutable publisher digest manifests plus inode/mtime/size before/after; no checkpoint hashing during timing.',
            'Record full managed application file set, exact native mapped path/SHA and process start ticks before/after each process. Copy the bound native into isolated bench-app; no alternate library search path.',
            'Record actual 3-GPU layer placement and memory on each rank, head-attachment banner and requested/effective KV. Do not infer placement from --tp3 alone.',
            'Use one telemetry sampler per process, interval1s, preserve /proc status/IO/start ticks, cgroupv1 usage/failcnt/pressure and GPU0,1,2 usage. Sampled peaks have1s resolution, not allocator peak precision.',
            'Build the private benchmark only against the bound app and recheck every referenced DLL afterwards. Local compile was compatibility-only, not the VM application.',
            'Require exact per-request input token digest/output tokens/finish, learned verify engagement, accepted proposals, exercised rollback and actual retained reuse. Missing coverage fails; do not substitute ngram or fallback.',
            'Run compare.py with all three warmed pairs and measured RSS/cgroup/GPU sampled peaks. Keep every5% failure; no performance claim when qualification or stable ordering/identity fails.',
            'Actual HTTP video_url is presently rejected by Qwen4Exp protocol. Keep its failure and unrun follow-up; ordered static-image checks are not video support.',
            'The32 HTTP probes are separate from32 model-free adapter unit rows and original29 policy cases. No HTTP tool call implies real shell/sandbox execution; separate46-case agent owner is required.',
            'Future v7 tokenizer/render must reproduce the new long case count/bytes or retain a failed input-parity gate; original67,154-token case remains unchanged.'
        ],
        'known_failures': ['Original Qwen38 v5 parser exposed XML instead of API calls, fixed in current managed build awaiting real replay.',
                           'Original required/no-tools and undeclared-named API validation fixed,152 model-free managed rows passed; actual32 replay unrun.',
                           'Original v5 candidate/baseline first8K concurrent decode wave had4 nonidentical greedy outputs; no cause or resolution claimed.',
                           'Pre-QSA long context is unqualified; original DeepSeek TP synthetic failure has not been waived.',
                           'Current Qwen4Exp advertises no retained/checkpoint clone or KV truncation support. Required reused tokens may remain zero; retain that coverage failure rather than claiming a clone.'],
        'files': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out / 'package.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(record['files']), 'package_sha256': sha(out / 'package.json'), 'status': record['status']}))


if __name__ == '__main__': main()
