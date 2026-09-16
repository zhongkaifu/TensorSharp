#!/usr/bin/env python3
"""Prepare immutable teacher-forced request/profile inputs; runs no model or remote command."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'docs/validation/ggml-no-patch-2026-09-15'
ARCHIVE = REPORTS / 'sparse-attention-candidate/deepseek41-sparse-attention'
PINNED_REPORTS = {
    'quality.json': 'd19b28b7d3aa3aad1aea9e64894fcafcc2b08198999f46542f4e26312c32d71e',
    'structured-tool-results.json': 'a6492db55d221313f53d1fc3e703d4f22eb8dfaa3650afbf32a49d5381e7ce84',
    'serial-tool-workflows.json': '27bb15609c7eb2846802ed5806c9423ab4f42c3d6059f8121297d22a706f98aa',
    'long-context.json': 'b8a8b43406eb7a3f5f35ac4967f77c688bf367683ab864bd988e9f089d91a69e',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_requests(report, report_name, report_sha):
    if report.get('weights_id') != '58d8ac86298fdf85a2440defee08b1abcad32e45-Q4_K_M':
        raise ValueError('Unexpected model weights identity')
    if report.get('thinking') is not False or not report.get('run_complete'):
        raise ValueError('Expected complete original nonthinking report')
    if len(report['cases']) != report['execution_plan']['expected_cases']:
        raise ValueError('Archived report is missing cases')
    rows = []
    for case_index, case in enumerate(report['cases']):
        for turn_index, turn in enumerate(case['turns']):
            request = copy.deepcopy(turn['request'])
            if not request.get('messages'):
                raise ValueError('Missing request history')
            if any(not isinstance(m.get('content'), (str, type(None))) for m in request['messages']):
                raise ValueError('Text-only plan cannot silently convert media content')
            rows.append({
                'id': f'{report_name[:-5]}/{case["tag"]}/turn{turn_index}',
                'source': {'report': report_name, 'sha256': report_sha, 'case_index': case_index,
                           'turn_index': turn_index, 'scenario': case['scenario'], 'tag': case['tag'],
                           'concurrency': case['concurrency'], 'case_status': case['status']},
                'request': request,
                'request_canonical_sha256': hashlib.sha256(canonical(request)).hexdigest(),
                'archived_assistant_message': copy.deepcopy(turn['metrics'].get('assistant_message')),
                'token_ids_status': 'not-exported',
                'continuation_status': 'not-recorded; use one frozen baseline raw-token trace, never retokenized API text',
            })
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate request identity')
    return rows


def build_plan(native_sha):
    if not re.fullmatch('[0-9a-f]{64}', native_sha):
        raise ValueError('An explicit lowercase native SHA256 is required')
    model_path = REPORTS / 'deepseek41-q4km.json'
    model = json.loads(model_path.read_text(encoding='utf-8'))
    if model['status'] != 'verified' or len(model['files']) != 11:
        raise ValueError('Expected the verified eleven-shard checkpoint')
    rows, sources = [], []
    for name, expected in PINNED_REPORTS.items():
        path = ARCHIVE / name
        if sha(path) != expected:
            raise ValueError('Archived request report changed: ' + name)
        report = json.loads(path.read_text(encoding='utf-8'))
        part = extract_requests(report, name, expected)
        rows.extend(part)
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': expected,
                        'cases': len(report['cases']), 'requests': len(part)})
    if len(rows) != 113:
        raise ValueError('Original four reports must contain 113 requests from 63 cases')
    source_profile_path = REPORTS / 'sparse-attention-candidate/sparse-attention-build/full-profile.json'
    if sha(source_profile_path) != '0a29cf8c4b1bd6c4567848314ec246e813f6f145ae8d7c40b14253d7f831639d':
        raise ValueError('Archived native execution profile changed')
    source_profile = json.loads(source_profile_path.read_text(encoding='utf-8'))
    variants = {}
    for name, tp in [('non-tp', 0), ('expert-tp7', 7)]:
        env = copy.deepcopy(source_profile['env'])
        env['TS_DSV41_TP'] = str(tp)
        variants[name] = {'status': 'prepared-not-executed', 'expected_native_sha256': native_sha,
                          'native_load': {'backend': 'CUDA', 'n_gpu': 7, 'n_ctx': 65536,
                                          'n_ubatch': 512, 'n_threads': 32, 'n_cpu_moe': 12},
                          'environment': env, 'requested_expert_tp_ranks': tp,
                          'effective_placement_status': 'unobserved'}
    return {'schema_version': 1, 'status': 'prepared-not-executed', 'qualified': False,
            'scope': 'Text-only raw model logits; no HTTP grammar/sampling, media, skills, or throughput qualification',
            'model': {'manifest': str(model_path.relative_to(ROOT)), 'manifest_sha256': sha(model_path),
                      'repository': model['repository'], 'revision': model['revision'],
                      'directory': model['directory'], 'files': model['files'],
                      'full_hash_policy': 'Reuse publisher-verified manifest with explicit immutable-file attestation; otherwise rehash outside timing'},
            'actual_tensor_types': {'path': 'docs/validation/ggml-no-patch-2026-09-15/tp7-checkpoint-v5/actual-q4km-routed-types.json',
                                   'sha256': sha(REPORTS / 'tp7-checkpoint-v5/actual-q4km-routed-types.json')},
            'sources': sources, 'request_count': len(rows), 'requests': rows, 'variants': variants,
            'implementation_status': {'request_export': 'implemented-model-free',
                                      'gguf_token_export': 'planned public API harness; not implemented',
                                      'native_logit_capture': 'planned existing C ABI driver; not implemented',
                                      'logit_comparison': 'specified in protocol; not implemented'},
            'protocol': {'context': 65536, 'prompt_chunk': 512, 'continuation_max_tokens': 64,
                         'capture': 'Every vocabulary F32 value from each Forward call, at its last input position',
                         'interior_prefill_rows': 'Not exposed by current V4.1 API; do not call ForwardSpec (it rejects V4.1)',
                         'primary_sequence': ['non-tp baseline records raw continuation token IDs',
                                              'non-tp replay same tokens/positions/chunks, including reset and slot controls',
                                              'expert-tp7 replay identical frozen tokens/positions/chunks',
                                              'expert-tp7 self-repeat; retain all failed requests'],
                         'boundary_probe_tail_lengths': [1, 4, 5, 8, 9, 16, 31, 32],
                         'boundary_probe_selection': 'quality initial turn, concurrency1, every scenario; same full prompt end position',
                         'cache_controls': ['fresh reset', 'same-handle exact replay', 'unrelated request then reset',
                                            'four independent native slots, round-robin', 'four-slot fused native decode'],
                         'thinking': 'Original requests are nonthinking. Separate thinking token exports required; no inferred coverage',
                         'media': 'Excluded; needs pinned image embeddings/masks plus vision companion and native vision API'},
            'gates': {'identity_and_positions': 'exact required', 'finite_logits': 'all vocabulary elements required',
                      'existing_fixture_allclose_diagnostic': {'atol': 2e-5, 'rtol': 2e-5,
                           'source': 'eng/tests/dsv41-inference.py', 'scope': 'Predeclared diagnostic; not an established full-Q4KM release bound'},
                      'native_tp_pairwise_1e_5': 'Original failed checkpoint fixture gate remains failed; untouched',
                      'argmax': 'Record every mismatch and both top1 margins; do not resume separate generated paths',
                      'kl_rms_full_model_limits': 'No established limits found; descriptive until independently reviewed',
                      'release_qualified': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output file required')
    plan = build_plan(args.native_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(args.output), 'sha256': sha(args.output), 'requests': plan['request_count'],
                      'status': plan['status'], 'qualified': False}))


if __name__ == '__main__':
    main()
