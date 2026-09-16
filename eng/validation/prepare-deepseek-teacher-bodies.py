#!/usr/bin/env python3
"""Capture the pinned benchmark's HTTP-body construction without any transport.

Only the original run_openai_chat AST node is executed. Its two transport
functions are replaced with recorders; imports, model loading and networking
are not executed. JSON byte representation here is order-preserving UTF-8 evidence,
not a claim about requests' historical whitespace/escaping on the wire.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Optional

ENGINE_SHA256 = 'd0c7b8dbacf70d6a663d39a51a6fa5e77cd36714d70591f8dc0e8ae9fe304ea6'
PROFILE_SHA256 = '0a29cf8c4b1bd6c4567848314ec246e813f6f145ae8d7c40b14253d7f831639d'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def ordered_json(value):
    # Tool schema property order reaches the production prompt renderer. Only
    # whitespace/escaping may be normalized; NEVER sort submitted object keys.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != expected:
        raise ValueError('SHA256 mismatch: ' + str(path))
    return raw


def engine_body_builder(engine_raw):
    if digest(engine_raw) != ENGINE_SHA256:
        raise ValueError('Original engines.py pin mismatch')
    tree = ast.parse(engine_raw.decode('utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_openai_chat']
    if len(nodes) != 1:
        raise ValueError('Expected one original run_openai_chat definition')
    function = nodes[0]
    # Source file is independently pinned; still refuse a newly introduced call.
    calls = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    if calls != {'_run_streaming', '_run_blocking'}:
        raise ValueError('Unexpected direct call in pinned request builder')
    def capture(url, body, timeout_s):
        return {'url': url, 'body': copy.deepcopy(body), 'timeout_s': timeout_s}
    namespace = {'Optional': Optional, '_run_streaming': capture, '_run_blocking': capture}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-run_openai_chat>', 'exec'), namespace)
    return namespace['run_openai_chat'], digest(ast.get_source_segment(engine_raw.decode('utf-8'), function).encode('utf-8'))


def prepare(plan_path, plan_sha, engine_path, profile_path, output):
    plan_raw = pinned(plan_path, plan_sha)
    plan = json.loads(plan_raw)
    profile = json.loads(pinned(profile_path, PROFILE_SHA256))
    engine_raw = pinned(engine_path, ENGINE_SHA256)
    build, function_sha = engine_body_builder(engine_raw)
    rows = plan['requests']
    ids = [r['id'] for r in rows]
    if len(rows) != 113 or plan['request_count'] != 113 or len(set(ids)) != 113:
        raise ValueError('Expected all 113 distinct original request IDs')
    args = profile['extra_args']
    if '--no-skills' not in args or '--max-tokens' in args or any(k in profile['env'] for k in ('MAX_TOKENS', 'TS_JSON_GRAMMAR')):
        raise ValueError('Review changed server preprocessing defaults before export')
    if profile['env']['MAX_CONTEXT'] != '65536':
        raise ValueError('Original runtime context changed')
    output = Path(output)
    if output.exists():
        raise ValueError('Output directory must be new')
    output.mkdir(parents=True)
    (output / 'requests').mkdir()
    records = []
    model_name = Path(profile['model']).stem
    for index, row in enumerate(rows):
        request = row['request']
        before = canonical(request)
        ordered_before = ordered_json(request)
        if digest(before) != row['request_canonical_sha256']:
            raise ValueError('Original request canonical hash mismatch: ' + row['id'])
        captured = build('http://teacher-export.invalid', model_name, **copy.deepcopy(request))
        if ordered_json(request) != ordered_before:
            raise ValueError('Request builder mutated archived input')
        body = captured['body']
        if body['think'] is not False:
            raise ValueError('Original 113-request schedule must stay nonthinking')
        body_bytes = ordered_json(body)
        request_name, body_name = f'requests/{index:03d}.engine.json', f'requests/{index:03d}.body.json'
        (output / request_name).write_bytes(ordered_before)
        (output / body_name).write_bytes(body_bytes)
        records.append({'id': row['id'], 'engine_arguments_file': request_name,
                        'engine_arguments_sha256': digest(ordered_before),
                        'engine_arguments_canonical_sha256': digest(before), 'body_file': body_name,
                        'body_sha256': digest(body_bytes), 'source': row['source']})
    result = {'schema_version': 1, 'status': 'bodies-captured-no-tokenizer-executed',
              'release_qualified': False, 'plan_sha256': plan_sha,
              'engine_sha256': ENGINE_SHA256, 'engine_function_utf8_sha256': function_sha,
              'source_profile_sha256': PROFILE_SHA256, 'model_name': model_name,
              'request_count': 113, 'ordered_request_ids_sha256': digest(canonical(ids)),
              'server_settings': {'architecture': 'deepseek41', 'context_limit': 65536,
                                  'default_max_tokens': 20000, 'max_tokens_pinned': False,
                                  'skills_enabled': False, 'sampling_defaults_pinned': False,
                                  'fresh_history_tracking': True, 'json_grammar_disabled': False},
              'requests': records,
              'limits': ['Transport was replaced, not executed.',
                         'Body and argument JSON preserve original property order and values, not historical wire whitespace/escaping.',
                         'Token export must bind actual compiled managed identities and checkpoint metadata.']}
    (output / 'bodies.json').write_bytes(canonical(result))
    (output / 'teacher-plan.json').write_bytes(plan_raw)
    (output / 'original-engines.py').write_bytes(engine_raw)
    (output / 'source-profile.json').write_bytes(pinned(profile_path, PROFILE_SHA256))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'engine', 'profile', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = prepare(args.plan, args.plan_sha256, args.engine, args.profile, args.output)
    print(json.dumps({'status': result['status'], 'requests': result['request_count'],
                      'bodies_sha256': digest((Path(args.output) / 'bodies.json').read_bytes())}))


if __name__ == '__main__':
    main()
