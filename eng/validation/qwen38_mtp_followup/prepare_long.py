#!/usr/bin/env python3
"""Create a NEW long_62k request using the archived corpus/oracle; never edit long_64k."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archived-package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = a.archived_package / 'package.json'
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'f99afed50db9a859320fa610367d9614ba74863a027dd9bf1bbf21b58df23378':
        raise ValueError('Original workload package changed')
    pins = json.loads(raw)
    hp = a.archived_package / 'harness/benchmarks/engine_comparison'
    os.environ['BENCH_CONFIG'] = str(hp / 'benchmark_config.json')
    modules = {}
    for name in ('config', 'engines', 'scenarios', 'validate_inference'):
        path = hp / (name + '.py')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pins['files'][path.relative_to(a.archived_package).as_posix()]:
            raise ValueError('Original helper changed: ' + name)
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(data, str(path), 'exec'), module.__dict__)
        modules[name] = module
    v = modules['validate_inference']
    # Confirm the corpus-selection environment reproduces the known failed
    # original before creating a differently named, shorter workload.
    old_tag = json.loads((a.archived_package.parent / 'qwen38-candidate-completed/long-context.json').read_bytes())['cases'][-1]['tag']
    original = v.case_spec('long_64k', old_tag)
    original_input = {**original, 'sampling': v.SAMPLING, 'thinking': False, 'stream': True}
    if v.digest(original_input) != '5112b84d0135c26ceee7adebcf7588a979e4720f0d508a1d3151340c994fb459':
        raise ValueError('Archived 67,154-token fixture no longer reconstructs exactly')
    tag = 'qwen38-mtp-new-counted-long62k-r1'
    spec = v.case_spec('long_62k', tag)
    initial = {**spec, 'sampling': v.SAMPLING, 'thinking': False, 'stream': True}
    # This case has no optional tools/response_format. Preserve canonical omission
    # from the pinned engine; explicit JSON null is not equivalent to omission.
    if spec['tools'] is not None or spec['response_format'] is not None:
        raise ValueError('Unexpected optional branch')
    body = {'model': 'Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf', 'messages': spec['messages'],
            'max_tokens': spec['max_tokens'], 'stream': True, 'stream_options': {'include_usage': True},
            **v.SAMPLING, **modules['engines'].thinking_body('tensorsharp', False)}
    a.output.mkdir(parents=True, exist_ok=False)
    body_bytes = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    (a.output / 'request.json').write_bytes(body_bytes)
    info = {'status': 'new-input-awaiting-actual-token-count', 'release_qualified': False, 'tag': tag,
            'scenario': 'long_62k', 'input_sha256': v.digest(initial), 'body_sha256': hashlib.sha256(body_bytes).hexdigest(),
            'expected': {'ALPHA': 'silver-4821', 'BETA': 'amber-7392', 'GAMMA': 'violet-1605'},
            'original_long64_input_sha256': v.digest(original_input), 'original_long64_actual_prompt_tokens': 67154,
            'original_unchanged': True, 'required_hard_prompt_maximum': 65535,
            'requested_generation_tokens': spec['max_tokens'], 'extra_verifier_reserve': 4,
            'oracle': 'Exact complete JSON; same ALPHA/BETA/GAMMA values as original, no fences or truncated output accepted',
            'qsa_qualified': False, 'source_package_sha256': hashlib.sha256(raw).hexdigest()}
    (a.output / 'input.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'body_sha256': info['body_sha256'], 'status': info['status']}))


if __name__ == '__main__':
    main()
