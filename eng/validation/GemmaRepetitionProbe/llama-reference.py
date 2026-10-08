#!/usr/bin/env python3
"""Capture a same-token independent llama.cpp completion from an already running server.

Does not start/build a server or modify its settings. Schedule the server and
TensorSharp separately so their GPU allocations cannot compete.
"""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request


def post(base, endpoint, payload, timeout):
    request = urllib.request.Request(base.rstrip('/') + endpoint,
                                     json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-record', required=True, type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8087')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-new', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--timeout', type=float, default=600,
                        help='Total request deadline in seconds; socket calls use the remaining time.')
    args = parser.parse_args()
    if args.max_new <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--max-new and --timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    prompt = json.loads(args.prompt_record.read_text(encoding='utf-8-sig'))
    deadline = time.monotonic() + args.timeout

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Overall request deadline exceeded')
        return seconds
    # Independently render/tokenize as well as replaying identical final IDs.
    # Preserve a mismatch: silently choosing one rendering would conceal a BOS,
    # whitespace, thought-channel or tokenizer defect before any kernel runs.
    try:
        rendered = post(args.server, '/apply-template', {
            'messages': [{'role': 'user', 'content': prompt['Prompt']}],
            'add_generation_prompt': True,
            'chat_template_kwargs': {'enable_thinking': prompt['Thinking']}}, remaining())
        text = rendered['prompt']
        tokenized = post(args.server, '/tokenize', {'content': text, 'add_special': True, 'parse_special': True}, remaining())
        write('independent-prompt.json', {'rendered': rendered, 'tokenized': tokenized,
                                       'same_tokens': tokenized['tokens'] == prompt['PromptTokens']})
    except Exception as error:
        write('independent-prompt.json', {'error': repr(error), 'same_tokens': None})

    body = {'prompt': prompt['PromptTokens'], 'n_predict': args.max_new, 'temperature': args.temperature,
            'top_k': args.top_k, 'top_p': args.top_p, 'min_p': 0.0, 'repeat_penalty': 1.0,
            'repeat_last_n': 0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
            'dry_multiplier': 0.0, 'seed': args.seed, 'cache_prompt': False,
            'return_tokens': True, 'n_probs': 20, 'post_sampling_probs': False,
            'samplers': ['top_k', 'top_p', 'temperature'], 'stream': False}
    write('request.json', body)
    write('execution.json', {'status': 'requesting', 'timeout_seconds': args.timeout, 'complete': False})
    try:
        result = post(args.server, '/completion', body, remaining())
    except Exception as error:
        write('execution.json', {'status': 'request-failed', 'error': repr(error), 'complete': False,
                                 'qualification': 'No completed generation or quality pass.'})
        raise
    write('completion.json', result)
    if not result.get('tokens') or len(result['tokens']) != result.get('tokens_predicted'):
        raise RuntimeError('Missing/incomplete raw token evidence; inspect completion.json')
    (args.output / 'output.txt').write_text(result['content'], encoding='utf-8')
    write('execution.json', {'status': 'captured', 'complete': result.get('stop_type') == 'eos',
                             'stop_type': result.get('stop_type'), 'tokens_predicted': result.get('tokens_predicted'),
                             'quality_passed': None, 'qualification': 'Semantic review still required.'})
    write('qualification.json', {
        'qualification': 'Independent completion captured; exit zero alone is not a quality or numerical-parity pass.',
        'model_sha256_requested': prompt['ModelSha256'],
        'limitation': 'Caller must verify server model hash/build identity. Top-20 probabilities do not establish full-logit equivalence.',
        'teacher_replay': 'Pass completion.json to GemmaRepetitionProbe --mode direct --teacher.'})


if __name__ == '__main__':
    main()
