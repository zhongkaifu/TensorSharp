#!/usr/bin/env python3
"""Run isolated, serial ABBA Flash expert-cache measurements.

Both directories must contain the same built Qwen4ExpExpertCacheProbe and
managed dependencies, with their respective GgmlOps native library. Control
disables demand prefetch; candidate uses the default feedback policy. Evidence
belongs under ignored artifacts/ or docs/validation/. No OS cache is flushed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-identity-report', type=Path, required=True)
    parser.add_argument('--control-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation', choices=('teacher-forced', 'greedy'), default='teacher-forced')
    parser.add_argument('--prompt', default='What is 17 plus 25? Answer with the number only.')
    parser.add_argument('--expected-output', help='Optional exact greedy answer after trimming surrounding whitespace; requires EOS.')
    parser.add_argument('--decode-tokens', type=int, default=32)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--expert-cache-mb', type=int, default=2048)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--candidate-first', action='store_true', help='Reverse the balanced order to candidate/control/control/candidate')
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--candidate-file-read', action='store_true', help='Compare mmap against exact file staging, keeping adaptive prefetch the same')
    experiment.add_argument('--candidate-lfu', action='store_true', help='Compare decaying-frequency eviction against LRU, both with file staging')
    args = parser.parse_args()
    if args.iterations < 2 or min(args.decode_tokens, args.max_context, args.expert_cache_mb, args.cpu_threads) < 1:
        parser.error('Require at least two measurements and positive token, cache and thread limits')
    if args.expected_output is not None and args.generation != 'greedy':
        parser.error('--expected-output requires --generation greedy')
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    output = args.output.resolve()
    if not any(output.is_relative_to(p) for p in (root / 'artifacts', root / 'docs/validation')):
        parser.error('Write generated evidence under ignored artifacts/ or docs/validation/')
    output.mkdir(parents=True, exist_ok=False)
    bench = load_module('process_bench', here / 'qwen-image21-bench.py')
    comparator = load_module('flash_compare', here / 'compare-flash-decode-runs.py')
    executions = []
    semantic_failures = []
    order = ('candidate','control','control','candidate') if args.candidate_first else ('control','candidate','candidate','control')
    for index, arm in enumerate(order):
        directory = output / f'{index}-{arm}'
        directory.mkdir()
        binary = args.control_dir if arm == 'control' else args.candidate_dir
        command = ['dotnet', str(binary.resolve() / 'Qwen4ExpExpertCacheProbe.dll'),
                   '--model', str(args.model.resolve()), '--model-identity-report', str(args.model_identity_report.resolve()),
                   '--placement', 'host', '--backend', 'ggml_cuda', '--generation', args.generation,
                   '--prompt', args.prompt, '--decode-tokens', str(args.decode_tokens),
                   '--max-context', str(args.max_context), '--iterations', str(args.iterations), '--warmup', '1',
                   '--output', str(directory / 'model.json')]
        environment = {k:v for k,v in os.environ.items() if not k.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'NCCL_'))}
        settings = dict(CUDA_VISIBLE_DEVICES=args.device, TS_CPU_MOE='1',
                        TS_CPU_MOE_THREADS=str(args.cpu_threads), TS_GGML_CPU_THREADS=str(args.cpu_threads),
                        OMP_NUM_THREADS=str(args.cpu_threads), TS_HOST_MOE_EXPERT_CACHE_MB=str(args.expert_cache_mb),
                        TS_HOST_MOE_EXPERT_CACHE_LAYERS='48', TS_HOST_MOE_PIN='0', TS_HOST_MOE_TIMING='0',
                        TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1', TS_HOST_MOE_DEVICE_MIN_BATCH='0',
                        TS_HOST_MOE_FILE_READ='0', TS_HOST_MOE_EXPERT_CACHE_LFU='0')
        if args.candidate_lfu:
            settings['TS_HOST_MOE_FILE_READ'] = '1'
            settings['TS_HOST_MOE_EXPERT_CACHE_LFU'] = '1' if arm == 'candidate' else '0'
        elif args.candidate_file_read:
            settings['TS_HOST_MOE_FILE_READ'] = '1' if arm == 'candidate' else '0'
        elif arm == 'control':
            settings['TS_HOST_MOE_EXPERT_CACHE_PREFETCH'] = '0'
        environment.update(settings)
        execution = dict(started_unix=time.time(), command=command, environment=settings, arm=arm,
                         scope='Serial ABBA; process-first request separated from repeated warm requests; OS pages uncontrolled.')
        path = directory / 'execution.json'
        path.write_text(json.dumps(execution, indent=2)+'\n', encoding='utf-8')
        print(f'Start {index}: {arm}', flush=True)
        execution.update(bench.run_process(command, directory / 'process.log', args.timeout, environment, True, 1.0))
        path.write_text(json.dumps(execution, indent=2)+'\n', encoding='utf-8')
        executions.append(path)
        if execution.get('exit_code') != 0 or execution.get('timed_out') is not False:
            raise RuntimeError(f'{arm} failed; inspect {path}')
        model = json.loads((directory / 'model.json').read_text(encoding='utf-8-sig'))
        print(json.dumps(dict(arm=arm, runs=[{k:r[k] for k in ('warmup','prefill_tps','decode_tps','finish_reason')} for r in model['runs']])), flush=True)
        if args.expected_output is not None:
            for i, row in enumerate(model['runs']):
                if row.get('finish_reason') != 'eos' or row.get('selected_text', '').strip() != args.expected_output.strip():
                    semantic_failures.append(f'Execution {index}, request {i}: expected answer/EOS mismatch')
    result = comparator.compare(executions, args.candidate_first, args.candidate_file_read, args.candidate_lfu)
    result['expected_output_checked'] = args.expected_output
    result['failures'].extend(semantic_failures)
    result['passed'] = not result['failures']
    (output / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
