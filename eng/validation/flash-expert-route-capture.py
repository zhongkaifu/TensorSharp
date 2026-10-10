#!/usr/bin/env python3
"""Capture a diagnostic Flash routing trace; do not use its timings as throughput."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-identity-report', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expert-cache-mb', type=int, default=9472)
    parser.add_argument('--iterations', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to(root / p) for p in ('artifacts', 'docs/validation')):
        parser.error('Output must stay in an ignored validation directory')
    if args.expert_cache_mb <= 0 or args.iterations <= 0:
        parser.error('Cache and iterations must be positive')
    output.mkdir(parents=True, exist_ok=False)
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'NCCL_'))}
    settings = dict(CUDA_VISIBLE_DEVICES='0', TS_CPU_MOE='1', TS_CPU_MOE_THREADS='8',
        TS_GGML_CPU_THREADS='8', OMP_NUM_THREADS='8', TS_HOST_MOE_EXPERT_CACHE_MB=str(args.expert_cache_mb),
        TS_HOST_MOE_EXPERT_CACHE_LAYERS='48', TS_HOST_MOE_PIN='0', TS_HOST_MOE_FILE_READ='1',
        TS_HOST_MOE_DEVICE_MIN_BATCH='0', TS_HOST_MOE_ROUTE_TRACE='1', TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1',
        TS_HOST_MOE_EXPERT_CACHE_LFU='0')
    environment.update(settings)
    command = ['dotnet', str(args.binary_dir.resolve()/'Qwen4ExpExpertCacheProbe.dll'), '--model', str(args.model.resolve()),
        '--model-identity-report', str(args.model_identity_report.resolve()), '--placement', 'host', '--backend', 'ggml_cuda',
        '--generation', 'greedy', '--prompt', args.prompt, '--decode-tokens', '256', '--max-context', '512',
        '--iterations', str(args.iterations), '--warmup', '1', '--output', str(output/'model.json')]
    with (output/'process.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    (output/'execution.json').write_text(json.dumps(dict(command=command, environment=settings,
        exit_code=completed.returncode, diagnostic_only=True,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n', encoding='utf-8')
    completed.check_returncode()
    subprocess.run([sys.executable, str(Path(__file__).with_name('flash-expert-cache-replay.py')),
        '--log', str(output/'process.log'), '--probe', str(output/'model.json'), '--output', str(output/'replay.json')], check=True)


if __name__ == '__main__':
    main()
