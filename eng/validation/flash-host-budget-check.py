#!/usr/bin/env python3
"""Exercise shared expert staging admission and release on a real Flash model.

The positive run releases device residency and trims the expert cache between
complete requests. The negative run must fail before allocating any host arena.
These are lifecycle/quality-regression checks, not warm throughput measurements.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-identity-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=('all', 'admitted', 'refused'), default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to(root/p) for p in ('artifacts','docs/validation')):
        parser.error('Output must stay in an ignored validation directory')
    output.mkdir(parents=True, exist_ok=False)
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('TS_','TENSORSHARP_','GGML_','NCCL_'))}
    settings = dict(CUDA_VISIBLE_DEVICES='0', TS_CPU_MOE='1', TS_CPU_MOE_THREADS='8',
        TS_GGML_CPU_THREADS='8', OMP_NUM_THREADS='8', TS_HOST_MOE_EXPERT_CACHE_MB='9472',
        TS_HOST_MOE_EXPERT_CACHE_LAYERS='48', TS_HOST_MOE_PIN='0', TS_HOST_MOE_DEVICE_MIN_BATCH='0',
        TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1')
    environment.update(settings) # Leave file source selection and LFU at their defaults.
    prompt = '用中文两段介绍《最终幻想 VII》的故事背景与主要角色，准确区分米德加、克劳德、蒂法和爱丽丝，控制在180字以内。不要使用列表。'
    result = dict(passed=False, runs={}, limitations=['Host quota covers file arena payload only; no total process RAM cap.',
        'Release/refill timings are not warm throughput. Full logits equality is not broad semantic quality.'])
    failures = []
    for name, cap in (('admitted', 32 << 20), ('refused', 1)):
        if args.scenario != 'all' and args.scenario != name:
            continue
        directory = output/name
        directory.mkdir()
        command = ['dotnet', str(args.binary_dir.resolve()/'Qwen4ExpExpertCacheProbe.dll'),
            '--model', str(args.model.resolve()), '--model-identity-report', str(args.model_identity_report.resolve()),
            '--placement','host','--backend','ggml_cuda','--max-context','512',
            '--device-budget-bytes', str(15 << 30), '--host-budget-bytes',str(cap), '--output',str(directory/'model.json')]
        command += ['--generation','greedy','--prompt',prompt,'--decode-tokens','256','--warmup','1','--iterations','2',
                    '--release-residency','true','--trim-target-bytes','0'] if name == 'admitted' else [
                    '--generation','teacher-forced','--prefill-tokens','4','--decode-tokens','2','--warmup','0','--iterations','1']
        with (directory/'process.log').open('w',encoding='utf-8') as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        execution = dict(command=command, environment=settings, exit_code=completed.returncode)
        (directory/'execution.json').write_text(json.dumps(execution,indent=2)+'\n',encoding='utf-8')
        model = json.loads((directory/'model.json').read_text(encoding='utf-8-sig'))
        cleanup = model['cleanup']
        if any(not cleanup.get(k) for k in ('model_disposed','cache_cleared','reuse_released','scope_detached','native_shutdown')) or cleanup.get('errors'):
            failures.append(name+': physical/budget cleanup failed')
        observations = model['budget_observations']
        host = [p for o in observations for p in o['pools'] if p['Pool']=='host']
        if not host or any(p['Committed']+p['Reserved'] > cap for p in host):
            failures.append(name+': missing host observations or quota exceeded')
        final = next((o for o in observations if o['stage']=='after-physical-cleanup-before-detach'),None)
        if not final or any(p['Reserved'] or p['Committed'] for p in final['pools']):
            failures.append(name+': shared credit remains after physical cleanup')
        if name == 'admitted':
            if completed.returncode != 0 or not model['passed'] or not model['run_complete']:
                failures.append(name+': expected successful complete model execution')
            chains = {r['full_logit_chain_sha256'] for r in model['runs']}
            expected = 'ac1cdce699bc35bf3606d03f56de3e530ca9e3c0963bb19b312ce59664693e65'
            if len(model['runs']) != 3 or chains != {expected} or any(r['finish_reason']!='eos' for r in model['runs']):
                failures.append(name+': complete vocabulary history or EOS changed')
            for r in model['runs']:
                if r['cache_stats_after_decode']['Calls']-r['cache_stats_after_prefill']['Calls'] != r['decode_tokens']*model['model_geometry']['layers']:
                    failures.append(name+': decode did not fully use the expert cache')
            trims = [o for o in observations if o['stage'].endswith('after-expert-trim')]
            if len(trims)!=2 or any(p['Committed'] or p['Reserved'] for o in trims for p in o['pools'] if p['Pool']=='host'):
                failures.append(name+': trim retained the host arena')
            if not any(p['Committed'] > 0 for p in host):
                failures.append(name+': host arena did not acquire shared credit')
            log = (directory/'process.log').read_text(encoding='utf-8')
            if log.count('[HOSTMOE-FILE]') < 3 or 'policy=lru' in log or 'policy=lfu' not in log:
                failures.append(name+': default file/LFU selection was not observed after reload')
        else:
            if completed.returncode == 0 or model['passed'] or 'Expert read workspace exceeds shared host budget' not in str(model.get('error')):
                failures.append(name+': expected host admission failure did not occur')
            if any(p['Committed'] or p['Reserved'] for p in host):
                failures.append(name+': refused arena consumed host credit')
        result['runs'][name] = dict(execution=execution, model=model)
    result['failures'] = failures
    result['passed'] = not failures
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],failures=failures)))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
