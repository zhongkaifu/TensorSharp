#!/usr/bin/env python3
"""Observe a new raw-token task with Maya; execution is NOT a quality pass.

A previously qualified plan verifies the executable, pack and checkpoint only.
Its reference output/oracle is never applied to this new request. Preserve full
wire IDs for a separate independent factual or serial/concurrent comparison.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('maya_observation_base', HERE/'maya-token-baseline.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
bench = base.bench


def prefill_rows(value):
    """Bound the diagnostic sweep independently of a historical task oracle."""
    if type(value) is not int or value not in (32, 64, 128, 256, 512):
        raise ValueError('Prefill rows must be one of32,64,128,256,512.')
    return value


def progress_record(line):
    match = re.fullmatch(r'PP ([0-9]+) ([0-9]+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?)', line)
    if not match:
        raise ValueError('Malformed prefill progress record.')
    done, total = map(int, match.group(1, 2))
    if not 0 < done <= total:
        raise ValueError('Invalid prefill progress counts.')
    elapsed, rate = map(float, match.group(3, 4))
    if not math.isfinite(elapsed) or not math.isfinite(rate):
        raise ValueError('Non-finite prefill progress interval/rate.')
    return dict(ComputedPromptTokens=done, TotalPromptTokens=total,
                EngineReportedCumulativeMs=elapsed, EngineReportedTokensPerSecond=rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--tokens', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-new', type=int, default=128)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--prefill', type=int, choices=(32, 64, 128, 256, 512), default=128)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_new <= 2048 or not 1 <= args.timeout <= 900:
        parser.error('Use bounded output and lifetime.')
    plan = json.loads(args.plan.read_text('utf-8-sig'))
    tokens = json.loads(args.tokens.read_text('utf-8-sig'))
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t < plan['vocab_size'] for t in tokens) \
            or len(tokens)+args.max_new > args.context:
        parser.error('Invalid exact raw IDs/context.')
    output = bench.evidence_directory(args.output); output.mkdir(parents=True, exist_ok=False)
    report = dict(ExecutionComplete=False, QualityPassed=None, Scope=__doc__, Tokens=tokens,
        ContextTokens=args.context, PrefillRows=prefill_rows(args.prefill), MaxNewTokens=args.max_new,
        BudgetEnvironment=plan['environment'],
        InputSha256=bench.sha(args.tokens), PlanSha256=bench.sha(args.plan), ToolSha256=bench.sha(__file__),
        RawOutputIds=[], RawTokenWallMs=[], PrefillProgress=[], Errors=[],
        TimingScope='Client wall starts immediately before sending GEN after READY. Raw first-token TTFT includes reasoning/control tokens; no published-content or equal-forward-work claim. Diagnostic flags, if present, change timing.')
    process = reader = watchdog = sampler = None
    timed_out = threading.Event(); lines = queue.Queue()
    def save():
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    save()
    try:
        for p in bench.psutil.process_iter(['pid','name','cmdline']):
            name=(p.info['name'] or '').lower(); command=' '.join(p.info['cmdline'] or []).lower()
            if any(n in name for n in ('llama-server','tensorsharp.server.host','strata.exe','strata-generate')) \
                    or name in ('dotnet','dotnet.exe') and any(n in command for n in ('glmunifiedsessionprobe.dll','qwenimagelogitprobe.dll')):
                raise RuntimeError('Existing model process is left untouched: '+str(p.info['pid']))
        _, files = base.verify_inputs(plan)
        report['VerifiedPackBefore'] = files
        tokenizer, _ = base.load_maya_text(plan['source_path'], plan['pack'])
        decoded = tokenizer.decode(tokens)
        if tokenizer.encode(decoded, parse_special=True) != tokens:
            raise ValueError('Independent Maya tokenizer does not round-trip the raw prompt.')
        report['DecodedPrompt'] = decoded
        command=[plan['exe'],'--glm-pack',plan['pack'],'--max-context',str(args.context),
                 '--prefill',str(args.prefill),'--serve']
        report['Command']=command
        started=time.monotonic(); deadline=started+args.timeout
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',env=base.cleaned_environment(plan['environment']),
            creationflags=bench.PROCESS_FLAGS)
        report['Pid']=process.pid
        def expire():
            if process.poll() is None:
                timed_out.set(); process.kill()
        watchdog=threading.Timer(args.timeout,expire); watchdog.start()
        def consume():
            try:
                with (output/'stdout.log').open('x',encoding='utf-8') as log:
                    for line in process.stdout:
                        log.write(line);log.flush();lines.put((time.monotonic(),line.rstrip('\r\n')))
            finally:lines.put((time.monotonic(),None))
        reader=threading.Thread(target=consume,daemon=True); reader.start()
        sampler=bench.Sampler(output/'telemetry.jsonl')
        sampler.process=bench.psutil.Process(process.pid)
        sampler.thread.start()
        def next_line():
            remaining=deadline-time.monotonic()
            if remaining <= 0 or timed_out.is_set():raise TimeoutError('Owned observation deadline')
            timestamp,value=lines.get(timeout=remaining)
            if value is None:raise RuntimeError('Process EOF before protocol completion')
            if value.startswith('ERR '):raise RuntimeError(value)
            return timestamp,value
        while not next_line()[1].startswith('READY '):pass
        actual=Path(bench.psutil.Process(process.pid).exe())
        if actual.resolve() != Path(plan['exe']).resolve() or bench.sha(actual) != plan['exe_sha256']:
            raise ValueError('Owned executable identity mismatch')
        report['ReadySeconds']=time.monotonic()-started
        request_started=time.monotonic()
        process.stdin.write('GEN '+str(args.max_new)+' temperature=0 '+','.join(map(str,tokens))+'\n')
        process.stdin.flush()
        while True:
            timestamp,line=next_line()
            if line.startswith('T '):
                parts=line.split()
                if len(parts)!=2:raise ValueError('Malformed token record')
                token=int(parts[1])
                if not 0 <= token < plan['vocab_size']:raise ValueError('Invalid output token')
                report['RawOutputIds'].append(token)
                report['RawTokenWallMs'].append((timestamp-request_started)*1000)
                if len(report['RawOutputIds'])>args.max_new:raise ValueError('Output exceeds bound')
            elif line.startswith('PP '):
                progress=progress_record(line)
                if progress['TotalPromptTokens']!=len(tokens):
                    raise ValueError('Prefill progress differs from raw input length.')
                progress['ClientWallMs']=(timestamp-request_started)*1000
                report['PrefillProgress'].append(progress)
            elif line.startswith('DONE '):
                report['Done']=base.parse_done(line)
                report['RequestWallMs']=(timestamp-request_started)*1000
                report['FirstRawTokenWallMs']=report['RawTokenWallMs'][0] if report['RawTokenWallMs'] else None
                break
        done=report['Done']
        if done['generated']!=len(report['RawOutputIds']) or done['prompt_tokens']!=len(tokens):
            raise ValueError('Wire usage differs from complete raw IDs')
        report['RawText']=tokenizer.decode(report['RawOutputIds'])
        # Native phase summaries are emitted by normal model teardown. Keep the
        # owned watchdog active and request QUIT rather than killing a completed
        # observation before its destructor has flushed those diagnostics.
        process.stdin.write('QUIT\n'); process.stdin.flush()
        process.wait(timeout=max(0.001, deadline-time.monotonic()))
        if process.returncode != 0 or time.monotonic() >= deadline or timed_out.is_set():
            raise RuntimeError('Owned observation did not complete normal bounded teardown.')
        report['ExecutionComplete']=True
    except Exception as error:
        report['Errors'].append(repr(error))
    finally:
        code, errors=base.cleanup_owned(process,watchdog,reader,sampler)
        report['ExitCode']=code;report['CleanupErrors']=errors;report['TimedOut']=timed_out.is_set()
        report['ExecutionComplete']=report['ExecutionComplete'] and not errors and not timed_out.is_set()
        if sampler is not None:
            report['SampledMemory']=bench.sampled_memory_summary(sampler.rows,sampler.errors)
            report['PerformanceObservationComplete']=report['ExecutionComplete'] and bool(sampler.rows) and not sampler.errors
            report['MemoryScope']='One-second process RSS/private and whole-board VRAM samples across process lifetime; peaks can be missed, board includes other apps, and process I/O counters are not physical SSD traffic.'
        try:
            if 'VerifiedPackBefore' in report:
                _, after=base.verify_inputs(plan)
                report['VerifiedPackAfter']=after
                if after!=report['VerifiedPackBefore']:raise ValueError('Verified pack bookends changed')
            if bench.sha(args.tokens)!=report['InputSha256'] or bench.sha(args.plan)!=report['PlanSha256'] \
                    or bench.sha(__file__)!=report['ToolSha256']:
                raise ValueError('Observation inputs/tool changed')
        except Exception as error:
            report['ExecutionComplete']=False;report['Errors'].append(repr(error))
        save()
    print(json.dumps({k:report[k] for k in ('ExecutionComplete','QualityPassed','Errors','CleanupErrors')}))
    return 0 if report['ExecutionComplete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
