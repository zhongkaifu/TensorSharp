#!/usr/bin/env python3
"""Prepare a fresh paired raw-token PP sweep; never load native code or a model.

The actual Host trace binds input IDs only. Limited outputs are observations, not
complete semantic quality passes. Requested widths must be checked at runtime:
admission can clamp/refuse a wide batch under the fixed resource ceilings.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve()
    if path.stat().st_size > 4 << 20:
        raise ValueError('Preparation only reads small evidence, never a model/native/executable.')
    raw = path.read_bytes()
    return dict(Path=str(path), Bytes=len(raw), Sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    pin(path)
    return json.loads(Path(path).read_text('utf-8-sig'))


def extract(text, request_id, expected_count):
    selected = []
    for number, line in enumerate(text.splitlines(), 1):
        if 'generation.validation.input ' not in line:
            continue
        if not re.search(r'\binternal='+re.escape(request_id)+r'\s', line):
            continue
        match = re.fullmatch(r'\s*generation\.validation\.input internal=(\S+) session=(\S+) '
            r'architecture=glm5-next compacted=False thinking=False maxNewTokens=20000 tokens=([0-9]+(?:,[0-9]+)*)\s*', line)
        if match and match.group(1) == request_id:
            words = match.group(3).split(',')
            tokens = [int(value) for value in words]
            if any(word != str(value) for word, value in zip(words, tokens)):
                raise ValueError('Actual trace IDs must be canonical decimal tokens.')
            if len(tokens) != expected_count or any(not 0 <= value < 154880 for value in tokens):
                raise ValueError('Actual input count/vocabulary differs from the explicit expectation.')
            selected.append(dict(Line=number, RequestId=request_id, SessionId=match.group(2),
                                 SourceLineUtf8Sha256=hashlib.sha256(line.encode('utf-8')).hexdigest(),
                                 SourceMaxNewTokens=20000, Tokens=tokens))
        else:
            raise ValueError('A targeted GLM input record is incomplete or has a different execution shape.')
    if len(selected) != 1:
        raise ValueError('Require exactly one complete, uncompacted, nonthinking GLM Host input trace for this request.')
    return selected[0]


def arguments(command):
    if not isinstance(command, list) or len(command) < 2 or (len(command)-2) % 2:
        raise ValueError('Invalid TS probe command.')
    values = {}
    for index in range(2, len(command), 2):
        key, value = command[index:index+2]
        if not isinstance(key, str) or not key.startswith('--') or key in values:
            raise ValueError('Require unique explicit probe arguments.')
        values[key] = value
    return values


def adaptive_options(options, enabled, sdk_supports_adaptive):
    """Legacy plans stay unchanged; new behavior needs explicit caller intent."""
    if type(enabled) is not bool or type(sdk_supports_adaptive) is not bool:
        raise ValueError('Adaptive planning policy requires explicit booleans.')
    result = dict(options)
    inherited = result.get('--adaptive-prefill')
    if inherited not in (None, 'true', 'false'):
        raise ValueError('Malformed inherited adaptive-prefill flag.')
    if not enabled:
        if inherited == 'true':
            raise ValueError('An adaptive template requires explicit --adaptive-prefill; do not enable by inheritance.')
        return result
    if result.get('--selected') != 'true' or not sdk_supports_adaptive:
        raise ValueError('Adaptive prefill requires selected=true and an explicitly supplied newer probe SDK.')
    result['--adaptive-prefill'] = 'true'
    return result


def adaptive_sdk_evidence(probe):
    """Bounded static CLI/setter evidence, not execution or native capability."""
    probe = Path(probe)
    if probe.name != 'GlmUnifiedSessionProbe.dll' or not 0 < probe.stat().st_size <= 4 << 20:
        raise ValueError('Require the caller-selected bounded actual GlmUnifiedSessionProbe DLL.')
    raw = probe.read_bytes()
    supported = ('adaptive-prefill'.encode('utf-16-le') in raw
                 and b'set_EnableAdaptiveGlmPrefill' in raw)
    return dict(ProbePath=str(probe.resolve()), ProbeBytes=len(raw),
                CliAndSessionSetterPresent=supported,
                WideAdaptiveSessionSetterPresent=b'set_AdaptiveGlmPrefillChunkTokens' in raw,
                Scope='Static literal/member presence in the explicit SDK only; no model, native or probe execution. Runtime report must confirm applied policy and actual work.')


def require_wide_adaptive_policy(widths, enabled, evidence):
    if any(width > 512 for width in widths) and (enabled is not True
            or evidence.get('CliAndSessionSetterPresent') is not True
            or evidence.get('WideAdaptiveSessionSetterPresent') is not True):
        raise ValueError('Widths above512 require explicit adaptive policy and the newer wide-admission probe SDK.')


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-log', 'request-id', 'ts-template', 'maya-template', 'ts-app', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--expected-tokens', type=int, default=1340)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--max-new', type=int, default=32)
    parser.add_argument('--widths', type=int, nargs='+', choices=(128, 256, 512, 1024), default=(128, 256, 512))
    parser.add_argument('--repetitions', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timing', action='store_true', help='Diagnostic only; keep quiet measurements separate.')
    parser.add_argument('--native-sha256', help='Explicit actual candidate native SHA; the template SHA is the historical default.')
    parser.add_argument('--adaptive-prefill', action='store_true',
                        help='Explicit newer probe SDK required; requested --prefill is a ceiling, not proof of actual batch work.')
    args = parser.parse_args()
    if not 1 <= args.expected_tokens <= 32768 or not 1 <= args.max_new <= 128 \
            or args.expected_tokens+args.max_new > args.context or not 1 <= args.context <= 32768 \
            or len(set(args.widths)) != len(args.widths):
        parser.error('Require bounded distinct shapes and actual request context capacity.')
    source = pin(args.source_log)
    bound = extract(Path(args.source_log).read_text('utf-8-sig'), args.request_id, args.expected_tokens)
    ts, maya = read(args.ts_template), read(args.maya_template)
    options = arguments(ts['Command'])
    if args.native_sha256 is not None:
        if not re.fullmatch(r'[0-9a-fA-F]{64}', args.native_sha256) or set(args.native_sha256) == {'0'}:
            raise ValueError('Require actual candidate native SHA, never a placeholder.')
        options['--native-sha256'] = args.native_sha256.lower()
    if options.get('--task') != 'unknown' or options.get('--selected') != 'true' \
            or options.get('--cache-token-snapshots') != 'false':
        raise ValueError('Require the original selected-expert non-gated unknown-task probe template.')
    app = Path(args.ts_app).resolve()
    probe = app / Path(ts['Command'][1]).name
    if not probe.is_file() or not (app/'GgmlOps.dll').is_file():
        raise ValueError('Require an already staged coherent private probe/native application.')
    adaptive_evidence = adaptive_sdk_evidence(probe) if args.adaptive_prefill else None
    require_wide_adaptive_policy(args.widths, args.adaptive_prefill, adaptive_evidence or {})
    options = adaptive_options(options, args.adaptive_prefill,
                               bool(adaptive_evidence and adaptive_evidence['CliAndSessionSetterPresent']))
    out = Path(args.output).absolute()
    if out.exists() or '..' in Path(args.output).parts:
        raise ValueError('Require a fresh explicit output directory without traversal.')
    out.mkdir(parents=True)
    with (out/'host-prompt.ids').open('x', encoding='utf-8') as stream:
        stream.write(','.join(map(str, bound['Tokens']))+'\n')
    write(out/'host-prompt.tokens.json', bound['Tokens'])
    receipt = dict(Schema='glm-actual-host-input-binding.v1', Source=source,
                   **{key: value for key, value in bound.items() if key != 'Tokens'},
                   TokenCount=len(bound['Tokens']), Csv=pin(out/'host-prompt.ids'),
                   Json=pin(out/'host-prompt.tokens.json'),
                   Scope='Original actual user prompt plus agent/tool catalog IDs; input binding only. Source request was intentionally cancelled, so its output is not a complete factual oracle.')
    write(out/'input-receipt.json', receipt)
    manifest = dict(Schema='glm-raw-prefill-paired-profile-plan.v1', ExecutionStatus='not-run', QualityPassed=None,
                    InputReceipt=pin(out/'input-receipt.json'), PreparationTool=pin(__file__),
                    TsTemplate=pin(args.ts_template), MayaTemplate=pin(args.maya_template),
                    ContextTokens=args.context, MaxNewTokens=args.max_new, Instrumented=args.timing,
                    AdaptivePrefill=args.adaptive_prefill, AdaptiveSdkEvidence=adaptive_evidence,
                    RequestedPrefillScope='Requested token ceiling only. Admission/scheduler caps and actual forward/chunk evidence must be recorded separately; prepared caps are not actual GPU batch work.',
                    Cases=[], Limits=['Fresh process per arm; two repeats use T-M-M-T at each requested width. OS file cache is uncontrolled.',
                                      'Original actual Host maxNew20000 is bounded here to limited output, retaining exactly its original raw input IDs.',
                                      'No complete factual answer or natural EOS quality acceptance is inferred from bounded observations.',
                                      'Requested width is not runtime admission proof. Record actual chunks, refusals, TTFT, attributed PP and all raw output IDs.',
                                      'Historical identity templates certify checkpoint/executable provenance only; old task output/oracle never grades the new input.',
                                      'Payload ceiling versus sampled process/board memory is not equivalent physical hard-budget ownership.',
                                      'TS forwards P+O including EOS whereas Maya forwards P+O-1; raw output counts/IDs must be checked before normalization.',
                                      'Opt-in timing changes execution; keep diagnostic and quiet performance evidence separate.'])
    for width in args.widths:
        sequence = [('tensorsharp','t1'), ('maya','m1')]
        if args.repetitions == 2:
            sequence += [('maya','m2'), ('tensorsharp','t2')]
        for engine, label in sequence:
            case = out / f'pp{width}-{label}'
            case.mkdir()
            if engine == 'tensorsharp':
                selected = dict(options)
                selected.update({'--output':str(case/'run'), '--tokens-file':str(out/'host-prompt.ids'),
                                 '--context':str(args.context), '--max-new':str(args.max_new),
                                 '--prefill':str(width), '--repetitions':'1'})
                command = [ts['Command'][0], str(probe)] + [part for pair in selected.items() for part in pair]
                env = dict(CUDA_VISIBLE_DEVICES='0', CUDA_MODULE_LOADING='EAGER',
                           TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1')
                if args.timing:
                    env['TS_HOST_MOE_EXPERT_CACHE_TIMING'] = '1'
                plan = dict(Command=command, Environment=env, ManagedDirectory=str(app),
                            NativeSha256=options['--native-sha256'], Output=str(case/'run'),
                            WrapperOutput=str(case/'wrapper'), ExpectedUnknownTaskExit=1,
                            SemanticPassed=None, AdaptivePrefill=args.adaptive_prefill,
                            RequestedPrefillCap=width,
                            ActualPrefillWorkStatus='not-run; require applied policy plus real chunk/forward/step evidence')
                plan_path = case/'ts-plan.json'
            else:
                plan = copy.deepcopy(maya)
                plan['environment'].pop('STRATA_PREFILL_TIMING', None)
                plan['environment'].pop('STRATA_GLM_PREFILL_PROF', None)
                if args.timing:
                    plan['environment']['STRATA_GLM_PREFILL_PROF'] = '1'
                plan['raw_observation_scope'] = 'Historical task identity/oracle is not the new request quality oracle.'
                plan_path = case/'maya-identity-plan.json'
            write(plan_path, plan)
            invocation = (['run-bounded-probe.py', '--output', plan['WrapperOutput'], '--timeout', '900',
                           '--telemetry', '--managed-directory', str(app), '--expected-native-sha256',
                           plan['NativeSha256'], '--']+plan['Command']) if engine == 'tensorsharp' else \
                ['maya-token-observation.py', '--plan', str(plan_path), '--tokens', str(out/'host-prompt.tokens.json'),
                 '--prefill', str(width), '--context', str(args.context), '--max-new', str(args.max_new),
                 '--timeout', '900', '--output', str(case/'run')]
            manifest['Cases'].append(dict(Engine=engine, Label=label, RequestedPrefillRows=width,
                                           Plan=pin(plan_path), OwnedWrapperArguments=invocation))
    write(out/'matrix-plan.json', manifest)
    print(json.dumps(dict(Prepared=True, ExecutionStatus='not-run', QualityPassed=None,
                         SourceLine=bound['Line'], Tokens=len(bound['Tokens']), Processes=len(manifest['Cases']),
                         Output=str(out))))


if __name__ == '__main__':
    main()
