#!/usr/bin/env python3
"""Prepare/run quiet, independently qualified T-old/T-new/T-new/T-old probes.

Only run mode launches an existing directly owned bounded wrapper. Full raw IDs,
natural EOS, exact admission width, runtime identities, cleanup and independent
task semantics are required. Payload ceilings are not physical hard budgets.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import types


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUIET = dict(CUDA_VISIBLE_DEVICES='0', CUDA_MODULE_LOADING='EAGER',
             TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='0', TS_HOST_MOE_EXPERT_CACHE_TIMING='0')
DIRECT_READ_KEY = 'TS_HOST_MOE_EXPERT_CACHE_DIRECT_READS'


def arm_environment(role, direct_candidate=False):
    if role not in ('baseline', 'candidate') or type(direct_candidate) is not bool:
        raise ValueError('Require an exact comparison role and explicit Boolean experiment.')
    environment = dict(QUIET)
    if direct_candidate and role == 'candidate':
        environment[DIRECT_READ_KEY] = '1'
    return environment


def direct_environment_verified(plan, report):
    expected = plan.get('Environment', {}).get(DIRECT_READ_KEY)
    return all(report.get(field, {}).get(DIRECT_READ_KEY) == expected
               for field in ('EnvironmentBefore', 'EnvironmentEffective', 'EnvironmentAfter'))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def pin(path):
    path = Path(path).resolve()
    if path.stat().st_size > 4 << 20:
        raise ValueError('Expected small evidence/source file; no model/DLL hashing here.')
    raw = path.read_bytes()
    return dict(Path=str(path), Bytes=len(raw), Sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    pin(path)
    return json.loads(Path(path).read_text('utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON: '+value)))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', 'utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean_environment(parent):
    result = {key: value for key, value in parent.items()
              if not key.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'GLM_', 'STRATA_', 'OMP_', 'GOMP_', 'NCCL_'))
              and key not in ('MAX_CONTEXT', 'MAX_TOKENS', 'KV_CACHE_DTYPE', 'CUDA_MPS_PIPE_DIRECTORY')}
    result.update(QUIET)
    return result


def comparison(cases):
    results = []
    for task in sorted({row['Task'] for row in cases}):
        rows = [row for row in cases if row['Task'] == task]
        complete = len(rows) == 4 and all(row.get('Passed') is True for row in rows)
        same = complete and all(row['RawOutputTokens'] == rows[0]['RawOutputTokens'] for row in rows)
        prompt = complete and all(row['RawPromptTokens'] == rows[0]['RawPromptTokens'] for row in rows)
        results.append(dict(Task=task, CompletedAndIndependentlyQualified=complete,
                            SameRawPromptIds=prompt if complete else None,
                            SameRawOutputIds=same if complete else None,
                            EqualForwardWork=bool(same and prompt) if complete else None,
                            Scope='Same public TS probe convention P+O including EOS. Two repeats are balanced observations, not statistical significance; different managed/native packages prevent native-only attribution.'))
    return results


def evidence_gate(plan, report, wrapper, qualified, wrapper_exit, checker_exit):
    timing = report.get('Timing', {})
    output, prompt = report.get('OutputTokens'), report.get('PromptTokens')
    counts = (isinstance(output, list) and bool(output) and isinstance(prompt, list)
              and prompt == plan['Binding']['PromptTokens']
              and timing.get('EngineOutputTokens') == timing.get('OutputForwardCount') == len(output)
              and report.get('ComputedTokens') == len(prompt)+len(output))
    telemetry = wrapper.get('telemetry', {})
    passed = (wrapper_exit == wrapper.get('exit_code') == plan['ExpectedProbeUnknownExit'] == 1
              and checker_exit == 0 and qualified.get('Passed') is True and counts
              and wrapper.get('complete') is True and wrapper.get('timed_out') is False
              and wrapper.get('cleanup_errors') == [] and telemetry.get('passed') is True
              and telemetry.get('errors') == [] and report.get('Errors') == []
              and report.get('LedgerZeroAfterCleanup') is True and report.get('NaturalEos') is True
              and direct_environment_verified(plan, report))
    return dict(Passed=bool(passed), CountsValid=bool(counts), RawPromptTokens=prompt,
                RawOutputTokens=output, NaturalEos=report.get('NaturalEos'),
                DirectReadEnvironmentVerified=direct_environment_verified(plan, report),
                IndependentSemantic=qualified.get('IndependentSemantic'),
                Performance={key: timing.get(key) for key in ('PrefillAttributedMs', 'DecodeAttributedMs',
                    'FirstPublishedTokenWallMs', 'RequestWallMs', 'EngineOutputTokens', 'OutputForwardCount')},
                ObservedMemory=telemetry.get('memory'))


def validate_manifest(manifest, qualifier):
    tasks = [('tool_json',128)] + ([('extraction',64)] if manifest['IncludeExtraction'] else [])
    expected = [(task, width, role, label) for task,width in tasks
                for role,label in [('baseline','b1'),('candidate','c1'),('candidate','c2'),('baseline','b2')]]
    actual = [(row['Task'],row['PrefillRows'],row['Role'],row['Label']) for row in manifest['Cases']]
    if actual != expected:
        raise ValueError('Require every exact independent balanced arm in its original order.')
    inputs = {}
    for row in manifest['Cases']:
        if pin(row['Plan']['Path']) != row['Plan']:
            raise ValueError('Prepared plan changed.')
        plan = read(row['Plan']['Path']); qualifier.verify_plan_files(plan)
        bound = manifest['Deployments'][row['Role']]
        if plan['NativeVariant'] != bound['Variant'] or plan['Native']['Sha256'] != bound['Sha256'] \
                or plan['AppDirectory'] != bound['Directory'] \
                or plan['Environment'] != arm_environment(row['Role'], manifest.get('CandidateDirectReads', False)) \
                or plan['Case'] != row['Task'] or plan.get('BenchmarkPrefillRows',32) != row['PrefillRows']:
            raise ValueError('Variant, app, task, exact width or quiet environment differs.')
        qualifier.controls(plan)
        tokens = plan['Binding']['PromptTokens']
        if len(tokens) != {'tool_json':85,'extraction':52}[row['Task']]:
            raise ValueError('Original real task input count differs.')
        if row['Task'] in inputs and inputs[row['Task']] != tokens:
            raise ValueError('Candidate does not consume identical original raw input IDs.')
        inputs[row['Task']] = tokens


def prepare(args):
    qualifier = module(HERE/'glm-unified-task-qualification.py','candidate_comparison_qualifier_prepare')
    out = Path(args.output).absolute()
    if out.exists() or '..' in Path(args.output).parts:
        raise ValueError('Fresh explicit preparation output required.')
    out.mkdir(parents=True)
    deployments = {}
    for role in ('baseline','candidate'):
        app, receipt, variant, expected = [getattr(args,role+'_'+field) for field in ('app','receipt','variant','sha256')]
        if not re.fullmatch('[0-9a-fA-F]{64}', expected) or set(expected) == {'0'}:
            raise ValueError('Require actual candidate/baseline native SHA, never a placeholder.')
        _, native, _, _ = qualifier.deployment(read(receipt),str(Path(app).resolve()),variant)
        if native['Sha256'].lower() != expected.lower():
            raise ValueError('Explicit native SHA differs from immutable deployment receipt.')
        deployments[role] = dict(Directory=str(Path(app).resolve()), Variant=variant,
                                Sha256=native['Sha256'], Receipt=pin(receipt))
    manifest = dict(Schema='glm-unified-candidate-comparison-plan.v1',ExecutionStatus='not-run',
                    IncludeExtraction=args.include_extraction,Deployments=deployments,Cases=[],
                    CandidateDirectReads=getattr(args, 'candidate_direct_reads', False),
                    PreparationTool=pin(__file__),Qualifier=pin(qualifier.__file__),
                    Scope='Quiet fresh N1 processes: tool85 PP128; optional extraction52 PP64. ctx1024/cpu8/temp0/max128/checkpointEOS; shared host12GiB/device14GiB/expert7000MiB payload ceilings. No physical-hard-budget, stockHost N3/32K or Maya parity claim.')
    if manifest['CandidateDirectReads']:
        manifest['Scope'] += ' Candidate-only DIRECT_READS=1 is an explicit experimental read-policy comparison, not default-on qualification.'
    for task,width in [('tool_json',128)]+([('extraction',64)] if args.include_extraction else []):
        for role,label in [('baseline','b1'),('candidate','c1'),('candidate','c2'),('baseline','b2')]:
            case = out/f'{task}-pp{width}-{label}'
            bound = deployments[role]
            qualifier.prepare(types.SimpleNamespace(app=bound['Directory'],variant=bound['Variant'],
                deployment_receipt=bound['Receipt']['Path'],output=str(case),prefill=width,
                source_directory=args.source_directory,model=args.model,model_identity=args.model_identity,
                relocation_receipt=args.relocation_receipt,relocation_source_manifest=args.relocation_source_manifest))
            plan_path = case/(task+'-plan.json')
            plan = read(plan_path); plan['Environment'] = arm_environment(role, manifest['CandidateDirectReads'])
            save(plan_path,plan)
            manifest['Cases'].append(dict(Task=task,PrefillRows=width,Role=role,Label=label,Plan=pin(plan_path)))
    validate_manifest(manifest,qualifier)
    first = {role:read(next(row['Plan']['Path'] for row in manifest['Cases'] if row['Role']==role))
             for role in ('baseline','candidate')}
    old,new = [first[role]['Managed'] for role in ('baseline','candidate')]
    manifest['ManagedPackageDifferences'] = [name for name in sorted(old.keys()|new.keys()) if old.get(name)!=new.get(name)]
    manifest['ManagedPackageScope'] = 'Different managed SDK identities mean a complete candidate package comparison, not a native-only causal experiment.'
    save(out/'matrix-plan.json',manifest)
    print(json.dumps(dict(Prepared=True,Processes=len(manifest['Cases']),ExecutionStatus='not-run',Output=str(out))))


def require_exclusive_processes():
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        name = (process.info['name'] or '').lower()
        cmd = ' '.join(process.info['cmdline'] or []).lower()
        if any(value in name for value in ('tensorsharp.server.host','llama-server','strata.exe','strata-generate')) \
                or name in ('dotnet','dotnet.exe') and any(value in cmd for value in
                    ('glmunifiedsessionprobe.dll','qwenimagelogitprobe.dll','qwenvisionembeddingprobe.dll')):
            raise RuntimeError('Existing model process left untouched: '+str(process.info['pid']))


def owned(script, argv, environment, log):
    old_args,old_env = sys.argv[:],dict(os.environ)
    try:
        sys.argv=[str(script)]+argv;os.environ.clear();os.environ.update(environment)
        with log.open('x',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            return runpy.run_path(str(script),run_name='glm_comparison_owned_wrapper')['main']()
    finally:
        sys.argv=old_args;os.environ.clear();os.environ.update(old_env)


def run(args):
    qualifier=module(HERE/'glm-unified-task-qualification.py','candidate_comparison_qualifier_run')
    manifest=read(args.matrix);validate_manifest(manifest,qualifier)
    if pin(qualifier.__file__)!=manifest['Qualifier'] or pin(__file__)!=manifest['PreparationTool']:
        raise ValueError('Frozen preparer/qualifier source changed.')
    base=Path(args.matrix).parent;output=base/'matrix-execution.json'
    if output.exists():raise ValueError('Never overwrite or silently resume an execution.')
    helpers={str(path):pin(path) for path in [HERE/name for name in
        ('run-bounded-probe.py','glm-unified-task-qualification.py','fixed-task-semantic.py')]}
    environment=clean_environment(os.environ)
    state=dict(Status='running',StartedUtc=utc(),Matrix=pin(args.matrix),Runner=pin(__file__),
               Helpers=helpers,ChildEnvironmentOverrides=dict(QUIET),Cases=[],Scope=manifest['Scope'],
               ManagedPackageDifferences=manifest['ManagedPackageDifferences'])
    current=None
    def flush():
        state['Comparability']=comparison(state['Cases']);save(output,state)
    flush()
    try:
        for row in manifest['Cases']:
            if (base/'stop-before-next-step').exists():
                state['Status']='stopped-between-cases';flush();return 3
            if pin(args.matrix)!=state['Matrix'] or pin(__file__)!=state['Runner'] \
                    or any(pin(path)!=bound for path,bound in helpers.items()):
                raise ValueError('Frozen matrix/runner/helper changed between arms.')
            require_exclusive_processes()
            if pin(row['Plan']['Path'])!=row['Plan']:raise ValueError('Frozen individual plan changed.')
            plan=read(row['Plan']['Path']);qualifier.verify_plan_files(plan)
            parent=Path(row['Plan']['Path']).parent
            if any(Path(plan[key]).exists() for key in ('WrapperOutput','ReportOutput')):
                raise ValueError('Fresh process output required; never resume an old partial run.')
            current=dict(row,Status='running',StartedUtc=utc());state['Cases'].append(current);flush()
            argv=['--output',plan['WrapperOutput'],'--timeout','900','--telemetry','--managed-directory',
                  plan['AppDirectory'],'--expected-native-sha256',plan['Native']['Sha256'],'--']+plan['Command']
            current['Command']=['run-bounded-probe.py']+argv;flush()
            print(json.dumps(dict(Stage='starting',Role=row['Role'],Task=row['Task'],Label=row['Label'])),flush=True)
            child_environment=dict(environment);child_environment.update(plan['Environment'])
            current['ChildEnvironmentOverrides']=plan['Environment'];flush()
            current['ExitCode']=owned(HERE/'run-bounded-probe.py',argv,child_environment,parent/'console.log')
            report_path=Path(plan['ReportOutput'])/'report.json';wrapper_path=Path(plan['WrapperOutput'])/'execution.json'
            missing=[str(path) for path in (report_path,wrapper_path) if not path.is_file()]
            if missing:
                current.update(Status='probe-or-deployment-failed',Passed=False,MissingReports=missing,
                               ModelQualityEvaluated=False,FailureScope='Missing capture is not a model quality/performance verdict.')
                state['Status']='probe-or-deployment-failed';state['FinishedUtc']=utc();flush();return 2
            command=[sys.executable,'-X','utf8','-B',str(HERE/'glm-unified-task-qualification.py'),'check',
                     '--plan',row['Plan']['Path'],'--report',str(report_path),'--wrapper',str(wrapper_path),
                     '--output',str(parent/'qualification')]
            with (parent/'checker-console.log').open('x',encoding='utf-8') as stream:
                current['CheckExit']=subprocess.run(command,cwd=REPO,env=environment,stdout=stream,stderr=subprocess.STDOUT).returncode
            qualified_path=parent/'qualification/qualification.json'
            if not qualified_path.is_file():raise ValueError('Independent checker did not emit a verdict; preserve its output.')
            current.update(evidence_gate(plan,read(report_path),read(wrapper_path),read(qualified_path),current['ExitCode'],current['CheckExit']))
            current['Status']='passed' if current['Passed'] else 'failed';current['FinishedUtc']=utc();flush()
            print(json.dumps(dict(Stage='completed',Role=row['Role'],Task=row['Task'],Label=row['Label'],
                                  Passed=current['Passed'],Performance=current['Performance'])),flush=True)
            if not current['Passed']:
                state['Status']='failed';state['FinishedUtc']=utc();flush();return 1
        comparable=all(row['EqualForwardWork'] is True for row in state['Comparability'])
        state['Status']='complete' if comparable else 'complete-output-difference'
        state['Passed']=comparable;state['FinishedUtc']=utc();flush()
        return 0 if comparable else 4
    except BaseException as error:
        state['Status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'supervisor-failed'
        state['Error']=repr(error);state['FinishedUtc']=utc()
        if current is not None and current['Status']=='running':current.update(Status=state['Status'],Passed=False,Error=repr(error))
        flush();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    prep=sub.add_parser('prepare')
    for role in ('baseline','candidate'):
        for field in ('app','receipt','variant','sha256'):prep.add_argument('--'+role+'-'+field,required=True)
    prep.add_argument('--output',required=True);prep.add_argument('--include-extraction',action='store_true')
    prep.add_argument('--candidate-direct-reads',action='store_true',
                      help='Explicit candidate-only direct I/O experiment; default native read path remains the baseline.')
    prep.add_argument('--source-directory',default='D:/TensorSharp-validation-cache/default-readiness-20261010')
    prep.add_argument('--model',default='C:/Works/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf')
    prep.add_argument('--model-identity',default='D:/TensorSharp-validation-cache/master-merge-20261013/glm-native18-c-wide-tail-abba-v1/glm-c-identity.json')
    prep.add_argument('--relocation-receipt',default='E:/TensorSharp-validation-cache/master-merge-20261013/storage-migration-v1/10-glm-to-c/receipt.json')
    prep.add_argument('--relocation-source-manifest',default='D:/TensorSharp-validation-cache/default-readiness-20261010/glm-e-identity-v1.json')
    execute=sub.add_parser('run');execute.add_argument('--matrix',required=True)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args);return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
