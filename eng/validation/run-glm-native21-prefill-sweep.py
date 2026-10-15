"""Consume a frozen paired shape plan through existing owned-process wrappers.

No direct model Popen here. Wrapper main executes in this process so interruption
unwinds its established child cleanup. First failed/unknown qualification stops.
This script is preparation only until root grants an exclusive GPU window.
"""
import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

REPO = Path('C:/Works/TensorSharp')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def pair_comparability(cases, repetitions):
    result = []
    for task in ('tool_json', 'extraction'):
        for width in (32, 64, 128):
            rows = [r for r in cases if r['Task'] == task and r['PrefillRows'] == width]
            terminal = [r for r in rows if r.get('Status') in ('passed', 'failed')]
            complete = len(terminal) == 2 * repetitions
            qualified = complete and all(r.get('Passed') is True for r in terminal)
            by_engine = {engine: [r for r in terminal if r['Engine'] == engine] for engine in ('tensorsharp', 'maya')}
            comparable = qualified and all(len(v) == repetitions for v in by_engine.values())
            result.append(dict(Task=task, PrefillRows=width, Complete=complete, IndependentQualityQualified=qualified,
                SameRawPromptIds=(all(r['RawPromptTokens'] == terminal[0]['RawPromptTokens'] for r in terminal) if comparable else None),
                SameRawOutputIds=(all(r['RawOutputTokens'] == terminal[0]['RawOutputTokens'] for r in terminal) if comparable else None),
                OutputTokenCountsByEngine={engine: [len(r['RawOutputTokens']) for r in values] for engine, values in by_engine.items()},
                ActualOutputForwardCountsByEngine={engine: [r['Performance']['OutputForwardCount'] for r in values] for engine, values in by_engine.items()},
                Scope='Equal raw output IDs do not equal forwarded work: TS includes an EOS forward; Maya does not. Different outputs prohibit same-work latency-ratio claims. No physical-budget equivalence or throughput winner is inferred.'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True)
    args = parser.parse_args()
    preparer = module(Path(__file__).with_name('prepare-glm-native21-prefill-sweep.py'), 'glm_shape_preparer')
    manifest = preparer.read(args.matrix); preparer.check_inputs(manifest)
    base = Path(args.matrix).parent
    output = base / 'matrix-execution.json'
    if output.exists(): raise ValueError('Never overwrite or silently resume an existing matrix')
    tools = [REPO / 'eng/validation' / name for name in
        ('run-bounded-probe.py', 'glm-unified-task-qualification.py', 'maya-token-baseline.py',
         'fixed-task-semantic.py', 'multimodel-quality-bench.py', 'linux-cgroup-telemetry.py')]
    pins = {str(path): preparer.pin(path) for path in tools}
    if preparer.pin(manifest['Qualifier']['Path']) != manifest['Qualifier']:
        raise ValueError('Frozen qualifier changed')
    state = dict(Status='running', StartedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        Matrix=preparer.pin(args.matrix), Runner=preparer.pin(__file__), Tools=pins, Cases=[],
        Scope=manifest['Scope'], PerfScope=manifest['PerfScope'])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TS_', 'TENSORSHARP_', 'GGML_', 'GLM_', 'STRATA_', 'OMP_', 'GOMP_', 'NCCL_'))
        and k not in ('MAX_CONTEXT', 'KV_CACHE_DTYPE', 'CUDA_MPS_PIPE_DIRECTORY')}
    env.update(CUDA_VISIBLE_DEVICES='0', CUDA_MODULE_LOADING='EAGER',
        TEMP='E:/TensorSharp-validation-cache/tmp', TMP='E:/TensorSharp-validation-cache/tmp',
        TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS='1')

    def save():
        output.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')

    def require_probe_reports(*paths):
        missing = [str(path) for path in paths if not Path(path).is_file()]
        if not missing:
            return True
        current.update(Status='probe-or-deployment-failed', Passed=False,
            FailureCategory='probe-or-deployment-failure', ModelQualityEvaluated=False,
            MissingReports=missing,
            FailureScope='Probe exited without required report evidence. Preserve wrapper/console; do not interpret this unqualified startup/deployment/crash as a model quality or performance regression.')
        state['Status'] = 'probe-or-deployment-failed'
        state['FinishedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        return False

    def owned(script, argv, log):
        old_args, old_env = sys.argv[:], dict(os.environ)
        try:
            sys.argv = [str(script)] + argv
            os.environ.clear(); os.environ.update(env)
            with log.open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                return runpy.run_path(str(script), run_name='glm_shape_owned_wrapper')['main']()
        finally:
            sys.argv = old_args; os.environ.clear(); os.environ.update(old_env)

    save()
    current = None
    try:
        for row in manifest['Cases']:
            if (base / 'stop-before-next-step').exists():
                state['Status'] = 'stopped-between-cases'; save(); return 3
            if any(preparer.pin(path) != bound for path, bound in pins.items()):
                raise ValueError('Frozen wrapper/helper changed between cases')
            if preparer.pin(args.matrix) != state['Matrix']:
                raise ValueError('Matrix changed between cases')
            active = subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command',
                '@(Get-Process -Name TensorSharp.Server.Host -ErrorAction SilentlyContinue).Count'], text=True)
            if active.strip() != '0': raise ValueError('User Host is active; do not interrupt it')
            plan = preparer.read(row['Plan']['Path'])
            if preparer.pin(row['Plan']['Path']) != row['Plan']: raise ValueError('Frozen plan changed')
            parent = Path(row['OutputDirectory'])
            current = dict(row, Status='running', StartedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            state['Cases'].append(current); save()
            print(json.dumps(dict(Stage='starting', Task=row['Task'], Engine=row['Engine'], Width=row['PrefillRows'], Label=row['Label'])), flush=True)
            if row['Engine'] == 'tensorsharp':
                for path in (plan['WrapperOutput'], plan['ReportOutput']):
                    if Path(path).exists(): raise ValueError('Execution output is not fresh')
                argv = ['--output', plan['WrapperOutput'], '--timeout', '900', '--telemetry',
                    '--managed-directory', plan['AppDirectory'], '--expected-native-sha256', manifest['NativeSha256'], '--'] + plan['Command']
                current['Command'] = ['run-bounded-probe.py'] + argv; save()
                current['ExitCode'] = owned(REPO / 'eng/validation/run-bounded-probe.py', argv, parent / 'console.log')
                report_path = Path(plan['ReportOutput']) / 'report.json'
                wrapper_path = Path(plan['WrapperOutput']) / 'execution.json'
                if not require_probe_reports(report_path, wrapper_path):
                    return 2
                report, wrapper = preparer.read(report_path), preparer.read(wrapper_path)
                # Preserve the existing public CLI check contract and its own
                # evidence pins; this child never owns or launches a model.
                check = [sys.executable, '-X', 'utf8', '-B', str(REPO / 'eng/validation/glm-unified-task-qualification.py'),
                    'check', '--plan', row['Plan']['Path'], '--report', str(report_path), '--wrapper', str(wrapper_path),
                    '--output', str(parent / 'qualification')]
                with (parent / 'checker-console.log').open('x', encoding='utf-8') as log:
                    current['CheckExit'] = subprocess.run(check, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                qualified = preparer.read(parent / 'qualification/qualification.json')
                output_ids, prompt_ids = report['OutputTokens'], report['PromptTokens']
                timing = report['Timing']; forwards = len(output_ids)
                counts = (timing['EngineOutputTokens'] == forwards == timing['OutputForwardCount']
                    and report['ComputedTokens'] == len(prompt_ids) + forwards)
                passed = (current['ExitCode'] == wrapper['exit_code'] == plan['ExpectedProbeUnknownExit'] == 1
                    and current['CheckExit'] == 0 and qualified.get('Passed') is True and counts
                    and wrapper['complete'] and not wrapper['timed_out'] and not wrapper['cleanup_errors']
                    and wrapper['telemetry']['passed'] and not wrapper['telemetry']['errors']
                    and not report['Errors'] and report['LedgerZeroAfterCleanup'])
                semantic = qualified.get('IndependentSemantic'); eos = report.get('NaturalEos')
                performance = dict(PrefillAttributedMs=timing.get('PrefillAttributedMs'), DecodeAttributedMs=timing.get('DecodeAttributedMs'),
                    FirstPublishedTokenWallMs=timing.get('FirstPublishedTokenWallMs'), RequestWallMs=timing.get('RequestWallMs'),
                    OutputForwardCount=forwards, Convention='P+O including EOS forward')
            else:
                for toolpin in plan['current_harness_pins'].values():
                    if preparer.pin(toolpin['path'])['Sha256'] != toolpin['sha256']: raise ValueError('Maya executing helper changed')
                target = parent / 'maya-run'
                if target.exists(): raise ValueError('Maya execution output is not fresh')
                argv = ['--plan', row['Plan']['Path'], '--output', str(target), '--timeout', '900']
                current['Command'] = ['maya-token-baseline.py'] + argv; save()
                current['ExitCode'] = owned(REPO / 'eng/validation/maya-token-baseline.py', argv, parent / 'console.log')
                if not require_probe_reports(target / 'report.json'):
                    return 2
                report = preparer.read(target / 'report.json')
                if len(report.get('rows', [])) != 1: raise ValueError('Expected exactly one completed Maya request')
                actual = report['rows'][0]; output_ids = actual['tokens']; prompt_ids = plan['bound_prompt_tokens']
                counts = (actual['generated'] == len(output_ids) and actual['prompt_tokens'] == len(prompt_ids))
                log_path = target / 'stdout.log'
                if log_path.stat().st_size > 4 << 20: raise ValueError('Unexpectedly large Maya protocol log')
                raw_log = log_path.read_text('utf-8-sig')
                headers = re.findall(r'^glm prefill: CUDA0 chunks of ([0-9]+) tokens \(--prefill ([0-9]+);', raw_log, re.M)
                progress = [(int(done), int(total)) for done, total in re.findall(r'^PP ([0-9]+) ([0-9]+) ', raw_log, re.M)]
                width, prompt_count = row['PrefillRows'], len(prompt_ids)
                expected_progress = [(done, prompt_count) for done in range(width, prompt_count - 1, width)]
                expected_progress += [(prompt_count - 1, prompt_count), (prompt_count, prompt_count)]
                width_verified = (headers == [(str(width), str(width))] and progress == expected_progress)
                passed = (current['ExitCode'] == 0 and report['quality_passed'] and report['process_success']
                    and report['identities_unchanged'] and not report['errors'] and counts and report['plan'] == plan
                    and width_verified and report['performance_observation_complete'] and not report['sampling_errors'])
                semantic = actual.get('semantic'); eos = actual.get('quality', {}).get('checkpoint_eos_observed')
                performance = dict(PrefillDoneMs=actual.get('prompt_ms'), DecodeDoneMs=actual.get('decode_ms'),
                    FirstTokenWallMs=actual.get('first_token_ms'), RequestWallMs=actual.get('wall_ms'),
                    OutputForwardCount=len(output_ids) - 1, ConfiguredPrefillRows=width,
                    ActualPrefillCumulativeRows=progress, ExactPrefillWidthVerified=width_verified,
                    Convention='P+O-1; Maya batches P-1 then scalar final prompt token; DONE decode includes first-token sampling')
            expected_ids = plan['Binding']['PromptTokens'] if row['Engine'] == 'tensorsharp' else plan['bound_prompt_tokens']
            passed = bool(passed and prompt_ids == expected_ids and eos is True)
            current.update(Passed=passed, Status='passed' if passed else 'failed', FinishedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                RawPromptTokens=prompt_ids, RawOutputTokens=output_ids, IndependentSemantic=semantic, NaturalEos=eos, CountsValid=counts, Performance=performance)
            state['PairComparability'] = pair_comparability(state['Cases'], manifest['Repetitions'])
            save(); print(json.dumps(dict(Stage='completed', Task=row['Task'], Engine=row['Engine'], Width=row['PrefillRows'], Passed=passed, Performance=performance)), flush=True)
            if not passed: state['Status'] = 'failed'; save(); return 1
        state['Status'] = 'complete'; state['FinishedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save(); return 0
    except BaseException as error:
        state['Status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'supervisor-failed'
        state['Error'] = repr(error)
        if current and current['Status'] == 'running': current.update(Status=state['Status'], Error=repr(error))
        state['FinishedUtc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
