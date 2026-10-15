"""Prepare/check frozen PP32/64/128 plans; no model, native or large-file hashing.

Requires the final deployment receipt and explicit real Native21 SHA. Reuses the
public qualification preparer and the unchanged Maya wrapper. Not executable
qualification until a separate owned runner consumes every listed case.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import types

REPO = Path('C:/Works/TensorSharp')
OLD = Path('D:/TensorSharp-validation-cache/master-merge-20261013/glm-native18-c-wide-tail-abba-v1')
TOOL = REPO / 'eng/validation/glm-unified-task-qualification.py'


def read(path):
    path = Path(path)
    if path.stat().st_size > 4 << 20:
        raise ValueError('Expected small evidence JSON')
    return json.loads(path.read_text('utf-8-sig'))


def pin(path):
    path = Path(path).resolve()
    if path.stat().st_size > 4 << 20:
        raise ValueError('No native/model/executable hashing in preparation')
    return dict(Path=str(path), Bytes=path.stat().st_size, Sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_module():
    spec = importlib.util.spec_from_file_location('glm_sweep_qualification', TOOL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def check_inputs(manifest):
    pairs = {}
    seen = set()
    repetitions = manifest['Repetitions']
    if type(repetitions) is not int or repetitions not in (1, 2) or len(manifest['Cases']) != 12 * repetitions:
        raise ValueError('Incomplete or invalid prepared matrix')
    for row in manifest['Cases']:
        label_key = (row['Task'], row['PrefillRows'], row['Engine'], row['Label'])
        if label_key in seen or row['Task'] not in ('tool_json', 'extraction') or row['PrefillRows'] not in (32, 64, 128):
            raise ValueError('Duplicate/unsupported matrix case')
        seen.add(label_key)
        plan = read(row['Plan']['Path'])
        if pin(row['Plan']['Path']) != row['Plan']:
            raise ValueError('Prepared plan changed')
        if row['Engine'] == 'tensorsharp':
            if plan['Case'] != row['Task']:
                raise ValueError('TS task binding differs')
            tokens = plan['Binding']['PromptTokens']
            if plan['NativeVariant'] != manifest['NativeVariant']:
                raise ValueError('TS native variant differs from explicit candidate')
            if plan['Native']['Sha256'].lower() != manifest['NativeSha256']:
                raise ValueError('Native SHA differs from explicit candidate')
            if plan.get('BenchmarkPrefillRows', 32) != row['PrefillRows']:
                raise ValueError('TS exact-width qualifier differs')
            if plan['RequestedOptions']['PrefillChunkTokens'] != row['PrefillRows']:
                raise ValueError('TS requested width differs')
            if plan['ExpectedProbeUnknownExit'] != 1 or plan['DeadlineSeconds'] != 900:
                raise ValueError('Do not weaken unknown-task or timeout contract')
            if plan['Model'] != manifest['Checkpoint']:
                raise ValueError('Checkpoint location differs')
        elif row['Engine'] == 'maya':
            if plan['case'] != row['Task']:
                raise ValueError('Maya task binding differs')
            tokens = plan['bound_prompt_tokens']
            if plan['prefill'] != row['PrefillRows'] or plan['max_context'] != 1024 or plan['repetitions'] != 1:
                raise ValueError('Maya shape/control differs')
            env = plan['environment']
            expected = {'STRATA_GLM_NO_MTP': '1', 'STRATA_GLM_NO_SPEC': '1', 'STRATA_GLM_NO_REUSE': '1',
                'STRATA_GLM_PREFILL_MIN': '1', 'STRATA_GLM_PREFILL_TAIL_SKIP': '0',
                'STRATA_GLM_VRAM_GB': '14', 'STRATA_GLM_POOL_GB': '6.8359375', 'STRATA_GLM_RAM_GB': '0.34'}
            if any(env.get(k) != value for k, value in expected.items()):
                raise ValueError('Maya budget/accuracy controls differ')
        else: raise ValueError('Unknown engine')
        if not isinstance(tokens, list) or not tokens or any(type(x) is not int or x < 0 or x >= 154880 for x in tokens):
            raise ValueError('Invalid raw prompt token IDs')
        expected_count = {'tool_json': 85, 'extraction': 52}[row['Task']]
        if len(tokens) != expected_count:
            raise ValueError('Historical task token count differs')
        key = (row['Task'], row['PrefillRows'])
        if key in pairs and pairs[key] != tokens:
            raise ValueError('Two engines/shapes do not consume the same raw IDs')
        pairs[key] = tokens
    for task in ('tool_json', 'extraction'):
        for width in (32, 64, 128):
            for engine, prefix in (('tensorsharp', 't'), ('maya', 'm')):
                for repeat in range(1, repetitions + 1):
                    if (task, width, engine, prefix + str(repeat)) not in seen:
                        raise ValueError('Missing actual matrix arm')
        if not (pairs[(task, 32)] == pairs[(task, 64)] == pairs[(task, 128)]):
            raise ValueError('Shape sweep changed the prompt')
    return dict(Passed=True, Tasks=2, Shapes=3, PreparedProcesses=len(manifest['Cases']),
        ExecutionStatus='not-run', LongPromptStatus='not-prepared: requires independently bound real500-token task; no padded/fabricated semantics')


def prepare(args):
    if not re.fullmatch('[0-9a-fA-F]{64}', args.native_sha256) or set(args.native_sha256) == {'0'}:
        raise ValueError('Require actual complete Native21 SHA; placeholders are forbidden')
    tool = load_module()
    receipt = read(args.deployment_receipt)
    _, native, _, _ = tool.deployment(receipt, str(Path(args.app).resolve()), args.variant)
    if native['Sha256'].lower() != args.native_sha256.lower():
        raise ValueError('Deployment receipt does not bind the requested Native21 SHA')
    out = Path(args.output)
    # Small evidence stays lexically under repo artifacts, including a declared
    # artifacts junction if selected. Do not weaken the Maya wrapper rule.
    lexical = Path(str(out.absolute()))
    if '..' in out.parts or not lexical.is_relative_to(REPO / 'artifacts'):
        raise ValueError('Output must be a fresh lexical repo artifacts directory')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(Schema='glm-prefill-paired-sweep.v1', ExecutionStatus='not-run', Repetitions=args.repetitions, Cases=[],
        RequestedWidths=[32, 64, 128], RuntimeAdmissionStatus='not-run: each Native21 exact width must be checked against actual runtime admission; no Native20 forecast cap is reused.',
        NativeVariant=args.variant, NativeSha256=args.native_sha256.lower(), Checkpoint='C:/Works/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf',
        DeploymentReceipt=pin(args.deployment_receipt), PreparationTool=pin(__file__), Qualifier=pin(TOOL),
        Scope='Fresh N1 processes; same raw C checkpoint/input IDs/PP ceiling per pair; ctx1024/cpu8/temp0/max128EOS/no draft/prefix reuse. TS host12/device14/expert7000MiB payload ceilings differ from Maya14GiB VRAM/7000MiB pool/0.34GiB RAM tier. No equivalent physical-hard-budget or stockHost-default claim.',
        PerfScope='Record PP/TTFT/post-first/whole request separately; TS actual forward counts include EOS while Maya decode=generated-1. Same IDs must be established after each run before normalizing work. No frame-count tokens. One repeat is only a pilot; two repeats form T-M-M-T per shape, not statistical significance.',
        LongPromptStatus='not-prepared; preserve a real independently bound500-token task and semantic rule before extending')
    for task in ('tool_json', 'extraction'):
        old_maya_path = OLD / (task + '-m1-maya-plan-v2.json')
        for width in (32, 64, 128):
            sequence = [('tensorsharp', 't1'), ('maya', 'm1')]
            if args.repetitions == 2: sequence += [('maya', 'm2'), ('tensorsharp', 't2')]
            elif width == 64: sequence.reverse()  # Pilot alternates order, explicitly not ABBA.
            for engine, label in sequence:
                case = out / f'{task}-pp{width}-{label}'
                if engine == 'tensorsharp':
                    tool.prepare(types.SimpleNamespace(deployment_receipt=args.deployment_receipt,
                        app=args.app, variant=args.variant, prefill=width, output=str(case),
                        source_directory='D:/TensorSharp-validation-cache/default-readiness-20261010',
                        model=manifest['Checkpoint'], model_identity=str(OLD / 'glm-c-identity.json'),
                        relocation_receipt='E:/TensorSharp-validation-cache/master-merge-20261013/storage-migration-v1/10-glm-to-c/receipt.json',
                        relocation_source_manifest='D:/TensorSharp-validation-cache/default-readiness-20261010/glm-e-identity-v1.json'))
                    plan_path = case / (task + '-plan.json')
                else:
                    case.mkdir()
                    plan = copy.deepcopy(read(old_maya_path))
                    binding = tool.bind_direct(read(plan['direct_report']), task)
                    if 'bound_prompt_tokens' in plan and plan['bound_prompt_tokens'] != binding['PromptTokens']:
                        raise ValueError('Existing Maya bound IDs differ from its historical Direct input')
                    plan['bound_prompt_tokens'] = binding['PromptTokens']
                    plan.update(prefill=width, repetitions=1, sweep_origin_plan=pin(old_maya_path),
                        controls_scope=manifest['Scope'], reference_scope='Historical Direct supplies input IDs only; independently fixed task semantics, not reference output ground truth.')
                    plan['current_harness_pins'] = {name: {'path': str(REPO / 'eng/validation' / name),
                        'sha256': pin(REPO / 'eng/validation' / name)['Sha256']} for name in
                        ('maya-token-baseline.py', 'fixed-task-semantic.py', 'multimodel-quality-bench.py', 'linux-cgroup-telemetry.py')}
                    plan_path = case / 'maya-plan.json'
                    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), 'utf-8')
                manifest['Cases'].append(dict(Task=task, Label=label, Engine=engine, PrefillRows=width,
                    Plan=pin(plan_path), OutputDirectory=str(case), DeadlineSeconds=900))
    manifest['InputQualification'] = check_inputs(manifest)
    (out / 'matrix-plan.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(manifest['InputQualification']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for name in ('app', 'deployment-receipt', 'native-sha256', 'output'): prep.add_argument('--' + name, required=True)
    prep.add_argument('--variant', choices=('native21',), default='native21')
    prep.add_argument('--repetitions', type=int, choices=(1, 2), default=1)
    check = sub.add_parser('check-inputs'); check.add_argument('--matrix', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args)
    else: print(json.dumps(check_inputs(read(args.matrix))))
