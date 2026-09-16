#!/usr/bin/env python3
"""Prepare bounded native comparisons and explicit learned-drafter coverage.

No model runs here. Capacity-review cells remain non-runnable until a concrete
placement is reviewed; all commands require a quiet, exclusively assigned lane.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from native_release_evidence import inventory_models, primary_files, require_sha256

BASELINE_SHA256 = 'b39514dca80b62b57975bef24c02dae6896b3795652b36a415b389b0bce6aa0e'

DRAFTS = {
    'gemma4-e4b-draft': 'gemma4-e4b', 'gemma4-26b-draft': 'gemma4-26b',
    'gemma4-12b-qat-draft': 'gemma4-12b', 'gemma4-26b-qat-draft': 'gemma4-26b-qat',
    'nemotron35-dspark': 'nemotron35', 'muse-dflash2': 'muse-glimmer',
    'qwen38-27b-dflash2': 'qwen38-27b-nvfp4', 'deepseek4-dspark': 'deepseek4',
}
MOE = {'gptoss20b', 'qwen36-moe-mtp', 'qwen35-35b', 'nemotron35',
       'nemotron-omni', 'gemma4-26b', 'gemma4-26b-qat', 'glm52', 'glm53', 'glm53-flash', 'qwen38'}


def build_plan(inventory, candidate_sha256, base, inventory_path, inventory_sha256, runtime_inventory=None):
    require_sha256(candidate_sha256)
    require_sha256(inventory_sha256)
    models = inventory_models(inventory)
    # Commands target Linux even when plans are prepared on Windows.
    from pathlib import PurePosixPath
    base = PurePosixPath(base)
    runtime_inventory = runtime_inventory or str(base / 'results/download-inventory-complete.json')
    plan = {'status': 'prepared-not-executed', 'executed': False, 'qualified': False,
            'inventory': str(inventory_path), 'inventory_sha256': inventory_sha256,
            'runtime_inventory': runtime_inventory,
            'inventory_verification_scope': inventory.get('verification_scope'),
            'candidate_sha256': candidate_sha256, 'candidate_qualification': 'not established by this plan',
            'baseline_sha256': BASELINE_SHA256,
            'model_identities': {name: {'repository': model['repo'], 'assets': model['files']} for name, model in models.items()},
            'status_contract': {'prepared': 'No execution; command/configuration only.',
                'capacity-review-required': 'No execution until memory/placement review and binding checks are complete.',
                'unavailable': 'Known unsupported feature or missing runner; no passing result.',
                'passed': 'Reserved for an executed result with all identity, feature, quality and timing gates satisfied.'},
            'execution_requirements': ['Freeze and verify the named native candidate before execution.',
                'Use identical managed app/dependencies and publisher-verified weights on both labels.',
                'Pause all downloads, builds and other inference during qualified timings.',
                'Inspect actual backend, layer/TP and KV/offload banners; a requested flag is not proof of coverage.',
                'Preserve every original failure; speculation or placement fallback is not a feature pass.'],
            'comparisons': [], 'drafters': [], 'capacity_review': []}

    def command(model, profile, scenarios, name, draft=None, learned=False):
        result = ['{python}', str(base / 'repo/eng/validation/run-native-comparison.py'),
                  '--repo', str(base / 'repo'), '--dotnet', str(base / 'dotnet/dotnet'),
                  '--baseline', str(base / 'native-baseline/build/libGgmlOps.so'),
                  '--candidate', str(base / 'repo/TensorSharp.GGML.Native/build/libGgmlOps.so'),
                  '--baseline-sha256', BASELINE_SHA256, '--candidate-sha256', candidate_sha256,
                  '--inventory', runtime_inventory, '--inventory-sha256', inventory_sha256,
                  '--catalog', str(base / 'repo/eng/validation/release-model-catalog.json'),
                  '--output', str(base / 'results/native-complete' / name),
                  '--model-ids', model, '--profiles', profile, '--gpu-ids', '0,1,2,3,4,5,6',
                  '--scenarios', scenarios, '--warmup', '2', '--repeats', '3', '--telemetry']
        if draft:
            result += ['--draft-model', draft]
        if learned:
            result += ['--require-learned-speculation']
        return result

    def cell(model, profile, scope, scenarios='short,long,json,conc'):
        name = model + '-' + profile
        plan['comparisons'].append({'id': name, 'model': model, 'profile': profile, 'status': 'prepared',
            'executed': False, 'qualified': False,
            'primary_asset_paths': [item['path'] for item in primary_files(models[model])],
            'scope': scope, 'command': command(model, profile, scenarios, name)})

    for name, model in models.items():
        if 'text' not in model.get('modalities', []) or name.startswith('deepseek'):
            continue
        files = primary_files(model)
        if model['status'] != 'downloaded-and-verified' or not files:
            raise ValueError('Incomplete primary: ' + name)
        weight_bytes = sum(item['bytes'] for item in files)
        minimum = math.ceil(weight_bytes / (42 * 1024**3))
        degrees = [degree for degree in (1, 2, 3, 4, 7) if degree >= minimum]
        if not degrees:
            plan['capacity_review'].append({'model': name, 'status': 'capacity-review-required', 'executed': False, 'qualified': False,
                'reason': 'Weights exceed conservative runtime headroom estimate; this is not proof of hardware infeasibility'})
            continue
        degree = degrees[0]
        if name.startswith('glm'):
            degree = 4 if name == 'glm53-flash' else 7
            cell(name, f'layer{degree}-f16', 'Native automatic whole-layer placement; no TP requested.')
            plan['capacity_review'].append({'model': name, 'desired_feature': 'native tensor parallelism',
                'status': 'capacity-review-required', 'executed': False, 'qualified': False,
                'proposed_degree': degree, 'required_review': 'Replicated MLA/indexer KV, concurrent slots, quant-block/head shard alignment and expert offload bytes.',
                'command_after_capacity_review': command(name, f'gpu{degree}-f16', 'short,long,json,conc', name + f'-tp{degree}-f16')})
            continue
        scope = ('Specialized translation model: token stability only; authored translation quality belongs to the HTTP lane.'
                 if name == 'hunyuan-dense' else 'GPU compute placement; permanent weight residency is separate and remains unknown without explicit diagnostics.')
        if name == 'qwen38':
            scope = 'Qwen4Exp shared GPU-degree selector performs whole-layer splitting; tensor parallelism is unsupported.'
        cell(name, f'gpu{degree}-f16', scope)
        if minimum <= 1:
            cell(name, 'gpu1-q8_0', 'Requested Q8 KV; require effective cache type or explicit unsupported verdict.')
            cell(name, 'gpu1-q4_0', 'Requested Q4 KV; require effective cache type or explicit unsupported verdict.')
            cell(name, 'cpu-f16', 'Bounded native CPU prompt/output sizes; a separate backend, not GPU offload.')
            if name not in ('hunyuan-dense', 'qwen38'):
                cell(name, 'gpu2-f16', 'Local tensor parallelism; require actual sharding and identical output across native variants.')
            if name in MOE:
                cell(name, 'cpu-moe4', 'Four leading routed-expert layers offloaded; shared dense work stays on GPU.')

    for draft_id, target in DRAFTS.items():
        artifact = models[draft_id]
        if artifact['status'] != 'downloaded-and-verified' or len(artifact['files']) != 1:
            raise ValueError('Unverified draft: ' + draft_id)
        draft = artifact['files'][0]
        item = {'id': draft_id, 'target': target, 'draft': draft, 'status': 'prepared',
                'executed': False, 'qualified': False,
                'draft_kind': 'assistant-mtp' if draft_id.startswith('gemma') else 'dspark' if 'dspark' in draft_id else 'dflash2',
                'primary_asset_paths': [asset['path'] for asset in primary_files(models[target])],
                'required_checks': ['Learned draft row present after attachment.', 'Drafted>0, VerifySteps>0, Accepted>0.',
                    'Record rollback count; zero rollbacks does not qualify partial-rejection recovery.',
                    'Strict greedy tokens and finish reason match plain decode; preserve divergence as a failed parity gate.',
                    'Before/after native tokens and workload shapes match; warmed performance stays within the configured gate.'],
                'scope': 'Solo learned drafting. Ngram tool/JSON rows do not imply learned-drafter composition; concurrency fallback is not engagement.'}
        if target == 'deepseek4':
            primary = primary_files(models[target])[0]
            rows = str(base / 'results/native-complete/deepseek4-dspark/candidate.json')
            item.update(status='capacity-review-required',
                comparison='candidate plain-versus-draft only; original native baseline is not a qualified DeepSeek precision reference',
                env={'CUDA_VISIBLE_DEVICES': '0,1,2', 'TENSORSHARP_TP_DEGREE': '1', 'TS_DSV4_NGPU': '3'},
                candidate_app=str(base / 'results/native-qualified-app/candidate'),
                invocation_after_review=[str(base / 'dotnet/dotnet'), str(base / 'results/native-qualified-app/candidate/AgentTurnBench.dll'),
                    '--model', primary['path'], '--backend', 'ggml_cuda', '--kv', 'f16', '--draft-model', draft['path'],
                    '--scenarios', 'spec', '--spec-new', '256', '--warmup', '2', '--out', rows],
                post_check=['{python}', str(base / 'repo/eng/validation/validate-speculative-rows.py'),
                    '--rows', rows, '--output', rows + '.engagement.json'],
                expected_native_sha256=candidate_sha256,
                prerequisite='Prepare isolated candidate app, execute through checked_process with the pinned expected_native SHA, validate every measured draft row, and require actual 3-GPU layer placement. The raw invocation/post-check alone cannot qualify native binding or placement.',
                unsupported_target='DeepSeek V4.1 refuses the V4 DSpark companion; this cell is V4 only.')
        else:
            item['command'] = command(target, 'gpu1-f16', 'spec,newchat,json', draft_id, draft['path'], True)
        plan['drafters'].append(item)
    for target in ('qwen36-moe-mtp', 'qwen38-27b-nvfp4'):
        plan['drafters'].append({'id': target + '-embedded-mtp', 'target': target, 'status': 'prepared',
            'executed': False, 'qualified': False, 'draft_kind': 'embedded-mtp',
            'scope': 'Embedded NextN; no separate drafter. One GPU avoids a borrowed LM head split across TP ranks.',
            'command': command(target, 'gpu1-f16', 'spec,newchat,json', target + '-embedded-mtp', learned=True)})
    for target in ('glm52', 'glm53', 'glm53-flash'):
        if target == 'glm53-flash':
            plan['drafters'].append({'id': target + '-embedded-mtp', 'target': target, 'status': 'unavailable',
                'executed': False, 'qualified': False, 'draft_kind': 'embedded-mtp',
                'reason': 'glm5next NextN/MTP is not implemented; the native loader serves standard decode.',
                'evidence': ['MODEL_DOWNLOADS.md GLM-5.3-Flash row', 'TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp glm5next NextN decline']})
            continue
        degree = 4 if target == 'glm53-flash' else 7
        plan['drafters'].append({'id': target + '-embedded-mtp', 'target': target, 'status': 'capacity-review-required',
            'executed': False, 'qualified': False, 'draft_kind': 'embedded-mtp',
            'env': {'TS_SPEC': '1'}, 'scope': 'Native loader must report a retained NextN block; missing/trunk-only weights cannot count as drafting.',
            'required_review': 'Extra draft layer and cache must fit with the layer-split trunk and requested context.',
            'command_after_capacity_review': command(target, f'layer{degree}-f16', 'spec', target + '-embedded-mtp', learned=True)
                + ['--enable-native-speculation']})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--base', default='/workspace/tensorsharp-no-patch-20260915')
    parser.add_argument('--runtime-inventory', help='Location of the identical pinned inventory on the execution machine')
    args = parser.parse_args()
    if (args.output / 'manifest.json').exists():
        parser.error('Refusing to overwrite an existing plan; use a new output directory')
    raw = args.inventory.read_bytes()
    try:
        plan = build_plan(json.loads(raw), args.candidate_sha256, args.base, args.inventory,
                          hashlib.sha256(raw).hexdigest(), args.runtime_inventory)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(f"Prepared {len(plan['comparisons'])} comparison cells, {len(plan['drafters'])} drafter cells, and {len(plan['capacity_review'])} capacity reviews; executed none.")


if __name__ == '__main__':
    main()
