#!/usr/bin/env python3
"""Compare two native bridges through identical AgentTurnBench applications.

Run --plan-only while other release workloads use the VM. Execution requires a
quiet machine; each application is isolated, actual mapped native libraries are
checked, and original JSON/token-delivery records are retained for every repeat.
HTTP semantic/tool execution suites remain separate release requirements.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from native_release_evidence import (inventory_models, placement_evidence,
                                     primary_files, require_sha256, validate_shards)


PROFILES = {
    'gpu1-f16': {'backend': 'ggml_cuda', 'gpus': 1, 'kv': 'f16'},
    'gpu1-q8_0': {'backend': 'ggml_cuda', 'gpus': 1, 'kv': 'q8_0'},
    'gpu1-q4_0': {'backend': 'ggml_cuda', 'gpus': 1, 'kv': 'q4_0'},
    'gpu2-f16': {'backend': 'ggml_cuda', 'gpus': 2, 'kv': 'f16'},
    'gpu3-f16': {'backend': 'ggml_cuda', 'gpus': 3, 'kv': 'f16'},
    'gpu4-f16': {'backend': 'ggml_cuda', 'gpus': 4, 'kv': 'f16'},
    'gpu7-f16': {'backend': 'ggml_cuda', 'gpus': 7, 'kv': 'f16'},
    'layer3-f16': {'backend': 'ggml_cuda', 'gpus': 3, 'kv': 'f16', 'tp': 1, 'native_layer_split': True},
    'layer4-f16': {'backend': 'ggml_cuda', 'gpus': 4, 'kv': 'f16', 'tp': 1, 'native_layer_split': True},
    'layer7-f16': {'backend': 'ggml_cuda', 'gpus': 7, 'kv': 'f16', 'tp': 1, 'native_layer_split': True},
    'cpu-f16': {'backend': 'ggml_cpu', 'gpus': 0, 'kv': 'f16', 'long': 512, 'new': 16},
    'cpu-moe4': {'backend': 'ggml_cuda', 'gpus': 1, 'kv': 'f16', 'env': {'TS_N_CPU_MOE': '4'}},
}
MOE_MODELS = {'gptoss20b', 'qwen36-moe-mtp', 'nemotron35', 'nemotron-omni',
              'qwen38', 'glm52', 'glm53', 'glm53-flash', 'gemma4-26b-qat', 'gemma4-26b', 'qwen35-35b'}


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected, cached=None):
    """A stat-matched prior hash remains explicitly weaker than a fresh read."""
    expected = require_sha256(expected)
    stat = path.stat()
    cached = cached or {}
    reused = (cached.get('size') == stat.st_size and cached.get('mtime_ns') == stat.st_mtime_ns
              and cached.get('sha256') == expected)
    actual = cached['sha256'] if reused else sha256(path)
    if actual != expected:
        raise ValueError(f'Asset/native SHA256 mismatch: {path}: expected {expected}, got {actual}')
    return {'path': str(path), 'sha256': actual, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'verification': 'prior-hash-with-matching-size-and-mtime' if reused else 'fresh-whole-file-sha256'}


def bind_model_identity(model, inventory):
    identity = inventory.get(model['id'])
    if identity is None:
        raise ValueError('Selected model absent from pinned inventory: ' + model['id'])
    primary = primary_files(identity)
    validate_shards(primary)
    if set(model['files']) != {item['path'] for item in primary}:
        raise ValueError('Actual checkpoint files differ from pinned inventory: ' + model['id'])
    if model['publisher'] != identity['repo']:
        raise ValueError('Catalog publisher differs from pinned inventory: ' + model['id'])
    return primary


def checkpoints(catalog, root, requested):
    plans = []
    for item in catalog['models']:
        if requested and item['id'] not in requested:
            continue
        if 'text' not in item.get('modalities', []) or item['id'].startswith('deepseek'):
            continue
        folder = root / item['id']
        primary = [path for path in sorted(folder.rglob('*.gguf'))
                   if not any(word in path.name.lower() for word in ('mmproj', 'dflash', 'dspark', 'mtp-gemma', 'assistant', 'drafter'))]
        plan = {'id': item['id'], 'publisher': item['repo'], 'files': [str(path) for path in primary]}
        if not primary:
            plan.update(status='not-run', reason='Primary checkpoint has not finished downloading')
        else:
            first = primary[0]
            try:
                validate_shards([{'path': str(path)} for path in primary])
                plan.update(status='planned', model=str(first), bytes=sum(path.stat().st_size for path in primary))
            except ValueError as error:
                plan.update(status='unavailable', reason=str(error))
        plans.append(plan)
    return plans


def checked_process(command, env, output, cwd, timeout, expected_native=None, telemetry_script=None, telemetry_interval=5):
    entry = {'command': command, 'started_unix': time.time(), 'timeout_seconds': timeout,
             'expected_native_sha256': expected_native,
             'environment_sha256': hashlib.sha256(json.dumps(env, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    started = time.monotonic()
    mapped = {}
    with output.open('w') as log:
        process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        entry['pid'] = process.pid
        telemetry = None
        if telemetry_script:
            telemetry_path = output.with_suffix('.telemetry.jsonl')
            telemetry = subprocess.Popen([sys.executable, str(telemetry_script), '--pid', str(process.pid),
                                          '--output', str(telemetry_path), '--interval', str(telemetry_interval)], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
            entry['telemetry_path'] = str(telemetry_path)
        try:
            while process.poll() is None:
                if expected_native:
                    maps = Path(f'/proc/{process.pid}/maps')
                    try:
                        for line in maps.read_text().splitlines():
                            fields = line.split(None, 5)
                            if len(fields) != 6:
                                continue
                            path = Path(fields[5])
                            if path.name == 'libGgmlOps.so' and path.is_file() and str(path) not in mapped:
                                mapped[str(path)] = sha256(path)
                    except FileNotFoundError:
                        pass  # The child can exit between poll and reading /proc.
                if time.monotonic() - started > timeout:
                    entry['error'] = 'Process timeout'
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            entry['exit_code'] = process.wait()
            entry.update(finished_unix=time.time(), wall_seconds=time.monotonic() - started)
            if telemetry:
                try:
                    telemetry.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    telemetry.terminate()
                    telemetry.wait(timeout=10)
                entry['telemetry_exit_code'] = telemetry.returncode
    if expected_native:
        entry['mapped_native_libraries'] = mapped
        entry['native_verified'] = bool(mapped) and set(mapped.values()) == {expected_native}
        if not entry['native_verified']:
            entry['error'] = 'Actual mapped native bridge does not match expected SHA256'
    return entry


def validate_learned_samples(output, count, comparator, speculative_gate):
    paths = ([Path(str(output) + f'.measure{i + 1}.json') for i in range(count)]
             if count > 1 else [output])
    reports = []
    for path in paths:
        report = speculative_gate.validate(list(comparator.load_rows(path).values()))
        reports.append({'rows': str(path), **report})
    return {'status': 'passed' if all(report['status'] == 'passed' for report in reports) else 'failed',
            'samples': reports, 'failures': [f'{report["rows"]}: {reason}' for report in reports
                                          for reason in report['failures']]}


def summarize_process_passes(output, count, comparator):
    """Retain all samples, then form one median observation per process."""
    paths = [Path(str(output) + f'.measure{i + 1}.json') for i in range(count)]
    samples = [comparator.load_rows(path) for path in paths]
    for path, sample in zip(paths[1:], samples[1:]):
        comparator.require_same_output(samples[0], sample, path)
    summary = []
    for key, first in samples[0].items():
        row = dict(first)
        for field in ('TtftMs', 'PrefillTps', 'DecodeTps', 'TotalMs'):
            row[field] = statistics.median(sample[key][field] for sample in samples)
        if 'TokenTimesMs' in row:
            row['TokenTimesMs'] = [statistics.median(sample[key]['TokenTimesMs'][i] for sample in samples)
                                   for i in range(len(row['TokenTimesMs']))]
        row.pop('StartedUnixMilliseconds', None)
        row.pop('RequestTimelines', None)
        row['Aggregation'] = f'Within-process median of {count} measured passes; original samples retained'
        summary.append(row)
    save(output, summary)
    return {'pass_count': count, 'samples': [str(path) for path in paths],
            'process_counters': str(output) + '.series.json', 'aggregation': 'within-process median'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--dotnet', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--baseline-sha256', required=True)
    ap.add_argument('--candidate-sha256', required=True)
    ap.add_argument('--inventory', type=Path, required=True, help='Publisher-verified asset inventory pinned by SHA256')
    ap.add_argument('--inventory-sha256', required=True)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--models-root', type=Path, default=Path('/workspace/models'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model-ids', default='')
    ap.add_argument('--draft-model', type=Path, help='Attach this draft GGUF; requires one exact --model-ids entry')
    ap.add_argument('--require-learned-speculation', action='store_true', help='Fail absent/inactive learned drafting and strict greedy parity divergence')
    ap.add_argument('--enable-native-speculation', action='store_true', help='Explicitly set TS_SPEC=1 for native embedded NextN loading')
    ap.add_argument('--profiles', default='gpu1-f16')
    ap.add_argument('--gpu-ids', default='0,1,2,3,4,5,6')
    ap.add_argument('--scenarios', default='short,long,tool,newchat,spec,json,conc')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--measure-passes', type=int, default=1)
    ap.add_argument('--long', type=int, default=8192)
    ap.add_argument('--new', type=int, default=64)
    ap.add_argument('--timeout', type=float, default=2400)
    ap.add_argument('--max-regression-percent', type=float, default=5)
    ap.add_argument('--verified-weights-manifest', type=Path,
                    help='Previously verified file hashes with matching size and mtime_ns; avoids rereading large checkpoints')
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--managed-dir', type=Path, help='Reuse this built managed application with --skip-build')
    ap.add_argument('--telemetry', action='store_true', help='Record GPU clocks, power and resource counters')
    ap.add_argument('--telemetry-interval', type=float, default=5)
    args = ap.parse_args()
    for value in (args.baseline_sha256, args.candidate_sha256, args.inventory_sha256):
        try:
            require_sha256(value)
        except ValueError as error:
            ap.error(str(error))
    repo = args.repo.resolve()
    if (args.output / 'comparison-manifest.json').exists():
        ap.error('Refusing to overwrite an existing comparison manifest; use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    requested_profiles = args.profiles.split(',')
    if set(requested_profiles) - set(PROFILES) or args.repeats < 1 or args.warmup < 0 or args.measure_passes < 1 or args.telemetry_interval < 1:
        ap.error('Unknown profile or invalid repeats, warmup, measurement passes, or telemetry interval')
    gpu_ids = args.gpu_ids.split(',')
    catalog = json.loads(args.catalog.read_text())
    verify_file(args.inventory, args.inventory_sha256)
    inventory = inventory_models(json.loads(args.inventory.read_text()))
    verified_weights = json.loads(args.verified_weights_manifest.read_text()).get('files', {}) if args.verified_weights_manifest else {}
    models = checkpoints(catalog, args.models_root, set(filter(None, args.model_ids.split(','))))
    if set(filter(None, args.model_ids.split(','))) - {model['id'] for model in models}:
        ap.error('Unknown, non-autoregressive, or DeepSeek model selection; this runner only qualifies non-DeepSeek text')
    if args.draft_model and (len(models) != 1 or not args.draft_model.is_file()):
        ap.error('--draft-model requires one selected model and an existing draft GGUF')
    if args.draft_model and not args.require_learned_speculation:
        ap.error('--draft-model requires --require-learned-speculation; attachment fallback cannot qualify the requested drafter')
    if args.require_learned_speculation and 'spec' not in args.scenarios.split(','):
        ap.error('--require-learned-speculation requires the spec scenario')
    if args.enable_native_speculation and not args.require_learned_speculation:
        ap.error('--enable-native-speculation requires --require-learned-speculation')
    for model in models:
        if model['status'] == 'planned':
            model['expected_assets'] = bind_model_identity(model, inventory)
    plan = {'status': 'prepared-not-executed', 'executed': False, 'qualified': False,
            'expected_native_sha256': {'baseline': args.baseline_sha256, 'candidate': args.candidate_sha256},
            'inventory': str(args.inventory), 'inventory_sha256': args.inventory_sha256,
            'models': models, 'profiles': {key: PROFILES[key] for key in requested_profiles},
            'scenarios': args.scenarios.split(','), 'repeats': args.repeats,
            'warmup_passes': args.warmup, 'measured_passes_per_process': args.measure_passes,
            'telemetry_interval_seconds': args.telemetry_interval,
            'draft_model': str(args.draft_model) if args.draft_model else None,
            'require_learned_speculation': args.require_learned_speculation,
            'enable_native_speculation': args.enable_native_speculation,
            'known_limitations': [
                'Missing effective backend, placement, KV or offload banners leave a completed run unqualified.',
                'Tool scenario supplies synthetic history; actual tool selection/execution requires HTTP/skills suites.',
                'Quantized KV support must be checked against model startup diagnostics.',
                'MoE offload profiles apply only to models with routed experts.',
                'These tests do not exercise DeepSeek, embedding models, or generative media.'],
            'runs': [], 'comparisons': []}
    manifest = args.output / 'comparison-manifest.json'
    save(manifest, plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    # Verify the explicitly named binaries before a build, checkpoint hash pass,
    # or inference process. A current-file hash is not a substitute for a pin.
    for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
        verify_file(path, plan['expected_native_sha256'][label])
    spec = importlib.util.spec_from_file_location('agent_turn_compare', repo / 'benchmarks/AgentTurnBench/compare.py')
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    spec = importlib.util.spec_from_file_location('speculative_gate', repo / 'eng/validation/validate-speculative-rows.py')
    speculative_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(speculative_gate)
    env = dict(os.environ)
    env.update({'PATH': str(args.dotnet.parent) + ':' + env.get('PATH', ''),
                'DOTNET_ROOT': str(args.dotnet.parent), 'TENSORSHARP_GGML_NATIVE_SKIP': 'true',
                'TENSORSHARP_MLX_NATIVE_SKIP': 'true'})
    managed = args.managed_dir.resolve() if args.managed_dir else args.output / 'managed'
    if not args.skip_build:
        build = [str(args.dotnet), 'build', str(repo / 'benchmarks/AgentTurnBench/AgentTurnBench.csproj'),
                 '-c', 'Release', '-o', str(managed), '-p:TensorSharpSkipGgmlNative=true',
                 '-p:TensorSharpSkipMlxNative=true', '--ignore-failed-sources']
        plan['build'] = checked_process(build, env, args.output / 'managed-build.log', repo, 900)
        save(manifest, plan)
        if plan['build']['exit_code']:
            return 1
    dll = managed / 'AgentTurnBench.dll'
    if not dll.is_file():
        raise RuntimeError(f'Missing built benchmark: {dll}')
    snapshots = {}
    for label, native in (('baseline', args.baseline), ('candidate', args.candidate)):
        directory = args.output / 'apps' / label
        shutil.copytree(managed, directory, dirs_exist_ok=True)
        shutil.copy2(native, directory / 'libGgmlOps.so')
        snapshots[label] = {'directory': str(directory), 'native_sha256': sha256(directory / 'libGgmlOps.so'),
                            'non_native_files_sha256': {str(path.relative_to(directory)): sha256(path)
                                for path in directory.rglob('*') if path.is_file() and path.name != 'libGgmlOps.so'}}
        if snapshots[label]['native_sha256'] != plan['expected_native_sha256'][label]:
            raise RuntimeError('Native binary changed while preparing isolated app: ' + label)
    if snapshots['baseline']['non_native_files_sha256'] != snapshots['candidate']['non_native_files_sha256']:
        raise RuntimeError('Baseline/candidate managed application or dependency files differ')
    plan['snapshots'] = snapshots
    plan['managed_sha256'] = sha256(dll)
    if args.draft_model:
        identities = [item for model in inventory.values() for item in model['files']
                      if item['path'] == str(args.draft_model)]
        if not identities or len({item['download_verified_sha256'] for item in identities}) != 1:
            raise ValueError('Draft asset missing or ambiguous in pinned inventory')
        plan['draft_identity'] = identities[0]
        plan['draft_verification'] = verify_file(args.draft_model, identities[0]['download_verified_sha256'],
                                                  verified_weights.get(str(args.draft_model)))
        plan['draft_model_sha256'] = plan['draft_verification']['sha256']
    save(manifest, plan)
    for model in models:
        if model['status'] != 'planned':
            continue
        hash_start = time.monotonic()
        model['checkpoint_sha256'] = {}
        model['asset_verification'] = []
        for asset in model['expected_assets']:
            filename = asset['path']
            path = Path(filename)
            if path.stat().st_size != asset['bytes']:
                raise ValueError('Checkpoint size differs from pinned publisher inventory: ' + filename)
            checked = verify_file(path, asset['download_verified_sha256'], verified_weights.get(filename))
            model['checkpoint_sha256'][filename] = checked['sha256']
            model['asset_verification'].append(checked)
        model['hash_phase_seconds'] = time.monotonic() - hash_start
        model['verified_hash_manifest'] = str(args.verified_weights_manifest) if args.verified_weights_manifest else None
        for profile_name in requested_profiles:
            profile = PROFILES[profile_name]
            unsupported = None
            if profile.get('native_layer_split') and not model['id'].startswith('glm'):
                unsupported = 'Native automatic whole-layer profile currently applies to GLM; Qwen4Exp uses the shared GPU-degree layer selector'
            if profile['gpus'] and model['bytes'] > profile['gpus'] * 42 * 1024**3:
                plan['comparisons'].append({'model': model['id'], 'profile': profile_name, 'status': 'unavailable', 'executed': False, 'qualified': False,
                    'reason': 'Capacity review required: weights exceed the conservative 42GiB per assigned A40 estimate; this is not proof of hardware infeasibility',
                    'minimum_gpus_by_weight_bytes': (model['bytes'] + 42 * 1024**3 - 1) // (42 * 1024**3)})
                save(manifest, plan)
                continue
            if profile['gpus'] > 1 and model['id'] == 'hunyuan-dense':
                unsupported = 'HunyuanDenseArchitecture explicitly supports one device only'
            if profile_name == 'cpu-moe4' and model['id'] not in MOE_MODELS:
                unsupported = 'This catalog checkpoint has no documented routed-expert offload path'
            if unsupported:
                plan['comparisons'].append({'model': model['id'], 'profile': profile_name,
                                            'status': 'unavailable', 'executed': False, 'qualified': False, 'reason': unsupported})
                save(manifest, plan)
                continue
            if profile['gpus'] > len(gpu_ids):
                plan['comparisons'].append({'model': model['id'], 'profile': profile_name,
                                            'status': 'unavailable', 'executed': False, 'qualified': False, 'reason': 'Insufficient assigned GPUs'})
                continue
            run_env = dict(env)
            for key in ('TENSORSHARP_TP_DEGREE', 'TS_N_CPU_MOE', 'TS_CPU_MOE',
                        'TS_SPEC', 'TS_SPEC_DRAFT_MODEL', 'TS_MTP_DRAFT_MODEL'):
                run_env.pop(key, None)
            run_env.update({'CUDA_VISIBLE_DEVICES': ','.join(gpu_ids[:profile['gpus']]),
                            'TENSORSHARP_TP_DEGREE': str(profile.get('tp', max(1, profile['gpus']))),
                            'TS_CPU_MOE_THREADS': '16', 'OMP_NUM_THREADS': '16'})
            run_env.update(profile.get('env', {}))
            if args.enable_native_speculation:
                run_env['TS_SPEC'] = '1'
            outputs = {'baseline': [], 'candidate': []}
            cell = args.output / model['id'] / profile_name
            cell.mkdir(parents=True, exist_ok=True)
            successful = True
            placement_qualified = True
            for repeat in range(args.repeats):
                for label in (('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')):
                    snapshot = snapshots[label]
                    output = cell / f'{label}-{repeat + 1}.json'
                    command = [str(args.dotnet), str(Path(snapshot['directory']) / 'AgentTurnBench.dll'),
                               '--model', model['model'], '--backend', profile['backend'], '--kv', profile['kv'],
                               '--chunk', '1024', '--max-batched', '4096', '--long', str(profile.get('long', args.long)),
                               '--tool', '3000', '--new', str(profile.get('new', args.new)), '--conc', '1,2,4',
                               '--scenarios', args.scenarios, '--warmup', str(args.warmup), '--out', str(output)]
                    if args.measure_passes > 1:
                        command += ['--measure-passes', str(args.measure_passes)]
                    if args.draft_model:
                        command += ['--draft-model', str(args.draft_model)]
                    print(f'Running {model["id"]} {profile_name} {label} repeat{repeat + 1}', flush=True)
                    entry = checked_process(command, run_env, output.with_suffix('.log'), repo,
                                            args.timeout, snapshot['native_sha256'],
                                            repo / 'eng/validation/sample-release-telemetry.py' if args.telemetry else None,
                                            args.telemetry_interval)
                    if args.measure_passes > 1 and entry['exit_code'] == 0:
                        try:
                            entry['within_process_measurement'] = summarize_process_passes(output, args.measure_passes, comparator)
                        except (OSError, ValueError) as error:
                            entry['error'] = 'Measured-pass validation failed: ' + str(error)
                            entry['exit_code'] = 1
                    if args.require_learned_speculation and entry['exit_code'] == 0 and output.is_file():
                        try:
                            gate = validate_learned_samples(output, args.measure_passes, comparator, speculative_gate)
                            entry['learned_speculation'] = gate
                            if gate['status'] != 'passed':
                                entry['error'] = '; '.join(gate['failures'])
                                entry['exit_code'] = 1
                        except (OSError, ValueError) as error:
                            entry['error'] = 'Learned-speculation validation failed: ' + str(error)
                            entry['exit_code'] = 1
                    entry.update(model=model['id'], profile=profile_name, variant=label, repeat=repeat + 1,
                                 executed=True, checkpoint_sha256=model['checkpoint_sha256'],
                                 draft_model_sha256=plan.get('draft_model_sha256'),
                                 output=str(output), environment={key: run_env[key] for key in
                                     ('CUDA_VISIBLE_DEVICES', 'TENSORSHARP_TP_DEGREE', 'TS_CPU_MOE_THREADS', 'OMP_NUM_THREADS', 'TS_N_CPU_MOE', 'TS_SPEC') if key in run_env})
                    log_text = output.with_suffix('.log').read_text(errors='replace')
                    entry['placement_gate'] = placement_evidence(log_text, model['id'], profile)
                    placement_qualified &= entry['placement_gate']['status'] == 'passed'
                    entry['status'] = ('failed' if entry['exit_code'] or not entry['native_verified'] or not output.is_file()
                                       else 'completed-pending-comparison' if entry['placement_gate']['status'] == 'passed' else 'unqualified')
                    entry['qualified'] = False
                    entry['placement_and_cache_diagnostics'] = [line for line in log_text.splitlines()
                        if re.search(r'tensor.parallel|layer.split|offload|KV.cache|kv.dtype|quantized.*cache', line, re.I)]
                    entry['requested_moe_layers'] = profile.get('env', {}).get('TS_N_CPU_MOE', '0')
                    entry['load_and_warmup_diagnostics'] = [line for line in log_text.splitlines()
                        if re.search(r'loaded .* in |warm.up|warmup', line, re.I)]
                    plan['runs'].append(entry)
                    save(manifest, plan)
                    if entry['exit_code'] or not entry['native_verified'] or not output.is_file():
                        successful = False
                    else:
                        outputs[label].append(output)
                if not successful:
                    break
            comparison = {'model': model['id'], 'profile': profile_name, 'status': 'failed',
                          'executed': True, 'qualified': False}
            if successful:
                command = [sys.executable, str(repo / 'benchmarks/AgentTurnBench/compare.py'),
                           str(outputs['baseline'][0]), str(outputs['candidate'][0]),
                           '--max-regression-percent', str(args.max_regression_percent)]
                for label in ('baseline', 'candidate'):
                    for output in outputs[label][1:]:
                        command += [f'--{label}-repeat', str(output)]
                comparison['process'] = checked_process(command, env, cell / 'comparison.log', repo, 60)
                comparison['token_and_performance_status'] = 'passed' if comparison['process']['exit_code'] == 0 else 'failed'
                comparison['status'] = ('failed' if comparison['process']['exit_code'] else
                                        'passed' if placement_qualified else 'unqualified')
                comparison['qualified'] = comparison['status'] == 'passed'
            for entry in plan['runs']:
                if entry['model'] == model['id'] and entry['profile'] == profile_name:
                    if entry['status'] == 'completed-pending-comparison':
                        entry['status'] = comparison['status']
                    entry['qualified'] = comparison['qualified'] and entry['status'] == 'passed'
            plan['comparisons'].append(comparison)
            save(manifest, plan)
    plan['finished_unix'] = time.time()
    plan['run_complete'] = True
    plan['executed'] = bool(plan['runs'])
    plan['qualified'] = (bool(plan['comparisons']) and all(cell['status'] == 'passed' for cell in plan['comparisons'])
                         and all(model['status'] == 'planned' for model in models))
    plan['status'] = ('failed' if any(cell['status'] == 'failed' for cell in plan['comparisons']) else
                      'passed' if plan['qualified'] else 'unqualified' if plan['executed'] else 'unavailable')
    save(manifest, plan)
    if not plan['runs']:
        return 2
    return 0 if plan['qualified'] else 1 if plan['status'] == 'failed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
