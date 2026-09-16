"""Small, fail-closed identity and log-evidence checks for native release runs.

Requested command-line flags are deliberately not effective-placement evidence.
Only existing model/native startup messages are recognized here. Missing banners
leave a feature unqualified; adding a parser rule requires an actual source/log.
"""
import re
from pathlib import Path


def require_sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Expected a lowercase SHA256 digest')
    return value


def primary_files(model):
    return sorted((item for item in model['files'] if item['path'].endswith('.gguf') and
                   not any(word in Path(item['path']).name.lower() for word in
                           ('mmproj', 'dflash', 'dspark', 'mtp-gemma', 'assistant', 'drafter'))),
                  key=lambda item: item['path'])


def validate_shards(files):
    """Require one complete checkpoint, rather than merely the expected count."""
    names = [Path(item['path']).name for item in files]
    if not names or len(set(names)) != len(names):
        raise ValueError('Missing or duplicate checkpoint files')
    matches = [re.fullmatch(r'(.+)-(\d{5})-of-(\d{5})\.gguf', name) for name in names]
    if not any(matches):
        if len(names) != 1:
            raise ValueError('Ambiguous primary checkpoint: multiple unsharded files')
        return
    if not all(matches):
        raise ValueError('Mixed sharded and unsharded primary checkpoint')
    identities = {(match[1], int(match[3])) for match in matches}
    if len(identities) != 1:
        raise ValueError('Mixed checkpoint shard families or totals')
    total = int(matches[0][3])
    if {int(match[2]) for match in matches} != set(range(1, total + 1)) or len(matches) != total:
        raise ValueError('Incomplete or duplicate checkpoint shard indices')


def inventory_models(inventory):
    models = {}
    for model in inventory['models']:
        name = model['id']
        if name in models:
            raise ValueError('Duplicate inventory model: ' + name)
        if model.get('status') != 'downloaded-and-verified' or not model.get('files'):
            raise ValueError('Unverified inventory model: ' + name)
        paths = set()
        for item in model['files']:
            if item['path'] in paths:
                raise ValueError('Duplicate inventory asset: ' + item['path'])
            paths.add(item['path'])
            require_sha256(item.get('download_verified_sha256'))
            if (type(item.get('bytes')) is not int or item['bytes'] <= 0 or
                    item.get('size') != item['bytes'] or not item.get('repository') or
                    not item.get('revision') or not item.get('publisher_digest_kind')):
                raise ValueError('Missing publisher identity or inconsistent asset size: ' + item['path'])
        if 'text' in model.get('modalities', []):
            validate_shards(primary_files(model))
        models[name] = model
    return models


def expected_placement(model_id, profile):
    if not profile['gpus']:
        return 'cpu'
    if profile.get('native_layer_split') or (model_id == 'qwen38' and profile['gpus'] > 1):
        return 'layer-split'
    return 'tensor-parallel' if profile['gpus'] > 1 else 'single-gpu'


def placement_evidence(log, model_id, profile):
    """Recognize existing banners, retaining exact supporting lines.

    Sources: ModelBase, Gemma4Model, each model's TensorParallel partial,
    Qwen4ExpModel.BuildLayerPlacement and ggml_ops_glm_dsa.cpp load summary.
    An individual fused-kernel decline does not invalidate genuine per-op TP.
    """
    lines = log.splitlines()
    evidence = []
    missing = []
    mismatch = []
    mode = expected_placement(model_id, profile)
    expected_backend = 'GgmlCuda' if profile['gpus'] else 'GgmlCpu'
    backend = {match[1] for line in lines if (match := re.fullmatch(r'\s*Backend:\s*(\w+)\s*', line))}
    evidence.extend(line for line in lines if re.fullmatch(r'\s*Backend:\s*(\w+)\s*', line))
    if not backend:
        missing.append('Effective model backend was not reported')
    elif backend != {expected_backend}:
        mismatch.append(f'Effective backend {sorted(backend)} differs from {expected_backend}')

    tp_degrees = set()
    layer_devices = {}
    for line in lines:
        # These banners are emitted only after weights/caches are distributed.
        match = re.search(r'\bTP weight sharding complete \((\d+) GPU|\[glm\] tensor parallel across (\d+) rank', line)
        if match:
            tp_degrees.add(int(match[1] or match[2]))
            evidence.append(line)
        if re.search(r'Layer split across \d+ GPUs:', line):
            layer_devices.update({int(device): int(count) for device, count in
                                  re.findall(r'gpu(\d+)=(\d+) layers/', line)})
            evidence.append(line)
        match = re.search(r'\[glm\]\s+device (\d+): layers (\d+)\.\.(\d+) \((\d+)\)', line)
        if match:
            if int(match[3]) - int(match[2]) + 1 != int(match[4]):
                mismatch.append('Inconsistent native layer range: ' + line.strip())
            layer_devices[int(match[1])] = int(match[4])
            evidence.append(line)
    active_layer_devices = sorted(device for device, count in layer_devices.items() if count > 0)
    if mode == 'tensor-parallel':
        if not tp_degrees:
            missing.append('No completed TP sharding banner; visible GPUs or requested degree do not prove TP')
        elif tp_degrees != {profile['gpus']}:
            mismatch.append(f'Actual TP degrees {sorted(tp_degrees)} differ from {profile["gpus"]}')
        if active_layer_devices:
            mismatch.append('Whole-layer placement cannot qualify tensor parallelism')
    elif mode == 'layer-split':
        if tp_degrees:
            mismatch.append('Tensor parallelism cannot qualify the requested whole-layer split')
        if active_layer_devices != list(range(profile['gpus'])):
            missing.append(f'Nonempty whole-layer placement on every assigned GPU was not reported: {active_layer_devices}')
    elif mode == 'single-gpu':
        if tp_degrees or len(active_layer_devices) > 1:
            mismatch.append('Actual distributed placement differs from one requested GPU')
        initialized = {int(match[1]) for line in lines if
                       (match := re.search(r'ggml_cuda_init: found (\d+) CUDA devices', line))}
        evidence.extend(line for line in lines if re.search(r'ggml_cuda_init: found \d+ CUDA devices', line))
        if initialized != {1}:
            missing.append('Exactly one initialized CUDA device was not established')

    kv_types = set()
    for line in lines:
        match = re.search(r'\bKV cache:.*\(dtype:\s*(f32|f16|bf16|q8_0|q4_0)\b', line, re.I)
        if not match:
            # Benchmark property is source-verified only for these three model
            # classes. Opaque native caches must provide their own diagnostics.
            match = re.fullmatch(r'\[agent-turn-bench\] effective managed KV storage dtype=(f32|f16|bf16|q8_0|q4_0) '
                                 r'model_type=(?:Gemma4Model|Qwen35Model|GptOssModel)', line)
        if match:
            kv_types.add(match[1].lower())
            evidence.append(line)
    if not kv_types:
        missing.append('Effective KV storage dtype was not reported; requested --kv is not evidence')
    elif kv_types != {profile['kv']}:
        mismatch.append(f'Effective KV types {sorted(kv_types)} differ from {profile["kv"]}')

    offload_layers = set()
    for line in lines:
        match = re.search(r'\[glm\] MoE CPU offload: (?:this rank\x27s experts for|routed experts of) layers 0\.\.(\d+)', line)
        if match:
            offload_layers.add(int(match[1]) + 1)
            evidence.append(line)
    requested_offload = int(profile.get('env', {}).get('TS_N_CPU_MOE', '0'))
    if requested_offload and offload_layers != {requested_offload}:
        missing.append(f'Actual expert CPU offload for {requested_offload} layers was not established')
    elif not requested_offload and offload_layers:
        mismatch.append('Unexpected expert CPU offload was reported')
    return {'status': 'unqualified' if missing or mismatch else 'passed',
            'requested_mode': mode, 'requested_gpu_count': profile['gpus'],
            'effective_backends': sorted(backend), 'effective_tp_degrees': sorted(tp_degrees),
            'permanent_weight_residency': 'unknown; compute placement does not establish permanent all-layer residency',
            'layer_counts_by_device': layer_devices, 'effective_kv_types': sorted(kv_types),
            'effective_offload_layer_counts': sorted(offload_layers),
            'evidence_lines': list(dict.fromkeys(evidence)), 'missing_evidence': missing,
            'contradictions': mismatch}
