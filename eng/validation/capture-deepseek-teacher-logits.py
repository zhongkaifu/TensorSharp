#!/usr/bin/env python3
"""Unqualified, serialized slot-0 full-vocabulary teacher-forced capture.

The CLI loads the explicitly pinned native model. Run only in a coordinated
exclusive model window. Unit tests exercise the engine with a fake native API;
importing this module neither loads a native library nor opens a model.
"""
import argparse
import contextlib
import copy
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import types

import numpy as np

COMPARATOR_SHA256 = 'b096ddef66b8fd8541dcc4c62c694b96dc7d1d93dbdf00137c51f8e489f3ed55'
SUPPORTED_SCOPE = 'primary-serialized-slot0'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def read_pinned(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Missing lowercase SHA256')
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, 'Pinned artifact changed: ' + str(path))
    return raw


def pinned_json(path, expected):
    return json.loads(read_pinned(path, expected))


def child(root, relative):
    path = Path(relative)
    require(not path.is_absolute(), 'Artifact path must be relative')
    result = (Path(root) / path).resolve()
    require(result.is_relative_to(Path(root).resolve()), 'Artifact path escapes root')
    return result


def load_comparator():
    path = Path(__file__).with_name('compare-deepseek-teacher-logits.py')
    raw = read_pinned(path, COMPARATOR_SHA256)
    module = types.ModuleType('teacher_capture_comparator')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def token_bytes(tokens):
    return struct.pack('<' + 'i' * len(tokens), *tokens)


def read_tokens(root, row):
    raw = read_pinned(child(root, row['call_input_file']), row['call_input_i32_sha256'])
    require(len(raw) == row['input_token_count'] * 4, 'Call token byte count differs')
    values = np.frombuffer(raw, dtype='<i4')
    require(bool(np.all((values >= 0) & (values < row['vocab_size']))), 'Input token outside vocabulary')
    return raw


def validate_export_identity(plan, plan_hash, identity, token_export):
    require(token_export['status'] == 'tokens-exported' and token_export['release_qualified'] is False,
            'Canonical token export is incomplete/compacted')
    require(token_export['plan_sha256'] == plan_hash, 'Token export belongs to a different teacher plan')
    require([row['id'] for row in token_export['requests']] == [row['id'] for row in plan['requests']],
            'Canonical request order/coverage differs')
    tokenizer = token_export['tokenizer']
    require(tokenizer['VocabSize'] == identity['tokenizer']['vocab_size']
            and tokenizer['EosTokenIds'] == identity['tokenizer']['eos_ids'], 'Tokenizer vocabulary/EOS identity differs')
    require(token_export['tokenizer_source']['original_first_shard_sha256'] == identity['tokenizer']['first_gguf_sha256'],
            'Tokenizer metadata comes from a different checkpoint')
    for observation in (token_export['before'], token_export['after']):
        require(observation['native_mappings'] == [], 'Canonical export loaded a native executor')
        require(observation['source_sha256'] == identity['exporter_source_sha256'], 'Exporter source identity differs')
        require(observation['assembly_sha256'] == identity['exporter_assemblies_sha256'], 'Exporter assembly identity differs')


def validate_serial_schedule(plan, plan_hash, schedule, schedule_root, token_export, token_root, continuation):
    compare = load_comparator()
    rows = compare.validate_schedule(plan, plan_hash, schedule)
    require(schedule.get('capture_scope') == SUPPORTED_SCOPE, 'Only primary serialized slot0 capture is implemented')
    validate_export_identity(plan, plan_hash, schedule['shared_identity'], token_export)
    require(continuation['schema_version'] == 1 and continuation['status'] == 'complete'
            and continuation['source_variant'] == 'non-tp', 'Continuations must come from a completed independent nonTP baseline')
    identity = schedule['shared_identity']
    require(continuation['token_export_sha256'] == identity['token_export_sha256']
            and continuation['native_sha256'] == identity['native_sha256'], 'Baseline continuation identity differs')
    require(continuation.get('source_capture_sha256') and re.fullmatch('[0-9a-f]{64}', continuation['source_capture_sha256']),
            'Missing independently recorded baseline trace pin')
    prompts = {item['id']: item for item in token_export['requests']}
    require(len(prompts) == len(token_export['requests']), 'Duplicate exported request')
    forced = {item['id']: item for item in continuation['requests']}
    require(len(forced) == len(continuation['requests']), 'Duplicate baseline continuation request')
    eos = set(identity['tokenizer']['eos_ids'])
    groups = []
    for row in rows:
        require(row['slot_id'] == 0 and row['phase'] == 'primary', 'Multi-slot/batched/boundary schedules must not be silently serialized')
        require(row.get('stage') in ('prompt', 'continuation'), 'Missing prompt/continuation stage')
        if not groups or groups[-1][0]['request_id'] != row['request_id']:
            require(all(group[0]['request_id'] != row['request_id'] for group in groups), 'Request groups cannot be interleaved/revisited in this scope')
            groups.append([])
        groups[-1].append(row)
    ordered_ids = [group[0]['request_id'] for group in groups]
    selected = set(ordered_ids)
    require(ordered_ids == [row['id'] for row in plan['requests'] if row['id'] in selected], 'Scheduled request order differs from original plan')
    for group in groups:
        request = group[0]['request_id']
        require(request in prompts and request in forced, 'Missing exported prompt or independent continuation')
        prompt = prompts[request]
        prompt_raw = read_pinned(child(token_root, prompt['prefix'] + '.final.i32'), prompt['TokensI32Sha256'])
        require(len(prompt_raw) == prompt['prompt_tokens'] * 4 and prompt['RemovedMessages'] == 0, 'Exported prompt count/compaction changed')
        continuation_tokens = forced[request]['tokens']
        require(all(type(t) is int and 0 <= t < group[0]['vocab_size'] for t in continuation_tokens), 'Invalid independent forced token')
        require(len(continuation_tokens) <= continuation['maximum_tokens'] == 64, 'Continuation cap differs from protocol')
        require(not any(t in eos for t in continuation_tokens[:-1]), 'Baseline continued after EOS')
        reason = forced[request]['stop_reason']
        require(reason in ('eos', 'maximum_tokens') and continuation_tokens, 'Incomplete baseline continuation')
        require((reason == 'eos' and continuation_tokens[-1] in eos)
                or (reason == 'maximum_tokens' and len(continuation_tokens) == 64 and continuation_tokens[-1] not in eos), 'Baseline stop condition is inconsistent')
        history = b''
        prompt_seen = b''
        decoded = []
        prior = None
        for index, row in enumerate(group):
            require(row['call_index'] == index and row['position_before'] == len(history) // 4, 'Non-contiguous call index/position')
            raw = read_tokens(schedule_root, row)
            if row['stage'] == 'prompt':
                require(not decoded and len(prompt_seen) < len(prompt_raw), 'Prompt resumed after continuation')
                expected_count = min(plan['variants']['non-tp']['native_load']['n_ubatch'], (len(prompt_raw) - len(prompt_seen)) // 4)
                require(row['input_token_count'] == expected_count, 'Primary prompt must use fixed ubatch-sized calls')
                prompt_seen += raw
                require(prompt_raw.startswith(prompt_seen), 'Schedule changes canonical prompt bytes')
            else:
                require(prompt_seen == prompt_raw and row['input_token_count'] == 1 and prior is not None, 'Continuation started before complete prompt or used a batched token')
                token = struct.unpack('<i', raw)[0]
                require(token == prior['next_forced_token_id'] and token not in eos, 'Teacher token differs from prior frozen prediction or feeds EOS')
                decoded.append(token)
            history += raw
            require(sha(history) == row['cumulative_input_i32_sha256'], 'Cumulative canonical history hash differs')
            require(row['position_after'] == len(history) // 4, 'Recorded next position differs')
            if len(prompt_seen) == len(prompt_raw):
                target_index = len(decoded)
                expected = continuation_tokens[target_index] if target_index < len(continuation_tokens) else None
                require(row['next_forced_token_id'] == expected, 'Independent baseline next-token identity differs')
            else:
                require(row['next_forced_token_id'] is None, 'Intermediate prompt row cannot invent a continuation token')
            prior = row
        require(prompt_seen == prompt_raw, 'Prompt tail omitted')
        # The final row already predicts the last recorded token (EOS or cap).
        # Do not feed that last token to manufacture an additional prediction.
        expected_feeds = continuation_tokens[:-1]
        require(decoded == expected_feeds, 'Continuation rows are missing, duplicated or reordered')
    return rows


def validate_baseline_trace(baseline, continuation, schedule, schedule_sha256, plan):
    require(baseline['status'] == 'complete' and baseline['phase'] == 'independent-baseline-continuation'
            and baseline['release_qualified'] is False and baseline['variant'] == 'non-tp', 'Wrong/incomplete baseline source phase')
    require(re.fullmatch('[0-9a-f]{64}', continuation['producer_sha256'])
            and baseline['producer_sha256'] == continuation['producer_sha256'], 'Baseline producer identity differs')
    require(baseline['original_failed_build_sha256'] == schedule['original_failed_build_sha256'], 'Baseline original failed gate differs')
    # The producer existed before the schedule was frozen. Validate its process
    # and row evidence against that later schedule without asserting it was a
    # teacher-forced run or modifying the original producer artifact.
    projected = copy.deepcopy(baseline)
    projected['schedule_sha256'] = schedule_sha256
    load_comparator().validate_capture(projected, schedule, schedule_sha256, plan)


def parse_placement(text, variant, expected_layers):
    load = variant['native_load']
    starts = re.findall(r'\[dsv4\] compute devices initialized: (\d+) (\S+)', text)
    require(len(starts) == 1 and int(starts[0][0]) == load['n_gpu'] and starts[0][1].upper() == load['backend'].upper(), 'Native device count/backend did not match')
    offload = re.findall(r'\[dsv4\] routed-expert CPU offload: (\d+) of (\d+) layer\(s\); (\d+) layer\(s\) on GPUs', text)
    require(len(offload) == 1 and tuple(map(int, offload[0])) == (load['n_cpu_moe'], expected_layers, expected_layers - load['n_cpu_moe']), 'CPU expert placement differs')
    layer_lines = re.findall(r'^\[dsv4\]   device (\d+): (layers (\d+)\.\.(\d+) \((\d+)\)|no layers), ([0-9.]+) GiB free after load$', text, re.M)
    require(len(layer_lines) == load['n_gpu'] and {int(x[0]) for x in layer_lines} == set(range(load['n_gpu'])), 'Missing/duplicate device-layer evidence')
    assigned = []
    for _, _, first, last, count, _ in layer_lines:
        if first:
            require(int(last) - int(first) + 1 == int(count), 'Inconsistent layer range')
            assigned += list(range(int(first), int(last) + 1))
    require(sorted(assigned) == list(range(expected_layers)), 'Layers are omitted/duplicated in actual placement')
    degrees = re.findall(r'\[dsv41\] routed-MoE tensor parallelism: (\d+) ranks, sharded gate/up/down weights;', text)
    tp = variant['requested_expert_tp_ranks']
    require(degrees == ([str(tp)] if tp else []), 'Requested TP activation was declined or unexpectedly enabled')
    shards = re.findall(r'^\[dsv41\]   rank (\d+): ([0-9.]+) GiB of routed-expert tensor shards$', text, re.M)
    require((not tp and not shards) or (len(shards) == tp and {int(x[0]) for x in shards} == set(range(tp))
            and all(float(x[1]) > 0 for x in shards)), 'Missing positive per-rank expert shard evidence')
    return {'effective_tp_ranks': tp, 'effective_gpu_count': load['n_gpu'], 'effective_cpu_moe_layers': load['n_cpu_moe'],
            'layer_placement_evidence': [x.group(0) for x in re.finditer(r'^\[dsv4\]   device .*free after load$', text, re.M)],
            'rank_shard_evidence': [x.group(0) for x in re.finditer(r'^\[dsv41\]   rank .*tensor shards$', text, re.M)]}


class Native:
    """Explicit v5-common ABI. No silent probing/fallback to a different reset API."""
    def __init__(self, path, reset_api):
        require(reset_api in ('v5-void', 'v6-checked'), 'Declare the reset ABI explicitly')
        self.reset_api = reset_api
        self.library = ctypes.CDLL(str(path))
        self.handle = None
        def bind(name, args, result):
            function = getattr(self.library, name)
            function.argtypes, function.restype = args, result
            return function
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        self._load = bind('TSGgml_Dsv4LoadModel', [ctypes.c_char_p, integer, integer, integer, integer, integer, ctypes.c_char_p], pointer)
        self._free = bind('TSGgml_Dsv4Free', [pointer], None)
        self._vocab = bind('TSGgml_Dsv4VocabSize', [pointer], integer)
        self._context = bind('TSGgml_Dsv4CtxSize', [pointer], integer)
        self._past = bind('TSGgml_Dsv4NPast', [pointer], integer)
        self._forward = bind('TSGgml_Dsv4Forward', [pointer, ctypes.POINTER(ctypes.c_int32), integer, ctypes.POINTER(ctypes.c_float)], integer)
        self._reset = bind('TSGgml_Dsv4Reset' if reset_api == 'v5-void' else 'TSGgml_Dsv4ResetChecked', [pointer], None if reset_api == 'v5-void' else integer)

    def load(self, path, settings):
        self.handle = self._load(str(path).encode('utf-8'), settings['n_gpu'], settings['n_ctx'], settings['n_ubatch'],
                                 settings['n_threads'], settings['n_cpu_moe'], settings['backend'].encode('ascii'))
        require(bool(self.handle), 'Native model load returned null')

    def free(self):
        if self.handle:
            self._free(self.handle)
            self.handle = None

    def vocab(self): return self._vocab(self.handle)
    def context(self): return self._context(self.handle)
    def past(self): return self._past(self.handle)
    def reset(self):
        value = self._reset(self.handle)
        if self.reset_api == 'v6-checked': require(value != 0, 'Checked native reset was refused')

    def forward(self, raw, output):
        tokens = (ctypes.c_int32 * (len(raw) // 4)).from_buffer_copy(raw)
        return self._forward(self.handle, tokens, len(tokens), output)


def capture_rows(native, rows, schedule_root, output, report):
    """Core injectable for model-free tests; never chooses a teacher token."""
    output = Path(output)
    require(not output.exists(), 'Capture output must be new')
    output.mkdir(parents=True)
    report.update(status='running', rows=[], release_qualified=False)
    active, history = None, b''
    def save():
        (output / 'capture.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        vocab = rows[0]['vocab_size']
        require(native.vocab() == vocab, 'Native vocabulary differs from tokenizer')
        require(native.context() == report['native_load']['n_ctx'], 'Native context differs from plan')
        with (output / 'logits.f32').open('xb') as stream, (output / 'nonfinite-coordinates.jsonl').open('x', encoding='utf-8') as coordinates, \
                (output / 'row-manifest.jsonl').open('x', encoding='utf-8') as row_log:
            for row in rows:
                if row['request_id'] != active:
                    native.reset()
                    require(native.past() == 0, 'Reset did not return slot0 to position0')
                    active, history = row['request_id'], b''
                require(native.past() == row['position_before'] == len(history) // 4, 'Native/input position before call differs')
                raw = read_tokens(schedule_root, row)
                history += raw
                require(sha(history) == row['cumulative_input_i32_sha256'], 'Call history changed after preflight')
                # The complete raw row is written even if Forward returns failure,
                # leaves a coordinate untouched, or reports a wrong post-position.
                buffer = (ctypes.c_float * vocab)(*([float('nan')] * vocab))
                forward_error = None
                try:
                    status = int(native.forward(raw, buffer))
                except Exception as error:
                    status, forward_error = -2147483648, repr(error)
                result_raw = ctypes.string_at(ctypes.addressof(buffer), vocab * 4)
                offset = stream.tell()
                stream.write(result_raw)
                stream.flush()
                vector = np.frombuffer(result_raw, dtype='<f4')
                nonfinite = np.flatnonzero(~np.isfinite(vector)).tolist()
                actual = {**copy.deepcopy(row), 'native_status': status, 'logits_file': 'logits.f32', 'byte_offset': offset,
                          'logits_sha256': sha(result_raw), 'nonfinite_count': len(nonfinite)}
                report['rows'].append(actual)
                if forward_error is not None:
                    actual['forward_exception'] = forward_error
                coordinates.write(json.dumps({'row_id': row['row_id'], 'nonfinite_coordinates': nonfinite}) + '\n')
                coordinates.flush()
                try:
                    actual['observed_position_after'] = native.past()
                except Exception as error:
                    actual['position_query_error'] = repr(error)
                # Append one complete metadata line per raw row. Rewriting the
                # entire growing JSON manifest on every token would add quadratic
                # IO. A fatal native crash leaves these raw/JSONL records plus an
                # explicitly incomplete running manifest; no completion inferred.
                row_log.write(json.dumps(actual, allow_nan=False) + '\n')
                row_log.flush()
                require(status == 0, 'Native Forward failed; complete output buffer preserved')
                require(not nonfinite, 'Native output has nonfinite/unwritten coordinates; raw row preserved')
                require(actual.get('observed_position_after') == row['position_after'], 'Native position after call differs; raw row preserved')
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
    return report


def stat_identity(path):
    value = Path(path).stat()
    return {'device': value.st_dev, 'inode': value.st_ino, 'bytes': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


class Observer:
    def __init__(self, config, schedule):
        self.config, self.identity = config, schedule['shared_identity']

    def audit(self):
        config, identity = self.config, self.identity
        native = Path(config['native_library']).resolve()
        require(file_sha(native) == identity['native_sha256'], 'Native library hash changed')
        for name, expected in identity['native_source_sha256'].items():
            require(file_sha(child(config['native_source_root'], name)) == expected, 'Native source changed: ' + name)
        require(file_sha(config['token_export_path']) == identity['token_export_sha256'], 'Token export changed')
        require(file_sha(__file__) == identity['capture_program_sha256'], 'Capture program changed')
        read_pinned(Path(__file__).with_name('compare-deepseek-teacher-logits.py'), COMPARATOR_SHA256)
        attestation = pinned_json(config['model_attestation_path'], identity['model_attestation_sha256'])
        require(attestation['verification'] == 'publisher-hash-reuse-with-immutable-stat-attestation'
                and attestation['model_manifest_sha256'] == identity['model_manifest_sha256'], 'Wrong independent immutable model attestation')
        stats = {}
        for category in ('checkpoint_files', 'engram_files'):
            require(set(attestation[category]) == set(identity[category]), 'Model attestation file set differs')
            for path, expected in identity[category].items():
                item = attestation[category][path]
                require(item['sha256'] == expected['sha256'] and item['bytes'] == expected['bytes'], 'Model hash/size attestation differs')
                observed = stat_identity(path)
                require(observed == item['stat'] and observed['bytes'] == expected['bytes'], 'Model file changed since immutable hash attestation')
                stats[path] = observed
        maps = {}
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and (Path(parts[5]).resolve() == native or 'ggmlops' in parts[5].lower()):
                require(not parts[5].endswith(' (deleted)'), 'Deleted native mapping')
                mapped = str(Path(parts[5]).resolve())
                if mapped not in maps:
                    maps[mapped] = file_sha(mapped)
        require(maps == {str(native): identity['native_sha256']}, 'Unexpected native mapping or alternate GgmlOps library')
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': os.getpid(), 'start_ticks': int(stat[19]), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'mapped_native_libraries': maps, 'native_source_sha256': identity['native_source_sha256'],
                'checkpoint_files': identity['checkpoint_files'], 'engram_files': identity['engram_files'],
                'token_export_sha256': identity['token_export_sha256'], 'model_stat_observations': stats,
                'model_attestation_sha256': identity['model_attestation_sha256'],
                'computation_environment': {name: value for name, value in os.environ.items()
                    if name.startswith(('TS_', 'GGML_', 'NVIDIA_', 'CUDA_')) or name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_PRELOAD', 'LD_LIBRARY_PATH')},
                'model_hash_audit_policy': 'Reuse independent publisher/full-hash identities only with unchanged attested inode/size/mtime/ctime; payloads are not rehashed during capture.'}


@contextlib.contextmanager
def native_stderr(path):
    saved = os.dup(2)
    with Path(path).open('xb') as target:
        os.dup2(target.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def audit_and_free(native, observer, report, schedule, schedule_sha256, plan):
    """Always try the final identity audit before freeing a loaded executor.

    A failed load/row remains the primary error even if auditing or cleanup
    also fails. Keeping this boundary independent makes both paths testable
    without opening a native library.
    """
    if observer is not None:
        try:
            report['observations']['after'] = observer.audit()
            if report['status'] == 'complete':
                require(report['observations']['before']['mapped_native_libraries'] == report['observations']['after']['mapped_native_libraries'], 'Native mapping set changed')
                require(report['observations']['before']['computation_environment'] == report['observations']['after']['computation_environment'], 'Computation environment changed')
                load_comparator().validate_capture(report, schedule, schedule_sha256, plan)
        except Exception as error:
            report.update(status='failed', after_audit_error=repr(error))
    try:
        if native is not None:
            native.free()
    except Exception as error:
        report.update(status='failed', cleanup_error=repr(error))


def validate_failed_gate(config, identity):
    raw = read_pinned(config['original_failed_build_path'], config['original_failed_build_sha256'])
    failed = json.loads(raw)
    require(failed['status'] == 'failed' and failed['native_sha256'] == identity['native_sha256']
            and failed['source_sha256'] == identity['native_source_sha256']
            and any(s['name'] == 'tp7-checkpoint-shape' and s['exit_code'] != 0 for s in failed['steps']),
            'Original failed TP gate/source/native evidence is required')
    require(all(step['exit_code'] == 0 for step in failed['steps'] if step['name'] != 'tp7-checkpoint-shape'),
            'This diagnostic does not bypass other failed build/native safety gates')
    return raw


def apply_environment(identity, variant):
    environment, common = variant['environment'], identity['driver_environment']
    require(isinstance(common, dict) and not set(environment).intersection(common), 'Separate common driver environment from variant overrides')
    desired = {**common, **environment}
    controlled = lambda name: name.startswith(('TS_', 'GGML_', 'NVIDIA_', 'CUDA_')) or name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_PRELOAD', 'LD_LIBRARY_PATH')
    for name in os.environ:
        if controlled(name):
            require(name in desired, 'Unpinned computation environment: ' + name)
    require(all(isinstance(name, str) and isinstance(value, str) for name, value in desired.items()), 'Environment names/values must be strings')
    os.environ.update(desired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(sys.platform.startswith('linux') and sys.byteorder == 'little', 'Actual native capture requires little-endian Linux with /proc identity evidence')
    config = pinned_json(args.config, args.config_sha256)
    require(config['release_qualified'] is False and config['scope'] == SUPPORTED_SCOPE, 'Explicit unqualified primary scope required')
    plan = pinned_json(config['plan_path'], config['plan_sha256'])
    schedule = pinned_json(config['schedule_path'], config['schedule_sha256'])
    identity = schedule['shared_identity']
    require(file_sha(__file__) == identity['capture_program_sha256'], 'Capture program source pin differs')
    failed_raw = validate_failed_gate(config, identity)
    require(config['original_failed_build_sha256'] == schedule['original_failed_build_sha256'], 'Schedule refers to different failed gate evidence')
    export = pinned_json(config['token_export_path'], identity['token_export_sha256'])
    continuation = pinned_json(config['continuation_path'], schedule['baseline_continuation_sha256'])
    rows = validate_serial_schedule(plan, config['plan_sha256'], schedule, Path(config['schedule_path']).parent,
                                    export, Path(config['token_export_path']).parent, continuation)
    baseline = pinned_json(config['baseline_capture_path'], continuation['source_capture_sha256'])
    validate_baseline_trace(baseline, continuation, schedule, config['schedule_sha256'], plan)
    variant = plan['variants'][config['variant']]
    environment = variant['environment']
    # Extra model/kernel overrides must be reviewed and pinned in BOTH variants,
    # rather than inherited invisibly from an interactive development shell.
    apply_environment(identity, variant)
    require(not args.output.exists(), 'Use a new capture output directory')
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_bytes(read_pinned(args.config, args.config_sha256))
    (args.output / 'original-failed-build.json').write_bytes(failed_raw)
    for stem, path_key, expected in (
        ('plan', 'plan_path', config['plan_sha256']), ('schedule', 'schedule_path', config['schedule_sha256']),
        ('token-export', 'token_export_path', identity['token_export_sha256']),
        ('baseline-continuation', 'continuation_path', schedule['baseline_continuation_sha256']),
        ('baseline-capture', 'baseline_capture_path', continuation['source_capture_sha256']),
        ('model-attestation', 'model_attestation_path', identity['model_attestation_sha256'])):
        (args.output / (stem + '.json')).write_bytes(read_pinned(config[path_key], expected))
    inputs = args.output / 'call-inputs'
    inputs.mkdir()
    for index, row in enumerate(rows):
        (inputs / f'{index:06d}.i32').write_bytes(read_tokens(Path(config['schedule_path']).parent, row))
    report = {'schema_version': 1, 'status': 'loading', 'release_qualified': False,
              'scope': SUPPORTED_SCOPE, 'reset_api': config['reset_api'],
              'limits': ['Serialized single native slot; no inference concurrency or HTTP scheduling claim.',
                         'v5 void reset plus NPast0 does not establish hidden slot health; every subsequent Forward/status/position is checked.',
                         'Original TP numerical gate remains failed; this is diagnostic evidence only.'],
              'shared_identity': identity, 'variant': config['variant'], 'native_load': variant['native_load'],
              'environment': environment, 'native_library_path': str(Path(config['native_library']).resolve()),
              'schedule_sha256': config['schedule_sha256'], 'original_failed_build_sha256': config['original_failed_build_sha256'],
              'baseline_continuation_sha256': schedule['baseline_continuation_sha256'], 'config_sha256': args.config_sha256,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'observations': {}}
    (args.output / 'lifecycle.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    native = observer = None
    try:
        require(file_sha(config['native_library']) == identity['native_sha256'], 'Native pin differs before dlopen')
        with native_stderr(args.output / 'native.log'):
            observer = Observer(config, schedule)
            native = Native(config['native_library'], config['reset_api'])
            report['observations']['before'] = observer.audit()
            native.load(Path(plan['model']['directory']) / plan['model']['files'][0]['path'], variant['native_load'])
            report['placement'] = parse_placement((args.output / 'native.log').read_text(errors='replace'), variant, config['model_layers'])
            capture_rows(native, rows, Path(config['schedule_path']).parent, args.output / 'rows', report)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        audit_and_free(native, observer, report, schedule, config['schedule_sha256'], plan)
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Raw row paths are relative to rows/capture.json, also used by the
        # comparator. The outer lifecycle file is a separate evidence record.
        if (args.output / 'rows').exists():
            (args.output / 'rows/capture.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (args.output / 'lifecycle.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'rows': len(report['rows']), 'release_qualified': False}))
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
