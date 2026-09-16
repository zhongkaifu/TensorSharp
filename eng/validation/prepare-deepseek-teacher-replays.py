#!/usr/bin/env python3
"""Derive unqualified replay configs only from a completed, pinned non-TP producer.

This command prepares JSON/argv only. It never loads a native library or model.
The original TP numerical failure remains an input to every replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def sha(raw): return hashlib.sha256(raw).hexdigest()
def pinned(path, digest):
    raw = Path(path).read_bytes()
    if sha(raw) != digest: raise ValueError('Input SHA256 differs: ' + str(path))
    return raw
def load_helper(path, digest):
    raw = pinned(path, digest)
    module = types.ModuleType('pinned_teacher_replay_helper')
    module.__file__ = str(Path(path).resolve())
    # Dataclasses need their module registered; execute exactly the checked bytes.
    sys.modules[module.__name__] = module
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


def derive_configs(producer, frozen, producer_root, output_root):
    if producer['release_qualified'] is not False or producer['variant'] != 'non-tp' or producer['maximum_tokens'] != 64:
        raise ValueError('Only an explicitly unqualified non-TP producer can define teacher replay')
    if frozen['status'] != 'frozen' or frozen['release_qualified'] is not False or frozen['requests'] != 113:
        raise ValueError('Completed 113-request frozen baseline required')
    root = Path(producer_root).resolve()
    common_names = ['release_qualified', 'scope', 'plan_path', 'plan_sha256', 'native_library', 'native_source_root',
                    'original_failed_build_path', 'original_failed_build_sha256', 'token_export_path',
                    'model_attestation_path', 'reset_api', 'model_layers']
    common = {key: producer[key] for key in common_names}
    common.update(schedule_path=str(root / 'teacher-schedule.json'), schedule_sha256=frozen['schedule_sha256'],
                  continuation_path=str(root / 'baseline-continuation.json'), baseline_capture_path=str(root / 'rows/capture.json'))
    # Same fixed order used to expose process/reload variance, with two controls
    # and two TP captures. This does not promise statistical performance power.
    result = []
    for label, variant in [('non-tp-r1', 'non-tp'), ('tp7-r1', 'expert-tp7'), ('tp7-r2', 'expert-tp7'), ('non-tp-r2', 'non-tp')]:
        result.append((label, {**common, 'variant': variant}, str(Path(output_root).resolve() / label)))
    return result


def prepare(producer_path, producer_sha, frozen_path, frozen_sha, output, capture_output):
    producer_raw = pinned(producer_path, producer_sha)
    producer = json.loads(producer_raw)
    frozen_raw = pinned(frozen_path, frozen_sha)
    frozen = json.loads(frozen_raw)
    helper = load_helper(producer['capture_helper_path'], producer['capture_helper_sha256'])
    helper.require(helper.file_sha(Path(producer['capture_helper_path']).with_name('generate-deepseek-teacher-baseline.py')) == producer['producer_sha256'], 'Baseline producer program differs')
    plan = helper.pinned_json(producer['plan_path'], producer['plan_sha256'])
    identity = helper.pinned_json(producer['shared_identity_path'], producer['shared_identity_sha256'])
    helper.require(identity['capture_program_sha256'] == producer['capture_helper_sha256'], 'Capture helper differs')
    helper.validate_failed_gate(producer, identity)
    root = Path(frozen_path).resolve().parent
    schedule = helper.pinned_json(root / 'teacher-schedule.json', frozen['schedule_sha256'])
    continuation = helper.pinned_json(root / 'baseline-continuation.json', frozen['continuation_sha256'])
    baseline = helper.pinned_json(root / 'rows/capture.json', frozen['source_capture_sha256'])
    exported = helper.pinned_json(producer['token_export_path'], identity['token_export_sha256'])
    helper.require(schedule['shared_identity'] == identity and baseline['config_sha256'] == producer_sha,
                   'Completed baseline was not produced by this exact config/identity')
    helper.require(continuation['producer_sha256'] == producer['producer_sha256']
                   and continuation['source_capture_sha256'] == frozen['source_capture_sha256'], 'Producer continuation binding differs')
    helper.validate_serial_schedule(plan, producer['plan_sha256'], schedule, root, exported,
                                    Path(producer['token_export_path']).parent, continuation)
    helper.validate_baseline_trace(baseline, continuation, schedule, frozen['schedule_sha256'], plan)
    helper.require(frozen['rows'] == len(schedule['rows']) and frozen['raw_f32_bytes'] == sum(r['vocab_size'] * 4 for r in schedule['rows']),
                   'Frozen row/byte summary differs')
    rows = derive_configs(producer, frozen, root, capture_output)
    output = Path(output)
    helper.require(not output.exists(), 'Replay config output must be new')
    output.mkdir(parents=True)
    commands = []
    for label, config, destination in rows:
        raw = (json.dumps(config, indent=2) + '\n').encode()
        path = output / (label + '.json')
        path.write_bytes(raw)
        commands.append({'label': label, 'variant': config['variant'], 'config_sha256': sha(raw),
                         'argv': [sys.executable, producer['capture_helper_path'], '--config', str(path.resolve()),
                                  '--config-sha256', sha(raw), '--output', destination]})
    manifest = {'status': 'prepared-not-executed', 'release_qualified': False, 'scope': producer['scope'],
                'program_sha256': helper.file_sha(__file__), 'producer_config_sha256': producer_sha, 'frozen_sha256': frozen_sha,
                'schedule_sha256': frozen['schedule_sha256'], 'baseline_continuation_sha256': frozen['continuation_sha256'],
                'source_capture_sha256': frozen['source_capture_sha256'], 'original_failed_build_sha256': producer['original_failed_build_sha256'],
                'native_sha256': identity['native_sha256'], 'commands': commands,
                'limits': ['Config derivation only; no native/model execution or raw-logit payload scan.',
                           'Completed producer already hashed/full-row-checked every payload before freezing; replays and comparator recheck their own inputs.',
                           'All raw logits remain VM-side. Original 1e-5 TP failure is preserved; no release or performance qualification.']}
    (output / 'replay-preparation.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-config', required=True, type=Path)
    parser.add_argument('--producer-config-sha256', required=True)
    parser.add_argument('--frozen', required=True, type=Path)
    parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--capture-output-root', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.producer_config, args.producer_config_sha256, args.frozen, args.frozen_sha256,
                     args.output, args.capture_output_root)
    print(json.dumps({'status': result['status'], 'release_qualified': False, 'commands': len(result['commands'])}))
if __name__ == '__main__': main()
