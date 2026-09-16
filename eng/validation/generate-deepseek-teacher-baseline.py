#!/usr/bin/env python3
"""Unqualified non-TP greedy continuation producer, followed by schedule freeze.

Actual execution loads a large model and requires an exclusive, coordinated VM
window. Import/model-free tests do not load native code. This phase runs BEFORE
teacher replay; candidate logits never choose the teacher continuation.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
import types


def load_helper(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Capture helper pin differs')
    module = types.ModuleType('pinned_teacher_baseline_capture')
    module.__file__ = str(Path(path).resolve())
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


class GreedyCalls:
    """Feed canonical prompt chunks, then at most63 raw greedy tokens.

    capture_rows owns every Forward/status/position/full-row guard. Resuming
    this iterator after yield means that call passed those guards. The final
    prompt row predicts token0; the final chosen EOS/cap token is never fed.
    """
    def __init__(self, helper, plan, plan_hash, identity, exported, token_root, output, report):
        self.helper, self.plan, self.identity = helper, plan, identity
        self.output, self.report = output, report
        self.comparator = helper.load_comparator()
        self.groups, self.requests = [], []
        helper.validate_export_identity(plan, plan_hash, identity, exported)
        inputs = output / 'call-inputs'
        inputs.mkdir()
        self.sequence = 0
        for item in exported['requests']:
            raw = helper.read_pinned(helper.child(token_root, item['prefix'] + '.final.i32'), item['TokensI32Sha256'])
            helper.require(item['RemovedMessages'] == 0 and len(raw) == item['prompt_tokens'] * 4 and raw,
                           'Prompt is empty, compacted or has a different token count')
            values = helper.np.frombuffer(raw, dtype='<i4')
            helper.require(bool(helper.np.all((values >= 0) & (values < identity['tokenizer']['vocab_size']))), 'Prompt token outside vocabulary')
            helper.require(item['prompt_tokens'] + 63 <= plan['protocol']['context'], '64-token continuation cannot fit without changing canonical history')
            group, history = [], b''
            step = plan['variants']['non-tp']['native_load']['n_ubatch'] * 4
            for offset in range(0, len(raw), step):
                part = raw[offset:offset+step]
                group.append(self.row(item['id'], len(group), history, part, 'prompt'))
                history += part
            self.groups.append((item['id'], group, history))
        prompt_schedule = {'schema_version': 1, 'teacher_plan_sha256': plan_hash, 'dtype': 'float32-le',
            'shared_identity': identity, 'rows': [row for _, group, _ in self.groups for row in group],
            'coverage': 'complete-primary', 'capture_scope': helper.SUPPORTED_SCOPE}
        self.comparator.validate_schedule(plan, plan_hash, prompt_schedule)

    def row(self, request, index, history, raw, stage):
        helper = self.helper
        relative = f'call-inputs/{self.sequence:06d}.i32'
        self.sequence += 1
        (self.output / relative).write_bytes(raw)
        before, after = len(history) // 4, (len(history) + len(raw)) // 4
        return {'row_id': f'{request}/primary/{index}', 'request_id': request, 'phase': 'primary', 'slot_id': 0,
                'call_index': index, 'position_before': before, 'position_after': after,
                'predicted_token_position': after, 'input_token_count': len(raw) // 4,
                'vocab_size': self.identity['tokenizer']['vocab_size'], 'stage': stage, 'call_input_file': relative,
                'call_input_i32_sha256': helper.sha(raw), 'cumulative_input_i32_sha256': helper.sha(history + raw),
                'next_forced_token_id': None}

    def __getitem__(self, index):
        if index != 0: raise IndexError(index)
        return self.groups[0][1][0]

    def __iter__(self):
        helper = self.helper
        eos = set(self.identity['tokenizer']['eos_ids'])
        with (self.output / 'greedy-decisions.jsonl').open('x', encoding='utf-8') as decisions:
            for request, group, history in self.groups:
                for row in group:
                    yield row
                tokens = []
                while True:
                    actual = self.report['rows'][-1]
                    helper.require(actual['request_id'] == request and actual['native_status'] == 0,
                                   'Greedy selection requires the current successful baseline row')
                    vector, raw = self.comparator.read_row(self.output / 'rows', actual)
                    helper.require(bool(helper.np.isfinite(vector).all()), 'Nonfinite baseline row cannot select continuation')
                    # np.argmax selects the lowest index when maxima tie. No
                    # temperature, penalties, grammar or token suppression.
                    token = int(helper.np.argmax(vector))
                    actual['next_forced_token_id'] = token
                    actual['baseline_greedy_token_id'] = token
                    tokens.append(token)
                    decisions.write(json.dumps({'request_id': request, 'row_id': actual['row_id'],
                        'token_index': len(tokens)-1, 'token_id': token, 'logits_sha256': helper.sha(raw),
                        'selection': 'raw-f32-argmax-lowest-id'}) + '\n')
                    decisions.flush()
                    if token in eos or len(tokens) == 64:
                        self.requests.append({'id': request, 'tokens': tokens,
                            'stop_reason': 'eos' if token in eos else 'maximum_tokens'})
                        break
                    part = helper.token_bytes([token])
                    row = self.row(request, len(group) + len(tokens)-1, history, part, 'continuation')
                    history += part
                    yield row


def freeze_schedule(helper, plan, plan_hash, identity, report, requests, exported, token_root, output, failed_sha):
    """Freeze only a complete audited non-TP trace, verifying every greedy row."""
    helper.require(report['status'] == 'complete' and report['variant'] == 'non-tp', 'Only completed nonTP baseline can freeze a teacher history')
    helper.require([r['id'] for r in requests] == [r['id'] for r in plan['requests']], 'Baseline request order/coverage differs')
    comparator = helper.load_comparator()
    comparator.payload_layout(output / 'rows', report['rows'])
    expected = []
    for row in report['rows']:
        vector, _ = comparator.read_row(output / 'rows', row)
        helper.require(bool(helper.np.isfinite(vector).all()), 'Baseline raw row has nonfinite coordinates')
        if row['next_forced_token_id'] is not None:
            helper.require(row['next_forced_token_id'] == int(helper.np.argmax(vector)), 'Frozen baseline choice differs from actual full-row greedy argmax')
        expected.append({key: row[key] for key in (*comparator.ROW_FIELDS, 'stage', 'call_input_file')})
    capture_path = output / 'rows/capture.json'
    capture_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    continuation = {'schema_version': 1, 'status': 'complete', 'source_variant': 'non-tp', 'release_qualified': False,
        'maximum_tokens': 64, 'selection': 'raw-f32-argmax-lowest-id', 'native_sha256': identity['native_sha256'],
        'token_export_sha256': identity['token_export_sha256'], 'source_capture_sha256': helper.file_sha(capture_path),
        'producer_sha256': helper.file_sha(__file__), 'requests': requests}
    continuation_path = output / 'baseline-continuation.json'
    continuation_raw = (json.dumps(continuation, indent=2, allow_nan=False) + '\n').encode()
    schedule = {'schema_version': 1, 'teacher_plan_sha256': plan_hash, 'dtype': 'float32-le', 'shared_identity': identity,
        'capture_scope': helper.SUPPORTED_SCOPE, 'coverage': 'complete-primary', 'rows': expected,
        'baseline_continuation_sha256': helper.sha(continuation_raw), 'original_failed_build_sha256': failed_sha,
        'release_qualified': False, 'selection_phase': 'independently recorded nonTP baseline; frozen before teacher replay'}
    helper.validate_serial_schedule(plan, plan_hash, schedule, output, exported, token_root, continuation)
    # Validate the dynamic producer's process and row evidence against the
    # newly frozen schedule. The producer did not claim that schedule existed
    # before it ran; only subsequent teacher captures bind its final digest.
    projected = copy.deepcopy(report)
    projected['schedule_sha256'] = '0' * 64
    comparator.validate_capture(projected, schedule, '0' * 64, plan)
    continuation_path.write_bytes(continuation_raw)
    schedule_path = output / 'teacher-schedule.json'
    schedule_path.write_text(json.dumps(schedule, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {'continuation_sha256': helper.file_sha(continuation_path), 'schedule_sha256': helper.file_sha(schedule_path),
            'source_capture_sha256': helper.file_sha(capture_path), 'requests': len(requests), 'rows': len(expected),
            'raw_f32_bytes': sum(row['vocab_size'] * 4 for row in expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.config_sha256: raise ValueError('Producer config pin differs')
    config = json.loads(raw)
    if not sys.platform.startswith('linux') or sys.byteorder != 'little': raise ValueError('Actual producer requires little-endian Linux')
    helper = load_helper(config['capture_helper_path'], config['capture_helper_sha256'])
    helper.require(config['release_qualified'] is False and config['scope'] == helper.SUPPORTED_SCOPE
                   and config['variant'] == 'non-tp' and config['maximum_tokens'] == 64, 'Explicit unqualified nonTP primary baseline required')
    helper.require(helper.file_sha(__file__) == config['producer_sha256'], 'Producer source pin differs')
    plan = helper.pinned_json(config['plan_path'], config['plan_sha256'])
    identity = helper.pinned_json(config['shared_identity_path'], config['shared_identity_sha256'])
    helper.require(identity['capture_program_sha256'] == config['capture_helper_sha256'], 'Capture-helper identity differs')
    exported = helper.pinned_json(config['token_export_path'], identity['token_export_sha256'])
    failed_raw = helper.validate_failed_gate(config, identity)
    variant = plan['variants']['non-tp']
    helper.require(variant['requested_expert_tp_ranks'] == 0 and variant['environment']['TS_DSV41_TP'] == '0', 'TP cannot select baseline continuation')
    helper.apply_environment(identity, variant)
    helper.require(not args.output.exists(), 'Use a fresh producer output directory')
    args.output.mkdir(parents=True)
    for target, content in [('config.json', raw), ('original-failed-build.json', failed_raw),
        ('plan.json', helper.read_pinned(config['plan_path'], config['plan_sha256'])),
        ('shared-identity.json', helper.read_pinned(config['shared_identity_path'], config['shared_identity_sha256'])),
        ('token-export.json', helper.read_pinned(config['token_export_path'], identity['token_export_sha256'])),
        ('model-attestation.json', helper.read_pinned(config['model_attestation_path'], identity['model_attestation_sha256']))]:
        (args.output / target).write_bytes(content)
    report = {'schema_version': 1, 'status': 'loading', 'phase': 'independent-baseline-continuation', 'release_qualified': False,
        'variant': 'non-tp', 'scope': helper.SUPPORTED_SCOPE, 'shared_identity': identity, 'native_load': variant['native_load'],
        'environment': variant['environment'], 'native_library_path': str(Path(config['native_library']).resolve()),
        'reset_api': config['reset_api'], 'producer_sha256': config['producer_sha256'], 'config_sha256': args.config_sha256,
        'original_failed_build_sha256': config['original_failed_build_sha256'], 'observations': {},
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'limits': ['Primary serialized slot0; no concurrency or HTTP sampler/grammar equivalence.',
                   'Raw greedy baseline, at most64 tokens; EOS/cap token is predicted but never fed.',
                   'Original TP gate remains failed; no release qualification.']}
    calls = GreedyCalls(helper, plan, config['plan_sha256'], identity, exported, Path(config['token_export_path']).parent, args.output, report)
    native = observer = None
    def save():
        (args.output / 'lifecycle.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    save()
    try:
        helper.require(helper.file_sha(config['native_library']) == identity['native_sha256'], 'Native source pin differs before load')
        with helper.native_stderr(args.output / 'native.log'):
            observer = helper.Observer(config, {'shared_identity': identity})
            native = helper.Native(config['native_library'], config['reset_api'])
            report['observations']['before'] = observer.audit()
            native.load(Path(plan['model']['directory']) / plan['model']['files'][0]['path'], variant['native_load'])
            report['placement'] = helper.parse_placement((args.output / 'native.log').read_text(errors='replace'), variant, config['model_layers'])
            helper.capture_rows(native, calls, args.output, args.output / 'rows', report)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if observer is not None:
            try:
                report['observations']['after'] = observer.audit()
                helper.require(helper.file_sha(__file__) == config['producer_sha256'], 'Producer source changed')
                helper.require(report['observations']['before']['computation_environment'] == report['observations']['after']['computation_environment'], 'Computation environment changed')
            except Exception as error:
                report.update(status='failed', after_audit_error=repr(error))
        try:
            if native is not None: native.free()
        except Exception as error:
            report.update(status='failed', cleanup_error=repr(error))
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if (args.output / 'rows').exists():
            (args.output / 'rows/capture.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        save()
    helper.require(report['status'] == 'complete', 'Baseline source/cleanup gate failed; no teacher schedule frozen')
    frozen = freeze_schedule(helper, plan, config['plan_sha256'], identity, report, calls.requests, exported,
                              Path(config['token_export_path']).parent, args.output, config['original_failed_build_sha256'])
    (args.output / 'frozen.json').write_text(json.dumps({'status': 'frozen', 'release_qualified': False, **frozen}, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'frozen', 'release_qualified': False, **frozen}))


if __name__ == '__main__': main()
