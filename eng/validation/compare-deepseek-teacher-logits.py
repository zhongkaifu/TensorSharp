#!/usr/bin/env python3
"""Compare complete F32 vocabulary captures against an independently pinned schedule.

This reads artifacts only. It never loads a model or changes the existing TP
gate. KL/RMS are descriptive; the 2e-5 allclose check is a predeclared diagnostic,
not a full-model release contract. See teacher-logits-protocol.md.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

ATOL = RTOL = 2e-5
ROW_FIELDS = ('row_id', 'request_id', 'phase', 'slot_id', 'call_index', 'position_before',
              'input_token_count', 'position_after', 'predicted_token_position',
              'cumulative_input_i32_sha256', 'call_input_i32_sha256', 'next_forced_token_id', 'vocab_size')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_value(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Missing lowercase SHA256')


def pinned_json(path, expected):
    hash_value(expected)
    raw = path.read_bytes()
    require(sha(raw) == expected, 'Pinned JSON changed: ' + str(path))
    return json.loads(raw)


def validate_schedule(plan, plan_hash, schedule):
    require(schedule['schema_version'] == 1 and schedule['teacher_plan_sha256'] == plan_hash, 'Wrong plan/schedule schema or pin')
    require(schedule['dtype'] == 'float32-le', 'Only raw little-endian full-vocabulary F32 rows are supported')
    identity = schedule['shared_identity']
    native_hash = identity['native_sha256']
    hash_value(native_hash)
    require(all(value['expected_native_sha256'] == native_hash for value in plan['variants'].values()), 'Variant native pins differ')
    off, on = plan['variants']['non-tp'], plan['variants']['expert-tp7']
    require(off['requested_expert_tp_ranks'] == 0 and on['requested_expert_tp_ranks'] == 7, 'Wrong planned TP degrees')
    require(off['native_load'] == on['native_load'], 'Planned native load settings differ')
    off_env, on_env = dict(off['environment']), dict(on['environment'])
    require(off_env.pop('TS_DSV41_TP') == '0' and on_env.pop('TS_DSV41_TP') == '7' and off_env == on_env,
            'Planned computation environments differ beyond TP')
    require(identity['model_manifest_sha256'] == plan['model']['manifest_sha256'], 'Wrong checkpoint manifest')
    for name in ('native_source_sha256', 'exporter_source_sha256', 'exporter_assemblies_sha256'):
        require(identity.get(name), 'Missing identity set: ' + name)
        for value in identity[name].values():
            hash_value(value)
    for name in ('token_export_sha256', 'capture_program_sha256'):
        hash_value(identity[name])
    files = identity['checkpoint_files']
    expected_files = {str(PurePosixPath(plan['model']['directory']) / item['path']):
                      {'bytes': item['size'], 'sha256': item['expected_sha256']} for item in plan['model']['files']}
    require(files == expected_files, 'Checkpoint file identities differ from the pinned teacher plan')
    require(identity.get('engram_files'), 'Engram sidecar identities are required')
    for item in identity['engram_files'].values():
        hash_value(item['sha256'])
        require(type(item['bytes']) is int and item['bytes'] > 0, 'Invalid Engram byte length')
    tokenizer = identity['tokenizer']
    require(type(tokenizer['vocab_size']) is int and tokenizer['vocab_size'] > 1, 'Invalid vocabulary')
    require(tokenizer['first_gguf_sha256'] == plan['model']['files'][0]['expected_sha256'], 'Tokenizer GGUF differs')
    require(tokenizer['eos_ids'] and len(set(tokenizer['eos_ids'])) == len(tokenizer['eos_ids']), 'Missing/duplicate EOS IDs')
    require(all(type(token) is int and 0 <= token < tokenizer['vocab_size'] for token in tokenizer['eos_ids']), 'Invalid EOS ID')
    gate = plan['gates']['existing_fixture_allclose_diagnostic']
    require(gate['atol'] == ATOL and gate['rtol'] == RTOL, 'Diagnostic tolerance changed')
    rows = schedule['rows']
    require(rows and len({row['row_id'] for row in rows}) == len(rows), 'Empty/duplicate scheduled rows')
    request_ids = {row['id'] for row in plan['requests']}
    require(len(request_ids) == plan['request_count'], 'Invalid teacher request identity set')
    for row in rows:
        require(all(name in row for name in ROW_FIELDS), 'Incomplete expected row metadata')
        require(row['request_id'] in request_ids, 'Unknown scheduled request')
        require(row['vocab_size'] == tokenizer['vocab_size'], 'Scheduled vocabulary differs')
        for field in ('slot_id', 'call_index', 'position_before', 'input_token_count', 'position_after', 'predicted_token_position'):
            require(type(row[field]) is int and row[field] >= 0, 'Invalid position/slot metadata: ' + field)
        require(row['input_token_count'] > 0 and row['position_after'] == row['position_before'] + row['input_token_count'], 'Inconsistent call positions')
        require(row['position_after'] <= plan['protocol']['context'], 'Scheduled position exceeds context')
        require(row['predicted_token_position'] == row['position_after'], 'Last-position prediction off by one')
        for field in ('cumulative_input_i32_sha256', 'call_input_i32_sha256'):
            hash_value(row[field])
        forced = row['next_forced_token_id']
        require(forced is None or type(forced) is int and 0 <= forced < row['vocab_size'], 'Forced token outside vocabulary')
    coverage = schedule['coverage']
    require(coverage in ('complete-primary', 'explicit-subset'), 'Missing schedule coverage declaration')
    if coverage == 'complete-primary':
        require({row['request_id'] for row in rows if row['phase'] == 'primary'} == request_ids,
                'Complete-primary schedule omits requests')
    else:
        require(schedule.get('subset_reason'), 'Subset cannot be silently described as full coverage')
    return rows


def validate_capture(capture, schedule, schedule_hash, plan):
    require(capture['schema_version'] == 1 and capture['status'] == 'complete', 'Capture is incomplete or failed')
    require(capture['schedule_sha256'] == schedule_hash, 'Capture belongs to a different frozen schedule')
    identity = schedule['shared_identity']
    require(capture['shared_identity'] == identity, 'Capture source/native/checkpoint/token identity differs')
    variant = plan['variants'][capture['variant']]
    require(capture['native_load'] == variant['native_load'] and capture['environment'] == variant['environment'], 'Capture computation settings differ')
    placement = capture['placement']
    require(placement['effective_tp_ranks'] == variant['requested_expert_tp_ranks'], 'Requested TP was declined or fell back')
    require(placement['effective_gpu_count'] == variant['native_load']['n_gpu'] and
            placement['effective_cpu_moe_layers'] == variant['native_load']['n_cpu_moe'], 'Effective placement differs')
    require(placement.get('layer_placement_evidence') and
            (variant['requested_expert_tp_ranks'] == 0 or placement.get('rank_shard_evidence')), 'Missing actual placement evidence')
    before, after = capture['observations']['before'], capture['observations']['after']
    require(type(before['pid']) is int and before['pid'] > 0 and before['start_ticks'] > 0, 'Missing actual process identity')
    require((before['pid'], before['start_ticks'], before['boot_id']) ==
            (after['pid'], after['start_ticks'], after['boot_id']), 'Capture process identity changed')
    library = capture['native_library_path']
    for observation in (before, after):
        require(observation['mapped_native_libraries'].get(library) == identity['native_sha256'], 'Actual native mapping does not match pin')
        require(observation['native_source_sha256'] == identity['native_source_sha256'], 'Native source audit differs')
        require(observation['checkpoint_files'] == identity['checkpoint_files'] and observation['engram_files'] == identity['engram_files'], 'Model file audit differs')
        require(observation['token_export_sha256'] == identity['token_export_sha256'], 'Exported tokens changed during capture')
    require(before['mapped_native_libraries'] == after['mapped_native_libraries'], 'Native mapping set changed during capture')
    rows = capture['rows']
    require(len(rows) == len(schedule['rows']), 'Capture row count differs; partial intersection is forbidden')
    for actual, expected in zip(rows, schedule['rows']):
        require({key: actual[key] for key in ROW_FIELDS} == {key: expected[key] for key in ROW_FIELDS},
                'Input/position/order metadata differs for ' + expected['row_id'])
        require(type(actual['native_status']) is int and actual['native_status'] == 0, 'Native row failed')
        hash_value(actual['logits_sha256'])
    return rows


def payload_layout(root, rows):
    files = {}
    for row in rows:
        relative = Path(row['logits_file'])
        require(not relative.is_absolute(), 'Logit artifact must be relative to capture manifest')
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), 'Logit artifact escapes capture directory')
        offset, size = row['byte_offset'], row['vocab_size'] * 4
        require(type(offset) is int and offset >= 0 and offset % 4 == 0, 'Invalid raw F32 offset')
        files.setdefault(path, []).append((offset, size))
    for path, spans in files.items():
        end = 0
        for offset, size in sorted(spans):
            require(offset == end, 'Overlapping/omitted raw logit bytes: ' + str(path))
            end = offset + size
        require(path.stat().st_size == end, 'Full-vocabulary payload is truncated or has unclaimed bytes')


def read_row(root, row):
    with (root / row['logits_file']).open('rb') as stream:
        stream.seek(row['byte_offset'])
        raw = stream.read(row['vocab_size'] * 4)
    require(len(raw) == row['vocab_size'] * 4 and sha(raw) == row['logits_sha256'], 'Raw full-vocabulary row changed')
    return np.frombuffer(raw, dtype='<f4'), raw


def log_softmax(value):
    shifted = value - np.max(value)
    return shifted - np.log(np.exp(shifted).sum(dtype=np.float64))


def compare_vectors(reference, candidate, forced=None):
    require(reference.ndim == candidate.ndim == 1 and reference.shape == candidate.shape and reference.size > 1, 'Full vector shape differs')
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        return {'finite': False, 'nonfinite_coordinates': np.flatnonzero(~finite).tolist(),
                'allclose_diagnostic_passed': False, 'exact_greedy_parity': False}, []
    difference = b - a
    absolute = np.abs(difference)
    allowed = ATOL + RTOL * np.abs(a)
    failing = np.flatnonzero(absolute > allowed)
    norm = float(np.linalg.norm(a))
    distance = float(np.linalg.norm(difference))
    centered = difference - difference.mean()
    log_a, log_b = log_softmax(a), log_softmax(b)
    prob_a, prob_b = np.exp(log_a), np.exp(log_b)
    mixture = np.logaddexp(log_a, log_b) - np.log(2.0)
    winner_a, winner_b = int(np.argmax(a)), int(np.argmax(b))
    max_index = int(np.argmax(absolute))
    guard_index = int(np.argmax(absolute / allowed))
    def margin(value):
        largest = np.partition(value, -2)[-2:]
        return float(largest.max() - largest.min())
    metrics = {'finite': True, 'max_absolute': float(absolute[max_index]), 'max_absolute_coordinate': max_index,
        'rms': float(np.sqrt(np.mean(difference * difference))), 'relative_l2': distance / norm if norm else (0.0 if not distance else None),
        'reference_l2_zero': norm == 0, 'difference_l2': distance, 'mean_logit_shift': float(difference.mean()),
        'centered_rms': float(np.sqrt(np.mean(centered * centered))), 'centered_max_absolute': float(np.max(np.abs(centered))),
        'kl_reference_to_candidate': float(np.dot(prob_a, log_a - log_b)),
        'kl_candidate_to_reference': float(np.dot(prob_b, log_b - log_a)),
        'jensen_shannon': float((np.dot(prob_a, log_a - mixture) + np.dot(prob_b, log_b - mixture)) / 2),
        'reference_argmax': winner_a, 'candidate_argmax': winner_b, 'exact_greedy_parity': winner_a == winner_b,
        'reference_top_two_margin': margin(a), 'candidate_top_two_margin': margin(b),
        'reference_winner_rank_in_candidate': 1 + int(np.count_nonzero(b > b[winner_a])),
        'candidate_winner_rank_in_reference': 1 + int(np.count_nonzero(a > a[winner_b])),
        'allclose_diagnostic_passed': len(failing) == 0, 'allclose_violations': len(failing),
        'worst_guard_coordinate': guard_index, 'worst_guard_ratio': float(absolute[guard_index] / allowed[guard_index])}
    if forced is not None:
        require(type(forced) is int and 0 <= forced < len(a), 'Forced token outside vocabulary')
        metrics.update(forced_token_id=forced, reference_forced_nll=float(-log_a[forced]),
                       candidate_forced_nll=float(-log_b[forced]), forced_nll_delta=float(log_a[forced] - log_b[forced]))
    coordinates = [{'coordinate': int(i), 'reference': float(a[i]), 'candidate': float(b[i]),
                    'absolute_difference': float(absolute[i]), 'allowed_difference': float(allowed[i])} for i in failing]
    return metrics, coordinates


def compare_artifacts(plan, plan_hash, schedule, schedule_hash, reference, candidate, reference_root, candidate_root, output):
    validate_schedule(plan, plan_hash, schedule)
    arows = validate_capture(reference, schedule, schedule_hash, plan)
    brows = validate_capture(candidate, schedule, schedule_hash, plan)
    payload_layout(reference_root, arows)
    payload_layout(candidate_root, brows)
    require(not output.exists(), 'Use a fresh comparison output')
    output.mkdir(parents=True)
    report = {'schema_version': 1, 'status': 'running', 'release_qualified': False,
              'teacher_plan_sha256': plan_hash, 'schedule_sha256': schedule_hash,
              'scope': 'Full-vocabulary numerical diagnostic; no throughput or model-task qualification.',
              'coverage': schedule['coverage'], 'reference_variant': reference['variant'], 'candidate_variant': candidate['variant'],
              'scripted_fixture': reference.get('scripted_fixture', False) or candidate.get('scripted_fixture', False),
              'comparator_sha256': sha(Path(__file__).read_bytes()), 'numpy_version': np.__version__,
              'atol': ATOL, 'rtol': RTOL, 'kl_rms_thresholds': None, 'native_tp_gate': 'Original failed 1e-5 gate remains unchanged.',
              'rows': []}
    try:
        with (output / 'coordinate-failures.jsonl').open('w', encoding='utf-8') as failures:
            for line, (arow, brow) in enumerate(zip(arows, brows), 1):
                avector, araw = read_row(reference_root, arow)
                bvector, braw = read_row(candidate_root, brow)
                metrics, coordinates = compare_vectors(avector, bvector, arow['next_forced_token_id'])
                metrics['bitwise_equal'] = araw == braw
                metrics['coordinate_failures_line'] = line
                failures.write(json.dumps({'row_id': arow['row_id'], 'violations': coordinates,
                    'nonfinite_coordinates': metrics.get('nonfinite_coordinates', [])}, allow_nan=False) + '\n')
                report['rows'].append({**{key: arow[key] for key in ROW_FIELDS}, **metrics})
        payload_layout(reference_root, arows)
        payload_layout(candidate_root, brows)
        report['all_finite'] = all(row['finite'] for row in report['rows'])
        report['bitwise_equal'] = all(row['bitwise_equal'] for row in report['rows'])
        report['allclose_diagnostic_passed'] = all(row['allclose_diagnostic_passed'] for row in report['rows'])
        report['exact_greedy_parity'] = all(row['exact_greedy_parity'] for row in report['rows'])
        report['compared_rows'] = len(report['rows'])
        report['status'] = 'passed-predeclared-diagnostic-only' if report['all_finite'] and report['allclose_diagnostic_passed'] and report['exact_greedy_parity'] else 'failed-diagnostic-or-greedy-parity'
        report['request_summary'] = {}
        for request_id in dict.fromkeys(row['request_id'] for row in report['rows']):
            rows = [row for row in report['rows'] if row['request_id'] == request_id]
            report['request_summary'][request_id] = {'rows': len(rows), 'nonfinite_rows': sum(not row['finite'] for row in rows),
                'allclose_failed_rows': sum(not row['allclose_diagnostic_passed'] for row in rows),
                'argmax_mismatch_rows': sum(not row['exact_greedy_parity'] for row in rows)}
        report['distributions'] = {}
        for metric in ('max_absolute', 'rms', 'relative_l2', 'centered_rms', 'kl_reference_to_candidate', 'kl_candidate_to_reference', 'jensen_shannon'):
            values = [(row[metric], row['row_id']) for row in report['rows'] if row.get(metric) is not None]
            if values:
                report['distributions'][metric] = {'percentiles': dict(zip(('0', '50', '90', '99', '100'),
                    map(float, np.percentile([value for value, _ in values], [0, 50, 90, 99, 100])))),
                    'worst_row_id': max(values)[1]}
        report['coordinate_failures_sha256'] = sha((output / 'coordinate-failures.jsonl').read_bytes())
    except Exception as error:
        report.update(status='failed-incomplete-comparison', error=repr(error))
        raise
    finally:
        (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'schedule', 'reference', 'candidate'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    documents = {name: pinned_json(getattr(args, name), getattr(args, name + '_sha256'))
                 for name in ('plan', 'schedule', 'reference', 'candidate')}
    result = compare_artifacts(documents['plan'], args.plan_sha256, documents['schedule'], args.schedule_sha256,
        documents['reference'], documents['candidate'], args.reference.parent, args.candidate.parent, args.output)
    try:
        for name in documents:
            pinned_json(getattr(args, name), getattr(args, name + '_sha256'))
    except Exception as error:
        result.update(status='failed-input-identity-after-comparison', error=repr(error))
        (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise
    result['input_manifest_sha256'] = {name: getattr(args, name + '_sha256') for name in documents}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result.get(key) for key in ('status', 'compared_rows', 'all_finite', 'allclose_diagnostic_passed', 'exact_greedy_parity', 'release_qualified')}))
    return int(result['status'] != 'passed-predeclared-diagnostic-only')


if __name__ == '__main__':
    raise SystemExit(main())
