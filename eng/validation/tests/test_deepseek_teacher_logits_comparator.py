"""Model-free full-vector and immutable capture-contract checks; no native load."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('teacher_compare', Path(__file__).parents[1] / 'compare-deepseek-teacher-logits.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def fixture(directory, second=None):
    native = 'a' * 64
    plan_hash, schedule_hash = 'b' * 64, 'c' * 64
    model = {'directory': '/models', 'manifest_sha256': 'd' * 64,
             'files': [{'path': 'model.gguf', 'size': 123, 'expected_sha256': 'e' * 64}]}
    load = {'backend': 'CUDA', 'n_gpu': 7, 'n_ctx': 65536, 'n_ubatch': 512, 'n_threads': 32, 'n_cpu_moe': 12}
    plan = {'variants': {name: {'expected_native_sha256': native, 'native_load': load,
             'environment': {'TS_DSV41_TP': str(degree), 'TS_DSV4_FA': '1', 'TS_DSV41_SPARSE_FA': '1'},
             'requested_expert_tp_ranks': degree} for name, degree in [('non-tp', 0), ('expert-tp7', 7)]},
            'model': model, 'request_count': 1, 'requests': [{'id': 'request0'}], 'protocol': {'context': 65536},
            'gates': {'existing_fixture_allclose_diagnostic': {'atol': 2e-5, 'rtol': 2e-5}}}
    identity = {'native_sha256': native, 'native_source_sha256': {'owned.cpp': 'f' * 64},
        'exporter_source_sha256': {'TokenExporter.cs': '1' * 64}, 'exporter_assemblies_sha256': {'Exporter.dll': '2' * 64},
        'token_export_sha256': '3' * 64, 'capture_program_sha256': '4' * 64,
        'model_manifest_sha256': model['manifest_sha256'],
        'checkpoint_files': {'/models/model.gguf': {'bytes': 123, 'sha256': 'e' * 64}},
        'engram_files': {'/models/engram.bin': {'bytes': 123, 'sha256': '5' * 64}},
        'tokenizer': {'first_gguf_sha256': 'e' * 64, 'vocab_size': 4, 'eos_ids': [3]}}
    rows = [{'row_id': f'request0/primary/{i}', 'request_id': 'request0', 'phase': 'primary',
             'slot_id': 0, 'call_index': i, 'position_before': i, 'input_token_count': 1,
             'position_after': i + 1, 'predicted_token_position': i + 1,
             'cumulative_input_i32_sha256': str(i + 6) * 64, 'call_input_i32_sha256': str(i + 6) * 64,
             'next_forced_token_id': 1 if i == 0 else None, 'vocab_size': 4} for i in range(2)]
    schedule = {'schema_version': 1, 'teacher_plan_sha256': plan_hash, 'dtype': 'float32-le',
                'shared_identity': identity, 'rows': rows, 'coverage': 'complete-primary'}
    observation = {'pid': 123, 'start_ticks': 456, 'boot_id': 'offline-fixture-boot',
                   'mapped_native_libraries': {'/app/libGgmlOps.so': native},
                   **{name: identity[name] for name in ('native_source_sha256', 'checkpoint_files', 'engram_files', 'token_export_sha256')}}
    first = np.array([[0, 2, -1, 0.5], [0.1, 3, 0, -2]], dtype='<f4')
    captures = []
    for index, (variant, vectors) in enumerate([('non-tp', first), ('expert-tp7', first if second is None else second)]):
        root = directory / str(index)
        root.mkdir()
        vectors = np.asarray(vectors, dtype='<f4')
        (root / 'logits.f32').write_bytes(vectors.tobytes())
        capture_rows = [{**row, 'logits_file': 'logits.f32', 'byte_offset': i * 16,
                         'logits_sha256': hashlib.sha256(vectors[i].tobytes()).hexdigest(), 'native_status': 0}
                        for i, row in enumerate(rows)]
        capture = {'schema_version': 1, 'status': 'complete', 'schedule_sha256': schedule_hash,
            'shared_identity': copy.deepcopy(identity), 'variant': variant,
            'native_load': copy.deepcopy(load), 'environment': copy.deepcopy(plan['variants'][variant]['environment']),
            'placement': {'effective_tp_ranks': index * 7, 'effective_gpu_count': 7, 'effective_cpu_moe_layers': 12,
                          'layer_placement_evidence': ['SCRIPTED test-only placement'], 'rank_shard_evidence': ['SCRIPTED test-only shards']},
            'observations': {'before': copy.deepcopy(observation), 'after': copy.deepcopy(observation)},
            'native_library_path': '/app/libGgmlOps.so', 'rows': capture_rows, 'scripted_fixture': True}
        captures.append(capture)
    return plan, plan_hash, schedule, schedule_hash, *captures, directory / '0', directory / '1', directory / 'output'


class TeacherVectorTests(unittest.TestCase):
    def test_identical_and_forced_token_metrics(self):
        values = np.array([0, 2, -3, 0.5], dtype=np.float32)
        result, failed = compare.compare_vectors(values, values, forced=2)
        self.assertTrue(result['allclose_diagnostic_passed'])
        self.assertTrue(result['exact_greedy_parity'])
        self.assertEqual(result['kl_reference_to_candidate'], 0)
        self.assertEqual(result['forced_nll_delta'], 0)
        self.assertEqual(failed, [])

    def test_common_shift_keeps_probability_metrics_but_fails_raw_guard(self):
        values = np.array([0, 2, -3, 0.5], dtype=np.float32)
        result, failed = compare.compare_vectors(values, values + 100)
        self.assertFalse(result['allclose_diagnostic_passed'])
        self.assertEqual(len(failed), 4)
        self.assertEqual(result['centered_rms'], 0)
        self.assertAlmostEqual(result['kl_reference_to_candidate'], 0, places=14)

    def test_equal_rms_does_not_hide_permutation_or_greedy_mismatch(self):
        a, b = np.array([1, -1], dtype=np.float32), np.array([-1, 1], dtype=np.float32)
        self.assertEqual(np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b)))
        result, failures = compare.compare_vectors(a, b)
        self.assertFalse(result['exact_greedy_parity'])
        self.assertGreater(result['kl_reference_to_candidate'], 1)
        self.assertEqual([x['coordinate'] for x in failures], [0, 1])

    def test_nonfinite_coordinate_is_a_failure_without_invalid_json_numbers(self):
        result, failures = compare.compare_vectors(np.array([0, 1]), np.array([0, np.nan]))
        self.assertFalse(result['finite'])
        self.assertEqual(result['nonfinite_coordinates'], [1])
        json.dumps(result, allow_nan=False)

    def test_large_logits_and_zero_reference_norm(self):
        a = np.array([3e38, -3e38, 0], dtype=np.float32)
        result, _ = compare.compare_vectors(a, a, forced=1)
        self.assertTrue(np.isfinite(result['reference_forced_nll']))
        self.assertEqual(result['jensen_shannon'], 0)
        result, _ = compare.compare_vectors(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        self.assertIsNone(result['relative_l2'])
        self.assertTrue(result['reference_l2_zero'])
        self.assertFalse(result['allclose_diagnostic_passed'])

    def test_near_tie_argmax_is_not_erased_by_passing_allclose(self):
        a = np.array([1, 1 + 1e-6, 0], dtype=np.float32)
        b = a[[1, 0, 2]]
        result, _ = compare.compare_vectors(a, b)
        self.assertTrue(result['allclose_diagnostic_passed'])
        self.assertFalse(result['exact_greedy_parity'])
        self.assertEqual(result['reference_winner_rank_in_candidate'], 2)


class TeacherArtifactTests(unittest.TestCase):
    def test_complete_identity_bound_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            result = compare.compare_artifacts(*args)
            self.assertEqual(result['compared_rows'], 2)
            self.assertTrue(result['bitwise_equal'])
            self.assertFalse(result['release_qualified'])
            self.assertTrue(result['scripted_fixture'])
            self.assertEqual(len(result['request_summary']), 1)

    def test_bitwise_signed_zero_is_reported_without_numerical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = np.array([[-0.0, 2, -1, 0.5], [0.1, 3, 0, -2]], dtype=np.float32)
            result = compare.compare_artifacts(*fixture(Path(directory), changed))
            self.assertFalse(result['bitwise_equal'])
            self.assertTrue(result['allclose_diagnostic_passed'])
            self.assertTrue(result['exact_greedy_parity'])

    def test_manifest_differences_and_partial_intersection_are_rejected(self):
        mutations = {
            'row-count': lambda c: c['rows'].pop(),
            'history': lambda c: c['rows'][0].update(cumulative_input_i32_sha256='0' * 64),
            'position': lambda c: c['rows'][0].update(position_after=2),
            'order': lambda c: c['rows'].reverse(),
            'native': lambda c: c['shared_identity'].update(native_sha256='0' * 64),
            'native-source': lambda c: c['observations']['after']['native_source_sha256'].update(changed='0' * 64),
            'checkpoint': lambda c: c['shared_identity']['checkpoint_files']['/models/model.gguf'].update(bytes=124),
            'status': lambda c: c['rows'][0].update(native_status=-1),
            'fallback': lambda c: c['placement'].update(effective_tp_ranks=0),
            'pid': lambda c: c['observations']['after'].update(start_ticks=999),
            'settings': lambda c: c['environment'].update(TS_DSV4_FA='0'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                args = list(fixture(Path(directory)))
                mutate(args[5])
                with self.assertRaises(ValueError):
                    compare.compare_artifacts(*args)
                self.assertFalse(args[-1].exists())

    def test_schedule_position_forced_coordinate_and_missing_requests_rejected(self):
        for mutation in ('prediction', 'forced', 'duplicate', 'missing-request', 'tolerance'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args = list(fixture(Path(directory)))
                if mutation == 'prediction':
                    args[2]['rows'][0]['predicted_token_position'] = 0
                elif mutation == 'forced':
                    args[2]['rows'][0]['next_forced_token_id'] = 4
                elif mutation == 'duplicate':
                    args[2]['rows'][1]['row_id'] = args[2]['rows'][0]['row_id']
                elif mutation == 'missing-request':
                    args[0]['requests'].append({'id': 'request1'}); args[0]['request_count'] = 2
                else:
                    args[0]['gates']['existing_fixture_allclose_diagnostic']['atol'] = 1
                with self.assertRaises(ValueError):
                    compare.compare_artifacts(*args)

    def test_truncated_payload_and_overlapping_rows_rejected(self):
        for mutation in ('truncate', 'overlap', 'escape'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args = list(fixture(Path(directory)))
                if mutation == 'truncate':
                    (args[7] / 'logits.f32').write_bytes(b'\0' * 31)
                elif mutation == 'overlap':
                    args[5]['rows'][1]['byte_offset'] = 0
                else:
                    args[5]['rows'][0]['logits_file'] = '../outside.f32'
                with self.assertRaises(ValueError):
                    compare.compare_artifacts(*args)

    def test_coordinate_guard_failures_remain_failed_in_full_report(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = np.array([[0, 2, -1, 0.5], [0.1, 3, 0, 9]], dtype=np.float32)
            args = fixture(Path(directory), changed)
            result = compare.compare_artifacts(*args)
            self.assertEqual(result['status'], 'failed-diagnostic-or-greedy-parity')
            self.assertFalse(result['exact_greedy_parity'])
            failures = [json.loads(line) for line in (args[-1] / 'coordinate-failures.jsonl').read_text().splitlines()]
            self.assertEqual(failures[0]['violations'], [])
            self.assertEqual(failures[1]['violations'][0]['coordinate'], 3)
            self.assertFalse(result['release_qualified'])

    def test_actual_nonfinite_row_not_hidden_by_finite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = np.array([[0, 2, -1, 0.5], [0.1, 3, 0, np.inf]], dtype=np.float32)
            args = fixture(Path(directory), changed)
            result = compare.compare_artifacts(*args)
            self.assertFalse(result['all_finite'])
            self.assertEqual(result['rows'][1]['nonfinite_coordinates'], [3])
            json.dumps(result, allow_nan=False)

    def test_raw_row_hash_mismatch_preserves_incomplete_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            args[5]['rows'][1]['logits_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'row changed'):
                compare.compare_artifacts(*args)
            result = json.loads((args[-1] / 'comparison.json').read_text())
            self.assertEqual(result['status'], 'failed-incomplete-comparison')
            self.assertEqual(len(result['rows']), 1)


if __name__ == '__main__':
    unittest.main()
