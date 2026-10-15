import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('glm_trace_refusal_tests',
    Path(__file__).parents[1] / 'check-glm-trace-refusal.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Function:
    def __init__(self, call):
        self.call = call
    def __call__(self, *arguments):
        return self.call(*arguments)


class FakeNative:
    """Observer fixture only; never counted as actual native execution."""
    def __init__(self, capture, fault=None, partial=False):
        self.capture, self.fault, self.partial = capture, fault, partial
        self.slots, self.active, self.backend_failed, self.freed = {}, 0, 0, False
        self.calls, self.need_reset, self.error = 0, False, b''
        functions = {
            'GlmLoadModel': self.load, 'GlmFree': self.free, 'GlmForward': self.forward,
            'GlmForwardBatchedDecode': self.joint, 'GlmNPast': lambda _: self.slots[self.active]['head'],
            'GlmRewind': self.rewind, 'GlmResetChecked': self.reset, 'GlmSlotAlloc': self.allocate,
            'GlmSetActiveSlot': self.select, 'GlmSlotFree': self.release,
            'GlmVocabSize': lambda _: 128, 'GetLastError': lambda: self.error,
            'HasBackendFailure': lambda: self.backend_failed,
            'GlmLastBatchedFailureNeedsReset': lambda _: int(self.need_reset),
            'GlmLastBatchedExecution': lambda *args: int(self.fault == 'false-publication'),
        }
        for name, function in functions.items():
            setattr(self, 'TSGgml_' + name, Function(function))

    def load(self, *args):
        self.capture.mkdir()
        self.slots = {0: {'head': 0, 'fenced': False}}
        return 1
    def free(self, _):
        self.freed = True
        self.slots.clear()
    def rewind(self, _, position):
        return int(not self.slots[self.active]['fenced'] and position == 0)
    def reset(self, handle):
        if not handle or self.fault == 'broken-reset' and self.slots[self.active]['fenced']:
            return 0
        self.slots[self.active] = {'head': 0, 'fenced': False}
        return 1
    def allocate(self, _):
        index = max(self.slots) + 1
        self.slots[index] = {'head': 0, 'fenced': False}
        return index if self.fault != 'duplicate-slot' else 1
    def select(self, _, slot):
        self.active = slot
        return int(slot in self.slots)
    def release(self, _, slot):
        if self.fault == 'bad-release':
            return 0
        del self.slots[slot]
        return 1
    def refuse(self, values, slots):
        self.calls += 1
        self.error = b'GLM trace capture failed before publication: GLM trace capture exceeds its configured process-wide byte bound'
        if self.partial and self.calls == 1:
            (self.capture / 'pass-000000.tensor-0000.bin').write_bytes(b'first packed tensor')
            (self.capture / 'pass-000000.incomplete').write_text('incomplete')
            self.error = b'GLM trace capture failed before publication: GLM trace output exists; refusing overwrite'
        if self.fault == 'lost-partial-charge' and self.calls == 2:
            (self.capture / 'pass-000001.incomplete').write_text('incorrect retry')
        if self.fault == 'wrong-error':
            self.error = b'unrelated native failure'
        if self.fault == 'sentinel-write':
            values[0] = 0
        for slot in slots:
            self.slots[slot]['fenced'] = self.fault != 'missing-fence'
            self.slots[slot]['head'] = int(self.fault == 'head-advanced')
        if self.fault == 'global-poison':
            self.backend_failed = 1
        return 0
    def forward(self, handle, tokens, count, values):
        return self.refuse(values, [self.active])
    def joint(self, handle, count, slots, tokens, positions, values):
        self.need_reset = self.fault != 'missing-joint-reset'
        return self.refuse(values, list(slots))


class TraceRefusalObserverTests(unittest.TestCase):
    def observe(self, fault=None, partial=False):
        with tempfile.TemporaryDirectory() as folder:
            native = FakeNative(Path(folder) / 'capture', fault, partial)
            result = M.exercise(native, Path(folder) / 'tiny.gguf', native.capture,
                                mode='partial-write' if partial else 'byte-budget')
            self.assertTrue(native.freed, 'Observer must destroy the model even after a failed check')
            return result

    def test_complete_observer_covers_n1_n3_and_43_publication_lifecycle_checks(self):
        report = self.observe()
        self.assertTrue(report['Passed'], report.get('Error'))
        self.assertEqual(43, len(report['Checks']))
        self.assertTrue(all(row['Passed'] for row in report['Checks']))

    def test_observer_rejects_each_publication_fence_reset_and_lifecycle_regression(self):
        for fault in ('wrong-error', 'sentinel-write', 'head-advanced', 'missing-fence', 'broken-reset',
                      'global-poison', 'duplicate-slot', 'missing-joint-reset', 'false-publication', 'bad-release'):
            with self.subTest(fault=fault):
                report = self.observe(fault)
                self.assertFalse(report['Passed'])
                self.assertTrue(any(not row['Passed'] for row in report['Checks']))
                self.assertIn('Error', report)

    def test_partial_write_retains_payload_and_incomplete_marker_without_new_pass(self):
        report = self.observe(partial=True)
        self.assertTrue(report['Passed'], report.get('Error'))
        self.assertEqual(44, len(report['Checks']))
        self.assertEqual(report['RetainedPartialBeforeRetry'], report['RetainedPartialAfterRetries'])
        self.assertEqual({'pass-000000.tensor-0000.bin', 'pass-000000.incomplete'}, set(report['RetainedPartialBeforeRetry']))

    def test_partial_write_observer_rejects_released_disk_credit_and_new_retry_files(self):
        report = self.observe('lost-partial-charge', partial=True)
        self.assertFalse(report['Passed'])
        self.assertTrue(any('Retained partial' in row['Name'] and not row['Passed'] for row in report['Checks']))


class TraceRefusalInputTests(unittest.TestCase):
    def receipt(self):
        before = {'upstream': {'revision': 'a' * 40, 'status': ''}}
        return {'passed': True, 'no_upstream_compilation': True, 'before': before, 'after': copy.deepcopy(before),
                'production_artifacts': {'GgmlOps.dll': 'b' * 64},
                'source_snapshot': {'TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp': 'c' * 64}}

    def test_receipt_binds_exact_binary_source_clean_revision_and_bookends(self):
        self.assertEqual('a' * 40, M.validate_receipt(self.receipt(), 'GgmlOps.dll', 'b' * 64, 'c' * 64))
        for field, value in [('passed', False), ('passed', 1), ('no_upstream_compilation', False), ('after', {})]:
            item = self.receipt()
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                M.validate_receipt(item, 'GgmlOps.dll', 'b' * 64, 'c' * 64)
        for binary, source in [('d' * 64, 'c' * 64), ('b' * 64, 'd' * 64)]:
            with self.assertRaises(ValueError):
                M.validate_receipt(self.receipt(), 'GgmlOps.dll', binary, source)

    def test_receipt_rejects_dirty_or_malformed_upstream_revision(self):
        for revision, status in [('a' * 41, ''), ('a' * 39, ''), ('g' * 40, ''), (None, ''), ('a' * 40, ' M file')]:
            item = self.receipt()
            item['before']['upstream'].update(revision=revision, status=status)
            item['after'] = copy.deepcopy(item['before'])
            with self.subTest(revision=revision, status=status), self.assertRaises(ValueError):
                M.validate_receipt(item, 'GgmlOps.dll', 'b' * 64, 'c' * 64)

    def test_cpu_environment_excludes_inherited_experiments_and_forces_one_byte(self):
        args = SimpleNamespace(threads=2, output=Path('/capture-parent'), prefill_rows=4,
                               mode='byte-budget', byte_budget=1, library_dir=[])
        with patch.dict(os.environ, {'TS_GGML_Q8_PARALLEL_VECTOR': '1', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE': 'bf16',
                                    'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '64', 'KEEP': 'yes'}, clear=True):
            result = M.environment(args)
        self.assertEqual('-1', result['CUDA_VISIBLE_DEVICES'])
        self.assertEqual('2', result['OMP_NUM_THREADS'])
        self.assertEqual('1', result['TS_GLM_TRACE_DUMP_MAX_BYTES'])
        self.assertNotIn('TS_GGML_Q8_PARALLEL_VECTOR', result)
        self.assertNotIn('GGML_CUDA_CUBLAS_COMPUTE_TYPE', result)
        self.assertEqual('yes', result['KEEP'])

    def test_reference_budget_requires_same_first_pass_and_complete_positive_byte_sum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pass-000000.json'
            good = {'schema': 'tensorsharp.glm-trace.v1', 'kind': 'trunk', 'p0': 0, 'nt': 4,
                    'total_bytes': 12, 'traces': [{'bytes': 4}, {'bytes': 8}]}
            path.write_text(json.dumps(good))
            self.assertEqual(12, M.reference_budget(path, 4))
            for field, value in [('nt', 3), ('kind', 'batched'), ('p0', 1), ('total_bytes', 13), ('total_bytes', True),
                                 ('traces', [{'bytes': 12}]), ('traces', [{'bytes': -4}, {'bytes': 16}])]:
                invalid = good | {field: value}
                path.write_text(json.dumps(invalid))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    M.reference_budget(path, 4)


if __name__ == '__main__':
    unittest.main()
