"""Qualification evidence gates without WSL, native loading or a model process."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'run-wsl-cuda-qualification.py'
SPEC = importlib.util.spec_from_file_location('wsl_qualification', PATH)
Q = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(Q)


class QualificationGateTests(unittest.TestCase):
    def valid(self):
        return dict(Executed=True, Error=None, NativeShutdown=True, Mode='adaptive', ModelSha256='a' * 64,
            Native=[dict(Sha256='b' * 64)], AfterDispose=[dict(Reserved=0, Committed=0)], Records=[
                dict(Run=n, Warmup=n < 0, DecodeCalls=63, Generated=[1] * 64, Before=dict(PhysicalPlan={})) for n in [-1, 0]])

    def test_incomplete_owner_or_identity_evidence_cannot_qualify(self):
        Q.adaptive_gate(self.valid(), 'adaptive', 1, 'a' * 64, 'b' * 64)
        for mutation in (
            lambda r: r.update(NativeShutdown=False), lambda r: r.update(Error='cleanup failed'),
            lambda r: r.update(Native=[]), lambda r: r['Native'][0].update(Sha256='c' * 64),
            lambda r: r['Records'].pop(), lambda r: r['Records'][1].update(Generated=[1] * 63),
            lambda r: r['Records'][1]['Before'].update(PhysicalPlan=None),
            lambda r: r['AfterDispose'][0].update(Committed=1), lambda r: r.update(AfterDispose=[])):
            report = self.valid(); mutation(report)
            with self.assertRaises(ValueError):
                Q.adaptive_gate(report, 'adaptive', 1, 'a' * 64, 'b' * 64)

    def test_environment_is_clean_before_child_and_restored(self):
        old = dict(PATH='preserved', TS_SPEC='1', GGML_CUDA_DISABLE_GRAPHS='1', TENSORSHARP_TP_DEGREE='8',
                   TS_GLM_MOE_MMAP='1', NVIDIA_TF32_OVERRIDE='0')
        with mock.patch.dict(os.environ, old, clear=True):
            with Q.environment(Path('/mnt/e/frozen'), dict(TS_CPU_MOE_THREADS='2')):
                self.assertNotIn('TS_SPEC', os.environ); self.assertNotIn('GGML_CUDA_DISABLE_GRAPHS', os.environ)
                self.assertNotIn('TS_GLM_MOE_MMAP', os.environ); self.assertNotIn('NVIDIA_TF32_OVERRIDE', os.environ)
                self.assertEqual('1', os.environ['TENSORSHARP_TP_DEGREE'])
                self.assertEqual('2', os.environ['TS_CPU_MOE_THREADS']); self.assertEqual('preserved', os.environ['PATH'])
            self.assertEqual(old, dict(os.environ))

    def test_missing_or_wrong_uploader_pools_cannot_qualify(self):
        valid = dict(passed=True, full_logits_bit_exact=True, logit_elements=512, refusals=1,
                     native=[dict(sha256='b' * 64)], final_budget=[
                         dict(Pool=name, Reserved=0, Committed=0) for name in ('host', 'device')])
        Q.upload_gate(valid, 'b' * 64)
        for pools in (None, [], [dict(Pool='host', Reserved=0, Committed=0)],
                      [dict(Pool='host', Reserved=0, Committed=0)] * 2,
                      [dict(Pool='host', Reserved=0, Committed=0), dict(Pool='device', Reserved=1, Committed=0)]):
            with self.subTest(pools=pools), self.assertRaises(ValueError):
                Q.upload_gate(dict(valid, final_budget=pools), 'b' * 64)

    def test_performance_requires_both_metrics_within_five_percent(self):
        for prefill, decode, expected in ((1, .95, True), (.94999, 2, False), (2, .94, False)):
            result = Q.performance_gate(dict(AdaptiveToResidentRatio=dict(
                PrefillTokensPerSecond=prefill, DecodeTokensPerSecond=decode)))
            self.assertIs(expected, result['passed'])
        for ratios in ({}, dict(PrefillTokensPerSecond=1),
                       dict(PrefillTokensPerSecond=float('nan'), DecodeTokensPerSecond=1)):
            with self.assertRaises(ValueError):
                Q.performance_gate(dict(AdaptiveToResidentRatio=ratios))

    def test_teacher_gate_binds_content_hash_and_consumed_history(self):
        report = dict(Generation='teacher-forced', Teacher=[1, 2, 3], TeacherSha256='A' * 64,
                      Records=[dict(Consumed=[1, 2])])
        Q.teacher_gate(report, [1, 2, 3], 'a' * 64)
        for mutation in (dict(Generation='raw-greedy'), dict(Teacher=[1, 2, 4]),
                         dict(TeacherSha256='b' * 64), dict(Records=[dict(Consumed=[1, 3])])):
            with self.assertRaises(ValueError):
                Q.teacher_gate(dict(report, **mutation), [1, 2, 3], 'a' * 64)

    def test_semantic_only_failure_is_distinct_from_infrastructure(self):
        report = dict(repeated_final_logits_exact=True, native_sha256={'libGgmlOps.so': 'B' * 64},
                      passed=True, quality_passed=True, quality=[dict(name=name, passed=True, eos=True,
                          reached_generation_limit=False) for name in ('math', 'extraction', 'squares')])
        self.assertTrue(Q.semantic_gate(report, 'b' * 64))
        report['quality'][2]['passed'] = False; report.update(passed=False, quality_passed=False)
        self.assertFalse(Q.semantic_gate(report, 'b' * 64))
        report['quality'].pop()
        with self.assertRaises(ValueError):
            Q.semantic_gate(report, 'b' * 64)

    def test_supplementary_freeze_rejects_changed_or_escaping_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'frozen-inputs.json').write_text('{}'); (root / 'script.py').write_text('pass')
            plan = dict(base_manifest_sha256=Q.digest(root / 'frozen-inputs.json'),
                        files=[dict(relative='script.py', sha256=Q.digest(root / 'script.py'))])
            Q.verify_extra(root, plan)
            bad = copy.deepcopy(plan); bad['files'][0]['relative'] = '../outside'
            with self.assertRaisesRegex(ValueError, 'escapes'):
                Q.verify_extra(root, bad)
            (root / 'script.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                Q.verify_extra(root, plan)


if __name__ == '__main__':
    unittest.main()
