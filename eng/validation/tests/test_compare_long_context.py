import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('long_context', Path(__file__).parents[1] / 'compare-long-context.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(arm):
    return dict(Arm=arm, Passed=True, Error=None, ModelSha256='model', GgmlRevision='upstream',
                ContextTokens=32768, MinimumPromptTokens=28000, Steps=2, Width=2,
                ChunkTokens=256, MaximumRunningRequests=2, PrefillChunkLimitEnforced=True,
                BulkRestoreEnabled=True, Native=[dict(FileName='GgmlOps.dll', Sha256='native')],
                Managed=[dict(Path='TensorSharp.Runtime.dll', Sha256='runtime')], TokenParity=True,
                RunningSamples=[dict(Running=2)], AfterDispose=[dict(Reserved=0, Committed=0)],
                WallSeconds=3, Requests=[dict(RequestId=str(i), PromptTokens=28001,
                  PromptHash=str(i), Tokens=[1, 2], Status='FinishedLengthCapped', FinishReason='max_tokens',
                  PrefillSeconds=1, DecodeSeconds=.1) for i in range(2)])


class ComparisonTests(unittest.TestCase):
    def test_matching_evidence(self):
        self.assertTrue(module.compare([fixture('reference')], [fixture('candidate')])['passed'])

    def test_unbudgeted_reference_has_no_ledger(self):
        reference = fixture('reference')
        reference['AfterDispose'] = None
        self.assertTrue(module.compare([reference], [fixture('candidate')])['passed'])

    def test_no_vacuous_or_serial_acceptance(self):
        for mutate in (lambda r: r.update(Requests=[]), lambda r: r.update(RunningSamples=[dict(Running=1)]),
                       lambda r: r.update(AfterDispose=[]), lambda r: r['Requests'][0].update(Tokens=[1]),
                       lambda r: r['AfterDispose'][0].update(Committed=1)):
            with self.subTest(mutate=mutate):
                changed = fixture('candidate')
                mutate(changed)
                with self.assertRaises(ValueError):
                    module.compare([fixture('reference')], [changed])

    def test_changed_tokens_or_binary_refused(self):
        for kind in ('tokens', 'binary'):
            changed = fixture('candidate')
            if kind == 'tokens': changed['Requests'][0]['Tokens'][1] = 3
            else: changed['Managed'][0]['Sha256'] = 'different'
            with self.assertRaises(ValueError): module.compare([fixture('reference')], [changed])

    def test_latency_regression_retains_exact_quality(self):
        changed = fixture('candidate')
        changed['WallSeconds'] *= 1.1
        result = module.compare([fixture('reference')], [changed])
        self.assertFalse(result['passed'])
        self.assertTrue(result['exact_tokens'])

    def test_changed_environment_or_duplicate_process_is_rejected(self):
        reference, candidate = fixture('reference'), fixture('candidate')
        for r, pid in ((reference, 1), (candidate, 2)):
            r.update(ProcessId=pid, StartedUtc='2026-10-10T00:00:00Z',
                     ExecutionEnvironment={'TS_EXAMPLE': '1'}, ProbeSha256='probe')
        self.assertTrue(module.compare([reference], [candidate])['process_and_environment_identity_checked'])
        candidate['ExecutionEnvironment'] = {'TS_EXAMPLE': '0'}
        with self.assertRaisesRegex(ValueError, 'environment'): module.compare([reference], [candidate])
        candidate['ExecutionEnvironment'] = reference['ExecutionEnvironment']
        candidate['ProcessId'] = reference['ProcessId']
        with self.assertRaisesRegex(ValueError, 'Repeated process'): module.compare([reference], [candidate])

    def test_candidate_budget_cannot_change_between_processes(self):
        candidates = [fixture('candidate'), fixture('candidate')]
        references = [fixture('reference'), fixture('reference')]
        for r in candidates:
            r['RequestedBudget'] = [dict(Pool='host', Bytes=100), dict(Pool='device', Bytes=200)]
        self.assertTrue(module.compare(references, candidates)['candidate_budget_identity_checked'])
        candidates[1]['RequestedBudget'][0]['Bytes'] = 101
        with self.assertRaisesRegex(ValueError, 'budgets differ'): module.compare(references, candidates)
        candidates[1].pop('RequestedBudget')
        with self.assertRaisesRegex(ValueError, 'budget identity'): module.compare(references, candidates)


if __name__ == '__main__':
    unittest.main()
