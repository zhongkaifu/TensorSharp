import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest


SPEC=importlib.util.spec_from_file_location('glm_candidate_comparison_supervision',
    Path(__file__).parents[1]/'glm-unified-candidate-comparison.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


def fixture():
    plan=dict(Binding=dict(PromptTokens=[1,2,3]),ExpectedProbeUnknownExit=1)
    report=dict(PromptTokens=[1,2,3],OutputTokens=[5,6,99],ComputedTokens=6,NaturalEos=True,
                Errors=[],LedgerZeroAfterCleanup=True,Timing=dict(EngineOutputTokens=3,OutputForwardCount=3))
    wrapper=dict(exit_code=1,complete=True,timed_out=False,cleanup_errors=[],
                 telemetry=dict(passed=True,errors=[],memory=dict(samples=2)))
    return plan,report,wrapper,dict(Passed=True,IndependentSemantic=dict(passed=True))


class CandidateComparisonSupervisionTests(unittest.TestCase):
    def test_unknown_exit_alone_is_never_quality(self):
        plan,report,wrapper,qualified=fixture()
        self.assertTrue(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])
        qualified['Passed']=False
        self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])
        qualified['Passed']=True
        self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,0,0)['Passed'])
        self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,1)['Passed'])

    def test_partial_timeout_cleanup_and_missing_telemetry_refuse(self):
        plan,report,original,qualified=fixture()
        for changes in (dict(complete=False),dict(timed_out=True),dict(cleanup_errors=['live child']),
                        dict(telemetry={}),dict(telemetry=dict(passed=True,errors=['identity unavailable']))):
            wrapper=copy.deepcopy(original);wrapper.update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])

    def test_forward_counts_eos_and_ledger_are_independent_gates(self):
        plan,original,wrapper,qualified=fixture()
        for changes in (dict(ComputedTokens=5),dict(PromptTokens=[1,2]),dict(OutputTokens=[]),
                        dict(NaturalEos=False),dict(LedgerZeroAfterCleanup=False),
                        dict(Timing=dict(EngineOutputTokens=3,OutputForwardCount=2))):
            report=copy.deepcopy(original);report.update(changes)
            with self.subTest(changes=changes):
                self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])

    def test_same_count_different_ids_never_equal_work(self):
        rows=[dict(Task='tool_json',Passed=True,RawPromptTokens=[1,2],RawOutputTokens=[5,99]) for _ in range(4)]
        self.assertTrue(MODULE.comparison(rows)[0]['EqualForwardWork'])
        rows[1]['RawOutputTokens']=[6,99]
        result=MODULE.comparison(rows)[0]
        self.assertTrue(result['CompletedAndIndependentlyQualified'])
        self.assertFalse(result['SameRawOutputIds']);self.assertFalse(result['EqualForwardWork'])
        self.assertIsNone(MODULE.comparison(rows[:3])[0]['EqualForwardWork'])

    def test_clean_environment_does_not_inherit_diagnostic_or_speculation_flags(self):
        result=MODULE.clean_environment(dict(PATH='ordinary',CUDA_PATH='installer',TS_HOST_MOE_EXPERT_CACHE_TIMING='1',
            TS_SPEC='1',GLM_X='1',GGML_X='1',STRATA_X='1',MAX_CONTEXT='9999',KV_CACHE_DTYPE='q4'))
        self.assertEqual(result,dict(PATH='ordinary',CUDA_PATH='installer',**MODULE.QUIET))

    def test_direct_experiment_is_explicit_and_candidate_only(self):
        self.assertEqual(MODULE.arm_environment('baseline',True),MODULE.QUIET)
        self.assertEqual(MODULE.arm_environment('candidate',False),MODULE.QUIET)
        enabled=MODULE.arm_environment('candidate',True)
        self.assertEqual(enabled[MODULE.DIRECT_READ_KEY],'1')
        self.assertEqual({k:v for k,v in enabled.items() if k!=MODULE.DIRECT_READ_KEY},MODULE.QUIET)
        for role,flag in [('unknown',True),('candidate','false'),('baseline',1)]:
            with self.subTest(role=role,flag=flag),self.assertRaises(ValueError):
                MODULE.arm_environment(role,flag)

    def test_direct_environment_requires_runtime_bookends(self):
        plan,report,wrapper,qualified=fixture()
        plan['Environment']=MODULE.arm_environment('candidate',True)
        self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])
        fields=('EnvironmentBefore','EnvironmentEffective','EnvironmentAfter')
        for field in fields:report[field]={MODULE.DIRECT_READ_KEY:'1'}
        self.assertTrue(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])
        for field in fields:
            changed=copy.deepcopy(report);changed[field][MODULE.DIRECT_READ_KEY]='0'
            self.assertFalse(MODULE.evidence_gate(plan,changed,wrapper,qualified,1,0)['Passed'])
        plan['Environment']=MODULE.arm_environment('baseline',True)
        self.assertFalse(MODULE.evidence_gate(plan,report,wrapper,qualified,1,0)['Passed'])

    def test_owned_wrapper_exception_restores_caller_environment_and_arguments(self):
        original_env,original_args=dict(os.environ),sys.argv[:]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fake_wrapper.py'
            path.write_text('import os,sys\ndef main():\n assert os.environ.get("TASK_TEST") == "child"\n assert sys.argv[1:] == ["--fake", "child"]\n raise RuntimeError("synthetic wrapper failure")\n','utf-8')
            with self.assertRaisesRegex(RuntimeError,'synthetic wrapper failure'):
                MODULE.owned(path,['--fake','child'],dict(TASK_TEST='child'),Path(directory)/'output.log')
        self.assertEqual(dict(os.environ),original_env);self.assertEqual(sys.argv,original_args)


if __name__=='__main__':unittest.main()
