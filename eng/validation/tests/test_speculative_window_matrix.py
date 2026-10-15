import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('window_matrix', Path(__file__).resolve().parents[1] / 'run-speculative-window-matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def rows():
    common = dict(Scenario='spec', Prompt=127, Tokens=[1, 2, 3], Finish='eos', Text='answer',
                  PrefillComputeMs=10., DecodeComputeMs=20., TotalMs=30., DecodeTps=50.,
                  Drafted=4, Accepted=3, VerifySteps=2, PlainSteps=1, Rollbacks=1)
    plain = dict(common, Label='plain greedy')
    speculative = dict(common, Label='draft head (auto)', Note='armed maxDraft=1',
                       SpeculationTimings={phase: 1. for phase in matrix.PHASES})
    return [plain, speculative]


class MatrixQualificationTests(unittest.TestCase):
    def test_complete_tokens_and_eos_both_required(self):
        sample = rows()
        self.assertTrue(matrix.qualify_rows(sample, 'plain-first', 1)['passed'])
        sample[1]['Tokens'] = [1, 2, 4]
        result = matrix.qualify_rows(sample, 'plain-first', 1)
        self.assertFalse(result['passed'])
        self.assertEqual(2, result['validator']['first_divergence'])
        sample = rows()
        for row in sample:
            row['Finish'] = 'length'
        result = matrix.qualify_rows(sample, 'plain-first', 1)
        self.assertEqual('passed', result['validator']['status'])
        self.assertFalse(result['passed'])

    def test_cross_pass_plain_and_prompt_continuity_are_independent(self):
        self.assertFalse(matrix.qualify_rows(rows(), 'plain-first', 1, [1, 2, 5])['passed'])
        self.assertFalse(matrix.qualify_rows(rows(), 'plain-first', 1, [1, 2, 3], 128)['passed'])
        sample = rows()
        sample[1]['Prompt'] = 128
        self.assertFalse(matrix.qualify_rows(sample, 'plain-first', 1)['passed'])

    def test_order_window_and_full_phase_evidence_are_hard_failures(self):
        for operation in (lambda x: x.reverse(), lambda x: x[1].update(Note='armed maxDraft=3'),
                          lambda x: x[1]['SpeculationTimings'].pop('DraftMs'),
                          lambda x: x[1]['SpeculationTimings'].update(DraftMs=float('nan'))):
            sample = rows()
            operation(sample)
            with self.assertRaises(ValueError):
                matrix.qualify_rows(sample, 'plain-first', 1)

    def test_absent_speculative_engagement_is_quality_failure(self):
        sample = rows()
        sample[1].update(Drafted=0, Accepted=0, VerifySteps=0)
        self.assertFalse(matrix.qualify_rows(sample, 'plain-first', 1)['passed'])

    def test_aggregation_excludes_warmup_and_retains_failed_quality_cost(self):
        check = matrix.qualify_rows(rows(), 'plain-first', 1)
        measured = dict(check, warmup=False)
        warmup = dict(check, warmup=True, plain_decode_ms=9999., speculative_decode_ms=1.)
        records = [dict(window=1, order=order, process_passed=True, quality_passed=True,
                        checks=[copy.deepcopy(warmup), copy.deepcopy(measured), copy.deepcopy(measured)])
                   for order in ('plain-first', 'spec-first')]
        result = matrix.aggregate(records)['1']
        self.assertEqual(4, result['measured_pairs'])
        self.assertEqual(20., result['plain_decode_ms_median'])
        self.assertTrue(result['all_quality_gates_passed'])
        records[0]['quality_passed'] = False
        result = matrix.aggregate(records)['1']
        self.assertFalse(result['all_quality_gates_passed'])
        self.assertEqual(20., result['speculative_decode_ms_median'])
        self.assertFalse(matrix.aggregate(records[1:])['1']['all_quality_gates_passed'])

    def test_counterbalanced_full_matrix(self):
        self.assertEqual(6, len(matrix.RUNS))
        self.assertEqual({(n, order) for n in (1, 3, 7) for order in ('plain-first', 'spec-first')}, set(matrix.RUNS))

    def test_auto_clears_inherited_speculation_controls(self):
        with mock.patch.dict(os.environ, {'TS_SPEC_DRAFT': '99', 'TS_SPEC_MIN': 'bad', 'MAX_CONTEXT': '9999', 'GGML_TEST': '1'}):
            automatic = matrix.child_environment(7)
            self.assertNotIn('TS_SPEC_DRAFT', automatic)
            self.assertNotIn('TS_SPEC_MIN', automatic)
            self.assertNotIn('GGML_TEST', automatic)
            self.assertEqual('2048', automatic['MAX_CONTEXT'])
            self.assertEqual('3', matrix.child_environment(3)['TS_SPEC_DRAFT'])

    def test_quality_failure_does_not_skip_remaining_independent_processes(self):
        # Mock only process/pin boundaries: run the real artifact reader,
        # qualification, continuation and final aggregation over six processes.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'matrix'
            plan = dict(entry=str(root / 'AgentTurnBench.dll'), repository=str(root), output=str(output),
                        managed_files={}, expected_native_sha256='0' * 64, timeout_per_process_seconds=1,
                        scope='mock process test', runs=[])
            for index, (window, order) in enumerate(matrix.RUNS, 1):
                owned = output / str(index)
                plan['runs'].append(dict(index=index, window=window, order=order, owned_output=str(owned),
                                         rows_base=str(owned / 'rows.json'), command=['dotnet', str(root / 'AgentTurnBench.dll'), '--case', str(index)]))
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            calls = []

            def simulate(command, **kwargs):
                index = len(calls)
                calls.append(command)
                item = plan['runs'][index]
                owned = Path(item['owned_output'])
                owned.mkdir()
                execution = dict(complete=True, exit_code=0, pid=index + 1, wall_seconds=1,
                                 telemetry=dict(passed=True, managed_files_before={}, managed_files_after={}))
                (owned / 'execution.json').write_text(json.dumps(execution))
                metadata = dict(ProcessId=index + 1, SpecOrder=item['order'], WarmupPasses=1, MeasuredPasses=2, Arguments=item['command'][2:])
                Path(item['rows_base'] + '.run.json').write_text(json.dumps(metadata))
                sample = rows()
                sample[1]['Note'] = 'armed maxDraft=' + str(item['window'])
                if index == 0:
                    sample[1]['Tokens'] = [1, 2, 4]
                if item['order'] == 'spec-first':
                    sample.reverse()
                for suffix in ('.warmup1.json', '.measure1.json', '.measure2.json'):
                    Path(item['rows_base'] + suffix).write_text(json.dumps(sample))
                return SimpleNamespace(returncode=0)

            with mock.patch.object(matrix, 'verify_pins'), mock.patch.object(matrix.subprocess, 'run', side_effect=simulate), contextlib.redirect_stdout(io.StringIO()):
                exit_code = matrix.execute(SimpleNamespace(plan=plan_path))
            report = json.loads((output / 'matrix.json').read_text())
            self.assertEqual(6, len(calls))
            self.assertEqual(2, exit_code)
            self.assertTrue(report['complete'])
            self.assertFalse(report['passed'])
            self.assertTrue(all(row['process_passed'] for row in report['runs']))
            self.assertFalse(report['runs'][0]['quality_passed'])
            self.assertTrue(all(row['quality_passed'] for row in report['runs'][1:]))


if __name__ == '__main__':
    unittest.main()
