import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import copy
import hashlib


SPEC = importlib.util.spec_from_file_location('readiness_report', Path(__file__).parents[1] / 'default-readiness-report.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class AuditSummaryCollectionTests(unittest.TestCase):
    def summary(self, status='passed'):
        return dict(passed=status == 'passed', status=status, errors=[] if status == 'passed' else ['audit failure'],
            display_rows=[dict(scenario='same run', status=status, quality='recorded quality',
                               performance='no new timing', resources='ledger only', scope='same process')])

    def write(self, root, name, data, kind='summary', **extra):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')
        return dict(path=name, kind=kind, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra)

    def index(self, root, current, history):
        (root / 'readiness-audit-index.json').write_text(json.dumps(dict(schema_version=1,
            groups=[dict(id='one physical run', current=current, history=history)])), encoding='utf-8')

    def test_discovers_versioned_summaries_without_inventing_version_relationships(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'case-summary-v2.json', self.summary())
            self.write(root, 'case-summary-v10.json', self.summary('failed'))
            self.write(root, 'nested/summary-v2.json', self.summary())
            self.write(root, 'case-summary-not-a-version.json', self.summary())
            rows, history, sources = REPORT.collected_audit_summaries(root)
            self.assertEqual(3, len(rows))
            self.assertFalse(history)
            self.assertEqual(3, len(sources))
            self.assertIn('failed', [row[1] for row in rows])

    def test_unindexed_display_rows_must_be_a_list_when_present(self):
        for value in (None, {}, 'rows', 1, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, 'case-summary.json', dict(display_rows=value))
                with self.assertRaisesRegex(ValueError, 'malformed display_rows: case-summary.json'):
                    REPORT.collected_audit_summaries(root)

    def test_unindexed_bad_row_cannot_be_silently_dropped_from_valid_rows(self):
        valid = self.summary()['display_rows'][0]
        malformed = [None, [], 'row']
        for field in valid:
            missing = dict(valid); missing.pop(field)
            malformed.extend((missing, {**valid, field: None}, {**valid, field: True}))
        for bad in malformed:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write(root, 'nested/summary.json', dict(display_rows=[valid, bad]))
                with self.assertRaisesRegex(ValueError, 'malformed display_rows: nested/summary.json'):
                    REPORT.collected_audit_summaries(root)

    def test_unindexed_absent_or_empty_display_rows_remain_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'absent-summary.json', dict(passed=True))
            self.write(root, 'empty-summary.json', dict(display_rows=[]))
            self.write(root, 'valid-summary.json', self.summary('failed'))
            rows, history, sources = REPORT.collected_audit_summaries(root)
            self.assertEqual(1, len(rows))
            self.assertEqual('failed', rows[0][1])
            self.assertFalse(history)
            self.assertEqual([root / 'valid-summary.json'], sources)

    def test_explicit_current_is_shown_once_and_old_failure_stays_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = self.write(root, 'case-summary-v2.json', self.summary())
            old = self.write(root, 'case-summary.json', self.summary('failed'), reason='Auditor parsing fixed; no new model run')
            snapshot = self.write(root, 'case-status-summary.json', self.summary(), reason='Earlier snapshot of the same run')
            self.index(root, current, [old, snapshot])
            rows, history, sources = REPORT.collected_audit_summaries(root)
            self.assertEqual(1, len(rows))
            self.assertEqual(2, len(history))
            self.assertIn('failed', history[0][3])
            self.assertIn('audit failure', history[0][5])
            self.assertIn('no new model run', history[0][4])
            self.assertEqual(4, len(sources))

    def test_current_failure_is_never_replaced_by_an_earlier_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = self.write(root, 'case-summary-v2.json', self.summary('failed'))
            old = self.write(root, 'case-summary.json', self.summary(), reason='Earlier audit')
            self.index(root, current, [old])
            rows, history, _ = REPORT.collected_audit_summaries(root)
            self.assertEqual('failed', rows[0][1])
            self.assertIn('passed', history[0][3])

    def test_terminal_encoding_failure_is_preserved_separately_from_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = self.write(root, 'case-summary-v2.json', self.summary())
            old = self.write(root, 'audit-terminal.json', dict(exit_code=1, stage='console rendering', error='UnicodeEncodeError'),
                             kind='terminal', reason='ASCII stdout repair only; underlying run not repeated')
            self.index(root, current, [old])
            rows, history, _ = REPORT.collected_audit_summaries(root)
            self.assertEqual('passed', rows[0][1])
            self.assertIn('"exit_code": 1', history[0][3])
            self.assertIn('UnicodeEncodeError', history[0][5])

    def test_malformed_summary_is_explicit_history_without_partial_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = self.write(root, 'case-summary-v2.json', self.summary())
            malformed = self.summary()
            malformed['display_rows'].append(dict(malformed['display_rows'][0]))
            malformed['display_rows'][1].pop('resources')
            old = self.write(root, 'case-summary.json', malformed, kind='malformed_summary',
                             reason='Display metadata repaired; original run unchanged')
            self.index(root, current, [old])
            rows, history, sources = REPORT.collected_audit_summaries(root)
            self.assertEqual(1, len(rows))
            self.assertEqual(1, len(history))
            self.assertIn('"passed": true', history[0][3])
            self.assertIn('原始 display_rows 格式错误', history[0][5])
            self.assertIn('不计为新增模型执行或模型质量失败', history[0][5])
            self.assertIn(root / old['path'], sources)

    def test_malformed_summary_cannot_be_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = self.summary(); malformed['display_rows'][0].pop('resources')
            current = self.write(root, 'case-summary.json', malformed, kind='malformed_summary')
            self.index(root, current, [])
            with self.assertRaises(ValueError): REPORT.collected_audit_summaries(root)

    def test_malformed_history_kind_requires_an_actually_invalid_display_field(self):
        for data in (self.summary(), dict(display_rows=[]), dict(passed=True), []):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                current = self.write(root, 'case-summary-v2.json', self.summary())
                old = self.write(root, 'case-summary.json', data, kind='malformed_summary',
                                 reason='Must not bypass the ordinary summary schema')
                self.index(root, current, [old])
                with self.assertRaises(ValueError): REPORT.collected_audit_summaries(root)

    def test_missing_or_changed_selected_evidence_fails_closed(self):
        for change in ('missing', 'changed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                current = self.write(root, 'case-summary-v2.json', self.summary())
                old = self.write(root, 'case-summary.json', self.summary(), reason='Older audit')
                self.index(root, current, [old])
                if change == 'missing': (root / current['path']).unlink()
                else: (root / current['path']).write_text(json.dumps(self.summary('failed')))
                with self.assertRaises(ValueError): REPORT.collected_audit_summaries(root)

    def test_index_rejects_escape_duplicate_or_malformed_display(self):
        for change in ('escape', 'duplicate', 'malformed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'evidence'; root.mkdir()
                data = self.summary()
                if change == 'malformed': data['display_rows'][0]['status'] = True
                name = '../outside.json' if change == 'escape' else 'case-summary-v2.json'
                current = self.write(root, name, data)
                history = [{**current, 'reason':'duplicate'}] if change == 'duplicate' else []
                self.index(root, current, history)
                with self.assertRaises(ValueError): REPORT.collected_audit_summaries(root)

    def test_unsupported_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in (2, True, '1'):
                (root / 'readiness-audit-index.json').write_text(json.dumps(dict(schema_version=version, groups=[])))
                with self.assertRaises(ValueError): REPORT.collected_audit_summaries(root)


class SpeculativeReadinessTests(unittest.TestCase):
    def rows(self):
        base = dict(Scenario='spec', Prompt=4, Reused=0, Fresh=4, OutTokens=2,
                    Tokens=[7, 8], Finish='eos', Note='', PrefillTps=10, DecodeTps=10, TtftMs=1,
                    Drafted=0, Accepted=0, VerifySteps=0, PlainSteps=0, Rollbacks=0)
        return [{**base, 'Label': 'plain greedy'},
                {**base, 'Label': 'draft head (auto)', 'Drafted': 4, 'Accepted': 2, 'VerifySteps': 1}]

    def summarize(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.json'
            path.write_text(json.dumps(rows))
            return REPORT.speculative_rows(path)

    def test_complete_engaged_pair_qualifies_without_inventing_missing_timing(self):
        result = self.summarize(self.rows())[0]
        self.assertTrue(result[2])
        self.assertEqual('未测量 → 未测量', result[5])

    def test_identically_truncated_pair_is_not_a_complete_pass(self):
        rows = self.rows()
        for row in rows:
            row['Finish'] = 'length'
        self.assertFalse(self.summarize(rows)[0][2])

    def test_attached_but_unaccepted_draft_is_not_coverage(self):
        rows = self.rows()
        rows[1]['Accepted'] = 0
        self.assertFalse(self.summarize(rows)[0][2])


class MayaReadinessTests(unittest.TestCase):
    def report(self):
        row = dict(tokens=[42, 9], generated=2, prompt_tokens=3, reused=0,
                   drafts_offered=0, drafts_accepted=0, finish='stop', prompt_ms=100, decode_ms=50,
                   quality=dict(passed=True), semantic=dict(passed=True, thinking_closed=True, tool_calls=[], content='42', raw_text='reason</think>42'))
        return dict(completed=True, process_success=True, exit_code=0, identities_unchanged=True,
                    errors=[], timed_out=False, prompt_tokenizer_exact=True,
                    performance_observation_complete=True, telemetry_samples=2, sampling_errors=[],
                    quality_passed=True, plan=dict(repetitions=2, eos_ids=[9], case='math'), rows=[row, copy.deepcopy(row)],
                    reference=dict(name='math', prompt=REPORT.fixed_tasks.PROMPTS['math'],
                        rendered_prompt='<|assistant|><think>',
                        prompt_tokens=[1, 2, 3], generated_tokens=[42, 9], passed=True, eos=True, parsed_text='42'))

    def test_semantics_and_exact_tokens_are_independent_and_decode_uses_forward_count(self):
        report = self.report()
        report['rows'][1]['tokens'] = [43, 9]
        result = REPORT.maya_summary(report)
        self.assertTrue(result['quality_passed'])
        self.assertTrue(result['rows'][0]['exact'])
        self.assertFalse(result['rows'][1]['exact'])
        self.assertEqual(20, result['rows'][0]['tg'])
        self.assertEqual(30, result['rows'][0]['pp'])

    def test_incomplete_or_failed_run_cannot_qualify_completed_rows(self):
        for key, value in [('completed', False), ('process_success', False), ('exit_code', 1),
                           ('identities_unchanged', False), ('errors', ['cleanup failed']), ('timed_out', True)]:
            report = self.report(); report[key] = value
            result = REPORT.maya_summary(report)
            self.assertFalse(result['quality_passed'], key)
            self.assertFalse(any(r['passed'] for r in result['rows']), key)
        report = self.report(); report['rows'].pop()
        self.assertFalse(REPORT.maya_summary(report)['quality_passed'])

    def test_claimed_success_cannot_override_non_eos_wire_or_parser_failure(self):
        for changes in [dict(finish='length'), dict(generated=1), dict(reused=1), dict(tokens=[9, 9]),
                        dict(semantic=dict(passed=True, thinking_closed=True, content='41', tool_calls=[]))]:
            report = self.report(); report['rows'][0].update(changes)
            self.assertFalse(REPORT.maya_summary(report)['quality_passed'])

    def test_missing_sampling_does_not_invalidate_semantics_or_invent_performance(self):
        report = self.report(); report.pop('performance_observation_complete')
        result = REPORT.maya_summary(report)
        self.assertTrue(result['quality_passed'])
        self.assertFalse(result['performance_complete'])

    def test_independent_recheck_preserves_original_false_and_policy(self):
        report = self.report()
        report['plan']['case'] = report['reference']['name'] = 'extraction'
        report['reference']['prompt'] = REPORT.fixed_tasks.PROMPTS['extraction']
        report['reference']['parsed_text'] = 'cedar, maple'
        report['quality_passed'] = False
        for row in report['rows']:
            row['semantic'].update(content='cedar,maple', raw_text='reason</think>cedar,maple', passed=False)
            row['quality']['passed'] = False
        result = REPORT.maya_summary(report)
        self.assertTrue(result['independent_quality_passed'])
        self.assertFalse(result['recorded_runner_quality_passed'])
        self.assertFalse(result['rows'][0]['recorded_runner_passed'])
        self.assertEqual(REPORT.fixed_tasks.LEGACY_MAYA_POLICY, result['recorded_runner_policy'])

    def test_model_reference_answer_is_never_ground_truth(self):
        report = self.report()
        report['reference']['parsed_text'] = '41'
        report['reference']['passed'] = False
        result = REPORT.maya_summary(report)
        self.assertTrue(result['independent_quality_passed'])
        self.assertFalse(result['reference_independent_semantic']['passed'])
        self.assertFalse(result['reference_recorded_passed'])
        report['rows'][0]['semantic']['content'] = '41'
        self.assertFalse(REPORT.maya_summary(report)['independent_quality_passed'])

    def test_reference_structure_is_separate_from_its_recorded_semantics(self):
        for change in [dict(eos=False), dict(prompt_tokens=[]), dict(generated_tokens=[42, 8]),
                       dict(generated_tokens=[9, 9])]:
            report = self.report(); report['reference'].update(change)
            self.assertFalse(REPORT.maya_summary(report)['independent_quality_passed'])

    def test_missing_case_or_wrong_prompt_cannot_use_answer_equality(self):
        for area, key, value in [('plan', 'case', None), ('reference', 'name', None),
                                 ('reference', 'prompt', ''), ('plan', 'case', 'unknown')]:
            report = self.report(); report[area][key] = value
            self.assertFalse(REPORT.maya_summary(report)['independent_quality_passed'])

    def test_independent_pass_cannot_override_incomplete_parser_or_json_fence(self):
        for change in [dict(raw_text='reason without closure'), dict(tool_calls=[dict(name='tool')]),
                       dict(content='```\n42\n```'), dict(content='{"answer":42}')]:
            report = self.report(); report['rows'][0]['semantic'].update(change)
            self.assertFalse(REPORT.maya_summary(report)['independent_quality_passed'])

    def test_report_recheck_uses_the_same_fixed_rule_vectors(self):
        for case, good, bad in [('math', '42', '41'), ('extraction', 'cedar,maple', 'maple,cedar'),
                                ('squares', ','.join(str(i*i) for i in range(1,21)), '1,4,9')]:
            for answer in (good, bad):
                report = self.report()
                report['plan']['case'] = report['reference']['name'] = case
                report['reference']['prompt'] = REPORT.fixed_tasks.PROMPTS[case]
                for row in report['rows']:
                    row['semantic']['content'] = answer
                    row['semantic']['raw_text'] = 'reason</think>' + answer
                expected = REPORT.fixed_tasks.evaluate(case, REPORT.fixed_tasks.PROMPTS[case], answer)['passed']
                self.assertEqual(expected, REPORT.maya_summary(report)['independent_quality_passed'])

    def test_current_closure_rejects_saved_true_reopened_or_missing_close(self):
        for suffix, raw in [('<think>\n', '42'), ('<think>\r\n', '42'),
                            ('<think>', 'reason</think>42<think>unfinished')]:
            report = self.report(); report['reference']['rendered_prompt'] = '<|assistant|>' + suffix
            report['rows'][0]['semantic']['raw_text'] = raw
            result = REPORT.maya_summary(report)
            self.assertFalse(result['independent_quality_passed'])
            check = result['rows'][0]['thinking_closure_recheck']
            self.assertTrue(check['saved_thinking_closed'])
            self.assertFalse(check['current_thinking_closed'])
            self.assertEqual('rechecked', check['status'])

    def test_raw_closure_recheck_can_correct_saved_false_without_mutation(self):
        report = self.report(); report['rows'][0]['semantic']['thinking_closed'] = False
        original = copy.deepcopy(report)
        result = REPORT.maya_summary(report)
        self.assertTrue(result['independent_quality_passed'])
        self.assertFalse(result['rows'][0]['thinking_closure_recheck']['saved_thinking_closed'])
        self.assertTrue(result['rows'][0]['thinking_closure_recheck']['current_thinking_closed'])
        self.assertEqual(original, report)

    def test_missing_original_structure_is_explicitly_unavailable_not_pass(self):
        for field in ('raw_text', 'rendered_prompt'):
            report = self.report()
            if field == 'raw_text': report['rows'][0]['semantic'].pop(field)
            else: report['reference'].pop(field)
            result = REPORT.maya_summary(report)
            self.assertFalse(result['independent_quality_passed'])
            check = result['rows'][0]['thinking_closure_recheck']
            self.assertEqual('structure_recheck_unavailable', check['status'])
            self.assertIsNone(check['current_thinking_closed'])

    def test_empty_incomplete_run_remains_visible(self):
        report = self.report(); report['rows'] = []; report['completed'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            rows = REPORT.maya_rows(path)
        self.assertEqual(1, len(rows))
        self.assertFalse(rows[0][1])
        self.assertFalse(rows[0][2])
        self.assertEqual('无完整请求', rows[0][4])


if __name__ == '__main__':
    unittest.main()
