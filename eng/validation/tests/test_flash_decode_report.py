import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('report', Path(__file__).resolve().parents[1] / 'flash-decode-report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
PROMPT = 'Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.'
ANSWER = ', '.join(str(n*n) for n in range(1, 21))


class ReportQualificationTests(unittest.TestCase):
    def group(self, answers):
        return dict(identity=dict(options=dict(prompt=PROMPT)), executions=[dict(runs=[
            dict(selected_text=text, finish_reason=finish) for text, finish in answers])])

    def test_every_request_including_first_must_satisfy_task(self):
        group = self.group([(ANSWER, 'eos'), (ANSWER, 'length')])
        result = report.qualify_semantics(group, 'squares')
        self.assertFalse(result['passed'])
        self.assertEqual((result['passed_requests'], result['requests']), (1, 2))

    def test_fenced_output_is_not_repaired_for_acceptance(self):
        result = report.qualify_semantics(self.group([('```\n'+ANSWER+'\n```', 'eos')]), 'squares')
        self.assertFalse(result['passed'])
        self.assertTrue(result['example_text'].startswith('```'))

    def test_wrong_prompt_cannot_borrow_an_unrelated_quality_label(self):
        with self.assertRaises(ValueError):
            report.qualify_semantics(self.group([(ANSWER, 'eos')]), 'tool_json')

    def test_empty_evidence_cannot_pass(self):
        with self.assertRaises(ValueError):
            report.qualify_semantics(self.group([]), 'squares')


if __name__ == '__main__':
    unittest.main()
