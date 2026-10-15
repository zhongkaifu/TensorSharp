import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('fixed_tasks', Path(__file__).parents[1] / 'fixed-task-semantic.py')
TASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TASKS)


class FixedTaskSemanticTests(unittest.TestCase):
    def check(self, name, answer):
        return TASKS.evaluate(name, TASKS.PROMPTS[name], answer)['passed']

    def test_math_uses_fixed_integer_not_reference(self):
        for answer in ('42', ' 42\n', '42.', '42!'):
            self.assertTrue(self.check('math', answer), answer)
        for answer in ('41', '142', 'The answer is 42', '```\n42\n```', '{"answer":42}', '42\n43', None):
            self.assertFalse(self.check('math', answer), answer)

    def test_extraction_accepts_only_comma_whitespace_and_case_variation(self):
        for answer in ('cedar,maple', 'cedar, maple', ' CEDAR , MAPLE\n'):
            self.assertTrue(self.check('extraction', answer), answer)
        for answer in ('maple,cedar', 'cedar,maple,pine', 'cedar maple', 'cedar;maple',
                       '["cedar","maple"]', '```\ncedar,maple\n```', 'Names: cedar,maple', ''):
            self.assertFalse(self.check('extraction', answer), answer)

    def test_squares_requires_complete_ordered_list(self):
        values = [str(i * i) for i in range(1, 21)]
        for answer in (','.join(values), ' , '.join(values), ','.join('0' + v for v in values)):
            self.assertTrue(self.check('squares', answer))
        for answer in (','.join(values[:-1]), ','.join(values + ['441']), ','.join(values[::-1]),
                       ','.join(values[:-1] + ['361']), '[' + ','.join(values) + ']',
                       '```\n' + ','.join(values) + '\n```', ','.join(['9' * 5000] + values[1:])):
            self.assertFalse(self.check('squares', answer))

    def test_unknown_missing_or_wrong_prompt_does_not_fall_back(self):
        for name, prompt in [('other', 'anything'), (None, None), ('math', ''),
                             ('math', TASKS.PROMPTS['extraction'])]:
            result = TASKS.evaluate(name, prompt, '42')
            self.assertFalse(result['passed'])
            self.assertFalse(result['prompt_matches'])

    def test_correct_answer_never_overrides_parser_structure(self):
        for closed, calls in [(False, []), (None, []), (True, [dict(name='x')]), (True, None)]:
            self.assertFalse(TASKS.parsed_answer('math', TASKS.PROMPTS['math'], '42', closed, calls)['passed'])
        self.assertTrue(TASKS.parsed_answer('math', TASKS.PROMPTS['math'], '42', True, [])['passed'])

    def test_extended_tasks_share_exact_prompts_and_strict_structural_fixtures(self):
        fixture = json.loads((Path(__file__).parents[1] / 'fixtures/fixed-extended-task-answers.json').read_text(encoding='utf-8'))
        for name, prompt in fixture['prompts'].items():
            self.assertEqual(prompt, TASKS.PROMPTS[name])
        for row in fixture['rows']:
            with self.subTest(row=row['id']):
                self.assertEqual(row['passed'], self.check(row['case'], row['answer']))
                self.assertFalse(TASKS.evaluate(row['case'], TASKS.PROMPTS[row['case']] + ' changed', row['answer'])['passed'])
                self.assertFalse(TASKS.parsed_answer(row['case'], TASKS.PROMPTS[row['case']], row['answer'], False, [])['passed'])
                self.assertFalse(TASKS.parsed_answer(row['case'], TASKS.PROMPTS[row['case']], row['answer'], True, [{'name': 'unexpected'}])['passed'])

    def test_thinking_closure_checks_prompt_whitespace_and_response_openers(self):
        for suffix in ('<think>', '<think>\n', '<think>\r\n', '<think> \t'):
            self.assertFalse(TASKS.thinking_closed('<|assistant|>' + suffix, '42'))
            self.assertTrue(TASKS.thinking_closed('<|assistant|>' + suffix, 'reason</think>42'))
        self.assertFalse(TASKS.thinking_closed('<|assistant|>', '<think>reason'))
        self.assertFalse(TASKS.thinking_closed('<think>', 'reason</think>42<think>unfinished'))


if __name__ == '__main__':
    unittest.main()
