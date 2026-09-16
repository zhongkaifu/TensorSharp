import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'prepare-deepseek-teacher-logits.py'
SPEC = importlib.util.spec_from_file_location('teacher_plan', PATH)
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def fixture():
    request = {'messages': [
        {'role': 'user', 'content': '\u5317\u4eac \U0001f680'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'b', 'function': {'name': 'second', 'arguments': '{"x":2}'}},
            {'id': 'a', 'function': {'name': 'first', 'arguments': '{"x":1}'}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'one'},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'two'}],
        'tools': [], 'response_format': {'type': 'json_object'},
        'extra_body': {'tool_choice': 'none', 'think': False}, 'max_tokens': 256, 'stream': True}
    return {'weights_id': '58d8ac86298fdf85a2440defee08b1abcad32e45-Q4_K_M',
            'thinking': False, 'run_complete': True, 'execution_plan': {'expected_cases': 1},
            'cases': [{'tag': 'agentic-c1-r0-i0', 'scenario': 'agentic', 'concurrency': 1,
                       'status': 'fail', 'turns': [{'request': request,
                        'metrics': {'assistant_message': {'role': 'assistant', 'content': ''}}}]}]}


class TeacherPlanTests(unittest.TestCase):
    def test_preserves_failed_case_unicode_ids_and_unmutated_history(self):
        data = fixture(); before = copy.deepcopy(data)
        row, = PLAN.extract_requests(data, 'quality.json', 'a' * 64)
        self.assertEqual(PLAN.canonical(row['request']), PLAN.canonical(before['cases'][0]['turns'][0]['request']))
        self.assertEqual(row['source']['case_status'], 'fail')
        self.assertEqual(data, before)
        row['request']['messages'][0]['content'] = 'changed after export'
        self.assertEqual(data, before)

    def test_rejects_missing_cases(self):
        data = fixture(); data['execution_plan']['expected_cases'] = 2
        with self.assertRaisesRegex(ValueError, 'missing cases'):
            PLAN.extract_requests(data, 'quality.json', 'a' * 64)

    def test_rejects_media_without_silent_text_conversion(self):
        data = fixture(); data['cases'][0]['turns'][0]['request']['messages'][0]['content'] = [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,fixture'}}]
        with self.assertRaisesRegex(ValueError, 'media'):
            PLAN.extract_requests(data, 'quality.json', 'a' * 64)

    def test_requires_explicit_native_digest(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            PLAN.build_plan('latest')

    def test_all_archived_requests_and_only_tp_environment_difference(self):
        plan = PLAN.build_plan('2' * 64)
        self.assertEqual(plan['request_count'], 113)
        self.assertEqual([s['requests'] for s in plan['sources']], [60, 25, 25, 3])
        self.assertFalse(plan['qualified'])
        a, b = [copy.deepcopy(v) for v in plan['variants'].values()]
        self.assertEqual(a['environment'].pop('TS_DSV41_TP'), '0')
        self.assertEqual(b['environment'].pop('TS_DSV41_TP'), '7')
        self.assertEqual(a['environment'], b['environment'])
        self.assertEqual(a['native_load'], b['native_load'])


if __name__ == '__main__':
    unittest.main()
