import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eng.validation.tests.fixtures import deepseek_metadata as FIXTURE

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('teacher_bodies', ROOT / 'eng/validation/prepare-deepseek-teacher-bodies.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
ENGINE = FIXTURE.ENGINE
PLAN_SPEC = importlib.util.spec_from_file_location('teacher_fixture_plan',ROOT / 'eng/validation/prepare-deepseek-teacher-logits.py')
PLAN_MOD = importlib.util.module_from_spec(PLAN_SPEC);PLAN_SPEC.loader.exec_module(PLAN_MOD)


class HistoricalPinTests(unittest.TestCase):
    def test_production_historical_inputs_still_refuse_synthetic_substitution(self):
        spec=importlib.util.spec_from_file_location('unmodified_teacher_pin',ROOT / 'eng/validation/prepare-deepseek-teacher-bodies.py')
        production=importlib.util.module_from_spec(spec);spec.loader.exec_module(production)
        self.assertEqual(production.ENGINE_SHA256,'d0c7b8dbacf70d6a663d39a51a6fa5e77cd36714d70591f8dc0e8ae9fe304ea6')
        self.assertEqual(production.PROFILE_SHA256,'0a29cf8c4b1bd6c4567848314ec246e813f6f145ae8d7c40b14253d7f831639d')
        with self.assertRaisesRegex(ValueError,'pin mismatch'):
            production.engine_body_builder(ENGINE.read_bytes())


class TeacherBodyTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        fixture=FIXTURE.bundle(directory.name);self.profile=fixture['sparse'];self.plan=Path(directory.name)/'teacher-plan.json'
        with patch.multiple(PLAN_MOD,ROOT=fixture['root'],REPORTS=fixture['reports'],ARCHIVE=fixture['archive'],
                            PINNED_REPORTS=fixture['pinned_reports'],SOURCE_PROFILE_SHA256=FIXTURE.digest(self.profile)):
            self.plan.write_bytes(MOD.canonical(PLAN_MOD.build_plan('2'*64)))
        pins=patch.multiple(MOD,ENGINE_SHA256=FIXTURE.digest(ENGINE),PROFILE_SHA256=FIXTURE.digest(self.profile))
        pins.start();self.addCleanup(pins.stop)
        self.build, _ = MOD.engine_body_builder(ENGINE.read_bytes())

    def test_exact_merge_priority_matches_independent_literal(self):
        request = {'messages': [{'role': 'user', 'content': 'caf\u00e9\n'}], 'max_tokens': 9,
                   'tools': [{'explicit': 1}], 'response_format': {'type': 'json_object'},
                   'extra_body': {'max_tokens': 7, 'tools': [{'extra': 1}],
                                  'response_format': {'type': 'text'}, 'tool_choice': 'none',
                                  'stream_options': {'wrong': True}, 'timings_per_token': False}}
        before = copy.deepcopy(request)
        got = self.build('http://invalid/', 'm', **request)
        self.assertEqual(got, {'url': 'http://invalid/v1/chat/completions', 'timeout_s': 1200.0,
                              'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'caf\u00e9\n'}],
                                       'max_tokens': 7, 'temperature': 0, 'stream': True,
                                       'tools': [{'explicit': 1}], 'response_format': {'type': 'json_object'},
                                       'tool_choice': 'none', 'stream_options': {'include_usage': True},
                                       'timings_per_token': True}})
        self.assertEqual(before, request)

    def test_empty_optional_values_do_not_overwrite_extra_and_blocking_has_no_stream_fields(self):
        got = self.build('http://invalid', 'm', [], stream=False, tools=[], response_format={},
                         extra_body={'tools': ['retained'], 'response_format': {'type': 'text'}})['body']
        self.assertEqual(got, {'model': 'm', 'messages': [], 'max_tokens': 128, 'temperature': 0,
                              'stream': False, 'tools': ['retained'], 'response_format': {'type': 'text'}})

    def test_tool_schema_property_order_survives_body_serialization(self):
        tools = [{'type': 'function', 'function': {'name': 'pick', 'description': 'ordered', 'parameters': {
            'type': 'object', 'properties': {'zebra': {'type': 'integer'}, 'apple': {'type': 'string'}},
            'required': ['zebra', 'apple']}}}]
        body = self.build('http://invalid', 'm', [{'role': 'user', 'content': 'x'}], tools=tools)['body']
        raw = MOD.ordered_json(body)
        self.assertIn(b'"function":{"name":"pick","description":"ordered","parameters":', raw)
        self.assertIn(b'"properties":{"zebra":{"type":"integer"},"apple":{"type":"string"}}', raw)
        self.assertNotEqual(raw, MOD.canonical(body))

    def test_changed_engine_and_plan_pin_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pin mismatch'):
            MOD.engine_body_builder(ENGINE.read_bytes() + b'\n')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                MOD.prepare(self.plan, '0' * 64, ENGINE, self.profile, Path(temp) / 'out')

    def test_all_113_fixture_ids_body_values_hashes_and_format_counts(self):
        plan = json.loads(self.plan.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'out'
            result = MOD.prepare(self.plan, MOD.digest(self.plan.read_bytes()), ENGINE, self.profile, output)
            self.assertEqual([r['id'] for r in result['requests']], [r['id'] for r in plan['requests']])
            self.assertEqual(len(result['requests']), 113)
            formats = {'json_object': 0, 'json_schema': 0, 'none': 0}
            for original, row in zip(plan['requests'], result['requests']):
                body_raw = (output / row['body_file']).read_bytes()
                self.assertEqual(hashlib.sha256(body_raw).hexdigest(), row['body_sha256'])
                body = json.loads(body_raw)
                args = original['request']
                self.assertEqual((output / row['engine_arguments_file']).read_bytes(), MOD.ordered_json(args))
                self.assertEqual(row['engine_arguments_canonical_sha256'], original['request_canonical_sha256'])
                expected_body = self.build('http://teacher-export.invalid', result['model_name'], **copy.deepcopy(args))['body']
                self.assertEqual(body_raw, MOD.ordered_json(expected_body))
                self.assertEqual(body['messages'], args['messages'])
                self.assertEqual(body['max_tokens'], args['max_tokens'])
                self.assertEqual(body.get('tools'), args['tools'])
                self.assertEqual(body.get('response_format'), args['response_format'])
                self.assertNotIn('extra_body', body)
                self.assertFalse(body['think'])
                for key, value in args['extra_body'].items():
                    self.assertEqual(body[key], value)
                formats[(body.get('response_format') or {}).get('type', 'none')] += 1
            self.assertEqual(formats, {'json_object': 20, 'json_schema': 5, 'none': 88})
            with self.assertRaisesRegex(ValueError, 'must be new'):
                MOD.prepare(self.plan, MOD.digest(self.plan.read_bytes()), ENGINE, self.profile, output)

    def test_duplicate_or_missing_original_ids_rejected(self):
        for change in ('duplicate', 'missing'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                plan = json.loads(self.plan.read_text(encoding='utf-8'))
                if change == 'duplicate':
                    plan['requests'][1]['id'] = plan['requests'][0]['id']
                else:
                    plan['requests'].pop()
                changed = Path(temp) / 'plan.json'
                changed.write_bytes(MOD.canonical(plan))
                with self.assertRaisesRegex(ValueError, '113 distinct'):
                    MOD.prepare(changed, MOD.digest(changed.read_bytes()), ENGINE, self.profile, Path(temp) / 'out')


if __name__ == '__main__':
    unittest.main()
