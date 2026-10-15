"""Model-free prompt/output linkage and exact consumed-history diagnostics."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'glm_prefix_token_diagnostic',
    Path(__file__).parents[1] / 'extract-glm-prefix-token-diagnostic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    row = {
        'name': 'synthetic', 'request_id': 'http1',
        'session': {'sessionId': 's1'}, 'completed': True, 'status': 200,
        'request': {
            'sessionId': 's1', 'messages': [{'role': 'user', 'content': 'test'}],
            'think': False, 'temperature': 0, 'topP': 1, 'topK': 0,
            'repeatPenalty': 1, 'presencePenalty': 0, 'frequencyPenalty': 0,
            'seed': 17, 'maxTokens': 3},
        'terminal': {
            'sessionId': 's1', 'done': True, 'aborted': False, 'error': None,
            'promptTokens': 10, 'tokenCount': 3, 'truncated': True,
            'kvReusedTokens': 8},
        'usage': {'prompt_tokens': 10, 'completion_tokens': 3,
                  'prompt_tokens_details': {'cached_tokens': 8}},
        'text': 'test', 'thinking': ''}

    def record(name, props, second):
        return {'ts': f'2026-10-14T00:00:0{second}Z',
                'event': {'name': name}, 'scope': {'RequestId': 'http1'},
                'props': props, 'template': name}

    records = [
        record('HttpRequestStarted', {'Method': 'POST', 'Path': '/api/chat'}, 0),
        record('generation.validation.input', {
            'InternalRequestId': 'chat1', 'SessionId': 's1',
            'Architecture': 'glm5-next', 'HistoryCompacted': False,
            'Thinking': False, 'MaxNewTokens': 3,
            'Tokens': ','.join(map(str, range(10)))}, 1),
        record('generation.timings', {
            'InternalRequestId': 'chat1', 'SessionId': 's1',
            'PromptTokens': 10, 'EngineCompletionAvailable': True,
            'EngineOutputTokens': 3, 'GeneratedTokens': 3,
            'FinishReason': 'max_tokens', 'KvReusedTokens': 8}, 2),
        record('generation.validation.output', {
            'InternalRequestId': 'chat1', 'FinishReason': 'max_tokens',
            'ComputedTokens': 13, 'Tokens': '10,11,12'}, 3),
        record('HttpRequestCompleted', {
            'Method': 'POST', 'Path': '/api/chat', 'StatusCode': 200}, 4)]
    return row, records


def extract(row, records):
    return module.extract(
        row, records, 'glm5-next',
        module.helpers.instant('2026-10-14T00:00:00Z'),
        module.helpers.instant('2026-10-14T00:00:05Z'),
        {'http': set(), 'internal': set()})


def trajectory_fixture():
    opening = extract(*fixture())
    opening['name'] = 'opening'
    followup = copy.deepcopy(opening)
    followup['name'] = 'followup'
    followup['prompt']['tokens'] = list(range(13)) + [99, 100]
    followup['cached_tokens'] = 13
    followup['log_lines']['http_start'] = 6
    return opening, followup


def rotated_fixture():
    source = '/capture/server-logs/tensorsharp-server-20261014.jsonl'
    snapshot = '/capture/http/server-log.jsonl'
    paths = [source,
             '/capture/server-logs/tensorsharp-server-20261014-0001.jsonl',
             '/capture/server-logs/tensorsharp-server-20261015.jsonl']
    chunks = [b'{"ordinal":0}\n', b'{"ordinal":1}\n', b'{"ordinal":2}\n']
    pins = [{'path': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for path, raw in zip(paths, chunks)]
    raw = b''.join(chunks)
    identity = {'schema': 'tensorsharp.rotated-log-capture.v1',
                'explicit_input': source, 'sources': pins,
                'combined_snapshot': {'path': snapshot, 'bytes': len(raw),
                                      'sha256': hashlib.sha256(raw).hexdigest()}}
    return identity, source, snapshot, raw


def validate_capture(identity, source, snapshot, raw):
    return module.validate_log_capture(identity, source, source, snapshot, snapshot, raw)


class GlmPrefixTokenDiagnosticTests(unittest.TestCase):
    def test_equal_token_hashes(self):
        result = module.compare_ids([1, 2], [1, 2])
        self.assertTrue(result['equal'])
        self.assertEqual(result['left_i32le_sha256'], result['right_i32le_sha256'])

    def test_first_divergent_token(self):
        result = module.compare_ids([1, 2, 3], [1, 4, 3])
        self.assertEqual({'index': 1, 'left': 2, 'right': 4}, result['first_difference'])

    def test_length_only_divergence(self):
        result = module.compare_ids([1, 2], [1, 2, 3])
        self.assertEqual({'index': 2, 'left': None, 'right': 3}, result['first_difference'])

    def test_prompt_plus_output_count(self):
        result = module.helpers.ar_commit_convention(10, 3, 13)
        self.assertTrue(result['terminal_output_forwarded'])

    def test_prompt_plus_output_minus_one_count(self):
        result = module.helpers.ar_commit_convention(10, 3, 12)
        self.assertFalse(result['terminal_output_forwarded'])

    def test_invalid_count_refused(self):
        with self.assertRaises(ValueError):
            module.helpers.ar_commit_convention(10, 3, 11)

    def test_invalid_token_csv_refused(self):
        with self.assertRaises(ValueError):
            module.helpers.token_csv('1,02')

    def test_exact_http_internal_count_linkage(self):
        result = extract(*fixture())
        self.assertEqual(13, result['output']['computed_tokens'])
        self.assertEqual(list(range(10)), result['prompt']['tokens'])
        self.assertEqual('http1', result['http_request_id'])
        self.assertEqual('chat1', result['internal_request_id'])

    def test_duplicate_generation_refused(self):
        row, records = fixture()
        with self.assertRaises(ValueError):
            extract(row, records[:3] + [records[2]] + records[3:])

    def test_wrong_internal_identity_refused(self):
        row, records = fixture()
        records[3]['props']['InternalRequestId'] = 'wrong'
        with self.assertRaises(ValueError):
            extract(row, records)

    def test_wrong_cached_count_refused(self):
        row, records = fixture()
        records[2]['props']['KvReusedTokens'] = 9
        with self.assertRaises(ValueError):
            extract(row, records)

    def test_complete_forwarded_history_cached_link(self):
        result = module.trajectory_check(*trajectory_fixture())
        self.assertTrue(result['cached_prefix_matches_observed_consumed_history'])
        self.assertTrue(result['cached_head_equals_opening_computed_head'])
        self.assertEqual(2, result['uncomputed_followup_tail_tokens'])

    def test_terminal_unforwarded_history_cached_link(self):
        opening, followup = trajectory_fixture()
        opening['output']['computed_tokens'] = 12
        followup['prompt']['tokens'] = list(range(12)) + [99, 100]
        followup['cached_tokens'] = 12
        result = module.trajectory_check(opening, followup)
        self.assertEqual(2, result['opening_output_consumed_count'])
        self.assertTrue(result['cached_prefix_matches_observed_consumed_history'])

    def test_raw_history_mismatch_reported_not_promoted(self):
        opening, followup = trajectory_fixture()
        followup['prompt']['tokens'][10] = 999
        result = module.trajectory_check(opening, followup)
        self.assertFalse(result['cached_prefix_matches_observed_consumed_history'])
        self.assertEqual(10, result['consumed_history_vs_followup_common_extent']['first_difference']['index'])

    def test_relative_and_absolute_cli_paths_are_identical(self):
        paths = {
            '--launch-audit': 'artifacts/example/launch-audit.json',
            '--http-report': 'artifacts/example/http/report.json',
            '--server-log': 'artifacts/example/http/server-log.jsonl',
            '--output': 'artifacts/example/diagnostic.json'}
        fixed = ['--expected-sdk-version', '26', '--expected-native-sha256', 'a' * 64]
        relative = module.parse_arguments([value for pair in paths.items() for value in pair] + fixed)
        absolute = module.parse_arguments([
            value for key, path in paths.items() for value in (key, str(Path(path).resolve()))] + fixed)
        self.assertEqual(vars(absolute), vars(relative))
        for name in ('launch_audit', 'http_report', 'server_log', 'output'):
            self.assertTrue(getattr(relative, name).is_absolute())


class GlmLogCaptureBindingTests(unittest.TestCase):
    def test_original_single_log_pin_remains_supported(self):
        _, source, snapshot, raw = rotated_fixture()
        identity = {'path': source, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        result = validate_capture(identity, source, snapshot, raw)
        self.assertEqual('single-file', result['kind'])

    def test_size_and_midnight_rotation_exact_byte_ranges(self):
        result = validate_capture(*rotated_fixture())
        self.assertEqual(3, result['source_count'])
        self.assertTrue(result['source_hashes_verified_against_snapshot_ranges'])
        self.assertFalse(result['live_source_files_reopened'])

    def test_same_size_source_tampering_is_refused(self):
        identity, source, snapshot, raw = rotated_fixture()
        identity['sources'][1]['sha256'] = 'a' * 64
        with self.assertRaises(ValueError):
            validate_capture(identity, source, snapshot, raw)

    def test_snapshot_pin_target_must_be_exact_http_snapshot(self):
        identity, source, snapshot, raw = rotated_fixture()
        identity['combined_snapshot']['path'] = '/capture/other/server-log.jsonl'
        with self.assertRaises(ValueError):
            validate_capture(identity, source, snapshot, raw)

    def test_source_order_is_checked_even_if_concatenation_is_self_consistent(self):
        identity, source, snapshot, raw = rotated_fixture()
        chunks = raw.splitlines(keepends=True)
        identity['sources'][0], identity['sources'][1] = identity['sources'][1], identity['sources'][0]
        raw = chunks[1] + chunks[0] + chunks[2]
        identity['combined_snapshot']['sha256'] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):
            validate_capture(identity, source, snapshot, raw)

    def test_rotation_gap_and_foreign_parent_are_refused(self):
        for field, value in (
                ('path', '/capture/server-logs/tensorsharp-server-20261014-0002.jsonl'),
                ('path', '/capture/other/tensorsharp-server-20261014-0001.jsonl')):
            with self.subTest(value=value):
                identity, source, snapshot, raw = rotated_fixture()
                identity['sources'][1][field] = value
                with self.assertRaises(ValueError):
                    validate_capture(identity, source, snapshot, raw)

    def test_duplicate_source_and_unbounded_date_family_are_refused(self):
        for path in ('/capture/server-logs/tensorsharp-server-20261014.jsonl',
                     '/capture/server-logs/tensorsharp-server-20261016.jsonl'):
            with self.subTest(path=path):
                identity, source, snapshot, raw = rotated_fixture()
                identity['sources'][2]['path'] = path
                with self.assertRaises(ValueError):
                    validate_capture(identity, source, snapshot, raw)

    def test_source_byte_sum_must_cover_exact_combined_payload(self):
        identity, source, snapshot, raw = rotated_fixture()
        identity['sources'] = identity['sources'][:-1]
        with self.assertRaises(ValueError):
            validate_capture(identity, source, snapshot, raw)

    def test_incomplete_prior_file_is_refused_without_inserting_newline(self):
        identity, source, snapshot, raw = rotated_fixture()
        chunks = raw.splitlines(keepends=True)
        chunks[0] = chunks[0][:-1]
        identity['sources'][0].update(bytes=len(chunks[0]), sha256=hashlib.sha256(chunks[0]).hexdigest())
        raw = b''.join(chunks)
        identity['combined_snapshot'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError):
            validate_capture(identity, source, snapshot, raw)

    def test_unknown_schema_and_explicit_input_mismatch_are_refused(self):
        for field, value in (('schema', 'unknown.v1'),
                             ('explicit_input', '/capture/server-logs/tensorsharp-server-20261013.jsonl')):
            with self.subTest(field=field):
                identity, source, snapshot, raw = rotated_fixture()
                identity[field] = value
                with self.assertRaises(ValueError):
                    validate_capture(identity, source, snapshot, raw)


if __name__ == '__main__':
    unittest.main()
