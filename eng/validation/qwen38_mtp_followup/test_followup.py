import copy
import importlib.util
import json
from pathlib import Path
import unittest


def load(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('qwen_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    # Execute exactly the read source; do not use a stale .pyc in an evidence test.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


c = load('compare')
h = load('http_contract')
b = load('bind_plan')


def fixture():
    rows, inputs = [], []
    for key in c.DENSE:
        counters = {'Drafted': 6, 'Accepted': 3, 'VerifySteps': 2, 'PlainSteps': 0, 'Rollbacks': 1}
        rows.append({'Label': key, 'Prompt': 3, 'Tokens': [10, 20, 30], 'OutTokens': 3, 'Finish': 'eos', 'Reused': 2,
                     **counters, 'TtftMs': 100, 'TotalMs': 300, 'DecodeTps': 10, 'PrefillTps': 30,
                     'RequestTimelines': [{'Id': key, 'Finish': 'eos', 'Error': None, 'OutTokens': 3, 'Speculation': counters.copy()}]})
        inputs.append({'id': key, 'tokens': [1, 2, 3], 'tokens_i32_sha256': c.token_sha([1, 2, 3]), 'prompt_tokens': 3,
                       'max_new_tokens': 128, 'verify_reserve': 4, 'maximum_position_exclusive': 135, 'media': key.startswith('image')})
    mtp = copy.deepcopy(rows)
    for row in rows:
        for field in ('Drafted', 'Accepted', 'VerifySteps', 'Rollbacks'):
            row[field] = row['RequestTimelines'][0]['Speculation'][field] = 0
    owner = {'exit_code': 0, 'before_identity_passed': True, 'after_identity_passed': True, 'process_gone': True, 'exclusive_timing': True,
             **dict.fromkeys(('native_sha256', 'managed_manifest_sha256', 'model_inventory_sha256', 'head_sha256'), 'a' * 64)}
    return {'plain': rows, 'mtp': mtp, 'plain_owner': owner.copy(), 'mtp_owner': owner.copy(), 'plain_inputs': {'inputs': copy.deepcopy(inputs), 'mode': 'plain', 'tier': 'dense', 'actual_head': 'PerToken'},
            'mtp_inputs': {'inputs': copy.deepcopy(inputs), 'mode': 'mtp', 'tier': 'dense', 'actual_head': 'PerToken'},
            'plain_memory': dict.fromkeys(('host_rss_peak_bytes', 'cgroup_current_peak_bytes', 'gpu_used_peak_bytes'), 1000),
            'mtp_memory': dict.fromkeys(('host_rss_peak_bytes', 'cgroup_current_peak_bytes', 'gpu_used_peak_bytes'), 1000)}


class Gates(unittest.TestCase):
    def pair(self, r):
        return c.pair(r['plain'], r['mtp'], r['plain_inputs'], r['mtp_inputs'])

    def test_complete_exact_pair_passes_but_not_release(self):
        result = self.pair(fixture())
        self.assertEqual('passed', result['status'])
        self.assertFalse(result['release_qualified'])

    def test_same_tokens_different_finish_fails(self):
        r = fixture(); r['mtp'][1]['Finish'] = 'max_tokens'
        self.assertEqual('failed', self.pair(r)['status'])

    def test_scheduler_token_budget_finish_requires_the_complete_budget(self):
        r = fixture()
        for mode in ('plain', 'mtp'):
            for row, inp in zip(r[mode], r[mode + '_inputs']['inputs']):
                row['Finish'] = row['RequestTimelines'][0]['Finish'] = 'max_tokens'
                inp['max_new_tokens'] = 3
                inp['maximum_position_exclusive'] = 10
        self.assertEqual('passed', self.pair(r)['status'])
        for mode in ('plain', 'mtp'):
            r[mode + '_inputs']['inputs'][1]['max_new_tokens'] = 4
            r[mode + '_inputs']['inputs'][1]['maximum_position_exclusive'] = 11
        self.assertTrue(any('incomplete token budget' in x for x in self.pair(r)['failures']))

    def test_longer_matching_prefix_is_not_full_parity(self):
        r = fixture(); r['mtp'][1]['Tokens'].append(40); r['mtp'][1]['OutTokens'] += 1
        self.assertTrue(any('parity' in x for x in self.pair(r)['failures']))

    def test_fallback_zero_verify_not_counted_as_engagement(self):
        r = fixture()
        for row in r['mtp']:
            row['VerifySteps'] = row['RequestTimelines'][0]['Speculation']['VerifySteps'] = 0
        self.assertTrue(any('never engaged' in x for x in self.pair(r)['failures']))

    def test_acceptance_and_rollback_must_actually_execute(self):
        for field in ('Accepted', 'Rollbacks'):
            r = fixture()
            for row in r['mtp']:
                row[field] = row['RequestTimelines'][0]['Speculation'][field] = 0
            self.assertEqual('failed', self.pair(r)['status'])

    def test_concurrent_aggregate_cannot_substitute_per_request_counters(self):
        r = fixture(); r['mtp'][7]['RequestTimelines'][0]['Speculation']['VerifySteps'] = 0
        self.assertTrue(any('disagree' in x for x in self.pair(r)['failures']))

    def test_concurrent_timeline_must_belong_to_the_actual_request(self):
        r = fixture(); r['mtp'][7]['RequestTimelines'][0]['Id'] = 'parallel4-i1'
        self.assertTrue(any('timeline identity' in x for x in self.pair(r)['failures']))

    def test_prompt_history_change_is_not_matched_workload(self):
        r = fixture(); inp = r['mtp_inputs']['inputs'][6]
        inp['tokens'] = [1, 2, 8]; inp['tokens_i32_sha256'] = c.token_sha(inp['tokens'])
        self.assertTrue(any('Rendered' in x for x in self.pair(r)['failures']))

    def test_dense_bound_includes_generation_and_verify_tail(self):
        r = fixture()
        for side in ('plain_inputs', 'mtp_inputs'):
            inp = r[side]['inputs'][1]; inp['max_new_tokens'] = 2045; inp['maximum_position_exclusive'] = 2052
        self.assertTrue(any('tier budget' in x for x in self.pair(r)['failures']))

    def test_missing_retained_reuse_not_called_supported(self):
        r = fixture(); r['mtp'][5]['Reused'] = 0
        self.assertTrue(any('actually reuse' in x for x in self.pair(r)['failures']))

    def test_three_pairs_required(self):
        with self.assertRaises(ValueError): c.compare_repeats([fixture()])

    def test_five_percent_timing_and_memory_fail_independently(self):
        repeats = [fixture() for _ in range(3)]
        for r in repeats:
            r['mtp'][1]['TtftMs'] = 106
            r['mtp_memory']['gpu_used_peak_bytes'] = 1060
        result = c.compare_repeats(repeats)
        self.assertTrue(any('TtftMs' in x for x in result['failures']))
        self.assertTrue(any('memory gate' in x for x in result['failures']))

    def test_one_outlier_does_not_change_three_repeat_median(self):
        repeats = [fixture() for _ in range(3)]; repeats[0]['mtp'][1]['TtftMs'] = 300
        self.assertEqual('passed', c.compare_repeats(repeats)['status'])

    def test_paired_concurrent_drift_is_not_repeat_determinism(self):
        repeats = [fixture() for _ in range(3)]
        for mode in ('plain', 'mtp'):
            repeats[1][mode][7]['Tokens'] = [10, 21, 30]
        self.assertTrue(all(self.pair(r)['status'] == 'passed' for r in repeats))
        result = c.compare_repeats(repeats)
        self.assertEqual('failed', result['status'])
        changed = [r for r in result['determinism'] if r['failures']]
        self.assertEqual(2, len(changed))
        self.assertTrue(all(r['id'] == 'parallel4-i0' and r['first_divergence'] == 1 for r in changed))

    def test_paired_finish_drift_is_not_repeat_determinism(self):
        repeats = [fixture() for _ in range(3)]
        for r in repeats:
            for mode in ('plain', 'mtp'):
                for inp in r[mode + '_inputs']['inputs']:
                    inp['max_new_tokens'] = 3
                    inp['maximum_position_exclusive'] = 10
        for mode in ('plain', 'mtp'):
            repeats[2][mode][7]['Finish'] = repeats[2][mode][7]['RequestTimelines'][0]['Finish'] = 'max_tokens'
        self.assertTrue(all(self.pair(r)['status'] == 'passed' for r in repeats))
        self.assertTrue(any('determinism failed' in x for x in c.compare_repeats(repeats)['failures']))

    def test_paired_prompt_drift_is_not_a_repeated_workload(self):
        repeats = [fixture() for _ in range(3)]
        for mode in ('plain', 'mtp'):
            inp = repeats[1][mode + '_inputs']['inputs'][7]
            inp['tokens'] = [1, 2, 8]; inp['tokens_i32_sha256'] = c.token_sha(inp['tokens'])
        self.assertTrue(all(self.pair(r)['status'] == 'passed' for r in repeats))
        self.assertTrue(any('changed across repeats' in x for x in c.compare_repeats(repeats)['failures']))

    def test_missing_memory_does_not_silently_pass(self):
        repeats = [fixture() for _ in range(3)]; del repeats[0]['mtp_memory']
        self.assertEqual('failed', c.compare_repeats(repeats)['status'])

    def test_failed_process_cannot_pass_matching_token_gate(self):
        repeats = [fixture() for _ in range(3)]; repeats[1]['mtp_owner']['exit_code'] = 1
        self.assertTrue(any('process failed' in x for x in c.compare_repeats(repeats)['failures']))

    def test_identity_or_competing_work_blocks_timing(self):
        repeats = [fixture() for _ in range(3)]; repeats[1]['mtp_owner']['exclusive_timing'] = False
        repeats[2]['mtp_owner']['native_sha256'] = 'b' * 64
        failures = c.compare_repeats(repeats)['failures']
        self.assertTrue(any('exclusive_timing' in x for x in failures)); self.assertTrue(any('same actual' in x for x in failures))

    def test_duplicate_row_rejected(self):
        r = fixture(); r['mtp'].append(r['mtp'][0])
        with self.assertRaises(ValueError): self.pair(r)

    def test_long_tier_never_release_qualified(self):
        r = fixture()
        for mode in ('plain', 'mtp'):
            r[mode] = r[mode][:-4]; r[mode + '_inputs']['inputs'] = r[mode + '_inputs']['inputs'][:-4]
            r[mode + '_inputs']['tier'] = 'long-qsa-unqualified'
        result = self.pair(r)
        self.assertEqual('passed', result['status']); self.assertFalse(result['release_qualified'])


class Http(unittest.TestCase):
    def test_exact_32_unique_stream_blocking_pairs(self):
        rows = h.catalog('m'); self.assertEqual(32, len(rows)); self.assertEqual(32, len({r['id'] for r in rows}))
        self.assertEqual(16, sum(r['expected_status'] == 400 for r in rows))
        self.assertEqual(16, sum(r['body']['stream'] for r in rows))
        self.assertTrue(all(r['body'].get('tools', []) is not None for r in rows))

    def test_invalid_contract_requires_actual_json_400(self):
        row = h.catalog('m')[0]; raw = json.dumps({'error': {'type': 'invalid_request_error', 'message': 'at least one client-declared function'}})
        self.assertTrue(h.judge(row, 400, 'application/json', raw)['api_rejected_before_stream'])
        with self.assertRaises(ValueError): h.judge(row, 200, 'application/json', raw)
        with self.assertRaises(ValueError): h.judge(row, 400, 'text/event-stream', raw)

    def test_raw_xml_is_not_a_successful_api_tool_call(self):
        row = next(r for r in h.catalog('m') if r['id'] == 'required-offered-blocking')
        raw = json.dumps({'choices': [{'message': {'content': '<tool_call><function=get_weather></function></tool_call>'}, 'finish_reason': 'stop'}],
                          'usage': {'completion_tokens': 12}})
        with self.assertRaises(ValueError): h.judge(row, 200, 'application/json', raw)

    def test_delta_arguments_reconstructed_without_losing_id(self):
        raw = '\n'.join('data: ' + json.dumps(x) for x in [
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'get_weather', 'arguments': '{"city":'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"Paris"}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'completion_tokens': 14}}]) + '\ndata: [DONE]\n'
        row = next(r for r in h.catalog('m') if r['id'] == 'required-offered-stream')
        result = h.judge(row, 200, 'text/event-stream', raw)
        self.assertEqual('call_1', result['message']['tool_calls'][0]['id'])
        with self.assertRaises(ValueError): h.judge(row, 200, 'text/event-stream', raw.replace('data: [DONE]', ''))

    def test_changed_streamed_tool_id_rejected(self):
        raw = '\n'.join('data: ' + json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': ident}]}}]}) for ident in ('a', 'b'))
        with self.assertRaises(ValueError): h.parse_success(raw, True)


class Bind(unittest.TestCase):
    def data(self):
        required = ['managed-build', 'c-abi-boundary'] + [f'qwen-target-{x}' for x in ('CPU', 'CUDA')] + [
            f'qwen-mtp-{x}-{capacity}-{alignment}' for x in ('CPU', 'CUDA') for capacity in (32, 512) for alignment in (4, 64)]
        managed = {'TensorSharp.Models.dll': 'a' * 64}
        build = {'final_source_and_binary_identity_passed': True, 'status': 'failed-gates', 'native_sha256': 'b' * 64,
                 'steps': [{'name': x, 'exit_code': 0, 'status': 'passed'} for x in required] + [{'name': 'tp-original', 'exit_code': 1, 'status': 'failed'}],
                 'managed_assemblies': managed}
        return build, {'status': 'passed', 'managed_assemblies': managed.copy()}, {'non_native_files_sha256': managed.copy()}

    def test_original_failed_gate_requires_explicit_unqualified_declaration(self):
        data = self.data()
        with self.assertRaises(ValueError): b.qualify(*data, False)
        report = b.qualify(*data, True)
        self.assertFalse(report['release_qualified']); self.assertEqual(['tp-original'], report['original_failed_steps'])

    def test_declaration_cannot_bypass_failed_qwen_fixture(self):
        data = self.data(); data[0]['steps'][2]['exit_code'] = 1
        with self.assertRaises(ValueError): b.qualify(*data, True)

    def test_declaration_cannot_bypass_unfinished_source_audit(self):
        data = self.data(); data[0]['final_source_and_binary_identity_passed'] = False
        with self.assertRaises(ValueError): b.qualify(*data, True)

    def test_unequal_managed_application_is_rejected(self):
        data = self.data(); data[2]['non_native_files_sha256']['TensorSharp.Models.dll'] = 'c' * 64
        with self.assertRaises(ValueError): b.qualify(*data, True)


if __name__ == '__main__':
    unittest.main()
