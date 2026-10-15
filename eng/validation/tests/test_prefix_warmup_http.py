"""Loopback fixture/controller validation only; no actual Host/model/device."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
import uuid

spec = importlib.util.spec_from_file_location('prefix_warmup_http_tests', Path(__file__).parents[1] / 'prefix-warmup-http.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Handler(BaseHTTPRequestHandler):
    def json(self, value, status=200):
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def do_GET(self):
        if self.path == '/api/models': self.json({'model': 'controller-fixture', 'unifiedMemory': True})
        elif self.path == '/api/queue/status': self.json({'busy': False, 'pending_requests': 0})
        elif self.path.startswith('/history/') and self.server.history_enabled:
            session = self.path.split('/')[-1]
            self.json({'sessionId': session, 'source': 'production-wire', 'messages': self.server.histories[session]})
        else: self.json({'error': 'No production history endpoint'}, 404)

    def do_DELETE(self):
        session = self.path.split('/')[-1]
        self.json({'ok': True, 'sessionId': session}, 503 if self.server.fail_cleanup else 200)

    def do_POST(self):
        if self.path == '/api/sessions':
            self.server.session_count += 1
            self.json({'sessionId': 'owned-' + str(self.server.session_count)})
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.bodies.append(body)
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Connection', 'close'); self.end_headers()
        def send(value):
            self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode('utf-8')); self.wfile.flush()
        prompt = body['messages'][-1]['content']
        fragments = ['FF7', ' literal answer'] if '最终幻想' in prompt else ['Time', ' literal answer']
        if self.server.different_repeat and len(self.server.bodies) == 3: fragments[-1] += '!'
        send({'status': 'preparing'})
        if self.server.tool_frame: send({'tool_calls': [{'name': 'unsupported-fixture', 'arguments': {}}]})
        if body['think']:
            reasoning = 'retained reasoning' + ('!' if self.server.different_thinking and len(self.server.bodies) == 3 else '')
            send({'thinking': reasoning})
        if not self.server.thinking_only:
            for fragment in fragments: send({'token': fragment})
        elif not body['think']:
            send({'thinking': 'hidden-only fixture'})
        followup = len(body['messages']) > 1
        cached = self.server.followup_cached if followup else self.server.cached
        if followup and body['sessionId'] == 'owned-1' and self.server.followup_older_cached is not None:
            cached = self.server.followup_older_cached
        prompt_tokens = 400 if followup else 300
        history = copy.deepcopy(body['messages'])
        history.append({'role': 'assistant', 'content': ''.join(fragments), 'thinking': 'retained reasoning' if body['think'] else ''})
        self.server.histories[body['sessionId']] = history
        props = dict(SessionId=body['sessionId'], InternalRequestId=uuid.uuid4().hex,
                     TimingBasis='attributed_forward', PromptTokens=prompt_tokens, KvReusedTokens=cached,
                     GeneratedTokens=body['maxTokens'], TotalNs=4000000, EngineCompletionAvailable=True,
                     PromptNs=2000000, EvalNs=1600000, EngineOutputTokens=body['maxTokens'], FinishReason='max_tokens')
        if self.server.cancel_log and body['maxTokens'] == 128: props['FinishReason'] = 'cancelled'
        if self.server.bad_log_cache: props['KvReusedTokens'] += 1
        if self.server.bad_log_session: props['SessionId'] = 'other-user-session'
        if not self.server.omit_log:
            with self.server.log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(template='generation.timings', scope={'RequestId': self.headers['X-TensorSharp-Request-Id']}, props=props)) + '\n')
        if not self.server.omit_terminal and not (self.server.omit_long_terminal and body['maxTokens'] == 256):
            try:
                send(dict(done=True, tokenCount=body['maxTokens'], promptTokens=prompt_tokens, kvReusedTokens=cached,
                          aborted=False, truncated=True, error=None, sessionId=body['sessionId']))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError): pass

    def log_message(self, *args): pass


class PrefixWarmupHttpTests(unittest.TestCase):
    def setUp(self):
        (bench.ROOT / 'artifacts').mkdir(exist_ok=True)
        self.tmp = TemporaryDirectory(dir=bench.ROOT / 'artifacts')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / 'server.jsonl'; self.log.write_text('', encoding='utf-8')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        for key, value in dict(session_count=0, bodies=[], cached=100, different_repeat=False, omit_terminal=False,
                               omit_log=False, bad_log_cache=False, bad_log_session=False, fail_cleanup=False,
                               thinking_only=False, different_thinking=False, cancel_log=False, history_enabled=False, histories={},
                               followup_cached=320, followup_older_cached=None, tool_frame=False, omit_long_terminal=False, log=self.log).items(): setattr(self.server, key, value)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def run_tool(self, *extra):
        output = self.root / 'run'
        result = bench.main(['--base-url', self.url, '--output', str(output), '--server-log', str(self.log), '--timeout', '2', *extra])
        return result, json.loads((output / 'report.json').read_text('utf-8'))

    def test_live_loopback_preserves_catalog_and_links_usage_and_timings(self):
        code, report = self.run_tool()
        self.assertEqual(code, 0); self.assertTrue(report['passed'])
        self.assertEqual([b['messages'][0]['content'] for b in self.server.bodies], list(bench.PROMPTS))
        for body in self.server.bodies:
            for field in ('tools', 'skills', 'skills_discovery', 'multi_agent', 'tool_choice'):
                self.assertNotIn(field, body)
        self.assertEqual(len({b['sessionId'] for b in self.server.bodies}), 3)
        self.assertTrue(report['server_timing_linkage']['passed'])
        self.assertEqual(report['public_prefix_observation']['cached_tokens'], [100, 100, 100])
        self.assertTrue(report['repeat_parity']['passed'])
        self.assertFalse(report['gpu_parallelism_claim']); self.assertFalse(report['model_semantic_quality_claim'])
        self.assertTrue(all(row['passed'] for row in report['sessions_cleanup']))
        self.assertEqual(report['source_pins_before'], report['source_pins_after'])

    def test_prefix_disabled_is_a_distinct_explicit_control(self):
        self.server.cached = 0
        code, report = self.run_tool('--cache-expectation', 'disabled')
        self.assertEqual(code, 0); self.assertTrue(report['public_prefix_observation']['passed'])

    def test_explicit_eight_token_cap_is_preserved_in_every_request(self):
        code, report = self.run_tool('--max-tokens', '8')
        self.assertEqual(code, 0)
        self.assertTrue(all(body['maxTokens'] == 8 for body in self.server.bodies))

    def test_enabled_prefix_missing_actual_reuse_fails(self):
        self.server.cached = 0
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['public_prefix_observation']['passed'])

    def test_observe_does_not_count_a_cache_gate_as_passing(self):
        self.server.cached = 0
        code, report = self.run_tool('--cache-expectation', 'observe')
        self.assertEqual(code, 0); self.assertIsNone(report['public_prefix_observation']['passed'])

    def test_repeat_output_difference_fails_literal_parity(self):
        self.server.different_repeat = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['repeat_parity']['checks']['exact_visible_text'])

    def test_missing_terminal_is_not_passing_stream_execution(self):
        self.server.omit_terminal = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertIn('Missing terminal', report['serial'][0]['error'])

    def test_missing_current_log_evidence_fails(self):
        self.server.omit_log = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['server_timing_linkage']['passed'])

    def test_cached_usage_must_match_the_actual_generation_log(self):
        self.server.bad_log_cache = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['server_timing_linkage']['usage_links'][0]['cached_matches'])

    def test_unrelated_session_timings_cannot_satisfy_the_gate(self):
        self.server.bad_log_session = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['server_timing_linkage']['passed'])

    def test_session_cleanup_failure_remains_a_failure(self):
        self.server.fail_cleanup = True
        code, report = self.run_tool()
        self.assertEqual(code, 1); self.assertFalse(report['sessions_cleanup'][0]['passed'])

    def test_thinking_and_content_ttft_are_separate_observations(self):
        code, report = self.run_tool('--think', 'on')
        self.assertEqual(code, 0)
        row = report['serial'][0]
        self.assertEqual(row['thinking'], 'retained reasoning')
        self.assertLessEqual(row['client_first_raw_ttft_ms'], row['client_first_content_ttft_ms'])

    def test_two_empty_visible_outputs_are_not_quality_or_parity_passes(self):
        self.server.thinking_only = True
        code, report = self.run_tool('--think', 'on')
        self.assertEqual(code, 1); self.assertEqual(report['repeat_parity']['status'], 'inconclusive_no_visible_output')
        self.assertIsNone(report['serial'][0]['client_first_content_ttft_ms'])

    def test_thinking_only_stream_gate_preserves_inconclusive_visible_quality(self):
        self.server.thinking_only = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream', '--cache-expectation', 'enabled')
        self.assertEqual(code, 0); self.assertTrue(report['passed'])
        self.assertTrue(report['stream_repeat_parity']['passed'])
        self.assertFalse(report['repeat_parity']['passed'])
        self.assertEqual(report['repeat_parity']['status'], 'inconclusive_no_visible_output')
        self.assertFalse(report['output_acceptance_gate']['visible_quality_passed'])
        for row in report['serial']:
            self.assertGreater(row['client_first_raw_ttft_ms'], 0)
            self.assertIsNone(row['client_first_content_ttft_ms'])
            self.assertEqual(row['thinking'], 'retained reasoning')

    def test_thinking_only_visible_gate_is_still_strict_when_explicitly_on(self):
        self.server.thinking_only = True
        code, report = self.run_tool('--think', 'on')
        self.assertEqual(code, 1); self.assertFalse(report['repeat_parity']['passed'])
        self.assertTrue(report['stream_repeat_parity']['passed'])
        self.assertEqual(report['output_acceptance_gate']['mode'], 'visible')

    def test_stream_gate_detects_changed_literal_thinking(self):
        self.server.thinking_only = True; self.server.different_thinking = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream')
        self.assertEqual(code, 1)
        self.assertFalse(report['stream_repeat_parity']['checks']['exact_thinking'])
        self.assertFalse(report['stream_repeat_parity']['checks']['exact_emitted_model_frames'])

    def test_stream_gate_detects_changed_content_even_when_thinking_matches(self):
        self.server.different_repeat = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream')
        self.assertEqual(code, 1)
        self.assertFalse(report['stream_repeat_parity']['checks']['exact_visible_text'])

    def test_stream_gate_refuses_tool_frames(self):
        self.server.tool_frame = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream')
        self.assertEqual(code, 1); self.assertFalse(report['stream_repeat_parity']['checks']['no_tool_frames'])

    def test_stream_gate_requires_explicit_thinking_on(self):
        with self.assertRaisesRegex(ValueError, 'explicit thinking on'):
            self.run_tool('--output-gate', 'stream')

    def test_stream_gate_does_not_claim_production_history_followup(self):
        for flag in ('--follow-up', '--client-visible-follow-up'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'production-history'):
                self.run_tool('--think', 'on', '--output-gate', 'stream', flag)

    def test_requested_followup_without_production_history_fails_inconclusively(self):
        code, report = self.run_tool('--follow-up')
        self.assertEqual(code, 1); self.assertEqual(report['follow_up']['status'], 'inconclusive_failure')
        self.assertEqual(len(self.server.bodies), 3)

    def test_cancellation_requires_actual_server_evidence_not_just_disconnect(self):
        code, report = self.run_tool('--cancellation')
        self.assertEqual(code, 1)
        self.assertTrue(report['cancelled_request']['cancelled_as_requested'])
        self.assertFalse(report['cancellation']['passed'])
        self.assertTrue(report['serial'][-1]['completed'])

    def test_server_cancel_queue_and_fresh_recovery_are_all_observed(self):
        self.server.cancel_log = True
        code, report = self.run_tool('--cancellation')
        self.assertEqual(code, 0)
        self.assertTrue(report['cancellation']['passed'])
        self.assertTrue(report['cancellation']['queue_recovered'])
        self.assertEqual(report['serial'][-1]['name'], 'recovery-new-chat')
        self.assertTrue(report['serial'][-1]['completed'])

    def test_no_reconstructed_history_when_the_endpoint_is_unavailable(self):
        code, report = self.run_tool('--follow-up', '--history-route', '/history/{sessionId}')
        self.assertEqual(code, 1); self.assertEqual(report['follow_up']['status'], 'inconclusive_failure')
        self.assertEqual(len(self.server.bodies), 3)

    def test_actual_production_history_and_private_reuse_are_both_required(self):
        self.server.history_enabled = True
        code, report = self.run_tool('--follow-up', '--history-route', '/history/{sessionId}', '--think', 'on')
        self.assertEqual(code, 0)
        self.assertTrue(report['follow_up']['passed'])
        self.assertEqual(report['follow_up']['reused_tokens'], [320, 320])
        self.assertEqual(report['follow_up']['beyond_prior_whole_prompt_threshold'], 300)
        self.assertTrue(report['follow_up']['repeat_parity']['passed'])
        self.assertEqual(self.server.bodies[3]['messages'][1]['thinking'], 'retained reasoning')

    def test_public_only_reuse_cannot_pass_a_private_followup_gate(self):
        self.server.history_enabled = True
        self.server.followup_cached = 100
        code, report = self.run_tool('--follow-up', '--history-route', '/history/{sessionId}')
        self.assertEqual(code, 1)
        self.assertFalse(report['follow_up']['passed'])
        self.assertEqual(report['follow_up']['reused_tokens'], [100, 100])

    def test_history_preserves_production_thinking_and_extra_wire_fields(self):
        session = {'sessionId': 'owned'}; prior = {'text': 'answer', 'thinking': 'unfinished thinking'}
        history = {'sessionId': 'owned', 'source': 'production-wire', 'messages': [
            {'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'answer', 'thinking': 'unfinished thinking', 'wire_field': 7}]}
        result = bench.production_history(history, session, prior)
        self.assertEqual(result, history['messages'])
        self.assertIsNot(result, history['messages'])
        result[-1]['content'] = 'changed'
        self.assertEqual(history['messages'][-1]['content'], 'answer')

    def test_mismatched_history_thinking_is_not_repaired(self):
        history = {'sessionId': 'owned', 'source': 'production-wire', 'messages': [
            {'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'answer', 'thinking': 'wrong'}]}
        with self.assertRaisesRegex(ValueError, 'thinking differs'):
            bench.production_history(history, {'sessionId': 'owned'}, {'text': 'answer', 'thinking': 'actual'})

    def test_wrong_session_or_unverified_history_source_is_rejected(self):
        for session_id, source in (('other', 'production-wire'), ('owned', 'reconstructed')):
            with self.subTest(session_id=session_id, source=source):
                history = {'sessionId': session_id, 'source': source, 'messages': [
                    {'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'answer'}]}
                with self.assertRaisesRegex(ValueError, 'production-wire session identity'):
                    bench.production_history(history, {'sessionId': 'owned'}, {'text': 'answer', 'thinking': ''})

    def test_replacement_frames_preserve_exact_visible_content(self):
        row = dict(frames=[], visible_frames=[], raw_frames=0, text='', thinking='', completed=False,
                   first_raw_at=None, first_content_at=None, tool_frames_observed=False)
        bench.consume_frame(row, {'token': 'provisional'}, 1)
        bench.consume_frame(row, {'replace': 'corrected'}, 2)
        bench.consume_frame(row, {'token': ' tail'}, 3)
        self.assertEqual(row['text'], 'corrected tail')
        self.assertEqual(row['visible_frames'], [{'token': 'provisional'}, {'replace': 'corrected'}, {'token': ' tail'}])

    def test_boolean_or_excess_cached_usage_is_rejected(self):
        terminal = dict(done=True, aborted=False, error=None, promptTokens=10, tokenCount=8, kvReusedTokens=0, truncated=True)
        for field, value in (('promptTokens', True), ('kvReusedTokens', 11)):
            with self.subTest(field=field):
                bad = copy.deepcopy(terminal); bad[field] = value
                with self.assertRaises(ValueError): bench.usage_of(bad)

    def test_client_visible_replay_uses_actual_content_and_thinking_without_history_provider(self):
        code, report = self.run_tool('--client-visible-follow-up', '--think', 'on')
        self.assertEqual(code, 0); self.assertFalse(self.server.history_enabled)
        self.assertTrue(report['client_visible_follow_up']['passed'])
        self.assertEqual(report['client_visible_follow_up']['reused_tokens'], [320, 320])
        self.assertTrue(report['client_visible_follow_up']['repeat_parity']['passed'])
        self.assertEqual(self.server.bodies[3]['sessionId'], 'owned-3')
        self.assertEqual(self.server.bodies[4]['sessionId'], 'owned-1')
        assistant = self.server.bodies[3]['messages'][1]
        self.assertEqual(assistant, {'role': 'assistant', 'content': 'FF7 literal answer', 'thinking': 'retained reasoning'})
        self.assertNotIn('RawOutputTokens', assistant); self.assertNotIn('rawOutputTokens', assistant)
        self.assertIsNone(report['follow_up']['passed'])

    def test_longer_client_visible_continuation_cap_preserves_initial_new_chat_workload(self):
        code, report = self.run_tool('--client-visible-follow-up', '--follow-up-max-tokens', '128')
        self.assertEqual(code, 0)
        self.assertEqual([body['maxTokens'] for body in self.server.bodies], [16, 16, 16, 128, 128])
        self.assertEqual([row['usage']['completion_tokens'] for row in report['serial']], [16, 16, 16, 128, 128])
        self.assertTrue(report['client_visible_follow_up']['repeat_parity']['passed'])
        self.assertEqual(report['arguments']['follow_up_max_tokens'], 128)

    def test_production_history_continuation_uses_its_separate_cap(self):
        self.server.history_enabled = True
        code, report = self.run_tool('--follow-up', '--history-route', '/history/{sessionId}', '--follow-up-max-tokens', '64')
        self.assertEqual(code, 0)
        self.assertEqual([body['maxTokens'] for body in self.server.bodies], [16, 16, 16, 64, 64])
        self.assertTrue(report['follow_up']['repeat_parity']['passed'])

    def test_followup_cap_without_requested_continuation_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'explicitly requested followup'):
            self.run_tool('--follow-up-max-tokens', '128')

    def test_long_new_chat_preserves_three_requests_and_records_visible_latency_without_quality_claim(self):
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream',
            '--long-new-chat-max-tokens', '256', '--long-new-chat-expected-cached-tokens', '100')
        self.assertEqual(code, 0)
        self.assertEqual([body['maxTokens'] for body in self.server.bodies], [16, 16, 16, 256])
        self.assertEqual(len({body['sessionId'] for body in self.server.bodies}), 4)
        self.assertEqual(self.server.bodies[3]['messages'], [{'role': 'user', 'content': bench.PROMPTS[0]}])
        probe = report['long_new_chat']
        self.assertTrue(probe['passed']); self.assertEqual(probe['visible_observation_status'], 'observed')
        self.assertGreater(probe['client_first_content_ttft_ms'], 0)
        self.assertLessEqual(probe['client_first_thinking_ttft_ms'], probe['client_first_content_ttft_ms'])
        self.assertIsNone(probe['semantic_quality_passed']); self.assertIsNone(probe['repeat_parity_passed'])
        self.assertTrue(all(row['passed'] for row in report['sessions_cleanup']))

    def test_long_new_chat_without_visible_content_stays_explicitly_inconclusive(self):
        self.server.thinking_only = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream',
            '--long-new-chat-max-tokens', '256', '--long-new-chat-expected-cached-tokens', '100')
        self.assertEqual(code, 0); self.assertTrue(report['long_new_chat']['passed'])
        self.assertEqual(report['long_new_chat']['visible_observation_status'], 'inconclusive_no_visible_output')
        self.assertIsNone(report['long_new_chat']['client_first_content_ttft_ms'])
        self.assertIsNone(report['long_new_chat']['semantic_quality_passed'])
        self.assertFalse(report['output_acceptance_gate']['visible_quality_passed'])

    def test_long_new_chat_requires_exact_public_cached_usage_and_actual_timing_link(self):
        for mutation in ('cache', 'timing'):
            with self.subTest(mutation=mutation):
                output = self.root / ('long-' + mutation)
                if mutation == 'timing': self.server.bad_log_cache = True
                code = bench.main(['--base-url', self.url, '--output', str(output), '--server-log', str(self.log),
                    '--timeout', '2', '--think', 'on', '--output-gate', 'stream', '--long-new-chat-max-tokens', '256',
                    '--long-new-chat-expected-cached-tokens', '101' if mutation == 'cache' else '100'])
                report = json.loads((output / 'report.json').read_text('utf-8'))
                self.assertEqual(code, 1); self.assertFalse(report['long_new_chat']['passed'])
                check = 'exact_public_cached_tokens' if mutation == 'cache' else 'measured_completion'
                self.assertFalse(report['long_new_chat']['checks'][check])

    def test_long_new_chat_refuses_wrong_mode_missing_expectation_or_mixed_scope_before_http(self):
        valid = ['--think', 'on', '--output-gate', 'stream', '--long-new-chat-max-tokens', '256']
        for extra in (valid, ['--long-new-chat-max-tokens', '256', '--long-new-chat-expected-cached-tokens', '100'],
                      valid + ['--long-new-chat-expected-cached-tokens', '0'],
                      valid + ['--long-new-chat-expected-cached-tokens', '100', '--cancellation']):
            with self.subTest(extra=extra), self.assertRaises(ValueError): self.run_tool(*extra)
        self.assertEqual(self.server.bodies, [])

    def test_long_new_chat_missing_terminal_is_not_a_completed_observation(self):
        self.server.omit_long_terminal = True
        code, report = self.run_tool('--think', 'on', '--output-gate', 'stream',
            '--long-new-chat-max-tokens', '256', '--long-new-chat-expected-cached-tokens', '100')
        self.assertEqual(code, 1); self.assertFalse(report['long_new_chat']['passed'])
        self.assertFalse(report['completed']); self.assertIn('did not complete', report['error'])
        self.assertTrue(all(row['passed'] for row in report['sessions_cleanup']))

    def test_client_visible_public_only_reuse_fails_private_gate(self):
        self.server.followup_cached = 100
        code, report = self.run_tool('--client-visible-follow-up')
        self.assertEqual(code, 1); self.assertFalse(report['client_visible_follow_up']['passed'])
        self.assertEqual(report['client_visible_follow_up']['reuse_candidate']['prior_prompt_tokens'], 300)

    def test_recent_hit_and_legitimate_older_eviction_still_allow_literal_reference(self):
        self.server.followup_older_cached = 100
        code, report = self.run_tool('--client-visible-follow-up')
        self.assertEqual(code, 0); self.assertTrue(report['client_visible_follow_up']['passed'])
        self.assertEqual(report['client_visible_follow_up']['reused_tokens'], [320, 100])
        self.assertTrue(report['client_visible_follow_up']['reuse_candidate']['passed'])
        self.assertFalse(report['client_visible_follow_up']['reference']['beyond_prior_prompt'])
        self.assertTrue(report['client_visible_follow_up']['repeat_parity']['passed'])

    def test_client_visible_tool_history_is_refused_without_replay(self):
        self.server.tool_frame = True
        code, report = self.run_tool('--client-visible-follow-up')
        self.assertEqual(code, 1); self.assertEqual(report['client_visible_follow_up']['status'], 'inconclusive_failure')
        self.assertEqual(len(self.server.bodies), 3)

    def test_client_visible_requires_initial_measured_engine_completion(self):
        self.server.omit_log = True
        code, report = self.run_tool('--client-visible-follow-up')
        self.assertEqual(code, 1); self.assertEqual(len(self.server.bodies), 3)
        self.assertFalse(report['client_visible_follow_up']['initial_timing_linkage']['passed'])

    @staticmethod
    def visible_prior():
        row = dict(frames=[], visible_frames=[], raw_frames=0, text='', thinking='', completed=False,
                   first_raw_at=None, first_content_at=None, tool_frames_observed=False, request_id='owned-request',
                   session={'sessionId': 'owned'}, request={'messages': [{'role': 'user', 'content': 'exact original'}]})
        bench.consume_frame(row, {'token': 'provisional reasoning'}, 1)
        bench.consume_frame(row, {'thinking': 'exact reasoning\n '}, 2)
        bench.consume_frame(row, {'replace': 'final visible '}, 3)
        bench.consume_frame(row, {'token': 'tail\n'}, 4)
        bench.consume_frame(row, dict(done=True, tokenCount=16, promptTokens=300, kvReusedTokens=100,
                                     aborted=False, truncated=True, error=None, sessionId='owned'), 5)
        measured = {'rows': [{'http_request_id': 'owned-request', 'engine_completion_available': True}]}
        return row, measured

    def test_client_visible_replays_replacement_result_and_literal_whitespace(self):
        prior, timings = self.visible_prior()
        messages = bench.client_visible_history(prior, timings)
        self.assertEqual(messages, [{'role': 'user', 'content': 'exact original'},
            {'role': 'assistant', 'content': 'final visible tail\n', 'thinking': 'exact reasoning\n '}])
        self.assertNotIn('provisional', messages[-1]['content'])
        messages[0]['content'] = 'changed'
        self.assertEqual(prior['request']['messages'][0]['content'], 'exact original')

    def test_client_visible_incomplete_empty_or_unsupported_metadata_is_refused(self):
        for field, value in (('completed', False), ('text', ''), ('tool_frames_observed', True)):
            with self.subTest(field=field):
                prior, timings = self.visible_prior(); prior[field] = value
                with self.assertRaises(ValueError): bench.client_visible_history(prior, timings)
        prior, timings = self.visible_prior()
        prior['frames'].insert(-1, [4.5, {'artifact': {'unavailable': True}}])
        with self.assertRaisesRegex(ValueError, 'unsupported frame metadata'): bench.client_visible_history(prior, timings)

    def test_client_visible_multiple_rounds_or_frames_after_terminal_are_refused(self):
        prior, timings = self.visible_prior(); timings['rows'] *= 2
        with self.assertRaisesRegex(ValueError, 'exactly one measured engine round'): bench.client_visible_history(prior, timings)
        prior, timings = self.visible_prior(); prior['frames'].append([6, {'token': 'after-terminal'}])
        with self.assertRaisesRegex(ValueError, 'terminal to be last'): bench.client_visible_history(prior, timings)


class RotatedLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(dir=bench.ROOT/'artifacts'); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.output = self.root/'snapshot'; self.output.mkdir()

    def write(self, name, value):
        path = self.root/name; path.write_bytes(value); return path

    def test_utc_and_size_rotation_preserve_order_each_original_pin_and_exact_snapshot(self):
        paths = [self.write(name, (json.dumps({'part': index})+'\n').encode()) for index,name in enumerate(
            ('tensorsharp-server-20261014.jsonl','tensorsharp-server-20261014-0001.jsonl','tensorsharp-server-20261015.jsonl'))]
        self.assertEqual(bench.logger_paths(paths[0]), paths)
        rows, identity = bench.read_records(paths[0], self.output)
        self.assertEqual(rows, [{'part': index} for index in range(3)])
        self.assertEqual(identity['sources'], [bench.pin(path) for path in paths])
        self.assertEqual((self.output/'server-log.jsonl').read_bytes(), b''.join(path.read_bytes() for path in paths))
        self.assertEqual(identity['combined_snapshot'], bench.pin(self.output/'server-log.jsonl'))

    def test_generic_single_log_retains_its_original_identity(self):
        path = self.write('fixture.jsonl', b'{"part":1}\n')
        rows, identity = bench.read_records(path, self.output)
        self.assertEqual(rows,[{'part':1}]);self.assertEqual(identity,bench.pin(path))

    def test_malformed_logger_member_and_calendar_date_are_refused(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{}\n')
        bad=self.write('tensorsharp-server-20261015-bad.jsonl',b'{}\n')
        with self.assertRaisesRegex(ValueError,'Malformed canonical'):bench.logger_paths(path)
        bad.unlink();bad=self.write('tensorsharp-server-20261314.jsonl',b'{}\n')
        with self.assertRaisesRegex(ValueError,'UTC date'):bench.logger_paths(bad)

    def test_rotation_index_gaps_and_duplicate_zero_are_refused(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{}\n')
        for name in ('tensorsharp-server-20261014-0002.jsonl','tensorsharp-server-20261014-0000.jsonl'):
            bad=self.write(name,b'{}\n')
            with self.subTest(name=name),self.assertRaises(ValueError):bench.logger_paths(path)
            bad.unlink()

    def test_nonadjacent_or_earlier_dates_are_refused(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{}\n')
        for day in ('20261013','20261016'):
            bad=self.write('tensorsharp-server-'+day+'.jsonl',b'{}\n')
            with self.subTest(day=day),self.assertRaisesRegex(ValueError,'adjacent2'):bench.logger_paths(path)
            bad.unlink()

    def test_rotation_file_count_is_bounded(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{}\n')
        for index in range(1,33):self.write(f'tensorsharp-server-20261014-{index:04}.jsonl',b'{}\n')
        with self.assertRaisesRegex(ValueError,'bounded32'):bench.logger_paths(path)

    def test_aggregate_bound_applies_across_individually_valid_files(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{"part":1}\n')
        self.write('tensorsharp-server-20261015.jsonl',b'{"part":2}\n')
        with patch.object(bench.checker,'MAX_LOG_BYTES',16),self.assertRaisesRegex(ValueError,'aggregate'):
            bench.read_records(path,self.output)

    def test_incomplete_prior_rotated_file_is_not_joined_into_an_invented_record(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{"part":1}')
        self.write('tensorsharp-server-20261015.jsonl',b'{"part":2}\n')
        with self.assertRaisesRegex(ValueError,'incomplete prior'):bench.read_records(path,self.output)

    def test_capture_retries_changed_family_and_refuses_persistent_churn(self):
        path=self.write('tensorsharp-server-20261014.jsonl',b'{"part":1}\n')
        second=self.write('tensorsharp-server-20261015.jsonl',b'{"part":2}\n')
        with patch.object(bench,'logger_paths',side_effect=[[path],[path,second],[path,second],[path,second]]),patch.object(bench.time,'sleep'):
            rows,identity=bench.read_records(path,self.output)
        self.assertEqual(rows,[{'part':1},{'part':2}]);self.assertEqual(len(identity['sources']),2)
        with patch.object(bench,'logger_paths',side_effect=[[path],[path,second]]*10),patch.object(bench.time,'sleep'),self.assertRaisesRegex(ValueError,'every capture'):
            bench.read_records(path,self.output,'must-not-exist.jsonl')
        self.assertFalse((self.output/'must-not-exist.jsonl').exists())


if __name__ == '__main__': unittest.main()
