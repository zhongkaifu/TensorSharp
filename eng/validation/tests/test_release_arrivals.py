"""Offline cancellation/cleanup gates; no HTTP requests or model execution."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('arrivals', Path(__file__).parents[1] / 'validate-release-arrivals.py')
arrivals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arrivals)


class Response:
    def __init__(self, status, body):
        self.status_code, self.body = status, body

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class ReleaseArrivalsTests(unittest.TestCase):
    def test_absent_engine_options_are_omitted_from_wire(self):
        for scenario in ('short', 'decode', 'decode_8k', 'long_8k'):
            body = arrivals.request_body('model', scenario, 'tag')
            self.assertNotIn('tools', body)
            self.assertNotIn('response_format', body)
            self.assertEqual(body['messages'], arrivals.case_spec(scenario, 'tag')['messages'])
            self.assertEqual(body['max_tokens'], 1024 if scenario.startswith('decode') else 256)

    def test_abort_wait_requires_exact_session_and_retains_prior_failure(self):
        log = Mock()
        log.read_text.side_effect = ['Web UI chat aborted by client (sessionId=other, partialTokens=8)',
                                    'Web UI chat aborted by client (sessionId=s, partialTokens=8)']
        now = [0.0]
        result = {'status': 'fail', 'session_id': 's', 'error': 'prior failure'}
        self.assertTrue(arrivals.wait_server_abort(result, log, timeout=1,
            clock=lambda: now[0], pause=lambda seconds: now.__setitem__(0, now[0] + seconds)))
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['error'], 'prior failure')
        self.assertEqual(result['server_abort_wait_seconds'], 0.25)

    def test_missing_abort_expires_and_cannot_pass(self):
        log = Mock()
        log.read_text.return_value = 'Web UI chat aborted by client (sessionId=ss, partialTokens=8)'
        now = [0.0]
        result = {'status': 'ok', 'session_id': 's'}
        self.assertFalse(arrivals.wait_server_abort(result, log, timeout=1,
            clock=lambda: now[0], pause=lambda seconds: now.__setitem__(0, now[0] + seconds)))
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['server_abort_wait_seconds'], 1)

    def test_peer_must_span_actual_disconnect(self):
        for start, finish, expected in ((2, 8, True), (5, 8, False), (6, 8, False),
                                        (1, 5, False), (1, 4, False), (None, 8, False)):
            result = {'status': 'ok', 'client_disconnected_monotonic': 5}
            with self.subTest(start=start, finish=finish):
                self.assertEqual(arrivals.qualify_cancel_peer(result,
                    {'started_monotonic': start, 'finished_monotonic': finish}), expected)
                self.assertEqual(result['status'], 'ok' if expected else 'fail')

    def test_missing_disconnect_cannot_qualify_overlap(self):
        result = {'status': 'fail', 'error': 'original transport failure'}
        self.assertFalse(arrivals.qualify_cancel_peer(result, {'started_monotonic': 1, 'finished_monotonic': 8}))
        self.assertEqual(result['error'], 'original transport failure')

    def test_cleanup_requires_success_and_matching_acknowledgment(self):
        for status, body, expected in ((200, {'ok': True, 'sessionId': 's'}, True),
                (404, {'ok': False}, False), (500, {'ok': True, 'sessionId': 's'}, False),
                (200, {'ok': False, 'sessionId': 's'}, False),
                (200, {'ok': True, 'sessionId': 'wrong'}, False),
                (200, ValueError('invalid JSON'), False)):
            result = {'status': 'ok', 'session_id': 's'}
            with self.subTest(status=status, body=body):
                self.assertEqual(arrivals.qualify_session_cleanup(result, Response(status, body)), expected)
                self.assertEqual(result['status'], 'ok' if expected else 'fail')

    def test_cleanup_transport_failure_preserves_prior_error(self):
        result = {'status': 'fail', 'session_id': 's', 'error': 'prior failure'}
        self.assertFalse(arrivals.qualify_session_cleanup(result, error=TimeoutError('cleanup timeout')))
        self.assertEqual(result['error'], 'prior failure')
        self.assertEqual(result['session_cleanup_error'], 'cleanup timeout')

    def test_successful_cleanup_does_not_clear_an_existing_failure(self):
        result = {'status': 'fail', 'session_id': 's'}
        self.assertTrue(arrivals.qualify_session_cleanup(result, Response(200, {'ok': True, 'sessionId': 's'})))
        self.assertEqual(result['status'], 'fail')


if __name__ == '__main__':
    unittest.main()
