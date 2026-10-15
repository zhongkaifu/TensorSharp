import copy
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('planned_chat', Path(__file__).resolve().parents[1] / 'llama-planned-chat.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PlannedChatTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'futureCommand': ['llama-server', '--host', '127.0.0.1', '--port', '51234', '--parallel', '1'],
                     'futureRequest': {'endpoint': 'http://127.0.0.1:51234/v1/chat/completions',
                                       'body': {'stream': False}}}
        self.response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '42'}}]}

    def test_normal_answer_is_scoped_not_exact_eos(self):
        result = probe.quality_result(self.response, '42')
        self.assertTrue(result['passed'])
        self.assertFalse(result['checkpoint_eos_id_observed'])

    def test_truncation_and_extra_prose_are_failures(self):
        self.response['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(probe.quality_result(self.response, '42')['passed'])
        self.response['choices'][0].update(finish_reason='stop', message={'content': 'Answer: 42'})
        self.assertFalse(probe.quality_result(self.response, '42')['passed'])

    def test_tool_calls_and_multiple_choices_are_failures(self):
        self.response['choices'][0]['message']['tool_calls'] = [{'id': 'unexpected'}]
        self.assertFalse(probe.quality_result(self.response, '42')['passed'])
        self.response['choices'].append(copy.deepcopy(self.response['choices'][0]))
        self.assertFalse(probe.quality_result(self.response, '42')['passed'])

    def test_matching_endpoint(self):
        self.assertEqual(probe.plan_endpoint(self.plan).port, 51234)

    def test_wrong_port_or_malformed_host_is_refused(self):
        for value in ('http://127.0.0.1:51235/v1/chat/completions', 'https://example.org/v1/chat/completions'):
            self.plan['futureRequest']['endpoint'] = value
            with self.assertRaises(ValueError):
                probe.plan_endpoint(self.plan)
        self.setUp()
        self.plan['futureCommand'].append('--host')
        with self.assertRaises(ValueError):
            probe.plan_endpoint(self.plan)

    def test_custom_stop_tools_and_streaming_are_refused(self):
        for field, value in (('stop', ['42']), ('tools', [{'type': 'function'}]), ('stream', True), ('n', 2)):
            self.setUp()
            self.plan['futureRequest']['body'][field] = value
            with self.assertRaises(ValueError):
                probe.plan_endpoint(self.plan)

    def test_failed_terminate_does_not_skip_kill_or_sampler_cleanup(self):
        server = Mock()
        server.poll.side_effect = [None, None, -9]
        server.terminate.side_effect = OSError('injected terminate error')
        server.wait.side_effect = [TimeoutError('injected wait error'), -9]
        timer = Mock()
        timer.is_alive.return_value = False
        sampler = SimpleNamespace(stop=Mock(), thread=Mock(), errors=[])
        sampler.thread.is_alive.return_value = False
        result = probe.cleanup_owned(server, timer, sampler)
        server.kill.assert_called_once()
        sampler.stop.set.assert_called_once()
        self.assertTrue(result['owned_server_exited'])
        self.assertTrue(result['sampler_stopped'])
        self.assertEqual(len(result['cleanup_errors']), 2)

    def test_failed_kill_and_sampler_join_are_recorded_together(self):
        server = Mock()
        server.poll.return_value = None
        server.wait.side_effect = TimeoutError('injected wait error')
        server.kill.side_effect = OSError('injected kill error')
        sampler = SimpleNamespace(stop=Mock(), thread=Mock(), errors=[])
        sampler.thread.join.side_effect = RuntimeError('injected join error')
        sampler.thread.is_alive.return_value = True
        result = probe.cleanup_owned(server, None, sampler)
        self.assertFalse(result['owned_server_exited'])
        self.assertFalse(result['sampler_stopped'])
        self.assertTrue(any('server.kill' in error for error in result['cleanup_errors']))
        self.assertTrue(any('sampler.join' in error for error in result['cleanup_errors']))


if __name__ == '__main__':
    unittest.main()
