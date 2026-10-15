"""Real CPU child processes exercise local reference-client lifetime boundaries."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('llama_local', Path(__file__).parents[1]/'llama-local-quality-bench.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LocalReferenceLifetimeTests(unittest.TestCase):
    def test_completed_client_keeps_nonzero_exit_and_is_reaped(self):
        for code in (0, 17):
            with self.subTest(code=code), tempfile.TemporaryFile(mode='w+') as log:
                result = {}
                runner.run_client([sys.executable, '-c', f'raise SystemExit({code})'], 10, log, result)
                self.assertEqual(code, result['client_exit'])
                self.assertTrue(result['client_cleanup']['owned_server_exited'])
                self.assertFalse(result['client_cleanup']['cleanup_errors'])

    def test_hanging_client_is_timed_out_and_reaped(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            result = {}
            with self.assertRaisesRegex(TimeoutError, 'Total reference lifetime'):
                runner.run_client([sys.executable, '-c', 'import time; time.sleep(30)'], .05, log, result)
            self.assertTrue(result['timed_out'])
            self.assertTrue(result['client_cleanup']['owned_server_exited'])
            self.assertIsNotNone(result['client_cleanup']['server_exit'])
            self.assertNotIn('client_exit', result)

    def test_interrupt_reaps_owned_client_before_propagating(self):
        launch = runner.subprocess.Popen
        def interrupted(*args, **kwargs):
            child = launch(*args, **kwargs)
            wait = child.wait
            calls = 0
            def interrupt_once(timeout=None):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt('injected')
                return wait(timeout=timeout)
            child.wait = interrupt_once
            return child
        with tempfile.TemporaryFile(mode='w+') as log, mock.patch.object(runner.subprocess, 'Popen', side_effect=interrupted):
            result = {}
            with self.assertRaises(KeyboardInterrupt):
                runner.run_client([sys.executable, '-c', 'import time; time.sleep(30)'], 10, log, result)
            self.assertTrue(result['client_cleanup']['owned_server_exited'])

    def test_nonfinite_timeout_is_rejected_before_evidence_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            for value in ('nan', 'inf', '-inf'):
                args = ['local', '--server', 'server', '--repo', 'repo', '--model', 'model',
                        '--output', str(output), '--timeout=' + value]
                with self.subTest(value=value), mock.patch.object(runner.sys, 'argv', args), \
                        mock.patch.object(runner.bench, 'evidence_directory', return_value=output), \
                        mock.patch.object(runner.sys, 'stderr', io.StringIO()), self.assertRaises(SystemExit) as error:
                    runner.main()
                self.assertEqual(2, error.exception.code)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
