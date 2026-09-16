"""Offline freshness/nonce checks; never starts a model, tunnel, or tool runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WAITER = Path(__file__).resolve().parents[1] / 'wait-external-suite.py'


class ExternalSuiteWaiterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = self.root / 'ready.json'
        self.completion = self.root / 'complete.json'
        self.nonce = 'offline-fixture-1234'
        self.identity.write_text(json.dumps({'nonce': self.nonce}))
        self.value = {'nonce': self.nonce, 'client_exit_code': 0, 'independent_code_exit_code': 0,
                      'remote_binding_postcheck_passed': True, 'client_report_sha256': 'a' * 64}
        self.identity_time = 1700000000000000000
        os.utime(self.identity, ns=(self.identity_time, self.identity_time))

    def run_waiter(self, *, offset=1000000000, identity=True):
        self.completion.write_text(json.dumps(self.value))
        stamp = self.identity_time + offset
        os.utime(self.completion, ns=(stamp, stamp))
        command = [sys.executable, str(WAITER), '--completion', str(self.completion),
                   '--nonce', self.nonce, '--timeout', '1']
        if identity:
            command += ['--ready-identity', str(self.identity)]
        return subprocess.run(command, capture_output=True, text=True, timeout=5)

    def test_existing_completion_after_or_at_identity_is_accepted(self):
        for offset in (0, 1000000000):
            with self.subTest(offset=offset):
                result = self.run_waiter(offset=offset)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"external_client_passed": true', result.stdout)

    def test_completion_before_identity_is_rejected(self):
        result = self.run_waiter(offset=-1000000000)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('predates the ready identity', result.stderr)

    def test_completion_or_identity_nonce_mismatch_is_rejected(self):
        self.value['nonce'] = 'different-completion'
        result = self.run_waiter()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('External completion belongs to another campaign', result.stderr)
        self.value['nonce'] = self.nonce
        self.identity.write_text(json.dumps({'nonce': 'different-identity'}))
        result = self.run_waiter()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Ready identity belongs to another campaign', result.stderr)

    def test_failed_completion_remains_failed(self):
        for key, value in (('client_exit_code', 1), ('independent_code_exit_code', 1),
                           ('remote_binding_postcheck_passed', False), ('client_report_sha256', 'invalid')):
            with self.subTest(key=key):
                previous = self.value[key]
                self.value[key] = value
                result = self.run_waiter()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('"external_client_passed": false', result.stdout)
                self.value[key] = previous

    def test_without_identity_existing_completion_remains_strictly_rejected(self):
        result = self.run_waiter(identity=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Completion file already exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
