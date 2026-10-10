import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("bounded_probe", Path(__file__).parents[1] / "run-bounded-probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class BoundedProbeTests(unittest.TestCase):
    def test_success_retains_stdout_and_does_not_claim_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = PROBE.run([sys.executable, "-c", "print('complete evidence')"], output, 10)
            self.assertTrue(result["complete"])
            self.assertFalse(result["timed_out"])
            self.assertEqual(0, result["exit_code"])
            self.assertIsNone(result["quality_passed"])
            self.assertIn("complete evidence", (output / "process.log").read_text())
            self.assertEqual(result, json.loads((output / "execution.json").read_text()))

    def test_nonzero_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result = PROBE.run([sys.executable, "-c", "raise SystemExit(17)"], Path(directory) / "run", 10)
            self.assertTrue(result["complete"])
            self.assertEqual(17, result["exit_code"])
            self.assertFalse(result["timed_out"])

    def test_timeout_waits_for_owned_child_exit_and_keeps_incomplete_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = PROBE.run([sys.executable, "-c", "import time; time.sleep(30)"], output, 0.2)
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["complete"])
            self.assertIsNotNone(result["exit_code"])
            self.assertNotEqual(0, result["exit_code"])
            self.assertEqual(result, json.loads((output / "execution.json").read_text()))


if __name__ == "__main__":
    unittest.main()
