#!/usr/bin/env python3
"""Failure/reporting tests; no model or accelerator is needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import abba_summary


class AbbaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def complete(self):
        self.write("run-plan.json", {"rounds": 2, "measure_passes": 3, "arms": ["base", "candidate"]})
        log = []
        for arm in ("base", "candidate"):
            for run in range(1, 3):
                log.extend([f"{arm}-{run} start time", f"{arm}-{run} rc=0 end time"])
                for measured in range(1, 4):
                    self.write(f"{arm}-{run}.json.measure{measured}.json", [
                        {"Scenario": "short", "Label": "turn 1", "PrefillTps": 100,
                         "DecodeTps": 10, "TtftMs": 5}])
                self.write(f"{arm}-{run}.json.series.json", [
                    {"Pass": p, "Failures": 0} for p in range(1, 4)])
        (self.directory / "runs.txt").write_text("\n".join(log + ["DONE"]) + "\n")

    def summarize(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = abba_summary.main([str(self.directory), "--baseline", "base", "--candidate", "candidate"])
        return rc, out.getvalue()

    def test_complete_balanced_run_passes(self):
        self.complete()
        rc, output = self.summarize()
        self.assertEqual(0, rc, output)
        self.assertIn("base=2, candidate=2", output)

    def test_missing_final_pass_cannot_hide_in_process_median(self):
        self.complete()
        (self.directory / "candidate-1.json.measure3.json").unlink()
        rc, output = self.summarize()
        self.assertEqual(2, rc)
        self.assertIn("incomplete measured passes", output)

    def test_missing_process_cannot_hide_in_arm_median(self):
        self.complete()
        for path in self.directory.glob("candidate-2.json*"):
            path.unlink()
        rc, output = self.summarize()
        self.assertEqual(2, rc)
        self.assertIn("process result", output)

    def test_failed_process_is_rejected_even_with_done_and_complete_json(self):
        self.complete()
        log = self.directory / "runs.txt"
        log.write_text(log.read_text().replace("candidate-1 rc=0", "candidate-1 rc=139"))
        rc, output = self.summarize()
        self.assertEqual(2, rc)
        self.assertIn("exit 139", output)

    def test_failed_pass_is_rejected(self):
        self.complete()
        self.write("candidate-1.json.series.json", [{"Pass": p, "Failures": int(p == 3)} for p in range(1, 4)])
        rc, output = self.summarize()
        self.assertEqual(2, rc)
        self.assertIn("failed pass series", output)

    def test_missing_workload_row_is_rejected_before_aggregation(self):
        self.complete()
        self.write("candidate-1.json.measure2.json", [])
        rc, output = self.summarize()
        self.assertEqual(2, rc)
        self.assertIn("empty or malformed", output)

    def test_runner_propagates_failure_and_preserves_previous_results(self):
        tools = self.directory / "tools"
        tools.mkdir()
        dotnet = tools / "dotnet"
        dotnet.write_text("#!/bin/sh\nexit 17\n")
        dotnet.chmod(0o755)
        result_dir = self.directory / "results"
        command = ["bash", str(Path(__file__).with_name("abba.sh")), str(result_dir), "1",
                   "--measure-passes 3", "base=/unused", "candidate=/unused"]
        env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(1, result.returncode, result.stderr)
        log = (result_dir / "runs.txt").read_text()
        self.assertEqual(2, log.count("rc=17"))
        self.assertTrue(log.endswith("FAILED 2 process(es)\n"))
        self.assertEqual(3, json.loads((result_dir / "run-plan.json").read_text())["measure_passes"])
        repeated = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(2, repeated.returncode)
        self.assertEqual(log, (result_dir / "runs.txt").read_text())


if __name__ == "__main__":
    unittest.main()
