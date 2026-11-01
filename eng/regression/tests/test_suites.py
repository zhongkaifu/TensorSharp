"""Test existing-suite orchestration with synthetic logs/TRX and mocked processes."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import suites


def trx(**overrides):
    values = dict(total=2, executed=2, passed=2, failed=0, error=0, timeout=0,
                  aborted=0, inconclusive=0, notRunnable=0, notExecuted=0)
    values.update(overrides)
    attributes = " ".join(f'{key}="{value}"' for key, value in values.items())
    return '<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010"><ResultSummary><Counters ' + attributes + '/></ResultSummary></TestRun>'


class TrxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tests.trx"

    def test_valid_counters_with_explicit_skips(self):
        self.path.write_text(trx(total=3, notExecuted=1))
        counts = suites.trx_counts(self.path)
        self.assertEqual(counts["passed"], 2)
        self.assertEqual(counts["notExecuted"], 1)

    def test_no_execution_and_missing_counters_rejected(self):
        for value in ("<TestRun/>", trx(total=0, executed=0, passed=0), trx(executed=1, passed=2)):
            with self.subTest(value=value):
                self.path.write_text(value)
                with self.assertRaises(ValueError):
                    suites.trx_counts(self.path)

    def test_impossible_or_negative_counters_rejected(self):
        for values in ({"total": 1, "executed": 2}, {"failed": -1},
                       {"passed": 1, "failed": 0, "executed": 2}, {"notExecuted": -1}):
            with self.subTest(values=values):
                self.path.write_text(trx(**values))
                with self.assertRaises(ValueError):
                    suites.trx_counts(self.path)


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.output = self.repo / "output"
        self.binary = self.repo / "fixture.dll"
        self.binary.write_bytes(b"never executed")
        self.spec = {"id": "fixture", "kind": "dotnet", "project": "fixture.csproj",
                     "built": self.binary, "filter": "Category!=Bench", "assertions": ["tests execute"]}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(suites, "list_suites", return_value=[self.spec]))
        self.dotnet = self.stack.enter_context(patch.object(suites.shutil, "which", return_value="dotnet"))
        self.process = self.stack.enter_context(patch.object(suites, "run_process"))
        self.xml = trx()

        def process(*args, **kwargs):
            (kwargs["artifact_dir"] / "tests.trx").write_text(self.xml)
            return {"exit_code": 0, "timed_out": False, "stdout": "", "stderr": ""}
        self.process.side_effect = process

    def invoke(self, mode="run", allow_builds=False):
        return suites.run_suites(self.repo, self.output, mode, allow_builds, 10)

    def test_plan_never_executes_or_builds(self):
        self.assertEqual(self.invoke(mode="plan", allow_builds=True)[0]["status"], "untested")
        self.process.assert_not_called()

    def test_missing_dotnet_or_assembly_skips_without_build(self):
        self.dotnet.return_value = None
        self.assertEqual(self.invoke()[0]["status"], "skip")
        self.dotnet.return_value = "dotnet"
        self.spec["built"] = self.repo / "missing.dll"
        self.assertEqual(self.invoke()[0]["status"], "skip")
        self.process.assert_not_called()

    def test_existing_dotnet_test_requires_results_and_discloses_skips(self):
        self.xml = trx(total=3, notExecuted=1)
        results = self.invoke()
        self.assertEqual({result["status"] for result in results}, {"pass", "skip"})
        command = self.process.call_args.args[0]
        self.assertIn("--no-build", command)
        self.assertIn("--no-restore", command)

    def test_failed_test_or_process_never_passes(self):
        self.xml = trx(passed=1, failed=1)
        self.assertEqual(self.invoke()[-1]["status"], "fail")
        self.process.side_effect = None
        self.process.return_value = {"exit_code": 1, "stdout": "PASS stale", "stderr": "failed"}
        self.assertEqual(self.invoke()[-1]["status"], "fail")

    def test_aborted_inconclusive_and_not_runnable_cannot_be_silent_pass(self):
        for key in ("aborted", "inconclusive", "notRunnable"):
            with self.subTest(key=key):
                self.xml = trx(passed=1, **{key: 1})
                results = self.invoke()
                self.assertTrue(any(result["status"] in {"fail", "skip", "untested"} for result in results), results)

    def test_http_selftest_pass_does_not_mask_missing_integration_assertions(self):
        self.spec.update(kind="client_http")
        self.process.side_effect = [
            {"exit_code": 0, "stdout": "", "stderr": ""},
            {"exit_code": 0, "stdout": "PASS answer oracle", "stderr": ""}]
        self.assertEqual(self.invoke()[-1]["status"], "fail")

    def test_http_both_assertion_programs_required(self):
        self.spec.update(kind="client_http")
        self.process.side_effect = [
            {"exit_code": 0, "stdout": "PASS payload\n35 checks passed.", "stderr": ""},
            {"exit_code": 0, "stdout": "PASS answer oracle", "stderr": ""}]
        self.assertEqual(self.invoke()[-1]["status"], "pass")

    def test_native_contract_requires_completion_boundary_evidence(self):
        self.spec.update(kind="client_native", example=self.binary)
        self.process.side_effect = None
        self.process.return_value = {"exit_code": 0, "stdout": "PASS unrelated", "stderr": ""}
        self.assertEqual(self.invoke()[-1]["status"], "fail")


if __name__ == "__main__":
    unittest.main()
