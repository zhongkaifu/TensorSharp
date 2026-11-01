"""No native builds/devices: discovery and result contract negative fixtures."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import ctest


def payload(*tests):
    return {"kind": "ctestInfo", "version": {"major": 1}, "tests": list(tests)}


def case(name, *args, properties=None):
    return {"name": name, "command": ["synthetic-test"] + list(args), "properties": properties or []}


class CTestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.build = self.repo / "build"
        self.build.mkdir()
        self.output = self.repo / "output"
        self.cache = self.build / "CMakeCache.txt"
        self.cache.write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={self.repo / 'TensorSharp.GGML.Native'}\nTENSORSHARP_GGML_NATIVE_BUILD_TESTS:BOOL=ON\n")
        self.xml = self.repo / "results.xml"
        self.hash_patch = patch.object(ctest, "file_hash", return_value="f" * 64)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def test_backend_uses_explicit_command_then_unambiguous_name(self):
        self.assertEqual(ctest.test_backend(case("metal-layout-cpu", "cpu")), "cpu")
        self.assertEqual(ctest.test_backend(case("embedding-cpu-rms")), "cpu")
        self.assertIsNone(ctest.test_backend(case("maybe-gpu")))
        self.assertIsNone(ctest.test_backend(case("cpu-cuda")))

    def test_no_device_fallback_and_unknown_is_untested(self):
        chosen, omitted = ctest.inventory(payload(case("foo-cpu"), case("foo-cuda"), case("descriptors")), "cpu", None)
        self.assertEqual([item["name"] for item in chosen], ["foo-cpu"])
        self.assertEqual([item["status"] for item in omitted], ["skip", "untested"])

    def test_cross_device_or_filtered_fixture_never_autoruns(self):
        setup = case("reference-cuda", properties=[{"name": "FIXTURES_SETUP", "value": ["reference"]}])
        use = case("compare-cpu", properties=[{"name": "FIXTURES_REQUIRED", "value": ["reference"]}])
        chosen, omitted = ctest.inventory(payload(setup, use), "cpu", None)
        self.assertFalse(chosen)
        self.assertIn("fixture setup", omitted[-1]["reason"])
        setup["name"] = "reference-cpu"
        chosen, omitted = ctest.inventory(payload(setup, use), "cpu", None)
        self.assertEqual(len(chosen), 2)
        self.assertFalse(omitted)

    def test_bad_or_empty_discovery_fails(self):
        for value in ({}, payload(), payload(case("same"), case("same"))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ctest.inventory(value, "cpu", None)

    def test_per_test_failure_skip_and_pass_preserved(self):
        self.xml.write_text('<testsuite><testcase name="one" time="1"/><testcase name="two"><skipped message="No device"/></testcase><testcase name="three"><failure>wrong row</failure></testcase></testsuite>')
        rows = ctest.parse_results(self.xml, [case("one"), case("two"), case("three")])
        self.assertEqual([row["status"] for row in rows], ["pass", "skip", "fail"])
        self.assertEqual(rows[2]["reason"], "wrong row")

    def test_missing_extra_duplicate_test_or_nonfinite_timing_fails(self):
        for content in ('<testsuite/>', '<testsuite><testcase name="other"/></testsuite>',
                        '<testsuite><testcase name="one"/><testcase name="one"/></testsuite>',
                        '<testsuite><testcase name="one" time="nan"/></testsuite>'):
            self.xml.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                ctest.parse_results(self.xml, [case("one")])

    def invoke(self, mode="run", **kwargs):
        return ctest.run_native_tests(self.repo, self.build, self.output, mode, **kwargs)

    def test_plan_does_not_launch_discovery_or_read_build(self):
        self.cache.unlink()
        with patch.object(ctest, "run_process") as process:
            self.assertEqual(self.invoke(mode="plan")[0]["status"], "untested")
            process.assert_not_called()

    def test_wrong_checkout_cache_or_disabled_backend_fails_before_launch(self):
        with patch.object(ctest.shutil, "which", return_value="ctest"), patch.object(ctest, "run_process") as process:
            self.assertEqual(self.invoke(backend="cuda")[0]["status"], "fail")
            self.cache.write_text("CMAKE_HOME_DIRECTORY:INTERNAL=C:/unrelated\nTENSORSHARP_GGML_NATIVE_BUILD_TESTS:BOOL=ON\n")
            self.assertEqual(self.invoke()[0]["status"], "fail")
            process.assert_not_called()

    def test_execution_serial_junit_evidence_and_no_auto_fixtures(self):
        calls = []
        def execute(command, **kwargs):
            calls.append((command, kwargs))
            if "--show-only=json-v1" in command:
                return {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
            self.output.joinpath("ctest.xml").write_text('<testsuite><testcase name="fixture-cpu"/></testsuite>')
            return {"exit_code": 0, "stdout": "100% tests passed", "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), patch.object(ctest, "run_process", side_effect=execute):
            results = self.invoke()
        self.assertTrue(all(row["status"] == "pass" for row in results))
        self.assertIn("--fixture-exclude-any", calls[1][0])
        self.assertEqual(calls[1][0][calls[1][0].index("--parallel") + 1], "1")
        self.assertEqual(calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_zero_exit_without_junit_or_only_skips_is_not_pass(self):
        def execute(command, **kwargs):
            if "--show-only=json-v1" in command:
                return {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
            return {"exit_code": 0, "stdout": "100% tests passed", "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), patch.object(ctest, "run_process", side_effect=execute):
            self.assertEqual(self.invoke()[0]["status"], "fail")
            self.output.joinpath("ctest.xml").write_text('<testsuite><testcase name="fixture-cpu"><skipped/></testcase></testsuite>')
            self.assertEqual(self.invoke()[0]["status"], "untested")

    def test_unverified_discovery_blocks_native_test_launch(self):
        discovery = {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": "",
                     "teardown": {"attempted": True, "verified": None}}
        with patch.object(ctest.shutil, "which", return_value="ctest"), \
             patch.object(ctest, "run_process", return_value=discovery) as execute:
            results = self.invoke()
        self.assertEqual(1, execute.call_count)
        self.assertEqual("fail", results[0]["status"])
        self.assertTrue(results[0]["containment_failure"])
        self.assertEqual("native_ctest::fixture-cpu", results[1]["id"])
        self.assertEqual("untested", results[1]["status"])

    def test_unverified_execution_never_accepts_stale_passing_junit(self):
        discovery = {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
        failed_cleanup = {"exit_code": 0, "containment_failure": True, "stdout": "all pass", "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), \
             patch.object(ctest, "run_process", side_effect=[discovery, failed_cleanup]), \
             patch.object(ctest, "parse_results") as parse:
            results = self.invoke()
        parse.assert_not_called()
        self.assertEqual(["fail", "untested"], [row["status"] for row in results])
        self.assertEqual("native_ctest::fixture-cpu", results[1]["id"])

    @unittest.skipUnless(shutil.which("ctest"), "CTest not installed")
    def test_real_ctest_synthetic_pass_skip_fail_without_build_or_device(self):
        # This CTest file runs only tiny Python expressions, not native/model code.
        executable = Path(sys.executable).as_posix()
        self.build.joinpath("CTestTestfile.cmake").write_text(
            f'add_test(synthetic-pass-cpu "{executable}" "-c" "assert 2 + 2 == 4")\n'
            f'add_test(synthetic-skip-cpu "{executable}" "-c" "raise SystemExit(77)")\n'
            'set_tests_properties(synthetic-skip-cpu PROPERTIES SKIP_RETURN_CODE 77)\n'
            f'add_test(synthetic-fail-cpu "{executable}" "-c" "assert 2 + 2 == 5")\n', encoding="utf-8")
        results = self.invoke(timeout_seconds=10)
        rows = {row["id"]: row["status"] for row in results}
        self.assertEqual(rows, {"native_ctest": "fail", "native_ctest::synthetic-pass-cpu": "pass",
                                "native_ctest::synthetic-skip-cpu": "skip", "native_ctest::synthetic-fail-cpu": "fail"}, results)


if __name__ == "__main__":
    unittest.main()
