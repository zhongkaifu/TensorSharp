"""No native builds/devices: discovery and result contract negative fixtures."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import ctest
from regression.processes import clean_environment


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

    def test_ctest_message_only_failure_retains_reason_and_assertion_output(self):
        # Actual CTest format from the native CPU regression; assertion details
        # are in system-out rather than inside the failure element.
        self.xml.write_text('<testsuite><testcase name="one" status="fail"><failure message="Failed"/>'
                            '<system-out>QUANT_STRIP_MISMATCH actual=0.643071175 expected=0.643071234</system-out>'
                            '</testcase></testsuite>')
        row = ctest.parse_results(self.xml, [case("one")])[0]
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["reason"], "Failed")
        self.assertIn("QUANT_STRIP_MISMATCH", row["stdout"])
        self.assertNotIn("skipped", row["reason"])

    def test_error_message_and_body_both_retained(self):
        self.xml.write_text('<testsuite><testcase name="one"><error message="Process crashed">'
                            'exception details</error></testcase></testsuite>')
        row = ctest.parse_results(self.xml, [case("one")])[0]
        self.assertEqual(row["status"], "fail")
        self.assertEqual(row["reason"], "Process crashed\nexception details")

    def test_empty_failure_never_uses_skip_reason_or_overrides_failure_with_skip(self):
        for extra in ('', '<skipped message="fixture unavailable"/>'):
            with self.subTest(extra=extra):
                self.xml.write_text('<testsuite><testcase name="one"><failure/>' + extra + '</testcase></testsuite>')
                row = ctest.parse_results(self.xml, [case("one")])[0]
                self.assertEqual(row["status"], "fail")
                self.assertEqual(row["reason"], "native test failed; see retained stdout/stderr")

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
        write_skipped_report = False
        def execute(command, **kwargs):
            if "--show-only=json-v1" in command:
                return {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
            if write_skipped_report:
                self.output.joinpath("ctest.xml").write_text('<testsuite><testcase name="fixture-cpu"><skipped/></testcase></testsuite>')
            return {"exit_code": 0, "stdout": "100% tests passed", "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), patch.object(ctest, "run_process", side_effect=execute):
            self.assertEqual(self.invoke()[0]["status"], "fail")
            write_skipped_report = True
            self.assertEqual(self.invoke()[0]["status"], "untested")

    def test_native_fixture_directories_are_fresh_owned_and_override_ambient_paths(self):
        ambient_tmp, ambient_fixtures = self.repo / "user-temp", self.repo / "user-models"
        ambient_tmp.mkdir()
        ambient_fixtures.mkdir()
        user_temp = ambient_tmp / "minimax-m3-exec-test.gguf"
        user_model = ambient_fixtures / "model.gguf"
        user_temp.write_text("existing user temp")
        user_model.write_text("existing user fixture")
        observed = []

        def execute(command, **kwargs):
            if "--show-only=json-v1" in command:
                return {"exit_code": 0, "stdout": json.dumps(payload(case("minimax-m3-dspark-cpu"))), "stderr": ""}
            effective = clean_environment(kwargs["env"])
            native_tmp = Path(effective["TMPDIR"])
            external = Path(effective["TS_MM3_FIXTURE_DIR"])
            self.assertEqual(native_tmp.name, "native-tmp")
            self.assertEqual(external.name, "external-fixtures")
            self.assertEqual(native_tmp.parent, external.parent)
            self.assertEqual(native_tmp.parent.parent, self.output.resolve())
            self.assertNotEqual(native_tmp, ambient_tmp)
            self.assertNotEqual(external, ambient_fixtures)
            self.assertEqual(effective["TMP"], str(native_tmp))
            self.assertEqual(effective["TEMP"], str(native_tmp))
            self.assertEqual(list(native_tmp.iterdir()), [])
            self.assertEqual(list(external.iterdir()), [])
            observed.append((native_tmp, external, kwargs["env"]))
            # Leave an old synthetic fixture behind: a rerun must not reuse it.
            (external / "model.gguf").write_text("previous run synthetic marker")
            self.output.joinpath("ctest.xml").write_text('<testsuite><testcase name="minimax-m3-dspark-cpu"><skipped message="fixture absent (77)"/></testcase></testsuite>')
            return {"exit_code": 0, "stdout": "test skipped", "stderr": ""}

        with patch.dict(os.environ, {"TMPDIR": str(ambient_tmp), "TMP": str(ambient_tmp),
                                     "TEMP": str(ambient_tmp), "TS_MM3_FIXTURE_DIR": str(ambient_fixtures)}), \
             patch.object(ctest.shutil, "which", return_value="ctest"), \
             patch.object(ctest, "run_process", side_effect=execute):
            first, second = self.invoke(), self.invoke()
        for result, (_, _, env) in zip((first, second), observed):
            self.assertEqual([row["status"] for row in result], ["untested", "skip"])
            recorded = result[0]["fixture_isolation"]
            self.assertEqual(recorded["environment"]["TMPDIR"], env["TMPDIR"])
            self.assertEqual(recorded["environment"]["TS_MM3_FIXTURE_DIR"], env["TS_MM3_FIXTURE_DIR"])
        self.assertNotEqual(observed[0][0], observed[1][0])
        self.assertNotEqual(observed[0][1], observed[1][1])
        self.assertEqual(user_temp.read_text(), "existing user temp")
        self.assertEqual(user_model.read_text(), "existing user fixture")

    def test_fixture_directory_creation_failure_prevents_native_execution(self):
        discovery = {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), \
             patch.object(ctest, "run_process", return_value=discovery) as process, \
             patch.object(ctest.tempfile, "mkdtemp", side_effect=PermissionError("cannot isolate native fixtures")):
            results = self.invoke()
        process.assert_called_once()
        self.assertEqual(results[0]["status"], "fail")
        self.assertIn("cannot isolate native fixtures", results[0]["reason"])

    def test_zero_exit_without_fresh_junit_rejects_old_passing_report(self):
        self.output.mkdir()
        junit = self.output / "ctest.xml"
        junit.write_text('<testsuite><testcase name="fixture-cpu"/></testsuite>')
        unrelated = self.output / "keep.txt"
        unrelated.write_text("preserve other diagnostic evidence")
        def execute(command, **kwargs):
            if "--show-only=json-v1" in command:
                return {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
            self.assertFalse(junit.exists(), "stale report must be removed before execution")
            return {"exit_code": 0, "stdout": "100% tests passed", "stderr": "no new report written"}
        with patch.object(ctest.shutil, "which", return_value="ctest"), patch.object(ctest, "run_process", side_effect=execute) as process:
            results = self.invoke()
        self.assertEqual(process.call_count, 2)
        self.assertEqual([row["status"] for row in results], ["fail"])
        self.assertIn("ctest.xml", results[0]["reason"])
        self.assertEqual(results[0]["processes"][-1]["stderr"], "no new report written")
        self.assertFalse(junit.exists())
        self.assertEqual(unrelated.read_text(), "preserve other diagnostic evidence")

    def test_junit_invalidation_failure_prevents_native_execution(self):
        discovery = {"exit_code": 0, "stdout": json.dumps(payload(case("fixture-cpu"))), "stderr": ""}
        with patch.object(ctest.shutil, "which", return_value="ctest"), \
             patch.object(ctest, "run_process", return_value=discovery) as process, \
             patch.object(Path, "unlink", side_effect=PermissionError("cannot invalidate stale ctest.xml")) as unlink:
            results = self.invoke()
        unlink.assert_called_once_with(missing_ok=True)
        process.assert_called_once()
        self.assertEqual(results[0]["status"], "fail")
        self.assertIn("cannot invalidate stale ctest.xml", results[0]["reason"])

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
