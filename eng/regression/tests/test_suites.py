"""Test existing-suite orchestration with synthetic logs/TRX and mocked processes."""
from contextlib import ExitStack
import hashlib
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


class SuiteSelectionTests(unittest.TestCase):
    def test_cpu_lanes_exclude_all_gpu_traits_and_retain_ggml_cpu(self):
        gpu_traits = {"Cuda", "Mlx", "GgmlCuda", "GgmlMetal", "GgmlVulkan"}
        for spec in suites.list_suites(Path("unused")):
            if spec["kind"] != "dotnet":
                continue
            with self.subTest(suite=spec["id"]):
                self.assertNotIn("|", spec["filter"], "an OR must not bypass GPU exclusions")
                predicates = spec["filter"].split("&")
                excluded = {item.removeprefix("Requires!=") for item in predicates
                            if item.startswith("Requires!=")}
                self.assertTrue(gpu_traits <= excluded, "GPU selection must not depend on assembly skip gates")
                self.assertNotIn("GgmlCpu", excluded, "native CPU correctness tests should remain eligible")
                self.assertIn("Models", excluded)
                self.assertIn("Category!=Bench", predicates)

    def test_legacy_model_fallbacks_are_excluded_without_relying_on_traits(self):
        specs = {spec["id"]: spec for spec in suites.list_suites(Path("unused"))}
        server = specs["server_unit"]
        predicates = server["filter"].split("&")
        self.assertNotIn("|", server["filter"], "an OR must not bypass model exclusions")
        self.assertIn("FullyQualifiedName!~InferenceWeb.Tests.WanVideoOracleTests", predicates)
        self.assertIn("FullyQualifiedName!=InferenceWeb.Tests.WanVideoTests.TokenizerMatchesHuggingFaceReference", predicates)
        self.assertNotIn("FullyQualifiedName!~InferenceWeb.Tests.WanVideoTests", predicates,
                         "synthetic Wan tests should still execute")
        self.assertEqual(2, len([item for item in server["excluded_fixtures"] if item["fixture_kind"] == "model_weights"]))
        self.assertTrue(all(item["reason"] for item in server["excluded_fixtures"]))
        self.assertFalse(specs["knowledge_unit"]["excluded_fixtures"])

    def test_external_optional_inputs_are_exact_exclusions_not_model_weight_coverage(self):
        expected = {
            "InferenceWeb.Tests.Dsv41QuantizationTests.QuantizesTheHarnessVectorsForEveryMode":
                ("quantization_vectors", "TS_DSV41_QUANT_VECTORS"),
            "InferenceWeb.Tests.ImageProcessorTests.UserSuppliedJpegSmokeTestWhenConfigured":
                ("user_media", "TENSORSHARP_JPEG_SMOKE_PATH"),
            "InferenceWeb.Tests.ImageProcessorTests.UserSuppliedHeicSmokeTestWhenConfigured":
                ("user_media", "TENSORSHARP_HEIC_SMOKE_PATH"),
        }
        spec = suites.list_suites(Path("unused"))[0]
        predicates = spec["filter"].split("&")
        records = {item["fully_qualified_name"]: item for item in spec["excluded_fixtures"]
                   if item["fixture_kind"] != "model_weights"}
        self.assertEqual(set(expected), set(records))
        for name, (kind, variable) in expected.items():
            with self.subTest(name=name):
                self.assertIn("FullyQualifiedName!=" + name, predicates)
                self.assertEqual("method", records[name]["match"])
                self.assertEqual(kind, records[name]["fixture_kind"])
                self.assertEqual(variable, records[name]["environment_variable"])
                self.assertIn("absent", records[name]["reason"])
        for name in ("InferenceWeb.Tests.Dsv41QuantizationTests", "InferenceWeb.Tests.ImageProcessorTests"):
            self.assertNotIn("FullyQualifiedName!~" + name, predicates, "other synthetic tests remain eligible")

        with patch.object(suites, "run_process") as launch:
            planned = suites.run_suites(Path("unused"), Path("unused-output"), "plan", False, 10)[0]
        launch.assert_not_called()
        self.assertEqual(5, len(planned["excluded_fixtures"]))
        self.assertTrue(all(item["status"] == "untested" for item in planned["excluded_fixtures"]))

    def test_prebuilt_assembly_command_and_report_keep_model_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            spec = suites.list_suites(repo)[0]
            spec["built"].parent.mkdir(parents=True)
            # An opaque old assembly has no corrected source traits to rely on.
            spec["built"].write_bytes(b"legacy untagged model fixture assembly")

            def process(command, **kwargs):
                test_filter = command[command.index("--filter") + 1]
                self.assertIn("&FullyQualifiedName!~InferenceWeb.Tests.WanVideoOracleTests", test_filter)
                self.assertIn("&FullyQualifiedName!=InferenceWeb.Tests.WanVideoTests.TokenizerMatchesHuggingFaceReference", test_filter)
                for item in suites.SERVER_EXTERNAL_FIXTURE_EXCLUSIONS:
                    self.assertIn("FullyQualifiedName!=" + item["fully_qualified_name"], test_filter.split("&"))
                for trait in ("Cuda", "Mlx", "GgmlCuda", "GgmlMetal", "GgmlVulkan"):
                    self.assertIn("Requires!=" + trait, test_filter.split("&"))
                self.assertNotIn("Requires!=GgmlCpu", test_filter.split("&"))
                self.assertEqual("cpu", kwargs["env"]["TS_TEST_GGML_BACKEND"])
                self.assertEqual("model_free_suite", kwargs["environment_mode"])
                self.assertIn("--no-build", command)
                self.assertIn("--no-restore", command)
                (kwargs["artifact_dir"] / "tests.trx").write_text(trx())
                return {"exit_code": 0, "stdout": "", "stderr": ""}

            with patch.object(suites, "list_suites", return_value=[spec]), \
                    patch.object(suites.shutil, "which", return_value="dotnet"), \
                    patch.object(suites, "source_identity", return_value={}), \
                    patch.object(suites, "run_process", side_effect=process) as launch:
                result = suites.run_suites(repo, repo / "output", "run", False, 10)[-1]
            self.assertEqual("pass", result["status"])
            self.assertEqual(1, launch.call_count)
            self.assertEqual(5, len(result["excluded_fixtures"]))
            self.assertTrue(all(item["status"] == "untested" for item in result["excluded_fixtures"]))
            self.assertEqual({"model_weights", "quantization_vectors", "user_media"},
                             {item["fixture_kind"] for item in result["excluded_fixtures"]})

    def test_new_assembly_sources_classify_external_wan_fixtures_as_models(self):
        repo = Path(__file__).resolve().parents[3]
        oracle = (repo / "InferenceWeb.Tests/WanVideoOracleTests.cs").read_text(encoding="utf-8-sig")
        tokenizer = (repo / "InferenceWeb.Tests/WanVideoTests.cs").read_text(encoding="utf-8-sig")
        self.assertRegex(oracle, r'\[Trait\("Requires", "Models"\)\]\s+public class WanVideoOracleTests')
        self.assertRegex(tokenizer, r'\[Trait\("Requires", "Models"\)\]\s+public void TokenizerMatchesHuggingFaceReference')


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
        self.stack.enter_context(patch.object(suites, "source_identity", return_value={"commit": "synthetic-suite-source"}))
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

    def test_all_managed_build_restore_test_commands_skip_native_targets(self):
        for kind in ("dotnet", "client_native", "client_http"):
            with self.subTest(kind=kind):
                self.spec.update(kind=kind, example=self.binary)
                self.process.reset_mock()

                def successful_process(command, **kwargs):
                    (kwargs["artifact_dir"] / "tests.trx").write_text(self.xml)
                    return {"exit_code": 0, "stdout": "PASS contracts; 7 completion boundary checks passed", "stderr": ""}

                self.process.side_effect = successful_process
                results = self.invoke(allow_builds=True)
                self.assertEqual("pass", results[-1]["status"])
                self.assertEqual("model_free_suite", results[-1]["environment_mode"])
                self.assertEqual({"managed_builds_allowed": True, "native_builds_allowed": False}, results[-1]["build_policy"])
                managed = []
                for call in self.process.call_args_list:
                    command = call.args[0]
                    self.assertEqual({"TS_TEST_GGML_BACKEND": "cpu"}, call.kwargs["env"])
                    self.assertEqual("model_free_suite", call.kwargs["environment_mode"])
                    if command[0] == "dotnet" and command[1] in {"build", "restore", "test"}:
                        managed.append(command)
                        self.assertEqual(1, command.count("-p:TensorSharpSkipGgmlNative=true"))
                        self.assertEqual(1, command.count("-p:TensorSharpSkipMlxNative=true"))
                        self.assertEqual(1, command.count("-p:TensorSharpSkipCudaPtx=true"))
                        if command[1] == "test":
                            self.assertIn("--no-build", command)
                            self.assertIn("--no-restore", command)
                        else:
                            self.assertNotIn("--no-build", command)
                        self.assertFalse(any("ENABLE_CUDA" in arg or "ENABLE_VULKAN" in arg for arg in command))
                self.assertTrue(managed)

    def test_no_build_dotnet_test_also_disables_native_msbuild_targets(self):
        self.invoke()
        command = self.process.call_args.args[0]
        self.assertIn("--no-build", command)
        self.assertIn("-p:TensorSharpSkipGgmlNative=true", command)
        self.assertIn("-p:TensorSharpSkipMlxNative=true", command)
        self.assertIn("-p:TensorSharpSkipCudaPtx=true", command)

    def test_identity_is_recorded_after_build_before_test_with_native_candidates(self):
        native = self.repo / "GgmlOps.dll"
        native.write_bytes(b"original native")
        kernel = self.repo / "cuda_kernels" / "tensorsharp_kernels.ptx"
        kernel.parent.mkdir()
        kernel.write_bytes(b"prebuilt PTX")
        built_contents = b"new managed assembly"

        def process(command, **kwargs):
            if command[1] == "build":
                self.binary.write_bytes(built_contents)
            else:
                # This mutation after launch must not replace pre-execution identity.
                self.binary.write_bytes(b"modified during test")
                (kwargs["artifact_dir"] / "tests.trx").write_text(self.xml)
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        self.process.side_effect = process
        result = self.invoke(allow_builds=True)[-1]
        self.assertEqual("pass", result["status"])
        self.assertEqual(["build", "test"], [call.args[0][1] for call in self.process.call_args_list])
        identity = result["execution_identity"]
        self.assertEqual("synthetic-suite-source", identity["source"]["commit"])
        self.assertEqual(hashlib.sha256(built_contents).hexdigest(), identity["assemblies"][0]["sha256"])
        candidates = {Path(item["path"]).name: item["sha256"] for item in identity["runtime_candidates"]}
        self.assertEqual(hashlib.sha256(b"original native").hexdigest(), candidates["GgmlOps.dll"])
        self.assertEqual(hashlib.sha256(b"prebuilt PTX").hexdigest(), candidates["tensorsharp_kernels.ptx"])

    def test_build_cannot_leave_trx_that_substitutes_for_test_execution(self):
        def process(command, **kwargs):
            if command[1] == "build":
                (kwargs["artifact_dir"] / "tests.trx").write_text(self.xml)
            return {"exit_code": 0, "stdout": "PASS build", "stderr": ""}

        self.process.side_effect = process
        self.assertEqual("fail", self.invoke(allow_builds=True)[-1]["status"])

    def test_successful_managed_build_without_contract_assembly_is_failure(self):
        for kind, expected_calls in (("client_native", 1), ("client_http", 2)):
            with self.subTest(kind=kind):
                self.spec.update(kind=kind, built=self.repo / "missing.dll", example=self.binary)
                self.process.reset_mock()
                self.process.side_effect = None
                self.process.return_value = {"exit_code": 0, "stdout": "PASS build; 7 completion boundary checks passed", "stderr": ""}
                results = self.invoke(allow_builds=True)
                self.assertEqual("fail", results[-1]["status"])
                self.assertIn("required contract assemblies", results[-1]["reason"])
                self.assertEqual(expected_calls, self.process.call_count, "missing assembly must block assertion process")

    def test_managed_build_success_cannot_substitute_for_test_assertions(self):
        self.spec.update(kind="client_http")
        self.process.side_effect = [
            {"exit_code": 0, "stdout": "PASS restore", "stderr": ""},
            {"exit_code": 0, "stdout": "PASS build", "stderr": ""},
            {"exit_code": 0, "stdout": "", "stderr": ""}]
        results = self.invoke(allow_builds=True)
        self.assertEqual("fail", results[-1]["status"])
        self.assertIn("no assertion evidence", results[-1]["reason"])

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

    def test_unsafe_teardown_halts_next_internal_command_and_later_suites(self):
        self.spec.update(kind="client_http")
        later = {**self.spec, "id": "must-not-start"}
        for process in ({"exit_code": 125, "containment_failure": True},
                        {"exit_code": 0, "teardown": {"attempted": True, "verified": None}}):
            with self.subTest(process=process), patch.object(suites, "list_suites", return_value=[self.spec, later]):
                self.process.reset_mock()
                self.process.side_effect = None
                self.process.return_value = {**process, "stdout": "PASS", "stderr": ""}
                results = self.invoke(allow_builds=True)
                self.assertEqual(1, self.process.call_count, "restore failure must block build, assertions, and next suite")
                self.assertEqual(["fail", "untested"], [row["status"] for row in results])
                self.assertTrue(results[0]["containment_failure"])
                self.assertIn("Unsafe owned resources", results[1]["reason"])
                self.assertEqual(3, len(results[0]["remaining_commands"]))
                self.assertTrue(all(command["status"] == "untested" for command in results[0]["remaining_commands"]))


if __name__ == "__main__":
    unittest.main()
