"""Offline end-to-end harness policy tests; no real executables or models run."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import reports, runner
from regression.configs import FileSpec, ResolvedConfig
from regression.downloads import PrepareError

GIB = 1024 ** 3


def machine():
    return {"memory": {"total_bytes": 32 * GIB, "available_bytes": 30 * GIB},
            "disk": {"free_bytes": 100 * GIB}, "gpus": [],
            "backends": {"ggml_cpu": {"available": True}, "cpu": {"available": True}}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Compare canonical identities, including Windows short TEMP aliases.
        self.repo = Path(self.temp.name).resolve()
        self.output = self.repo / "artifacts" / "result"
        self.model = self.repo / "models" / "tiny.gguf"
        self.model.parent.mkdir()
        self.model.write_bytes(b"GGUF-test-fixture")
        self.binary = self.repo / "fake-cli"
        self.binary.write_bytes(b"not executed")
        self.catalog = self.repo / "catalog.json"
        self.catalog_data = {"schema_version": 1,
            "presets": [{"id": "tiny", "config": "config/tiny.json", "kind": "text", "family": "test",
                         "backends": ["ggml_cpu", "cpu"], "scenarios": ["cli_text"]}],
            "scenarios": [{"id": "cli_text", "kind": "cli_text", "status": "automated", "fixture": {"prompt": "2+2"}}]}
        self.catalog.write_text(json.dumps(self.catalog_data))
        self.resolved = ResolvedConfig(self.repo / "config" / "tiny.json", {"model": str(self.model)},
                                      [FileSpec("model", self.model)])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.audit = self.stack.enter_context(patch.object(runner, "audit_catalog", return_value={
            "status": "pass", "errors": [], "config_count": 1, "scenario_count": 1}))
        self.stack.enter_context(patch.object(runner, "source_identity", return_value={"commit": "test", "branch": "test"}))
        self.stack.enter_context(patch.object(runner, "find_binary", return_value=[str(self.binary)]))
        self.hardware = self.stack.enter_context(patch.object(runner, "detect_hardware", return_value=machine()))
        self.config = self.stack.enter_context(patch.object(runner, "read_config", return_value=self.resolved))
        self.availability = self.stack.enter_context(patch.object(runner, "availability", return_value=(True, "available")))
        self.prepare = self.stack.enter_context(patch.object(runner, "prepare_files", return_value=[{"sha256": "fixture-only"}]))
        self.run = self.stack.enter_context(patch.object(runner, "run_scenario", return_value={"status": "pass", "reason": "assertions passed"}))

    def invoke(self, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = runner.main(["--repo", str(self.repo), "--catalog", str(self.catalog),
                                  "--output", str(self.output), "--backend", "ggml_cpu", *extra])
        return result, json.loads((self.output / "report.json").read_text())

    def test_plan_is_zero_execution_and_marks_untested(self):
        exit_code, report = self.invoke("--mode", "plan")
        self.assertEqual(0, exit_code)
        self.assertEqual("untested", report["results"][-1]["status"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_audit_success_is_only_inventory_pass(self):
        exit_code, report = self.invoke("--mode", "audit")
        self.assertEqual(0, exit_code)
        self.assertEqual(["config-coverage-audit"], [r["id"] for r in report["results"]])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_unknown_model_selector_is_configuration_error(self):
        exit_code, report = self.invoke("--mode", "run", "--models", "does-not-exist")
        self.assertEqual(2, exit_code)
        self.assertIn("model selector", report["results"][-1]["reason"])
        self.run.assert_not_called()

    def test_unknown_scenario_selector_is_configuration_error(self):
        exit_code, report = self.invoke("--mode", "run", "--scenarios", "does-not-exist")
        self.assertEqual(2, exit_code)
        self.assertIn("scenario selector", report["results"][-1]["reason"])

    def add_unbound_recipes(self):
        ids = ("embedding_media", "distributed", "client_live", "native_unit")
        self.catalog_data["scenarios"].extend({"id": identifier, "status": "recipe",
            "entry_point": "eng/fixture.py", "recipe": "Run with separately prepared fixture",
            "requirements": ["Explicit fixture and supported hardware"],
            "assertions": ["A meaningful result must be checked"], "evidence": ["eng/fixture.py"]}
            for identifier in ids)
        self.catalog.write_text(json.dumps(self.catalog_data))
        return ids

    def test_explicit_unbound_recipe_is_catalog_untested_not_selection_error(self):
        for identifier in self.add_unbound_recipes():
            with self.subTest(identifier=identifier):
                self.output = self.repo / "artifacts" / identifier
                exit_code, report = self.invoke("--mode", "run", "--scenarios", identifier)
                self.assertEqual(3, exit_code)
                rows = [row for row in report["results"] if row["id"] != "config-coverage-audit"]
                self.assertEqual(1, len(rows))
                self.assertEqual(("catalog::" + identifier, "catalog", "untested"),
                                 (rows[0]["id"], rows[0]["scope"], rows[0]["status"]))
                self.assertNotIn("preset", rows[0])
                self.assertNotIn("family", rows[0])
                self.assertNotIn("backend", rows[0])
                self.assertEqual(["A meaningful result must be checked"], rows[0]["unexecuted_assertions"])
                self.assertIn(identifier, (self.output / "report.md").read_text(encoding="utf-8"))
        self.config.assert_not_called()
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_broad_run_discloses_unbound_recipes_without_false_pass(self):
        ids = self.add_unbound_recipes()
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(0, exit_code)
        rows = [row for row in report["results"] if row.get("scope") == "catalog"]
        self.assertEqual(set(ids), {row["scenario"] for row in rows})
        self.assertTrue(all(row["status"] == "untested" and "backend" not in row for row in rows))
        self.assertEqual(1, self.run.call_count)
        self.assertEqual(["tiny::ggml_cpu::cli_text"], [row["id"] for row in report["results"]
                         if row["status"] == "pass" and row["id"] != "config-coverage-audit"])

    def test_strict_broad_run_keeps_unbound_recipe_coverage_incomplete(self):
        self.add_unbound_recipes()
        exit_code, report = self.invoke("--mode", "run", "--strict")
        self.assertEqual(3, exit_code)
        self.assertEqual(4, len([row for row in report["results"] if row.get("scope") == "catalog"]))

    def test_broad_plan_preserves_bound_recipe_and_separate_catalog_gaps(self):
        ids = self.add_unbound_recipes()
        self.catalog_data["presets"][0]["scenarios"].append("bound_recipe")
        self.catalog_data["scenarios"].append({"id": "bound_recipe", "status": "recipe", "recipe": "Existing model recipe"})
        self.catalog.write_text(json.dumps(self.catalog_data))
        exit_code, report = self.invoke("--mode", "plan")
        self.assertEqual(0, exit_code)
        self.assertEqual(set(ids), {row["scenario"] for row in report["results"] if row.get("scope") == "catalog"})
        bound = [row for row in report["results"] if row.get("scenario") == "bound_recipe"]
        self.assertEqual(1, len(bound))
        self.assertEqual(("tiny", "ggml_cpu", "untested"), (bound[0]["preset"], bound[0]["backend"], bound[0]["status"]))
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_unselected_catalog_recipes_do_not_change_explicit_model_case(self):
        self.add_unbound_recipes()
        exit_code, report = self.invoke("--mode", "run", "--scenarios", "cli_text", "--strict")
        self.assertEqual(0, exit_code)
        self.assertFalse(any(row.get("scope") == "catalog" for row in report["results"]))

    def test_inventory_pass_does_not_hide_zero_executions(self):
        self.availability.return_value = (False, "CLI unavailable")
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(3, exit_code)
        self.assertEqual("skip", report["results"][-1]["status"])
        self.run.assert_not_called()

    def containment_fixture(self, process):
        first = self.catalog_data["presets"][0]
        first["scenarios"] = ["cli_text", "cli_decode"]
        self.catalog_data["presets"] = [first, {**first, "id": "next-model"}]
        self.catalog_data["scenarios"].append({"id": "cli_decode", "kind": "cli_decode", "status": "automated"})
        self.catalog.write_text(json.dumps(self.catalog_data))
        # A buggy adapter claiming success must not hide uncertain cleanup.
        self.run.side_effect = [{"status": "pass", "reason": "semantic fixture passed", "processes": [process]},
                                AssertionError("later scenario must never start")]

    def test_containment_failure_halts_remaining_scenarios_models_backends_and_suites(self):
        self.containment_fixture({"stage": "decode", "exit_code": 125, "parent_exit_code": 0,
            "containment_failure": True, "teardown": {"attempted": True, "verified": False, "reason": "child remains"}})
        with patch("regression.suites.run_suites") as suite_run, patch("regression.ctest.run_native_tests") as native_run, \
             patch.object(runner, "sizes_for", wraps=runner.sizes_for) as size_probe:
            exit_code, report = self.invoke("--mode", "run", "--backend", "all", "--include-suites",
                "--native-test-build-dir", str(self.repo / "native-build"), "--allow-download")
        self.assertEqual(1, exit_code)
        self.assertEqual(1, self.run.call_count)
        self.assertEqual(1, self.prepare.call_count, "next model must not prepare/download")
        self.assertEqual(1, self.config.call_count, "next model must not resolve files")
        self.assertEqual(1, size_probe.call_count, "next model must not probe network metadata")
        suite_run.assert_not_called()
        native_run.assert_not_called()
        model_results = [row for row in report["results"] if row.get("preset")]
        self.assertEqual(8, len(model_results))
        self.assertEqual("fail", model_results[0]["status"])
        self.assertTrue(all(row["status"] == "untested" for row in model_results[1:]))
        self.assertIn("Unsafe owned resources", report["execution_halt"]["reason"])
        remaining_suites = [row for row in report["results"] if row["id"] not in {"config-coverage-audit"} and not row.get("preset")]
        self.assertEqual(6, len(remaining_suites))
        self.assertTrue(all(row["status"] == "untested" for row in remaining_suites))
        self.assertEqual("child remains", model_results[0]["processes"][0]["teardown"]["reason"])

    def test_attempted_teardown_unknown_halts_even_without_failure_flag(self):
        self.containment_fixture({"stage": "server", "exit_code": 0,
                                  "teardown": {"attempted": True, "verified": None}})
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(1, exit_code)
        self.assertEqual(1, self.run.call_count)
        self.assertEqual(1, self.prepare.call_count)
        model_results = [row for row in report["results"] if row.get("preset")]
        self.assertEqual(["fail", "untested", "untested", "untested"], [row["status"] for row in model_results])

    def test_suite_containment_failure_blocks_later_native_suite(self):
        suite_result = {"id": "server_unit", "status": "fail", "reason": "cleanup uncertain",
                        "processes": [{"exit_code": 0, "teardown": {"attempted": True, "verified": None}}]}
        with patch("regression.suites.run_suites", return_value=[suite_result]), \
             patch("regression.ctest.run_native_tests") as native_run:
            exit_code, report = self.invoke("--mode", "run", "--include-suites",
                "--native-test-build-dir", str(self.repo / "native-build"))
        self.assertEqual(1, exit_code)
        native_run.assert_not_called()
        self.assertEqual("native_ctest", report["results"][-1]["id"])
        self.assertEqual("untested", report["results"][-1]["status"])

    def test_adapter_omitting_failed_process_cannot_bypass_central_guard(self):
        self.containment_fixture({})
        self.run.side_effect = [{"status": "pass", "reason": "adapter forgot process metadata"},
                                AssertionError("must not run")]
        state = {"containment_failure": True, "teardown": {"attempted": True, "verified": False}}
        with patch.object(runner, "unsafe_process_state", side_effect=[None, state]):
            exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(1, exit_code)
        self.assertEqual(1, self.run.call_count)
        self.assertEqual(1, self.prepare.call_count)
        self.assertIn("execution_halt", report)

    def test_cancellation_with_unverified_cleanup_fails_and_marks_remaining_untested(self):
        self.containment_fixture({})
        self.run.side_effect = KeyboardInterrupt()
        state = {"containment_failure": True, "teardown": {"attempted": True, "verified": False}}
        with patch.object(runner, "unsafe_process_state", side_effect=[None, state]):
            exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(1, exit_code)
        self.assertTrue(report["cancelled"])
        self.assertEqual(1, self.prepare.call_count)
        models = [row for row in report["results"] if row.get("preset")]
        self.assertEqual(4, len(models))
        self.assertTrue(all(row["status"] == "untested" for row in models))

    def test_reproduce_output_is_unique_sibling_and_preserves_model_locations(self):
        original = self.repo / "selected-volume" / "original-results"
        original.mkdir(parents=True)
        marker = original / "existing-evidence.txt"
        marker.write_text("unchanged")
        model_root = self.repo / "model-volume"
        existing_root = self.repo / "read-only-model-volume"
        args = runner.parse_args(["--output", str(original), "--model-root", str(model_root),
                                  "--existing-root", str(existing_root)])
        preset, scenario = self.catalog_data["presets"][0], self.catalog_data["scenarios"][0]
        first = runner.reproduce_args(args, self.repo, model_root, preset, "cpu", scenario)
        second = runner.reproduce_args(args, self.repo, model_root, preset, "cpu", scenario)
        one, two = (Path(command[command.index("--output") + 1]) for command in (first, second))
        self.assertEqual(original.parent, one.parent)
        self.assertEqual(one.parent, two.parent)
        self.assertNotEqual(one, two)
        self.assertNotEqual(original, one)
        self.assertFalse(one.exists())
        self.assertEqual("unchanged", marker.read_text())
        self.assertEqual(str(model_root), first[first.index("--model-root") + 1])
        self.assertEqual(str(existing_root), first[first.index("--existing-root") + 1])

    def test_backend_unavailable_never_falls_back(self):
        self.hardware.return_value = machine()
        self.hardware.return_value["backends"]["ggml_cpu"]["available"] = False
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(3, exit_code)
        self.assertIn("backend unavailable", report["results"][-1]["reason"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_run_records_assertion_failure_not_process_success(self):
        self.run.return_value = {"status": "fail", "reason": "expected 4, got 5",
                                 "assertions": [{"expected": "4", "actual": "5", "passed": False}]}
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(1, exit_code)
        self.assertEqual("5", report["results"][-1]["assertions"][0]["actual"])
        self.assertTrue(report["results"][-1]["reproduce"])

    def test_prepare_failure_preserves_stage_and_error_code(self):
        self.prepare.side_effect = PrepareError("verify", "hash_mismatch", self.model, "bad SHA256")
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(1, exit_code)
        self.assertEqual("verify", report["results"][-1]["stage"])
        self.assertEqual("hash_mismatch", report["results"][-1]["error_code"])
        self.run.assert_not_called()

    def test_memory_budget_skip_does_not_prepare_or_execute(self):
        exit_code, report = self.invoke("--mode", "run", "--ram-budget-gib", "1")
        self.assertEqual(3, exit_code)
        self.assertIn("RAM", report["results"][-1]["reason"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_auto_backend_records_resource_rejection_and_cpu_selection(self):
        self.catalog_data["presets"][0]["backends"] = ["ggml_cuda", "ggml_cpu"]
        self.catalog.write_text(json.dumps(self.catalog_data))
        self.hardware.return_value["backends"]["ggml_cuda"] = {"available": True}
        self.hardware.return_value["gpus"] = [{"index": 0, "memory_free_bytes": 2 * GIB}]
        exit_code, report = self.invoke("--mode", "run", "--backend", "auto")
        self.assertEqual(0, exit_code)
        self.assertEqual("ggml_cpu", report["results"][-1]["backend"])
        selection = report["backend_selections"][0]
        self.assertEqual("ggml_cpu", selection["selected"])
        cuda_plan = next(p for p in selection["attempts"] if p["backend"] == "ggml_cuda")
        self.assertFalse(cuda_plan["eligible"])

    def cpu_moe_fixture(self, family="qwen4exp", layout="moe"):
        self.catalog_data["presets"][0].update(family=family, layout=layout,
            backends=["cuda", "ggml_cuda", "ggml_cpu", "cpu"])
        self.catalog.write_text(json.dumps(self.catalog_data))
        self.hardware.return_value["backends"].update(ggml_cuda={"available": True}, cuda={"available": True})
        self.hardware.return_value["gpus"] = [{"index": 0, "memory_free_bytes": 16 * GIB,
            "compute_capability": 8.6, "driver_version": "580.100"}]

    def test_explicit_cpu_moe_profile_fits_large_file_only_with_declared_requirements(self):
        self.cpu_moe_fixture()
        with patch.object(runner, "sizes_for", return_value=(75 * GIB, 0)):
            exit_code, report = self.invoke("--mode", "run", "--backend", "ggml_cuda", "--cpu-moe",
                "--ram-required-gib", "16", "--vram-required-gib", "6")
        self.assertEqual(0, exit_code)
        result = report["results"][-1]
        self.assertEqual("ggml_cuda", result["backend"])
        self.assertEqual(75 * GIB, result["resource_plan"]["estimates"]["weight_bytes"])
        self.assertEqual(16 * GIB, result["resource_plan"]["estimates"]["ram_required_bytes"])
        self.assertEqual(6 * GIB, result["resource_plan"]["estimates"]["vram_required_bytes"])
        context = self.run.call_args.args[1]
        self.assertEqual({"CUDA_VISIBLE_DEVICES": "0", "TS_CPU_MOE": "1"}, context["env"])
        self.assertEqual("cpu_moe", context["offload_policy"]["name"])
        self.assertEqual("explicit_user_requirements", report["policy"]["offload"]["estimate_source"])
        self.assertIn("--cpu-moe", result["reproduce"])
        self.assertIn("--ram-required-gib", result["reproduce"])
        self.assertIn("--vram-required-gib", result["reproduce"])

    def test_default_large_model_is_still_conservatively_skipped(self):
        self.cpu_moe_fixture()
        with patch.object(runner, "sizes_for", return_value=(75 * GIB, 0)):
            exit_code, report = self.invoke("--mode", "run", "--backend", "ggml_cuda")
        self.assertEqual(3, exit_code)
        self.assertFalse(report["policy"]["offload"]["requested"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_cpu_moe_cannot_override_current_ram_or_gpu_budget(self):
        self.cpu_moe_fixture()
        for ram, vram, expected in (("16", "17", "VRAM budget"), ("31", "6", "RAM exceeds budget")):
            with self.subTest(ram=ram, vram=vram):
                self.output = self.repo / "artifacts" / ("budget-" + ram + "-" + vram)
                exit_code, report = self.invoke("--mode", "run", "--backend", "ggml_cuda", "--cpu-moe",
                    "--ram-required-gib", ram, "--vram-required-gib", vram)
                self.assertEqual(3, exit_code)
                self.assertIn(expected, report["results"][-1]["reason"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_cpu_moe_still_requires_known_size_and_full_download_budget(self):
        self.cpu_moe_fixture()
        for total, missing, expected in ((None, None, "size unknown"), (75 * GIB, 75 * GIB, "download budget")):
            with self.subTest(total=total, missing=missing):
                self.output = self.repo / "artifacts" / ("download-" + str(total))
                with patch.object(runner, "sizes_for", return_value=(total, missing)):
                    exit_code, report = self.invoke("--mode", "run", "--backend", "ggml_cuda", "--cpu-moe",
                        "--ram-required-gib", "16", "--vram-required-gib", "6", "--allow-download")
                self.assertEqual(3, exit_code)
                self.assertIn(expected, report["results"][-1]["reason"])
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_cpu_moe_auto_no_supported_backend_rejects_before_metadata_probe(self):
        self.cpu_moe_fixture()
        self.catalog_data["presets"][0]["backends"] = ["cuda"]
        self.catalog.write_text(json.dumps(self.catalog_data))
        with patch.object(runner, "sizes_for") as probe:
            exit_code, report = self.invoke("--mode", "run", "--backend", "auto", "--cpu-moe",
                "--ram-required-gib", "16", "--vram-required-gib", "6", "--allow-download")
        self.assertEqual(3, exit_code)
        self.assertEqual("unsupported", report["results"][-1]["status"])
        self.config.assert_not_called()
        probe.assert_not_called()
        self.prepare.assert_not_called()

    def test_cpu_moe_invalid_family_layout_or_backend_never_prepares_files(self):
        for family, layout, backend in (("qwen35", "dense", "auto"), ("test", "moe", "ggml_cuda"),
                                        ("qwen4exp", "moe", "cuda")):
            with self.subTest(family=family, layout=layout, backend=backend):
                self.output = self.repo / "artifacts" / (family + "-" + layout + "-" + backend)
                self.cpu_moe_fixture(family, layout)
                exit_code, report = self.invoke("--mode", "run", "--backend", backend, "--cpu-moe",
                    "--ram-required-gib", "16", "--vram-required-gib", "6")
                self.assertEqual(3, exit_code)
                self.assertEqual("unsupported", report["results"][-1]["status"])
        self.config.assert_not_called()
        self.prepare.assert_not_called()
        self.run.assert_not_called()

    def test_cpu_moe_auto_does_not_choose_unsupported_direct_cuda(self):
        self.cpu_moe_fixture("qwen35")
        self.hardware.return_value["backends"]["ggml_cuda"]["available"] = False
        with patch.object(runner, "sizes_for", return_value=(75 * GIB, 0)):
            exit_code, report = self.invoke("--mode", "run", "--backend", "auto", "--cpu-moe",
                "--ram-required-gib", "16", "--vram-required-gib", "6")
        self.assertEqual(0, exit_code)
        self.assertEqual("ggml_cpu", report["results"][-1]["backend"])
        direct_cuda = next(row for row in report["backend_selections"][0]["attempts"] if row["backend"] == "cuda")
        self.assertFalse(direct_cuda["eligible"])
        self.assertEqual({"TS_CPU_MOE": "1"}, self.run.call_args.args[1]["env"])

    def test_default_run_passes_no_cpu_moe_environment(self):
        self.invoke("--mode", "run")
        self.assertNotIn("TS_CPU_MOE", self.run.call_args.args[1]["env"])
        self.assertFalse(self.run.call_args.args[1]["offload_policy"]["requested"])

    def test_cumulative_download_cap_stops_later_model_before_prepare(self):
        first = self.catalog_data["presets"][0]
        self.catalog_data["presets"] = [{**first, "id": f"tiny-{i}"} for i in range(3)]
        self.catalog.write_text(json.dumps(self.catalog_data))
        self.run.side_effect = lambda *_: {"status": "pass", "reason": "fixture passed"}
        with patch.object(runner, "sizes_for", return_value=(10 * GIB, 10 * GIB)):
            exit_code, report = self.invoke("--mode", "run", "--allow-download", "--max-download-gib", "20")
        self.assertEqual(0, exit_code)
        self.assertEqual(2, self.prepare.call_count)
        self.assertEqual(2, self.run.call_count)
        self.assertEqual("skip", report["results"][-1]["status"])
        self.assertIn("whole-run download budget", report["results"][-1]["reason"])

    def test_disk_inventory_uses_model_volume_and_explicit_runtime_dirs(self):
        model_root = self.repo / "selected-model-volume"
        runtime_dir = self.repo / "native-runtime"
        self.invoke("--mode", "plan", "--model-root", str(model_root), "--runtime-dir", str(runtime_dir))
        self.assertEqual(model_root, self.hardware.call_args.args[0])
        self.assertIn(str(runtime_dir), self.hardware.call_args.kwargs["runtime_dirs"])

    def test_reproduction_preserves_budget_and_existing_root(self):
        existing_root = self.repo / "existing-models"
        _, report = self.invoke("--mode", "run", "--ram-budget-gib", "12", "--existing-root", str(existing_root),
                                "--assume-backend-available")
        command = report["results"][-1]["reproduce"]
        self.assertIn("--ram-budget-gib", command)
        self.assertIn("--assume-backend-available", command)
        self.assertIn(str(existing_root), command)
        reproduction_output = Path(command[command.index("--output") + 1])
        self.assertEqual(self.output.parent, reproduction_output.parent)
        self.assertNotEqual(self.output, reproduction_output)
        self.assertFalse(reproduction_output.exists())

    def test_external_engine_source_is_separate_and_preserved_for_reproduction(self):
        engine = self.repo / "engine-checkout"
        _, report = self.invoke("--mode", "run", "--engine-repo", str(engine))
        self.assertIn("caller-declared", report["engine_source"]["provenance"])
        self.assertIn("--engine-repo", report["results"][-1]["reproduce"])
        self.assertIn(str(engine), report["results"][-1]["reproduce"])

    def test_cancellation_returns_130_and_preserves_report(self):
        self.run.side_effect = KeyboardInterrupt()
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(130, exit_code)
        self.assertTrue(report["cancelled"])
        self.assertEqual("untested", report["results"][-1]["status"])

    def test_audit_failure_prevents_all_execution(self):
        self.audit.return_value = {"status": "fail", "errors": ["missing config"], "config_count": 1, "scenario_count": 1}
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(2, exit_code)
        self.assertEqual("fail", report["results"][0]["status"])
        self.hardware.assert_not_called()
        self.run.assert_not_called()

    def test_nan_infinite_negative_and_zero_timeout_rejected(self):
        for flag, value in (("--max-download-gib", "nan"), ("--vram-budget-gib", "inf"),
                            ("--disk-headroom-gib", "-1"), ("--timeout-seconds", "0"),
                            ("--ram-required-gib", "0"), ("--vram-budget-gib", "0")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    runner.parse_args([flag, value])
                self.assertEqual(2, error.exception.code)

    def test_explicit_backend_list_keeps_requested_identity(self):
        self.assertEqual(["cuda"], runner.select_backends("cuda", ["cpu"], machine()))

    def test_existing_unique_local_weights_preferred(self):
        other = self.repo / "external" / "tiny.gguf"
        other.parent.mkdir()
        other.write_bytes(b"GGUF")
        missing = self.repo / "download" / "tiny.gguf"
        resolved = ResolvedConfig(self.resolved.source, {"model": str(missing)}, [FileSpec("model", missing)])
        options, files = runner.use_existing(resolved, {"tiny.gguf": [other]})
        self.assertEqual(str(other), options["model"])
        self.assertEqual(other, files[0].path)

    def test_ambiguous_local_weights_are_not_chosen(self):
        missing = self.repo / "download" / "tiny.gguf"
        resolved = ResolvedConfig(self.resolved.source, {"model": str(missing)}, [FileSpec("model", missing)])
        _, files = runner.use_existing(resolved, {"tiny.gguf": [self.repo / "a" / "tiny.gguf", self.repo / "b" / "tiny.gguf"]})
        self.assertEqual(missing, files[0].path)

    def test_split_shards_cannot_be_spread_across_roots(self):
        first = self.repo / "a" / "model-00001-of-00002.gguf"
        second = self.repo / "b" / "model-00002-of-00002.gguf"
        resolved = ResolvedConfig(self.resolved.source, {"model": str(first)},
                                  [FileSpec("model", first), FileSpec("model", second)])
        with self.assertRaisesRegex(ValueError, "different directories"):
            runner.use_existing(resolved, {})

    def test_effective_config_drops_user_agent_permissions(self):
        options = runner.effective_options({"model": "a.gguf", "code-exec": True,
            "code-exec-allow-network": True, "skills-dir": "/private", "host": "0.0.0.0",
            "draft-model": "draft.gguf", "mmproj": "vision.gguf", "spec": True}, ["cli_text"], "cpu")
        self.assertEqual("a.gguf", options["model"])
        for name in ("code-exec", "code-exec-allow-network", "skills-dir", "host", "draft-model", "mmproj", "spec"):
            self.assertNotIn(name, options)

    def test_ngram_and_image_smoke_only_prepare_used_companions(self):
        options = runner.effective_options({"model": "base.gguf", "draft-model": "draft.gguf", "spec": "dflash"},
                                           ["speculation_ngram"], "cpu")
        self.assertEqual(options, {"model": "base.gguf"})
        options = runner.effective_options({"model": "base.gguf", "qwen-image-mmproj": "editing.gguf",
                                           "qwen-image-vae": "vae.gguf", "qwen-image-vl": "encoder.gguf"},
                                           ["image_generation_smoke"], "cpu")
        self.assertNotIn("qwen-image-mmproj", options)
        files = [FileSpec(option, self.repo / path, optional=True) for option, path in options.items()]
        files = runner.required_scenario_files(files, [{"id": "image_generation_smoke"}])
        self.assertTrue(all(not spec.optional for spec in files if spec.option != "model"))


class ReportPolicyTests(unittest.TestCase):
    def test_status_aggregation(self):
        cases = [([], 3), (["skip"], 3), (["unsupported"], 3), (["untested"], 3),
                 (["pass"], 0), (["pass", "fail"], 1), (["unknown"], 2)]
        for states, expected in cases:
            self.assertEqual(expected, reports.aggregate_exit([{"status": state} for state in states]))
        self.assertEqual(3, reports.aggregate_exit([{"status": "pass"}, {"status": "skip"}], strict=True))
        self.assertEqual(130, reports.aggregate_exit([], cancelled=True))

    def test_diagnostics_redact_secrets_and_omit_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.gguf").write_bytes(b"not part of diagnostics")
            (root / "stdout.log").write_text("Authorization: Bearer test-secret\napi_key=private-key")
            report = {"results": [{"id": "case", "status": "fail", "reason": "incorrect output"}],
                      "source": {"commit": "test"}, "api_key": "private-key", "mode": "run"}
            reports.write_reports(root, report)
            import zipfile
            with zipfile.ZipFile(root / "diagnostics.zip") as archive:
                self.assertNotIn("model.gguf", archive.namelist())
                content = "\n".join(archive.read(name).decode() for name in archive.namelist())
            self.assertNotIn("test-secret", content)
            self.assertNotIn("private-key", content)
            self.assertIn("<redacted>", content)


if __name__ == "__main__":
    unittest.main()
