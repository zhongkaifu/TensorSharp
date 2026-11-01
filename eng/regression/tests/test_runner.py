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
        self.repo = Path(self.temp.name)
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

    def test_inventory_pass_does_not_hide_zero_executions(self):
        self.availability.return_value = (False, "CLI unavailable")
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(3, exit_code)
        self.assertEqual("skip", report["results"][-1]["status"])
        self.run.assert_not_called()

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
