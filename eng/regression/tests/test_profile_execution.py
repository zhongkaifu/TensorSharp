"""Tiny offline full-CLI profile integrations; no binaries or network execute."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from eng.regression import profile_execution, runner, scenario_profiles, scenarios
from eng.regression import downloads
from eng.regression.configs import loads_jsonc

GIB = 1024 ** 3


def machine(available=30 * GIB):
    return {"memory": {"total_bytes": 32 * GIB, "available_bytes": available},
            "disk": {"free_bytes": 100 * GIB}, "gpus": [],
            "backends": {"cpu": {"available": True}}}


class ProfileExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.output = self.repo / "artifacts" / "run"
        self.cache = self.repo / "models"
        self.cache.mkdir()
        (self.repo / "config").mkdir()
        self.binary = self.repo / "synthetic-host"
        self.binary.write_bytes(b"never executed")
        self.model = self.cache / "tiny.gguf"
        self.model.write_bytes(b"GGUF-only-a-tiny-policy-fixture")
        self.config = {"backend": "cpu", "model": self.file_config(self.model)}
        self.config_path = self.repo / "config" / "tiny.json"
        self.config_path.write_text(json.dumps(self.config))
        self.catalog = {"schema_version": 1, "presets": [
            {"id": "tiny", "config": "config/tiny.json", "kind": "text", "family": "synthetic", "backends": ["cpu"], "scenarios": ["image_generation"]}],
            "scenarios": [{"id": "image_generation", "status": "automated", "execution": "profile", "kind": "image_generation"}]}
        self.catalog_path = self.repo / "catalog.json"
        self.profile = {"id": "first", "scenario": "image_generation", "preset": "tiny", "backend": "cpu",
                        "models": {}, "fixtures": {}, "parameters": {}, "expected": {}, "resources": {"ram_required_gib": 1}}
        self.profiles = [self.profile]
        self.profile_path = self.repo / "profiles.json"
        self.executed = []
        self.availability_calls = []
        self.needs_hardware = False
        self.run_effect = None
        self.module = SimpleNamespace(validate_profile=lambda profile: None, required_options=lambda profile: set(),
                                      availability=self.available, run=self.execute)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runner, "audit_catalog", return_value={"status": "pass", "errors": [], "config_count": 1, "scenario_count": 1}))
        self.stack.enter_context(patch.object(runner, "source_identity", return_value={"commit": "synthetic", "branch": "synthetic"}))
        self.stack.enter_context(patch.object(runner, "find_binary", return_value=[str(self.binary)]))
        self.stack.enter_context(patch.object(runner, "detect_hardware", return_value=machine()))
        self.hardware = self.stack.enter_context(patch.object(profile_execution, "detect_hardware", return_value=machine()))
        self.stack.enter_context(patch.object(scenario_profiles, "adapter", return_value=self.module))
        self.stack.enter_context(patch.object(profile_execution, "adapter", return_value=self.module))
        self.prepare = self.stack.enter_context(patch.object(profile_execution, "prepare_files", wraps=downloads.prepare_files))
        self.sizes = self.stack.enter_context(patch.object(runner, "sizes_for", wraps=runner.sizes_for))

    def file_config(self, path, **extra):
        content = path.read_bytes() if path.is_file() else b"missing-synthetic-weight"
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content), **extra}

    def available(self, case, context):
        self.availability_calls.append(context)
        if self.needs_hardware and not context.get("hardware", {}).get("backends", {}).get("cpu", {}).get("available"):
            return False, "adapter needs actual detected hardware in context"
        return True, "synthetic prerequisites available"

    def execute(self, case, context, result):
        self.executed.append(context)
        # Exercise the real host config validation, without launching its binary.
        args = scenarios._base_args(context, "image_generation")
        self.assertIn("--config", args)
        self.assertEqual("cpu", args[args.index("--backend") + 1])
        identities = context["profile_models"]["primary"]["identity"]
        self.assertTrue(identities and identities[0]["sha256"] == hashlib.sha256(self.model.read_bytes()).hexdigest())
        if self.run_effect:
            self.run_effect(context, result)
        else:
            scenarios._require(result, "synthetic domain oracle", 6 * 7 == 42, 42, 42)

    def invoke(self, *extra):
        self.catalog_path.write_text(json.dumps(self.catalog))
        self.config_path.write_text(json.dumps(self.config))
        self.profile_path.write_text(json.dumps({"schema_version": 1, "profiles": self.profiles}))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(["--repo", str(self.repo), "--catalog", str(self.catalog_path), "--scenario-profiles", str(self.profile_path),
                                "--model-root", str(self.cache), "--output", str(self.output), "--backend", "cpu", "--disk-headroom-gib", "0", *extra])
        report = json.loads((self.output / "report.json").read_text())
        return code, report, [row for row in report["results"] if row["id"].startswith("profile::")]

    def add_fixture(self):
        path = self.repo / "approved fixture.txt"
        path.write_bytes(b"small synthetic reference")
        self.profile["fixtures"]["reference"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return path

    def test_typed_profile_full_positive_uses_sanitized_effective_config(self):
        code, report, rows = self.invoke("--mode", "run")
        self.assertEqual(0, code, rows)
        self.assertEqual("pass", rows[0]["status"])
        self.assertEqual(1, len(self.executed))
        self.assertTrue(rows[0]["model_identity"])
        config = json.loads((self.output / "profiles/first/effective-config.json").read_text())
        self.assertNotIn("backend", config)
        self.assertIn("model", config)

    def test_plan_never_prepares_or_executes_weights(self):
        code, _, rows = self.invoke("--mode", "plan", "--allow-download")
        self.assertEqual(0, code)
        self.assertEqual("untested", rows[0]["status"])
        self.prepare.assert_not_called()
        self.assertFalse(self.executed)

    def test_missing_fixture_skips_before_weights(self):
        self.add_fixture().unlink()
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(3, code)
        self.assertEqual("skip", rows[0]["status"])
        self.assertIn("SHA-pinned local fixtures", rows[0]["reason"])
        self.prepare.assert_not_called()
        self.assertFalse(self.executed)

    def test_corrupt_fixture_fails_before_weights(self):
        self.add_fixture().write_bytes(b"changed")
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(1, code)
        self.assertEqual("fail", rows[0]["status"])
        self.assertIn("SHA256", rows[0]["reason"])
        self.prepare.assert_not_called()

    def test_missing_required_draft_does_not_execute_primary_alone(self):
        missing = self.cache / "draft.gguf"
        (self.repo / "config/draft.json").write_text(json.dumps({"model": self.file_config(missing)}))
        self.catalog["presets"].append({"id": "draft", "config": "config/draft.json", "kind": "text", "family": "synthetic", "backends": ["cpu"], "scenarios": []})
        self.profile["models"]["draft"] = "draft"
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(3, code)
        self.assertEqual("skip", rows[0]["status"])
        self.assertIn("weights missing", rows[0]["reason"])
        self.prepare.assert_not_called()
        self.assertFalse(self.executed)

    def test_unconsumed_optional_gated_companion_never_probed_or_prepared(self):
        self.config["mmproj"] = self.file_config(self.cache / "gated-projector.gguf", optional=True,
            urls=["https://huggingface.co/gated/project/resolve/" + "a" * 40 + "/gated-projector.gguf"])
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(0, code, rows)
        self.assertTrue(all(spec.option == "model" for call in self.sizes.call_args_list for spec in call.args[0]))
        self.assertTrue(all(spec.option == "model" for call in self.prepare.call_args_list for spec in call.args[0]))
        self.assertNotIn("mmproj", self.executed[0]["model_options"])

    def test_actual_ltx_config_keeps_required_model_and_omits_optional_same_option_child(self):
        # Only the shipped JSONC metadata is read: no LTX payload exists or is fetched.
        source = Path(__file__).resolve().parents[3] / "config/ltx-2.5-distilled.json"
        self.config = loads_jsonc(source.read_text(encoding="utf-8-sig"))
        code, _, rows = self.invoke("--mode", "plan")
        self.assertEqual(0, code, rows)
        files = rows[0]["required_files"]
        names = [Path(item["path"]).name for item in files]
        self.assertIn("ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf", names)
        self.assertFalse(any("ingredients" in name for name in names))
        self.assertEqual(1, sum(item["option"] == "model" for item in files))
        options = self.availability_calls[0]["model_options"]
        self.assertTrue(options["model"].endswith("ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"))
        self.assertIn("video-upscaler", options)
        self.prepare.assert_not_called()
        self.assertFalse(self.executed)

    def test_hardware_reaches_preflight_and_runtime_adapter(self):
        self.needs_hardware = True
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(0, code, rows)
        self.assertTrue(self.availability_calls[0]["hardware"]["backends"]["cpu"]["available"])
        self.assertTrue(self.executed[0]["hardware"]["backends"]["cpu"]["available"])

    def test_ram_drop_after_preparation_skips_without_loading(self):
        self.hardware.side_effect = lambda *args, **kwargs: machine(512 * 1024 ** 2) if self.prepare.called else machine()
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(3, code)
        self.prepare.assert_called_once()
        self.assertFalse(self.executed)
        self.assertIn("Resources changed after preparation", rows[0]["reason"])
        self.assertFalse(rows[0]["execution_resource_plan"]["eligible"])

    def test_output_volume_budget_is_independent_of_cache_drive(self):
        self.profile["resources"]["scratch_gib"] = 2
        with patch.object(profile_execution.shutil, "disk_usage", return_value=SimpleNamespace(free=GIB)):
            code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(3, code)
        self.assertIn("artifact volume", rows[0]["reason"])
        self.assertTrue(rows[0]["resource_plan"]["eligible"])
        self.prepare.assert_not_called()
        self.assertFalse(self.executed)

    def test_scratch_on_output_drive_does_not_require_cache_drive_space(self):
        self.profile["resources"]["scratch_gib"] = 2
        hardware = machine()
        hardware["disk"]["free_bytes"] = GIB
        self.hardware.return_value = hardware
        with patch.object(profile_execution, "same_volume", return_value=False), \
             patch.object(profile_execution.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * GIB)):
            code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(0, code, rows)
        self.assertEqual("pass", rows[0]["status"])
        self.assertEqual(0, rows[0]["resource_plan"]["estimates"]["disk_required_bytes"])
        self.assertEqual(2 * GIB, rows[0]["artifact_disk_plan"]["scratch_bytes"])

    def test_runtime_model_receipts_are_report_serializable(self):
        def serialize(context, result):
            result["inputs"]["models"] = context["profile_models"]
            scenarios._require(result, "synthetic domain oracle", True)
        self.run_effect = serialize
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(0, code, rows)
        self.assertEqual("pass", rows[0]["status"])
        self.assertEqual("tiny", rows[0]["inputs"]["models"]["primary"]["preset"]["id"])

    def test_fixture_change_during_prepare_fails_before_adapter(self):
        fixture = self.add_fixture()
        def prepare(files, policy):
            identities = downloads.prepare_files(files, policy)
            fixture.write_bytes(b"changed during model preparation")
            return identities
        self.prepare.side_effect = prepare
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(1, code)
        self.assertIn("SHA256 mismatch", rows[0]["reason"])
        self.assertFalse(self.executed)
        self.assertTrue(self.model.is_file())

    def test_unavailable_volume_root_terminates_without_parent_loop(self):
        class MissingRoot:
            checks = 0
            @property
            def parent(self):
                return self
            def resolve(self):
                return self
            def exists(self):
                self.checks += 1
                if self.checks > 5:
                    raise RuntimeError("synthetic watchdog: unavailable drive root loop")
                return False
            def __str__(self):
                return "synthetic-unmounted-root"
        missing = MissingRoot()
        with patch.object(profile_execution, "Path", return_value=missing), self.assertRaises((OSError, ValueError)):
            profile_execution.same_volume("unused", "unused")
        self.assertLessEqual(missing.checks, 5)

    def test_backend_fallback_cannot_become_pass(self):
        def fallback(context, result):
            proof = scenarios.backend_evidence("Backend: cpu\nfalling back to ggml_cpu", context["backend"])
            scenarios._require(result, "actual requested backend", proof["verified"], context["backend"], proof)
        self.run_effect = fallback
        code, _, rows = self.invoke("--mode", "run")
        self.assertEqual(1, code)
        self.assertEqual("fail", rows[0]["status"])
        self.assertIn("actual requested backend", rows[0]["reason"])

    def test_containment_failure_halts_later_profiles_and_global_suites(self):
        self.profiles.append({**copy.deepcopy(self.profile), "id": "second"})
        def unsafe(context, result):
            result["processes"].append({"stage": "synthetic-owned-child", "teardown": {"attempted": True, "verified": False}})
            scenarios._require(result, "cleanup verified", False)
        self.run_effect = unsafe
        with patch("eng.regression.suites.run_suites") as run_suites, patch("eng.regression.suites.list_suites", return_value=[{"id": "managed-fixture"}]):
            code, report, rows = self.invoke("--mode", "run", "--include-suites")
        self.assertEqual(1, code)
        self.assertEqual(["fail", "untested"], [row["status"] for row in rows])
        self.assertEqual("execution_halted", rows[1]["stage"])
        self.assertEqual(1, len(self.executed))
        self.prepare.assert_called_once()
        run_suites.assert_not_called()
        suite = next(row for row in report["results"] if row["id"] == "managed-fixture")
        self.assertEqual(("untested", "execution_halted"), (suite["status"], suite["stage"]))


if __name__ == "__main__":
    unittest.main()
