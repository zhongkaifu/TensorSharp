"""Synthetic auto-planning failures must not stop independent selected models."""
import json
import unittest
from unittest.mock import patch

import test_runner as fixture
from regression.downloads import PrepareError
from regression.configs import FileSpec


class PlanningFailureTests(unittest.TestCase):
    setUp = fixture.RunnerTests.setUp
    invoke = fixture.RunnerTests.invoke

    def test_auto_probe_failure_continues_next_preset_with_reproduction(self):
        first = self.catalog_data["presets"][0]
        self.catalog_data["presets"] = [{**first, "id": "blocked"}, {**first, "id": "working"}]
        self.catalog.write_text(json.dumps(self.catalog_data))
        failed = PrepareError("probe", "credentials_or_license", self.repo / "models/private.gguf", "Access refused")
        with patch("regression.runner.sizes_for", side_effect=[failed, (16, 0), (16, 0)]) as sizes:
            exit_code, report = self.invoke("--mode", "run", "--backend", "auto", "--probe-network")
        self.assertEqual(exit_code, 1)
        self.assertEqual(self.run.call_count, 1)
        self.assertEqual(self.prepare.call_count, 1)
        self.assertEqual(sizes.call_count, 3)
        failure = next(result for result in report["results"] if result.get("preset") == "blocked")
        self.assertEqual(failure["status"], "fail")
        self.assertEqual(failure["stage"], "probe")
        self.assertEqual(failure["error"]["path"], str(self.repo / "models/private.gguf"))
        self.assertEqual(failure["error_code"], "credentials_or_license")
        self.assertIn("--probe-network", failure["reproduce"])
        self.assertTrue(failure["required_files"])
        self.assertEqual(report["backend_selections"][0]["selected"], None)
        self.assertEqual(report["results"][-1]["status"], "pass")

    def test_probe_failure_does_not_claim_recipe_execution(self):
        self.catalog_data["scenarios"].append({"id": "recipe", "status": "recipe", "recipe": "manual hardware recipe"})
        self.catalog_data["scenarios"].append({"id": "not_supported", "status": "unsupported", "reason": "not implemented"})
        self.catalog_data["presets"][0]["scenarios"].extend(["recipe", "not_supported"])
        self.catalog.write_text(json.dumps(self.catalog_data))
        with patch("regression.runner.sizes_for", side_effect=PrepareError("probe", "timeout", self.model, "Timed out")) as sizes:
            _, report = self.invoke("--mode", "plan", "--backend", "auto", "--probe-network")
        by_scenario = {result.get("scenario"): result for result in report["results"]}
        self.assertEqual(by_scenario["cli_text"]["status"], "fail")
        self.assertEqual(by_scenario["recipe"]["status"], "untested")
        self.assertEqual(by_scenario["not_supported"]["status"], "unsupported")
        self.assertEqual(sizes.call_count, 1)
        self.prepare.assert_not_called()

    def test_model_limit_counts_unique_presets_not_backend_runs(self):
        first = self.catalog_data["presets"][0]
        self.catalog_data["presets"] = [{**first, "id": "first"}, {**first, "id": "second"}]
        self.catalog.write_text(json.dumps(self.catalog_data))
        self.run.side_effect = lambda *_: {"status": "pass", "reason": "fixture passed"}
        exit_code, report = self.invoke("--mode", "run", "--backend", "all", "--max-models", "1")
        self.assertEqual(exit_code, 0)
        first_results = [result for result in report["results"] if result.get("preset") == "first"]
        self.assertEqual(len(first_results), 2)
        self.assertTrue(all(result["status"] == "pass" for result in first_results))
        second_results = [result for result in report["results"] if result.get("preset") == "second"]
        self.assertTrue(all(result["status"] == "skip" for result in second_results))
        self.assertEqual(self.prepare.call_count, 2)
        self.assertEqual(self.run.call_count, 2)

    def test_missing_media_fixture_does_not_plan_or_prepare_projector(self):
        self.catalog_data["scenarios"].append({"id": "cli_image", "status": "automated"})
        self.catalog_data["presets"][0]["scenarios"].append("cli_image")
        self.catalog.write_text(json.dumps(self.catalog_data))
        projector = self.repo / "models" / "missing-projector.gguf"
        self.resolved.options["mmproj"] = str(projector)
        self.resolved.files.append(FileSpec("mmproj", projector, size_bytes=100 * 1024**3, optional=True))
        self.availability.side_effect = lambda case, _: (False, "media fixture is absent") if case["id"] == "cli_image" else (True, "ready")
        exit_code, report = self.invoke("--mode", "run", "--backend", "auto", "--allow-download")
        self.assertEqual(exit_code, 0)
        self.assertEqual(self.run.call_count, 1)
        self.assertTrue(all(spec.option != "mmproj" for spec in self.prepare.call_args.args[0]))
        self.assertNotIn("mmproj", self.run.call_args.args[1]["model_options"])
        image = next(result for result in report["results"] if result.get("scenario") == "cli_image")
        self.assertEqual(image["status"], "skip")
        self.assertIn("media fixture", image["reason"])

    def test_selected_media_projector_is_required_and_skipped_companion_not_used(self):
        self.catalog_data["scenarios"].append({"id": "cli_image", "status": "automated"})
        self.catalog_data["presets"][0]["scenarios"].append("cli_image")
        self.catalog.write_text(json.dumps(self.catalog_data))
        projector = self.repo / "models" / "projector.gguf"
        projector.write_bytes(b"synthetic projector fixture")
        self.resolved.options["mmproj"] = str(projector)
        self.resolved.files.append(FileSpec("mmproj", projector, optional=True))
        self.prepare.return_value = [{"path": str(self.model), "status": "local"},
                                     {"path": str(projector), "status": "optional-skipped", "reason": "synthetic unavailable companion"}]
        exit_code, report = self.invoke("--mode", "run")
        self.assertEqual(exit_code, 0)
        prepared_projector = next(spec for spec in self.prepare.call_args.args[0] if spec.option == "mmproj")
        self.assertFalse(prepared_projector.optional)
        self.assertEqual(self.run.call_count, 1)
        self.assertNotIn("mmproj", self.run.call_args.args[1]["model_options"])
        image = next(result for result in report["results"] if result.get("scenario") == "cli_image")
        self.assertEqual(image["status"], "skip")
        self.assertIn("companion unavailable", image["reason"])
