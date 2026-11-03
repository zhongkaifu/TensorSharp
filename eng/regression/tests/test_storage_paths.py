"""Portable mock-volume checks: source, models, existing files and reports differ.

Directories stand in for separate mounts/drives, with mocked per-volume free
space. These tests neither access actual D:/E: storage nor download models.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import downloads, hardware, processes, scenarios
from regression.configs import FileSpec, ResolvedConfig
from regression.downloads import DownloadPolicy, PrepareError
import test_runner as runner_fixture

GIB = 1024 ** 3


class StorageVolumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source_volume = self.root / "system-volume"
        self.model_volume = self.root / "model-volume"
        self.report_volume = self.root / "report-volume"
        for directory in (self.source_volume, self.model_volume, self.report_volume):
            directory.mkdir()

    def disk_usage(self, path):
        path = Path(path).resolve()
        free = 100 * GIB if path.is_relative_to(self.model_volume) else GIB
        return SimpleNamespace(total=200 * GIB, used=200 * GIB - free, free=free)

    def test_model_inventory_uses_selected_volume_for_a_not_yet_created_root(self):
        target = self.model_volume / "models" / "future-cache"
        with patch.object(hardware.shutil, "disk_usage", side_effect=self.disk_usage) as probe:
            inventory = hardware._disk(target)
        self.assertEqual(inventory["path"], str(self.model_volume))
        self.assertEqual(inventory["free_bytes"], 100 * GIB)
        probe.assert_called_once_with(self.model_volume)
        self.assertFalse(target.exists())

    def test_download_probe_uses_actual_target_ancestor_not_cwd_or_report_volume(self):
        target = self.model_volume / "models" / "model.gguf"
        with patch.object(downloads.shutil, "disk_usage", side_effect=self.disk_usage) as probe:
            available = downloads._free_bytes(target)
        self.assertEqual(available, 100 * GIB)
        probe.assert_called_once_with(self.model_volume)

    def test_insufficient_model_headroom_blocks_before_transfer_with_target_path(self):
        cache = self.model_volume / "models"
        spec = FileSpec("model", cache / "model.gguf", (), "a" * 64, 3 * GIB)
        policy = DownloadPolicy(cache, 10 * GIB, GIB, True)
        planned = [{"path": str(spec.path), "exists": False, "status": "available",
                    "size_bytes": 3 * GIB, "remaining_bytes": 3 * GIB, "source": None}]
        with patch.object(downloads, "probe_files", return_value=planned), \
             patch.object(downloads, "_free_bytes", return_value=3 * GIB) as free, \
             patch.object(downloads, "_download", side_effect=AssertionError("must not start transfer")) as transfer:
            with self.assertRaises(PrepareError) as error:
                downloads.prepare_files([spec], policy)
        self.assertEqual(error.exception.code, "disk_budget")
        self.assertEqual(error.exception.path, str(cache.resolve()))
        self.assertIn("headroom", str(error.exception))
        free.assert_called_once_with(cache.resolve() / "placeholder")
        transfer.assert_not_called()
        self.assertFalse(cache.exists())

    def test_roomy_model_drive_does_not_hide_full_artifact_drive(self):
        artifact_dir = self.report_volume / "artifacts" / "case"
        result = {"evidence": {}}
        with patch.object(scenarios.shutil, "disk_usage", side_effect=self.disk_usage) as probe:
            with self.assertRaisesRegex(scenarios.ScenarioUntested, "disk.*headroom"):
                scenarios._ensure_logit_disk({"artifact_dir": artifact_dir, "disk_headroom_gib": 5}, result)
        probe.assert_called_once_with(self.report_volume)
        evidence = result["evidence"]["artifact_disk_checks"][0]
        self.assertEqual(evidence["path"], str(self.report_volume))
        self.assertEqual(evidence["available_bytes"], GIB)
        self.assertEqual(evidence["headroom_bytes"], 5 * GIB)
        self.assertGreater(evidence["required_bytes"], evidence["available_bytes"])

    def test_artifact_guard_accepts_selected_spacious_output_volume(self):
        result = {"evidence": {}}
        with patch.object(scenarios.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * GIB)) as probe:
            scenarios._ensure_logit_disk({"artifact_dir": self.report_volume / "case", "disk_headroom_gib": 10}, result, sequence=True)
        probe.assert_called_once_with(self.report_volume)
        self.assertEqual(result["evidence"]["artifact_disk_checks"][0]["required_bytes"], 11 * GIB)

    def test_existing_weights_on_another_volume_are_verified_without_copy_or_download(self):
        existing = self.source_volume / "existing.gguf"
        payload = b"GGUF-synthetic-existing-weights"
        existing.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        spec = FileSpec("model", existing, sha256=digest)
        policy = DownloadPolicy(self.model_volume / "new-models", 0, 5 * GIB, False)
        with patch.object(downloads, "_open", side_effect=AssertionError("no network")), \
             patch.object(downloads, "_free_bytes", side_effect=AssertionError("no new disk allocation")):
            result = downloads.prepare_files([spec], policy)
        self.assertEqual(result[0].status, "local")
        self.assertEqual(result[0].sha256, digest)
        self.assertEqual(existing.read_bytes(), payload)
        self.assertFalse(policy.cache_root.exists())

    def test_host_temporary_paths_are_explicit_and_artifact_caches_follow_output(self):
        temporary = self.model_volume / "temp"
        output = self.report_volume / "run-001"
        with patch.dict("os.environ", {"TEMP": str(temporary), "TMP": str(temporary), "TMPDIR": str(temporary)}, clear=True):
            env = processes.clean_environment(artifact_dir=output)
        self.assertEqual(env["TEMP"], str(temporary))
        self.assertEqual(env["TMP"], str(temporary))
        self.assertEqual(env["TMPDIR"], str(temporary))
        for key in ("TENSORSHARP_PREFIX_CACHE_DIR", "TENSORSHARP_UPLOAD_DIR", "TENSORSHARP_LOG_DIR", "TS_STUDIO_DIRECTORY"):
            self.assertTrue(Path(env[key]).is_relative_to(output))


class StorageRunnerTests(unittest.TestCase):
    setUp = runner_fixture.RunnerTests.setUp
    invoke = runner_fixture.RunnerTests.invoke

    def test_model_existing_and_output_roots_stay_independent(self):
        model_volume = self.repo / "different-model-volume"
        existing_volume = self.repo / "existing-volume"
        report_volume = self.repo / "different-report-volume"
        existing_volume.mkdir()
        payload = b"GGUF-external-synthetic-weights"
        existing = existing_volume / "tiny.gguf"
        existing.write_bytes(payload)
        target = model_volume / "new-models" / "tiny.gguf"
        self.resolved = ResolvedConfig(self.resolved.source, {"model": str(target)}, [FileSpec("model", target)])
        self.config.return_value = self.resolved
        self.output = report_volume / "reports" / "new-run"
        code, report = self.invoke("--mode", "run", "--model-root", str(target.parent),
                                  "--existing-root", str(existing_volume))
        self.assertEqual(code, 0)
        self.assertEqual(self.hardware.call_args.args[0], target.parent.resolve())
        selected_files, policy = self.prepare.call_args.args
        self.assertEqual(selected_files[0].path, existing)
        self.assertEqual(policy.cache_root, target.parent.resolve())
        context = self.run.call_args.args[1]
        self.assertTrue(Path(context["artifact_dir"]).is_relative_to(self.output))
        self.assertEqual(context["model_path"], str(existing))
        self.assertEqual(report["policy"]["model_root"], str(target.parent.resolve()))
        self.assertTrue((self.output / "report.json").is_file())
        self.assertFalse(target.exists())
        self.assertEqual(existing.read_bytes(), payload)
        self.assertIn(str(existing_volume), report["results"][-1]["reproduce"])

    def test_low_model_volume_headroom_is_a_clear_skip_without_execution(self):
        model_root = self.repo / "selected-model-volume"
        report_root = self.repo / "spacious-report-volume"
        self.output = report_root / "new-run"
        self.hardware.return_value["disk"] = {"path": str(model_root), "free_bytes": 4 * GIB}
        with patch("regression.runner.sizes_for", return_value=(GIB, GIB)):
            code, report = self.invoke("--mode", "run", "--model-root", str(model_root),
                                      "--allow-download", "--disk-headroom-gib", "5")
        self.assertEqual(code, 3)
        result = report["results"][-1]
        self.assertEqual(result["status"], "skip")
        self.assertIn("disk budget after headroom", result["reason"])
        self.assertEqual(result["resource_plan"]["estimates"]["disk_headroom_bytes"], 5 * GIB)
        self.prepare.assert_not_called()
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
