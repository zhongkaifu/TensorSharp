import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_device_cache_comparison as fixtures

spec = importlib.util.spec_from_file_location("workspace_cache_comparison",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-workspace-cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkspaceCacheEvidence(unittest.TestCase):
    def fixture(self, root):
        paths = fixtures.DeviceCacheEvidence().fixture(root)
        for path, enabled in zip(paths, (False, True, True, False)):
            r = json.loads(path.with_name("report.json").read_text())
            r["RequestedOptions"].update({"--device-cache-bytes": "0", "--workspace-cache-bytes": "128" if enabled else "0"})
            for index, row in enumerate(r["Records"], 1):
                row["Streaming"].update(FileBytesRead=index * 1000, DeviceCacheBytes=0, PeakDeviceCacheBytes=0,
                    DeviceCacheHits=0, DeviceCacheHitBytes=0, WeightUploadBytes=index * 1000,
                    DeviceWorkspaceCacheBytes=128 if enabled else 0, PeakDeviceWorkspaceCacheBytes=128 if enabled else 0,
                    DeviceWorkspaceReuses=index * 8 if enabled else 0, DeviceSessionCreations=index * (2 if enabled else 10))
            path.with_name("report.json").write_text(json.dumps(r))
        return paths

    def test_complete_logits_and_actual_allocation_reduction_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            result = module.compare(self.fixture(Path(folder)))
            self.assertTrue(result["ComparableAndBitwiseEqual"], result)
            self.assertEqual(.2, result["OnToOffRatio"]["DeviceSessionCreationsPerRequest"])
            self.assertEqual(1, result["OnToOffRatio"]["WeightUploadBytesPerRequest"])

    def test_false_workspace_evidence_refuses(self):
        for mutation in ("no-reuse", "negative-creations", "different-operations", "weight-cache-competition", "owner-overrun",
                         "cache-overrun", "changed-weight-ceiling", "resident", "leak", "logits", "timeout"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                paths = self.fixture(Path(folder)); path = paths[1].with_name("report.json")
                r = json.loads(path.read_text())
                usage = r["Records"][1]["Streaming"]
                if mutation == "no-reuse": usage["DeviceWorkspaceReuses"] = 0
                if mutation == "negative-creations": usage["DeviceSessionCreations"] = -1
                if mutation == "different-operations": usage["DeviceSessionCreations"] += 1
                if mutation == "weight-cache-competition":
                    for index, row in enumerate(r["Records"], 1):
                        row["Streaming"].update(FileBytesRead=index * 900, DeviceCacheHitBytes=index * 100, WeightUploadBytes=index * 900)
                if mutation == "owner-overrun": usage["PeakDeviceOwnedBytes"] = 257
                if mutation == "cache-overrun": usage["PeakDeviceWorkspaceCacheBytes"] = 129
                if mutation == "changed-weight-ceiling": r["RequestedOptions"]["--device-cache-bytes"] = "32"
                if mutation == "resident": r["Plan"]["SelectedCandidate"]["Placement"] = 0
                if mutation == "leak": r["AfterDispose"][0]["Committed"] = 1
                if mutation == "logits": r["Records"][1]["LogitsSha256"] = "b" * 64
                if mutation == "timeout":
                    execution = json.loads(paths[1].read_text()); execution["timed_out"] = True
                    paths[1].write_text(json.dumps(execution))
                path.write_text(json.dumps(r))
                self.assertFalse(module.compare(paths)["ComparableAndBitwiseEqual"])

    def competition_fixture(self, root, mixed=False):
        paths = self.fixture(root)
        for path, enabled in zip(paths, (False, True, True, False)):
            r = json.loads(path.with_name("report.json").read_text())
            r["RequestedOptions"].pop("--device-cache-bytes")
            if enabled: r["RequestedOptions"].pop("--workspace-cache-bytes") # Actual default, not explicit infinity.
            if mixed:
                r["Prompts"] = [r["Prompt"], r["Prompt"] * 2]
                r["RequestedOptions"]["--alternate-prompt-tokens"] = "4"
            reads = uploads = hits = 0
            for index, row in enumerate(r["Records"]):
                prompt_index = 1 if mixed and index in (1, 3) else 0
                row.update(PromptIndex=prompt_index, PromptTokens=4 if prompt_index else 2,
                           PrefillTokensPerSecond=2000 if prompt_index else 1000, LogitsSha256=("b" if prompt_index else "a") * 64)
                consumed = 1200 if prompt_index else 1000
                reused = 100 if enabled else 200
                reads += consumed - reused; uploads += consumed - reused; hits += reused
                row["Streaming"].update(DeviceCacheBytes=64, PeakDeviceCacheBytes=64, DeviceCacheHitBytes=hits,
                                        FileBytesRead=reads, WeightUploadBytes=uploads)
            path.with_name("report.json").write_text(json.dumps(r))
        return paths

    def test_auto_competition_reports_extra_uploads_and_stratifies_mixed_prompts(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed), tempfile.TemporaryDirectory() as folder:
                paths = self.competition_fixture(Path(folder), mixed)
                result = module.compare(paths, competition=True)
                self.assertTrue(result["ComparableAndBitwiseEqual"], result)
                self.assertGreater(result["OnToOffRatio"]["WeightUploadBytesPerRequest"], 1)
                self.assertEqual({"2", "4"} if mixed else {"2"}, set(result["ByPromptTokens"]))
                self.assertFalse(module.compare(paths)["ComparableAndBitwiseEqual"])

    def test_competition_cannot_hide_changed_work_missing_weights_or_wrong_prompt(self):
        for mutation in ("conservation", "no-weights", "schedule", "prompt-content", "denominator", "logits"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                paths = self.competition_fixture(Path(folder), True)
                path = paths[1].with_name("report.json"); r = json.loads(path.read_text())
                if mutation == "conservation": r["Records"][1]["Streaming"]["FileBytesRead"] += 1
                if mutation == "no-weights": r["Records"][1]["Streaming"]["DeviceCacheBytes"] = 0
                if mutation == "schedule": r["Records"][1]["PromptIndex"] = 0
                if mutation == "prompt-content": r["Prompts"][1][0] = 2
                if mutation == "denominator": r["Records"][1]["PrefillTokensPerSecond"] = 1000
                if mutation == "logits": r["Records"][1]["LogitsSha256"] = "c" * 64
                path.write_text(json.dumps(r))
                self.assertFalse(module.compare(paths, competition=True)["ComparableAndBitwiseEqual"])

    def test_competition_accepts_reuse_followed_by_full_idle_reclamation(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = self.competition_fixture(Path(folder), True)
            for path in paths[1:3]:
                report_path = path.with_name("report.json"); r = json.loads(report_path.read_text())
                r["Records"][1]["Streaming"]["DeviceWorkspaceCacheBytes"] = 0
                report_path.write_text(json.dumps(r))
            self.assertTrue(module.compare(paths, competition=True)["ComparableAndBitwiseEqual"])


if __name__ == "__main__": unittest.main()
