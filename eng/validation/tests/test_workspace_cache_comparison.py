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


if __name__ == "__main__": unittest.main()
