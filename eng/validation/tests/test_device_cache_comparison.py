import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_host_cache_comparison as fixtures

spec = importlib.util.spec_from_file_location("device_cache_comparison",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-device-cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeviceCacheEvidence(unittest.TestCase):
    def fixture(self, root):
        paths = fixtures.HostCacheEvidence().fixture(root)
        for path, enabled in zip(paths, (False, True, True, False)):
            r = json.loads(path.with_name("report.json").read_text())
            r["RequestedOptions"].update({"--host-cache-bytes": "0", "--device-cache-bytes": "128" if enabled else "0",
                                           "--device-bytes": "256"})
            for index, row in enumerate(r["Records"], 1):
                row["Streaming"].update(HostCacheBytes=0, PeakHostCacheBytes=0, HostCacheHitBytes=0, HostCacheHits=0,
                    DeviceCacheBytes=128 if enabled else 0, PeakDeviceCacheBytes=128 if enabled else 0,
                    DeviceCacheHits=index if enabled else 0, DeviceCacheHitBytes=index * 200 if enabled else 0,
                    WeightUploadBytes=index * (800 if enabled else 1000), PeakDeviceOwnedBytes=200)
            path.with_name("report.json").write_text(json.dumps(r))
        return paths

    def test_complete_logits_and_actual_upload_reduction_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            result = module.compare(self.fixture(Path(folder)))
            self.assertTrue(result["ComparableAndBitwiseEqual"], result)
            self.assertEqual(.8, result["OnToOffRatio"]["WeightUploadBytesPerRequest"])

    def test_false_device_evidence_refuses(self):
        for mutation in ("no-hits", "no-uploads", "negative-uploads", "wrong-consumption", "owner-overrun",
                         "changed-host-cache", "resident", "leak", "logits", "timeout"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                paths = self.fixture(Path(folder)); path = paths[1].with_name("report.json")
                r = json.loads(path.read_text())
                if mutation == "no-hits": r["Records"][1]["Streaming"]["DeviceCacheHitBytes"] = 0
                if mutation == "no-uploads":
                    for i, row in enumerate(r["Records"], 1): row["Streaming"]["WeightUploadBytes"] = i * 1000
                if mutation == "negative-uploads": r["Records"][1]["Streaming"]["WeightUploadBytes"] = -1
                if mutation == "wrong-consumption": r["Records"][1]["Streaming"]["DeviceCacheHitBytes"] += 1
                if mutation == "owner-overrun": r["Records"][1]["Streaming"]["PeakDeviceOwnedBytes"] = 257
                if mutation == "changed-host-cache": r["RequestedOptions"]["--host-cache-bytes"] = "32"
                if mutation == "resident": r["Plan"]["SelectedCandidate"]["Placement"] = 0
                if mutation == "leak": r["AfterDispose"][0]["Committed"] = 1
                if mutation == "logits": r["Records"][1]["LogitsSha256"] = "b" * 64
                if mutation == "timeout":
                    execution = json.loads(paths[1].read_text()); execution["timed_out"] = True
                    paths[1].write_text(json.dumps(execution))
                path.write_text(json.dumps(r))
                self.assertFalse(module.compare(paths)["ComparableAndBitwiseEqual"])


if __name__ == "__main__": unittest.main()
