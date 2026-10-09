import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_adaptive_parallel_comparison as fixtures

spec = importlib.util.spec_from_file_location("host_cache_comparison",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-host-cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HostCacheEvidence(unittest.TestCase):
    def fixture(self, root):
        seeds = fixtures.ParallelPerformanceEvidence().fixture(root)
        paths = [seeds["serial"][0], *seeds["parallel"], seeds["serial"][1]]
        for path, enabled in zip(paths, (False, True, True, False)):
            r = json.loads(path.with_name("report.json").read_text())
            r.update(Mode="adaptive", Plan={"Accepted": True, "SelectedCandidate": {"Placement": 2}},
                     AfterDispose=[{"Reserved": 0, "Committed": 0}])
            r["Environment"][module.evidence.FLAG] = "0"
            r["RequestedOptions"]["--host-cache-bytes"] = "128" if enabled else "0"
            for index, row in enumerate(r["Records"], 1):
                row["LogitsSha256"] = "a" * 64
                row["Streaming"] = {"FileBytesRead": index * (800 if enabled else 1000), "LinearTiles": index,
                    "HostCacheBytes": 128 if enabled else 0, "PeakHostCacheBytes": 128 if enabled else 0,
                    "HostCacheHitBytes": index * 200 if enabled else 0, "HostCacheHits": index if enabled else 0}
                row["Before"] = row["After"] = {"Pools": [{"Capacity": 256, "Reserved": 0, "Committed": 128}]}
            path.with_name("report.json").write_text(json.dumps(r))
            path.with_name("process.log").write_text("")
        return paths

    def test_complete_logits_and_measured_read_reduction_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            result = module.compare(self.fixture(Path(folder)))
            self.assertTrue(result["ComparableAndBitwiseEqual"], result)
            self.assertEqual(.8, result["OnToOffRatio"]["FileBytesPerRequest"])
            self.assertEqual(6, result["Measurements"]["on"]["DecodeTokensPerSecond"]["Samples"])

    def test_missing_cache_coverage_changed_logits_or_budget_overrun_refuses(self):
        for mutation in ("resident", "no-hits", "logits", "overrun", "no-reduction", "leak"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                paths = self.fixture(Path(folder)); path = paths[1].with_name("report.json")
                r = json.loads(path.read_text())
                if mutation == "resident": r["Plan"]["SelectedCandidate"]["Placement"] = 0
                if mutation == "no-hits": r["Records"][1]["Streaming"]["HostCacheHitBytes"] = 0
                if mutation == "logits": r["Records"][1]["LogitsSha256"] = "b" * 64
                if mutation == "overrun": r["Records"][1]["After"]["Pools"][0]["Committed"] = 257
                if mutation == "leak": r["AfterDispose"][0]["Committed"] = 1
                if mutation == "no-reduction":
                    for i, row in enumerate(r["Records"], 1): row["Streaming"]["FileBytesRead"] = i * 1400
                path.write_text(json.dumps(r))
                self.assertFalse(module.compare(paths)["ComparableAndBitwiseEqual"])


if __name__ == "__main__": unittest.main()
