import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_adaptive_parallel_comparison as fixtures

spec = importlib.util.spec_from_file_location("prefill_tiles",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-prefill-tiles.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PrefillTileEvidence(unittest.TestCase):
    def fixture(self, root):
        seeds = fixtures.ParallelPerformanceEvidence().fixture(root)
        seed = seeds["parallel"][0]
        report = json.loads(seed.with_name("report.json").read_text())
        execution = json.loads(seed.read_text())
        paths = []
        for i, tile in enumerate((32, 64, 128, 128, 64, 32)):
            folder = root / f"tile-{i}"; folder.mkdir()
            r, e = copy.deepcopy(report), copy.deepcopy(execution)
            r.update(ProcessId=1000+i, StartedUtc=f"2026-01-01T00:0{i}:01Z", FinishedUtc=f"2026-01-01T00:0{i}:10Z")
            r["Environment"][module.FLAG] = str(tile)
            r["RequestedOptions"]["--output"] = str(folder)
            e.update(pid=1000+i, started_utc=f"2026-01-01T00:0{i}:00Z", finished_utc=f"2026-01-01T00:0{i}:11Z")
            e["command"][-1] = str(folder)
            (folder / "report.json").write_text(json.dumps(r))
            (folder / "execution.json").write_text(json.dumps(e))
            (folder / "process.log").write_text(module.evidence.MARKER + ".\n" +
                (f"[q8-f32] Experimental K-ordered prefill column tile selected: {tile}.\n" if tile != 32 else ""))
            paths.append(folder / "execution.json")
        return paths

    def test_balanced_complete_logits_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = module.compare(self.fixture(Path(directory)))
            self.assertTrue(result["ComparableAndBitwiseEqual"], result)
            self.assertEqual(6, result["Measurements"]["128"]["PrefillTokensPerSecond"]["Samples"])

    def test_changed_logits_settings_exit_order_or_selection_refuse(self):
        for mutation in ("logits", "settings", "exit", "order", "selection", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                paths = self.fixture(Path(directory)); path = paths[2]
                r = json.loads(path.with_name("report.json").read_text())
                e = json.loads(path.read_text())
                if mutation == "logits": r["Records"][2]["LogitsSha256"] = "0"*64
                if mutation == "settings": r["Environment"]["CUDA_VISIBLE_DEVICES"] = "2"
                if mutation == "exit": e["exit_code"] = 9
                if mutation == "order": r["Environment"][module.FLAG] = "64"
                if mutation == "selection": path.with_name("process.log").write_text(module.evidence.MARKER)
                if mutation == "duplicate": paths[5] = paths[0]
                path.with_name("report.json").write_text(json.dumps(r)); path.write_text(json.dumps(e))
                self.assertFalse(module.compare(paths)["ComparableAndBitwiseEqual"])

    def auto_fixture(self, root, serial=False):
        paths = self.fixture(root)[:4]
        for path, tile in zip(paths, ("32", None, None, "32")):
            r = json.loads(path.with_name("report.json").read_text())
            r["Environment"][module.FLAG] = tile
            if serial: r["Environment"][module.evidence.FLAG] = "0"
            path.with_name("report.json").write_text(json.dumps(r))
            path.with_name("process.log").write_text(("" if serial else module.evidence.MARKER + ".\n") +
                ("[q8-f32] Automatic K-ordered prefill tiling: device=0 SMs=48 active64=3 active128=2; no global scratch.\n" if tile is None else ""))
        return paths

    def test_automatic_default_and_serial_vector_pass(self):
        for serial in (False, True):
            with self.subTest(serial=serial), tempfile.TemporaryDirectory() as directory:
                paths = self.auto_fixture(Path(directory), serial)
                result = module.compare(paths, automatic=True, vector_arm="serial" if serial else "parallel")
                self.assertTrue(result["ComparableAndBitwiseEqual"], result)
                self.assertEqual(6, result["Measurements"]["auto"]["PrefillTokensPerSecond"]["Samples"])

    def test_automatic_refuses_explicit_flag_or_missing_native_selection(self):
        for mutation in ("explicit", "missing", "wrong-arm"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                paths = self.auto_fixture(Path(directory)); path = paths[1]
                r = json.loads(path.with_name("report.json").read_text())
                if mutation == "explicit": r["Environment"][module.FLAG] = "auto"
                if mutation == "wrong-arm": r["Environment"][module.FLAG] = "32"
                if mutation == "missing": path.with_name("process.log").write_text(module.evidence.MARKER)
                path.with_name("report.json").write_text(json.dumps(r))
                self.assertFalse(module.compare(paths, automatic=True)["ComparableAndBitwiseEqual"])


if __name__ == "__main__": unittest.main()
