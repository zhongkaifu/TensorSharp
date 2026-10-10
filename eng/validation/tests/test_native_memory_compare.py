import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import test_adaptive_parallel_comparison as fixtures

SPEC = importlib.util.spec_from_file_location("native_compare",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-native-runs.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class NativeComparisonTests(unittest.TestCase):
    def execution_fixture(self, root):
        seeds = fixtures.ParallelPerformanceEvidence().fixture(root)
        paths = [seeds["serial"][0], *seeds["parallel"], seeds["serial"][1]]
        for index, path in enumerate(paths):
            report_path = path.with_name("report.json")
            report = json.loads(report_path.read_text())
            report.update(Model="same.gguf", Context=64)
            report["Environment"][COMPARE.evidence.FLAG] = "0"
            report["Native"][0]["Sha256"] = ("a" if index in (0, 3) else "b") * 64
            for row in report["Records"]: row["LogitsSha256"] = "e" * 64
            report_path.write_text(json.dumps(report))
            path.with_name("process.log").write_text("")
            path.with_name("prompt.json").write_text(json.dumps(report["Prompt"]))
        return paths

    def test_strict_executions_require_balanced_completed_processes(self):
        with tempfile.TemporaryDirectory() as root:
            result = COMPARE.compare_executions(self.execution_fixture(Path(root)))
            self.assertTrue(result["ComparableAndBitwiseEqual"], result)
            self.assertEqual(6, result["Measurements"]["candidate"]["DecodeTokensPerSecond"]["Samples"])

    def test_strict_executions_reject_false_or_changed_evidence(self):
        for change in ("exit", "capture", "shutdown", "warmup", "settings", "probe", "order", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                paths = self.execution_fixture(Path(root)); target = paths[1]
                report = json.loads(target.with_name("report.json").read_text())
                execution = json.loads(target.read_text())
                if change == "exit": execution["exit_code"] = 1
                if change == "capture": report["CaptureLogits"] = True
                if change == "shutdown": report["NativeShutdown"] = False
                if change == "warmup": report["Records"][0]["LogitsSha256"] = "0" * 64
                if change == "settings": report["Environment"]["CUDA_VISIBLE_DEVICES"] = "2"
                if change == "probe": report["ProbeAssemblySha256"] = "0" * 64
                if change == "order": report["Native"][0]["Sha256"] = "a" * 64
                if change == "duplicate": paths[3] = paths[0]
                target.with_name("report.json").write_text(json.dumps(report)); target.write_text(json.dumps(execution))
                self.assertFalse(COMPARE.compare_executions(paths)["ComparableAndBitwiseEqual"])

    def run_comparison(self, change=None):
        with tempfile.TemporaryDirectory() as root:
            paths = {"control": [], "candidate": []}
            for arm in paths:
                for index in range(2):
                    folder = Path(root) / f"{arm}-{index}"
                    folder.mkdir()
                    (folder / "prompt.json").write_text("[1,2,3]", encoding="utf-8")
                    record = {"Warmup": False, "LogitsSha256": "all-rows", "PromptTokens": 3,
                              "DecodeCalls": 15, "PrefillTokensPerSecond": 200,
                              "DecodeTokensPerSecond": 100 if arm == "candidate" else 50}
                    report = {"Mode": "resident", "Executed": True, "Error": None,
                              "Model": "same.gguf", "ModelSha256": "same-model", "ModelBytes": 1234,
                              "Context": 128, "ModelsAssemblySha256": "same-assembly",
                              "Native": [{"Sha256": arm}],
                              "Records": [dict(record, Warmup=True), record]}
                    if change and arm == "candidate":
                        change(report)
                    file = folder / "report.json"
                    file.write_text(json.dumps(report), encoding="utf-8")
                    paths[arm].append(file)
            return COMPARE.compare(paths["control"], paths["candidate"])

    def test_only_native_changes_and_complete_logits_match(self):
        result = self.run_comparison()
        self.assertTrue(result["ComparableAndBitwiseEqual"])
        self.assertEqual(2, result["CandidateToControlRatio"]["DecodeTokensPerSecond"])

    def test_different_checkpoint_or_managed_code_cannot_claim_kernel_speedup(self):
        for field in ("ModelSha256", "ModelsAssemblySha256"):
            with self.subTest(field=field):
                self.assertFalse(self.run_comparison(lambda r: r.update({field: "other"}))["ComparableAndBitwiseEqual"])

    def test_same_native_or_different_logits_are_rejected(self):
        for change in (lambda r: r.update(Native=[{"Sha256": "control"}]),
                       lambda r: r["Records"][1].update(LogitsSha256="different")):
            self.assertFalse(self.run_comparison(change)["ComparableAndBitwiseEqual"])

    def test_failed_or_nonfinite_run_is_not_performance_evidence(self):
        for change in (lambda r: r.update(Executed=False),
                       lambda r: r["Records"][1].update(DecodeTokensPerSecond=float("nan"))):
            self.assertFalse(self.run_comparison(change)["ComparableAndBitwiseEqual"])


if __name__ == "__main__":
    unittest.main()
