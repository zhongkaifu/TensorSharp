import importlib.util
import json
import pathlib
import tempfile
import unittest


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-runs.py"
SPEC = importlib.util.spec_from_file_location("adaptive_compare", SOURCE)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class AdaptiveComparisonTests(unittest.TestCase):
    def run_comparison(self, change=None):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for mode in ("resident", "adaptive"):
                folder = pathlib.Path(temporary) / mode
                folder.mkdir()
                (folder / "prompt.json").write_text("[1,2,3]", encoding="utf-8")
                report = {
                    "Mode": mode, "Model": "same.gguf", "ModelBytes": 1234,
                    "ModelSha256": "checkpoint", "Context": 128,
                    "ModelsAssemblySha256": "assembly", "Native": [{"Sha256": "native"}],
                    "Executed": True, "Error": None,
                    "AfterDispose": [{"Reserved": 0, "Committed": 0}],
                    "Records": [{"Warmup": False, "LogitsSha256": "all-logits",
                                 "PromptTokens": 3, "DecodeCalls": 15,
                                 "PrefillTokensPerSecond": 200, "DecodeTokensPerSecond": 50}],
                }
                if mode == "adaptive" and change:
                    change(report)
                path = folder / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(path)
            return COMPARE.compare(paths)

    def test_identical_completed_runs_report_equal_medians(self):
        result = self.run_comparison()
        self.assertTrue(result["CorrectnessAndLifecyclePassed"])
        self.assertEqual(1, result["AdaptiveToResidentRatio"]["DecodeTokensPerSecond"])

    def test_replaced_checkpoint_at_same_path_is_not_parity(self):
        result = self.run_comparison(lambda r: r.update(ModelSha256="replaced"))
        self.assertFalse(result["CorrectnessAndLifecyclePassed"])

    def test_exit_success_does_not_hide_retained_or_unobserved_owners(self):
        for state in (None, [], [{"Reserved": 0, "Committed": 8}]):
            with self.subTest(state=state):
                result = self.run_comparison(lambda r: r.update(AfterDispose=state))
                self.assertFalse(result["CorrectnessAndLifecyclePassed"])

    def test_invalid_throughput_cannot_become_a_speedup(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_comparison(lambda r: r["Records"][0].update(DecodeTokensPerSecond=value))


if __name__ == "__main__":
    unittest.main()
