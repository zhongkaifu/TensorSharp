import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("parallel_perf",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-parallel-runs.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ParallelPerformanceEvidence(unittest.TestCase):
    def fixture(self, root):
        paths = {"serial": [], "parallel": []}
        for order, arm in enumerate(("serial", "parallel", "parallel", "serial")):
            folder = root / str(order); folder.mkdir()
            execution_path = folder / "execution.json"
            bin_path = root / arm / "AdaptiveMemoryProbe.dll"
            native_path = root / arm / "GgmlOps.dll"
            report = {"Executed": True, "Error": None, "NativeShutdown": True, "ProcessId": 100 + order,
                "StartedUtc": f"2026-01-01T00:0{order}:01.1234567Z", "FinishedUtc": f"2026-01-01T00:0{order}:10.1234567Z",
                "CaptureLogits": False, "LogitCapture": None,
                "RequestedOptions": {"--output": str(folder), "--steps": "2", "--repeats": "3"},
                "Environment": {module.FLAG: "0" if arm == "serial" else "1", "CUDA_VISIBLE_DEVICES": "0"},
                "Native": [{"FileName": str(native_path), "Sha256": "a" * 64}],
                "ModelSha256": "b" * 64, "ModelBytes": 123, "ModelsAssemblySha256": "c" * 64,
                "ProbeAssemblySha256": "d" * 64, "ProbeAssemblyPath": str(bin_path),
                "ManagedAssembliesSha256": {"TensorSharp.Models": "c" * 64},
                "ModelGeometry": {"Architecture": "test", "HiddenSize": 4, "NumLayers": 2,
                    "NumHeads": 2, "NumKVHeads": 1, "Vocabulary": 3, "Context": 64},
                "Steps": 2, "Repeats": 3, "Prompt": [0, 1], "Mode": "resident",
                "Generation": "raw-greedy", "Teacher": None,
                "Records": [{"Run": run, "Warmup": run < 0, "PromptTokens": 2, "DecodeCalls": 1,
                    "Generated": [2, 1], "Consumed": [2], "LogitsSha256": ("e" if arm == "serial" else "f") * 64,
                    "PrefillMilliseconds": 2., "PrefillTokensPerSecond": 1000.,
                    "DecodeMilliseconds": 1. if arm == "serial" else .5,
                    "DecodeTokensPerSecond": 1000. if arm == "serial" else 2000.} for run in (-1, 0, 1, 2)]}
            execution = {"command": ["dotnet", str(bin_path), "--output", str(folder)],
                "complete": True, "timed_out": False, "exit_code": 0, "pid": 100 + order,
                "started_utc": f"2026-01-01T00:0{order}:00Z", "finished_utc": f"2026-01-01T00:0{order}:11Z"}
            execution_path.write_text(json.dumps(execution))
            (folder / "report.json").write_text(json.dumps(report))
            (folder / "process.log").write_text(module.MARKER + "\n" if arm == "parallel" else "serial run\n")
            paths[arm].append(execution_path)
        return paths

    @staticmethod
    def mutate(path, update):
        data = json.loads(path.read_text()); update(data); path.write_text(json.dumps(data))

    def test_complete_abba_same_tokens_accepts_different_raw_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            result = module.compare(**paths)
            self.assertTrue(result["ComparableForTiming"], result)
            self.assertEqual(6, result["Measurements"]["serial"]["DecodeTokensPerSecond"]["Samples"])
            self.assertEqual(2, result["ParallelToSerialRatio"]["DecodeTokensPerSecond"])
            self.assertNotEqual(result["RawLogitHashes"]["serial"], result["RawLogitHashes"]["parallel"])

    def test_nonzero_exit_timeout_or_pid_mismatch_refuses_timing(self):
        for changed in ({"exit_code": 1}, {"complete": False}, {"timed_out": True}, {"pid": 999}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                self.mutate(paths["serial"][0], lambda data: data.update(changed))
                self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_actual_flag_selection_log_is_required_and_once_only(self):
        for text in ("no selection", module.MARKER + module.MARKER):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                (paths["parallel"][0].parent / "process.log").write_text(text)
                self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_small_batch_flag_cannot_hide_in_vector_only_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            # Even enabling it in every arm is outside this comparator's
            # vector-only qualification and must not be accepted as equal.
            for executions in paths.values():
                for path in executions:
                    self.mutate(path.parent / "report.json",
                        lambda data: data["Environment"].update(TS_GGML_Q8_PARALLEL_SMALL_BATCH="1"))
            self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_capture_or_changed_runtime_knob_or_same_bin_refuses_timing(self):
        for mode in ("capture", "env", "binary"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                target = paths["parallel"][0].parent / "report.json"
                def update(data):
                    if mode == "capture": data["CaptureLogits"] = True
                    if mode == "env": data["Environment"]["CUDA_VISIBLE_DEVICES"] = "1"
                    if mode == "binary": data["Native"][0]["FileName"] = str(Path(temporary) / "serial" / "GgmlOps.dll")
                self.mutate(target, update)
                self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_missing_requests_bad_denominator_and_argmax_changes_fail(self):
        for mode in ("requests", "denominator", "argmax", "managed", "shutdown"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                target = paths["parallel"][0].parent / "report.json"
                def update(data):
                    if mode == "requests": data["Records"].pop()
                    if mode == "denominator": data["Records"][1]["DecodeTokensPerSecond"] = 5000.
                    if mode == "argmax": data["Records"][1]["Generated"][-1] = 0
                    if mode == "managed": data["ProbeAssemblySha256"] = "0" * 64
                    if mode == "shutdown": data["NativeShutdown"] = False
                self.mutate(target, update)
                self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_equal_reused_evidence_is_not_two_fresh_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.fixture(Path(temporary))
            paths["serial"][1] = paths["serial"][0]
            self.assertFalse(module.compare(**paths)["ComparableForTiming"])

    def test_non_abba_or_overlapping_processes_are_rejected(self):
        for mode in ("order", "overlap"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                paths = self.fixture(Path(temporary))
                if mode == "order":
                    self.mutate(paths["serial"][1], lambda data: data.update(started_utc="2026-01-01T00:00:30Z"))
                else:
                    self.mutate(paths["serial"][0], lambda data: data.update(finished_utc="2026-01-01T00:01:05Z"))
                self.assertFalse(module.compare(**paths)["ComparableForTiming"])


if __name__ == "__main__":
    unittest.main()
