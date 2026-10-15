import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("adaptive_captures",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "compare-captures.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class AdaptiveCaptureTests(unittest.TestCase):
    def write(self, folder, vectors=None, teacher=True):
        folder.mkdir()
        vectors = vectors or [[1., 2., 3.], [2., 3., 4.]]
        chunks = [struct.pack("<fff", *row) for row in vectors]
        data = b"".join(chunks)
        data_path, index_path, report_path = (folder / name for name in ("logits.f32", "logits.json", "report.json"))
        rows = [{"Run": 0, "Step": i, "Stage": "prefill" if i == 0 else "decode",
                 "InputTokens": [0, 1] + [2] * i, "Elements": 3, "ByteOffset": 12 * i,
                 "Sha256": hashlib.sha256(chunks[i]).hexdigest(),
                 "Argmax": max(range(3), key=vectors[i].__getitem__)} for i in range(2)]
        capture = {"Format": "f32le", "DataPath": str(data_path), "IndexPath": str(index_path),
                   "Bytes": len(data), "Sha256": hashlib.sha256(data).hexdigest(), "Rows": rows}
        record = {"Run": 0, "Warmup": False, "PromptTokens": 2, "DecodeCalls": 1,
                  "Generated": [row["Argmax"] for row in rows], "Consumed": [2],
                  "LogitsSha256": hashlib.sha256(data).hexdigest()}
        report = {"Executed": True, "Error": None, "NativeShutdown": True, "CaptureLogits": True,
            "ModelSha256": "a" * 64, "ModelBytes": 123, "ModelsAssemblySha256": "b" * 64,
            "ProbeAssemblySha256": "e" * 64, "ManagedAssembliesSha256": {"TensorSharp.Models": "b" * 64},
            "Native": [{"Sha256": "c" * 64}], "ModelGeometry": {"Architecture": "test", "HiddenSize": 4,
                "NumLayers": 2, "NumHeads": 2, "NumKVHeads": 1, "Vocabulary": 3, "Context": 64},
            "Steps": 2, "Repeats": 1, "Prompt": [0, 1], "Mode": "resident",
            "Generation": "teacher-forced" if teacher else "raw-greedy", "Teacher": [2, 1] if teacher else None,
            "TeacherSha256": "d" * 64 if teacher else None, "LogitCapture": capture,
            "Records": [{**record, "Run": -1, "Warmup": True}, record]}
        data_path.write_bytes(data)
        self.save(report_path, report)
        return report_path, report

    @staticmethod
    def save(path, report):
        capture = report["LogitCapture"]
        Path(capture["IndexPath"]).write_text(json.dumps({"Format": "f32le", "DataPath": capture["DataPath"],
                                                        "Rows": capture["Rows"]}))
        path.write_text(json.dumps(report))

    def test_identical_and_small_different_rows_pass_fixed_numerical_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, _ = self.write(directory / "left")
            right, _ = self.write(directory / "right", [[1., 2., 3.00001], [2., 3., 4.00001]])
            self.assertTrue(module.compare(left, left)["Passed"])
            result = module.compare(left, right)
            self.assertTrue(result["Passed"], result)
            self.assertFalse(result["Rows"][0]["BitwiseEqual"])
            self.assertGreater(result["Rows"][0]["MaximumAbsoluteError"], 0)

    def test_same_argmax_large_error_fails_without_claiming_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, _ = self.write(directory / "left")
            right, _ = self.write(directory / "right", [[1., 2., 3.1], [2., 3., 4.]])
            result = module.compare(left, right)
            self.assertFalse(result["Passed"])
            self.assertTrue(result["RunComplete"])

    def test_identically_truncated_and_failed_shutdown_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, report = self.write(Path(temporary) / "data")
            for field in ("Executed", "NativeShutdown"):
                changed = copy.deepcopy(report); changed[field] = False; self.save(path, changed)
                self.assertFalse(module.compare(path, path)["Passed"])
            report["LogitCapture"]["Rows"].pop()
            self.save(path, report)
            self.assertFalse(module.compare(path, path)["RunComplete"])

    def test_teacher_count_range_and_recorded_consumed_history_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, report = self.write(Path(temporary) / "data")
            for teacher in ([2], [2, 3], [True, 1], [1.2, 1]):
                changed = copy.deepcopy(report); changed["Teacher"] = teacher; self.save(path, changed)
                self.assertFalse(module.compare(path, path)["Passed"])
            report["Records"][1]["Consumed"] = [1]
            self.save(path, report)
            self.assertFalse(module.compare(path, path)["Passed"])

    def test_changed_source_history_or_checkpoint_is_not_comparable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left, _ = self.write(directory / "left")
            right, report = self.write(directory / "right")
            report["ModelSha256"] = "f" * 64; self.save(right, report)
            self.assertFalse(module.compare(left, right)["RunComplete"])
            report["ModelSha256"] = "a" * 64
            report["LogitCapture"]["Rows"][1]["InputTokens"] = [0, 1, 1]
            self.save(right, report)
            self.assertFalse(module.compare(left, right)["RunComplete"])

    def test_nonfinite_or_replaced_payload_and_full_run_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, report = self.write(directory / "data")
            report["Records"][1]["LogitsSha256"] = "0" * 64; self.save(path, report)
            self.assertFalse(module.compare(path, path)["RunComplete"])
            other, _ = self.write(directory / "nonfinite", [[1., 2., float("inf")], [2., 3., 4.]])
            self.assertFalse(module.compare(other, other)["RunComplete"])
            Path(report["LogitCapture"]["DataPath"]).write_bytes(b"\0" * 24)
            self.assertFalse(module.compare(path, path)["RunComplete"])

    def test_adaptive_live_credit_refuses_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, report = self.write(Path(temporary) / "data")
            report["Mode"] = "adaptive"
            report["AfterDispose"] = [{"Reserved": 0, "Committed": 0}]
            self.save(path, report)
            self.assertTrue(module.compare(path, path)["Passed"])
            report["AfterDispose"][0]["Committed"] = 1; self.save(path, report)
            self.assertFalse(module.compare(path, path)["RunComplete"])


if __name__ == "__main__":
    unittest.main()
