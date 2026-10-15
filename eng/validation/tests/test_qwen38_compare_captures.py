import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "qwen38_compare", Path(__file__).resolve().parents[1] / "qwen38-compare-captures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MatchedRowEvidence(unittest.TestCase):
    def write(self, directory, name, values, history=None):
        data = struct.pack("<" + "f" * len(values), *values)
        blob = directory / (name + ".f32")
        blob.write_bytes(data)
        index = directory / (name + ".json")
        index.write_text(json.dumps({"format": "f32le", "data_path": str(blob), "rows": [{
            "elements": len(values), "byte_offset": 0, "sha256": hashlib.sha256(data).hexdigest(),
            "stage": "prefill", "iteration": 0, "warmup": False,
            "input_tokens": history if history is not None else [1, 2]}]}))
        return index

    def test_identical_rows_pass_and_small_same_argmax_error_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = self.write(directory, "a", [1., 2., 3.])
            same = self.write(directory, "same", [1., 2., 3.])
            changed = self.write(directory, "changed", [1., 2.01, 3.])
            self.assertTrue(module.compare(left, same, 0)[0]["passed"])
            row = module.compare(left, changed, 0)[0]
            self.assertEqual([2, 2], row["argmax"])
            self.assertFalse(row["passed"])

    def test_different_teacher_histories_are_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = self.write(directory, "a", [1., 2., 3.])
            right = self.write(directory, "b", [1., 2., 3.], [1, 0])
            with self.assertRaises(ValueError):
                module.compare(left, right, 0)

    def test_zero_reference_large_finite_mismatch_remains_serializable_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = self.write(directory, "zero", [0., 0., 0.])
            right = self.write(directory, "large", [1e38, 1e38, 1e38])
            row = module.compare(left, right, 0)[0]
            self.assertFalse(row["passed"])
            self.assertTrue(math.isfinite(row["relative_l2"]))
            json.dumps(row, allow_nan=False)

    def test_missing_iteration_and_partial_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = self.write(directory, "a", [1., 2., 3.])
            with self.assertRaises(ValueError):
                module.compare(left, left, 1)
            right = self.write(directory, "b", [1., 2., 3.])
            (directory / "b.f32").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                module.compare(left, right, 0)


if __name__ == "__main__":
    unittest.main()
