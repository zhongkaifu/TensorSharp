import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "qwen38_llama_teacher", Path(__file__).resolve().parents[1] / "qwen38-llama-teacher.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def completion(values):
    return {"completion_probabilities": [{"top_logprobs": [
        {"id": index, "logprob": value} for index, value in enumerate(values)]}]}


class CompleteVocabularyEvidence(unittest.TestCase):
    def test_normalization_offset_is_removed_without_hiding_pairwise_error(self):
        same = module.compare([3., 1., -2.], completion([-7., -9., -12.]))
        self.assertEqual(0., same["relative_l2_after_common_token_offset"])
        changed = module.compare([3., 1., -2.], completion([-7., -8.5, -12.]))
        self.assertEqual(.5, changed["max_abs_pairwise_logit_error"])
        self.assertGreater(changed["relative_l2_after_common_token_offset"], 0.)

    def test_reference_argmax_is_reported_even_if_it_differs(self):
        result = module.compare([4., 1., 0.], completion([-3., -1., -4.]))
        self.assertEqual([0, 1], result["argmax"])
        self.assertEqual(1, result["reference_token"])

    def test_incomplete_or_duplicate_vocabulary_is_rejected(self):
        for candidate in (completion([-2.]), completion([-2., -3.])):
            if len(candidate["completion_probabilities"][0]["top_logprobs"]) == 2:
                candidate["completion_probabilities"][0]["top_logprobs"][1]["id"] = 0
            with self.assertRaises(ValueError):
                module.compare([2., 1.], candidate)

    def test_underflow_and_nonfinite_reference_are_not_numerical_passes(self):
        for value in (-3.402823466e38, math.inf, math.nan):
            with self.assertRaises(ValueError):
                module.compare([2., 1.], completion([-1., value]))

    def test_capture_hash_and_range_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.f32"
            data = struct.pack("<ff", 1.25, -2.5)
            path.write_bytes(b"skip" + data)
            row = {"elements": 2, "byte_offset": 4, "sha256": hashlib.sha256(data).hexdigest()}
            self.assertEqual([1.25, -2.5], list(module.read_row(path, row)))
            for invalid in ({**row, "byte_offset": 8}, {**row, "sha256": "0" * 64},
                            {**row, "elements": True}, {**row, "byte_offset": 1}):
                with self.assertRaises(ValueError):
                    module.read_row(path, invalid)

    def test_nonfinite_capture_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.f32"
            data = struct.pack("<f", math.nan)
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                module.read_row(path, {"elements": 1, "byte_offset": 0,
                                      "sha256": hashlib.sha256(data).hexdigest()})


if __name__ == "__main__":
    unittest.main()
