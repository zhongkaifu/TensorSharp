import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("qwen_layer_trace", Path(__file__).parents[1] / "compare-qwen4exp-layer-traces.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def serial(row, values, prefix="serial-2-0-"):
    points = " ".join(f"{i}:{value}.0@{value:016x}" for i, value in enumerate(values))
    return (f"diagnostic serial rebuild: id={prefix}{row}, row={row}, step=22, position=1501, token={100 + row}\n"
            f"[q4e-trace] span 0..2 T=1 pos=1501: {points}\n")


def arena(row, values):
    return "".join(f"[q4e-arena-trace] row={row} pos=1501 l={i // 2} half={i % 2} l2={value}.0 hash={value:016x} head=0 0\n"
                   for i, value in enumerate(values))


class LayerTraceTests(unittest.TestCase):
    def parse(self, text, **kwargs):
        return MOD.analyze(text, position=1501, layers=2, rows=2, **kwargs)

    def good(self):
        return serial(0, [1, 2, 3, 4]) + serial(1, [11, 12, 13, 14]) + arena(0, [1, 2, 3, 4]) + arena(1, [11, 12, 13, 14])

    def test_swapped_native_rows_localize_without_assuming_request_order(self):
        result = self.parse(serial(0, [1, 2, 3, 4]) + serial(1, [11, 12, 13, 14])
                            + arena(0, [11, 12, 13, 14]) + arena(1, [1, 2, 33, 44]))
        self.assertEqual("localized", result["status"])
        pair = result["selected_mapping"]["pairs"][0]
        self.assertEqual(1, pair["native_sorted_row"])
        self.assertEqual(2, pair["equal_prefix_count"])
        self.assertEqual(1, pair["first_difference"]["arena"]["layer"])
        self.assertEqual(0, pair["first_difference"]["arena"]["half"])

    def test_ambiguous_identical_rows_do_not_claim_a_mapping(self):
        result = self.parse(serial(0, [1, 2, 3, 4]) + serial(1, [1, 2, 3, 4])
                            + arena(0, [1, 2, 3, 4]) + arena(1, [1, 2, 3, 4]))
        self.assertEqual("ambiguous", result["status"])
        self.assertIsNone(result["selected_mapping"])

    def test_no_trace_is_incomplete(self):
        self.assertEqual("incomplete", self.parse("model loaded\n")["status"])

    def test_missing_half_is_not_a_complete_comparison(self):
        result = self.parse(self.good().replace(arena(1, [11, 12, 13, 14]), arena(1, [11, 12, 13])))
        self.assertEqual("incomplete", result["status"])
        self.assertFalse(result["trace_complete"])

    def test_duplicate_native_half_is_invalid(self):
        self.assertEqual("invalid", self.parse(self.good() + arena(0, [1]))["status"])

    def test_duplicate_serial_marker_is_invalid(self):
        self.assertEqual("invalid", self.parse(self.good() + serial(0, [1, 2, 3, 4]))["status"])

    def test_nonfinite_norm_is_invalid(self):
        result = self.parse(self.good().replace("l2=11.0", "l2=nan"))
        self.assertEqual("invalid", result["status"])
        self.assertTrue(result["errors"])

    def test_malformed_hash_is_invalid(self):
        self.assertEqual("invalid", self.parse(self.good().replace("hash=000000000000000b", "hash=xyz"))["status"])

    def test_wrong_position_does_not_borrow_other_history(self):
        self.assertEqual("incomplete", self.parse(self.good().replace("1501", "1500"))["status"])

    def test_unlabelled_solo_trace_is_not_attributed_to_last_request(self):
        extra = "[q4e-trace] span 0..2 T=1 pos=1501: 0:1.0@0000000000000001\n"
        self.assertEqual("matched", self.parse(self.good() + extra)["status"])

    def test_cohort_prefix_filters_other_repetitions(self):
        extra = serial(0, [31, 32, 33, 34], prefix="serial-2-1-")
        self.assertEqual("matched", self.parse(self.good() + extra, serial_id_prefix="serial-2-0-")["status"])

    def test_full_unique_equal_trace_is_matched_not_model_pass(self):
        result = self.parse(self.good())
        self.assertEqual("matched", result["status"])
        self.assertNotIn("passed", result)

    def test_incomplete_serial_span_is_invalid(self):
        self.assertEqual("invalid", self.parse(self.good().replace("span 0..2", "span 0..3", 1))["status"])

    def test_invalid_geometry_rejected(self):
        for kwargs in ({"position": -1}, {"position": 1, "layers": 0}, {"position": 1, "rows": 9}):
            with self.assertRaises(ValueError):
                MOD.analyze("", **kwargs)

    def test_output_is_restricted_to_generated_evidence_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual((root / "artifacts/result.json").resolve(), MOD.output_path(root / "artifacts/result.json", root))
            with self.assertRaises(ValueError):
                MOD.output_path(root / "eng/result.json", root)


if __name__ == "__main__":
    unittest.main()
