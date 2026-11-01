"""Tiny model-free byte and metadata oracles; no process/network/native code."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SOURCE = Path(__file__).with_name("moe-metal-overlap-quality.py")
if not SOURCE.is_file():
    SOURCE = Path(__file__).parents[1] / "moe-metal-overlap-quality.py"
SPEC = importlib.util.spec_from_file_location("moe_quality", SOURCE)
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def summary(**changes):
    fields = {"mode": 1, "considered": 20, "ineligible": 10, "eligible": 10, "async": 10,
              "serial": 0, "rejected": 0, "submit_failures": 0, "drain_failures": 0, "fast_fallbacks": 0}
    fields.update(changes)
    return "[HOSTMOE-METAL-OVERLAP] " + " ".join(f"{key}={value}" for key, value in fields.items())


class GenericQualityOracles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def vector(self, name, rows):
        path = self.root / name
        path.write_bytes(b"".join(struct.pack("<" + "f" * len(row), *row) for row in rows))
        return path

    def test_dynamic_vocab_and_exact_all_rows(self):
        for vocab in (3, 7):
            with self.subTest(vocab=vocab):
                rows = [list(range(vocab)), list(reversed(range(vocab)))]
                a, b = self.vector("a", rows), self.vector("b", rows)
                result = QUALITY.compare_vectors(a, b, 2, vocab, [vocab - 1, 0])
                self.assertTrue(result["bit_identical"])
                self.assertEqual(2 * vocab * 4, result["bytes_per_file"])
                rows[1][-1] += .000001
                result = QUALITY.compare_vectors(a, self.vector("changed", rows), 2, vocab, [vocab - 1, 0])
                self.assertFalse(result["bit_identical"])
                self.assertEqual([1], result["different_rows"])

    def test_identical_nonfinite_rows_cannot_pass(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                a, b = self.vector("a", [[1., value, 2.]]), self.vector("b", [[1., value, 2.]])
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    QUALITY.compare_vectors(a, b, 1, 3, [2])

    def test_false_top1_metadata_and_truncated_extra_or_wrong_width_files_fail(self):
        a = self.vector("a", [[1., 3., 2.], [4., 2., 1.]])
        with self.assertRaisesRegex(ValueError, "argmax"):
            QUALITY.compare_vectors(a, a, 2, 3, [0, 0])
        for b, vocab in ((self.vector("short", [[1., 3., 2.]]), 3),
                         (self.vector("extra", [[1., 3., 2.]] * 3), 3), (a, 7)):
            with self.subTest(file=b, vocab=vocab), self.assertRaisesRegex(ValueError, "size"):
                QUALITY.compare_vectors(a, b, 2, vocab)

    def test_vocab_requires_actual_integer_metadata(self):
        for value in (None, True, False, 3., "3", 0, 1, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                QUALITY.validate_vocab_size(value)
        self.assertEqual(7, QUALITY.validate_vocab_size(7))

    def test_active_layer_count_and_complete_engagement(self):
        stats = QUALITY.overlap_engagement(summary(), True, 2, "active", 10)
        self.assertEqual(10, stats["async"])
        with self.assertRaises(ValueError):
            QUALITY.overlap_engagement(summary(), True, 4, "active", 20)
        with self.assertRaises(ValueError):
            QUALITY.overlap_engagement(summary(eligible=0, **{"async": 0}), True, 2, "active", 10)

    def test_active_errors_rejection_or_fast_fallback_fail(self):
        for field in ("rejected", "submit_failures", "drain_failures", "fast_fallbacks"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                QUALITY.overlap_engagement(summary(**{field: 1}), True, 2, "active", 10)

    def test_dense_and_unsupported_layout_fallback_require_no_async(self):
        self.assertFalse(QUALITY.overlap_engagement("", True, 0, "fallback")["summary_reported"])
        with self.assertRaises(ValueError):
            QUALITY.overlap_engagement("", True, 1, "fallback")
        rejected = summary(considered=20, ineligible=5, rejected=15, eligible=0, **{"async": 0})
        log = rejected + "\n[HOSTMOE-METAL-OVERLAP] rejected_operation=15\n"
        self.assertEqual(0, QUALITY.overlap_engagement(log, True, 1, "fallback")["async"])
        with self.assertRaises(ValueError):
            QUALITY.overlap_engagement(summary(), True, 1, "fallback")

    def test_reset_summaries_sum_and_off_reference_is_quiet(self):
        part = summary(considered=10, ineligible=5, eligible=5, **{"async": 5})
        self.assertEqual(10, QUALITY.overlap_engagement(part + "\n" + part, True, 2, "active", 10)["async"])
        self.assertFalse(QUALITY.overlap_engagement("", False, 2, "active")["summary_reported"])
        with self.assertRaises(ValueError):
            QUALITY.overlap_engagement(summary(), False, 2, "active")

    def test_exact_switch_variable_and_plaintext_public_cases(self):
        env = QUALITY.environment(QUALITY.FEATURE_ENV, 1, 4)
        self.assertEqual("4", env["TS_N_CPU_MOE"])
        self.assertEqual("1", env["TS_HOST_MOE_METAL_OVERLAP"])
        with self.assertRaises(ValueError):
            QUALITY.environment("TS_Q4E_METAL_HOST_OVERLAP", 1, 4)
        cases = QUALITY.default_cases()
        self.assertEqual(QUALITY.EXPECTED_CASES, {case["name"] for case in cases})
        self.assertTrue(all("<|im_start|>" not in case["prompt"] for case in cases))

    def test_off_on_vocab_size_mismatch_is_rejected(self):
        cases = []
        for item in QUALITY.default_cases():
            cases.append({"name": item["name"], "steps": 2, "prompt_tokens": 8,
                          "metrics": [{"step": i, "finite": True, "top1_equal": True,
                                       "reference_top": 1, "candidate_top": 1} for i in range(2)]})
        document = {"complete": True, "all_finite": True, "cases_done": 5, "cases_total": 5,
                    "run_id": "test", "backend": "ggml_metal", "max_context": 8192,
                    "cpu_moe_threads": 12, "vocab_size": 7, "cpu_moe": "first 2 layer(s)",
                    "qualified": True, "cases": cases}
        (self.root / "metrics.json").write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "vocab_size differs"):
            QUALITY.read_metrics(self.root, "test", 2, True, 2, "active", expected_vocab=3)
        for value in (True, 1.0):
            with self.subTest(candidate_top=value):
                document["cases"][0]["metrics"][0]["candidate_top"] = value
                (self.root / "metrics.json").write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "Top1 differs"):
                    QUALITY.read_metrics(self.root, "test", 2, True, 2, "active", expected_vocab=7)


if __name__ == "__main__":
    unittest.main()
