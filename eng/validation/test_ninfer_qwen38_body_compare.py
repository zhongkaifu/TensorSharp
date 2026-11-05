import array
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("body_compare", Path(__file__).with_name("ninfer-qwen38-body-compare.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BodyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.make_run(self.reference, "independent/reference")
        self.make_run(self.candidate, "TensorSharp/native-cuda")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def make_run(path, engine, values=(0.0, 2.0, 1.0, -1.0)):
        path.mkdir()
        inputs = json.dumps({"schema": "ninfer-qwen38-body-inputs/1", "cases": [
            {"id": "case", "promptTokens": [0, 1, 2], "continuationTokens": [1, 0], "chunkSize": 2}
        ]}).encode()
        (path / "inputs.json").write_bytes(inputs)
        captures = []
        for kind, position in (("prefill", 2), ("prefill", 3), ("decode", 4), ("decode", 5)):
            file = f"{kind}.{position}.f32"
            raw = struct.pack("<4f", *values)
            (path / file).write_bytes(raw)
            captures.append({"caseId": "case", "pass": 0, "kind": kind, "position": position,
                             "file": file, "elements": 4, "sha256": hashlib.sha256(raw).hexdigest(), "argmax": 1})
        run = {"schema": MODULE.SCHEMA, "complete": True, "engine": engine, "vocabSize": 4,
               "validVocabSize": 3, "inputsSha256": hashlib.sha256(inputs).hexdigest(),
               "artifactDirectorySha256": "a" * 64, "frontendResourceHashes": {"tokenizer": "b" * 64},
               "captures": captures}
        (path / "run.json").write_text(json.dumps(run))

    def change_run(self, fn):
        file = self.candidate / "run.json"
        data = json.loads(file.read_text())
        fn(data)
        file.write_text(json.dumps(data))

    def test_identical_logits_pass_only_explicit_numeric_gate(self):
        result = MODULE.compare(self.reference, self.candidate, require_argmax=True, max_relative_l2=0)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual((result["positions"], result["argmaxMatches"]), (4, 4))
        self.assertEqual(result["modelQuality"], "NOT_RUN")
        self.assertTrue(all(r["fullIncludingPaddingBitsEqual"] for r in result["rows"]))

    def test_no_threshold_does_not_certify_a_gate(self):
        result = MODULE.compare(self.reference, self.candidate)
        self.assertFalse(result["gate"]["configured"])
        self.assertIsNone(result["gate"]["passed"])

    def test_relative_l2_threshold_failure_retains_all_rows(self):
        for file in self.candidate.glob("*.f32"):
            file.write_bytes(struct.pack("<4f", 0, 3, 1, -1))
        self.change_run(lambda data: [cap.update(sha256=hashlib.sha256((self.candidate / cap["file"]).read_bytes()).hexdigest()) for cap in data["captures"]])
        result = MODULE.compare(self.reference, self.candidate, max_relative_l2=0.01)
        self.assertFalse(result["gate"]["passed"])
        self.assertEqual(result["positions"], 4)

    def test_missing_capture_is_refused(self):
        self.change_run(lambda data: data["captures"].pop())
        with self.assertRaisesRegex(ValueError, "Missing"):
            MODULE.compare(self.reference, self.candidate)

    def test_corrupt_float_file_is_refused(self):
        file = next(self.candidate.glob("*.f32"))
        file.write_bytes(file.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "size/SHA"):
            MODULE.compare(self.reference, self.candidate)

    def test_nan_in_valid_vocab_is_refused(self):
        a = array.array("f", [0, 1, float("nan"), 0])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            MODULE.metrics(a, a, 3, 2)

    def test_padding_is_preserved_but_excluded_from_argmax(self):
        a = array.array("f", [0, 2, 1, float("inf")])
        b = array.array("f", [0, 2, 1, float("-inf")])
        result = MODULE.metrics(a, b, 3, 2)
        self.assertTrue(result["argmaxEqual"])
        self.assertTrue(result["validFloatBitsEqual"])
        self.assertFalse(result["fullIncludingPaddingBitsEqual"])
        self.assertTrue(result["referencePaddedRowWins"])
        self.assertFalse(result["candidatePaddedRowWins"])
        self.assertEqual(result["referenceHeadArgmax"], 3)
        self.assertEqual(result["referenceNonfinitePadding"], 1)

    def test_same_engine_label_is_not_an_independent_oracle(self):
        self.change_run(lambda data: data.update(engine="independent/reference"))
        with self.assertRaisesRegex(ValueError, "Independent"):
            MODULE.compare(self.reference, self.candidate)

    def test_changed_input_manifest_is_refused(self):
        (self.candidate / "inputs.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Input manifest SHA"):
            MODULE.compare(self.reference, self.candidate)

    def test_duplicate_capture_is_refused(self):
        self.change_run(lambda data: data["captures"].append(data["captures"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.compare(self.reference, self.candidate)

    def test_incomplete_producer_is_refused(self):
        self.change_run(lambda data: data.update(complete=False))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            MODULE.compare(self.reference, self.candidate)

    def test_untrue_argmax_metadata_is_refused(self):
        self.change_run(lambda data: data["captures"][0].update(argmax=2))
        with self.assertRaisesRegex(ValueError, "argmax metadata"):
            MODULE.compare(self.reference, self.candidate)

    def test_zero_vector_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Zero-norm"):
            MODULE.metrics(array.array("f", [0, 0]), array.array("f", [0, 0]), 2, 2)


if __name__ == "__main__":
    unittest.main()
