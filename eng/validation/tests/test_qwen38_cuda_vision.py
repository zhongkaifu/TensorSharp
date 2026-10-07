#!/usr/bin/env python3
"""Evidence rejection checks; these do not execute a model or CUDA device."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("qwen38_cuda_vision", Path(__file__).resolve().parents[1] / "qwen38-cuda-vision.py")
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def turn(answer="TensorSharp", reason="eos", thinking=""):
    body = "Assistant: " + (f"[thinking] {thinking}\n[answer] " if thinking else "") + answer
    return body + ("\n[turn complete: tokens=15 prefillMs=10 decodeMs=20 tps=750.0 "
                   f"ttftMs=11 reason={reason} kvPlan=Prefill]\n")


class EvidenceChecks(unittest.TestCase):
    def interactive(self, text):
        return HARNESS.interactive_result(text, "TensorSharp")

    def ocr(self, records, budget=256):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records))
            return HARNESS.ocr_result(path, "TensorSharp", budget)

    def test_complete_interactive_turn_and_retained_followup(self):
        result = self.interactive(turn("A TensorSharp banner.", thinking="Read the image.") + turn())
        self.assertTrue(result["passed"])
        self.assertEqual(result["turns"][0]["answer"], "A TensorSharp banner.")
        self.assertEqual(result["turns"][0]["prefill_ms"], 10)

    def test_one_or_three_turns_are_incomplete_evidence(self):
        self.assertFalse(self.interactive(turn())["passed"])
        self.assertFalse(self.interactive(turn() * 3)["passed"])

    def test_error_followed_by_success_is_not_a_pass(self):
        text = "05:24:47 fail: TensorSharp.Cli[0] Step failed for sequence\n[error] Cannot grow TensorSharp CUDA matmul scratch: out of memory\n"
        self.assertFalse(self.interactive(text + turn() * 2)["passed"])

    def test_budget_exhaustion_or_repetition_is_not_completion(self):
        for reason in ("max_tokens", "repetition", "cancelled", "error", "aborted"):
            with self.subTest(reason=reason):
                self.assertFalse(self.interactive(turn() + turn(reason=reason))["passed"])

    def test_reasoning_or_startup_does_not_count_as_an_answer(self):
        startup = "Loaded TensorSharp model from /root/TensorSharp/\n"
        self.assertFalse(self.interactive(startup + turn("Unknown") * 2)["passed"])
        self.assertFalse(self.interactive(turn("Unknown", thinking="TensorSharp") + turn())["passed"])
        self.assertFalse(self.interactive(turn("[thinking] TensorSharp") + turn())["passed"])

    def test_log_namespace_cannot_satisfy_expected_answer(self):
        self.assertFalse(self.interactive(turn("05:24:47 info: TensorSharp.Cli[0] inference running\nA generic picture") + turn())["passed"])

    def test_missing_assistant_marker_is_not_an_answer(self):
        self.assertFalse(self.interactive(turn().replace("Assistant: ", "") + turn())["passed"])

    def test_greedy_ocr_pass(self):
        result = self.ocr([{"id": "banner-ocr", "output": "TensorSharp", "tokens_generated": 4}])
        self.assertTrue(result["passed"])

    def test_ocr_incomplete_or_error_is_not_a_pass(self):
        for change in ({"error": "OOM"}, {"tokens_generated": 256}, {"tokens_generated": 0},
                       {"output": "<think>TensorSharp"}, {"output": "TensorSharp<|im_end|>"},
                       {"output": "It does not say TensorSharp"}, {"output": None},
                       {"tokens_generated": True},
                       {"output": "<think>TensorSharp</think>Unknown"}):
            row = {"id": "banner-ocr", "output": "TensorSharp", "tokens_generated": 4, **change}
            with self.subTest(change=change):
                self.assertFalse(self.ocr([row])["passed"])

    def test_wrong_missing_duplicate_ocr_result(self):
        row = {"id": "banner-ocr", "output": "TensorSharp", "tokens_generated": 4}
        self.assertFalse(self.ocr([])["passed"])
        self.assertFalse(self.ocr([row, row])["passed"])
        self.assertFalse(self.ocr(["TensorSharp"])["passed"])
        self.assertFalse(self.ocr([{**row, "id": "another-request"}])["passed"])


if __name__ == "__main__":
    unittest.main()
