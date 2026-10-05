#!/usr/bin/env python3
"""Verify that the vision isolation validator rejects incomplete evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("qwen38_parallel_vision",
    Path(__file__).resolve().parents[1] / "validation" / "qwen38-parallel-vision.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def case(concurrency=1):
    return {"scenario": "image_follow_up", "tag": "image_follow_up-r0-i0",
            "status": "ok", "concurrency": concurrency, "turns": [
                {"request_sha256": "first-request", "expected": {"code": "4821"},
                 "metrics": {"assistant_message": {"content": '{"code":"4821"}'}, "finish_reason": "stop"}},
                {"request_sha256": "second-request", "expected": {"color": "red"},
                 "metrics": {"assistant_message": {"content": '{"color":"red"}'}, "finish_reason": "stop"}}]}


class VisionIsolationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.control = {"run_complete": True, "cases": [case()]}
        self.candidate = [case(2)]

    def compare(self):
        return validator.compare_sequential_baseline(self.candidate, self.control)

    def test_identical_complete_multi_turn_requests_pass(self):
        self.assertTrue(self.compare()["passed"])

    def test_correct_final_turn_cannot_hide_corrupted_first_turn(self):
        self.candidate[0]["turns"][0]["metrics"]["assistant_message"]["content"] = '{"code":"9364"}'
        self.assertFalse(self.compare()["passed"])

    def test_different_rendered_request_cannot_establish_parity(self):
        self.candidate[0]["turns"][0]["request_sha256"] = "different-request"
        self.assertFalse(self.compare()["passed"])

    def test_token_limit_is_not_completed_inference(self):
        self.candidate[0]["turns"][1]["metrics"]["finish_reason"] = "length"
        self.assertFalse(self.compare()["passed"])

    def test_incomplete_control_fails(self):
        self.control["run_complete"] = False
        self.assertFalse(self.compare()["passed"])

    def test_failed_control_fails(self):
        self.control["cases"][0]["status"] = "fail"
        self.assertFalse(self.compare()["passed"])

    def test_missing_image_control_fails(self):
        additional = copy.deepcopy(self.candidate[0])
        additional["tag"] = "image_follow_up-r0-i1"
        self.candidate.append(additional)
        self.assertFalse(self.compare()["passed"])

    def test_duplicate_controls_fail(self):
        self.control["cases"].append(copy.deepcopy(self.control["cases"][0]))
        self.assertFalse(self.compare()["passed"])

    def test_empty_turn_evidence_fails(self):
        self.control["cases"][0]["turns"] = []
        self.candidate[0]["turns"] = []
        self.assertFalse(self.compare()["passed"])

    def test_matching_first_turn_cannot_hide_missing_follow_up(self):
        self.control["cases"][0]["turns"] = self.control["cases"][0]["turns"][:1]
        self.candidate[0]["turns"] = self.candidate[0]["turns"][:1]
        self.assertFalse(self.compare()["passed"])

    def test_extra_follow_up_turns_do_not_match_the_scenario(self):
        for item in [self.control["cases"][0], self.candidate[0]]:
            item["turns"].append(copy.deepcopy(item["turns"][-1]))
        self.assertFalse(self.compare()["passed"])

    def test_duplicate_candidates_fail(self):
        self.candidate.append(copy.deepcopy(self.candidate[0]))
        self.assertFalse(self.compare()["passed"])

    def test_solo_candidates_cannot_hide_a_missing_parallel_image(self):
        second = case()
        second["tag"] = "image_follow_up-r0-i1"
        self.control["cases"].append(second)
        self.candidate.extend(copy.deepcopy(self.control["cases"]))
        # Both C=1 images are present, but C=2 still has only the first image.
        self.assertFalse(self.compare()["passed"])

    def test_one_turn_image_scenario_passes(self):
        for item in [self.control["cases"][0], self.candidate[0]]:
            item["scenario"] = "image_ocr"
            item["turns"] = item["turns"][:1]
        self.assertTrue(self.compare()["passed"])

    def test_declared_complete_baseline_with_missing_planned_cases_fails(self):
        self.control["execution_plan"] = {"expected_cases": 2}
        self.assertFalse(self.compare()["passed"])

    def test_unknown_scenario_cannot_establish_complete_coverage(self):
        for item in [self.control["cases"][0], self.candidate[0]]:
            item["scenario"] = "unknown"
            item["turns"] = item["turns"][:1]
        self.assertFalse(self.compare()["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
