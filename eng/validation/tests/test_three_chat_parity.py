import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("three_parity", Path(__file__).resolve().parents[1] / "unified-three-chat-parity.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ThreeChatParityTests(unittest.TestCase):
    def gate(self, a, b, actual_eos=False, serial_eos=False, **extra):
        return m.prefix_control_gate({"text": a}, {"text": b, "completed": True, **extra}, actual_eos, serial_eos)

    def test_identical_complete_answers_have_complete_parity(self):
        result = self.gate("answer", "answer", True, True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["complete_answer_text_equal"])

    def test_early_eos_cannot_pass_by_matching_a_longer_control_prefix(self):
        self.assertFalse(self.gate("answer", "answer continues", True, False)["passed"])
        self.assertFalse(self.gate("answer", "answer", True, False)["passed"])

    def test_serial_eos_shorter_than_actual_fails(self):
        self.assertFalse(self.gate("answer continues", "answer", True, True)["passed"])

    def test_bounded_controls_and_intentional_cancel_remain_prefix_only(self):
        for args in (("answer continues", "answer", True, False), ("ans", "answer", False, True)):
            result = self.gate(*args)
            self.assertTrue(result["passed"])
            self.assertFalse(result["complete_answer_text_equal"])

    def test_empty_divergent_failed_or_unfinished_controls_fail(self):
        for a, b, extra in (("", "", {}), ("answer", "other", {}), ("answer", "answer", {"error": "fail"}),
                            ("answer", "answer", {"completed": False}), ("answer", "answer", {"cleanup_error": "fail"})):
            self.assertFalse(self.gate(a, b, **extra)["passed"])
