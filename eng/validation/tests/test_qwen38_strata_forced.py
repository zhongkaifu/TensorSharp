"""Reject forced benchmarks whose engines consume different target positions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "qwen38-strata-forced.py"
spec = importlib.util.spec_from_file_location("strata_forced_compare", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ForcedStrataTests(unittest.TestCase):
    def setUp(self):
        self.plan = runner.conditioning([11, 22, 33], [44, 55], 99)

    def test_final_prompt_and_unconsumed_sentinel_align_every_target_position(self):
        plan = self.plan
        self.assertEqual([11, 22], plan["tensor_prompt"])
        self.assertEqual([33, 44, 55], plan["tensor_forced"])
        self.assertEqual([44, 55, 99], plan["strata_follow"])
        tensor_context = list(plan["tensor_prompt"])
        strata_context = list(plan["strata_prompt"][:-1])
        strata_inputs = [plan["strata_prompt"][-1]] + plan["strata_follow"][:-1]
        for left, right in zip(plan["tensor_forced"], strata_inputs):
            tensor_context.append(left)
            strata_context.append(right)
            self.assertEqual(tensor_context, strata_context)
        self.assertEqual(plan["consumed_tokens"], tensor_context)
        self.assertNotIn(99, tensor_context)
        self.assertEqual(2, plan["prefill_tokens"])
        self.assertEqual(3, plan["decode_tokens"])

    def test_default_sentinel_does_not_change_consumed_context(self):
        plan = runner.conditioning([11, 22, 33], [44, 55])
        self.assertEqual([44, 55, 55], plan["strata_follow"])
        self.assertEqual(self.plan["consumed_tokens"], plan["consumed_tokens"])

    def test_invalid_prefix_continuation_or_token_is_refused(self):
        for prompt, continuation, sentinel in (([1], [2], 3), ([1, 2], [], 3),
                                                ([1, -2], [3], 4), ([1, 2], [3], -4),
                                                ([1, 2], [True], 4), ([1, 2], [3.0], 4)):
            with self.subTest(prompt=prompt, continuation=continuation, sentinel=sentinel), self.assertRaises(ValueError):
                runner.conditioning(prompt, continuation, sentinel)

    def tensor(self):
        return {"passed": True, "synthetic": False, "decode_mode": "teacher-forced",
                "final_logits": [0.25, -0.5],
                "prompt_tokens": self.plan["tensor_prompt"], "forced_tokens": self.plan["tensor_forced"],
                "runs": [{"warmup": False, "finish_reason": "teacher-forced", "prefill_tokens": 2,
                          "decode_tokens": 3, "decode_step_ms": [9, 10, 11], "logit_rows": 4,
                          "prefill_ms": 20, "decode_ms": 30, "prefill_tps": 0, "decode_tps": 0}]}

    def test_tensor_requires_exact_inputs_steps_and_denominators(self):
        row = runner.validate_tensor(self.tensor(), self.plan)
        self.assertEqual(100, row["decode_tps"])
        self.assertEqual(50, row["combined_target_ms"])
        for key, value in (("passed", False), ("synthetic", True), ("decode_mode", "greedy"),
                           ("prompt_tokens", [11, 22, 33]), ("forced_tokens", [44, 55, 99]),
                           ("final_logits", []), ("final_logits", [0.25, float("nan")])):
            data = self.tensor()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_tensor(data, self.plan)
        for key, value in (("prefill_tokens", 3), ("decode_tokens", 2), ("logit_rows", 3),
                           ("decode_step_ms", [9, 10]), ("decode_step_ms", [9, float("nan"), 11]),
                           ("warmup", True), ("decode_ms", 0),
                           ("prefill_ms", float("nan"))):
            data = self.tensor()
            data["runs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_tensor(data, self.plan)

    LOG = ("prompt  : 11 22 33\noutput  : 44 55 99\n"
           "decode  3 tokens in 30.0 ms -> 100.00 tok/s\n"
           "prefill  2 tokens in 20.0 ms -> 100.00 tok/s (time to first token 45.0 ms)\n"
           "speculation  3 rounds of 2, drafts accepted 0 of 0 (0.000), 1.00 tokens per round\n"
           "window sizes  T1:3 T2:0 (min draft probability 1.00)\n"
           "follow  2 of 3 emitted tokens differ from the argmax\n")

    def test_strata_requires_complete_scalar_forced_execution(self):
        row = runner.validate_strata(self.LOG, self.plan)
        self.assertEqual(2, row["forced_argmax_disagreements"])
        self.assertEqual(100, row["decode_tps"])
        for wrong in (self.LOG.replace("44 55 99", "44 99 55"),
                      self.LOG.replace("T1:3 T2:0", "T1:1 T2:1"),
                      self.LOG.replace("2 of 3 emitted", "2 of 2 emitted"),
                      self.LOG.replace("2 of 3 emitted", "4 of 3 emitted"),
                      self.LOG.replace("follow  2 of 3 emitted tokens differ from the argmax\n", ""),
                      self.LOG + "follow  2 of 3 emitted tokens differ from the argmax\n"):
            with self.subTest(log=wrong), self.assertRaises(ValueError):
                runner.validate_strata(wrong, self.plan)

    def test_token_file_preserves_exact_order_and_rejects_invalid_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.ids"
            path.write_text("\ufeff11,22\n33\t44 ", encoding="utf-8")
            self.assertEqual([11, 22, 33, 44], runner.read_ids(path))
            for text in ("", "-1,2", "1.5,2", "1,no"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    runner.read_ids(path)


if __name__ == "__main__":
    unittest.main()
