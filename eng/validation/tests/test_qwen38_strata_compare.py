"""Reject incomplete model evidence, wrong token conditioning and weak answers."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "qwen38-strata-compare.py"
spec = importlib.util.spec_from_file_location("qwen38_strata_compare", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StrataComparisonTests(unittest.TestCase):
    def test_instruction_format_is_not_repaired_before_scoring(self):
        for case, text in (("math", "42."), ("math", "42!"),
                           ("extraction", "cedar | maple"), ("extraction", "cedar maple"),
                           ("code", "```python\ndef square(x):\n    return x*x\n```")):
            with self.subTest(case=case, text=text):
                self.assertFalse(runner.semantic_check(case, text, True)["passed"])

    def test_tool_arguments_are_complete_json_without_duplicate_keys(self):
        answer = '{"name":"get_weather","arguments":{"city":"Hangzhou","unit":"celsius"}}'
        self.assertTrue(runner.semantic_check("tool_json", answer, True)["passed"])
        for text, eos in ((answer, False), (answer.replace('celsius', 'fahrenheit'), True),
                          (answer.replace('"name":', '"name":"wrong","name":'), True),
                          ('```json\n' + answer + '\n```', True),
                          (answer[:-1] + ',"extra":1}', True)):
            with self.subTest(text=text, eos=eos):
                self.assertFalse(runner.semantic_check("tool_json", text, eos)["passed"])

    LOG = "prompt  : 11 22 33\noutput  : 42 99\ndecode  2 tokens in 123.0 ms -> 16.26 tok/s\nprefill  2 tokens in 51.0 ms -> 39.22 tok/s (time to first token 72.0 ms)\n"

    def test_completed_ids_and_different_timing_denominators(self):
        data = runner.parse_strata_log(self.LOG, [11, 22, 33], 64)
        self.assertEqual([42, 99], data["generated_tokens"])
        self.assertEqual(2, data["decode_tokens"])
        self.assertEqual(2, data["prefill_tokens"])
        self.assertEqual(72.0, data["time_to_first_token_ms"])

    def test_missing_duplicate_wrong_conditioning_and_denominators_fail(self):
        for log, prompt in ((self.LOG.replace("output  : 42 99\n", ""), [11, 22, 33]),
                            (self.LOG + "output  : 42 99\n", [11, 22, 33]),
                            (self.LOG, [11, 22, 34]),
                            (self.LOG.replace("decode  2", "decode  1"), [11, 22, 33]),
                            (self.LOG.replace("prefill  2", "prefill  3"), [11, 22, 33])):
            with self.subTest(log=log, prompt=prompt), self.assertRaises(ValueError):
                runner.parse_strata_log(log, prompt, 64)

    def test_eos_stopped_dump_has_known_final_prediction_position(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / "logits.bin"
            # Three prompt positions, two generated IDs: four consumed positions.
            dump.write_bytes(struct.pack("<ii8f", 2, 3 - 1 + 64, 1, 2, 3, 4, 5, 6, 7, 8))
            result = runner.parse_strata_final_logits(dump, 2, 3, 2, 64)
            self.assertEqual([7.0, 8.0], result["final_logits"])
            self.assertEqual(3, result["prediction_position"])
            for raw in (b"\0", struct.pack("<ii2f", 2, 66, 7, 8), struct.pack("<ii8f", 2, 66, 1, 2, 3, 4, 5, 6, float("nan"), 8)):
                dump.write_bytes(raw)
                with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                    runner.parse_strata_final_logits(dump, 2, 3, 2, 64)

    def test_semantics_are_independent_and_require_eos(self):
        for case, answer in (("math", "42"), ("extraction", "cedar, maple"), ("code", "def square(x):\n    return x * x")):
            with self.subTest(case=case):
                self.assertTrue(runner.semantic_check(case, answer, True)["passed"])
                self.assertFalse(runner.semantic_check(case, answer, False)["passed"])
        for case, answer in (("math", "17+25=43"), ("extraction", "cedar, maple, pine"),
                             ("code", "def square(x):\n    return x + x"),
                             ("code", "import os\ndef square(x):\n    return x*x"),
                             ("code", "@dangerous\ndef square(x):\n    return x*x")):
            with self.subTest(case=case, answer=answer):
                self.assertFalse(runner.semantic_check(case, answer, True)["passed"])

    def test_missing_cases_arms_repetitions_and_timeout_cannot_pass(self):
        passed = {"passed": True, "complete": True, "semantic_check": {"passed": True}}
        suite = {"math": {"runs": [{"arms": {"tensor": passed, "strata": passed}}]}}
        self.assertTrue(runner.completed_suite(suite, ["math"], ["tensor", "strata"], 1))
        for actual, cases, arms, repetitions in (({}, ["math"], ["tensor", "strata"], 1),
                                               (suite, ["math", "code"], ["tensor", "strata"], 1),
                                               (suite, ["math"], ["tensor", "strata", "cache"], 1),
                                               (suite, ["math"], ["tensor", "strata"], 2)):
            self.assertFalse(runner.completed_suite(actual, cases, arms, repetitions))
        timed_out = json.loads(json.dumps(suite))
        timed_out["math"]["runs"][0]["arms"]["strata"] = {"passed": False, "failure": "timeout"}
        self.assertFalse(runner.completed_suite(timed_out, ["math"], ["tensor", "strata"], 1))

    def test_supplemental_order_can_run_larger_cache_before_smaller_cache(self):
        self.assertEqual(["tensor-cache9472", "strata", "tensor-cache0", "tensor-cache8192"],
                         runner.engine_order([8192, 9472], 0, 9472))
        self.assertEqual(["strata", "tensor-cache0", "tensor-cache8192", "tensor-cache9472"],
                         runner.engine_order([8192, 9472], 1, 9472))
        self.assertEqual(["tensor-cache0", "tensor-cache8192", "tensor-cache9472", "strata"],
                         runner.engine_order([8192, 9472], 0))
        with self.assertRaises(ValueError):
            runner.engine_order([8192, 9472], 0, 4096)

    def test_strata_can_run_first_with_reused_conditioning_for_full_counterbalance(self):
        self.assertEqual(["strata", "tensor-cache0", "tensor-cache8192", "tensor-cache9472"],
                         runner.engine_order([8192, 9472], 0, strata_first=True))
        self.assertEqual(["tensor-cache0", "tensor-cache8192", "tensor-cache9472", "strata"],
                         runner.engine_order([8192, 9472], 1, strata_first=True))

    def test_long_numeric_answer_requires_all_twenty_correct_ordered_squares(self):
        answer = ", ".join(str(value * value) for value in range(1, 21))
        self.assertTrue(runner.semantic_check("squares", answer, True)["passed"])
        for wrong, complete in ((answer, False), (answer.rsplit(",", 1)[0], True),
                                (answer.replace("400", "401"), True),
                                ("The answer is " + answer, True),
                                (answer + ", 441", True)):
            with self.subTest(answer=wrong, complete=complete):
                self.assertFalse(runner.semantic_check("squares", wrong, complete)["passed"])

    def test_download_provenance_must_cover_exact_paths_sizes_and_publisher_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.gguf"
            checkpoint.write_bytes(b"checkpoint")
            report = Path(directory) / "download.json"
            shard = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                     "sha256": "a" * 64, "actual_sha256": "a" * 64, "verified": True}
            good = {"status": "verified", "shards": [shard]}
            report.write_text(json.dumps(good))
            self.assertEqual("verified", runner.verified_download(report, [checkpoint])["status"])
            for key, value in (("bytes", 1), ("path", str(Path(directory) / "other.gguf")),
                               ("verified", False), ("actual_sha256", "b" * 64), ("sha256", "bogus")):
                wrong = json.loads(json.dumps(good))
                wrong["shards"][0][key] = value
                report.write_text(json.dumps(wrong))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    runner.verified_download(report, [checkpoint])
            report.write_text(json.dumps({"status": "downloading", "shards": [shard]}))
            with self.assertRaises(ValueError):
                runner.verified_download(report, [checkpoint])

    def test_single_row_verifier_requires_observed_zero_drafts_and_complete_windows(self):
        log = "speculation   4 rounds of 2, drafts accepted 0 of 0 (0.000), 1.00 tokens per round\nwindow sizes   T1:4 T2:0 (min draft probability 1.00)\n"
        data = runner.parse_strata_scalar_execution(log, 2, 4)
        self.assertEqual(1, data["observed_target_window"])
        for wrong in (log.replace("T1:4 T2:0", "T1:3 T2:1"), log.replace("0 of 0", "0 of 4"),
                      log.replace("T1:4 T2:0", "T1:4"), log.replace("rounds of 2", "rounds of 4"),
                      log.replace("4 rounds", "3 rounds"), ""):
            with self.subTest(log=wrong), self.assertRaises(ValueError):
                runner.parse_strata_scalar_execution(wrong, 2, 4)


if __name__ == "__main__":
    unittest.main()
