import copy
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("adaptive_llama_throughput",
    Path(__file__).resolve().parents[1] / "AdaptiveMemoryProbe" / "llama-throughput.py")
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


def response():
    settings = CLIENT.make_request(list(range(643)))
    settings["speculative.types"] = "none"
    return {"tokens": list(range(64)), "tokens_predicted": 64, "tokens_evaluated": 643,
            "tokens_cached": 706, "stop": True, "stop_type": "limit", "truncated": False,
            "generation_settings": settings,
            "timings": {"cache_n": 0, "prompt_n": 643, "predicted_n": 64,
                        "prompt_ms": 100, "predicted_ms": 1000,
                        "prompt_per_second": 6430, "predicted_per_second": 63}}


class AdaptiveLlamaThroughputTests(unittest.TestCase):
    def test_prompt_and_work_preserved_without_penalties_or_cache(self):
        tokens = list(range(643))
        body = CLIENT.make_request(tokens)
        self.assertEqual(tokens, body["prompt"])
        self.assertEqual(64, body["n_predict"])
        self.assertFalse(body["cache_prompt"])
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(1, body["repeat_penalty"])
        self.assertEqual("none", body["speculative.type"])

    def test_correct_denominator_is_63_not_64_and_slot_occupancy_is_not_a_cache_hit(self):
        row = CLIENT.validate_response(response(), list(range(64)))
        self.assertEqual(63, row["decode_tokens_per_second"])
        self.assertEqual(63, row["decode_calls"])
        self.assertTrue(row["raw_token_histories_match"])

    def test_cache_reuse_is_rejected(self):
        for value in (1, 643, None, False):
            candidate = response()
            candidate["timings"]["cache_n"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cache_n"):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_truncated_or_missing_truncation_evidence_is_rejected(self):
        for value in (True, None, 0):
            candidate = response()
            candidate["truncated"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "truncat"):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_short_or_noninteger_tokens_are_rejected(self):
        for tokens in (list(range(63)), list(range(65)), [True] * 64, [-1] * 64, None):
            candidate = response()
            candidate["tokens"] = tokens
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "token history"):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_eos_termination_and_mismatched_counters_are_rejected(self):
        changes = [("stop_type", "eos"), ("tokens_evaluated", 642), ("tokens_predicted", 63)]
        for field, value in changes:
            candidate = response()
            candidate[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CLIENT.validate_response(candidate, list(range(64)))
        for field, value in (("prompt_n", 642), ("predicted_n", 63)):
            candidate = response()
            candidate["timings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_wrong_generation_denominator_is_rejected(self):
        candidate = response()
        candidate["timings"]["predicted_per_second"] = 64
        with self.assertRaisesRegex(ValueError, "denominator"):
            CLIENT.validate_response(candidate, list(range(64)))

    def test_nonfinite_or_missing_timings_are_rejected(self):
        for value in (float("nan"), float("inf"), 0, -1, None):
            candidate = response()
            candidate["timings"]["predicted_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_speculation_or_sampling_cannot_silently_change_the_workload(self):
        for field, value in (("temperature", 1), ("speculative.types", "draft-mtp"), ("repeat_penalty", 1.1)):
            candidate = response()
            candidate["generation_settings"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_duplicate_none_speculation_entries_are_still_disabled(self):
        candidate = response()
        candidate["generation_settings"]["speculative.types"] = "none,none"
        self.assertTrue(CLIENT.validate_response(candidate, list(range(64)))["raw_token_histories_match"])
        for value in ("", "none,", "none,draft-mtp", None):
            candidate["generation_settings"]["speculative.types"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "speculative"):
                CLIENT.validate_response(candidate, list(range(64)))

    def test_raw_token_divergence_is_visible_without_a_quality_claim(self):
        expected = list(range(64))
        expected[37] = 999
        row = CLIENT.validate_response(response(), expected)
        self.assertFalse(row["raw_token_histories_match"])
        self.assertEqual([37], row["mismatched_token_positions"])
        self.assertEqual(37, row["first_mismatch"])

    def test_different_checkpoint_or_server_setup_is_rejected(self):
        identity = {"source_revision": CLIENT.SOURCE_REVISION, "binary_sha256": "a" * 64,
                    "model_sha256": "b" * 64, "command": ["llama-server", "--ctx-size", "2048"],
                    "startup_evidence": "server-startup.log", "configuration": copy.deepcopy(CLIENT.CONFIGURATION)}
        CLIENT.validate_identity(identity, {"ModelSha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "identities differ"):
            CLIENT.validate_identity(identity, {"ModelSha256": "c" * 64})
        identity["configuration"]["all_gpu"] = False
        with self.assertRaisesRegex(ValueError, "configuration"):
            CLIENT.validate_identity(identity, {"ModelSha256": "b" * 64})

    def test_warmup_is_excluded_from_medians(self):
        rows = [{"prefill_tokens_per_second": 1, "decode_tokens_per_second": 1}]
        rows += [{"prefill_tokens_per_second": 100, "decode_tokens_per_second": 20} for _ in range(3)]
        ts = [{"PromptTokens": 643, "DecodeCalls": 63, "PrefillMilliseconds": 6430, "DecodeMilliseconds": 3150}] * 4
        summary = CLIENT.summarize(rows, ts)
        self.assertEqual(3, summary["decode"]["llama_tokens_per_second"]["samples"])
        self.assertEqual(20, summary["decode"]["llama_tokens_per_second"]["median"])
        self.assertEqual(1, summary["decode"]["tensorsharp_to_llama_ratio"])


if __name__ == "__main__":
    unittest.main()
