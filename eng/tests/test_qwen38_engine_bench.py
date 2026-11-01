#!/usr/bin/env python3
"""Regression tests for benchmark integrity; no GPU or model is required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
        Path(__file__).resolve().parents[1] / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("qwen38-engine-bench")
report = load("qwen38-engine-report")


def successful_case(repeat=0):
    return {"scenario": "short_chat", "repeat": repeat, "pass": True, "verdict": "PASS",
        "checks": [{"name": "answer", "ok": True}], "metrics": {}, "requests": [{
            "error": None, "ttft_s": .1, "total_s": .2, "decode_tok_s": 100,
            "completion_tokens": 64, "completion_tokens_source": "usage",
            "request_sha256": "request", "response_sha256": "response", "finish_reason": "stop"}]}


class BenchmarkIntegrityTests(unittest.TestCase):
    def test_http_failure_after_correct_content_is_not_a_pass(self):
        rec = bench.case("short_chat", 0, [{"name": "answer", "ok": True}], {},
                         [{"content": "391", "error": "connection reset"}], 100)
        self.assertFalse(rec["pass"])

    def test_context_capacity_error_is_explicitly_recorded(self):
        rec = bench.case("long_prompt_250k", 0, [{"name": "needle", "ok": False}], {},
                         [{"error": "HTTP 400: maximum context length exceeded"}], 100)
        self.assertEqual(rec["availability"], "capacity_rejected")
        self.assertFalse(rec["pass"])

    def test_short_haystack_does_not_silently_cap_long_prompt_at_four_copies(self):
        paras = bench.take_prose({"paragraphs": ["a paragraph"]}, 1000)
        self.assertGreaterEqual(sum(len(p) + 2 for p in paras), 1000)
        self.assertGreater(len(paras), 4)

    def test_fit_counts_instructions_as_well_as_document(self):
        prompt, n = bench.fit_prompt_tokens(lambda s: "prefix:" + s + ":suffix", "a" * 200, 100, len)
        self.assertEqual(n, 100)
        self.assertEqual(len(prompt), 100)
        self.assertTrue(prompt.startswith("prefix:"))
        self.assertTrue(prompt.endswith(":suffix"))

    def test_real_128k_and_250k_construction_checks_three_depths_and_actual_usage(self):
        class FakeClient:
            tokenizer = object()
            count_tokens = staticmethod(len)

            def chat(self, messages, max_tokens, extra=None):
                prompt = messages[-1]["content"]
                self.prompt = prompt
                codes = dict(re.findall(r"Archive record: the (\w+) vault code is ([A-Z0-9-]+)\.", prompt))
                return {"content": json.dumps(codes), "error": None,
                        "prompt_tokens": len(prompt) + 128, "completion_tokens": 60, "finish_reason": "stop"}

        args = SimpleNamespace(nonce_seed=1234, chars_per_token=4.3, answer_tokens=512,
                               keep_text=1500, long_prompt_tolerance=.01)
        for k in (128, 250):
            client = FakeClient()
            ctx = bench.Ctx(args, client, {"paragraphs": ["The archive contains a historical record. " * 5]})
            rec = bench.run_long_prompt(ctx, 0, k, f"long_prompt_{k}k")
            self.assertTrue(rec["pass"])
            self.assertEqual(rec["metrics"]["prompt_tokens"], k * 1024)
            self.assertEqual(rec["metrics"]["user_prompt_tokens"], k * 1024 - 128)
            self.assertEqual(len(rec["checks"]), 5)
            positions = [client.prompt.index(f"Archive record: the {key}") / len(client.prompt)
                         for key in ("alpha", "beta", "gamma")]
            for actual, target in zip(positions, (.05, .5, .95)):
                self.assertAlmostEqual(actual, target, delta=.005)

    def test_128k_without_tokenizer_refuses_character_estimate(self):
        args = SimpleNamespace(nonce_seed=1234)
        ctx = bench.Ctx(args, SimpleNamespace(tokenizer=None), {"paragraphs": ["A document."]})
        with self.assertRaisesRegex(RuntimeError, "require --tokenizer"):
            bench.run_long_prompt(ctx, 0, 128, "long_prompt_128k")

    def test_five_needle_order_and_neighbors_are_measured_without_extra_requests(self):
        class OrderedClient:
            tokenizer = object()
            count_tokens = staticmethod(len)
            wrong_order = False
            calls = 0

            def chat(self, messages, max_tokens, extra=None):
                self.calls += 1
                prompt = messages[-1]["content"]
                records = re.findall(r"Archive record: the (\w+) vault code is ([A-Z0-9-]+)\.", prompt)
                actual = dict(records)
                order = [key for key, code in records]
                gamma = order.index("gamma")
                actual.update({"order": sorted(order) if self.wrong_order else order,
                    "before_gamma": order[gamma - 1] if gamma > 0 else None,
                    "after_gamma": order[gamma + 1] if gamma + 1 < len(order) else None})
                return {"content": json.dumps(actual), "error": None, "finish_reason": "stop",
                        "prompt_tokens": len(prompt) + 128, "completion_tokens": 128}

        args = SimpleNamespace(nonce_seed=1234, chars_per_token=4.3, answer_tokens=512,
                               keep_text=1500, long_prompt_tolerance=.01, long_order_quality=True)
        client = OrderedClient()
        ctx = bench.Ctx(args, client, {"paragraphs": ["The archive contains a historical record. " * 5]})
        rec = bench.run_long_prompt(ctx, 0, 128, "long_prompt_128k")
        self.assertTrue(rec["pass"])
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(rec["notes"]["physical_order"]), 5)
        self.assertEqual(rec["metrics"]["needle_depths"], (.05, .25, .5, .75, .95))
        self.assertTrue(any(c["name"] == "position_before_gamma" for c in rec["checks"]))
        client.wrong_order = True
        wrong = bench.run_long_prompt(ctx, 0, 128, "long_prompt_128k")
        self.assertFalse(wrong["pass"], "correct literal recall must not hide damaged positional order")
        self.assertTrue(all(c["ok"] for c in wrong["checks"] if c["name"].startswith("needle_")))

    def test_public_gsm8k_fixture_has_pinned_source_and_license(self):
        fixture = json.loads(bench.GSM8K_FIXTURE.read_text())
        self.assertRegex(fixture["upstream_revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(fixture["upstream_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(fixture["license"], "MIT")
        self.assertEqual(len(fixture["examples"]), 12)
        self.assertEqual(len({e["id"] for e in fixture["examples"]}), 12)
        license_text = (bench.GSM8K_FIXTURE.parent / fixture["license_file"]).read_text()
        self.assertIn("Copyright (c) 2021 OpenAI", license_text)
        meta = bench.file_metadata(bench.GSM8K_FIXTURE)
        self.assertEqual(meta["sha256"], hashlib.sha256(bench.GSM8K_FIXTURE.read_bytes()).hexdigest())

    def test_gsm8k_evaluates_final_answer_not_substring(self):
        examples = json.loads(bench.GSM8K_FIXTURE.read_text())["examples"]

        class MathClient:
            wrong = False

            def chat(self, messages, max_tokens, extra=None):
                found = next(e for e in examples if messages[-1]["content"].startswith(e["question"]))
                text = found["answer"] if not self.wrong else "The answer might be " + found["answer"] + " or 12345."
                return {"content": text, "error": None, "finish_reason": "stop"}

        client = MathClient()
        ctx = bench.Ctx(SimpleNamespace(nonce_seed=1234, gsm8k_limit=2, answer_tokens=256, keep_text=1500), client, {})
        self.assertTrue(bench.run_quality_gsm8k(ctx, 0)["pass"])
        client.wrong = True
        self.assertFalse(bench.run_quality_gsm8k(ctx, 0)["pass"])

    def test_strict_numeric_grading_rejects_prose_and_nonfinite_answers(self):
        self.assertEqual(bench.numeric_answer("1,234.00"), bench.numeric_answer("1234"))
        for answer in ("1234 or 4321", "The answer is 1234", "NaN", "Infinity", "1234%"):
            self.assertIsNone(bench.numeric_answer(answer))

    def test_missing_usage_does_not_count_speculative_sse_chunks_as_tokens(self):
        client = object.__new__(bench.Client)
        client.tokenizer = None
        obs = {"usage": None, "timings": None, "content": ["many tokens per event"], "reasoning": [],
               "tool_calls": {}, "n_content": 1, "n_reasoning": 0, "n_tool_deltas": 0,
               "first": 1., "last": 2., "t0": 0., "t_end": 2., "error": None,
               "http_status": 200, "finish_reason": "stop"}
        result = client.finish(obs)
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["decode_tok_s"])

    def test_correct_text_with_unterminated_stream_fails(self):
        rec = bench.case("short_chat", 0, [{"name": "answer", "ok": True}], {},
                         [{"content": "391", "error": None, "finish_reason": None}], 100)
        self.assertFalse(rec["pass"])

    def test_request_identity_ignores_generated_transport_ids(self):
        first = {"model": "alias-one", "messages": [{"role": "assistant", "tool_calls": [
            {"id": "random-1", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "random-1", "content": "hello"}]}
        second = json.loads(json.dumps(first).replace("random-1", "random-2"))
        second["model"] = "alias-two"
        self.assertEqual(bench.request_identity(first), bench.request_identity(second))

    def test_baseline_changed_output_fails_even_if_functional_checks_pass(self):
        reference = successful_case()
        candidate = copy.deepcopy(reference)
        candidate["requests"][0]["response_sha256"] = "changed"
        bench.compare_quality_baseline(candidate, reference)
        self.assertFalse(candidate["pass"])

    def test_baseline_changed_input_does_not_establish_parity(self):
        reference = successful_case()
        candidate = copy.deepcopy(reference)
        candidate["requests"][0]["request_sha256"] = "changed"
        bench.compare_quality_baseline(candidate, reference)
        self.assertFalse(candidate["pass"])

    def test_baseline_matching_full_outputs_pass(self):
        reference = successful_case()
        candidate = copy.deepcopy(reference)
        bench.compare_quality_baseline(candidate, reference)
        self.assertTrue(candidate["pass"])

    def test_malformed_scenario_is_reported_unknown(self):
        self.assertIsNone(bench.dispatch("long_prompt_bogusk"))
        self.assertIsNone(bench.dispatch("parallel_0"))


class QualityGatedReportTests(unittest.TestCase):
    def test_fast_wrong_answer_is_excluded_from_all_comparison_numbers(self):
        good, bad = successful_case(0), successful_case(1)
        bad.update({"pass": False, "verdict": "FAIL", "checks": [{"name": "answer", "ok": False}]})
        bad["requests"][0]["ttft_s"] = .001
        row = report.summarise([good, bad], 32, 2)["short_chat"]
        self.assertEqual(row["quality_gate"], "FAIL")
        self.assertEqual(row["pass"], 1)
        for metric in ("ttft_s", "decode_tok_s", "prefill_tok_s", "total_s"):
            self.assertIsNone(row[metric])

    def test_partial_run_is_not_passing_validation(self):
        row = report.summarise([successful_case()], 32, 3)["short_chat"]
        self.assertEqual(row["quality_gate"], "INCOMPLETE")
        self.assertIsNone(row["ttft_s"])

    def test_duplicate_repeats_do_not_fill_missing_samples(self):
        row = report.summarise([successful_case(), successful_case()], 32, 2)["short_chat"]
        self.assertEqual(row["quality_gate"], "INCOMPLETE")

    def test_legacy_delta_counts_never_enter_token_rates(self):
        case = successful_case()
        case["requests"][0]["completion_tokens_source"] = "deltas"
        row = report.summarise([case], 32, 1)["short_chat"]
        self.assertIsNone(row["decode_tok_s"])
        self.assertEqual(row["ttft_s"], .1)

    def test_passed_run_reports_repeat_bootstrap_and_p90(self):
        cases = [successful_case(i) for i in range(3)]
        for i, case in enumerate(cases):
            case["requests"][0]["ttft_s"] = .1 * (i + 1)
        row = report.summarise(cases, 32, 3)["short_chat"]
        self.assertEqual(row["quality_gate"], "PASS")
        self.assertAlmostEqual(row["ttft_s"], .2)
        self.assertAlmostEqual(row["ttft_p90_s"], .28)
        self.assertEqual(row["ttft_s_ci95_low"], .1)
        self.assertAlmostEqual(row["ttft_s_ci95_high"], .3)

    def test_failed_transport_cannot_claim_a_pass_from_legacy_record(self):
        case = successful_case()
        case["requests"][0]["error"] = "timeout"
        self.assertFalse(report.quality_passed(case))


class SemanticStreamMetricTests(unittest.TestCase):
    class WordTokenizer:
        def __init__(self):
            self.vocabulary = {}

        def encode(self, text, add_special_tokens=False):
            ids = []
            for word in text.split():
                self.vocabulary.setdefault(word, len(self.vocabulary))
                ids.append(self.vocabulary[word])
            return SimpleNamespace(ids=ids)

    @staticmethod
    def delta(value, finish=None):
        return {"choices": [{"index": 0, "delta": value, "finish_reason": finish}]}

    def observe(self, deltas, usage=6, tokenizer=True, timings=None):
        payloads = list(deltas) + [{"choices": [], "usage": {"prompt_tokens": 100,
                    "completion_tokens": usage, "prompt_tokens_details": {"cached_tokens": 0}}}]
        if timings:
            payloads[-1]["timings"] = timings

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                yield from [("data: " + json.dumps(value) + "\n\n").encode() for value in payloads]
                yield b"data: [DONE]\n\n"

        # Only semantic deltas should consume observation timestamps. Transport
        # role/id announcements do not start TTFT or inflate delta counts.
        n = sum(any((choice.get("delta") or {}).get(key) for key in ("content", "reasoning", "reasoning_content"))
                or any((tc.get("function") or {}).get("name") or (tc.get("function") or {}).get("arguments")
                       for tc in (choice.get("delta") or {}).get("tool_calls", []))
                for value in deltas for choice in value.get("choices", []))
        clock = [0.] + [1. + 2. * i for i in range(n)] + [2. + 2. * n]
        with patch.object(bench.urllib.request, "urlopen", return_value=Response()), \
                patch.object(bench.time, "perf_counter", side_effect=clock):
            obs = bench.stream_chat("http://mock", {"stream": True}, timeout=5)
        client = object.__new__(bench.Client)
        client.tokenizer = self.WordTokenizer() if tokenizer else None
        return obs, client.finish(obs)

    def test_multi_token_first_payload_is_entirely_excluded_from_decode_numerator(self):
        obs, result = self.observe([
            self.delta({"role": "assistant"}), self.delta({"content": "alpha beta gamma "}),
            self.delta({"content": "delta epsilon"}, "stop")])
        self.assertEqual(obs["n_model_deltas"], 2)
        self.assertEqual(obs["first_payload"]["content"], "alpha beta gamma ")
        self.assertEqual(result["first_payload_content_tokens"], 3)
        self.assertEqual(result["visible_content_tokens"], 5)
        self.assertEqual(result["decode_numerator_tokens"], 2)
        self.assertEqual(result["decode_duration_s"], 2.)
        self.assertEqual(result["decode_tok_s"], 1.)
        self.assertEqual(result["legacy_completion_minus_one_decode_tok_s"], 2.5)
        self.assertEqual(result["decode_usage_minus_visible_tokens"], 1)
        self.assertEqual(result["ttft_s"], 1.)
        self.assertIn("semantic", result["ttft_definition"])
        self.assertIn("not_pure_gpu_prefill", result["prefill_tok_s_source"])

    def test_fully_buffered_tool_call_has_no_decode_and_id_announcement_is_not_ttft(self):
        obs, result = self.observe([
            self.delta({"role": "assistant", "tool_calls": [{"index": 0, "id": "generated-id", "type": "function"}]}),
            self.delta({"tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]}, "tool_calls")])
        self.assertEqual(obs["n_tool_metadata_deltas"], 1)
        self.assertEqual(obs["n_model_deltas"], 1)
        self.assertEqual(result["ttft_s"], 1.)
        self.assertIsNone(result["decode_tok_s"])
        self.assertIn("tool_call_parser", result["decode_unavailable_reason"])
        self.assertEqual(result["first_payload"]["tool_calls"][0]["name"], "get_weather")

    def test_single_content_payload_cannot_produce_decode_rate(self):
        _, result = self.observe([self.delta({"content": "alpha beta gamma delta"}, "stop")])
        self.assertEqual(result["first_payload_content_tokens"], 4)
        self.assertEqual(result["decode_numerator_tokens"], 0)
        self.assertIsNone(result["decode_tok_s"])
        self.assertIsNone(result["legacy_completion_minus_one_decode_tok_s"])
        self.assertIn("single_semantic_payload", result["decode_unavailable_reason"])

    def test_reasoning_then_content_is_not_a_plain_content_decode_stream(self):
        _, result = self.observe([self.delta({"reasoning_content": "alpha beta "}),
                                  self.delta({"content": "gamma delta"}, "stop")])
        self.assertIsNone(result["decode_tok_s"])
        self.assertIn("reasoning", result["decode_unavailable_reason"])

    def test_missing_tokenizer_cannot_assume_one_token_in_first_payload(self):
        _, result = self.observe([self.delta({"content": "alpha beta gamma "}),
                                  self.delta({"content": "delta epsilon"}, "stop")], tokenizer=False)
        self.assertEqual(result["completion_tokens"], 6)
        self.assertIsNone(result["decode_tok_s"])
        self.assertIn("tokenizer_required", result["decode_unavailable_reason"])

    def test_zero_usage_remains_zero_and_server_timings_are_separate(self):
        _, result = self.observe([self.delta({"content": "alpha beta "}), self.delta({"content": "gamma"}, "stop")],
                                  usage=0, timings={"predicted_n": 10, "predicted_per_second": 999.})
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["completion_tokens_source"], "usage")
        self.assertIsNone(result["decode_tok_s"])
        self.assertEqual(result["decode_unavailable_reason"], "zero_completion_token_usage")
        self.assertEqual(result["server_decode_tok_s"], 999.)
        self.assertIn("separate", result["server_timing_source"])

    def test_partial_token_boundary_cannot_establish_first_payload_token_count(self):
        _, result = self.observe([self.delta({"content": "alpha"}), self.delta({"content": "beta"}, "stop")])
        self.assertFalse(result["decode_token_prefix_stable"])
        self.assertIsNone(result["decode_tok_s"])
        self.assertIn("not_a_stable_prefix", result["decode_unavailable_reason"])

    def test_summariser_uses_corrected_rate_and_never_the_legacy_diagnostic(self):
        _, result = self.observe([self.delta({"content": "alpha beta gamma "}),
                                  self.delta({"content": "delta epsilon"}, "stop")])
        record = bench.case("long_generation", 0, [{"name": "answer", "ok": True}], {}, [result], 1500)
        summary = report.summarise([record], 2, 1)["long_generation"]
        self.assertEqual(summary["quality_gate"], "PASS")
        self.assertEqual(summary["decode_tok_s"], 1.)


if __name__ == "__main__":
    unittest.main()
