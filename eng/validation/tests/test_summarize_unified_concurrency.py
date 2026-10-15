import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).parents[1] / "summarize-unified-concurrency.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def run_summary(self, report, samples=None, case_directory="concurrency"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / case_directory).mkdir()
            (root / case_directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
            if samples is None:
                samples = [{"rss": 10, "private": 20, "board_gpu": "GPU-a, 100, 200, 30"},
                           {"rss": 12, "private": 18, "board_gpu": "N/A"}]
            (root / "memory-samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            return summary.summarize(root, case_directory)

    def test_three_chat_verified_graphs_are_reported_without_claiming_full_continuity(self):
        result = self.run_summary({"passed": True, "native": {
            "passed": True, "n3_steps": 17, "n2_sibling_steps": 260}}, case_directory="three-chat")
        self.assertTrue(result["actual_joint_engagement_verified"])
        self.assertEqual(277, result["joint_steps"])
        self.assertFalse(result["full_interval_continuity_verified"])

    def test_three_chat_requires_both_live_cohorts_and_verified_native_proof(self):
        for native in ({"passed": False, "n3_steps": 17, "n2_sibling_steps": 260},
                       {"passed": True, "n3_steps": 0, "n2_sibling_steps": 260},
                       {"passed": True, "n3_steps": 17, "n2_sibling_steps": 0}, None):
            with self.subTest(native=native):
                result = self.run_summary({"passed": True, "native": native}, case_directory="three-chat")
                self.assertFalse(result["actual_joint_engagement_verified"])
                self.assertIsNone(result["joint_steps"])

    def generation(self, **changes):
        row = {"http_request_id": "r", "internal_request_id": "g1", "prompt_tokens": 1000,
               "kv_reused_tokens": 990, "attributed_prefill_ns": 500000000,
               "attributed_decode_ns": 250000000, "recorded_pipeline_tokens": 20,
               "engine_output_tokens": 21, "finish_reason": "eos", "engine_completion_available": True}
        row.update(changes)
        return row

    def report(self, generations, **changes):
        report = {"concurrent": [{"name": "one", "request_id": "r",
            "terminal": {"tokenCount": 20, "tokPerSec": 999},
            "phase_observations": {"first_content_to_terminal_seconds": 2}}],
            "generation_timings": {"rows": generations}}
        report.update(changes)
        return report

    def test_cache_hit_tokens_are_not_counted_as_prefill_work(self):
        result = self.run_summary({
            "passed": False, "overlap": {"passed": True},
            "concurrent": [{"name": "one", "request_id": "r", "terminal": {"tokenCount": 20, "tokPerSec": 80},
                            "phase_observations": {"first_content_to_terminal_seconds": 2}}],
            "generation_timings": {"rows": [self.generation()]}})
        self.assertFalse(result["actual_joint_engagement_verified"])
        self.assertFalse(result["acceptance_passed"])
        row = result["requests"][0]
        self.assertEqual(20, row["attributed_prefill_tokens_per_second"])
        self.assertEqual(10, row["pipeline_tokens_per_first_content_to_terminal_second"])
        self.assertEqual(80, row["attributed_decode_tokens_per_second"])
        self.assertEqual(20, row["recorded_pipeline_output_tokens"])
        self.assertEqual(21, row["engine_output_tokens"])
        self.assertNotIn("visible_output_tokens", row)
        self.assertEqual(12, result["process_sampled_peak_rss_bytes"])
        self.assertEqual(100, result["whole_board_sampled_peak_used_mib"])

    def test_absent_completion_timing_remains_unavailable(self):
        result = self.run_summary({"concurrent": [{"terminal": None}]})
        row = result["requests"][0]
        self.assertIsNone(row["attributed_prefill_seconds"])
        self.assertIsNone(row["attributed_prefill_tokens_per_second"])
        self.assertIsNone(row["pipeline_tokens_per_first_content_to_terminal_second"])
        self.assertIsNone(row["finish_reason"])

    def test_multiple_rounds_sum_matching_tokens_and_durations(self):
        rounds = [self.generation(finish_reason="tool_calls"),
                  self.generation(internal_request_id="g2", prompt_tokens=1200, kv_reused_tokens=1170,
                      attributed_prefill_ns=1500000000, attributed_decode_ns=750000000,
                      recorded_pipeline_tokens=30, engine_output_tokens=31)]
        result = self.run_summary(self.report(rounds))
        row = result["requests"][0]
        self.assertTrue(row["timing_complete"])
        self.assertEqual(2, row["generation_count"])
        self.assertEqual(rounds, row["generation_timings"])
        self.assertEqual(["tool_calls", "eos"], row["generation_finish_reasons"])
        self.assertEqual("eos", row["finish_reason"])
        self.assertEqual(40, row["uncached_prompt_tokens"])
        self.assertEqual(2160, row["reused_prompt_tokens"])
        self.assertEqual(20, row["attributed_prefill_tokens_per_second"])
        self.assertEqual(50, row["recorded_pipeline_output_tokens"])
        self.assertEqual(52, row["engine_output_tokens"])
        self.assertEqual(50, row["attributed_decode_tokens_per_second"])
        self.assertEqual(25, row["pipeline_tokens_per_first_content_to_terminal_second"])
        # The saved terminal value is retained but never used as measured decode.
        self.assertEqual(999, row["terminal_reported_tokens_per_second"])

    def test_missing_completion_in_any_round_cannot_use_later_or_terminal_rate(self):
        rounds = [self.generation(engine_completion_available=False, attributed_prefill_ns=None,
                      attributed_decode_ns=None, engine_output_tokens=None),
                  self.generation(internal_request_id="g2")]
        row = self.run_summary(self.report(rounds))["requests"][0]
        self.assertFalse(row["timing_complete"])
        self.assertEqual(40, row["recorded_pipeline_output_tokens"])
        for key in ("engine_output_tokens", "attributed_prefill_seconds", "attributed_decode_seconds",
                    "attributed_prefill_tokens_per_second", "attributed_decode_tokens_per_second"):
            self.assertIsNone(row[key], key)
        self.assertEqual(999, row["terminal_reported_tokens_per_second"])

    def test_zero_decode_duration_does_not_turn_wall_fallback_into_attributed_rate(self):
        row = self.run_summary(self.report([self.generation(attributed_decode_ns=0)]))["requests"][0]
        self.assertTrue(row["timing_complete"])
        self.assertEqual(0, row["attributed_decode_seconds"])
        self.assertIsNone(row["attributed_decode_tokens_per_second"])
        self.assertEqual(999, row["terminal_reported_tokens_per_second"])

    def test_other_http_generations_are_not_added(self):
        rounds = [self.generation(), self.generation(http_request_id="other", internal_request_id="g2")]
        row = self.run_summary(self.report(rounds))["requests"][0]
        self.assertEqual(1, row["generation_count"])
        self.assertEqual(20, row["recorded_pipeline_output_tokens"])

    def test_duplicate_generation_identity_rejects_double_counting(self):
        row = self.run_summary(self.report([self.generation(), self.generation()]))["requests"][0]
        self.assertFalse(row["timing_complete"])
        self.assertTrue(row["timing_errors"])
        self.assertIsNone(row["recorded_pipeline_output_tokens"])
        self.assertIsNone(row["attributed_decode_tokens_per_second"])

    def test_timing_parser_errors_do_not_allow_partial_rate(self):
        report = self.report([self.generation()])
        report["generation_timings"]["errors"] = ["another generation was malformed"]
        row = self.run_summary(report)["requests"][0]
        self.assertFalse(row["timing_complete"])
        self.assertIsNone(row["attributed_decode_tokens_per_second"])

    def test_invalid_reused_count_is_not_hidden_by_another_round(self):
        rounds = [self.generation(prompt_tokens=1, kv_reused_tokens=2),
                  self.generation(internal_request_id="g2")]
        row = self.run_summary(self.report(rounds))["requests"][0]
        self.assertIsNone(row["uncached_prompt_tokens"])
        self.assertIsNone(row["attributed_prefill_tokens_per_second"])

    def test_linux_unknown_private_and_error_samples_remain_unknown(self):
        samples = [{"rss": 100, "private": None}, {"error": "process observation unavailable"},
                   {"rss": 80, "private": None}]
        result = self.run_summary({}, samples)
        self.assertEqual(100, result["process_sampled_peak_rss_bytes"])
        self.assertIsNone(result["process_sampled_peak_private_bytes"])
        self.assertIsNone(result["whole_board_sampled_peak_used_mib"])
        self.assertEqual(2, result["resource_sample_counts"]["rss_measured"])
        self.assertEqual(0, result["resource_sample_counts"]["private_measured"])
        self.assertEqual(1, result["resource_sample_counts"]["samples_with_error"])

    def test_mixed_unknown_and_valid_resource_values_keep_only_measurements(self):
        samples = [{"rss": None, "private": None}, {"rss": 0, "private": 30},
                   {"rss": 10, "private": 20}, {"rss": True, "private": -1},
                   {"board_gpu": "GPU-a, NaN, 100, 20\nGPU-b, inf, 100, 20\nGPU-c, -1, 100, 20"}]
        result = self.run_summary({}, samples)
        self.assertEqual(10, result["process_sampled_peak_rss_bytes"])
        self.assertEqual(30, result["process_sampled_peak_private_bytes"])
        self.assertEqual(2, result["resource_sample_counts"]["rss_measured"])
        self.assertIsNone(result["whole_board_sampled_peak_used_mib"])

    def test_empty_or_all_error_resource_samples_are_not_zero(self):
        for samples in ([], [{"error": "gone"}]):
            with self.subTest(samples=samples):
                result = self.run_summary({}, samples)
                self.assertIsNone(result["process_sampled_peak_rss_bytes"])
                self.assertIsNone(result["process_sampled_peak_private_bytes"])


if __name__ == "__main__":
    unittest.main()
