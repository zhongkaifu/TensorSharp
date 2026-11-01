#!/usr/bin/env python3
"""Tests that the offline report cannot promote failed or absent measurements."""
import copy
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("qwen38_eval_report",
    Path(__file__).resolve().parents[1] / "qwen38-eval-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def header(repeats=1, scenarios=None):
    return {"kind": "run", "engine": "tensorsharp", "mode": "plain", "label": "candidate",
            "run_id": "test-run", "harness": "qwen38-engine-bench/2", "repeats": repeats,
            "scenarios": scenarios or ["short_chat"], "model_artifacts": [{"sha256": "abc123", "bytes": 42}]}


def case(repeat=0, scenario="short_chat"):
    return {"kind": "case", "run_id": "test-run", "repeat": repeat, "scenario": scenario,
            "pass": True, "verdict": "PASS", "checks": [{"name": "answer", "ok": True}], "metrics": {},
            "requests": [{"error": None, "finish_reason": "stop", "completion_tokens": 64,
                          "completion_tokens_source": "usage", "ttft_s": .123, "total_s": .456,
                          "decode_tok_s": 189.9, "prefill_tok_s": 8123.4, "prompt_tokens": 1000}]}


class OfflineReportTests(unittest.TestCase):
    def setUp(self):
        # Validation evidence and scratch stay in the ignored evidence directory.
        evidence = report.ROOT / "docs" / "validation" / "qwen38-report-tests"
        evidence.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=evidence)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, records, name="run.jsonl"):
        path = self.root / name
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return path

    def build(self, records, context=None):
        return report.build_summary([self.write(records)], context or {})

    def test_no_data_is_not_fabricated_as_zero_latency(self):
        summary = report.build_summary([], {"missingEngines": ["ignis"]})
        html = report.render_html(summary)
        self.assertEqual(summary["runs"], [])
        self.assertIn("没有可核验的引擎运行数据", html)
        self.assertIn("不能判断修复收益", html)
        self.assertNotIn("0.000", html)

    def test_failed_quality_hides_every_performance_measurement(self):
        bad = case()
        bad.update({"pass": False, "verdict": "FAIL", "checks": [{"name": "answer", "ok": False}]})
        summary = self.build([header(), bad])
        self.assertEqual(summary["rows"][0]["status"], "FAIL")
        self.assertEqual(summary["rows"][0]["metrics"], {})
        self.assertNotIn("189.9", report.render_html(summary))

    def test_missing_repeats_are_incomplete(self):
        summary = self.build([header(3), case()])
        row = summary["rows"][0]
        self.assertEqual(row["status"], "INCOMPLETE")
        self.assertEqual(row["expected_repeats"], 3)
        self.assertEqual(row["metrics"], {})

    def test_capacity_rejection_is_unsupported_not_a_fast_answer(self):
        rejected = case(scenario="long_prompt_250k")
        rejected.update({"availability": "capacity_rejected", "pass": False, "verdict": "ERROR",
                         "errors": ["HTTP 400: maximum context length exceeded"]})
        rejected["requests"][0]["error"] = rejected["errors"][0]
        summary = self.build([header(scenarios=["long_prompt_250k"]), rejected])
        row = summary["rows"][0]
        self.assertEqual(row["status"], "UNSUPPORTED")
        self.assertEqual(row["metrics"], {})
        self.assertIsNone(row["quality_rate"])
        self.assertIn("不支持 / 容量拒绝", report.render_html(summary))

    def test_header_plans_without_cases_are_visible(self):
        summary = self.build([header(scenarios=["short_chat", "long_prompt_128k"]), case()])
        row = next(row for row in summary["rows"] if row["scenario"] == "long_prompt_128k")
        self.assertEqual(row["status"], "INCOMPLETE")
        self.assertEqual(row["metrics"], {})

    def test_unrecorded_repeat_plan_cannot_claim_complete_validation(self):
        h = header()
        h.pop("repeats")
        row = self.build([h, case()])["rows"][0]
        self.assertEqual(row["status"], "INCOMPLETE")
        self.assertEqual(row["metrics"], {})

    def test_success_retains_raw_model_identity_and_actual_metrics(self):
        summary = self.build([header(), case()])
        self.assertEqual(summary["rows"][0]["metrics"]["decode_tok_s"], 189.9)
        self.assertEqual(summary["runs"][0]["header"]["model_artifacts"][0]["sha256"], "abc123")
        html = report.render_html(summary)
        self.assertIn("189.9", html)
        self.assertIn("abc123", html)
        self.assertIn("不同检查点和精度", html)
        self.assertIn("差异小于 5% 不能宣称领先", html)

    def test_text_context_is_escaped_not_executed(self):
        summary = self.build([header(), case()], {"title": '<script>alert("x")</script>',
            "summary": '<img src=x onerror="alert(1)">',
            "sources": [{"title": "unsafe", "url": "javascript:alert(1)"}]})
        html = report.render_html(summary)
        self.assertNotIn('<script>alert("x")</script>', html)
        self.assertNotIn('<img src=x', html)
        self.assertNotIn('href="javascript:', html)
        self.assertIn("&lt;script&gt;", html)

    def test_last_same_key_run_is_used_without_merging(self):
        bad = case()
        bad["verdict"], bad["pass"] = "FAIL", False
        summary = self.build([header(), case(), header(), bad])
        self.assertEqual(len(summary["runs"]), 1)
        self.assertEqual(summary["rows"][0]["status"], "FAIL")
        self.assertTrue(any("替换" in message for message in summary["diagnostics"]))

    def test_wrong_run_id_is_rejected_not_merged_into_quality(self):
        wrong = case()
        wrong["run_id"] = "another-run"
        summary = self.build([header(), wrong])
        self.assertEqual(summary["rows"][0]["status"], "INCOMPLETE")
        self.assertTrue(any("run_id" in message for message in summary["diagnostics"]))

    def test_missing_engine_retains_status_and_does_not_get_zero_scores(self):
        summary = self.build([header(), case()], {"missingEngines": [
            {"engine": "infernix", "status": "build_failed", "reason": "missing dependency"}]})
        row = next(row for row in summary["rows"] if row["engine"] == "infernix")
        self.assertEqual(row["status"], "BUILD_FAILED")
        self.assertEqual(row["metrics"], {})
        self.assertIsNone(row["quality_rate"])

    def test_actual_run_overrides_stale_context_missing_engine_status(self):
        summary = self.build([header(), case()], {"missingEngines": ["tensorsharp"]})
        self.assertEqual(summary["missing_engines"], [])
        self.assertEqual(len(summary["rows"]), 1)
        self.assertEqual(summary["rows"][0]["status"], "PASS")

    def test_successful_retry_does_not_hide_a_different_attempt_failure(self):
        summary = self.build([header(), case()], {"missingEngines": [
            {"engine": "tensorsharp", "mode": "plain", "label": "startup-attempt-1",
             "status": "ERROR", "reason": "server exited before readiness"},
            {"engine": "tensorsharp", "mode": "plain", "label": "candidate",
             "status": "NOT_RUN", "reason": "stale status before completed retry"}]})
        by_label = {row["label"]: row for row in summary["rows"]}
        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(by_label["candidate"]["status"], "PASS")
        self.assertEqual(by_label["startup-attempt-1"]["status"], "ERROR")
        self.assertEqual(by_label["startup-attempt-1"]["metrics"], {})

    def test_optional_figure_is_embedded_and_caption_is_escaped(self):
        picture = self.root / "plot.png"
        # A known one-pixel PNG; the report embeds bytes without importing image libraries.
        picture.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="))
        summary = self.build([header(), case()], {"figures": [
            {"path": str(picture), "title": "Measured plot", "caption": "<script>bad()</script>"},
            {"path": str(self.root / "missing.png"), "title": "Missing plot"}]})
        html = report.render_html(summary)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", html)
        self.assertNotIn("<script>bad()", html)
        self.assertNotIn("<h3>Missing plot</h3>", html)

    def test_cli_writes_standalone_html_and_machine_readable_summary(self):
        source = self.write([header(), case()])
        target = self.root / "report" / "evaluation.html"
        self.assertEqual(report.main([str(source), "--output", str(target)]), 0)
        self.assertTrue(target.is_file())
        summary = json.loads(target.with_name("summary.json").read_text())
        self.assertEqual(summary["rows"][0]["status"], "PASS")
        html = target.read_text()
        self.assertNotIn('<script src=', html)
        self.assertNotIn('cdn.', html)

    def test_globs_and_directories_are_deduplicated(self):
        source = self.write([header(), case()])
        paths, warnings = report.collect_inputs([str(self.root), str(self.root / "*.jsonl")])
        self.assertEqual(paths, [source.resolve()])
        self.assertEqual(warnings, [])

    def test_multi_turn_workflow_uses_sum_of_turns_for_e2e(self):
        multi = case(scenario="multi_turn")
        multi["metrics"]["turn_ttft_s"] = [.123, .123]
        multi["requests"].append(copy.deepcopy(multi["requests"][0]))
        summary = self.build([header(scenarios=["multi_turn"]), multi])
        self.assertAlmostEqual(summary["rows"][0]["metrics"]["end_to_end_s"], .912)


if __name__ == "__main__":
    unittest.main()
