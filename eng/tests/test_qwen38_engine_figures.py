#!/usr/bin/env python3
"""Data gates for offline figures; no plotting dependency or model execution."""
import builtins
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("qwen38_engine_figures",
    Path(__file__).resolve().parents[1] / "qwen38-engine-figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def header(repeats=3, engine="tensorsharp", mode="plain", label="candidate"):
    return {"kind": "run", "engine": engine, "mode": mode, "label": label, "run_id": "fixture",
            "harness": "qwen38-engine-bench/3", "repeats": repeats, "quality_tasks": 8, "gsm8k_limit": 8,
            "scenarios": ["long_generation", "quality_tasks", "quality_gsm8k", "long_prompt_128k", "long_prompt_250k"]}


def request(rate=200.0, tokens=256, suffix="code"):
    return {"error": None, "finish_reason": "stop", "completion_tokens": tokens,
            "completion_tokens_source": "usage", "visible_content_tokens": tokens,
            "first_payload_content_tokens": 2, "decode_numerator_tokens": tokens - 2,
            "decode_duration_s": (tokens - 2) / rate, "decode_tok_s": rate,
            "decode_tok_s_source": figures.RATE_SOURCE, "decode_numerator_source": figures.RATE_SOURCE,
            "decode_token_prefix_stable": True, "decode_unavailable_reason": None,
            "n_content_deltas": 10, "n_model_deltas": 10, "reasoning_chars": 0,
            "n_reasoning_deltas": 0, "n_tool_deltas": 0, "ttft_s": 2.0, "total_s": 3.0,
            "prompt_tokens": 131072, "content_sha256": "content-" + suffix,
            "request_sha256": "request-" + suffix, "response_sha256": "response-" + suffix}


def case(repeat=0, scenario="long_generation"):
    reqs = [request(suffix=kind) for kind in figures.KINDS] if scenario == "long_generation" else [request()]
    metrics = {"per_prompt": [{"kind": kind, "content_sha256": req["content_sha256"],
                               "completion_tokens": req["completion_tokens"], "decode_tok_s": req["decode_tok_s"]}
                              for kind, req in zip(figures.KINDS, reqs)]} if scenario == "long_generation" else {}
    return {"kind": "case", "run_id": "fixture", "scenario": scenario, "repeat": repeat,
            "pass": True, "verdict": "PASS", "checks": [{"name": "original_check", "ok": True}],
            "metrics": metrics, "requests": reqs, "errors": []}


def quality_case(repeat=0, scenario="quality_gsm8k", correct=3):
    c = case(repeat, scenario)
    ids = [f"gsm8k-test-{i}" for i in range(8)] if scenario == "quality_gsm8k" else [f"task{i}" for i in range(8)]
    c["requests"] = [request(suffix=name) for name in ids]
    c["checks"] = [{"name": name, "ok": i < correct} for i, name in enumerate(ids)]
    c["checks"].append({"name": "stream_completed", "ok": True})
    c["notes"] = {"ids": ids} if scenario == "quality_gsm8k" else {"expected": [{}] * 8}
    c["metrics"] = {"n_correct": correct, "n_questions": 8}
    c["pass"], c["verdict"] = correct == 8, "PASS" if correct == 8 else "FAIL"
    return c


def sglang_error_records():
    fixture = Path(__file__).with_name("fixtures") / "qwen38-sglang-quality-request-errors.json"
    return json.loads(fixture.read_text())["records"]


class FigureDataTests(unittest.TestCase):
    def setUp(self):
        parent = figures.ROOT / "docs" / "validation" / "qwen38-figure-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, records, name="raw.jsonl"):
        path = self.root / name
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return path

    def build(self, records, **kwargs):
        return figures.build_data([self.write(records)], **kwargs)

    def test_failed_original_check_withholds_all_three_decode_speeds(self):
        cases = [case(i) for i in range(3)]
        cases[1].update({"pass": False, "verdict": "FAIL", "checks": [{"name": "copy", "ok": False}]})
        data = self.build([header()] + cases)
        for row in data["decode"]["plain"]:
            self.assertFalse(row["qualified"])
            self.assertIsNone(row["median"])
            self.assertIsNone(row["ci95"])
            self.assertEqual(row["repeat_samples"], [])
            self.assertEqual(row["original_gate"], "FAIL")

    def test_missing_duplicate_and_bool_repeat_ids_do_not_show_speed(self):
        for ids in ([0, 1], [0, 1, 1], [False, 1, 2]):
            with self.subTest(ids=ids):
                data = self.build([header()] + [case(i) for i in ids])
                self.assertTrue(all(not row["qualified"] for row in data["decode"]["plain"]))

    def test_127_tokens_or_event_count_is_not_a_decode_speed(self):
        for change in ({"completion_tokens": 127}, {"completion_tokens_source": "sse_events"},
                       {"decode_tok_s_source": "legacy_completion_minus_one"},
                       {"decode_token_prefix_stable": False}, {"decode_duration_s": .1}):
            with self.subTest(change=change):
                cases = [case(i) for i in range(3)]
                cases[0]["requests"][0].update(change)
                cases[0]["metrics"]["per_prompt"][0]["completion_tokens"] = cases[0]["requests"][0]["completion_tokens"]
                data = self.build([header()] + cases)
                row = next(row for row in data["decode"]["plain"] if row["kind"] == "code")
                self.assertFalse(row["qualified"])
                self.assertIsNone(row["median"])

    def test_valid_decode_median_and_ci_are_by_whole_repeats(self):
        cases = [case(i) for i in range(3)]
        for c, speed in zip(cases, (100.0, 200.0, 500.0)):
            c["requests"][0] = request(speed)
            c["metrics"]["per_prompt"][0]["decode_tok_s"] = speed
        data = self.build([header()] + cases)
        row = next(row for row in data["decode"]["plain"] if row["kind"] == "code")
        self.assertTrue(row["qualified"])
        self.assertEqual(row["median"], 200)
        self.assertEqual(row["ci95"], list(figures.metrics.repeat_median_ci([[100], [200], [500]])))
        self.assertEqual([sample["repeat"] for sample in row["repeat_samples"]], [0, 1, 2])

    def test_ambiguous_per_prompt_hash_mapping_withholds_instead_of_assuming_order(self):
        cases = [case(i) for i in range(3)]
        cases[1]["metrics"]["per_prompt"][0]["content_sha256"] = "wrong-request"
        rows = self.build([header()] + cases)["decode"]["plain"]
        self.assertTrue(all(not row["qualified"] for row in rows))

    def test_gsm8k_failures_show_actual_counts_but_only_eight_unique_questions(self):
        data = self.build([header()] + [quality_case(i) for i in range(3)])
        row = next(row for row in data["quality"] if row["scenario"] == "quality_gsm8k")
        self.assertEqual(row["original_gate"], "FAIL")
        self.assertTrue(row["qualified"])
        self.assertEqual((row["correct"], row["total"]), (9, 24))
        self.assertEqual(row["unique_questions"], 8)
        self.assertFalse(row["independent_question_samples"])
        self.assertIsNone(row["confidence_interval"])
        self.assertIn("not 24 independent", row["sample_note"])
        self.assertEqual(row["request_availability"]["requests_with_errors"], 0)

    def test_actual_sglang_request_failures_are_withheld_not_model_zero(self):
        records = sglang_error_records()
        original = json.dumps(records, sort_keys=True)
        source = self.write(records)
        data = figures.build_data([source], context={"measurement_groups": {figures.report.digest_file(source): "vm1"}})
        self.assertEqual(json.dumps(records, sort_keys=True), original)
        for row in data["quality"]:
            with self.subTest(scenario=row["scenario"]):
                self.assertEqual(row["measurement_group"], "vm1")
                self.assertEqual(row["original_gate"], "FAIL")
                self.assertFalse(row["qualified"])
                self.assertTrue(row["counts_complete"])
                self.assertEqual(row["original_answer_counts"], {"correct": 0, "total": 24})
                self.assertIsNone(row["correct"])
                self.assertIsNone(row["total"])
                self.assertEqual(row["request_availability"]["requests_observed"], 24)
                self.assertEqual(row["request_availability"]["requests_with_errors"], 24)
                self.assertEqual(row["withheld_reason_code"], "request_error")
                self.assertEqual(figures.quality_withheld_label(row), "withheld (request errors 24/24)")

    def test_one_request_error_withholds_whole_quality_group(self):
        for scenario in ("quality_tasks", "quality_gsm8k"):
            with self.subTest(scenario=scenario):
                cases = [quality_case(i, scenario) for i in range(3)]
                cases[0]["requests"][7]["error"] = "ConnectionRefusedError: connection refused"
                cases[0]["verdict"] = "ERROR"
                row = next(row for row in self.build([header()] + cases)["quality"] if row["scenario"] == scenario)
                self.assertFalse(row["qualified"])
                self.assertIsNone(row["correct"])
                self.assertEqual(row["original_answer_counts"], {"correct": 9, "total": 24})
                self.assertEqual(row["request_availability"]["requests_with_errors"], 1)
                self.assertEqual(row["withheld_reason_code"], "request_error")

    def test_unavailable_quality_is_not_a_model_zero_even_without_request_error(self):
        for status in ("UNSUPPORTED", "ERROR", "INCOMPLETE"):
            with self.subTest(status=status):
                cases = [quality_case(i, "quality_tasks", 0) for i in range(3)]
                if status == "UNSUPPORTED":
                    for c in cases: c["availability"] = "unsupported"
                else:
                    cases[0]["verdict"] = status
                row = next(row for row in self.build([header()] + cases)["quality"] if row["scenario"] == "quality_tasks")
                self.assertFalse(row["qualified"])
                self.assertIsNone(row["correct"])
                self.assertEqual(row["original_gate"], "UNSUPPORTED" if status == "UNSUPPORTED" else "FAIL")
                self.assertEqual(row["withheld_reason_code"], "unavailable_case")
                self.assertIn(status, row["withheld_reason"])
                self.assertEqual(figures.quality_withheld_label(row), "withheld (unavailable case)")

    def test_quality_missing_or_duplicate_repeat_never_qualifies_with_or_without_error(self):
        for ids in ([0, 1], [0, 1, 1]):
            for error in (None, "Connection refused"):
                with self.subTest(ids=ids, error=error):
                    cases = [quality_case(i) for i in ids]
                    cases[0]["requests"][7]["error"] = error
                    row = next(row for row in self.build([header()] + cases)["quality"] if row["scenario"] == "quality_gsm8k")
                    self.assertFalse(row["qualified"])
                    self.assertFalse(row["counts_complete"])
                    self.assertIsNone(row["correct"])
                    self.assertEqual(row["withheld_reason_code"], "request_error" if error else "incomplete_counts")

    def test_incomplete_questions_and_inconsistent_original_scores_are_withheld(self):
        for change in ("missing_request", "wrong_correct", "wrong_total", "duplicate_check", "duplicate_id"):
            with self.subTest(change=change):
                cases = [quality_case(i) for i in range(3)]
                c = cases[1]
                if change == "missing_request": c["requests"].pop()
                elif change == "wrong_correct": c["metrics"]["n_correct"] = 8
                elif change == "wrong_total": c["metrics"]["n_questions"] = 7
                elif change == "duplicate_check": c["checks"][0]["name"] = c["checks"][1]["name"]
                else: c["notes"]["ids"][0] = c["notes"]["ids"][1]
                row = next(row for row in self.build([header()] + cases)["quality"] if row["scenario"] == "quality_gsm8k")
                self.assertFalse(row["qualified"])
                self.assertIsNone(row["correct"])
                self.assertIsNone(row["total"])

    def test_generated_tasks_keep_strict_original_count_not_stream_check(self):
        row = next(row for row in self.build([header()] + [quality_case(i, "quality_tasks", 0) for i in range(3)])["quality"]
                   if row["scenario"] == "quality_tasks")
        self.assertTrue(row["qualified"])
        self.assertEqual((row["correct"], row["total"]), (0, 24))
        self.assertEqual(row["request_availability"]["requests_with_errors"], 0)

    def test_long_failure_or_missing_repeat_never_gets_a_zero_or_point(self):
        for failure in ("quality", "missing", "unfinished_stream"):
            with self.subTest(failure=failure):
                cases = [case(i, "long_prompt_128k") for i in range(3)]
                if failure == "quality":
                    cases[1].update({"pass": False, "verdict": "FAIL", "checks": [{"name": "needle", "ok": False}]})
                elif failure == "missing": cases.pop()
                else: cases[1]["requests"][0]["finish_reason"] = None
                row = self.build([header()] + cases)["long_context"]["long_prompt_128k"][0]
                self.assertFalse(row["qualified"])
                self.assertIsNone(row["median"])
                self.assertEqual(row["repeat_samples"], [])

    def test_passing_long_context_requires_a_ttft_in_every_repeat(self):
        cases = [case(i, "long_prompt_250k") for i in range(3)]
        data = self.build([header()] + cases)
        self.assertTrue(data["long_context"]["long_prompt_250k"][0]["qualified"])
        cases[1]["requests"][0]["ttft_s"] = None
        self.assertFalse(self.build([header()] + cases)["long_context"]["long_prompt_250k"][0]["qualified"])

    def test_label_filters_and_alphabetical_order_do_not_rank_by_speed(self):
        records = []
        for name in ("zeta", "alpha", "old-baseline"):
            records += [header(label=name)] + [case(i) for i in range(3)]
        data = self.build(records, include=("a*", "z*", "old-*"), exclude=("old-*",))
        self.assertEqual([r["label"] for r in data["runs"]], ["alpha", "zeta"])
        self.assertEqual(data["selection"]["excluded_runs"], 1)

    def test_last_same_key_run_is_selected_without_merging(self):
        records = [header()] + [case(i) for i in range(3)] + [header(), case(0)]
        rows = self.build(records)["decode"]["plain"]
        self.assertTrue(all(not row["qualified"] for row in rows))

    def test_explicit_groups_keep_same_profile_and_compute_ci_within_each_run(self):
        sources = []
        samples = {"vm1": [100., 200., 300.], "vm2": [500., 600., 700.]}
        for group, rates in samples.items():
            records = [header()] + [case(i) for i in range(3)]
            records[0]["run_id"] = group
            for c, rate in zip(records[1:], rates):
                c["run_id"] = group
                c["requests"][0] = request(rate)
                c["metrics"]["per_prompt"][0]["decode_tok_s"] = rate
            sources.append(self.write(records, group + ".jsonl"))
        context = {"measurement_groups": {figures.report.digest_file(p): group for p, group in zip(sources, samples)}}
        data = figures.build_data(sources, context=context)
        self.assertEqual(len(data["runs"]), 2)
        rows = {r["measurement_group"]: r for r in data["decode"]["plain"] if r["kind"] == "code"}
        for group, rates in samples.items():
            self.assertTrue(rows[group]["qualified"])
            self.assertEqual(rows[group]["median"], rates[1])
            self.assertEqual(rows[group]["ci95"], list(figures.metrics.repeat_median_ci([[x] for x in rates])))
            self.assertEqual(len(rows[group]["repeat_samples"]), 3)
        with self.assertRaisesRegex(ValueError, "multiple measurement groups"):
            figures.render(data, self.root / "must-not-render")
        selected = figures.build_data(sources, context=context, measurement_group="vm1")
        self.assertEqual(len(selected["runs"]), 1)
        self.assertEqual(figures.render_measurement_group(selected), "vm1")

    def test_same_group_last_wins_and_unknown_sources_are_isolated(self):
        old = self.write([header()] + [case(i) for i in range(3)], "old.jsonl")
        new = self.write([header(), case()], "new.jsonl")
        mapping = {"measurement_groups": {figures.report.digest_file(p): "vm1" for p in (old, new)}}
        grouped = figures.build_data([old, new], context=mapping)
        self.assertEqual(len(grouped["runs"]), 1)
        self.assertTrue(all(not row["qualified"] for row in grouped["decode"]["plain"]))
        unknown = figures.build_data([old, new])
        self.assertEqual(len(unknown["runs"]), 2)
        self.assertEqual({r["measurement_group"] for r in unknown["runs"]}, {"UNSPECIFIED"})
        with self.assertRaisesRegex(ValueError, "multiple UNSPECIFIED inputs"):
            figures.render(unknown, self.root / "must-not-render")
        self.assertEqual(figures.render_measurement_group(figures.build_data([old])), "UNSPECIFIED")

    def test_bad_group_filter_and_mixed_cli_render_fail_before_creating_output(self):
        import contextlib
        import io
        sources = [self.write([header(label=name), case()], name + ".jsonl") for name in ("one", "two")]
        context = {"measurement_groups": {figures.report.digest_file(p): group for p, group in zip(sources, ("vm1", "vm2"))}}
        for group in ("", " vm1", "not-recorded"):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "measurement-group"):
                figures.build_data(sources, context=context, measurement_group=group)
        context_path = self.root / "context.json"
        context_path.write_text(json.dumps(context))
        out = self.root / "mixed-render"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            figures.main([*(str(p) for p in sources), "--output-dir", str(out), "--context-json", str(context_path)])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("multiple measurement groups", stderr.getvalue())
        self.assertFalse(out.exists())
        self.assertEqual(figures.main([*(str(p) for p in sources), "--output-dir", str(out), "--context-json", str(context_path), "--data-only"]), 0)
        self.assertEqual(json.loads((out / "data.json").read_text())["selection"]["measurement_groups"], ["vm1", "vm2"])

    def test_malformed_later_header_cannot_promote_an_earlier_complete_run(self):
        source = self.write([header()] + [case(i) for i in range(3)])
        source.write_text(source.read_text() + '{"kind":"run","engine":\n')
        data = figures.build_data([source])
        self.assertTrue(any("无效 JSON" in warning for warning in data["diagnostics"]))
        for row in data["decode"]["plain"]:
            self.assertFalse(row["qualified"])
            self.assertIsNone(row["median"])

    def test_single_complete_repeat_can_show_median_but_no_invented_interval(self):
        row = self.build([header(repeats=1), case()])["decode"]["plain"][0]
        self.assertTrue(row["qualified"])
        self.assertEqual(row["median"], 200)
        self.assertIsNone(row["ci95"])

    def test_data_only_cli_does_not_import_matplotlib_and_records_sources(self):
        source = self.write([header()] + [case(i) for i in range(3)])
        out = self.root / "figures"
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise AssertionError("data extraction imported matplotlib")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            self.assertEqual(figures.main([str(source), "--output-dir", str(out), "--data-only"]), 0)
        data = json.loads((out / "data.json").read_text())
        self.assertFalse(data["rendered"])
        self.assertEqual(data["sources"][0]["sha256"], figures.report.digest_file(source))
        self.assertEqual(len(data["tools"]), 3)
        self.assertEqual(data["figures"], [])

    def test_minimum_cannot_be_weakened_below128(self):
        with self.assertRaises(ValueError):
            figures.build_data([], min_decode_tokens=32)


if __name__ == "__main__":
    unittest.main()
