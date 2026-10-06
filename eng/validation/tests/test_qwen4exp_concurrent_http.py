"""Validate real HTTP streaming/admission and benchmark qualification locally."""
import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("qwen4exp_concurrent", Path(__file__).parents[1] / "qwen4exp-concurrent-http.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

FF7 = "《最终幻想7》是一款在1997年发行的角色扮演游戏。克劳德是故事的主角，神罗公司是重要组织。米德加是故事初期的舞台。" + "它的剧情围绕主角们的经历与星球的命运展开，玩家可以探索场景、参与战斗并了解各个角色。" * 2
TIME = "《时间简史》是霍金于1988年出版的科普著作。它讨论宇宙、大爆炸、黑洞和相对论，向普通读者解释物理学的重要问题。" + "书中使用容易理解的语言介绍这些概念，并探讨空间、时间以及宇宙的起源与演化。" * 2


class Handler(BaseHTTPRequestHandler):
    omit_done = False
    first_content_delay = .02
    step_delay = .003
    finish_in_first_delta = False
    entire_answer_first_delta = False
    get_count = 0
    fail_get_after = None

    def do_GET(self):
        Handler.get_count += 1
        if self.fail_get_after is not None and Handler.get_count > self.fail_get_after:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"unhealthy"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fixture-model"}]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = next(message["content"] for message in body["messages"] if message["role"] == "user")
        fixture = next((item for item in bench.FIXTURES.values() if item["prompt"] == prompt), None)
        answer = fixture["expected"] if fixture["kind"] == "exact" else FF7 if "最终幻想" in prompt else TIME
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        def send(event):
            wire = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)
            self.wfile.write(("data: " + wire + "\n\n").encode())
            self.wfile.flush()
        try:
            # Empty role headers must not count as TTFT or trigger cancellation.
            send({"choices": [{"delta": {"role": "assistant"}}]})
            time.sleep(self.first_content_delay)
            chunk_size = len(answer) if self.entire_answer_first_delta else 8
            for start in range(0, len(answer), chunk_size):
                choice = {"delta": {"content": answer[start:start + chunk_size]}}
                if self.finish_in_first_delta and start == 0:
                    choice["finish_reason"] = "stop"
                send({"choices": [choice]})
                time.sleep(self.step_delay)
            send({"choices": [{"delta": {}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 12, "completion_tokens": len(answer),
                            "prompt_tokens_details": {"cached_tokens": 0}}})
            if not self.omit_done:
                send("[DONE]")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # An actual intentional client disconnect reaches this handler.
            pass

    def log_message(self, *args):
        pass


class Qwen4ExpConcurrentHttpTests(unittest.TestCase):
    def setUp(self):
        Handler.omit_done = False
        Handler.finish_in_first_delta = False
        Handler.entire_answer_first_delta = False
        Handler.get_count = 0
        Handler.fail_get_after = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_real_stream_serial_parallel_staggered_and_cancellation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "concurrent.json"
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                result = bench.main(["--url", self.url, "--output", str(output),
                                     "--admission-delay-ms", "1", "--require-exact-parity", "--cancellation"])
            self.assertEqual(result, 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["groups"]), 7)
            for group in report["groups"]:
                if group["mode"] == "serial":
                    self.assertFalse(group["metrics"]["client_request_spans_overlap"])
                else:
                    self.assertTrue(group["metrics"]["client_request_spans_overlap"])
                if group["mode"] == "staggered":
                    self.assertTrue(group["admitted_during_generation"])
                for row in group["requests"]:
                    self.assertGreater(row["ttft_ms"], Handler.first_content_delay * 500)
                    self.assertEqual(row["request"]["temperature"], 0)
                    self.assertEqual(row["request"]["top_k"], 1)
                    self.assertEqual(row["request"]["seed"], 12345)
            cancelled, survivor, reused = report["groups"][-1]["requests"]
            self.assertTrue(cancelled["cancelled"])
            self.assertFalse(cancelled["done"])
            self.assertEqual(survivor["status"], "passed")
            self.assertEqual(reused["answer"], "EXACT_GAMMA=81")
            self.assertTrue(report["groups"][-1]["reuse_admitted_while_survivor_active"])
            self.assertTrue(report["groups"][-1]["cancelled_and_survivor_spans_overlap"])
            self.assertEqual(bench.verify_report_integrity(report)["retained_groups"], 7)

    def test_incomplete_sse_and_missing_fused_evidence_fail(self):
        Handler.omit_done = True
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "failed.json"
            log = root / "server.log"
            log.write_text("Batched fused decode accepted 2 sequences in one graph. Reported once.\n")
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                result = bench.main(["--url", self.url, "--output", str(output), "--groups", "markers",
                                     "--modes", "serial,parallel", "--server-log", str(log), "--require-fused"])
            report = json.loads(output.read_text())
            self.assertEqual(result, 1)
            self.assertFalse(report["fused_decode_evidence"]["new_acceptance_observed"])
            self.assertIn("without [DONE]", report["groups"][0]["requests"][0]["error"])
            self.assertFalse(report["groups"][1]["serial_comparison"]["performance_qualified"])

    def test_long_context_overlap_health_and_follow_up_gates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "long-context.json"
            argv = ["--url", self.url, "--output", str(output), "--groups", "topics",
                    "--modes", "parallel", "--concurrent-only", "--min-topic-total-tokens", "100",
                    "--require-generation-overlap", "--check-server-health", "--follow-up"]
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                self.assertEqual(bench.main(argv), 0)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(report["groups"][0]["metrics"]["client_generation_spans_overlap"])
                self.assertEqual(bench.verify_report_integrity(report)["retained_groups"], 1)
                unexercised = copy.deepcopy(report)
                unexercised["configuration"]["min_topic_total_tokens"] = 10000
                with self.assertRaisesRegex(ValueError, "unexercised long-context"):
                    bench.verify_report_integrity(unexercised)
                self.assertEqual(report["follow_up"]["answer"], "EXACT_GAMMA=81")
                self.assertEqual([item["status"] for item in report["server_health_checks"]], ["passed", "passed"])
                context = root / "system-context.txt"
                context.write_text("A retained neutral context prefix.", encoding="utf-8")
                self.assertEqual(bench.main(argv + ["--topic-system-prompt-file", str(context)]), 0)
                contextual = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(contextual["coverage"]["topic_prompt_variant"], "contextual_system_prefix")
                self.assertEqual(contextual["groups"][0]["requests"][0]["request"]["messages"], [
                    {"role": "system", "content": context.read_text(encoding="utf-8")},
                    {"role": "user", "content": bench.FIXTURES["ff7"]["prompt"]}])
                self.assertEqual(len(contextual["follow_up"]["request"]["messages"]), 1)
                self.assertEqual(bench.verify_report_integrity(contextual)["retained_groups"], 1)
                self.assertEqual(bench.main(argv + ["--min-topic-total-tokens", "10000"]), 1)
                failed = json.loads(output.read_text(encoding="utf-8"))
                self.assertIn("coverage was not exercised", failed["groups"][0]["requests"][0]["error"])
                Handler.entire_answer_first_delta = True
                self.assertEqual(bench.main(argv), 1)
                failed = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(failed["groups"][0]["requests"][0]["status"], "passed")
                self.assertFalse(failed["groups"][0]["metrics"]["client_generation_spans_overlap"])
                self.assertIn("generation spans did not overlap", failed["groups"][0]["error"])

    def test_health_failure_after_completed_streams_fails_report(self):
        Handler.fail_get_after = 1
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "unhealthy.json"
            log = root / "server.log"
            log.write_text("Startup log before the probe.\n", encoding="utf-8")
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                result = bench.main(["--url", self.url, "--output", str(output), "--groups", "markers",
                                     "--modes", "parallel", "--concurrent-only", "--check-server-health",
                                     "--server-log", str(log)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(report["groups"][0]["status"], "passed")
            self.assertEqual(report["server_health_checks"][0]["status"], "failed")
            self.assertIn("HTTP 503", report["failures"][0])
            self.assertFalse(report["fused_decode_evidence"]["new_acceptance_observed"])

    def test_integrity_rejects_missing_optional_checks_and_forged_overlap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "integrity-gates.json"
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                self.assertEqual(bench.main(["--url", self.url, "--output", str(output), "--groups", "topics",
                                             "--modes", "parallel", "--concurrent-only", "--follow-up",
                                             "--check-server-health", "--require-generation-overlap"]), 0)
            original = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(bench.verify_report_integrity(original)["follow_up_verified"])
            changes = [lambda report: report.pop("follow_up"),
                       lambda report: report.pop("server_health_checks"),
                       lambda report: report["follow_up"].update(status="failed", done=False),
                       lambda report: report["follow_up"].update(answer="EXACT_GAMMA=wrong"),
                       lambda report: report["server_health_checks"].pop(),
                       lambda report: report["server_health_checks"][0].update(status="failed", error="unavailable"),
                       lambda report: report["server_health_checks"][0].update(models=[{"id": "wrong-model"}])]
            for index, change in enumerate(changes):
                corrupt = copy.deepcopy(original)
                change(corrupt)
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    bench.verify_report_integrity(corrupt)
            forged = copy.deepcopy(original)
            group = forged["groups"][0]
            first, second = group["requests"]
            second["started_monotonic_s"] = first["ended_monotonic_s"] + 1
            second["ended_monotonic_s"] = second["started_monotonic_s"] + second["elapsed_ms"] / 1000
            first["last_delta_ms"] = (second["ended_monotonic_s"] - first["started_monotonic_s"] + 1) * 1000
            group["metrics"] = bench.group_metrics(group["requests"], group["metrics"]["wall_ms"])
            self.assertFalse(group["metrics"]["client_generation_spans_overlap"])
            with self.assertRaisesRegex(ValueError, "last_delta_ms disagrees with SSE"):
                bench.verify_report_integrity(forged)
            legacy = copy.deepcopy(original)
            for key in ("follow_up", "check_server_health", "require_generation_overlap", "min_topic_total_tokens", "topic_system_prompt"):
                legacy["configuration"].pop(key, None)
            legacy.pop("follow_up")
            legacy.pop("server_health_checks")
            legacy["groups"][0]["metrics"].pop("client_generation_spans_overlap")
            self.assertEqual(bench.verify_report_integrity(legacy)["status"], "verified")

    def test_concurrent_only_omits_local_controls_and_rejects_unavailable_gates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "concurrent-only.json"
            argv = ["--url", self.url, "--output", str(output), "--groups", "markers",
                    "--modes", "parallel", "--concurrent-only"]
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                self.assertEqual(bench.main(argv), 0)
                for unavailable in (["--min-parallel-speedup", "1"], ["--require-exact-parity"], ["--modes", "serial"]):
                    with self.subTest(gate=unavailable), self.assertRaises(ValueError):
                        bench.parse_args(argv + unavailable)
            report = json.loads(output.read_text())
            self.assertFalse(report["coverage"]["local_serial_controls_run"])
            self.assertEqual(report["configuration"]["modes"], ["parallel"])
            self.assertEqual(len(report["groups"]), 1)
            group = report["groups"][0]
            self.assertEqual(group["mode"], "parallel")
            self.assertEqual(group["local_serial_control"], "not_run_concurrent_only")
            self.assertNotIn("serial_comparison", group)
            self.assertNotIn("baseline_comparison", report)

    def test_concurrent_only_external_controls_gate_exact_parity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "artifacts" / "controls.json"
            candidate = root / "artifacts" / "candidate.json"
            base = ["--url", self.url, "--groups", "markers", "--modes", "parallel"]
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                # Default behavior still inserts a serial control even when
                # the caller selects only a concurrent mode.
                self.assertEqual(bench.main(base + ["--output", str(baseline)]), 0)
                controls = json.loads(baseline.read_text())
                self.assertEqual([group["mode"] for group in controls["groups"]], ["serial", "parallel"])
                argv = base + ["--output", str(candidate), "--concurrent-only", "--compare-with", str(baseline),
                               "--require-exact-parity", "--min-baseline-speedup", ".001"]
                self.assertEqual(bench.main(argv), 0)
                report = json.loads(candidate.read_text())
                comparison = report["baseline_comparison"]
                self.assertEqual(comparison["coverage_relation"], "candidate_subset")
                self.assertEqual(comparison["baseline_only_groups_not_compared"], [["markers", "serial", 1]])
                self.assertTrue(comparison["all_selected_answers_identical"])
                self.assertTrue(comparison["all_performance_comparisons_qualified"])
                controls["groups"][1]["requests"][0]["answer_sha256"] = "different"
                baseline.write_text(json.dumps(controls))
                self.assertEqual(bench.main(argv), 1)
                failed = json.loads(candidate.read_text())
                self.assertIn("retained answer hash", failed["failures"][0])
                self.assertNotIn("baseline_comparison", failed)
                self.assertEqual(failed["comparison_baseline_file"]["sha256"], bench.sha256(baseline.read_bytes()))

    def test_baseline_integrity_rejects_corrupt_hashes_duplicates_and_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "baseline.json"
            with patch.object(bench, "ROOT", root), patch("builtins.print"):
                self.assertEqual(bench.main(["--url", self.url, "--groups", "markers", "--modes", "parallel",
                                             "--output", str(output)]), 0)
            original = json.loads(output.read_text())
            self.assertEqual(bench.verify_report_integrity(original)["status"], "verified")
            changes = [lambda report: report["groups"][0]["requests"][0].update(answer="corrupt retained answer"),
                       lambda report: report["groups"][0]["requests"][0]["request"].update(seed=777),
                       lambda report: report["groups"][0]["requests"][0].update(answer_sha256="stale"),
                       lambda report: report["groups"][0]["requests"][0].pop("request_sha256"),
                       lambda report: report["groups"].append(copy.deepcopy(report["groups"][0])),
                       lambda report: report["groups"][0]["requests"].append(copy.deepcopy(report["groups"][0]["requests"][0])),
                       lambda report: report["groups"].pop(),
                       lambda report: report["groups"][0]["metrics"].update(wall_ms=float("inf")),
                       lambda report: report["groups"][0]["metrics"].update(aggregate_completion_tokens_per_s=999),
                       lambda report: report["groups"][0]["requests"][0]["usage"].update(completion_tokens=1),
                       lambda report: report["groups"][0]["requests"][0]["quality"].update(passed=False),
                       lambda report: report["groups"][0]["requests"][0]["events"].pop(),
                       lambda report: report["groups"][0]["requests"][0]["events"][1]["data"]["choices"][0]["delta"].update(content="corrupt SSE delta")]
            for index, change in enumerate(changes):
                with self.subTest(index=index):
                    report = copy.deepcopy(original)
                    change(report)
                    with self.assertRaises(ValueError):
                        bench.verify_report_integrity(report)
            # Recomputing the digest must not bless a different workload or a
            # corrupted answer whose stored quality result claims success.
            corrupt = copy.deepcopy(original)
            row = corrupt["groups"][0]["requests"][0]
            row["request"]["seed"] = 777
            row["request_sha256"] = bench.sha256(json.dumps(row["request"], ensure_ascii=False, sort_keys=True).encode())
            with self.assertRaisesRegex(ValueError, "declared model/fixture/sampling"):
                bench.verify_report_integrity(corrupt)
            corrupt = copy.deepcopy(original)
            row = corrupt["groups"][0]["requests"][0]
            row["answer"] = "wrong answer"
            row["answer_sha256"] = bench.sha256(row["answer"].encode())
            with self.assertRaisesRegex(ValueError, "SSE deltas"):
                bench.verify_report_integrity(corrupt)

    def test_cancellation_rejects_finished_or_complete_first_delta(self):
        args = SimpleNamespace(model="fixture-model", max_tokens=384, seed=12345)
        client = bench.Client(self.url, 5)
        Handler.finish_in_first_delta = True
        row = bench.run_request(args, client, "marker_long", cancel_after_deltas=1)
        self.assertEqual(row["status"], "failed")
        self.assertIn("finish reason", row["error"])
        Handler.finish_in_first_delta = False
        Handler.entire_answer_first_delta = True
        row = bench.run_request(args, client, "marker_long", cancel_after_deltas=1)
        self.assertEqual(row["status"], "failed")
        self.assertIn("Complete task answer", row["error"])

    def test_cancellation_requires_fresh_admission_while_survivor_active(self):
        class ImmediatelyFinishedSurvivor:
            def chat(self, body, row, first_delta=None, cancel_after_deltas=None):
                started = time.perf_counter()
                if cancel_after_deltas:
                    time.sleep(.03)  # Ensure the survivor has ended before reuse.
                    answer, done, cancelled = "\n", False, True
                else:
                    answer = FF7 if body["messages"][0]["content"] == bench.FIXTURES["ff7"]["prompt"] else "EXACT_GAMMA=81"
                    done, cancelled = True, False
                row.update(answer=answer, done=done, cancelled=cancelled, finish_reason="stop" if done else None,
                           started_monotonic_s=started, ended_monotonic_s=time.perf_counter(), ttft_ms=1,
                           completion_tokens=len(answer) if done else None)
                if first_delta:
                    first_delta.set()
        args = SimpleNamespace(model="fixture-model", max_tokens=384, seed=12345, timeout=5)
        group = bench.cancellation_and_reuse(args, ImmediatelyFinishedSurvivor())
        self.assertEqual(group["status"], "failed")
        self.assertFalse(group["reuse_admitted_while_survivor_active"])
        self.assertEqual([row["status"] for row in group["requests"]], ["cancelled_as_requested", "passed", "passed"])

    def test_quality_detects_leakage_wrong_marker_and_topic_drift(self):
        self.assertTrue(bench.quality_check(bench.FIXTURES["ff7"], FF7)["passed"])
        self.assertTrue(bench.quality_check(bench.FIXTURES["time_history"], TIME)["passed"])
        self.assertFalse(bench.quality_check(bench.FIXTURES["ff7"], TIME)["passed"])
        self.assertFalse(bench.quality_check(bench.FIXTURES["ff7"], FF7 + "霍金")["passed"])
        self.assertFalse(bench.quality_check(bench.FIXTURES["marker_short"], "EXACT_ALPHA=73")["passed"])
        self.assertFalse(bench.quality_check(bench.FIXTURES["marker_long"], "EXACT_BETA=73")["passed"])

    def test_report_comparison_suppresses_incomparable_speedups(self):
        request = {"fixture": "marker_short", "status": "passed", "request_sha256": "request",
                   "answer_sha256": "answer", "completion_tokens": 12, "finish_reason": "stop"}
        group = {"name": "markers", "mode": "parallel", "repetition": 1, "status": "passed",
                 "requests": [request], "metrics": {"wall_ms": 100, "aggregate_completion_tokens_per_s": 120}}
        baseline = {"model": "same", "status": "passed", "groups": [group]}
        candidate = copy.deepcopy(baseline)
        candidate["groups"][0]["metrics"].update(wall_ms=50, aggregate_completion_tokens_per_s=240)
        comparison = bench.compare_reports(baseline, candidate)
        self.assertTrue(comparison["all_performance_comparisons_qualified"])
        self.assertEqual(comparison["pairs"][0]["baseline_to_candidate_wall_speedup"], 2)
        for mutation in ("model", "tokens", "request", "finish", "status", "group_status", "report_status"):
            changed = copy.deepcopy(candidate)
            row = changed["groups"][0]["requests"][0]
            if mutation == "model":
                changed["model"] = "different"
            elif mutation == "tokens":
                row["completion_tokens"] = 10
            elif mutation == "request":
                row["request_sha256"] = "different"
            elif mutation == "finish":
                row["finish_reason"] = "length"
            elif mutation == "group_status":
                changed["groups"][0]["status"] = "failed"
            elif mutation == "report_status":
                changed["status"] = "failed"
            else:
                row["status"] = "failed"
            with self.subTest(mutation=mutation):
                result = bench.compare_reports(baseline, changed)
                self.assertFalse(result["all_performance_comparisons_qualified"])
                self.assertIsNone(result["pairs"][0]["baseline_to_candidate_wall_speedup"])

    def test_report_comparison_rejects_missing_or_duplicate_group_and_request_coverage(self):
        requests = [{"fixture": name, "status": "passed", "request_sha256": name,
                     "answer_sha256": name, "completion_tokens": 12, "finish_reason": "stop"}
                    for name in ("marker_short", "marker_long")]
        group = {"name": "markers", "mode": "parallel", "repetition": 1, "status": "passed",
                 "requests": requests, "metrics": {"wall_ms": 100, "aggregate_completion_tokens_per_s": 240}}
        baseline = {"model": "same", "status": "passed", "groups": [group]}
        for mutation in ("missing_group", "duplicate_group", "missing_request", "duplicate_request"):
            candidate = copy.deepcopy(baseline)
            if mutation == "missing_group":
                candidate["groups"][0]["mode"] = "staggered"
            elif mutation == "duplicate_group":
                candidate["groups"].append(copy.deepcopy(group))
            elif mutation == "missing_request":
                candidate["groups"][0]["requests"].pop()
            else:
                candidate["groups"][0]["requests"][1] = copy.deepcopy(requests[0])
            with self.subTest(mutation=mutation):
                result = bench.compare_reports(baseline, candidate)
                self.assertFalse(result["all_performance_comparisons_qualified"])
                self.assertFalse(result["all_selected_answers_identical"])

    def test_log_slice_ignores_old_acceptance_and_records_declines(self):
        with TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            old = b"Batched fused decode accepted 2 sequences in one graph. Reported once.\n"
            new = b"Batched fused decode accepted 3 sequences in one graph. Reported once.\nDefault batched fused-decode path declined a step; serving sequences round-robin\n"
            log.write_bytes(old + new)
            result = bench.log_evidence(log, len(old))
            self.assertEqual(result["accepted_batch_widths"], [3])
            self.assertTrue(result["new_acceptance_observed"])
            self.assertEqual(len(result["decline_lines"]), 1)

    def test_output_rejects_non_evidence_directory(self):
        with self.assertRaisesRegex(ValueError, "ignored artifacts"):
            bench.evidence_path(ROOT_FOR_INVALID_OUTPUT / "report.json")


ROOT_FOR_INVALID_OUTPUT = Path(__file__).parents[1]


if __name__ == "__main__":
    unittest.main()
