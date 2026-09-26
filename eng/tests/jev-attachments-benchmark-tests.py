#!/usr/bin/env python3
"""Harness correctness tests. HTTP fixtures are not model-quality evidence."""
import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("jev_attachments_benchmark", Path(__file__).parents[1] / "jev-attachments-benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FixtureServer(BaseHTTPRequestHandler):
    payloads = []
    uploads = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/api/upload":
            self.uploads.append(raw)
            response = {"ok": True, "file": "generated-ticket.txt"}
        else:
            payload = json.loads(raw)
            self.payloads.append(payload)
            response = {"model": "mock-not-a-model", "answers": {"team": {
                "type": "choice", "choice": "billing", "confidence": 1,
                "probabilities": {"billing": 1, "technical": 0}}},
                "usage": {"input_tokens": 16, "output_tokens": 3},
                "diagnostics": {"attachments": [{"name": "ticket.txt", "kind": "text", "textCharacters": 101,
                                                 "imageCount": 0, "cacheHit": False}],
                                "timing": {"preprocessing_ms": 1, "inference_ms": 2, "total_ms": 3}}}
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AttachmentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.args = SimpleNamespace(model="jev-latest", fixtures=bench.FIXTURES, timeout=5, api_key=None,
                                    systemone_url=base + "/v1/systemone", upload_url=base + "/api/upload")
        self.case = {"id": "txt", "attachments": [{"field": "files", "path": "ticket.txt", "kind": "text"}],
                     "questions": {"team": {"type": "choice", "instructions": "Which team?",
                                             "criteria": {"billing": "Billing", "technical": "Technical"}}},
                     "expected": {"team": "billing"}}

    def test_checked_fixture_integrity_and_coverage(self):
        fixtures = bench.load_fixtures(bench.FIXTURES)
        self.assertGreaterEqual(len(fixtures["cases"]), 10)
        kinds = {a["kind"] for case in fixtures["cases"] for a in case["attachments"]}
        self.assertTrue({"text", "pdf", "document", "audio", "video", "image"} <= kinds)

    def test_modified_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ticket.txt").write_text("changed")
            (root / "manifest.json").write_text(json.dumps({"version": 1, "files": {"ticket.txt": {"bytes": 1, "sha256": "wrong"}}}))
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                bench.load_fixtures(root / "manifest.json")

    def test_inline_data_url_and_upload_reach_distinct_http_routes(self):
        source = (bench.FIXTURES.parent / "ticket.txt").read_bytes()
        for mode in ("inline", "data-url", "upload"):
            with self.subTest(mode=mode):
                row = bench.run_case(self.args, self.case, mode, 0)
                self.assertEqual("passed", row["status"], row)
                item = FixtureServer.payloads[-1]["files"][0]
                if mode == "upload":
                    self.assertEqual("generated-ticket.txt", item["file"])
                    self.assertNotIn("data", item)
                    self.assertIn(source, FixtureServer.uploads[-1])
                    self.assertGreater(row["upload_ms"], 0)
                else:
                    encoded = item["data"].split(",", 1)[-1]
                    self.assertEqual(source, base64.b64decode(encoded))
                    self.assertEqual(mode == "data-url", item["data"].startswith("data:"))
                self.assertGreaterEqual(row["end_to_end_ms"], row["request_ms"])
                self.assertEqual(1, row["preprocessing_ms"])
                self.assertEqual(2, row["inference_ms"])

    def test_wrong_answer_fails_quality_even_when_http_contract_is_valid(self):
        self.case["expected"]["team"] = "technical"
        row = bench.run_case(self.args, self.case, "inline", 0)
        self.assertEqual("failed", row["status"])
        self.assertIn("differs from gold", row["error"])

    def test_unavailable_is_not_a_pass_and_has_no_success_timings(self):
        with patch.object(bench.BENCH, "post", return_value={"status": 503, "body": {"error": "ASR unavailable"}, "elapsed_ms": 12}):
            row = bench.run_case(self.args, self.case, "inline", 0)
        self.assertEqual("unavailable", row["status"])
        summary = bench.summarize([row], 1, maximum_p95=1000)
        self.assertEqual(0, summary["passed"])
        self.assertEqual(1, summary["unavailable"])
        self.assertIsNone(summary["timings"]["end_to_end_ms"]["p95"])
        self.assertEqual("failed", summary["performance_budget_status"])

    def test_empty_attachment_diagnostics_cannot_hide_ignored_input(self):
        for attachment in ({"kind": "text", "textCharacters": 0}, {"kind": "image", "imageCount": 1}):
            with self.subTest(attachment=attachment), self.assertRaises(ValueError):
                bench.validate_diagnostics(self.case, {"diagnostics": {"attachments": [attachment]}})

    def test_bad_server_timings_are_rejected(self):
        row = bench.run_case(self.args, self.case, "inline", 0)
        for value in (float("nan"), -1, True, None):
            body = copy.deepcopy(row["response"]["body"])
            body["diagnostics"]["timing"]["preprocessing_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.validate_diagnostics(self.case, body)

    def test_internal_errors_do_not_pass_invalid_input_validation(self):
        for code in (200, 500, 503, None):
            with self.subTest(code=code), patch.object(bench.BENCH, "post", return_value={"status": code}):
                self.assertTrue(all(r["status"] == "failed" for r in bench.run_invalid(self.args)))
        with patch.object(bench.BENCH, "post", return_value={"status": 422}):
            self.assertTrue(all(r["status"] == "passed" for r in bench.run_invalid(self.args)))

    def test_performance_requires_explicit_budget_and_all_requests_pass(self):
        row = bench.run_case(self.args, self.case, "inline", 0)
        self.assertEqual("not_evaluated", bench.summarize([row], 1)["performance_budget_status"])
        self.assertEqual("passed", bench.summarize([row], 1, 10000)["performance_budget_status"])
        self.assertEqual("failed", bench.summarize([row, {**row, "status": "failed"}], 1, 10000)["performance_budget_status"])

    def test_evidence_directory_rejects_source_tree(self):
        with self.assertRaisesRegex(ValueError, "ignored"):
            bench.output_directory(bench.ROOT / "eng/generated-evidence")

    def test_references_are_reused_across_serial_and_concurrent_requests(self):
        manifest = bench.load_fixtures(bench.FIXTURES)
        calls = []
        def fake_run(args, case, mode, repetition, references):
            calls.append((case, mode, references.copy()))
            return {"case": case["id"], "status": "passed"}
        with patch.object(bench, "upload", side_effect=[
                {"status": 200, "body": {"file": "a.txt"}}, {"status": 200, "body": {"file": "b.txt"}}]) as uploads, \
                patch.object(bench, "run_case", side_effect=fake_run):
            rows = bench.run_reference_isolation(self.args, manifest)
        self.assertEqual(2, uploads.call_count)
        self.assertEqual(4, len(rows))
        self.assertEqual(2, sum(r["phase"] == "concurrent-reference-reuse" for r in rows))
        self.assertTrue(all(calls[0][2] == call[2] for call in calls))
        self.assertTrue(set(calls[0][0]["questions"]).isdisjoint(calls[1][0]["questions"]))

    def test_plain_control_has_no_attachment_answer_leakage(self):
        manifest = bench.load_fixtures(bench.FIXTURES)
        control = next(case for case in manifest["cases"] if case["id"] == "plain-control")
        self.assertEqual([], control["attachments"])
        payload = bench.make_payload(control, "jev-latest", 0)
        self.assertEqual(control["state"], payload["state"])
        attached = bench.make_payload(manifest["cases"][0], "jev-latest", 0)
        self.assertNotIn("billing", json.dumps(attached["state"]))


if __name__ == "__main__":
    unittest.main()
