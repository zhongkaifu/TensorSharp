import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("unified_http_checks", Path(__file__).resolve().parents[1] / "unified-server-http-checks.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def good_answer(content="42"):
    return {"status": 200, "transport_complete": True, "request_id_echo": True,
            "natural_eos_telemetry": True, "json": {"choices": [{"message": {"content": content},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 2}}}


class HttpCheckerTests(unittest.TestCase):
    def test_output_limit_omission_is_explicit_and_existing_payload_keeps_64(self):
        ordinary = m.chat_payload("test.gguf", m.TEXT)
        inherited = m.chat_payload("test.gguf", m.TEXT, use_default_output_limit=True)
        self.assertEqual(64, ordinary.pop("max_tokens"))
        self.assertEqual(ordinary, inherited)
        self.assertNotIn("max_tokens", inherited)
        self.assertNotIn("max_completion_tokens", inherited)

    def test_default_limit_requires_actual_omission_correct_answer_and_eos(self):
        def row():
            value = good_answer()
            value["request"] = {"method": "POST", "route": "/v1/chat/completions",
                                "body": m.chat_payload("test.gguf", m.TEXT, use_default_output_limit=True)}
            return value
        self.assertTrue(m.default_output_limit_gate(row())["passed"])
        for key in ("max_tokens", "max_completion_tokens"):
            for value in (None, 64, 2048):
                bad = row(); bad["request"]["body"][key] = value
                with self.subTest(key=key, value=value):
                    self.assertFalse(m.default_output_limit_gate(bad)["passed"])
        for key, value in (("request", None), ("status", 503), ("request_id_echo", False),
                           ("natural_eos_telemetry", False), ("transport_complete", False)):
            bad = row(); bad[key] = value
            self.assertFalse(m.default_output_limit_gate(bad)["passed"])
        bad = row(); bad["json"]["choices"][0]["message"]["content"] = "43"
        self.assertFalse(m.default_output_limit_gate(bad)["passed"])
        bad = row(); bad["request"]["body"]["messages"][0]["content"] = m.OTHER
        self.assertFalse(m.default_output_limit_gate(bad)["passed"])

    def test_optional_case_cannot_be_missing_duplicate_or_explicitly_capped(self):
        rows = [{"name": name, "quality": {"passed": True}} for name in
                ("text", "image", "concurrent-42", "concurrent-81", "after-cancel", "after-reload")]
        self.assertTrue(m.answer_suite_gate(rows)["passed"])
        self.assertFalse(m.answer_suite_gate(rows, True)["passed"])
        default = good_answer()
        default.update(name="default-output-limit", quality={"passed": True},
                       request={"method": "POST", "route": "/v1/chat/completions",
                                "body": m.chat_payload("test.gguf", m.TEXT, use_default_output_limit=True)})
        rows.append(default)
        self.assertTrue(m.answer_suite_gate(rows, True)["passed"])
        self.assertFalse(m.answer_suite_gate(rows)["passed"])
        self.assertFalse(m.answer_suite_gate(rows + [default], True)["passed"])
        default["request"]["body"]["max_tokens"] = 64
        self.assertFalse(m.answer_suite_gate(rows, True)["passed"])

    def test_default_output_limit_cli_flag_is_opt_in(self):
        argv = ["checker", "--url", "http://localhost", "--model", "test.gguf", "--image", "card.png",
                "--output", "artifacts/test"]
        for enabled in (False, True):
            with patch.object(sys, "argv", argv + (["--default-output-limit-check"] if enabled else [])), patch.object(m, "run", return_value=0) as run:
                self.assertEqual(0, m.main())
                self.assertIs(enabled, run.call_args.args[0].default_output_limit_check)

    def test_ignored_namespace_allows_resolved_junction_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "checkout"
            requested = root / "artifacts" / "fresh"
            target = Path(folder) / "other-volume" / "fresh"
            with patch.object(Path, "resolve", return_value=target):
                self.assertEqual(target, m.evidence_output_path(requested, root))
            with self.assertRaises(ValueError): m.evidence_output_path(root / "src" / "wrong", root)
            with self.assertRaises(ValueError): m.evidence_output_path(root / "artifacts" / ".." / "wrong", root)

    def test_explicit_unified_status_rejects_unknown_or_disabled(self):
        body = {"loaded": "test.gguf", "architecture": "qwen4exp", "visionReady": True}
        m.require_model_status(body, "test.gguf")
        for value in (None, False, "true", 1):
            with self.assertRaises(ValueError): m.require_model_status(dict(body, unifiedMemory=value), "test.gguf", True)
        m.require_model_status(dict(body, unifiedMemory=True), "test.gguf", True)

    def test_exact_answer_and_eos_pass(self):
        self.assertTrue(m.answer_gate(good_answer(), "42")["passed"])

    def test_glm_status_requires_its_explicit_architecture_and_unified_owner(self):
        body = {"loaded": "glm.gguf", "architecture": "glm5-next", "visionReady": True, "unifiedMemory": True}
        m.require_model_status(body, "glm.gguf", True, "glm5-next")
        for architecture in ("qwen4exp", "glm5next", "unknown"):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                m.require_model_status(body, "glm.gguf", True, architecture)

    def test_protocol_stop_without_eos_is_unknown_not_pass(self):
        row = good_answer(); del row["natural_eos_telemetry"]
        self.assertFalse(m.answer_gate(row, "42")["passed"])

    def test_wrong_incomplete_tool_and_error_outputs_rejected(self):
        variants = []
        for key, value in (("status", 503), ("transport_complete", False), ("request_id_echo", False),
                           ("error", "disconnect"), ("cleanup_errors", ["close failed"])):
            row = good_answer(); row[key] = value; variants.append(row)
        row = good_answer("43"); variants.append(row)
        row = good_answer(); row["json"]["choices"][0]["finish_reason"] = "length"; variants.append(row)
        row = good_answer(); row["json"]["choices"][0]["message"]["tool_calls"] = [{"id": "x"}]; variants.append(row)
        row = good_answer(); row["json"]["usage"]["completion_tokens"] = 0; variants.append(row)
        for row in variants:
            with self.subTest(row=row): self.assertFalse(m.answer_gate(row, "42")["passed"])

    def test_unknown_response_never_passes(self):
        for body in (None, {}, {"choices": []}, {"choices": [None]}, {"choices": [1, 2]}):
            row = good_answer(); row["json"] = body
            self.assertFalse(m.answer_gate(row, "42")["passed"])

    def test_ocr_is_exact(self):
        self.assertTrue(m.answer_gate(good_answer("4821"), "4821")["passed"])
        for text in ("4822", "Code: 4821", "4821 4821"):
            self.assertFalse(m.answer_gate(good_answer(text), "4821")["passed"])

    def test_queue_requires_complete_consistent_counters(self):
        idle = {"busy": False, "processing": 0, "pending_requests": 0, "total_processed": 1}
        self.assertTrue(m.queue_state(idle))
        for changes in ({"processing": True}, {"processing": -1}, {"busy": True}, {"pending_requests": None}):
            with self.assertRaises(ValueError): m.queue_state(dict(idle, **changes))
        with self.assertRaises(ValueError): m.queue_state({})
        self.assertFalse(m.queue_state(dict(idle, pending_requests=1)))

    def test_overlap_requires_real_finite_intersection(self):
        rows = [{"started_monotonic": 1., "ended_monotonic": 3.}, {"started_monotonic": 2., "ended_monotonic": 4.}]
        self.assertTrue(m.overlap_gate(rows)["passed"])
        rows[1]["started_monotonic"] = 3.
        self.assertFalse(m.overlap_gate(rows)["passed"])
        rows[1]["started_monotonic"] = float("nan")
        self.assertFalse(m.overlap_gate(rows)["passed"])

    def test_json_logs_bind_scope_not_other_request_or_properties(self):
        records = [{"message": "chat.complete finishReason=eos", "scopes": {"RequestId": "abc"}},
                   {"message": "chat.cancelled because request aborted", "scopes": [{"RequestId": "abc"}]},
                   {"message": "chat.complete finishReason=eos", "scopes": {"RequestId": "abcd"}}]
        result = m.log_evidence("\n".join(json.dumps(r) for r in records), "abc")
        self.assertTrue(result["natural_eos"] and result["cancelled"])
        self.assertEqual(2, len(result["lines"]))
        self.assertFalse(m.log_evidence(json.dumps({"message": "chat.complete finishReason=eos", "properties": {"RequestId": "abc"}}), "abc")["natural_eos"])

    def test_actual_singular_scope_schema_binds_eos_and_cancellation(self):
        # Matches LogEntryFormatter's production JSONL schema, including the
        # nested HTTP RequestId overriding ASP.NET's earlier trace RequestId.
        eos = '{"message":"chat.complete tokens=3 finishReason=eos assistantOutput=\\"42\\"","scope":{"RequestId":"aspnet-trace","Client":"openai","RequestId":"abc"}}'
        cancel = '{"message":"chat.cancelled tokens=5 promptTokens=36","scope":{"RequestId":"aspnet-trace","RequestId":"cancel-id"}}'
        text = eos + "\n" + cancel
        self.assertEqual({"natural_eos": True, "cancelled": False, "lines": [eos]}, m.log_evidence(text, "abc"))
        self.assertEqual({"natural_eos": False, "cancelled": True, "lines": [cancel]}, m.log_evidence(text, "cancel-id"))
        for request_id in ("aspnet-trace", "abcd", "cancel", "missing"):
            self.assertEqual({"natural_eos": False, "cancelled": False, "lines": []}, m.log_evidence(text, request_id))

    def test_canonical_scope_cannot_be_overridden_by_legacy_or_properties(self):
        for canonical in (None, {}, {"RequestId": "other"}):
            record = {"message": "chat.complete finishReason=eos", "scope": canonical,
                      "scopes": {"RequestId": "abc"}, "props": {"RequestId": "abc"}}
            self.assertFalse(m.log_evidence(json.dumps(record), "abc")["natural_eos"])
        for message in ("chat.complete finishReason=length", "chat.complete finishReason=stop",
                        "User said chat.complete finishReason=eos", "HTTP user quoted chat.cancelled"):
            record = {"message": message, "scope": {"RequestId": "abc"}}
            result = m.log_evidence(json.dumps(record), "abc")
            self.assertFalse(result["natural_eos"] or result["cancelled"])

    def test_duplicate_singular_request_id_uses_innermost_not_earlier_value(self):
        record = '{"message":"chat.complete finishReason=eos","scope":{"RequestId":"abc","RequestId":"other"}}'
        self.assertFalse(m.log_evidence(record, "abc")["natural_eos"])
        self.assertTrue(m.log_evidence(record, "other")["natural_eos"])

    def test_text_logs_bind_exact_id_and_actual_finish_reason(self):
        text = "12:00 INF Cat: chat.complete finishReason=eos => RequestId=abc Client=x\n12:01 INF Cat: HTTP POST /api/generate threw OperationCanceledException => RequestId=abc\n12:01 INF Cat: chat.complete finishReason=eos => RequestId=abcd"
        result = m.log_evidence(text, "abc")
        self.assertTrue(result["natural_eos"] and result["cancelled"])
        self.assertEqual(2, len(result["lines"]))
        self.assertFalse(m.log_evidence("Cat: chat.complete finishReason=length => RequestId=abc", "abc")["natural_eos"])

    def test_disconnect_and_empty_queue_are_not_cancellation_proof(self):
        row = {"status": 200, "request_id_echo": True, "content_events": 3, "client_closed": True,
               "terminal_before_close": False, "busy_before_close": True}
        self.assertTrue(m.cancellation_gate(row, True, True)["passed"])
        self.assertFalse(m.cancellation_gate(row, True, False)["passed"])
        self.assertFalse(m.cancellation_gate(row, False, True)["passed"])
        row["terminal_before_close"] = True
        self.assertFalse(m.cancellation_gate(row, True, True)["passed"])

    def test_timeout_is_finite_positive_and_bounded(self):
        self.assertEqual(300, m.finite_timeout("300"))
        for value in ("nan", "inf", -1, 0, 301):
            with self.assertRaises(ValueError): m.finite_timeout(value)

    def test_deadline_unblocks_owned_socket_read(self):
        left, right = socket.socketpair()
        class Response:
            status = 200
            def getheaders(self): return []
            def getheader(self, name): return None
            def read(self, count): return left.recv(count)
            def close(self): left.close()
        class Connection:
            def __init__(self, *args, **kwargs): self.sock = left
            def connect(self): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self): return Response()
            def close(self): left.close()
        try:
            with tempfile.TemporaryDirectory() as folder, patch.object(m.http.client, "HTTPConnection", Connection):
                start = time.monotonic()
                row = m.Client("http://localhost", folder, .1).call("deadline", "/test")
                self.assertLess(time.monotonic() - start, 2)
                self.assertFalse(row["transport_complete"])
                self.assertTrue(row.get("error"))
                self.assertEqual(-1, left.fileno())
                # Either the per-socket deadline or absolute timer can win.
                self.assertTrue(row["timed_out"] or "TimeoutError" in row["error"])
        finally:
            left.close(); right.close()

    def test_stream_closes_after_exactly_three_nonempty_content_events(self):
        left, right = socket.socketpair()
        events = iter(json.dumps(value).encode() + b"\n" for value in (
            {"response": "", "done": False}, {"response": "1", "done": False},
            {"response": "\n", "done": False}, {"response": "2", "done": False},
            {"response": "3", "done": False}))
        class Response:
            status = 200
            def getheaders(self): return []
            def getheader(self, name): return None
            def readline(self, count): return next(events)
            def close(self): left.close()
        class Connection:
            def __init__(self, *args, **kwargs): self.sock = left
            def connect(self): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self): return Response()
            def close(self): left.close()
        try:
            with tempfile.TemporaryDirectory() as folder, patch.object(m.http.client, "HTTPConnection", Connection):
                row = m.Client("http://localhost", folder, 1).call("cancel", "/test", {}, cancel_after_three=True, on_third=lambda: True)
                self.assertEqual(3, row["content_events"])
                self.assertEqual(4, len(row["events"]))
                self.assertTrue(row["client_closed"] and row["busy_before_close"])
                self.assertFalse(row["terminal_before_close"])
                self.assertNotIn("error", row)
                self.assertEqual([], row["cleanup_errors"])
                self.assertEqual(-1, left.fileno())
        finally:
            left.close(); right.close()


if __name__ == "__main__":
    unittest.main()
