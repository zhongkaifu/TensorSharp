"""Exercise incremental HTTP timing, completion integrity, and comparison gates."""
import copy
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("qwen_chat_cache", Path(__file__).parents[1] / "qwen-chat-cache-benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class DelayedSseHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b'data: {"token":"hello"}\n\n')
        self.wfile.flush()
        time.sleep(.25)
        final = {"done": True, "tokenCount": 1, "promptTokens": 5, "kvReusedTokens": 3,
                 "truncated": False, "error": None, "aborted": False}
        self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
        self.wfile.flush()

    def log_message(self, *args):
        pass


class QwenChatCacheBenchmarkTests(unittest.TestCase):
    def test_webui_discovery_uses_loaded_model_and_preserves_host_metadata(self):
        loaded = "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf"
        metadata = {"models": [loaded], "loaded": loaded, "visionReady": True,
                    "loadedMmProj": "mmproj-BF16.gguf", "loadedBackend": "ggml_cuda", "contextTokens": 32768}
        for supplied in (None, loaded):
            with self.subTest(supplied=supplied), TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "artifacts" / "report.json"
                client = Mock()
                def request(method, route):
                    if (method, route) == ("GET", "/api/models"):
                        return copy.deepcopy(metadata)
                    if (method, route) == ("POST", "/api/sessions"):
                        return {"sessionId": "session"}
                    self.assertEqual((method, route), ("DELETE", "/api/sessions/session"))
                    return {}
                client.json.side_effect = request
                def chat(route, body, protocol, row):
                    self.assertEqual((route, protocol), ("/api/chat", "webui"))
                    row.update(answer="answer", done=True, ttft_ms=1, cached_tokens=0,
                               prompt_tokens=5, completion_tokens=1, finish_reason="stop")
                client.chat.side_effect = chat
                argv = ["--url", "http://127.0.0.1:12345", "--model-discovery", "webui",
                        "--cases", "text", "--turns", "1", "--output", str(output)]
                if supplied:
                    argv += ["--model", supplied]
                with patch.object(bench, "ROOT", root), patch.object(bench, "Client", return_value=client), \
                        patch("builtins.print"):
                    self.assertEqual(bench.main(argv), 0)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["model"], loaded)
                self.assertEqual(report["server_models"], [{"id": loaded}])
                self.assertEqual(report["server_webui_models"], metadata)
                self.assertEqual(report["configuration"]["model_discovery"], "webui")
                self.assertEqual(client.json.call_args_list[0].args, ("GET", "/api/models"))
                self.assertFalse(any(call.args[1] == "/v1/models" for call in client.json.call_args_list))
                client.chat.assert_called_once()

    def test_webui_discovery_rejects_missing_or_mismatched_models_before_chat(self):
        cases = [
            ({"models": ["loaded"]}, None, "no loaded model"),
            ({"models": ["loaded"], "loaded": ""}, None, "no loaded model"),
            ({"models": [], "loaded": "loaded"}, None, "not present"),
            ({"models": ["loaded"], "loaded": "loaded"}, "other", "--model must match"),
        ]
        for metadata, supplied, reason in cases:
            with self.subTest(reason=reason, supplied=supplied), TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "artifacts" / "report.json"
                client = Mock()
                client.json.return_value = metadata
                argv = ["--url", "http://127.0.0.1:12345", "--model-discovery", "webui",
                        "--cases", "text", "--turns", "1", "--output", str(output)]
                if supplied:
                    argv += ["--model", supplied]
                with patch.object(bench, "ROOT", root), patch.object(bench, "Client", return_value=client), \
                        patch("builtins.print"):
                    self.assertEqual(bench.main(argv), 1)
                client.json.assert_called_once_with("GET", "/api/models")
                client.chat.assert_not_called()
                client.upload.assert_not_called()
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["runs"], [])
                self.assertIn(reason, " ".join(report["failures"]))

    def test_webui_discovery_rejects_openai_protocol_before_creating_client(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(bench, "ROOT", root), patch.object(bench, "Client") as client:
                with self.assertRaisesRegex(ValueError, "requires --protocol webui"):
                    bench.main(["--url", "http://127.0.0.1:12345", "--model-discovery", "webui",
                                "--protocol", "openai", "--cases", "text",
                                "--output", str(root / "artifacts" / "report.json")])
                client.assert_not_called()

    def test_connection_file_authenticates_json_sse_and_upload_without_leaking_cookie(self):
        class Response(io.BytesIO):
            status = 200

            def __init__(self, payload, content_type="application/json"):
                super().__init__((payload if isinstance(payload, str) else json.dumps(payload)).encode())
                self.content_type = content_type

            def getheader(self, name, default=None):
                return self.content_type if name.lower() == "content-type" else default

        cookie = "TensorAgentAuth=test-only-authentication-value"
        final = {"done": True, "tokenCount": 1, "promptTokens": 5,
                 "kvReusedTokens": 0, "truncated": False}
        connection = Mock()
        connection.getresponse.side_effect = [
            Response({"data": [{"id": "qwen"}]}),
            Response({"ok": True, "file": "uploaded.jpg"}),
            Response({"sessionId": "session"}),
            Response('data: {"token":"answer"}\n\ndata: ' + json.dumps(final) + '\n\n', "text/event-stream"),
            Response({}),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"test-image-payload")
            connection_file = root / "connection.json"
            # The host can write a UTF-8 BOM, and its URL can have a trailing slash.
            connection_file.write_text(json.dumps({"baseUrl": "http://127.0.0.1:12345/", "cookie": cookie}),
                                       encoding="utf-8-sig")
            output = root / "artifacts" / "result.json"
            with patch.object(bench, "ROOT", root), patch.object(bench.http.client, "HTTPConnection", return_value=connection), \
                    patch("builtins.print"):
                result = bench.main(["--url", "http://127.0.0.1:12345", "--connection-file", str(connection_file),
                                     "--cases", "image", "--image", str(image), "--turns", "1", "--output", str(output)])
            self.assertEqual(result, 0)
            requests = connection.request.call_args_list
            self.assertEqual([call.args[1] for call in requests],
                             ["/v1/models", "/api/upload", "/api/sessions", "/api/chat", "/api/sessions/session"])
            self.assertTrue(all(call.args[3]["Cookie"] == cookie for call in requests))
            self.assertEqual(requests[0].args[3]["Content-Type"], "application/json")
            self.assertTrue(requests[1].args[3]["Content-Type"].startswith("multipart/form-data; boundary="))
            self.assertIn(image.read_bytes(), requests[1].args[2])
            self.assertEqual(requests[3].args[3]["Content-Type"], "application/json")
            self.assertEqual(connection.close.call_count, len(requests))
            report_text = output.read_text(encoding="utf-8")
            report = json.loads(report_text)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["runs"][0]["answer"], "answer")
            self.assertNotIn(cookie, report_text)
            self.assertNotIn(str(connection_file), report_text)
            self.assertNotIn("cookie", report["configuration"])
            self.assertNotIn("connection_file", report["configuration"])

    def test_untrusted_connection_file_never_opens_http_and_does_not_record_its_cookie(self):
        invalid = [
            ("http://127.0.0.1:12346", "test-only-cookie", "baseUrl"),
            ("http://127.0.0.1:12345", "test-only-cookie\rInjected: value", "Invalid connection cookie"),
            ("http://127.0.0.1:12345", "test-only-cookie\nInjected: value", "Invalid connection cookie"),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            connection_file = root / "connection.json"
            output = root / "artifacts" / "result.json"
            for base_url, cookie, reason in invalid:
                with self.subTest(base_url=base_url, reason=reason):
                    connection_file.write_text(json.dumps({"baseUrl": base_url, "cookie": cookie}), encoding="utf-8")
                    with patch.object(bench, "ROOT", root), patch.object(bench.http.client, "HTTPConnection") as http, \
                            patch.object(bench.http.client, "HTTPSConnection") as https, patch("builtins.print"):
                        result = bench.main(["--url", "http://127.0.0.1:12345", "--connection-file", str(connection_file),
                                             "--cases", "text", "--turns", "1", "--output", str(output)])
                    self.assertEqual(result, 1)
                    http.assert_not_called()
                    https.assert_not_called()
                    report_text = output.read_text(encoding="utf-8")
                    report = json.loads(report_text)
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["runs"], [])
                    self.assertIn(reason, " ".join(report["failures"]))
                    self.assertNotIn("test-only-cookie", report_text)

    def test_history_replay_uses_baseline_answers_with_current_session_and_image(self):
        class RecordingClient:
            def __init__(self):
                self.bodies = []

            def json(self, method, route):
                return {"sessionId": "new-session"} if method == "POST" else {}

            def chat(self, route, body, protocol, row):
                self.bodies.append(copy.deepcopy(body))
                row.update(answer=f"different candidate {len(self.bodies)}", done=True,
                           ttft_ms=1, cached_tokens=0, prompt_tokens=20, completion_tokens=3,
                           finish_reason="length")

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(model="qwen", protocol="webui", thinking=False, max_tokens=64,
                text_prompt="text", continuation_prompt="continue", turns=3, repeats=1,
                branches=True, invalidation=True, cases=["image"], require_reuse=False,
                require_full_reuse=False, output=Path(directory) / "candidate.json",
                replay_history_from=Path(directory) / "baseline.json")
            first = {"role": "user", "content": bench.IMAGE_PROMPT, "imagePaths": ["old-upload.jpg"]}
            baseline = {"model": "qwen", "protocol": "webui", "label": "baseline", "runs": []}
            history = [first]
            initial = None
            for index in range(3):
                if index:
                    history.append({"role": "user", "content": args.continuation_prompt})
                canonical = bench.canonical_request(bench.make_body(args, history, "old-session"), "image-digest")
                baseline["runs"].append({"scenario": "image", "repetition": 1, "turn": f"turn-{index + 1}",
                    "status": "passed", "done": True, "answer": f"baseline answer {index + 1}",
                    "request_sha256": bench.digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode())})
                history.append({"role": "assistant", "content": f"baseline answer {index + 1}"})
                if index == 0:
                    initial = copy.deepcopy(history)
            for turn, extra in (("branch", initial + [{"role": "user", "content": "请补充说明刚才提到的细节。"}]),
                                ("invalidation", [{"role": "user", "content": bench.INVALIDATION_PROMPT}])):
                canonical = bench.canonical_request(bench.make_body(args, extra, "old-session"), "image-digest")
                baseline["runs"].append({"scenario": "image", "repetition": 1, "turn": turn,
                    "status": "passed", "done": True, "answer": "579" if turn == "invalidation" else "baseline branch",
                    "request_sha256": bench.digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode())})
            args.replay_history_from.write_text(json.dumps(baseline), encoding="utf-8")
            report = {"model": "qwen", "runs": [], "failures": [], "image": {"sha256": "image-digest"}}
            bench.load_replay_history(args, report)
            client = RecordingClient()
            # Invalidation tests the arithmetic invariant separately; return its exact answer here.
            original_chat = client.chat
            def chat(route, body, protocol, row):
                original_chat(route, body, protocol, row)
                if row["turn"] == "invalidation":
                    row["answer"] = "579"
            client.chat = chat
            current = dict(first, imagePaths=["new-upload.jpg"])
            bench.run_workflow(args, client, "image", 1, current, report)
            self.assertFalse(report["failures"])
            self.assertEqual(len(client.bodies), 5)
            self.assertEqual(client.bodies[2]["messages"][1]["content"], "baseline answer 1")
            self.assertEqual(client.bodies[2]["messages"][3]["content"], "baseline answer 2")
            self.assertEqual(client.bodies[3]["messages"][1]["content"], "baseline answer 1")
            self.assertEqual(client.bodies[2]["messages"][0]["imagePaths"], ["new-upload.jpg"])
            self.assertTrue(all(body["sessionId"] == "new-session" for body in client.bodies))
            self.assertFalse(report["runs"][0]["answer_matches_replay"])

            # The default still appends the candidate's actual generated answers.
            del args._replay_rows
            args.branches = args.invalidation = False
            fresh = RecordingClient()
            bench.run_workflow(args, fresh, "image", 1, current,
                               {"runs": [], "failures": [], "image": {"sha256": "image-digest"}})
            self.assertEqual(fresh.bodies[2]["messages"][1]["content"], "different candidate 1")
            self.assertEqual(fresh.bodies[2]["messages"][3]["content"], "different candidate 2")

    def test_history_replay_rejects_missing_failed_or_duplicate_baseline_requests(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            args = SimpleNamespace(replay_history_from=path, protocol="webui", turns=1,
                branches=False, invalidation=False, cases=["text"], repeats=1)
            valid = {"scenario": "text", "repetition": 1, "turn": "turn-1", "status": "passed",
                     "done": True, "answer": "answer", "request_sha256": "hash"}
            for rows in ([], [dict(valid, done=False)], [valid, valid]):
                path.write_text(json.dumps({"model": "qwen", "protocol": "webui", "runs": rows}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    bench.load_replay_history(args, {"model": "qwen"})

    def test_webui_uses_loaded_model_and_openai_names_the_model(self):
        args = SimpleNamespace(model="qwen", thinking=False, protocol="webui", max_tokens=64)
        web = bench.make_body(args, [{"role": "user", "content": "hello"}], session="session")
        self.assertNotIn("model", web)
        self.assertNotIn("backend", web)
        self.assertEqual(web["sessionId"], "session")
        self.assertEqual(web["maxTokens"], 64)
        args.protocol = "openai"
        api = bench.make_body(args, [], session="ignored")
        self.assertEqual(api["model"], "qwen")
        self.assertNotIn("sessionId", api)
        self.assertEqual(api["max_tokens"], 64)

    def test_http_ttft_observes_incremental_delta_before_completion(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), DelayedSseHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = bench.Client(f"http://127.0.0.1:{server.server_port}", 5)
            evidence = {}
            message = client.chat("/api/chat", {"messages": []}, "webui", evidence)
            self.assertEqual(message["content"], "hello")
            self.assertGreater(evidence["elapsed_ms"] - evidence["ttft_ms"], 180)
            self.assertEqual(evidence["cached_tokens"], 3)
            self.assertEqual(evidence["prefilled_tokens"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_openai_reasoning_is_ttft_and_answer_latency_is_separate(self):
        state = bench.StreamState("openai")
        state.add({"choices": [{"delta": {"role": "assistant"}}]}, 2)
        state.add({"choices": [{"delta": {"reasoning_content": "consider"}}]}, 5)
        state.add({"choices": [{"delta": {"content": "42"}}]}, 15)
        state.add({"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 8, "completion_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": 7}}}, 20)
        state.add("[DONE]", 22)
        state.validate()
        self.assertEqual(state.result()["ttft_ms"], 5)
        self.assertEqual(state.result()["first_answer_ms"], 15)
        self.assertEqual(state.message(), {"role": "assistant", "content": "42"})

    def test_webui_replace_sets_the_answer_that_is_sent_back(self):
        # Text the page showed and then took back as reasoning (Nemotron-H Reasoning-128K
        # closing a block its thinking-off prompt had closed) arrives as its reasoning and a
        # whole-answer replace. The next turn sends back what remains, or the server's
        # transcript no longer recognises the turn and the benchmark measures a re-prefill.
        state = bench.StreamState("webui")
        state.add({"token": "Okay, the user asks."}, 5)
        state.add({"thinking": "Okay, the user asks."}, 8)
        state.add({"replace": ""}, 8)
        state.add({"token": "Paris."}, 12)
        state.add({"done": True, "tokenCount": 6, "promptTokens": 8, "kvReusedTokens": 0}, 15)
        state.validate()
        self.assertEqual(state.message(), {"role": "assistant", "content": "Paris."})
        self.assertEqual(state.result()["reasoning"], "Okay, the user asks.")

    def test_unfinished_or_failed_streams_never_pass_validation(self):
        for final in (None, {"done": True, "tokenCount": 2, "promptTokens": 3},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 4},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 0, "aborted": True},
                      {"done": True, "tokenCount": 2, "promptTokens": 3, "kvReusedTokens": 0, "error": "OOM"}):
            with self.subTest(final=final):
                state = bench.StreamState("webui")
                state.add({"token": "answer"}, 10)
                if final:
                    state.add(final, 15)
                with self.assertRaises(ValueError):
                    state.validate()

    def test_canonical_history_retains_answer_text_and_compares_image_identity(self):
        body = {"sessionId": "one", "messages": [{"role": "user", "content": "describe", "imagePaths": ["a.jpg"]},
            {"role": "assistant", "content": "actual first answer"}, {"role": "user", "content": "continue"}]}
        candidate = copy.deepcopy(body)
        candidate["sessionId"] = "two"
        candidate["messages"][0]["imagePaths"] = ["b.jpg"]
        self.assertEqual(bench.canonical_request(body, "digest"), bench.canonical_request(candidate, "digest"))
        self.assertEqual(body["messages"][0]["imagePaths"], ["a.jpg"])
        candidate["messages"][1]["content"] = "changed first answer"
        self.assertNotEqual(bench.canonical_request(body, "digest"), bench.canonical_request(candidate, "digest"))

    def test_openai_evidence_does_not_duplicate_base64_image_bytes(self):
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]}
        canonical = bench.canonical_request(body, "digest")
        self.assertEqual(canonical["messages"][0]["content"][0]["image_url"]["url"], "sha256:digest")
        self.assertIn("base64", body["messages"][0]["content"][0]["image_url"]["url"])

    def test_changed_history_or_incomplete_streams_do_not_qualify_speedup(self):
        old = {"scenario": "image", "repetition": 1, "turn": "turn-2", "request_sha256": "history",
               "answer_sha256": "answer", "done": True, "ttft_ms": 100, "completion_tokens": 20,
               "finish_reason": "length", "cached_tokens": 0}
        new = dict(old, ttft_ms=10, cached_tokens=15)
        report = bench.compare_reports({"runs": [old]}, {"runs": [new]})
        self.assertEqual(report["pairs"][0]["ttft_speedup"], 10)
        for changed in (dict(new, request_sha256="different-history"), dict(new, done=False), dict(new, error="OOM")):
            report = bench.compare_reports({"runs": [old]}, {"runs": [changed]})
            self.assertIsNone(report["pairs"][0]["ttft_speedup"])
        for baseline, candidate in (({"model": "first", "runs": [old]}, {"model": "other", "runs": [new]}),
                                    ({"runs": [old, old]}, {"runs": [new, new]}),
                                    ({"runs": [old]}, {"runs": [dict(new, done=False)]})):
            report = bench.compare_reports(baseline, candidate)
            self.assertFalse(report["performance_qualified"])


if __name__ == "__main__":
    unittest.main()
