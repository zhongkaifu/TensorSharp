import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "validate-qwen38-tool-calls.py"
spec = importlib.util.spec_from_file_location("qwen38_tool_calls", SOURCE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

USAGE = {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
CALL_ID = "call_actual-server-id_42"


def tool_message(scenario="string_payload", arguments=None):
    scenario_spec = harness.SPECS[scenario]
    return {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": CALL_ID, "type": "function", "function": {
                "name": scenario_spec["name"],
                "arguments": json.dumps(arguments if arguments is not None else scenario_spec["expected"]),
            },
        }],
    }


def json_response(message, finish="tool_calls", usage=USAGE):
    return 200, "application/json", json.dumps({
        "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage,
    }).encode()


def sse_response(events, done=True):
    payload = b": heartbeat\n\n" + b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )
    if done:
        payload += b"data: [DONE]\n\n"
    return 200, "text/event-stream", payload


def delta_event(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def streamed_message(message, finish="tool_calls", done=True):
    events = [delta_event({"role": "assistant", "reasoning_content": "Plan "}),
              delta_event({"reasoning_content": "complete."})]
    for index, call in enumerate(message.get("tool_calls", [])):
        name = call["function"]["name"]
        arguments = call["function"]["arguments"]
        events.append(delta_event({"tool_calls": [{
            "index": index, "id": call["id"], "type": "function",
            "function": {"name": name[:3], "arguments": ""},
        }]}))
        events.append(delta_event({"tool_calls": [{
            "index": index, "function": {"name": name[3:]},
        }]}))
        # Split at every character, including JSON escapes and marker strings.
        for character in arguments:
            events.append(delta_event({"tool_calls": [{
                "index": index, "function": {"arguments": character},
            }]}))
    for character in message.get("content", ""):
        events.append(delta_event({"content": character}))
    events.extend([delta_event({}, finish), {"choices": [], "usage": USAGE}])
    return sse_response(events, done)


class Qwen38ToolCallHarnessTests(unittest.TestCase):
    def serve(self, responder):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.respond(body)

            def do_GET(self):
                self.respond(None)

            def respond(self, body):
                requests.append({"method": self.command, "path": self.path, "body": body})
                status, content_type, response = responder(body, len(requests))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        args = SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", model="fixture-model", timeout=2)
        return args, requests

    def test_model_discovery_returns_the_advertised_id_verbatim(self):
        model_id = "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003"
        args, requests = self.serve(lambda *_: (
            200, "application/json", json.dumps({"data": [{"id": model_id}]}).encode()))
        self.assertEqual(harness.hosted_model(args.url + "/proxy/", args.timeout), model_id)
        self.assertEqual(requests, [{"method": "GET", "path": "/proxy/v1/models", "body": None}])

    def test_model_discovery_requires_explicit_selection_without_one_usable_id(self):
        for models in ([], [{"id": "one"}, {"id": "two"}], [{}], [{"id": ""}]):
            with self.subTest(models=models):
                args, _ = self.serve(lambda *_: (
                    200, "application/json", json.dumps({"data": models}).encode()))
                with self.assertRaisesRegex(ValueError, "Specify --model"):
                    harness.hosted_model(args.url, args.timeout)

    def test_model_discovery_rejects_http_errors(self):
        args, _ = self.serve(lambda *_: (503, "text/plain", b"not ready"))
        with self.assertRaisesRegex(ValueError, "Model discovery returned HTTP 503"):
            harness.hosted_model(args.url, args.timeout)

    def roundtrip_responder(self, message):
        def respond(body, number):
            if number == 1:
                output, finish = message, "tool_calls"
            else:
                tool_result = next(item for item in body["messages"] if item["role"] == "tool")
                output = {"role": "assistant", "content": json.loads(tool_result["content"])["receipt"]}
                finish = "stop"
            return streamed_message(output, finish) if body["stream"] else json_response(output, finish)
        return respond

    def test_roundtrip_checks_all_payloads_and_replays_original_call_id(self):
        for scenario in harness.SPECS:
            for stream in (False, True):
                for thinking in (False, True):
                    with self.subTest(scenario=scenario, stream=stream, thinking=thinking):
                        original = tool_message(scenario)
                        args, requests = self.serve(self.roundtrip_responder(original))
                        result = harness.run_case(args, scenario, stream, thinking)
                        self.assertEqual(result["status"], "ok", result.get("error"))
                        self.assertEqual(result["arguments"], harness.SPECS[scenario]["expected"])
                        self.assertEqual(len(requests), 2)
                        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
                        self.assertEqual(requests[0]["body"]["think"], thinking)
                        replay = requests[1]["body"]
                        self.assertEqual(replay["tool_choice"], "none")
                        self.assertEqual(replay["messages"][1]["tool_calls"], original["tool_calls"])
                        self.assertEqual(replay["messages"][2]["tool_call_id"], CALL_ID)
                        if stream:
                            self.assertTrue(result["turns"][0]["done"])
                            self.assertEqual(replay["messages"][1]["reasoning_content"], "Plan complete.")

    def test_stream_assembly_keeps_parallel_call_indexes_separate(self):
        events = [delta_event({"tool_calls": [
            {"index": 1, "id": "second", "function": {"name": "two", "arguments": '{"b":'}},
            {"index": 0, "id": "first", "function": {"name": "one", "arguments": '{"a":'}},
        ]}), delta_event({"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}},
            {"index": 1, "function": {"arguments": "2}"}},
        ]}, "tool_calls"), {"choices": [], "usage": USAGE}]
        args, _ = self.serve(lambda *_: sse_response(events))
        message, finish, usage = harness.request(args.url, {"stream": True}, args.timeout, {})
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(usage, USAGE)
        self.assertEqual([call["id"] for call in message["tool_calls"]], ["first", "second"])
        self.assertEqual([json.loads(call["function"]["arguments"]) for call in message["tool_calls"]],
                         [{"a": 1}, {"b": 2}])

    def test_requested_token_budget_and_effort_apply_to_both_tool_turns(self):
        args, requests = self.serve(self.roundtrip_responder(tool_message()))
        args.max_tokens = 768
        args.reasoning_effort = "low"
        result = harness.run_case(args, "string_payload", False, False)
        self.assertEqual(result["status"], "ok")
        for request in requests:
            self.assertEqual(request["body"]["max_tokens"], 768)
            self.assertEqual(request["body"]["reasoning_effort"], "low")

    def test_missing_done_rejects_even_a_complete_call(self):
        args, requests = self.serve(lambda *_: streamed_message(tool_message(), done=False))
        result = harness.run_case(args, "string_payload", True, False)
        self.assertEqual(result["status"], "fail")
        self.assertIn("no [DONE]", result["error"])
        self.assertFalse(result["turns"][0]["done"])
        self.assertEqual(len(requests), 1)

    def test_malformed_and_truncated_json_preserve_failure_evidence(self):
        for stream, wire in ((False, b'{"choices": ['), (True, b'data: {"choices": [\n\n')):
            with self.subTest(stream=stream):
                args, requests = self.serve(lambda *_: (200, "text/plain", wire))
                result = harness.run_case(args, "string_payload", stream, False)
                self.assertEqual(result["status"], "fail")
                self.assertIn("JSONDecodeError", result["error"])
                evidence = result["turns"][0]
                if stream:
                    self.assertEqual(evidence["raw_events"], ['{"choices": ['])
                    self.assertFalse(evidence["done"])
                else:
                    self.assertEqual(evidence["body"], wire.decode())
                self.assertEqual(len(requests), 1)

    def test_argument_corruption_fails_before_tool_result_replay(self):
        corruptions = [
            ("string_payload", {"key": 123, "content": '{"answer":42}'}),
            ("string_payload", {"key": "123", "content": {"answer": 42}}),
            ("code_payload", {"key": "main.py", "content": harness.PAYLOAD.rstrip()}),
            ("code_payload", {"key": "main.py", "content": harness.PAYLOAD.replace("<tool_call></tool_call>", "")}),
        ]
        for scenario, arguments in corruptions:
            for stream in (False, True):
                with self.subTest(scenario=scenario, arguments=arguments, stream=stream):
                    message = tool_message(scenario, arguments)
                    args, requests = self.serve(self.roundtrip_responder(message))
                    result = harness.run_case(args, scenario, stream, False)
                    self.assertEqual(result["status"], "fail")
                    self.assertIn("Arguments differ", result["error"])
                    self.assertEqual(len(requests), 1)

    def test_duplicate_argument_keys_fail_before_tool_result_replay(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                message = copy.deepcopy(tool_message("weather"))
                message["tool_calls"][0]["function"]["arguments"] = '{"city":"wrong","city":"Paris","units":"celsius"}'
                args, requests = self.serve(self.roundtrip_responder(message))
                result = harness.run_case(args, "weather", stream, False)
                self.assertEqual(result["status"], "fail")
                self.assertIn("Duplicate tool argument keys", result["error"])
                self.assertEqual(len(requests), 1)

    def test_invalid_call_metadata_and_incomplete_arguments_fail(self):
        for mutation in ("missing_id", "wrong_name", "truncated_arguments", "wrong_finish", "missing_usage"):
            with self.subTest(mutation=mutation):
                message = copy.deepcopy(tool_message())
                finish, usage = "tool_calls", USAGE
                if mutation == "missing_id":
                    message["tool_calls"][0]["id"] = ""
                elif mutation == "wrong_name":
                    message["tool_calls"][0]["function"]["name"] = "wrong"
                elif mutation == "truncated_arguments":
                    message["tool_calls"][0]["function"]["arguments"] = '{"key": "123"'
                elif mutation == "wrong_finish":
                    finish = "length"
                else:
                    usage = None
                response = json_response(message, finish, usage)
                args, requests = self.serve(lambda *_: response)
                result = harness.run_case(args, "string_payload", False, False)
                self.assertEqual(result["status"], "fail")
                self.assertEqual(len(requests), 1)

    def test_http_and_sse_errors_fail_with_evidence(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = sse_response([{"error": {"message": "model failed"}}]) if stream else (
                    503, "application/json", b'{"error":"model failed"}')
                args, requests = self.serve(lambda *_: response)
                result = harness.run_case(args, "string_payload", stream, False)
                self.assertEqual(result["status"], "fail")
                self.assertIn("model failed", result["error"])
                self.assertEqual(len(requests), 1)

    def test_incorrect_receipt_never_counts_as_a_successful_roundtrip(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                def respond(body, number):
                    message = tool_message() if number == 1 else {"role": "assistant", "content": "receipt-invented"}
                    finish = "tool_calls" if number == 1 else "stop"
                    return streamed_message(message, finish) if body["stream"] else json_response(message, finish)

                args, requests = self.serve(respond)
                result = harness.run_case(args, "string_payload", stream, False)
                self.assertEqual(result["status"], "fail")
                self.assertIn("Tool-result round trip failed", result["error"])
                self.assertEqual(len(requests), 2)


if __name__ == "__main__":
    unittest.main()
