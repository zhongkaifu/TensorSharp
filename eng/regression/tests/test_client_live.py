"""Synthetic client captures and raw controls; no client/server/model is launched."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import client_live, runner, scenarios

REPO = Path(__file__).resolve().parents[3]
MODEL = "fixture-model"
ENDPOINT = "http://127.0.0.1:12345/v1/"


def captured(client="extensions", mode="stream"):
    return {"exit_code": 0, "timed_out": False, "cancelled": False, "output_truncated": False,
            "teardown": {"attempted": True, "verified": True}, "stage": "synthetic",
            "stdout": "4\n12\n",
            "stderr": f"TensorSharp HTTP chat via {client}; mode={mode}; model={MODEL}; endpoint={ENDPOINT}\n"}


class FakeHttp:
    base = "http://127.0.0.1:12345"
    def __init__(self):
        self.calls = []
        self.response = {"object": "chat.completion", "model": MODEL,
            "choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        self.streaming = {"content": "4", "done": True, "finish_reason": "stop",
                          "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}, "chunks": []}
    def json(self, route, payload):
        self.calls.append((route, payload))
        return copy.deepcopy(self.response)
    def stream(self, route, payload):
        self.calls.append((route, payload))
        return copy.deepcopy(self.streaming)
    def remaining(self):
        return 90.25


class ClientLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = client_live.load_oracle(REPO)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.binary = self.root / "synthetic-client.dll"
        self.binary.write_bytes(b"never executed")
        self.context = {"repo": REPO, "artifact_dir": self.root, "http_client_command": ["dotnet", str(self.binary)],
                        "env": {"CUDA_VISIBLE_DEVICES": "0"}, "timeout_seconds": 600}
        self.result = {"assertions": [], "inputs": {}, "evidence": {}, "processes": []}

    def validate(self, item):
        return client_live.validate_client_output(item, client="extensions", mode="stream", model=MODEL, endpoint=ENDPOINT, oracle=self.oracle)

    def test_strict_two_answer_lines_accept_only_semantic_answers(self):
        for text in ("4\n12\n", " 4.\n 12. \n"):
            item = captured()
            item["stdout"] = text
            self.assertEqual(len(self.validate(item)), 2)
        for text in ("4\n13\n", "Answer: 4\n12", "5, not 4\n12", "4\n12\nlog", "Log says 4\n12", "4 12", "4\n120", "4\n12\n12"):
            item = captured()
            item["stdout"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.validate(item)

    def test_exact_identity_header_and_no_stderr_warnings(self):
        for text in ("", captured()["stderr"].replace("fixture-model", "other-model"),
                     captured()["stderr"].replace("mode=stream", "mode=response"),
                     captured()["stderr"].replace("12345", "12346"),
                     captured()["stderr"] + "Generation reached the token limit.\n",
                     captured()["stderr"] * 2):
            item = captured()
            item["stderr"] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.validate(item)

    def test_process_failure_timeout_truncation_and_containment_never_pass(self):
        for change in ({"exit_code": 1}, {"timed_out": True}, {"cancelled": True},
                       {"output_truncated": True}, {"teardown": {}},
                       {"teardown": {"attempted": True, "verified": False}}):
            item = captured()
            item.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(item)

    def test_endpoint_cannot_be_redirected_to_other_local_or_remote_service(self):
        self.assertEqual(client_live.local_endpoint("http://127.0.0.1:12345"), ENDPOINT)
        for value in ("http://example.com:12345", "http://127.0.0.1:12345/other", "http://user:pass@127.0.0.1:12345", "http://127.0.0.1:12345?x=1", "https://127.0.0.1:12345", "http://127.0.0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client_live.local_endpoint(value)

    def test_all_four_sdk_mode_invocations_use_owned_process_and_semantic_checks(self):
        http = FakeHttp()
        def execute(argv, **kwargs):
            return captured(argv[argv.index("--client") + 1], argv[argv.index("--mode") + 1])
        with patch.object(client_live, "run_process", side_effect=execute) as process:
            client_live.run_http_client(http, MODEL, self.context, self.result)
        self.assertEqual(process.call_count, 4)
        self.assertEqual(len(self.result["processes"]), 4)
        self.assertEqual(len(http.calls), 2)
        self.assertTrue(all(row["passed"] for row in self.result["evidence"]["http_client"]["invocations"]))
        seen = set()
        for call in process.call_args_list:
            argv = call.args[0]
            seen.add((argv[argv.index("--client") + 1], argv[argv.index("--mode") + 1]))
            self.assertEqual(argv[argv.index("--base-url") + 1], ENDPOINT)
            self.assertEqual(argv[argv.index("--follow-up") + 1], client_live.FOLLOW_UP)
            self.assertNotIn("--seed", argv)
            self.assertNotIn("--temperature", argv)
            self.assertEqual(call.kwargs["timeout_seconds"], 90.25)
            self.assertEqual(call.kwargs["env"]["HTTP_PROXY"], "")
            self.assertEqual(call.kwargs["env"]["NO_PROXY"], "127.0.0.1,localhost,::1")
        self.assertEqual(seen, {(client, mode) for client in ("extensions", "openai") for mode in ("stream", "response")})

    def test_failed_first_client_stops_later_invocations_and_keeps_evidence(self):
        item = captured()
        item["teardown"] = {"attempted": True, "verified": False}
        item["containment_failure"] = True
        with patch.object(client_live, "run_process", return_value=item) as process, self.assertRaises(ValueError):
            client_live.run_http_client(FakeHttp(), MODEL, self.context, self.result)
        self.assertEqual(process.call_count, 1)
        self.assertTrue(self.result["processes"][0]["containment_failure"])
        self.assertFalse(self.result["evidence"]["http_client"]["invocations"][0]["passed"])

    def test_bad_raw_controls_prevent_sdk_launches(self):
        for change in ({"content": "5"}, {"done": False}, {"finish_reason": "length"},
                       {"usage": {"prompt_tokens": 1, "completion_tokens": 65, "total_tokens": 66}},
                       {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 99}}):
            http = FakeHttp()
            http.streaming.update(change)
            with self.subTest(change=change), patch.object(client_live, "run_process") as process, self.assertRaises(ValueError):
                client_live.run_http_client(http, MODEL, self.context, self.result)
            process.assert_not_called()

    def test_raw_complete_model_identity_must_match(self):
        http = FakeHttp()
        http.response["model"] = "other-model"
        with patch.object(client_live, "run_process") as process, self.assertRaises(scenarios.ScenarioFailure):
            client_live.run_http_client(http, MODEL, self.context, self.result)
        process.assert_not_called()

    def test_missing_prebuilt_client_is_a_prerequisite_skip(self):
        context = {"repo": REPO, "server_command": ["not-executed"], "preparing_model": True, "backend": "ggml_cpu"}
        self.assertFalse(scenarios.availability({"id": "client_http_live"}, context)[0])
        context["http_client_command"] = ["not-executed"]
        self.assertTrue(scenarios.availability({"id": "client_http_live"}, context)[0])
        context.pop("server_command")
        self.assertFalse(scenarios.availability({"id": "client_http_live"}, context)[0])

    def test_runner_option_and_reproduce_keep_explicit_client_path(self):
        args = runner.parse_args(["--http-client", str(self.binary)])
        self.assertEqual(args.http_client, self.binary)
        argv = runner.reproduce_args(args, REPO, self.root, {"id": "qwen3.5-0.8b"}, "ggml_cpu", {"id": "client_http_live"})
        self.assertEqual(argv[argv.index("--http-client") + 1], str(self.binary))


if __name__ == "__main__":
    unittest.main()
