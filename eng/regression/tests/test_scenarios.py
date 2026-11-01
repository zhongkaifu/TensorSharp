"""Model-free negative regressions for runtime adapters and subprocess isolation."""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch
from types import SimpleNamespace

from eng.regression.processes import clean_environment, redact, run_process
from eng.regression.scenarios import (
    ScenarioFailure, ScenarioUntested, _Http, availability, backend_evidence, extract_assistant,
    run_scenario, validate_chat, validate_embeddings, validate_logit_metrics,
    validate_sequence_metrics,
    _history_hashes,
)


def completion(text="56"):
    return {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}


def logit_metrics(comparison=False):
    row = {"finite": True}
    if comparison:
        row.update(relative_l2=0.00001, cosine=0.9999999, kl=0.000001, max_abs=0.0001, top1_equal=True)
    return {"run_id": "fresh", "complete": True, "all_finite": True, "qualified": True if comparison else None,
            "cases_done": 1, "cases_total": 1,
            "cases": [{"name": "prefill", "steps": 16, "metrics": [dict(row, step=i) for i in range(16)]}]}


class ScenarioContracts(unittest.TestCase):
    def test_backend_requires_positive_runtime_banner(self):
        self.assertFalse(backend_evidence("requested backend=ggml_cuda", "ggml_cuda")["verified"])
        self.assertTrue(backend_evidence("Backend: GgmlCuda\n", "ggml_cuda")["verified"])

    def test_backend_fallback_is_never_coverage(self):
        self.assertFalse(backend_evidence("Backend: Cpu\n", "ggml_cuda")["verified"])
        self.assertFalse(backend_evidence("Backend: GgmlCuda\nbackend unavailable, falling back to CPU", "ggml_cuda")["verified"])

    def test_prompt_or_thinking_cannot_satisfy_answer(self):
        text = "info: prompt mentions 56\n=== Generated Output ===\n--- Thinking ---\n56\n--- End Thinking ---\nwrong\ninfo: done"
        self.assertEqual(extract_assistant(text), "wrong")
        self.assertEqual(extract_assistant("prompt 56\nno output marker"), "")

    def test_semantic_and_usage_assertions(self):
        self.assertEqual(validate_chat(completion()), "56")
        for value in ("", "The answer is 55", "Error: 56", "56 is wrong, 55 is right"):
            with self.subTest(value=value), self.assertRaises(ScenarioFailure):
                validate_chat(completion(value))
        value = completion()
        value["usage"]["total_tokens"] = 99
        with self.assertRaises(ScenarioFailure):
            validate_chat(value)

    def test_length_capped_answer_does_not_pass(self):
        value = completion()
        value["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ScenarioFailure):
            validate_chat(value)

    def test_embedding_shapes_nonfinite_and_insensitivity(self):
        value = {"object": "list", "data": [{"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0, 0.0]}, {"index": 2, "embedding": [0.8, 0.6]},
                    {"index": 3, "embedding": [0.0, 1.0]}]}
        self.assertEqual(validate_embeddings(value, 2)["duplicate_cosine"], 1)
        for vector in ([float("nan"), 0], [0, 0], [1, 0], [0, 1, 0]):
            bad = copy.deepcopy(value)
            bad["data"][3]["embedding"] = vector
            with self.subTest(vector=vector), self.assertRaises(ScenarioFailure):
                validate_embeddings(bad, 2)

    def test_embedding_semantic_ordering_not_just_difference(self):
        value = {"object": "list", "data": [{"index": i, "embedding": vector} for i, vector in enumerate(
            ([1, 0], [1, 0], [0, 1], [0.8, 0.6]))]}
        with self.assertRaisesRegex(ScenarioFailure, "relevance"):
            validate_embeddings(value, 2)

    def test_stale_incomplete_and_nonfinite_logit_metrics(self):
        value = logit_metrics()
        self.assertIsNone(validate_logit_metrics(value, "fresh", comparison=False))
        for key, actual in (("run_id", "stale"), ("complete", False), ("all_finite", False), ("cases_done", 0)):
            bad = copy.deepcopy(value)
            bad[key] = actual
            with self.subTest(key=key), self.assertRaises(ScenarioFailure):
                validate_logit_metrics(bad, "fresh", comparison=False)

    def test_logit_first_divergence_not_summary_only(self):
        value = logit_metrics(True)
        value["cases"][0]["metrics"][7]["relative_l2"] = 0.2
        self.assertEqual(validate_logit_metrics(value, "fresh", comparison=True)["step"], 7)
        value["cases"][0]["metrics"][3]["cosine"] = None
        self.assertEqual(validate_logit_metrics(value, "fresh", comparison=True)["step"], 3)

    def test_missing_reference_rows_rejected(self):
        value = logit_metrics(True)
        value["cases"][0]["metrics"].pop()
        with self.assertRaises(ScenarioFailure):
            validate_logit_metrics(value, "fresh", comparison=True)

    def sequence_metrics(self, order):
        schedule = [{"name": f"{name}-{i}", "reference_case": name} for i, name in enumerate(order)]
        value = logit_metrics(True)
        template = value["cases"][0]
        template["metrics"] = [dict(template["metrics"][0], step=i) for i in range(32)]
        value["cases"] = [dict(copy.deepcopy(template), name=case["name"], reference_case=case["reference_case"],
                               reference_scope="external", reset_kv_cache=True, session_case_index=i, steps=32,
                               prompt_tokens=32 if case["reference_case"] == "short" else 512)
                          for i, case in enumerate(schedule)]
        value["cases_done"] = value["cases_total"] = len(schedule)
        return value, schedule

    def test_length_order_divergence_fails_when_same_length_passes(self):
        stable, stable_schedule = self.sequence_metrics(["short", "short"])
        self.assertIsNone(validate_sequence_metrics(stable, stable_schedule, "fresh"))
        changed, changed_schedule = self.sequence_metrics(["short", "long", "short"])
        changed["cases"][2]["metrics"][0].update(top1_equal=False, relative_l2=0.12, cosine=0.91)
        first = validate_sequence_metrics(changed, changed_schedule, "fresh")
        self.assertEqual((first["case"], first["step"]), ("short-2", 0))

    def test_length_order_requires_explicit_reset_and_fresh_reference(self):
        for key, actual in (("reference_scope", "process"), ("reset_kv_cache", False), ("prompt_tokens", 31)):
            value, schedule = self.sequence_metrics(["long", "short"])
            value["cases"][1][key] = actual
            with self.subTest(key=key), self.assertRaises(ScenarioFailure):
                validate_sequence_metrics(value, schedule, "fresh")

    def test_length_order_history_hash_cannot_silently_change(self):
        value, schedule = self.sequence_metrics(["short", "long", "short"])
        references = {name: {"prompt_tokens": [1] * count, "generated_tokens": [2] * 32}
                      for name, count in (("short", 32), ("long", 512))}
        for actual, expected in zip(value["cases"], schedule):
            actual.update(_history_hashes(references[expected["reference_case"]]))
        self.assertIsNone(validate_sequence_metrics(value, schedule, "fresh", references))
        value["cases"][2]["forced_tokens_sha256"] = "0" * 64
        with self.assertRaisesRegex(ScenarioFailure, "history hash"):
            validate_sequence_metrics(value, schedule, "fresh", references)

    def test_environment_permissions_and_secrets_are_not_inherited(self):
        with patch.dict(os.environ, {"TS_CODE_EXEC": "1", "TS_KNOWLEDGE_DIRECTORY": "private", "TENSORSHARP_BACKEND": "cuda", "HF_TOKEN": "secret", "MAX_CONTEXT": "999999"}):
            env = clean_environment({"TS_CODE_EXEC": "1"}, "artifacts")
            self.assertEqual(env["TS_CODE_EXEC"], "0")
            self.assertEqual(env["MAX_CONTEXT"], "2048")
            for key in ("HF_TOKEN", "TS_KNOWLEDGE_DIRECTORY", "TENSORSHARP_BACKEND"):
                self.assertNotIn(key, env)
            self.assertTrue(env["TENSORSHARP_PREFIX_CACHE_DIR"].endswith("prefix-cache"))

    def test_redaction(self):
        value = redact("Bearer abcd.123 https://user:pass@example.test/x?token=abcd password=private literal_secret", ["literal_secret"])
        for secret in ("abcd.123", "user:pass", "token=abcd", "password=private", "literal_secret"):
            self.assertNotIn(secret, value)


class RealSubprocessRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "synthetic.gguf"
        self.model.write_bytes(b"GGUFfixture")
        self.context = {"repo": self.root, "artifact_dir": self.root / "result", "backend": "cpu",
                        "model_path": self.model, "timeout_seconds": 3}

    def tearDown(self):
        self.temp.cleanup()

    def fake_cli(self, answer="56", backend="Cpu", finish="eos", exit_code=0):
        output = f"Backend: {backend}\ninfo: cli.inference decode complete: tokens=1 ms=1 tokensPerSec=1000\ninfo: cli.inference finishReason={finish} tokens=1\n=== Generated Output ===\n{answer}\n"
        return [sys.executable, "-c", "import sys; print(" + repr(output) + "); sys.exit(" + str(exit_code) + ")"]

    def test_real_subprocess_validates_and_repeats(self):
        self.context["cli_command"] = self.fake_cli()
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(len(value["processes"]), 2)
        self.assertTrue((self.root / "result/scenario.json").is_file())

    def test_exit_zero_wrong_answer_is_failure(self):
        self.context["cli_command"] = self.fake_cli("55")
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertIn("semantic", value["reason"])

    def test_backend_mismatch_is_failure(self):
        self.context["cli_command"] = self.fake_cli(backend="GgmlCuda")
        self.assertEqual(run_scenario({"id": "cli_text"}, self.context)["status"], "fail")

    def test_oom_exit_captures_diagnostic(self):
        self.context["cli_command"] = [sys.executable, "-c", "import sys; print('CUDA out of memory',file=sys.stderr); sys.exit(9)"]
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertEqual(value["processes"][0]["exit_code"], 9)
        self.assertIn("out of memory", Path(value["processes"][0]["stderr_path"]).read_text())

    def test_timeout_kills_owned_process_and_records_stage(self):
        value = run_process([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(30)"],
                            cwd=self.root, artifact_dir=self.root, stage="timeout", timeout_seconds=0.2)
        self.assertTrue(value["timed_out"])
        self.assertIsNotNone(value["exit_code"])
        self.assertIn("started", value["stdout"])

    def test_reject_untrusted_effective_config_before_start(self):
        config = self.root / "config.json"
        config.write_text('{"code-exec": true}')
        self.context.update(cli_command=self.fake_cli(), effective_config_path=config)
        value = run_scenario({"id": "cli_text"}, self.context)
        self.assertEqual(value["status"], "fail")
        self.assertEqual(value["processes"], [])

    def test_reject_download_object_before_start(self):
        config = self.root / "config.json"
        config.write_text('{"model":{"path":"x","url":"https://example.test/model"}}')
        self.context.update(cli_command=self.fake_cli(), effective_config_path=config)
        self.assertEqual(run_scenario({"id": "cli_text"}, self.context)["status"], "fail")

    def test_download_preflight_does_not_require_existing_weights(self):
        self.context.update(cli_command=self.fake_cli(), model_path=self.root / "missing.gguf")
        self.assertFalse(availability({"id": "cli_text"}, self.context)[0])
        self.context["preparing_model"] = True
        self.assertTrue(availability({"id": "cli_text"}, self.context)[0])

    def test_missing_media_and_unknown_adapter_remain_untested(self):
        self.context["cli_command"] = self.fake_cli()
        self.assertEqual(run_scenario({"id": "cli_image"}, self.context)["status"], "untested")
        self.assertEqual(run_scenario({"id": "distributed"}, self.context)["status"], "untested")

    def test_logit_artifact_disk_guard_stops_before_launch(self):
        self.context["logit_command"] = [sys.executable, "-c", "raise RuntimeError('must not launch')"]
        with patch("eng.regression.scenarios.shutil.disk_usage", return_value=SimpleNamespace(free=1024)):
            result = run_scenario({"id": "forced_logits"}, self.context)
        self.assertEqual(result["status"], "untested")
        self.assertEqual(result["processes"], [])
        self.assertIn("disk", result["reason"])

    def test_nonfinite_benchmark_failure_still_writes_valid_json(self):
        output = "Backend: Cpu\ninference benchmark starting: prefillTokens=128 decodeTokens=0 runs=1\nbenchmark summary: bestPrefillMs=1 bestPrefillTps=NaN bestDecodeMs=0 bestDecodeTps=NaN\n"
        self.context["cli_command"] = [sys.executable, "-c", "print(" + repr(output) + ")"]
        value = run_scenario({"id": "cli_prefill"}, self.context)
        self.assertEqual(value["status"], "fail")
        json.loads((self.root / "result/scenario.json").read_text())

    def test_owned_server_bootstrap_and_real_loopback_replay(self):
        script = self.root / "synthetic_server.py"
        script.write_text('''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
port = int(sys.argv[sys.argv.index('--port') + 1])
model = Path(sys.argv[sys.argv.index('--model') + 1]).name
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/api/models': value = {'loaded':model,'loadedBackend':'cpu'}
        else: value = {'data':[{'id':model}]}
        self.reply(value)
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.reply({'object':'chat.completion','choices':[{'message':{'role':'assistant','content':'56'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':12,'completion_tokens':1,'total_tokens':13}})
    def reply(self, value):
        data = json.dumps(value).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
server = HTTPServer(('127.0.0.1', port), Handler)
print('Now listening on: http://127.0.0.1:' + str(port), flush=True)
server.serve_forever()
''', encoding="utf-8")
        self.context["server_command"] = [sys.executable, str(script)]
        self.context["timeout_seconds"] = 10
        value = run_scenario({"id": "server_chat"}, self.context)
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(value["evidence"]["server_shutdown"], "owned process tree terminated after scenario")


class HttpStreamRegressions(unittest.TestCase):
    def setUp(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if getattr(owner, "redirect", None):
                    self.send_response(302)
                    self.send_header("Location", owner.redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(owner.body.encode())
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.result = {"inputs": {}, "evidence": {}}
        self.http = _Http(self.server.server_port, time.monotonic() + 5, self.result)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def stream_body(self, content="56", done=True, finish="stop"):
        value = {"object": "chat.completion.chunk", "choices": [{"delta": {"content": content}, "finish_reason": None}]}
        final = {"object": "chat.completion.chunk", "choices": [{"delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}}
        return "data: " + json.dumps(value) + "\n\ndata: " + json.dumps(final) + "\n\n" + ("data: [DONE]\n\n" if done else "")

    def test_valid_stream_reconstructs_answer(self):
        self.body = self.stream_body()
        self.assertEqual(self.http.stream("/v1/chat/completions", {})["content"], "56")

    def test_loopback_redirect_is_refused(self):
        self.redirect = "http://example.invalid/never-request-this"
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.http.json("/v1/chat/completions", {})
        self.assertEqual(caught.exception.code, 302)
        self.assertEqual(self.result["evidence"]["http_errors"][0]["status"], 302)

    def test_stream_missing_terminal_does_not_pass(self):
        self.body = self.stream_body(done=False)
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_stream_duplicate_terminal_does_not_pass(self):
        self.body = self.stream_body() + "data: [DONE]\n\n"
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_completed_stream_cannot_count_as_cancellation_coverage(self):
        self.body = self.stream_body()
        with patch.object(self.http, "json", return_value={"processing": 0, "pending_requests": 0}):
            with self.assertRaises(ScenarioUntested):
                self.http.stream("/v1/chat/completions", {}, cancel=True)

    def test_stream_error_and_budget_exhaustion(self):
        self.body = self.stream_body(content="Error: bad model")
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})
        self.body = self.stream_body(finish="length")
        with self.assertRaises(ScenarioFailure):
            self.http.stream("/v1/chat/completions", {})

    def test_web_ui_replace_is_string(self):
        events = [{"token": "obsolete"}, {"replace": "56"},
                  {"done": True, "aborted": False, "truncated": False, "promptTokens": 12, "tokenCount": 1}]
        self.body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        self.assertEqual(self.http.stream("/api/chat", {}, web_ui=True)["content"], "56")


if __name__ == "__main__":
    unittest.main()
