"""Model-free regression tests for the Metal performance evidence harness.

Every HTTP connection/process is mocked. No listener, model or device is used.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

SPEC = importlib.util.spec_from_file_location("qwen38_metal_performance", Path(__file__).parents[1] / "qwen38-metal-performance.py")
PERF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PERF)


def event(content=None, finish=None, **delta):
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


USAGE = {"prompt_tokens": 80, "completion_tokens": 3, "total_tokens": 83,
         "prompt_tokens_details": {"cached_tokens": 64}}


class Response:
    def __init__(self, events, status=200, content_type="text/event-stream"):
        self.events, self.status, self.content_type = events, status, content_type

    def getheader(self, name, default=None):
        return self.content_type if name.lower() == "content-type" else default

    def read(self, size=None):
        return b"synthetic HTTP error"

    def __iter__(self):
        yield b": keepalive\n"
        for item in self.events:
            if isinstance(item, bytes):
                yield item
            else:
                raw = item if isinstance(item, str) else json.dumps(item)
                yield b"data: " + raw.encode() + b"\n\n"


class SseTests(unittest.TestCase):
    def request(self, events, status=200, content_type="text/event-stream"):
        connection = Mock()
        connection.getresponse.return_value = Response(events, status, content_type)
        # This exact module access catches the original def http()/import http
        # name collision before any real HTTP or model process is needed.
        with patch.object(PERF.http_client, "HTTPConnection", return_value=connection) as constructor:
            result = PERF.request_sse(18779, "synthetic", [{"role": "user", "content": "test"}], 32, 5)
        constructor.assert_called_once_with("127.0.0.1", 18779, timeout=5)
        connection.close.assert_called_once()
        connection.getresponse.assert_called_once()
        method, path, body, headers = connection.request.call_args.args
        self.assertEqual(("POST", "/v1/chat/completions"), (method, path))
        self.assertEqual("application/json", headers["Content-Type"])
        self.assertTrue(json.loads(body)["stream_options"]["include_usage"])
        return result

    def test_valid_stream_and_usage_after_finish_survive_module_name_collision(self):
        result = self.request([event(role="assistant"), event("5"), event("79", "stop"),
                               {"choices": [], "usage": USAGE}, "[DONE]"])
        self.assertEqual([], result["errors"])
        self.assertEqual("579", result["answer"])
        self.assertEqual("stop", result["finish_reason"])
        self.assertTrue(result["done"])
        self.assertEqual(USAGE, result["usage"])
        self.assertEqual(64, result["cached_tokens"])
        self.assertEqual(16, result["prefilled_tokens"])
        self.assertEqual(2, result["content_delta_count"])
        self.assertIsNotNone(result["post_first_delta_transport_tokens_per_second"])
        content_times = [e["elapsed_ms"] for e in result["events"]
                         if any(c.get("delta", {}).get("content") for c in e["data"].get("choices", []))]
        self.assertEqual(content_times[0], result["ttft_ms"])

    def test_role_only_stream_does_not_create_ttft(self):
        result = self.request([event(role="assistant"), event(finish="stop"), {"usage": USAGE}, "[DONE]"])
        self.assertIsNone(result["ttft_ms"])
        self.assertTrue(result["errors"])
        self.assertIsNone(result["post_first_delta_transport_tokens_per_second"])

    def test_missing_done_finish_or_usage_fails(self):
        for events in ([event("579", "stop"), {"usage": USAGE}],
                       [event("579"), {"usage": USAGE}, "[DONE]"],
                       [event("579", "stop"), "[DONE]"]):
            with self.subTest(events=events):
                self.assertTrue(self.request(events)["errors"])

    def test_server_error_and_reasoning_are_not_success(self):
        for problem in ({"error": {"message": "synthetic failure"}}, event(reasoning_content="reason")):
            with self.subTest(problem=problem):
                result = self.request([problem, event("579", "stop"), {"usage": USAGE}, "[DONE]"])
                self.assertTrue(result["errors"])

    def test_one_content_chunk_has_no_decode_rate(self):
        result = self.request([event("579", "stop"), {"usage": USAGE}, "[DONE]"])
        self.assertEqual([], result["errors"])
        self.assertIsNone(result["post_first_delta_transport_tokens_per_second"])

    def test_http_failure_and_malformed_sse_close_connection(self):
        for response in (Response([], status=503), Response([], content_type="application/json"),
                         Response([b"data: {invalid-json}\n"])):
            connection = Mock()
            connection.getresponse.return_value = response
            with self.subTest(response=response), patch.object(PERF.http_client, "HTTPConnection", return_value=connection):
                with self.assertRaises(ValueError):
                    PERF.request_sse(18779, "synthetic", [], 32, 5)
                connection.close.assert_called_once()


def benchmark_line(run=1, total=1):
    return (f"benchmark run {run}/{total}: prefillMs=101 prefillTps=1267.3 decodeMs=3000 "
            "decodeTps=21.3 msPerTok=46.9 decodeModelMs=2999 decodeModelTps=21.3 "
            "greedySampleMs=0.0 decodeMode=random-inference")


class CliParserTests(unittest.TestCase):
    def test_collects_each_run_with_cache_scope(self):
        rows = PERF.parse_runs("\n".join(benchmark_line(i, 3) for i in range(1, 4)), 3)
        self.assertEqual([1, 2, 3], [r["run"] for r in rows])
        self.assertEqual(101, rows[0]["prefillMs"])
        self.assertEqual("random-inference", rows[0]["decodeMode"])
        self.assertIn("process-first", rows[0]["cache_label"])
        self.assertIn("graph-warm", rows[1]["cache_label"])

    def test_missing_duplicate_out_of_order_or_summary_only_fail(self):
        for log in (benchmark_line(1, 2), benchmark_line() + "\n" + benchmark_line(),
                    benchmark_line(2, 2) + "\n" + benchmark_line(1, 2),
                    "benchmark summary: bestPrefillMs=101 bestDecodeMs=3000"):
            with self.subTest(log=log), self.assertRaises(ValueError):
                PERF.parse_runs(log, 2)

    def test_missing_model_timing_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "decodeModelMs"):
            PERF.parse_runs(benchmark_line().replace("decodeModelMs=2999 ", ""), 1)


class LogitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def vector(self, name, values=(1., 2., 3.), **metadata):
        path = self.directory / name
        raw = struct.pack("<" + "f" * len(values), *values)
        path.write_bytes(raw)
        meta = {"format": "f32le", "semantics": "first-generated-token", "count": len(values),
                "vector_sha256": hashlib.sha256(raw).hexdigest(), "prompt_tokens": 128,
                "prompt_fingerprint": "same-token-stream", "media_spans": 0, "prefix_reused_tokens": 0}
        meta.update(metadata)
        Path(str(path) + ".json").write_text(json.dumps(meta))
        return path

    def test_full_vector_comparison_and_top1_change(self):
        a = self.vector("a")
        exact = PERF.compare_logits(a, self.vector("same"))
        self.assertTrue(exact["bit_identical"])
        self.assertEqual(0, exact["max_abs"])
        drift = PERF.compare_logits(a, self.vector("drift", (1., 2., 3.000001)))
        self.assertFalse(drift["bit_identical"])
        self.assertTrue(drift["top1_equal"])
        self.assertGreater(drift["relative_l2"], 0)
        flip = PERF.compare_logits(a, self.vector("flip", (1., 4., 3.)))
        self.assertFalse(flip["top1_equal"])
        self.assertEqual(2, flip["max_abs"])

    def test_stale_hash_bad_count_format_and_nonfinite_fail(self):
        cases = [((1., 2., 3.), {"vector_sha256": "stale"}),
                 ((1., 2., 3.), {"count": 4}),
                 ((1., 2., 3.), {"format": "f16le"}),
                 ((1., 2., float("nan")), {}), ((1., 2., float("inf")), {})]
        for index, (values, metadata) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                PERF.read_logits(self.vector(str(index), values, **metadata))

    def test_mismatched_or_missing_prompt_identity_fails(self):
        a = self.vector("a")
        for index, metadata in enumerate(({"prompt_fingerprint": "different"}, {"prompt_fingerprint": None},
                                          {"prompt_tokens": 129}, {"prefix_reused_tokens": 64})):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                PERF.compare_logits(a, self.vector(str(index), **metadata))


class EnvironmentTests(unittest.TestCase):
    def arguments(self):
        return SimpleNamespace(context=None, chunk=None, threads=None,
            environment={"TS_CPU_MOE_THREADS": "0"},
            variant_environment={"candidate": {"TS_CPU_MOE_THREADS": "12"}})

    def test_variant_override_is_isolated_and_inherited_settings_are_removed(self):
        with patch.dict(PERF.os.environ, {"TS_UNRELATED": "discard", "GGML_METAL_FUSION_DISABLE": "1"}):
            baseline = PERF.environment(self.arguments(), "baseline", Path("baseline.f32"))
            candidate = PERF.environment(self.arguments(), "candidate", Path("candidate.f32"))
        self.assertEqual("0", baseline["TS_CPU_MOE_THREADS"])
        self.assertEqual("12", candidate["TS_CPU_MOE_THREADS"])
        self.assertNotIn("TS_UNRELATED", candidate)
        self.assertNotIn("GGML_METAL_FUSION_DISABLE", candidate)
        self.assertEqual("candidate.f32", candidate["TS_DUMP_LOGITS"])

    def test_variant_cannot_override_evidence_path_or_unlisted_settings(self):
        for key in ("TS_DUMP_LOGITS", "UNLISTED", "TS_Q4E_METAL_HC_FUSED", "TS_HOST_MOE_DECODE_MATRIX_ORDER"):
            args = self.arguments()
            args.variant_environment = {"candidate": {key: "forbidden"}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                PERF.environment(args, "candidate", Path("owned.f32"))

    def test_defaults_preserve_engine_choices_and_common_override_wins(self):
        args = self.arguments()
        args.environment, args.variant_environment = {}, {}
        env = PERF.environment(args, "baseline", Path("owned.f32"))
        for key in ("MAX_CONTEXT", "TS_PREFILL_CHUNK", "TS_CPU_MOE_THREADS", "TS_GGML_CPU_THREADS", "TS_SPEC"):
            self.assertNotIn(key, env)
        args.threads = 12
        args.environment = {"TS_CPU_MOE_THREADS": "0"}
        env = PERF.environment(args, "tuned", Path("owned.f32"))
        self.assertEqual("0", env["TS_CPU_MOE_THREADS"])
        self.assertEqual("12", env["TS_GGML_CPU_THREADS"])


class ConfigurationTests(unittest.TestCase):
    def test_variant_config_argv_and_sha_are_recorded_in_cli_and_http_dry_run(self):
        for mode in ("cli", "http"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                config = root / "tuned.json"
                config.write_text('{"CpuMoeThreads":12}\n')
                output = io.StringIO()
                argv = ["qwen38-metal-performance.py", mode, "--variant", "baseline=" + str(root),
                        "--variant", "tuned=" + str(root), "--variant-config", "tuned=" + str(config),
                        "--model", "unused.gguf", "--output", str(root / "artifacts/dry"), "--cases", "128:64"]
                with patch.object(PERF, "ROOT", root), patch.object(PERF.sys, "argv", argv), \
                        patch.object(PERF.subprocess, "Popen", side_effect=AssertionError("dry run launched a process")), \
                        contextlib.redirect_stdout(output):
                    PERF.main()
                baseline, tuned = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertIsNone(baseline["config"])
                self.assertNotIn("--config", baseline["argv"])
                self.assertNotIn("--n-cpu-moe", baseline["argv"])
                self.assertEqual(str(config), tuned["argv"][tuned["argv"].index("--config") + 1])
                self.assertEqual({"file": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}, tuned["config"])
                self.assertFalse((root / "artifacts/dry").exists())


class IdentityTests(unittest.TestCase):
    def test_transitive_tensorsharp_assemblies_are_hashed_and_entry_native_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app"
            app.mkdir()
            files = {"TensorSharp.Cli.dll": b"entry", "libGgmlOps.dylib": b"native",
                     "TensorSharp.Models.dll": b"models", "TensorSharp.Runtime.dll": b"runtime",
                     "TensorSharp.Chat.dll": b"chat"}
            for name, raw in files.items():
                (app / name).write_bytes(raw)
            with patch.object(PERF, "capture", return_value={"exit_code": 0, "output": "synthetic"}):
                provenance = PERF.identity(root, app / "TensorSharp.Cli.dll")
                self.assertEqual({"app/" + name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                                 provenance["binaries_sha256"])
                for name in ("TensorSharp.Cli.dll", "libGgmlOps.dylib"):
                    (app / name).unlink()
                    with self.subTest(missing=name), self.assertRaises(ValueError):
                        PERF.identity(root, app / "TensorSharp.Cli.dll")
                    (app / name).write_bytes(files[name])


class HttpVectorTests(unittest.TestCase):
    def run_mocked_servers(self, changed=False, missing=False, pid_text=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = root / "synthetic.gguf"
            model.write_bytes(b"not a model")
            config = root / "tuned.json"
            config.write_text('{"CpuMoeThreads":12}')
            output = root / "artifacts/run"
            stopped, launched = [], []
            compare = PERF.compare_logits

            def launch(argv, **kwargs):
                launched.append(argv)
                kwargs["stderr"].write("123456 maximum resident set size\n")
                kwargs["stderr"].flush()
                dump = Path(kwargs["env"]["TS_DUMP_LOGITS"])
                if pid_text is not None:
                    (dump.parent / "server.pid").write_text(pid_text)
                if not (missing and len(launched) == 2):
                    raw = struct.pack("<3f", 1., 2., 3.01 if changed and len(launched) == 2 else 3.)
                    dump.write_bytes(raw)
                    Path(str(dump) + ".json").write_text(json.dumps({"format": "f32le", "semantics": "first-generated-token",
                        "count": 3, "vector_sha256": hashlib.sha256(raw).hexdigest(), "prompt_tokens": 80,
                        "prompt_fingerprint": "arithmetic", "media_spans": 0, "prefix_reused_tokens": 0}))
                return Mock(pid=10000 + len(launched), poll=Mock(return_value=None), returncode=None)

            def stop(process, child_pid=None):
                if pid_text in ("", "malformed", "-1"):
                    self.assertIsNone(child_pid, "invalid PID must use owned-group fallback")
                stopped.append(process.pid)
                process.returncode = 0

            def compare_after_exit(first, second):
                self.assertEqual(2, len(stopped), "HTTP vectors compared before owned server exit")
                return compare(first, second)

            def request(port, model, messages, max_tokens, timeout):
                prompt = messages[0]["content"]
                answer = "579" if prompt.startswith("Only output") else '{"answer":42,"ok":true}' if prompt.startswith("Output only") else "中文回答"
                return {"answer": answer, "usage": USAGE, "errors": [], "finish_reason": "stop", "ttft_ms": 1, "e2e_ms": 2}

            connection = Mock()
            connection.getresponse.return_value = Mock(status=200, read=Mock(return_value=b'{"data":[{"id":"synthetic"}]}'))
            argv = ["qwen38-metal-performance.py", "http", "--variant", "baseline=" + str(root),
                    "--variant", "tuned=" + str(root), "--variant-config", "tuned=" + str(config),
                    "--model", str(model), "--output", str(output), "--runs", "1", "--execute"]
            with patch.object(PERF, "ROOT", root), patch.object(PERF.sys, "argv", argv), \
                    patch.object(PERF, "identity", return_value={"source": "synthetic"}), \
                    patch.object(PERF.socket, "socket", return_value=MagicMock()), \
                    patch.object(PERF.subprocess, "Popen", side_effect=launch), \
                    patch.object(PERF.http_client, "HTTPConnection", return_value=connection), \
                    patch.object(PERF, "request_sse", side_effect=request), \
                    patch.object(PERF, "stop_owned", side_effect=stop), \
                    patch.object(PERF, "compare_logits", side_effect=compare_after_exit), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    PERF.main()
                except SystemExit as error:
                    self.assertTrue(missing)
                    self.assertEqual(1, error.code)
            return json.loads((output / "report.json").read_text()), launched

    def test_http_vectors_are_compared_after_owned_exit_and_summary_records_identity(self):
        report, argv = self.run_mocked_servers()
        self.assertTrue(report["passed"])
        self.assertEqual(2, report["quality_summary"]["first_logit_vectors_validated"])
        self.assertEqual(1, report["quality_summary"]["first_logit_vectors_compared"])
        self.assertTrue(report["quality_summary"]["first_logit_all_bit_identical"])
        self.assertNotIn("--config", argv[0])
        self.assertIn("--config", argv[1])
        self.assertEqual(123456, report["processes"][1]["peak_process_rss_bytes"])

    def test_http_bit_drift_and_missing_vector_cannot_look_identical(self):
        report, _ = self.run_mocked_servers(changed=True)
        self.assertEqual(1, report["quality_summary"]["first_logit_bit_mismatches"])
        self.assertFalse(report["quality_summary"]["first_logit_all_bit_identical"])
        report, _ = self.run_mocked_servers(missing=True)
        self.assertFalse(report["passed"])
        self.assertTrue(report["processes"][1]["errors"])

    def test_empty_malformed_and_nonpositive_pid_files_still_stop_owned_servers(self):
        for text in ("", "malformed", "-1"):
            with self.subTest(pid_file=text):
                report, _ = self.run_mocked_servers(pid_text=text)
                self.assertTrue(report["complete"])
                self.assertTrue(report["passed"])
                self.assertEqual([0, 0], [p["exit_code_after_owned_shutdown"] for p in report["processes"]])


class MainPersistenceTests(unittest.TestCase):
    def invoke(self, work):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = root / "synthetic.gguf"
            model.write_bytes(b"not a model; process calls are mocked")
            output = root / "artifacts/run"
            argv = ["qwen38-metal-performance.py", "http", "--variant", "base=" + str(root),
                    "--model", str(model), "--output", str(output), "--execute"]
            with patch.object(PERF, "ROOT", root), patch.object(PERF.sys, "argv", argv), \
                    patch.object(PERF, "http", side_effect=work), \
                    patch.object(PERF.subprocess, "Popen", side_effect=AssertionError("no real process permitted")), \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    PERF.main()
                except Exception as error:
                    caught = error
                except SystemExit as error:
                    caught = error
                else:
                    caught = None
            return caught, json.loads((output / "report.json").read_text())

    def test_recorded_quality_errors_force_nonzero_and_persist(self):
        def work(args, report):
            report["processes"].append({"requests": [{"errors": ["incorrect JSON answer"]}]})
        caught, report = self.invoke(work)
        self.assertIsInstance(caught, SystemExit)
        self.assertEqual(1, caught.code)
        self.assertTrue(report["complete"])
        self.assertFalse(report["passed"])
        self.assertEqual(1, report["quality_summary"]["failed_requests"])

    def test_stop_failure_keeps_incomplete_report(self):
        def work(args, report):
            report["processes"].append({"requests": [{"errors": [], "answer": "579"}]})
            process = Mock(pid=123456)
            process.poll.return_value = None
            with patch.object(PERF.os, "killpg", side_effect=PermissionError("synthetic shutdown failure")):
                PERF.stop_owned(process)
        caught, report = self.invoke(work)
        self.assertIsInstance(caught, PermissionError)
        self.assertFalse(report["complete"])
        self.assertFalse(report.get("passed", False))
        self.assertEqual("579", report["processes"][0]["requests"][0]["answer"])

    def test_stream_exception_keeps_incomplete_report(self):
        def work(args, report):
            report["processes"].append({"requests": []})
            raise ValueError("synthetic malformed SSE")
        caught, report = self.invoke(work)
        self.assertIsInstance(caught, ValueError)
        self.assertFalse(report["complete"])
        self.assertFalse(report.get("passed", False))


if __name__ == "__main__":
    unittest.main()
