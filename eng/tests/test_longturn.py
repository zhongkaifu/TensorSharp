#!/usr/bin/env python3
"""Tests for eng/longturn.py and eng/longturn-report.py; no GPU, model or server is required."""
import contextlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"),
        Path(__file__).resolve().parents[1] / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


longturn = load("longturn")
report = load("longturn-report")


def make_checkout(root):
    """A git checkout with three TensorSharp.Runtime sources (one with CRLF line ends), committed, and an untracked
    obj/ source a build would leave."""
    rt = Path(root, "TensorSharp.Runtime")
    (rt / "Sub").mkdir(parents=True)
    (rt / "Alpha.cs").write_bytes(b"class Alpha {}\r\n// crlf\r\n" + b"x" * 3000 + b"\n")
    (rt / "Sub" / "Beta.cs").write_text("class Beta {}\n" + "y" * 3000 + "\n")
    (rt / "Zeta.cs").write_text("class Zeta {}\n" + "z" * 3000 + "\n")
    git = ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "corpus"], check=True)
    (rt / "obj" / "linux").mkdir(parents=True)
    (rt / "obj" / "linux" / "Generated.cs").write_text("// generated\n")


class CorpusTests(unittest.TestCase):
    def test_a_revision_reads_what_the_issue_script_reads_from_a_clean_tree(self):
        with tempfile.TemporaryDirectory() as root:
            make_checkout(root)
            from_git = longturn.corpus_files(root, "HEAD")
            from_disk = longturn.corpus_files(root, longturn.WORKTREE)
            self.assertEqual(["Alpha.cs", os.path.join("Sub", "Beta.cs"), "Zeta.cs"], [p for p, _ in from_git])
            # The working tree adds the build's obj/ source (it sorts after the upper-case names), nothing else differs.
            self.assertEqual(from_git, [f for f in from_disk if not f[0].startswith("obj")])
            self.assertIn(os.path.join("obj", "linux", "Generated.cs"), [p for p, _ in from_disk])
            # Text-mode reads turn CRLF into LF; blobs read from git get the same.
            self.assertTrue(from_git[0][1].startswith("class Alpha {}\n// crlf\n"))

    def test_a_missing_revision_names_the_way_out(self):
        with tempfile.TemporaryDirectory() as root:
            make_checkout(root)
            with self.assertRaises(SystemExit) as caught:
                longturn.corpus_files(root, "0123456789abcdef")
            self.assertIn("--corpus-rev worktree", str(caught.exception))

    def test_messages_follow_the_issue_and_fresh_rounds_change_only_the_system_turn(self):
        corpus = "".join(chr(97 + i % 26) for i in range(20000))
        a, b = longturn.conversation(corpus, 1000, 0, fresh_rounds=False)
        self.assertEqual(2, len(a))
        self.assertEqual(a, b[:2])
        self.assertEqual(["system", "user", "assistant", "user"], [m["role"] for m in b])
        self.assertIn(corpus[:3200], a[1]["content"])
        self.assertNotIn(corpus[:3201], a[1]["content"])
        self.assertIn(corpus[-4300:], b[3]["content"])
        a2, _ = longturn.conversation(corpus, 1000, 1, fresh_rounds=True)
        self.assertNotEqual(a[0]["content"], a2[0]["content"])
        self.assertEqual(a[1], a2[1])
        self.assertEqual(a, longturn.conversation(corpus, 1000, 1, fresh_rounds=False)[0])


class FakeServer(http.server.ThreadingHTTPServer):
    """/v1/models and /v1/chat/completions; a request whose max_tokens is 1 and has 4 messages fails with 500 when its
    document is longer than fail_above characters."""

    def __init__(self, fail_above=None, truncate_c=False):
        super().__init__(("127.0.0.1", 0), FakeHandler)
        self.bodies = []
        self.fail_above = fail_above
        self.truncate_c = truncate_c   # every C's answer stops mid-body, as from a server that dies while answering


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"data": [{"id": "fake-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        doc = len(body["messages"][1]["content"])
        if self.server.fail_above is not None and len(body["messages"]) == 4 and body["max_tokens"] == 1 \
                and doc > self.server.fail_above:
            self._send(500, {"error": {"message": "CUDA error 2: out of memory"}})
            return
        if self.server.truncate_c and body["max_tokens"] > 1:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"id": "chatcmpl-cut", "us')
            return
        n = len(self.server.bodies)
        self._send(200, {"id": f"chatcmpl-{n}", "choices": [{"finish_reason": "length"}],
                         "usage": {"prompt_tokens": doc // 3, "completion_tokens": body["max_tokens"]}})


@contextlib.contextmanager
def serving(**kw):
    server = FakeServer(**kw)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class ClientTests(unittest.TestCase):
    def run_client(self, root, server, extra):
        out = os.path.join(root, "out")
        args = longturn.parse_args([f"127.0.0.1:{server.server_address[1]}", "--corpus", root, "--corpus-rev", "HEAD",
                                    "--out", out] + extra)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = longturn.run(args)
        with open(os.path.join(out, "client.jsonl")) as f:
            records = [json.loads(line) for line in f]
        with open(os.path.join(out, "run.json")) as f:
            meta = json.load(f)
        return rc, records, meta

    def test_each_size_sends_a_b_c_with_the_pinned_sampling(self):
        with tempfile.TemporaryDirectory() as root, serving() as server:
            make_checkout(root)
            rc, records, meta = self.run_client(root, server, ["--sizes", "100", "200", "--rounds", "2",
                                                               "--temperature", "0.3", "--top-k", "20", "--top-p", "0.8",
                                                               "--max-tokens", "64"])
            self.assertEqual(0, rc)
            self.assertEqual([(100, 0, "A"), (100, 0, "B"), (100, 0, "C"), (100, 1, "A"), (100, 1, "B"), (100, 1, "C"),
                              (200, 0, "A"), (200, 0, "B"), (200, 0, "C"), (200, 1, "A"), (200, 1, "B"), (200, 1, "C")],
                             [(r["size"], r["round"], r["step"]) for r in records])
            self.assertEqual([1, 1, 64] * 4, [b["max_tokens"] for b in server.bodies])
            self.assertEqual([2, 4, 4] * 4, [len(b["messages"]) for b in server.bodies])
            for body in server.bodies:
                self.assertEqual(("fake-model", 0.3, 20, 0.8), (body["model"], body["temperature"], body["top_k"], body["top_p"]))
                self.assertNotIn("seed", body)
            # B and C send the same prompt; a second round repeats the first without --fresh-rounds.
            self.assertEqual(server.bodies[1]["messages"], server.bodies[2]["messages"])
            self.assertEqual(server.bodies[0]["messages"], server.bodies[3]["messages"])
            self.assertEqual(["chatcmpl-1", 200, None], [records[0]["id"], records[0]["status"], records[0]["error"]])
            self.assertEqual({"temperature": 0.3, "top_k": 20, "top_p": 0.8}, meta["sampling"])
            self.assertEqual(("HEAD", 3), (meta["corpus_rev"], meta["corpus_files"]))

    def test_without_sampling_flags_the_server_defaults_apply_and_a_failure_ends_its_size(self):
        with tempfile.TemporaryDirectory() as root, serving(fail_above=500) as server:
            make_checkout(root)
            rc, records, meta = self.run_client(root, server, ["--sizes", "100", "400", "100"])
            self.assertEqual(1, rc)
            # 400 tokens is a document of 1,280 characters: its B fails, so its C is not sent; the next size still runs.
            self.assertEqual([(100, "A"), (100, "B"), (100, "C"), (400, "A"), (400, "B"), (100, "A"), (100, "B"), (100, "C")],
                             [(r["size"], r["step"]) for r in records])
            self.assertEqual(500, records[4]["status"])
            self.assertIn("out of memory", records[4]["error"])
            self.assertIsNone(meta["sampling"])
            for body in server.bodies:
                self.assertEqual({"model", "messages", "max_tokens"}, set(body))

    def test_an_answer_cut_off_mid_body_is_a_failed_request_not_a_crash(self):
        # http.client's IncompleteRead is not an OSError: it had escaped chat() and ended the run without a record.
        with tempfile.TemporaryDirectory() as root, serving(truncate_c=True) as server:
            make_checkout(root)
            rc, records, _ = self.run_client(root, server, ["--sizes", "100", "200"])
            self.assertEqual(1, rc)
            self.assertEqual([(100, "A"), (100, "B"), (100, "C"), (200, "A"), (200, "B"), (200, "C")],
                             [(r["size"], r["step"]) for r in records])
            for c in (records[2], records[5]):
                self.assertEqual(200, c["status"])
                self.assertIn("IncompleteRead", c["error"])

    def test_stop_on_error_stops_at_the_first_failure(self):
        with tempfile.TemporaryDirectory() as root, serving(fail_above=500) as server:
            make_checkout(root)
            rc, records, _ = self.run_client(root, server, ["--sizes", "400", "100", "--stop-on-error"])
            self.assertEqual(1, rc)
            self.assertEqual([(400, "A"), (400, "B")], [(r["size"], r["step"]) for r in records])

    def test_stamp_prefixes_every_line(self):
        out = io.StringIO()
        longturn.stamp(io.StringIO("one\ntwo\n"), out)
        lines = out.getvalue().splitlines()
        self.assertEqual(2, len(lines))
        for line, text in zip(lines, ("one", "two")):
            self.assertRegex(line, r"^\d\d:\d\d:\d\d\.\d{3} " + text + "$")


SERVER_LOG = """\
23:59:58.000 info: TensorSharp.Server.Host.Startup[1000]
23:59:58.001 Model: qwen35 on the shared runtime
23:59:59.000 info: TensorSharp.Server.ModelService[1500]
23:59:59.000       chat.start arch=qwen35 origin=prefix-cache-warmup maxTokens=1 thinking=False
23:59:59.050 [step] t=23:59:59.050 gap=0.0 ms=40.0 Unified warm:P13
23:59:59.060       chat.complete tokens=1 origin=prefix-cache-warmup promptTokens=13 kvReused=0 ttftMs=50 elapsedMs=60.0
00:00:01.000       /v1/chat/completions request: id=chatcmpl-a1 model=m stream=False maxTokens=1
00:00:01.001       chat.start arch=qwen35 maxTokens=1 thinking=False tools=0
00:00:01.010       prompt.shared_prefix tokens=0 hash=- boundaries= of 9000 prompt tokens (requestId=chat-0000aaaa, sessionId=x)
00:00:01.020       Radix prompt reuse for chat-0000aaaa: 0/9000 tokens; 9000 token(s) to prefill (no resumable radix prefix).
00:00:02.040 [step] t=00:00:02.040 gap=10.0 ms=1000.0 Unified aaaa:P8192
00:00:02.200 [step] t=00:00:02.200 gap=0.5 ms=150.0 Unified aaaa:P808
00:00:02.230 [step] t=00:00:02.230 gap=0.2 ms=20.0 Unified aaaa:D1
00:00:02.250       chat.complete tokens=1 promptTokens=9000 kvReused=0 kvReusePercent=0.0 ttftMs=1200 elapsedMs=1249.0 tokensPerSec=1.0 finishReason=max_tokens assistantOutput="OK"
00:00:02.300       /v1/chat/completions request: id=chatcmpl-b2 model=m stream=False maxTokens=1
00:00:02.301       chat.start arch=qwen35 maxTokens=1 thinking=False tools=0
00:00:02.400       [prefix-retain] chat-0000aaaa length=9001 primary=False captured=True origin=Donation
00:00:04.301       Radix prompt reuse for chat-0000bbbb: 9001/10000 tokens from radix end state; 999 token(s) to prefill.
00:00:04.600 [step] t=00:00:04.600 gap=4000.0 ms=250.0 Unified bbbb:P999
00:00:04.630 [step] t=00:00:04.630 gap=0.1 ms=20.0 Unified bbbb:D1
00:00:04.640       chat.complete tokens=1 promptTokens=10000 kvReused=9001 kvReusePercent=90.0 ttftMs=2300 elapsedMs=2339.0 tokensPerSec=1.0 finishReason=max_tokens assistantOutput="OK"
00:00:04.700       /v1/chat/completions request: id=chatcmpl-c3 model=m stream=False maxTokens=8
00:00:04.701       chat.start arch=qwen35 maxTokens=8 thinking=False tools=0
00:00:04.710       Radix prompt reuse for chat-0000cccc: 10000/10000 tokens from radix end state; 0 token(s) to prefill.
00:00:04.750 [step] t=00:00:04.750 gap=1.0 ms=30.0 Unified cccc:D1
00:00:04.800 [step] t=00:00:04.800 gap=0.1 ms=40.0 Unified cccc:D4
00:00:04.850 [step] t=00:00:04.850 gap=0.1 ms=44.0 Unified cccc:D3
00:00:04.860       Speculative decoding stats for chat-0000cccc: drafted=14 accepted=6 acceptance=43 % verifySteps=2 plainSteps=0 rollbacks=2 | phaseMs draft=10 verify=70 snapshot=0 rollback=0 catchUp=2 plain=0 | governor plain=0.0ms/tok spec=0.0ms/tok wins=0 losses=0 parked=0
00:00:04.861       chat.complete tokens=8 promptTokens=10000 kvReused=10000 kvReusePercent=100.0 ttftMs=49 elapsedMs=160.0 tokensPerSec=50.0 finishReason=max_tokens assistantOutput="x"
00:00:05.000       /v1/chat/completions request: id=chatcmpl-d4 model=m stream=False maxTokens=1
00:00:05.001       chat.start arch=qwen35 maxTokens=1 thinking=False tools=0
00:00:05.010       Radix prompt reuse for chat-0000dddd: 0/30000 tokens; 30000 token(s) to prefill (no resumable radix prefix).
00:00:05.200 fail: TensorSharp.Server.ModelService[0]
00:00:05.200       Unified step failed for sequence chat-0000dddd
00:00:05.200       TensorSharp.Cuda.Interop.CudaException: CUDA error 2: out of memory
00:00:05.200          at TensorSharp.Cuda.Interop.CudaErrorHelper.ThrowOnError(Int32 result) in /x/CudaErrorHelper.cs:line 13
00:00:05.200          at TensorSharp.Cuda.CudaAllocator.AllocateDeviceMemory(Int64 allocationBytes) in /x/CudaAllocator.cs:line 299
00:00:05.200          at TensorSharp.Models.StandardDecoderModel.AllocateCaches(Int32 capacity) in /x/Slots.cs:line 471
00:00:05.200          at TensorSharp.Models.StandardDecoderModel.EnsureSequenceSlot(String requestId, Int32 reserveTokens) in /x/Unified.cs:line 149
00:00:05.200          at TensorSharp.Runtime.Scheduling.BatchExecutor.ExecuteStepUnified(IUnifiedBatchModel model, SchedulerOutput output) in /x/BatchExecutor.cs:line 2478
00:00:05.201 [step] t=00:00:05.201 gap=33.9 ms=190.0 Unified dddd:D0
00:00:05.210 fail: TensorSharp.Server.Logging.RequestLoggingMiddleware[1102]
00:00:05.210       HTTP POST /v1/chat/completions threw CudaException after 209.7 ms
"""


class ReportTests(unittest.TestCase):
    def write_run(self, root):
        Path(root, "server.log").write_text(SERVER_LOG)
        records = [
            {"size": 9000, "round": 0, "step": "A", "sent": "2026-11-18T00:00:00.990Z", "done": "2026-11-18T00:00:02.260Z",
             "status": 200, "id": "chatcmpl-a1", "error": None, "wall_s": 1.27},
            {"size": 9000, "round": 0, "step": "B", "sent": "2026-11-18T00:00:02.290Z", "done": "2026-11-18T00:00:04.650Z",
             "status": 200, "id": "chatcmpl-b2", "error": None, "wall_s": 2.36},
            {"size": 9000, "round": 0, "step": "C", "sent": "2026-11-18T00:00:04.690Z", "done": "2026-11-18T00:00:04.870Z",
             "status": 200, "id": "chatcmpl-c3", "error": None, "wall_s": 0.18},
            {"size": 30000, "round": 0, "step": "A", "sent": "2026-11-18T00:00:04.990Z", "done": "2026-11-18T00:00:05.220Z",
             "status": 500, "id": None, "error": "HTTP 500: out of memory", "wall_s": 0.23},
        ]
        Path(root, "client.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_a_run_reads_into_the_issue_s_measurements(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root)
            run = report.load_run(os.path.join(root, "server.log"), os.path.join(root, "client.jsonl"),
                                  [report.DEFAULT_NOTES])
            a, b, c, d = run["rows"]
            self.assertEqual(4, run["requests"])   # the warm-up is not a client request
            # A: 9,000 fresh tokens; its 1-token answer completed 49 ms after the first token.
            self.assertEqual(("A", 9000, 0, 1200, 49), (a["step"], a["prompt"], a["reused"], a["ttft_ms"], a["tail_ms"]))
            self.assertEqual((2, 1150, 1), (a["prompt_steps"], a["prompt_ms"], a["decode_steps"]))
            self.assertEqual((39, "P8192", 1000.0), (a["admit_ms"], a["first_rows"], a["first_ms"]))
            self.assertEqual(7500, a["prefill_tok_s"])
            # B waited 2 s for its radix line and 2.05 s for its first step, which came after an idle gap of 4 s.
            self.assertEqual((2000, 2049, 4000.0, 250.0, "P999"),
                             (b["radix_ms"], b["admit_ms"], b["first_gap_ms"], b["first_ms"], b["first_rows"]))
            self.assertEqual([(99, "[prefix-retain] chat-0000aaaa length=9001 primary=False captured=True origin=Donation")],
                             b["notes"])
            # C: decode steps and the speculative line per verify step.
            self.assertEqual((3, 40.0, 2.67), (c["decode_steps"], c["decode_ms_med"], c["tokens_per_decode_step"]))
            self.assertEqual((2, "43", 5.0, 35.0, 1.0), (c["verify_steps"], c["acceptance"], c["draft_ms"], c["verify_ms"],
                                                         c["catchUp_ms"]))
            self.assertNotIn("commit_ms", c)
            # The failed request: its prompt from the radix line, the client's status and where the server threw.
            self.assertEqual((30000, 0), (d["prompt"], d["reused"]))
            self.assertIn("HTTP 500", d["error"])
            self.assertIn("CudaException: CUDA error 2: out of memory", d["error"])
            self.assertIn("ExecuteStepUnified -> EnsureSequenceSlot -> AllocateCaches (in AllocateDeviceMemory)", d["error"])

    def test_the_per_verify_step_section_is_read_when_the_line_has_one(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root)
            log = Path(root, "server.log")
            log.write_text(log.read_text().replace(
                "phaseMs draft=10 verify=70 snapshot=0 rollback=0 catchUp=2 plain=0 |",
                "phaseMs draft=10 verify=70 commit=3 snapshot=0 rollback=0 catchUp=2 plain=0 "
                "| perVerifyMs draft=5.2 verify=34.9 commit=1.4 catchUp=0.9 |"))
            run = report.load_run(str(log), os.path.join(root, "client.jsonl"), [])
            c = run["rows"][2]
            self.assertEqual((5.2, 34.9, 1.4, 0.9), (c["draft_ms"], c["verify_ms"], c["commit_ms"], c["catchUp_ms"]))

    def test_markdown_puts_a_size_s_requests_side_by_side(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                report.main([root])
            text = out.getvalue()
            self.assertIn("| 9000 r0 | 1.20 | 49 | 2.30 | 2049 | 250.0 | 0.05 | 10000 | 50.0 | 35.0 / 5.0 /  |  |", text)
            self.assertIn("| 30000 r0 |  |  |  |  |  |  |  |  |  | A: HTTP 500", text)

    def test_the_issue_scripts_client_log_still_reads(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "server.log").write_text(SERVER_LOG)
            Path(root, "client.log").write_text(
                "00:00:02.260 9000 A 9000 1.27 s\n00:00:04.650 9000 B 10000 2.36 s\n00:00:04.870 9000 C 10000 0.18 s\n"
                "00:00:05.221 Traceback (most recent call last):\n"
                "00:00:05.222 urllib.error.HTTPError: HTTP Error 500: Internal Server Error\n")
            run = report.load_run(os.path.join(root, "server.log"), os.path.join(root, "client.log"), [])
            self.assertEqual([("A", 9000), ("B", 9000), ("C", 9000), ("A", None)],
                             [(r["step"], r["size"]) for r in run["rows"]])
            self.assertEqual(2049, run["rows"][1]["admit_ms"])
            self.assertIn("HTTP 500", run["rows"][3]["error"])
            self.assertEqual(30000, run["rows"][3]["prompt"])


if __name__ == "__main__":
    unittest.main()
