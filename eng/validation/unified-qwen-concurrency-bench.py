#!/usr/bin/env python3
"""Measure actual Web UI token-stream overlap in two distinct Qwen sessions.

Connects to an already running host. Does not start/stop servers. The second
request starts after eight nonempty token frames from the first, not merely
when both sockets are open. Optional serial controls require exact greedy text
parity. Finite token limits qualify bounded generation, not natural EOS.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid

PROMPTS = ["请详细介绍最终幻想7", "请详细介绍时间简史"]
ROOT = Path(__file__).resolve().parents[2]
FUSED_CHECKER = Path(__file__).with_name("verify-fused-batch-evidence.py")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def overlap_gate(rows):
    if len(rows) != 2 or any(r.get("error") or r.get("cleanup_error") or not r.get("completed") or not r.get("tokens") for r in rows):
        return {"passed": False, "reason": "Both streams must complete and emit tokens"}
    for row in rows:
        times = [row.get("terminal_at")] + [t for t, _ in row["tokens"]]
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in times):
            return {"passed": False, "reason": "Finite terminal and token timestamps are required"}
        if row["tokens"][-1][0] > row["terminal_at"] or any(
                a[0] > b[0] for a,b in zip(row["tokens"],row["tokens"][1:])):
            return {"passed": False, "reason": "Invalid token/terminal ordering"}
    a, b = rows
    start, end = max(a["tokens"][0][0], b["tokens"][0][0]), min(a["terminal_at"], b["terminal_at"])
    progress = [sum(start <= t < end for t, _ in r["tokens"]) for r in rows]
    return {"passed": end > start and all(n >= 2 for n in progress),
            "overlap_seconds": max(0, end-start), "frames_in_overlap": progress,
            "second_first_token_before_first_end": b["tokens"][0][0] < a["terminal_at"],
            "scope": "Both requests emit at least two content frames while both are active; no claim of simultaneous GPU kernels."}


def phase_observations(row):
    """Client content timing is observable; it does not isolate prefill/queue."""
    def elapsed(start, end):
        return end-start if (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and end >= start) else None
    tokens = row.get("tokens", [])
    first, last = (tokens[0][0], tokens[-1][0]) if tokens else (None, None)
    terminal = row.get("terminal", {})
    return {"dispatch_to_headers_seconds": elapsed(row.get("started"), row.get("headers_at")),
            "dispatch_to_first_content_seconds": elapsed(row.get("started"), first),
            "first_to_last_content_seconds": elapsed(first, last),
            "first_content_to_terminal_seconds": elapsed(first, row.get("terminal_at")),
            "content_frames": len(tokens), "server_token_count": terminal.get("tokenCount"),
            "server_elapsed_seconds": terminal.get("elapsed"),
            "server_prompt_tokens": terminal.get("promptTokens"),
            "server_kv_reused_tokens": terminal.get("kvReusedTokens"),
            "server_truncated": terminal.get("truncated"),
            "scope": "Client content frames are not tokenizer tokens; first-content latency includes queue, preparation, prefill, hidden thinking/tool rounds and network time. Server fields retain their declared scope; no isolated PP/TG rate is inferred."}


def capture_fused_evidence(report, server_log, output, minimum_steps=2):
    checker_hash = hashlib.sha256(FUSED_CHECKER.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("fused_batch_evidence", FUSED_CHECKER)
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    raw, pin = checker.read_input(server_log, checker.MAX_LOG_BYTES)
    # A finished SSE can be followed by a final HTTP-completed logger write.
    # Retry boundedly for a complete JSONL tail; never discard malformed rows.
    for attempt in range(10):
        (output / "fused-server-log.jsonl").write_bytes(raw)
        try:
            records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
            if any(not isinstance(record, dict) for record in records):
                raise ValueError("Every JSONL record must be an object")
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            if attempt == 9: raise
            time.sleep(.1)
            raw, pin = checker.read_input(server_log, checker.MAX_LOG_BYTES)
    result = checker.verify(report, records, minimum_steps)
    result["server_log"] = pin
    result["tool_sha256"] = checker_hash
    if hashlib.sha256(FUSED_CHECKER.read_bytes()).hexdigest() != checker_hash:
        raise RuntimeError("Fused checker changed during verification")
    save(output / "fused-evidence.json", result)
    return result


def attach_fused_evidence(report, evidence):
    """Keep diagnostic engagement visible without weakening the final gate."""
    report["fused_evidence"] = evidence
    report["batched_fused_graph_verified"] = evidence.get("passed") is True
    for key in ("actual_joint_engagement_verified", "full_interval_continuity_verified"):
        report[key] = evidence.get(key) is True
    report["fused_joint_steps"] = evidence.get("joint_steps")
    report["fused_continuity_gaps"] = evidence.get("continuity_gaps")
    report["fused_contiguous_joint_runs"] = evidence.get("contiguous_joint_runs")
    report["generation_timings"] = evidence.get("generation_timings")


class Client:
    def __init__(self, url, output, timeout):
        self.url = urlsplit(url)
        if self.url.scheme not in ("http", "https") or not self.url.hostname or self.url.username:
            raise ValueError("Invalid HTTP URL")
        self.output, self.timeout = output, timeout

    def connect(self):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=self.timeout)

    def json(self, method, route, body=None):
        conn = self.connect()
        sock = None
        timer = None
        def expire():
            if sock:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
        try:
            conn.connect(); sock = conn.sock
            timer = threading.Timer(self.timeout, expire); timer.daemon = True; timer.start()
            conn.request(method, self.url.path.rstrip("/")+route,
                         None if body is None else json.dumps(body).encode(), {"Content-Type":"application/json", "Connection":"close"})
            response = conn.getresponse()
            raw = response.read((8 << 20)+1)
            if len(raw) > 8 << 20: raise RuntimeError("Control response exceeds evidence bound")
            if response.status != 200: raise RuntimeError(f"{route}: HTTP {response.status} {raw[:500]!r}")
            return json.loads(raw)
        finally:
            if timer: timer.cancel()
            conn.close()

    def stream(self, name, prompt, max_tokens, release=None):
        folder = self.output/name
        folder.mkdir()
        row = {"name":name, "tokens":[], "frames":[], "completed":False,
               "request_id":"concurrency-"+uuid.uuid4().hex}
        session_id=None
        conn=self.connect(); response=None; sock=None; timer=None
        def expire():
            if sock:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
        try:
            session = self.json("POST", "/api/sessions")
            session_id = session["sessionId"]
            body = {"sessionId":session_id, "messages":[{"role":"user", "content":prompt}],
                    "maxTokens":max_tokens, "think":False, "temperature":0, "topP":1,
                    "topK":0, "repeatPenalty":1, "seed":17}
            row.update(session=session, request=body)
            save(folder/"request.json", row)
            row["started"] = time.perf_counter()
            conn.connect(); sock=conn.sock
            timer=threading.Timer(self.timeout, expire); timer.daemon=True; timer.start()
            conn.request("POST", self.url.path.rstrip("/")+"/api/chat", json.dumps(body,ensure_ascii=False).encode(),
                         {"Content-Type":"application/json", "Connection":"close", "X-TensorSharp-Request-Id":row["request_id"]})
            response=conn.getresponse(); row["status"]=response.status; row["headers_at"]=time.perf_counter()
            if response.status!=200: raise RuntimeError(f"HTTP {response.status}: {response.read(4096)!r}")
            with (folder/"response.sse").open("wb") as raw:
                total=0
                while True:
                    line=response.readline(1 << 20)
                    if not line: break
                    total+=len(line)
                    if total > 8 << 20: raise RuntimeError("Response exceeds evidence bound")
                    raw.write(line)
                    if not line.startswith(b"data:"): continue
                    frame=json.loads(line[5:].strip())
                    now=time.perf_counter(); row["frames"].append([now,frame])
                    if frame.get("error"): raise RuntimeError(str(frame["error"]))
                    if frame.get("aborted"): raise RuntimeError("Server aborted stream")
                    token=frame.get("token")
                    if isinstance(token,str) and token:
                        row["tokens"].append([now,token])
                        if release and len(row["tokens"])>=8: release.set()
                    if frame.get("done") is True:
                        row["completed"]=True; row["terminal"]=frame; row["terminal_at"]=now
            row["text"]="".join(t for _,t in row["tokens"])
        except Exception as error:
            row["error"]=f"{type(error).__name__}: {error}"
        finally:
            row["ended"]=time.perf_counter()
            if timer: timer.cancel()
            if response: response.close()
            conn.close()
            if release: release.set()
            if session_id:
                try: row["session_cleanup"]=self.json("DELETE", "/api/sessions/"+session_id)
                except Exception as error: row["cleanup_error"]=str(error)
            if row["tokens"]: row["ttft_seconds"]=row["tokens"][0][0]-row["started"]
            row["elapsed_seconds"]=row["ended"]-row.get("started",row["ended"])
            row["phase_observations"]=phase_observations(row)
            save(folder/"result.json",row)
        return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--max-tokens",type=int,default=160)
    parser.add_argument("--timeout",type=float,default=300)
    parser.add_argument("--compare-serial",action="store_true")
    parser.add_argument("--server-log", help="Actual JSONL with opt-in TS_BATCHED_FUSED_TRACE=1; preserves a bounded full copy")
    parser.add_argument("--require-fused-graph", "--require-fused-batch", dest="require_fused_graph", action="store_true", help="Require repeated actual native joint graphs for the two current HTTP generations")
    parser.add_argument("--min-joint-steps", type=int, default=2)
    parser.add_argument("--expect-interleaved", "--expect-concurrent", dest="expect_interleaved", action="store_true",
                        help="Require overlapping generation progress; this does not prove a fused multi-request graph")
    args=parser.parse_args()
    if not (0<args.max_tokens<=2048 and 0<args.timeout<=600): parser.error("Invalid bounded tokens/timeout")
    if args.min_joint_steps < 2 or (args.require_fused_graph and not args.server_log):
        parser.error("Fused qualification needs --server-log and --min-joint-steps >= 2")
    lexical=Path(os.path.abspath(args.output))
    if ".." in Path(args.output).parts or not any(lexical.is_relative_to(ROOT/p) for p in ("artifacts","docs/validation")):
        parser.error("Output must be a fresh ignored artifact directory")
    output=lexical.resolve(); output.mkdir(parents=True,exist_ok=False)
    client=Client(args.url,output,args.timeout)
    report={"started_utc":datetime.now(timezone.utc).isoformat(),"args":vars(args),"completed":False,
            "tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "batched_fused_graph_verified":False,
            "validation_scope":"HTTP generation overlap and optional greedy parity only; fused multi-request execution needs independent engine/native evidence."}
    try:
        report["model"]=client.json("GET","/api/models")
        if report["model"].get("unifiedMemory") is not True: raise RuntimeError("Unified session is not active")
        ready=threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(client.stream,"concurrent-ff7",PROMPTS[0],args.max_tokens,ready)
            if not ready.wait(args.timeout): raise TimeoutError("First stream did not reach trigger")
            second=pool.submit(client.stream,"concurrent-hawking",PROMPTS[1],args.max_tokens)
            report["concurrent"]=[first.result(),second.result()]
        report["overlap"]=overlap_gate(report["concurrent"])
        if args.compare_serial:
            report["serial"]=[client.stream("serial-"+str(i),p,args.max_tokens) for i,p in enumerate(PROMPTS)]
            report["greedy_text_parity"]=[a.get("text")==b.get("text") and bool(a.get("text"))
                for a,b in zip(report["concurrent"],report["serial"])]
        report["final_queue"]=client.json("GET","/api/queue/status")
        all_rows=report["concurrent"]+report.get("serial",[])
        report["passed"]=all(r["completed"] and not r.get("error") and not r.get("cleanup_error") for r in all_rows)
        report["passed"] &= not args.expect_interleaved or report["overlap"]["passed"]
        report["passed"] &= not args.compare_serial or all(report["greedy_text_parity"])
        report["completed"]=True
        report["http_checks_passed"]=report["passed"]
        if args.server_log:
            attach_fused_evidence(report,capture_fused_evidence(report,args.server_log,output,args.min_joint_steps))
            report["validation_scope"]="HTTP progress/optional greedy parity and separately verified native joint logical graph engagement. Host seams may require multiple backend dispatches; bounded outputs do not qualify complete answer quality."
        report["passed"] &= not args.require_fused_graph or report["batched_fused_graph_verified"]
    except Exception as error: report.update(error=f"{type(error).__name__}: {error}",passed=False)
    finally: save(output/"report.json",report)
    print(json.dumps({k:report.get(k) for k in ("completed","passed","error","overlap")},ensure_ascii=False))
    return 0 if report.get("passed") else 1


if __name__=="__main__": raise SystemExit(main())
