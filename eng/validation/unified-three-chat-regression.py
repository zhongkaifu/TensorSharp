#!/usr/bin/env python3
"""Three real WebUI chats, 20K allowance, disconnect cancellation and native N3/N2 evidence.

Never launches a host. Output must be fresh ignored evidence. Absolute socket
deadlines bound every request; cancelling a client is not itself server proof.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ["请详细介绍最终幻想7", "请详细介绍时间简史", "请详细介绍最终幻想8"]
BASE = Path(__file__).with_name("unified-qwen-concurrency-bench.py")
spec = importlib.util.spec_from_file_location("concurrency_client", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
CHECKER = base.FUSED_CHECKER
spec = importlib.util.spec_from_file_location("fused_checker", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Stream:
    def __init__(self, client, name, prompt, limit, plain_chat=False):
        self.client = client
        self.plain_chat = plain_chat
        self.folder = client.output / name
        self.folder.mkdir()
        self.row = dict(name=name, prompt=prompt, tokens=[], frames=[], completed=False,
                        request_id="three-chat-" + uuid.uuid4().hex)
        self.limit = limit
        self.socket = None
        self.cancel_at = None
        self.lock = threading.Lock()

    def cancel(self):
        with self.lock:
            if self.cancel_at is None:
                self.cancel_at = time.perf_counter()
            sock = self.socket
        if sock:
            try: sock.shutdown(socket.SHUT_RDWR)
            except OSError: pass

    def run(self):
        row = self.row
        client = self.client
        conn = client.connect()
        response = timer = None
        try:
            session = client.json("POST", "/api/sessions")
            row["session"] = session
            body = dict(sessionId=session["sessionId"], messages=[dict(role="user", content=row["prompt"])],
                        maxTokens=self.limit, think=False, temperature=0, topP=1, topK=0, repeatPenalty=1, seed=17)
            if self.plain_chat:
                body.update(skills=[], skills_discovery=False, multi_agent=False, tools=[])
            row["request"] = body
            save(self.folder / "request.json", row)
            row["started"] = time.perf_counter()
            conn.connect()
            with self.lock: self.socket = conn.sock
            timer = threading.Timer(client.timeout, self.cancel)
            timer.daemon = True
            timer.start()
            conn.request("POST", client.url.path.rstrip("/") + "/api/chat",
                         json.dumps(body, ensure_ascii=False).encode(),
                         {"Content-Type": "application/json", "Connection": "close",
                          "X-TensorSharp-Request-Id": row["request_id"]})
            response = conn.getresponse()
            row.update(status=response.status, headers_at=time.perf_counter())
            if response.status != 200: raise RuntimeError(f"HTTP {response.status}: {response.read(4096)!r}")
            with (self.folder / "response.sse").open("wb") as raw:
                total = 0
                while True:
                    line = response.readline(1 << 20)
                    if not line: break
                    total += len(line)
                    if total > 16 << 20: raise RuntimeError("SSE exceeds evidence bound")
                    raw.write(line)
                    if not line.startswith(b"data:"): continue
                    frame = json.loads(line[5:].strip())
                    now = time.perf_counter()
                    row["frames"].append([now, frame])
                    if frame.get("error"): raise RuntimeError(str(frame["error"]))
                    token = frame.get("token")
                    if isinstance(token, str) and token: row["tokens"].append([now, token])
                    if frame.get("done") is True:
                        row.update(completed=True, terminal=frame, terminal_at=now)
        except Exception as error:
            row["transport_error"] = f"{type(error).__name__}: {error}"
        finally:
            row["ended"] = time.perf_counter()
            if timer: timer.cancel()
            if response: response.close()
            conn.close()
            with self.lock: self.socket = None
            row["client_cancelled_at"] = self.cancel_at
            row["text"] = "".join(t for _, t in row["tokens"])
            row["phase_observations"] = base.phase_observations(row)
            save(self.folder / "result.json", row)
        return row


def native_proof(rows, records):
    """Attribute only current chats; validate successful logical graphs and positions."""
    targets = {r["request_id"]: r["session"]["sessionId"] for r in rows}
    bindings, errors = {}, []
    for record in records:
        if not checker.event_is(record, checker.MAP_EVENT): continue
        try:
            http_id = checker.request_scope(record).get("RequestId")
            if http_id not in targets: continue
            p = record["props"]
            internal, session = p["InternalRequestId"], p["SessionId"]
            if not isinstance(internal, str) or not internal or session != targets[http_id]:
                raise ValueError("Bad current HTTP/sequence binding")
            if internal in bindings and bindings[internal] != http_id: raise ValueError("Ambiguous owner")
            bindings[internal] = http_id
        except Exception as error: errors.append(str(error))
    if set(bindings.values()) != set(targets): errors.append("Missing current HTTP bindings")
    cancelled_lines = [n for n, record in enumerate(records) if checker.event_is(record, "generation.timings")
        and checker.request_scope(record).get("RequestId") == rows[0]["request_id"]
        and record.get("props", {}).get("FinishReason") == "cancelled"]
    if len(cancelled_lines) != 1: errors.append("Exactly one first-chat cancellation completion required")
    cancel_line = cancelled_lines[0] if len(cancelled_lines) == 1 else None
    n3, n2, seen, positions = [], [], set(), {}
    siblings = {r["request_id"] for r in rows[1:]}
    for line_number, record in enumerate(records):
        if not checker.event_is(record, checker.BATCH_EVENT): continue
        p = record.get("props", {})
        try:
            ids = checker.array(p, "SequenceRequestIdsJson")
            if not set(ids).intersection(bindings): continue
            before, after = checker.array(p, "PositionsBeforeJson"), checker.array(p, "PositionsAfterJson")
            count = p.get("ActiveSequenceCount")
            eid = checker.native_execution_id(p.get("NativeAcceptedExecutionId"))
            if eid in seen: raise ValueError("Repeated native execution")
            seen.add(eid)
            if (count != len(ids) or count < 2 or len(set(ids)) != count or len(before) != count or len(after) != count
                    or any(not checker.integer(a) or not checker.integer(b) or b != a + 1 for a, b in zip(before, after))
                    or p.get("ManagedPublishCompleted") is not True or p.get("NativeGraphCount") != 1
                    or not checker.integer(p.get("NativeNodeCount"), 1)):
                raise ValueError("Incomplete/invalid native joint acceptance")
            segment, kind = p.get("NativeHostSegmentCount"), p.get("NativeExecutionKind")
            if not checker.integer(segment) or kind not in (0, 1) or (kind == 1) != (segment > 0):
                raise ValueError("Invalid native seam identity")
            owners = {bindings[i] for i in ids if i in bindings}
            for internal, a, b in zip(ids, before, after):
                if internal not in bindings: continue
                if a < positions.get(internal, a): raise ValueError("Replayed position")
                positions[internal] = b
            entry = dict(line=line_number, native_id=eid, ids=ids, before=before, after=after)
            if owners == set(targets) and count == 3 and cancel_line is not None and line_number < cancel_line:
                n3.append(entry)
            if owners == siblings and count == 2 and cancel_line is not None and line_number > cancel_line:
                n2.append(entry)
        except Exception as error: errors.append(str(error))
    return dict(passed=not errors and len(n3) >= 2 and len(n2) >= 2, errors=errors,
                cancellation_line=cancel_line, n3_steps=len(n3), n2_sibling_steps=len(n2), bindings=bindings, n3=n3, n2=n2,
                scope="Repeated actual logical N3 graphs for all current chats and N2 graphs for only the surviving siblings; no claim of full-lifetime continuity or simultaneous CUDA kernels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--join-timeout", type=float, default=60)
    parser.add_argument("--expect-stall", action="store_true", help="Record the immutable pre-fix failure, not a passing fix")
    parser.add_argument("--plain-chat", action="store_true",
        help="Explicit controlled prose-only requests; disables skills/agent/tool injection. Does not qualify agentic work.")
    args = parser.parse_args()
    if not 32 <= args.max_tokens <= 20000 or not 1 <= args.timeout <= 1200 or not 1 <= args.join_timeout <= 120:
        parser.error("Invalid bounded test limits")
    lexical = Path(os.path.abspath(args.output))
    if ".." in Path(args.output).parts or not any(lexical.is_relative_to(ROOT / n) for n in ("artifacts", "docs/validation")):
        parser.error("Evidence must be ignored")
    output = lexical.resolve()
    output.mkdir(parents=True, exist_ok=False)
    client = base.Client(args.url, output, args.timeout)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args), passed=False,
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), completed=False)
    streams, futures = [], []
    try:
        report["model"] = client.json("GET", "/api/models")
        if report["model"].get("unifiedMemory") is not True: raise RuntimeError("No unified session")
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, prompt in enumerate(PROMPTS):
                stream = Stream(client, "chat-" + str(i), prompt, args.max_tokens, args.plain_chat)
                streams.append(stream)
                futures.append(pool.submit(stream.run))
                if i == 0:
                    deadline = time.perf_counter() + args.join_timeout
                    while len(stream.row["tokens"]) < 8 and not futures[0].done() and time.perf_counter() < deadline:
                        time.sleep(.1)
                    if len(stream.row["tokens"]) < 8:
                        stream.cancel()
                        raise RuntimeError("First chat did not generate trigger")
            deadline = time.perf_counter() + args.join_timeout
            while min(len(s.row["tokens"]) for s in streams) < 16 and time.perf_counter() < deadline:
                time.sleep(.1)
            report["frames_before_cancel"] = [len(s.row["tokens"]) for s in streams]
            report["queue_before_cancel"] = client.json("GET", "/api/queue/status")
            streams[0].cancel()
            report["cancel_at"] = streams[0].cancel_at
            if args.expect_stall:
                deadline = time.perf_counter() + args.join_timeout
                while time.perf_counter() < deadline and not all(f.done() for f in futures[1:]): time.sleep(.2)
                report["frames_after_cancel"] = [len(s.row["tokens"]) for s in streams]
                for s in streams[1:]: s.cancel()
            report["concurrent"] = [f.result() for f in futures]
        deadline = time.perf_counter() + 15
        while True:
            report["final_queue"] = client.json("GET", "/api/queue/status")
            if not report["final_queue"].get("busy") or time.perf_counter() >= deadline: break
            time.sleep(.2)
        raw, pin = checker.read_input(args.server_log, checker.MAX_LOG_BYTES)
        for attempt in range(10):
            try:
                records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
                if any(not isinstance(r, dict) for r in records): raise ValueError("Log record must be object")
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                if attempt == 9: raise
                time.sleep(.1)
                raw, pin = checker.read_input(args.server_log, checker.MAX_LOG_BYTES)
        (output / "server-log.jsonl").write_bytes(raw)
        report["server_log"] = pin
        report["native"] = native_proof(report["concurrent"], records)
        report["generation_timings"] = checker.generation_timing_evidence(report, records)
        first_id = streams[0].row["request_id"]
        report["cancel_server_evidence"] = [r for r in records if checker.event_is(r, "generation.timings")
                and checker.request_scope(r).get("RequestId") == first_id
                and str(r.get("props", {}).get("FinishReason", "")).lower() == "cancelled"]
        siblings = report["concurrent"][1:]
        report["completed"] = True
        report["passed"] = (not args.expect_stall and report["native"]["passed"]
                and bool(report["cancel_server_evidence"])
                and all(r.get("completed") and not r.get("transport_error") and not r["terminal"].get("error")
                        and not r["terminal"].get("aborted") and r["tokens"] for r in siblings)
                and all(sum(t > report["cancel_at"] for t, _ in r["tokens"]) >= 8 for r in siblings)
                and not report["final_queue"].get("busy") and report["final_queue"].get("pending_requests") == 0)
        report["validation_scope"] = (f"{args.max_tokens}-token output allowance admission, actual N3→N2 execution and cancelled first chat; exact prompts but fixed greedy controls. "
            + ("Controlled plain chat with skills/agents/tools disabled; no agentic qualification. " if args.plain_chat else "Default tool/agent routing retained. ")
            + "Natural EOS only when sibling terminal and attributed finish reason say EOS. Read answers separately for factual quality.")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        for stream in streams: stream.cancel()
    finally:
        for stream in streams:
            session = stream.row.get("session", {}).get("sessionId")
            if session:
                try: stream.row["session_cleanup"] = client.json("DELETE", "/api/sessions/" + session)
                except Exception as error: report.setdefault("cleanup_errors", []).append(str(error))
        if report.get("cleanup_errors"): report["passed"] = False
        save(output / "report.json", report)
    print(json.dumps({k: report.get(k) for k in ("passed", "error", "frames_before_cancel", "frames_after_cancel")}, ensure_ascii=False))
    return 0 if report.get("passed") else 1


if __name__ == "__main__": raise SystemExit(main())
