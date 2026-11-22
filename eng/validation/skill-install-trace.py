#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
# Licensed under the BSD-3-Clause license in the repository root.
"""Send one Web UI chat turn and record its whole skill tool-call trace.

Works against TensorAgent's loopback host (the Mac app launched by
TensorAgent/scripts/run-mac.sh, or the simulator) and against TensorSharp.Server,
which share the /api/chat route. The goal is to see what a turn really does with
the skill hub: which skills_* tools were declared, each call's arguments, and the
result the host returned, ending with the answer.

Two sources make up the trace, both read after the turn ends:
  * the SSE frames (skill_step, tool_progress, tool_calls, the answer and done);
  * the host's console log (--log): every round's `chat.start` line carries the
    declared tool count and the prompt (each message capped at 512 characters by
    InferenceTelemetry), its `chat.complete` line the model's full output (so the
    tool calls with their arguments), and `skills.tool` lines name each executed
    call with ok/bytes. The tool results are read back out of the NEXT round's
    prompt, which is where the host put them.

Usage:
  skill-install-trace.py --log <app stdout log> --prompt "..." --out trace.json
  skill-install-trace.py --base http://127.0.0.1:5000/ --log server.log --prompt "..." --out trace.json
  skill-install-trace.py --log device-console.log --log-only --out trace.json

--log-only sends nothing: it reads a whole log that a turn already wrote, such as the
console of a device launched with TENSORAGENT_DEMO_PROMPT (a phone's loopback host is not
reachable from the Mac, so its trace can only come from the console).

The --log of TensorAgent also supplies the Debug build's entry URL and launch token.
Output belongs in ignored artifacts/ or docs/validation/. This is a model integration
check, not a unit test: it records, and never decides pass/fail on the model's prose.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def entry_from_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    found = re.findall(r"entry URL (http://127\.0\.0\.1:\d+/)\?token=([0-9a-f]+)", text)
    return found[-1] if found else (None, None)


class Host:
    def __init__(self, base, token):
        self.base = base if base.endswith("/") else base + "/"
        self.cookie = f"tensoragent_token={token}" if token else None

    def request(self, method, path, body=None, timeout=60):
        headers = {"Cookie": self.cookie} if self.cookie else {}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        elif method == "POST":
            data = b""
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, headers=headers, method=method)
        return urllib.request.urlopen(req, timeout=timeout)

    def json(self, method, path, body=None):
        with self.request(method, path, body) as response:
            return json.loads(response.read().decode())


def run_turn(host, prompt, max_tokens, think, timeout):
    created = host.json("POST", "api/sessions?conversation=new")
    session = created["sessionId"]
    body = {"sessionId": session, "messages": [{"role": "user", "content": prompt}],
            "maxTokens": max_tokens, "think": think, "knowledgeCollections": []}
    frames, answer, thinking = [], [], []
    start = time.monotonic()
    error = None
    try:
        with host.request("POST", "api/chat", body, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                try:
                    frame = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                at = round(time.monotonic() - start, 2)
                if isinstance(frame.get("token"), str):
                    answer.append(frame["token"])
                    continue
                if isinstance(frame.get("thinking"), str):
                    thinking.append(frame["thinking"])
                    continue
                if isinstance(frame.get("replace"), str):
                    answer = [frame["replace"]]
                if frame.get("tool_progress") == "running" and frame.get("text") and not frame.get("detail"):
                    continue
                frames.append({"t": at, **frame})
    except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
        error = f"transport: {ex}"
    return {"sessionId": session, "seconds": round(time.monotonic() - start, 1), "error": error,
            "answer": "".join(answer), "thinking": "".join(thinking), "frames": frames}


ENTRY = re.compile(r"^(trce|dbug|info|warn|fail|crit): ([^\[]+)\[(\d+)\]\s*$")
# The Mac/iOS app's stdout goes through NSLog, which prefixes every line and splits a long
# one into chunks that each get the prefix again.
NSLOG = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ [^\[\s]+\[\d+:(\d+)\] ?")


def log_entries(path, offset):
    """Console-logger entries written after byte `offset`: (level, category, message).

    An NSLog chunk continues the entry its own thread was writing; other threads' lines
    can sit between the chunks."""
    with open(path, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", errors="replace")
    entries, open_by_thread, last_length, current = [], {}, {}, None
    for raw in text.splitlines():
        prefixed = NSLOG.match(raw)
        thread = prefixed.group(1) if prefixed else ""
        line = raw[prefixed.end():] if prefixed else raw
        chunk = prefixed is not None and last_length.get(thread, 0) >= 4000
        last_length[thread] = len(line.encode("utf-8"))
        head = None if chunk else ENTRY.match(line)
        if head:
            current = [head.group(1), head.group(2).strip(), []]
            entries.append(current)
            open_by_thread[thread] = current
            continue
        owner = open_by_thread.get(thread)
        if owner is None:
            # File-logger lines (TensorSharp.Server) are one line each.
            if " chat.start " in line or " chat.complete " in line or " skills." in line:
                entries.append(["file", "", [line]])
            continue
        if chunk and owner[2]:
            owner[2][-1] += line   # an NSLog chunk of the same line
        elif line.startswith("      ") or not owner[2]:
            owner[2].append(line.strip())
        else:
            open_by_thread.pop(thread, None)
    return [(level, category, " ".join(body)) for level, category, body in entries]


def rounds_from_log(path, offset):
    rounds, tools = [], []
    for level, category, message in log_entries(path, offset):
        if "chat.start " in message:
            m = re.search(r"tools=(\d+)", message)
            full = re.search(r"fullInput=(.*)$", message)
            rounds.append({"tools": int(m.group(1)) if m else None, "input": full.group(1) if full else None})
        elif "chat.complete " in message or "chat.cancelled " in message:
            out = re.search(r'assistantOutput="(.*)"\s*$', message)
            if rounds:
                rounds[-1]["output"] = out.group(1) if out else message
        elif "skills.tool " in message or "skills.loop" in message or "skills.completion" in message:
            tools.append(message[message.find("skills."):])
    return rounds, tools


def tool_results(rounds):
    """The tool results the host fed back, read from the last round's prompt."""
    if not rounds or not rounds[-1].get("input"):
        return []
    try:
        messages = json.loads(rounds[-1]["input"])
    except json.JSONDecodeError:
        return []
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", help="host URL; default: the entry URL in --log")
    p.add_argument("--log", required=True, help="the host's console log (read for the rounds and, for TensorAgent, the token)")
    p.add_argument("--connection-file", help="connection.json of eng/validation/TensorAgentHost (its baseUrl and cookie)")
    p.add_argument("--prompt", help="the turn to send (required unless --log-only)")
    p.add_argument("--log-only", action="store_true", help="send nothing; trace the turns already in --log")
    p.add_argument("--max-tokens", type=int, default=262144)
    p.add_argument("--think", action="store_true")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    if a.log_only:
        rounds, tools = rounds_from_log(a.log, 0)
        trace = {"log": a.log, "rounds": rounds, "toolLog": tools, "toolResults": tool_results(rounds)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(rounds)} rounds")
        for line in tools:
            print("  log:", line[:200])
        return
    if not a.prompt:
        p.error("--prompt is required unless --log-only")
    if a.connection_file:
        connection = json.loads(Path(a.connection_file).read_text(encoding="utf-8"))
        base, token = connection["baseUrl"], connection["cookie"].split("=", 1)[1]
    else:
        base, token = (a.base, None) if a.base else entry_from_log(a.log)
    if not base:
        sys.exit(f"No 'entry URL' line in {a.log}; pass --base for a TensorSharp.Server.")
    offset = Path(a.log).stat().st_size
    turn = run_turn(Host(base, token), a.prompt, a.max_tokens, a.think, a.timeout)
    time.sleep(1.0)  # the last log lines are flushed after the stream closes
    rounds, tools = rounds_from_log(a.log, offset)
    trace = {"prompt": a.prompt, **turn, "rounds": rounds, "toolLog": tools, "toolResults": tool_results(rounds)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")

    steps = [f for f in turn["frames"] if "skill_step" in f]
    print(f"{turn['seconds']} s, {len(rounds)} rounds, error={turn['error']}")
    for f in steps:
        print(f"  round {f.get('round')}: {f['skill_step']} skill={f.get('skill')} detail={f.get('detail')} ok={f.get('ok')}")
    for line in tools:
        print("  log:", line[:200])
    print("answer:", turn["answer"][:600].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
