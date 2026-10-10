#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Thinking-off Nemotron-H Reasoning-128K end to end, against a RUNNING TensorSharp.Server.

Usage:
  nemotron-thinkoff-e2e.py --base http://127.0.0.1:5077/ --out artifacts/nemotron-thinkoff-e2e
                           [--repeats 3] [--skip-webui] [--skip-openai] [--server-log server.log]

The server must have a Nemotron-H Reasoning-128K GGUF loaded, and --code-exec on for the
tool chats. With thinking off this model sometimes reasons after its closed
<think></think> and closes the block itself, mostly in tool rounds; that reasoning and the
literal </think> used to be the answer. What it checks:

  webui    /api/chat as the pages read it (token appends, replace sets the whole answer,
           thinking appends) and as they send it back (content and thinking). A plain
           four-turn chat, then a tool chat on an attached file, repeated, thinking off,
           and both once more with thinking on. Fails if any answer ever SHOWED a </think>, or a
           finished answer still holds reasoning the turn's thinking also holds. Reports
           how often text was taken back (replace frames), and every turn's prompt reuse:
           a follow-up that stopped reusing the cache after a retraction is the silent
           failure to look for.
  openai   /v1/chat/completions streamed, thinking off: the first-round prompt that
           reasoned past its closed block over the API (tools in the system prompt,
           {'reasoning': False}), seeds 1-3, and three ordinary answers (one sentence,
           about 400 words, about 900 words). Fails if a content delta holds </think>, or if
           the stream ever went more than --max-gap seconds without a byte: while undecided
           text is held the server sends an SSE ': keep-alive' comment every 15 s, so a
           proxy's idle timeout does not cut the stream. Reports how long the first content
           delta took and how much it carried, which is the price of holding undecided text
           for a stream that cannot take it back, and how many keep-alives came before it.
           With --server-log (the running server's log file), a one-round case also reports
           the server's own time to first token and the hold: first content minus that.
  json     response_format with thinking off, json_schema (streamed and not) and
           json_object (streamed), for an object whose strings quote both tags. The grammar
           shapes the reply from its first token, so it must arrive whole: fails on a
           non-200 answer or content that is not the requested JSON.

Writes turns.jsonl and summary.json under --out (keep it under ignored artifacts/).
Exits non-zero when a check fails. Whether the model reasons past its block on a given
run is up to sampling: a run with no stray close validates nothing about retraction, and
the summary says how many it saw.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

SHA_TASK = ("Run a shell command that prints the SHA-256 hex digest of the exact ASCII string tensoragent "
            "(no trailing newline), then reply with only the first 12 hex characters.")
AB_TOOLS = [
    {"name": "shell", "description": "Run a bash command in the user's working directory and return its standard output and exit code. Use it to compute values, inspect files and run programs.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to run."}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a text file from the working directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Relative path of the file."}}, "required": ["path"]}},
    {"name": "write_file", "description": "Create or overwrite a text file in the working directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "web_fetch", "description": "Fetch a web page and return its text.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
]


def hermes_system(tools):
    """The tool-format A/B's Hermes system prompt, byte for byte, with reasoning off."""
    s = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
         "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
    for t in tools:
        s += "\n" + json.dumps({"type": "function", "function": t})
    s += ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
          "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
          "<args-json-object>}\n</tool_call>\nFunction results are returned to you inside "
          "<tool_response></tool_response> tags.")
    return s + "\n{'reasoning': False}"


class Server:
    def __init__(self, base):
        self.base = base if base.endswith("/") else base + "/"

    def open(self, path, body=None, content_type="application/json", timeout=900):
        data = None
        headers = {}
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path.lstrip("/"), data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        return urllib.request.urlopen(req, timeout=timeout)

    def json(self, path, body=None):
        with self.open(path, body if body is not None else b"") as r:
            return json.loads(r.read().decode())

    def upload(self, path):
        boundary = "----thinkoff" + uuid.uuid4().hex
        payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
                   f"Content-Type: application/octet-stream\r\n\r\n").encode()
        payload += open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        with self.open("api/upload", payload, f"multipart/form-data; boundary={boundary}") as r:
            return json.loads(r.read().decode())

    def sse(self, path, body, gaps=None):
        """The stream's data frames with their arrival times. A keep-alive comment comes
        as {"keepAlive": True}; GAPS, when given, collects the time between any two lines."""
        start = time.monotonic()
        last = start
        with self.open(path, body) as r:
            for raw in r:
                now = time.monotonic()
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if gaps is not None and line:
                    gaps.append(now - last)
                    last = now
                if line.startswith(":"):
                    yield now - start, {"keepAlive": True}
                elif line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    yield now - start, json.loads(line[6:])


def webui_turn(server, session, history, think, max_tokens):
    """One /api/chat turn, read the way both pages read it."""
    body = {"sessionId": session, "messages": history, "maxTokens": max_tokens, "think": think}
    answer, thinking, replaces, shown_close, first, done, tools, error = "", "", 0, False, None, {}, 0, None
    for at, frame in server.sse("api/chat", body):
        if frame.get("thinking"):
            thinking += frame["thinking"]
            first = first if first is not None else at
        if isinstance(frame.get("token"), str):
            answer += frame["token"]
            first = first if first is not None else at
        if isinstance(frame.get("replace"), str):
            answer = frame["replace"]
            replaces += 1
        shown_close = shown_close or "</think>" in answer
        if frame.get("skill_step") or frame.get("tool_calls"):
            tools += 1
        if frame.get("error"):
            error = frame["error"]
        if frame.get("done") is True:
            done = frame
    return {"answer": answer, "thinking": thinking, "replaces": replaces, "closeShown": shown_close,
            "ttft": first, "tools": tools, "error": error, "promptTokens": done.get("promptTokens", 0),
            "reused": done.get("kvReusedTokens", 0), "reusePct": done.get("kvReusePercent", 0.0),
            "tokenCount": done.get("tokenCount", 0)}


def leaked_reasoning(turn):
    """A finished answer that still holds reasoning the turn reported as reasoning."""
    thinking = turn["thinking"].strip()
    return bool(thinking) and len(thinking) > 40 and thinking[:40] in turn["answer"]


def run_webui(server, out, repeats, notes_path):
    rows = []

    def chat(label, think, turns):
        session = server.json("api/sessions")["sessionId"]
        history = []
        for prompt, extra in turns:
            msg = {"role": "user", "content": prompt}
            msg.update(extra or {})
            history.append(msg)
            turn = webui_turn(server, session, history, think, 2048 if think else 512)
            entry = {"role": "assistant", "content": turn["answer"]}
            if turn["thinking"]:
                entry["thinking"] = turn["thinking"]
            history.append(entry)
            row = {"surface": "webui", "chat": label, "think": think, "turn": prompt[:48],
                   "answer": turn["answer"][:160], "thinkingChars": len(turn["thinking"]),
                   "replaces": turn["replaces"], "closeShown": turn["closeShown"],
                   "leaked": leaked_reasoning(turn), "tools": turn["tools"], "error": turn["error"],
                   "promptTokens": turn["promptTokens"], "reused": turn["reused"],
                   "reusePct": round(turn["reusePct"], 1), "ttft": round(turn["ttft"] or 0, 2),
                   "tokenCount": turn["tokenCount"]}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

    capitals = [("What is the capital of France? One word.", None),
                ("And of Italy? One word.", None),
                ("And of Japan? One word.", None),
                ("Which of those three cities did I ask about first? One word.", None)]
    chat("capitals", False, capitals)
    chat("capitals", True, capitals)
    for think, count in ((False, repeats), (True, 1)):
        for i in range(count):
            up = server.upload(notes_path)
            att = {"attachments": [{"file": up["file"], "fileName": "notes.txt", "mediaType": "file"}],
                   "attachmentPaths": [up["file"]]}
            chat(f"tool-{i + 1}", think, [("How many lines does the attached notes.txt have? Count them with a tool, don't guess.", att),
                                          ("What is its last line?", None)])
    return rows


def openai_stream(server, model, messages, seed, max_tokens, response_format=None):
    body = {"model": model, "messages": messages, "temperature": 0.7, "top_p": 0.95, "top_k": 40,
            "seed": seed, "max_tokens": max_tokens, "stream": True}
    if response_format is not None:
        body["response_format"] = response_format
    content, reasoning, first_any, first_content, first_size, finish = "", "", None, None, 0, None
    total = 0.0
    gaps, keep_alives = [], 0
    for at, event in server.sse("v1/chat/completions", body, gaps):
        total = at
        if event.get("keepAlive"):
            keep_alives += 1 if first_content is None else 0
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason") or finish
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
                first_any = first_any if first_any is not None else at
            if delta.get("content"):
                if first_content is None:
                    first_content, first_size = at, len(delta["content"])
                first_any = first_any if first_any is not None else at
                content += delta["content"]
    return {"content": content, "reasoning": reasoning, "firstAny": first_any, "firstContent": first_content,
            "firstContentChars": first_size, "total": total, "finish": finish,
            "keepAlivesBeforeContent": keep_alives, "maxGap": max(gaps) if gaps else 0.0}


def completions_since(server_log, offset):
    """The server's chat.complete records written after byte OFFSET of its log: (tokens, ttftMs)."""
    if not server_log or not os.path.exists(server_log):
        return []
    with open(server_log, "rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", errors="replace")
    return [(int(m.group(1)), int(m.group(2)))
            for m in re.finditer(r"chat\.complete tokens=(\d+) .*?ttftMs=(\d+)", text)]


def run_openai(server, out, server_log, max_gap):
    rows = []
    with server.open("api/tags") as r:
        model = json.loads(r.read().decode())["models"][0]["name"]
    cases = [(f"ab-sha-seed{seed}", [{"role": "system", "content": hermes_system(AB_TOOLS)},
                                     {"role": "user", "content": SHA_TASK}], seed, 512) for seed in (1, 2, 3)]
    cases.append(("ordinary-sentence", [{"role": "user", "content":
                  "In one sentence, what does the HTTP status code 404 mean?"}], 7, 200))
    cases.append(("ordinary-400w", [{"role": "user", "content":
                  "Explain in about 400 words how TCP slow start and congestion avoidance work together."}], 7, 900))
    cases.append(("ordinary-900w", [{"role": "user", "content":
                  "Explain in about 900 words how a B-tree index works in a relational database, "
                  "from page layout to splits and range scans."}], 7, 1600))
    for label, messages, seed, max_tokens in cases:
        offset = os.path.getsize(server_log) if server_log and os.path.exists(server_log) else 0
        r = openai_stream(server, model, messages, seed, max_tokens)
        rounds = completions_since(server_log, offset)
        server_ttft = rounds[0][1] / 1000.0 if rounds else None
        hold = (round(r["firstContent"] - server_ttft, 2)
                if len(rounds) == 1 and r["firstContent"] is not None else None)
        row = {"surface": "openai", "case": label, "contentHasClose": "</think>" in r["content"],
               "reasoningChars": len(r["reasoning"]), "contentChars": len(r["content"]),
               "firstContentSec": None if r["firstContent"] is None else round(r["firstContent"], 2),
               "firstContentChars": r["firstContentChars"], "totalSec": round(r["total"], 2),
               "rounds": len(rounds), "serverTtftSec": server_ttft, "holdSec": hold,
               "keepAlivesBeforeContent": r["keepAlivesBeforeContent"], "maxGapSec": round(r["maxGap"], 2),
               "gapTooLong": r["maxGap"] > max_gap,
               "finish": r["finish"], "content": r["content"][:160], "reasoningHead": r["reasoning"][:120]}
        rows.append(row)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


TAGS_PROMPT = ("Return a JSON object whose \"tags\" array holds exactly these two strings, in this order: "
               "<think> and </think>. Set \"count\" to the number of strings in the array.")
TAGS_SCHEMA = {"type": "json_schema", "json_schema": {"name": "tags", "strict": True, "schema": {
    "type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}, "count": {"type": "integer"}},
    "required": ["tags", "count"], "additionalProperties": False}}}


def run_json(server, out, repeats):
    """response_format with thinking off: the reply is the JSON from its first token."""
    rows = []
    with server.open("api/tags") as r:
        model = json.loads(r.read().decode())["models"][0]["name"]
    messages = [{"role": "user", "content": TAGS_PROMPT}]
    for seed in range(1, repeats + 1):
        for label, fmt, stream in (("json_schema-stream", TAGS_SCHEMA, True), ("json_schema", TAGS_SCHEMA, False),
                                   ("json_object-stream", {"type": "json_object"}, True)):
            status, content = 200, ""
            try:
                if stream:
                    content = openai_stream(server, model, messages, seed, 200, fmt)["content"]
                else:
                    body = {"model": model, "messages": messages, "temperature": 0.7, "seed": seed,
                            "max_tokens": 200, "stream": False, "response_format": fmt}
                    content = server.json("v1/chat/completions", body)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                status, content = e.code, e.read().decode(errors="replace")[:300]
            try:
                value = json.loads(content)
            except ValueError:
                value = None
            tags = value.get("tags") if isinstance(value, dict) else None
            row = {"surface": "json", "case": label, "seed": seed, "status": status,
                   "valid": isinstance(tags, list) and isinstance(value.get("count"), int),
                   "quotesClose": isinstance(tags, list) and "</think>" in tags, "content": content[:200]}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=os.path.join("artifacts", "nemotron-thinkoff-e2e"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-webui", action="store_true")
    ap.add_argument("--skip-openai", action="store_true")
    ap.add_argument("--skip-json", action="store_true")
    ap.add_argument("--max-gap", type=float, default=20.0,
                    help="seconds a stream may go without a byte (keep-alives come every 15 s)")
    ap.add_argument("--server-log", help="the running server's log file, for its time to first token")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    server = Server(args.base)
    notes = os.path.join(tempfile.mkdtemp(prefix="thinkoff-"), "notes.txt")
    with open(notes, "w") as f:
        f.write("alpha\nbravo\ncharlie\ndelta\necho\nfoxtrot\ngolf\n")
    with open(os.path.join(args.out, "turns.jsonl"), "a", encoding="utf-8") as out:
        web = [] if args.skip_webui else run_webui(server, out, args.repeats, notes)
        api = [] if args.skip_openai else run_openai(server, out, args.server_log, args.max_gap)
        js = [] if args.skip_json else run_json(server, out, args.repeats)
    failures = [r for r in web if r["closeShown"] or r["leaked"] or r["error"]]
    failures += [r for r in api if r["contentHasClose"] or r["gapTooLong"]]
    failures += [r for r in js if r["status"] != 200 or not r["valid"]]
    follow_ups = [r for r in web if r["turn"].startswith(("And of", "Which of", "What is its"))]
    summary = {
        "webuiTurns": len(web),
        "webuiTurnsWithRetraction": sum(1 for r in web if r["replaces"] > 0),
        "webuiCloseShown": sum(1 for r in web if r["closeShown"]),
        "webuiLeaked": sum(1 for r in web if r["leaked"]),
        "followUpReusePct": [r["reusePct"] for r in follow_ups if not r["think"]],
        "followUpReusePctThinkingOn": [r["reusePct"] for r in follow_ups if r["think"]],
        "openaiCases": len(api),
        "openaiWithReasoning": sum(1 for r in api if r["reasoningChars"] > 0),
        "openaiContentWithClose": sum(1 for r in api if r["contentHasClose"]),
        "openaiHoldSec": {r["case"]: r["holdSec"] for r in api if r["holdSec"] is not None},
        "openaiKeepAlivesBeforeContent": {r["case"]: r["keepAlivesBeforeContent"] for r in api},
        "openaiMaxGapSec": {r["case"]: r["maxGapSec"] for r in api},
        "jsonCases": len(js),
        "jsonValid": sum(1 for r in js if r["status"] == 200 and r["valid"]),
        "jsonQuotingTheClose": sum(1 for r in js if r["quotesClose"]),
        "failures": len(failures),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
