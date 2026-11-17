#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Shared OpenAI-compatible client for the LFM3 end-to-end suite (standard library only).

One client talks to TensorSharp.Server.Host and to llama-server alike, so every scenario and
quality set sends the two engines the same request body. Reasoning is read from
`reasoning_content` (streaming deltas and messages; llama.cpp, vLLM and SGLang spell it so)
and, failing that, from TensorSharp's older non-streaming `thinking` field.
"""
import http.client
import json
import math
import time
import urllib.parse


class Client:
    def __init__(self, url, model=None, timeout=3600):
        self.url = url.rstrip("/")
        u = urllib.parse.urlparse(self.url)
        self.host, self.port = u.hostname, u.port
        self.timeout = timeout
        self.model = model or self.models()[0]

    def _conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout or self.timeout)

    def get(self, path):
        c = self._conn(60)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()

    def models(self):
        status, data = self.get("/v1/models")
        return [m["id"] for m in json.loads(data)["data"]]

    def post(self, path, body, timeout=None):
        c = self._conn(timeout)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()

    # ------------------------------------------------------------------ chat completions

    def chat(self, messages, *, stream=False, thinking=False, sampling=None, on_delta=None, abort_after=None, **extra):
        """One /v1/chat/completions call.

        thinking: True/False sends chat_template_kwargs.enable_thinking (both servers read it);
        None sends no switch (the server's default, or a "think" field in extra).
        sampling: dict of temperature/top_k/top_p/min_p/repeat_penalty/seed (sent as given; both
        servers read these names). Returns content, reasoning, tool_calls (name + raw arguments),
        finish_reason, usage, ttft, total, the parsed deltas (streaming) and the HTTP status.
        abort_after: close the connection after this many streamed deltas (client disconnect)."""
        body = {"model": self.model, "messages": messages, "stream": stream}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if sampling:
            body.update(sampling)
        body.update(extra)
        return self._chat_body(body, on_delta=on_delta, abort_after=abort_after)

    def _chat_body(self, body, on_delta=None, abort_after=None):
        c = self._conn()
        t0 = time.monotonic()
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None,
               "ttft": None, "status": r.status, "deltas": []}
        if r.status != 200:
            out["error"] = r.read()[:2000].decode("utf-8", "replace")
            out["total"] = time.monotonic() - t0
            return out
        if not body.get("stream"):
            d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            out["content"] = msg.get("content") or ""
            out["reasoning"] = msg.get("reasoning_content") or msg.get("thinking") or ""
            out["reasoning_field"] = ("reasoning_content" if "reasoning_content" in msg
                                      else "thinking" if "thinking" in msg else None)
            out["tool_calls"] = [{"id": tc.get("id"), "name": tc["function"]["name"],
                                  "arguments": tc["function"].get("arguments") or ""}
                                 for tc in msg.get("tool_calls") or []]
            out["finish_reason"] = d["choices"][0].get("finish_reason")
            out["usage"] = d.get("usage")
            out["raw"] = d
            out["total"] = time.monotonic() - t0
            return out
        calls = {}
        n = 0
        try:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    out["done_marker"] = True
                    break
                ev = json.loads(data)
                if ev.get("error"):
                    out["error"] = json.dumps(ev["error"])[:2000]
                    break
                if ev.get("usage"):
                    out["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    out["deltas"].append(d)
                    if on_delta:
                        on_delta(d)
                    if out["ttft"] is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                        out["ttft"] = time.monotonic() - t0
                    out["content"] += d.get("content") or ""
                    out["reasoning"] += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = slot["id"] or tc.get("id")
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
                n += 1
                if abort_after is not None and n >= abort_after:
                    out["aborted"] = True
                    c.close()
                    break
        except (http.client.IncompleteRead, ConnectionError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
        out["tool_calls"] = [calls[k] for k in sorted(calls)]
        out["total"] = time.monotonic() - t0
        return out

    # ------------------------------------------------------------------ responses API

    def responses(self, input_, *, stream=False, reasoning=None, sampling=None, **extra):
        body = {"model": self.model, "input": input_, "stream": stream}
        if reasoning is not None:
            body["reasoning"] = reasoning
        if sampling:
            body.update({k: v for k, v in sampling.items() if k in ("temperature", "top_p", "seed")})
            for k in ("top_k", "min_p", "repeat_penalty"):
                if k in sampling:
                    body[k] = sampling[k]
        body.update(extra)
        c = self._conn()
        t0 = time.monotonic()
        c.request("POST", "/v1/responses", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        out = {"status": r.status, "events": [], "text": "", "reasoning": "", "function_calls": []}
        if r.status != 200:
            out["error"] = r.read()[:2000].decode("utf-8", "replace")
            return out
        if not stream:
            d = json.loads(r.read())
            out["raw"] = d
            for item in d.get("output") or []:
                if item.get("type") == "message":
                    for part in item.get("content") or []:
                        if part.get("type") in ("output_text", "text"):
                            out["text"] += part.get("text") or ""
                elif item.get("type") == "reasoning":
                    for part in (item.get("content") or []) + (item.get("summary") or []):
                        out["reasoning"] += part.get("text") or ""
                elif item.get("type") == "function_call":
                    out["function_calls"].append({"name": item.get("name"), "arguments": item.get("arguments"),
                                                  "call_id": item.get("call_id")})
            out["total"] = time.monotonic() - t0
            return out
        event = None
        for line in r:
            line = line.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                kind = ev.get("type") or event
                out["events"].append(kind)
                if kind == "response.output_text.delta":
                    out["text"] += ev.get("delta") or ""
                elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                    out["reasoning"] += ev.get("delta") or ""
                elif kind == "response.output_item.done" and (ev.get("item") or {}).get("type") == "function_call":
                    item = ev["item"]
                    out["function_calls"].append({"name": item.get("name"), "arguments": item.get("arguments"),
                                                  "call_id": item.get("call_id")})
                elif kind == "response.completed":
                    out["raw"] = ev.get("response")
        out["total"] = time.monotonic() - t0
        return out

    # ------------------------------------------------------------------ TensorSharp Web UI

    def webchat(self, messages, *, think=False, max_tokens=256, session=None, new_chat=False, plain=True, **extra):
        """POST /api/chat (TensorSharp's Web UI SSE stream). Returns thinking, text, the done frame."""
        payload = {"messages": messages, "maxTokens": max_tokens, "think": think, "newChat": new_chat}
        if session:
            payload["sessionId"] = session
        if plain:
            payload.update({"skills": [], "skills_discovery": False, "multi_agent": False})
        payload.update(extra)
        c = self._conn()
        c.request("POST", "/api/chat", json.dumps(payload), {"Content-Type": "application/json"})
        r = c.getresponse()
        out = {"status": r.status, "thinking": "", "text": "", "done": None, "frames": 0}
        if r.status != 200:
            out["error"] = r.read()[:2000].decode("utf-8", "replace")
            return out
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            out["frames"] += 1
            data = json.loads(line[6:])
            if data.get("error"):
                out["error"] = data["error"]
                break
            out["thinking"] += data.get("thinking") or ""
            out["text"] += data.get("token") or ""
            if data.get("replace") is not None:
                out["text"] = data["replace"]
            if data.get("done"):
                out["done"] = data
        return out

    def new_session(self):
        status, data = self.post("/api/sessions", {}, timeout=60)
        return json.loads(data or b"{}").get("sessionId")


# ---------------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    """Wilson score interval for k successes of n (95% by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b (A right, B wrong) and c (A wrong, B right) discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_summary(a, b):
    """a, b: lists of booleans over the same items. Accuracy, Wilson CIs, discordant counts, McNemar p."""
    n = len(a)
    ka, kb = sum(a), sum(b)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {"n": n, "a": ka, "b": kb, "a_acc": ka / n if n else None, "b_acc": kb / n if n else None,
            "a_ci": wilson(ka, n), "b_ci": wilson(kb, n), "only_a": only_a, "only_b": only_b,
            "mcnemar_p": mcnemar_exact(only_a, only_b)}


GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}
CARD = {"temperature": 0.1, "top_k": 50, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.1}
