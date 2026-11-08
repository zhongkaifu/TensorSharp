#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""
End-to-end serving scenarios against any OpenAI-compatible chat endpoint (standard library only).

Each scenario checks what a client would see (answers, tool calls, JSON against its schema, a memory across turns,
a needle in a long prompt, stop and length handling) and records its timings (time to first token, total, decode
rate) and the server's reused prompt tokens (usage.prompt_tokens_details.cached_tokens). Run it against two builds
of the same model and compare with --compare to find quality or performance regressions.

  python3 eng/serving-e2e.py --url http://127.0.0.1:5088 --label after --out after.jsonl
  python3 eng/serving-e2e.py --compare before.jsonl after.jsonl

Scenarios: chat_exact, stream_identity, stop_sequence, max_tokens, thinking, json_object, json_schema,
tool_call, tool_call_parallel, agent_loop, multi_turn_reuse, needle_4k, needle_16k, concurrency_mixed.
Greedy (temperature 0) and thinking off unless a scenario says otherwise.
"""
import argparse
import http.client
import json
import random
import re
import sys
import threading
import time
import urllib.parse

MODEL = None


def _conn(url, timeout):
    u = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)


def model_id(url):
    c = _conn(url, 30)
    c.request("GET", "/v1/models")
    return json.loads(c.getresponse().read())["data"][0]["id"]


def chat(url, messages, *, stream=True, timeout=1800, thinking=False, **extra):
    """One chat completion. Returns content, reasoning, tool_calls, finish_reason, usage, ttft and total seconds."""
    body = {"model": MODEL, "messages": messages, "stream": stream, "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(extra)
    c = _conn(url, timeout)
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    out = {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": None, "ttft": None,
           "status": r.status}
    if r.status != 200:
        out["error"] = r.read()[:500].decode("utf-8", "replace")
        out["total"] = time.monotonic() - t0
        return out
    if not stream:
        d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        out["content"] = msg.get("content") or ""
        out["reasoning"] = msg.get("reasoning_content") or ""
        out["tool_calls"] = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or ""}
                             for tc in msg.get("tool_calls") or []]
        out["finish_reason"] = d["choices"][0].get("finish_reason")
        out["usage"] = d.get("usage")
        out["total"] = time.monotonic() - t0
        return out
    calls = {}
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            out["usage"] = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if out["ttft"] is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                out["ttft"] = time.monotonic() - t0
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                out["finish_reason"] = ch["finish_reason"]
    out["tool_calls"] = [calls[k] for k in sorted(calls)]
    out["total"] = time.monotonic() - t0
    return out


def cached(res):
    u = res.get("usage") or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens")


def decode_rate(res):
    u = res.get("usage") or {}
    n = u.get("completion_tokens")
    if not n or res.get("ttft") is None or res["total"] <= res["ttft"]:
        return None
    return (n - 1) / (res["total"] - res["ttft"]) if n > 1 else None


class Result:
    def __init__(self, name):
        self.name, self.checks, self.metrics, self.notes = name, [], {}, []

    def check(self, label, ok, detail=None):
        self.checks.append({"check": label, "ok": bool(ok), "detail": detail})
        return ok

    @property
    def ok(self):
        return all(c["ok"] for c in self.checks) and bool(self.checks)


def _json(text):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)


# ---------------------------------------------------------------- scenarios

def s_chat_exact(url, R):
    res = chat(url, [{"role": "user", "content": "What is 7 multiplied by 8? Reply with only the integer."}], max_tokens=16)
    R.check("answer is 56", re.fullmatch(r"\s*56[.!]?\s*", res["content"]), res["content"])
    R.metrics.update(ttft=res["ttft"], total=res["total"])


def s_stream_identity(url, R):
    msgs = [{"role": "user", "content": "Write exactly two sentences about the ocean."}]
    a = chat(url, msgs, stream=True, max_tokens=64)
    b = chat(url, msgs, stream=False, max_tokens=64)
    R.check("streamed and non-streamed greedy text identical", a["content"] == b["content"],
            {"stream": a["content"][:120], "plain": b["content"][:120]})
    R.metrics.update(ttft=a["ttft"], decode_tps=decode_rate(a))


def s_stop_sequence(url, R):
    res = chat(url, [{"role": "user", "content": "Count from 1 to 10, separated by commas, nothing else."}],
               max_tokens=48, stop=["6"])
    R.check("finish_reason stop", res["finish_reason"] == "stop", res["finish_reason"])
    R.check("stop text not emitted", "6" not in res["content"], res["content"])
    R.check("text before the stop kept", "5" in res["content"], res["content"])


def s_max_tokens(url, R):
    res = chat(url, [{"role": "user", "content": "Write a long story about a lighthouse keeper."}], max_tokens=8)
    u = res.get("usage") or {}
    R.check("finish_reason length", res["finish_reason"] == "length", res["finish_reason"])
    R.check("8 completion tokens", u.get("completion_tokens") == 8, u.get("completion_tokens"))


def s_thinking(url, R):
    res = chat(url, [{"role": "user", "content": "What is 17 multiplied by 23? Give the number."}], thinking=True,
               max_tokens=1024)
    R.check("reasoning returned", len(res["reasoning"]) > 0, len(res["reasoning"]))
    R.check("answer 391 in content", "391" in res["content"], res["content"][-200:])
    R.metrics.update(ttft=res["ttft"], total=res["total"], decode_tps=decode_rate(res))


def s_json_object(url, R):
    res = chat(url, [{"role": "system", "content": "You answer in JSON."},
                     {"role": "user", "content": "Give a JSON object with keys \"city\" (the capital of Japan) and "
                                                 "\"population_millions\" (a number)."}],
               response_format={"type": "json_object"}, max_tokens=128)
    try:
        obj = _json(res["content"])
        R.check("parses as a JSON object", isinstance(obj, dict), res["content"])
        R.check("city is Tokyo", str(obj.get("city", "")).lower().startswith("tokyo"), obj)
        R.check("population is a number", isinstance(obj.get("population_millions"), (int, float)), obj)
    except Exception as e:
        R.check("parses as a JSON object", False, f"{e}: {res['content'][:200]}")
    R.metrics.update(total=res["total"])


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "age", "skills", "address"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                         "skills": {"type": "array", "items": {"type": "string", "enum": ["python", "c#", "sql", "go"]}},
                         "address": {"type": "object", "additionalProperties": False, "required": ["city", "zip"],
                                     "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}}}}


def _validate(value, schema, path="$"):
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return f"{path}: not an object"
        for k in schema.get("required", []):
            if k not in value:
                return f"{path}: missing {k}"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return f"{path}: extra keys {sorted(extra)}"
        for k, v in value.items():
            if k in props:
                err = _validate(v, props[k], f"{path}.{k}")
                if err:
                    return err
    elif t == "array":
        if not isinstance(value, list):
            return f"{path}: not an array"
        for i, v in enumerate(value):
            err = _validate(v, schema.get("items", {}), f"{path}[{i}]")
            if err:
                return err
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path}: not an integer"
    elif t == "string":
        if not isinstance(value, str):
            return f"{path}: not a string"
        if "enum" in schema and value not in schema["enum"]:
            return f"{path}: {value!r} not in enum"
    return None


def s_json_schema(url, R):
    res = chat(url, [{"role": "user", "content": "Create a profile for a fictional software engineer named Mia, age 31, "
                                                 "who knows python and sql and lives in Berlin (zip 10115)."}],
               response_format={"type": "json_schema", "json_schema": {"name": "profile", "strict": True, "schema": SCHEMA}},
               max_tokens=200)
    try:
        obj = _json(res["content"])
        err = _validate(obj, SCHEMA)
        R.check("valid against the schema", err is None, err or obj)
        R.check("facts carried over", obj.get("name") == "Mia" and obj.get("age") == 31
                and obj.get("address", {}).get("zip") == "10115", obj)
    except Exception as e:
        R.check("valid against the schema", False, f"{e}: {res['content'][:200]}")
    R.metrics.update(total=res["total"])


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   "required": ["city"]}}}


def _args(tc):
    try:
        return json.loads(tc["arguments"] or "{}")
    except Exception:
        return None


def s_tool_call(url, R):
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    res = chat(url, msgs, tools=[WEATHER], max_tokens=200)
    tcs = res["tool_calls"]
    R.check("one get_weather call", len(tcs) >= 1 and tcs[0]["name"] == "get_weather", tcs)
    a = _args(tcs[0]) if tcs else None
    R.check("arguments parse, city Paris", isinstance(a, dict) and "paris" in str(a.get("city", "")).lower(), tcs)
    R.check("finish_reason tool_calls", res["finish_reason"] == "tool_calls", res["finish_reason"])
    R.metrics.update(ttft=res["ttft"], call_total=res["total"])
    if not tcs:
        return
    msgs += [{"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": tcs[0]["arguments"]}}]},
             {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "light rain"})}]
    fin = chat(url, msgs, tools=[WEATHER], max_tokens=120)
    R.check("final answer uses the tool result", "17" in fin["content"], fin["content"][:200])
    R.metrics.update(answer_ttft=fin["ttft"], answer_cached=cached(fin))


def s_tool_call_parallel(url, R):
    msgs = [{"role": "user", "content": "Get the current weather for both Tokyo and Madrid with the tool."}]
    seen, rounds = set(), 0
    while rounds < 3:
        rounds += 1
        res = chat(url, msgs, tools=[WEATHER], max_tokens=200)
        if not res["tool_calls"]:
            break
        calls = []
        for i, tc in enumerate(res["tool_calls"]):
            a = _args(tc) or {}
            city = str(a.get("city", "")).lower()
            seen.add("tokyo" if "tokyo" in city else "madrid" if "madrid" in city else city)
            calls.append({"id": f"c{rounds}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
        msgs.append({"role": "assistant", "content": None, "tool_calls": calls})
        for c in calls:
            a = _args({"arguments": c["function"]["arguments"]}) or {}
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": json.dumps({"city": a.get("city"), "temp_c": 21 if "tok" in str(a.get("city", "")).lower() else 25})})
    R.check("both cities requested", {"tokyo", "madrid"} <= seen, sorted(seen))
    R.metrics.update(rounds=rounds)


FS_TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List the files of the project.",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a file's whole content.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
]


def s_agent_loop(url, R):
    fs = {"README.md": "A tiny calculator.\n", "src/main.py": "from math_utils import add\nprint(add(2, 3))\n",
          "src/math_utils.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"}
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools to inspect and edit files. Be concise."},
            {"role": "user", "content": "src/main.py should print 5 but prints -1. Find the bug, fix it with the tools, "
                                        "then reply with one sentence saying what you changed."}]
    turns, ttfts, reuse, bad = 0, [], [], 0
    final = ""
    while turns < 10:
        turns += 1
        res = chat(url, msgs, tools=FS_TOOLS, max_tokens=400)
        ttfts.append(res["ttft"])
        reuse.append(cached(res))
        if not res["tool_calls"]:
            final = res["content"]
            break
        calls = []
        for i, tc in enumerate(res["tool_calls"]):
            calls.append({"id": f"t{turns}_{i}", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
        msgs.append({"role": "assistant", "content": res["content"] or None, "tool_calls": calls})
        for c in calls:
            a = _args({"arguments": c["function"]["arguments"]})
            name = c["function"]["name"]
            if a is None:
                bad += 1
                out = "error: arguments are not valid JSON"
            elif name == "list_files":
                out = "\n".join(sorted(fs))
            elif name == "read_file":
                out = fs.get(a.get("path", ""), f"error: no such file {a.get('path')}")
            elif name == "write_file":
                fs[a.get("path", "")] = a.get("content", "")
                out = "ok"
            else:
                bad += 1
                out = f"error: no tool {name}"
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    fixed = fs.get("src/math_utils.py", "")
    R.check("bug fixed (return a + b)", re.search(r"return\s+a\s*\+\s*b", fixed) is not None, fixed)
    R.check("mul kept", "return a * b" in fixed, fixed)
    R.check("finished with a summary", bool(final.strip()), final[:200])
    R.check("every tool call parsed", bad == 0, bad)
    R.metrics.update(turns=turns, ttft_first=ttfts[0], ttft_later=[round(t, 3) for t in ttfts[1:] if t is not None],
                     cached=reuse)


def s_multi_turn_reuse(url, R):
    msgs = [{"role": "system", "content": "You are a helpful assistant. Keep answers short."},
            {"role": "user", "content": "My name is Ada and my favourite number is 42. Please remember both."}]
    asks = ["What is the capital of France?", "What is 3 plus 4?", "Name a primary colour.",
            "What are my name and my favourite number?"]
    ttfts, reuse, prompt = [], [], []
    res = None
    for i in range(len(asks) + 1):
        res = chat(url, msgs, max_tokens=60)
        ttfts.append(res["ttft"])
        reuse.append(cached(res))
        prompt.append((res.get("usage") or {}).get("prompt_tokens"))
        msgs.append({"role": "assistant", "content": res["content"]})
        if i < len(asks):
            msgs.append({"role": "user", "content": asks[i]})
    R.check("remembers the name", "ada" in res["content"].lower(), res["content"])
    R.check("remembers the number", "42" in res["content"], res["content"])
    later = [r for r in reuse[1:] if r is not None]
    R.check("later turns reuse the cached prefix", len(later) == len(reuse) - 1 and all(r > 0 for r in later), reuse)
    R.metrics.update(ttft=ttfts, cached=reuse, prompt_tokens=prompt)


def _needle(url, R, target_tokens):
    rnd = random.Random(target_tokens)
    code = f"KX-{rnd.randint(1000, 9999)}-QZ"
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    lines = [f"Entry {i:05d}: " + " ".join(rnd.choice(words) for _ in range(8)) + "." for i in range(target_tokens // 16)]
    lines.insert(len(lines) * 2 // 3, f"Important: the secret access code is {code}.")
    prompt = "\n".join(lines) + "\n\nWhat is the secret access code? Reply with the code only."
    res = chat(url, [{"role": "user", "content": prompt}], max_tokens=24)
    R.check("needle found", code in res["content"], res["content"])
    R.metrics.update(ttft=res["ttft"], prompt_tokens=(res.get("usage") or {}).get("prompt_tokens"))


def s_needle_4k(url, R):
    _needle(url, R, 4096)


def s_needle_16k(url, R):
    _needle(url, R, 16384)


def s_concurrency_mixed(url, R):
    jobs = [("tool_call", s_tool_call), ("json_schema", s_json_schema), ("needle_4k", s_needle_4k),
            ("chat_exact", s_chat_exact)]
    subs = {n: Result(n) for n, _ in jobs}
    errs = {}

    def run(n, f):
        try:
            f(url, subs[n])
        except Exception as e:
            errs[n] = repr(e)

    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(n, f)) for n, f in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    for n, _ in jobs:
        R.check(f"{n} passes when run beside the others", n not in errs and subs[n].ok,
                errs.get(n) or [c for c in subs[n].checks if not c["ok"]])
    R.metrics.update(wall=wall)


SCENARIOS = {"chat_exact": s_chat_exact, "stream_identity": s_stream_identity, "stop_sequence": s_stop_sequence,
             "max_tokens": s_max_tokens, "thinking": s_thinking, "json_object": s_json_object,
             "json_schema": s_json_schema, "tool_call": s_tool_call, "tool_call_parallel": s_tool_call_parallel,
             "agent_loop": s_agent_loop, "multi_turn_reuse": s_multi_turn_reuse, "needle_4k": s_needle_4k,
             "needle_16k": s_needle_16k, "concurrency_mixed": s_concurrency_mixed}


def compare(a_path, b_path):
    def load(p):
        return {r["scenario"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
    a, b = load(a_path), load(b_path)
    print(f"{'scenario':20s} {'A':>5s} {'B':>5s}  timings (A -> B)")
    regress = 0
    for name in SCENARIOS:
        if name not in a and name not in b:
            continue
        ra, rb = a.get(name, {}), b.get(name, {})
        oa, ob = ra.get("ok"), rb.get("ok")
        if oa and ob is False:
            regress += 1
        keys = sorted(set(ra.get("metrics", {})) | set(rb.get("metrics", {})))
        timing = []
        for k in keys:
            va, vb = ra.get("metrics", {}).get(k), rb.get("metrics", {}).get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                timing.append(f"{k} {va:.3g}->{vb:.3g}")
        flag = "  <-- REGRESSED" if oa and ob is False else ""
        print(f"{name:20s} {str(oa):>5s} {str(ob):>5s}  {'; '.join(timing)}{flag}")
    return 1 if regress else 0


def main():
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    MODEL = model_id(args.url)
    failed = 0
    out = open(args.out, "a") if args.out else None
    for name in args.scenarios.split(","):
        R = Result(name)
        t0 = time.monotonic()
        try:
            SCENARIOS[name](args.url, R)
        except Exception as e:
            R.check("ran without an exception", False, repr(e))
        row = {"label": args.label, "scenario": name, "ok": R.ok, "seconds": round(time.monotonic() - t0, 2),
               "checks": R.checks, "metrics": R.metrics}
        failed += 0 if R.ok else 1
        bad = [c for c in R.checks if not c["ok"]]
        print(f"{name:20s} {'PASS' if R.ok else 'FAIL'} {row['seconds']:7.2f}s  "
              + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in R.metrics.items()})
              + (f"  failed: {json.dumps(bad)[:400]}" if bad else ""), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(f"{len(args.scenarios.split(',')) - failed}/{len(args.scenarios.split(','))} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
