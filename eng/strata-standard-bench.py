#!/usr/bin/env python3
"""Replay Strata's frozen benchmark bundle (bench/prompts/standard-v1 in github.com/Niko1221/Strata) against any
OpenAI-compatible server, so TensorSharp and Strata are measured on Strata's own workload as well as on
eng/qwen38-engine-bench.py's.

The bundle is read from a Strata checkout (it is not copied here) and checked against its MANIFEST.sha256. Every
request is sent the way Strata's runner (strata_cli/bench_standard.py, protocol strata-qwen38-thinking-v1-draft.1)
sends it: streamed, thinking on (chat_template_kwargs enable_thinking / preserve_thinking, reasoning_effort xhigh),
temperature 1.0, top_p 0.95, top_k 20, the request's seed and output cap, its tools. Sanity answers are scored with
the same rules (strict JSON with equal types, the exact text, or exactly one tool call with equal arguments).

Timings are the client's, for every engine alike (an engine's own `timings` are recorded when it sends them):
  ttft_s      dispatch -> first nonempty delta (reasoning, answer or tool call)
  prompt_tps  prompt tokens / ttft_s
  decode_tps  (completion tokens - 1) / (last delta - first delta)
Concurrency rounds send their four requests at once and report total completion tokens / round wall time.

  python3 eng/strata-standard-bench.py --bundle ~/work/Strata-latest/bench/prompts/standard-v1 \\
      --url http://127.0.0.1:5088 --label ts-mtp --out artifacts/strata-standard/ts-mtp.jsonl
  python3 eng/strata-standard-bench.py --summarize artifacts/strata-standard/*.jsonl
"""
import argparse
import hashlib
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

# strata_cli/bench_standard.py: SAMPLING, TEMPLATE_KWARGS and payload_of.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0, "reasoning_effort": "xhigh"}
TEMPLATE_KWARGS = {"enable_thinking": True, "preserve_thinking": True}


def load_bundle(directory):
    directory = Path(directory).expanduser()
    build = json.loads((directory / "build.json").read_text())
    manifest = {}
    for line in (directory / "MANIFEST.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            manifest[name.strip().lstrip("*")] = digest
    specs = []
    for entry in build["requests"]:
        rel = f"requests/{entry['id']}.json"
        data = (directory / rel).read_bytes()
        expected = manifest.get(rel) or manifest.get("./" + rel)
        if expected is None or hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"{rel}: not in MANIFEST.sha256 or its hash differs - not the frozen bundle")
        specs.append(json.loads(data))
    return build, specs


def payload_of(spec, model):
    body = {"model": model, "messages": spec["messages"], "stream": True,
            "stream_options": {"include_usage": True}, "max_tokens": spec["max_output_tokens"], "seed": spec["seed"],
            **SAMPLING, "reasoning_budget_tokens": 0, "chat_template_kwargs": dict(TEMPLATE_KWARGS)}
    if spec.get("tools"):
        body["tools"] = spec["tools"]
        if spec.get("tool_choice") is not None:
            body["tool_choice"] = spec["tool_choice"]
    return body


def model_id(url):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read())
    return data["data"][0]["id"]


def stream_chat(url, spec, model, timeout_s, gate=None):
    u = urllib.parse.urlparse(url)
    raw = {"reasoning": [], "content": [], "tool_calls": {}, "usage": None, "timings": None, "finish_reason": None,
           "error": None, "marks": []}
    if gate is not None:
        gate.wait()
    t0 = time.monotonic()
    raw["t0"] = t0
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout_s)
        conn.request("POST", "/v1/chat/completions", json.dumps(payload_of(spec, model)),
                     {"Content-Type": "application/json", "Accept": "text/event-stream"})
        resp = conn.getresponse()
        if resp.status != 200:
            raw["error"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return raw
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                raw["usage"] = event["usage"]
            if isinstance(event.get("timings"), dict):
                raw["timings"] = event["timings"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                now = time.monotonic() - t0
                if delta.get("reasoning_content"):
                    raw["reasoning"].append(delta["reasoning_content"]); raw["marks"].append(now)
                if delta.get("content"):
                    raw["content"].append(delta["content"]); raw["marks"].append(now)
                for call in delta.get("tool_calls") or []:
                    slot = raw["tool_calls"].setdefault(call.get("index", len(raw["tool_calls"])),
                                                        {"name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                    raw["marks"].append(now)
                if choice.get("finish_reason"):
                    raw["finish_reason"] = choice["finish_reason"]
        conn.close()
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raw["error"] = f"{type(exc).__name__}: {exc}"[:300]
    raw["wall_s"] = time.monotonic() - t0
    return raw


# strata_cli/bench_standard.py: strict_json, same_json, evaluate.
def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate key {key!r}")
            out[key] = value
        return out

    def constant(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(text.strip(" \t\r\n\x0b\x0c"), object_pairs_hook=pairs, parse_constant=constant)


def same_json(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def evaluate(spec, status, answer, calls):
    expected = spec["expected"]
    if status != "completed":
        return False, {"truncated": "the output cap was reached before the answer was complete",
                       "failed": "the request failed"}[status]
    kind = expected["kind"]
    if kind == "tool_call":
        if len(calls) != 1:
            return False, f"expected one structured tool call, got {len(calls)}"
        if calls[0]["name"] != expected["name"]:
            return False, f"wrong function {calls[0]['name']!r}"
        try:
            arguments = strict_json(calls[0]["arguments"] or "{}")
        except ValueError as exc:
            return False, f"the tool arguments are not strict JSON ({exc})"
        ok = isinstance(arguments, dict) and same_json(arguments, expected["arguments"])
        return ok, "the expected tool call" if ok else "the tool arguments differ"
    if kind == "text":
        ok = answer.strip(" \t\r\n\x0b\x0c") == expected["value"]
        return ok, "the exact expected text" if ok else "the answer text differs"
    try:
        value = strict_json(answer)
    except ValueError as exc:
        return False, f"the answer is not one strict JSON value ({exc})"
    ok = same_json(value, expected["value"])
    return ok, "the expected JSON value" if ok else "the JSON value differs"


def record_of(spec, raw, label):
    usage = raw.get("usage") or {}
    prompt_tokens, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    marks = raw["marks"]
    ttft = marks[0] if marks else None
    decode_tps = None
    if completion and completion > 1 and len(marks) > 1 and marks[-1] > marks[0]:
        decode_tps = (completion - 1) / (marks[-1] - marks[0])
    status = "failed" if raw["error"] else ("truncated" if raw["finish_reason"] == "length" else "completed")
    rec = {"label": label, "id": spec["id"], "phase": spec["phase"], "case": spec["case_id"],
           "repetition": spec["repetition"], "status": status, "finish_reason": raw["finish_reason"],
           "error": raw["error"], "prompt_tokens": prompt_tokens, "frozen_prompt_tokens": spec["input_tokens_frozen"],
           "cached_tokens": cached, "completion_tokens": completion, "ttft_s": ttft,
           "prompt_tps": (prompt_tokens - (cached or 0)) / ttft if prompt_tokens and ttft else None,
           "decode_tps": decode_tps, "wall_s": raw.get("wall_s"), "engine_timings": raw.get("timings"),
           "reasoning_chars": sum(map(len, raw["reasoning"])), "answer": "".join(raw["content"])[:400]}
    if spec.get("expected"):
        calls = [raw["tool_calls"][k] for k in sorted(raw["tool_calls"])]
        rec["passed"], rec["reason"] = evaluate(spec, status, "".join(raw["content"]), calls)
    return rec


def run(args):
    build, specs = load_bundle(args.bundle)
    model = args.model or model_id(args.url)
    phases = set(args.phases.split(","))
    print(f"protocol {build.get('protocol_id')} bundle {build.get('bundle')}: {len(specs)} requests; "
          f"{args.url} model={model} label={args.label}", flush=True)
    out = open(args.out, "a")

    def emit(rec):
        out.write(json.dumps(rec) + "\n"); out.flush()
        speed = (f"ttft {rec['ttft_s']:.2f}s " if rec["ttft_s"] is not None else "") + \
                (f"decode {rec['decode_tps']:.1f} tok/s " if rec["decode_tps"] else "")
        verdict = "" if "passed" not in rec else (" PASS" if rec["passed"] else f" FAIL ({rec['reason']})")
        print(f"  {rec['id']}: {rec['status']} prompt {rec['prompt_tokens']} out {rec['completion_tokens']} {speed}{verdict}",
              flush=True)

    i = 0
    while i < len(specs):
        spec = specs[i]
        if spec["phase"] not in phases and spec["phase"] != "warmup":
            i += 1
            continue
        if spec["phase"] == "concurrency":
            group = [s for s in specs[i:i + spec["concurrency"]] if s["round"] == spec["round"]]
            gate, raws, threads = threading.Event(), {}, []
            for s in group:
                t = threading.Thread(target=lambda s=s: raws.__setitem__(s["id"], stream_chat(args.url, s, model, args.timeout, gate)))
                t.start(); threads.append(t)
            t0 = time.monotonic(); gate.set()
            for t in threads:
                t.join()
            wall = time.monotonic() - t0
            recs = [record_of(s, raws[s["id"]], args.label) for s in group]
            for rec in recs:
                emit(rec)
            total = sum(r["completion_tokens"] or 0 for r in recs)
            summary = {"label": args.label, "id": f"concurrency-round-{spec['round']}", "phase": "concurrency-round",
                       "requests": len(group), "completion_tokens": total, "wall_s": wall, "total_tps": total / wall}
            out.write(json.dumps(summary) + "\n"); out.flush()
            print(f"  concurrency round {spec['round']}: {len(group)} at once, {total / wall:.1f} tok/s total", flush=True)
            i += len(group)
            continue
        emit(record_of(spec, stream_chat(args.url, spec, model, args.timeout), args.label))
        i += 1


def summarize(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    for label in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == label]
        print(f"== {label}")
        for case in ("short", "prompt-4k", "prompt-16k"):
            cs = [r for r in rs if r.get("phase") == "performance" and r.get("case") == case and r["status"] != "failed"]
            if not cs:
                continue
            med = lambda k: statistics.median([r[k] for r in cs if r.get(k) is not None]) if any(r.get(k) for r in cs) else None
            fmt = lambda v, f: f.format(v) if v is not None else "-"
            print(f"  {case:11s} n={len(cs)} ttft {fmt(med('ttft_s'), '{:.2f}')} s  prompt {fmt(med('prompt_tps'), '{:.0f}')} tok/s  "
                  f"decode {fmt(med('decode_tps'), '{:.1f}')} tok/s  capped {sum(r['status'] == 'truncated' for r in cs)}/{len(cs)}")
        sanity = [r for r in rs if r.get("phase") == "sanity"]
        if sanity:
            print(f"  sanity {sum(bool(r.get('passed')) for r in sanity)} of {len(sanity)}"
                  + "".join(f"\n    {r['id']}: {r['reason']}" for r in sanity if not r.get("passed")))
        rounds = [r for r in rs if r.get("phase") == "concurrency-round"]
        if rounds:
            print("  concurrency total tok/s: " + ", ".join(f"{r['total_tps']:.1f}" for r in rounds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="Strata's bench/prompts/standard-v1 directory")
    ap.add_argument("--url", help="server base URL, e.g. http://127.0.0.1:5088")
    ap.add_argument("--model", help="model id (default: the first of /v1/models)")
    ap.add_argument("--label", default="engine")
    ap.add_argument("--out", help="JSONL to append to")
    ap.add_argument("--phases", default="performance,sanity", help="comma list of performance, sanity, concurrency")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--summarize", nargs="+", metavar="JSONL", help="print medians, sanity and concurrency per label")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.bundle and args.url and args.out:
        run(args)
    else:
        ap.error("--bundle, --url and --out (or --summarize)")


if __name__ == "__main__":
    main()
