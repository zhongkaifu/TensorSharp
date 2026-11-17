#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Function-calling accuracy of one OpenAI-compatible server on toolcall_cases.CASES.

    toolcall_eval.py --url http://127.0.0.1:5731 --out ts-tools.jsonl [--think] [--stream]
    compare.py ts-tools.jsonl llama-tools.jsonl --show 10

Greedy (temperature 0, top_k 1, no repetition penalty). A case is correct when the parsed calls match
the expected ones (names, then each expected argument by its matcher; parallel calls in any order),
every call is schema-valid (declared parameters only, required ones present, JSON types right) and,
for a no-call case, there is no call (plus answer_contains, when given). --stream assembles the calls
from streamed tool-call deltas instead (the two runs should agree).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY, wilson  # noqa: E402
from toolcall_cases import CASES, TOOLS  # noqa: E402

JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def match_value(matcher, value):
    if isinstance(matcher, dict):
        if "contains" in matcher:
            return isinstance(value, str) and matcher["contains"].lower() in value.lower()
        if "any_of" in matcher:
            return any(match_value(m, value) for m in matcher["any_of"])
        if matcher.get("present"):
            return value is not None
    if isinstance(matcher, bool) or isinstance(value, bool):
        return matcher is value or matcher == value and type(matcher) is type(value)
    if isinstance(matcher, (int, float)):
        try:
            return abs(float(value) - matcher) < 1e-9
        except (TypeError, ValueError):
            return False
    return isinstance(value, str) and value.strip().lower() == str(matcher).strip().lower()


def schema_problems(call, tools):
    t = tools.get(call["name"])
    if t is None:
        return [f"undeclared function {call['name']}"]
    params = t["function"]["parameters"]
    props, required = params.get("properties", {}), params.get("required", [])
    args = call.get("parsed")
    if not isinstance(args, dict):
        return ["arguments are not a JSON object"]
    out = [f"undeclared parameter {k}" for k in args if k not in props]
    out += [f"missing required {k}" for k in required if k not in args]
    for k, v in args.items():
        if k in props:
            ty = props[k].get("type")
            want = JSON_TYPES.get(ty)
            if want and not (isinstance(v, want) and not (ty in ("integer", "number") and isinstance(v, bool))):
                out.append(f"{k}: {type(v).__name__} is not {ty}")
            if "enum" in props[k] and v not in props[k]["enum"]:
                out.append(f"{k}: {v!r} not in enum")
    return out


def calls_match(expected, got):
    if len(expected) != len(got):
        return False
    remaining = list(got)
    for e in expected:
        hit = next((g for g in remaining if g["name"] == e["name"] and isinstance(g.get("parsed"), dict)
                    and all(k in g["parsed"] and match_value(m, g["parsed"][k]) for k, m in e["args"].items())), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--only", help="comma-separated case ids")
    args = ap.parse_args()
    cl = Client(args.url)
    only = set(args.only.split(",")) if args.only else None
    n = k = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for case in CASES:
            if only and case["id"] not in only:
                continue
            tools = {name: TOOLS[name] for name in case["tools"]}
            res = cl.chat(case["messages"], stream=args.stream, thinking=args.think, sampling=GREEDY,
                          max_tokens=args.max_tokens if not args.think else max(args.max_tokens, 4096),
                          tools=list(tools.values()), **case.get("extra", {}))
            calls = []
            for tc in res["tool_calls"]:
                try:
                    parsed = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    parsed = None
                calls.append({"name": tc["name"], "arguments": tc["arguments"], "parsed": parsed})
            problems = [p for c in calls for p in schema_problems(c, tools)]
            # Call markup left in the answer is a call the client never received.
            if any(tag in res["content"] for tag in ("<tool_call>", "<function=", "</think>")):
                problems.append("tool-call or reasoning markup in the content")
            alternatives = [case["expected"]] + case.get("alt_expected", [])
            ok = res["status"] == 200 and not problems and any(calls_match(e, calls) for e in alternatives)
            if ok and not case["expected"]:
                ok = all(s.lower() in res["content"].lower() for s in case.get("answer_contains", []))
                ok = ok and bool(res["content"].strip())
            if ok and case["expected"] and res["finish_reason"] != "tool_calls":
                problems.append(f"finish_reason {res['finish_reason']}")
            n += 1
            k += ok
            rec = {"id": case["id"], "correct": ok, "calls": [{k2: c[k2] for k2 in ("name", "arguments")} for c in calls],
                   "problems": problems, "content": res["content"], "reasoning": res["reasoning"],
                   "finish": res["finish_reason"], "status": res["status"], "error": res.get("error"),
                   "reply": json.dumps([{k2: c[k2] for k2 in ("name", "parsed")} for c in calls], sort_keys=True)
                   + "|" + res["content"].strip()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{case['id']:22s} {'OK ' if ok else 'BAD'} {rec['calls']} {problems} {res['content'][:80]!r}", flush=True)
    lo, hi = wilson(k, n)
    print(json.dumps({"n": n, "correct": k, "acc": k / n, "ci95": [lo, hi]}))


if __name__ == "__main__":
    main()
