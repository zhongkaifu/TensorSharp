#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Speculative decoding on the server: greedy replies with and without speculation must be identical.

    spec.py --plain http://127.0.0.1:5733 --spec http://127.0.0.1:5736 --spec-log ts-spec.log --out spec.jsonl

--plain is a TensorSharp.Server.Host without speculation, --spec the same build and GGUF started with
--spec --spec-type ngram (or another drafter). Each prompt runs non-streaming and streaming on the
speculative server and once on the plain one, greedy, thinking off and on. The speculative server's
log lines "Speculative decoding stats for ..." give drafted / accepted tokens and verify steps.
Prints per-prompt identity, wall time on both servers (informational: the Mac is shared) and the
acceptance totals. Exit status 1 when any reply differs.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY  # noqa: E402

LIST = ", ".join(f"item-{i}" for i in range(80))
PROMPTS = [
    ("copy-list", f"Repeat this list exactly, then repeat it once more: {LIST}.", False, 700),
    ("code-edit", "Here is a Python function:\n\ndef area(r):\n    import math\n    return math.pi * r * r\n\n"
                  "def perimeter(r):\n    import math\n    return 2 * math.pi * r\n\nRewrite both functions with type "
                  "hints and docstrings, keeping the code otherwise identical.", False, 400),
    ("json-table", "Produce a JSON array of 12 objects with fields id (1-12), name (\"user-<id>\") and active "
                   "(true for even ids). Output only the JSON.", False, 600),
    ("chat", "Explain in three short paragraphs why the sky is blue.", False, 400),
    ("math-think", "What is 23*47? Show your work briefly.", True, 1200),
    ("story", "Write a 150-word story about a lighthouse keeper.", False, 400),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--spec-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plain, spec = Client(args.plain), Client(args.spec)
    start = os.path.getsize(args.spec_log) if os.path.exists(args.spec_log) else 0
    differ = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for name, prompt, think, max_tokens in PROMPTS:
            msgs = [{"role": "user", "content": prompt}]
            a = plain.chat(msgs, thinking=think, sampling=GREEDY, max_tokens=max_tokens)
            b = spec.chat(msgs, thinking=think, sampling=GREEDY, max_tokens=max_tokens)
            c = spec.chat(msgs, stream=True, thinking=think, sampling=GREEDY, max_tokens=max_tokens)
            same = (a["content"], a["reasoning"]) == (b["content"], b["reasoning"]) == (c["content"], c["reasoning"])
            differ += not same
            ct = (a.get("usage") or {}).get("completion_tokens")
            rec = {"id": name, "identical": same, "completion_tokens": ct, "plain_s": round(a["total"], 2),
                   "spec_s": round(b["total"], 2), "plain": a["content"][-200:], "spec": b["content"][-200:]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{name:12s} identical={same} tokens={ct} plain={a['total']:.2f}s spec={b['total']:.2f}s", flush=True)
    with open(args.spec_log, encoding="utf-8", errors="replace") as lf:
        lf.seek(start)
        stats = re.findall(r"Speculative decoding stats for \S+: drafted=(\d+) accepted=(\d+) acceptance=[^v]* "
                           r"verifySteps=(\d+) plainSteps=(\d+)", lf.read())
    drafted = sum(int(s[0]) for s in stats)
    accepted = sum(int(s[1]) for s in stats)
    print(json.dumps({"prompts": len(PROMPTS), "differ": differ, "requests_with_stats": len(stats), "drafted": drafted,
                      "accepted": accepted, "acceptance": accepted / drafted if drafted else None,
                      "verify_steps": sum(int(s[2]) for s in stats), "plain_steps": sum(int(s[3]) for s in stats)}))
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
