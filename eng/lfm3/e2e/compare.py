#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Paired comparison of two eval result files (gsm8k.py, ifeval.py, toolcall_eval.py output).

    compare.py A.jsonl B.jsonl [--show 5]

Each line holds an "id" and a boolean "correct". Prints both accuracies with 95% Wilson intervals,
the items only one engine got right, the exact McNemar p-value of that split, and how many replies
were identical text (a greedy pair that matches token for token gives identical replies).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import paired_summary  # noqa: E402


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=0, help="print this many discordant items")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ids = sorted(set(A) & set(B), key=lambda x: (str(type(x)), x))
    a = [bool(A[i]["correct"]) for i in ids]
    b = [bool(B[i]["correct"]) for i in ids]
    s = paired_summary(a, b)
    same = sum(1 for i in ids if A[i].get("reply") is not None and A[i].get("reply") == B[i].get("reply"))
    s["identical_replies"] = same
    s["only_a_ids"] = [i for i, x, y in zip(ids, a, b) if x and not y]
    s["only_b_ids"] = [i for i, x, y in zip(ids, a, b) if y and not x]
    if args.json:
        print(json.dumps(s))
        return
    print(f"items {s['n']}  identical replies {same}")
    print(f"A {args.a}: {s['a']}/{s['n']} = {s['a_acc']:.3f}  95% CI [{s['a_ci'][0]:.3f}, {s['a_ci'][1]:.3f}]")
    print(f"B {args.b}: {s['b']}/{s['n']} = {s['b_acc']:.3f}  95% CI [{s['b_ci'][0]:.3f}, {s['b_ci'][1]:.3f}]")
    print(f"only A right {s['only_a']}  only B right {s['only_b']}  exact McNemar p = {s['mcnemar_p']:.3f}")
    for i in (s["only_a_ids"] + s["only_b_ids"])[:args.show]:
        print("---", i, "A:", json.dumps(A[i].get("reply", ""))[:300])
        print("   ", i, "B:", json.dumps(B[i].get("reply", ""))[:300])


if __name__ == "__main__":
    main()
