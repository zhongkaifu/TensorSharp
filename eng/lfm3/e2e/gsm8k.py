#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""GSM8K accuracy of one OpenAI-compatible server (TensorSharp or llama-server), zero-shot.

    gsm8k.py --url http://127.0.0.1:5731 --data test.jsonl --n 200 --out ts-nothink.jsonl
    gsm8k.py --url http://127.0.0.1:5732 --data test.jsonl --n 50 --think --max-tokens 8192 --out llama-think.jsonl
    compare.py ts-nothink.jsonl llama-nothink.jsonl

The data is the public test split (openai/grade-school-math, grade_school_math/data/test.jsonl), first
--n questions after --offset. Greedy by default (temperature 0, top_k 1, no repetition penalty), so two
engines given the same prompt differ only where their numerics or prompt tokens do. The answer is the
number after the last "Answer:" in the reply (else the reply's last number), compared with the
reference after "####". Thinking off, the reply scored is the whole generated text (stray reasoning
the server moved to reasoning_content, "</think>", the content), so a parser difference between
engines does not change the score; content_correct scores the content alone. Each line of --out records the question id, the reply, the parsed and expected
answers and whether the reply ended in a stray </think> (thinking off). Resumes an existing --out.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY, wilson  # noqa: E402

PROMPT = ("Solve the following math problem. Explain your reasoning briefly, then give the final answer "
          "on the last line in the form \"Answer: <number>\".\n\n{question}")

NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize(s):
    s = s.replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v == int(v) else round(v, 6)


def extract(reply):
    m = list(re.finditer(r"Answer\s*[:：]\s*\**\s*(-?\$?\d[\d,]*(?:\.\d+)?)", reply, re.I))
    if m:
        return normalize(m[-1].group(1))
    boxed = list(re.finditer(r"\\boxed\{\s*(-?\$?\d[\d,]*(?:\.\d+)?)", reply))
    if boxed:
        return normalize(boxed[-1].group(1))
    nums = NUM.findall(reply)
    return normalize(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sampling", default="greedy",
                    help="greedy (the default), server (send no sampling fields: the server's defaults), "
                         "or a JSON object of request fields")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cl = Client(args.url)
    sampling = GREEDY if args.sampling == "greedy" else None if args.sampling == "server" else json.loads(args.sampling)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][args.offset:args.offset + args.n]
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    with open(args.out, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            qid = args.offset + i
            if qid in done:
                continue
            expected = normalize(row["answer"].split("####")[-1])
            res = cl.chat([{"role": "user", "content": PROMPT.format(question=row["question"])}],
                          thinking=args.think, sampling=sampling, max_tokens=args.max_tokens)
            reply = res["content"]
            # Thinking off, the whole generated text is scored, so two engines that generate the
            # same tokens score alike whether or not their parser moved stray reasoning (text
            # before a </think> the model wrote itself) out of the content; content_correct is
            # what a client reading only the content got. Thinking on, the answer is the content.
            full = reply if args.think or not res["reasoning"] else res["reasoning"] + "</think>" + reply
            got = extract(full)
            got_content = extract(reply)
            rec = {"id": qid, "correct": got is not None and expected is not None and abs(got - expected) < 1e-6,
                   "content_correct": got_content is not None and expected is not None and abs(got_content - expected) < 1e-6,
                   "got": got, "expected": expected, "finish": res["finish_reason"], "status": res["status"],
                   "stray_think_close": "</think>" in full, "reply": full, "content": reply, "reasoning": res["reasoning"],
                   "completion_tokens": (res.get("usage") or {}).get("completion_tokens"),
                   "seconds": round(res["total"], 2), "error": res.get("error")}
            done[qid] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            k = sum(1 for r in done.values() if r["correct"])
            print(f"[{len(done)}/{len(rows)}] id={qid} got={got} exp={expected} ok={rec['correct']} acc={k}/{len(done)}",
                  flush=True)
    k = sum(1 for r in done.values() if r["correct"])
    lo, hi = wilson(k, len(done))
    print(json.dumps({"n": len(done), "correct": k, "acc": k / len(done), "ci95": [lo, hi],
                      "content_correct": sum(1 for r in done.values() if r.get("content_correct", r["correct"])),
                      "stray_think_close": sum(1 for r in done.values() if r.get("stray_think_close")),
                      "length_cut": sum(1 for r in done.values() if r.get("finish") == "length")}))


if __name__ == "__main__":
    main()
