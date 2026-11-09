#!/usr/bin/env python3
"""Speculative-decoding draft acceptance per language, and answer identity between two server configurations.

Sends greedy long answers in English, Chinese and code to a running OpenAI-compatible server (TensorSharp, a draft
head or drafter configured), saves each answer's text, and reads the server log's per-request speculative stats
("Speculative decoding stats for ...: drafted=N accepted=M") to report the acceptance per language. A second command
compares two runs' texts request by request - the way to show that a drafting change (a draft vocabulary, a drafter)
changes no output: run both configurations with the answers made history-independent (for Qwen3.8 Flash Next on
--backend cuda: TS_MOE_ROUTE_TAIL_SKIP=0 TS_MOE_TIER_ADAPT=0) and expect identical texts.

  # 1. answers (thinking off, max_tokens 700, temperature 0), appended to a JSON-lines file
  python3 eng/draft-acceptance.py run --url http://127.0.0.1:5088 --out artifacts/x/lang-on.jsonl --repeats 2
  # 2. acceptance per language: the run's requests are the server log's last speculative-stats lines, in order
  python3 eng/draft-acceptance.py stats --log artifacts/x/server-on.log --runs artifacts/x/lang-on.jsonl
  # 3. two runs' texts, and the share of output tokens at or past a token id (a prefix draft vocabulary)
  python3 eng/draft-acceptance.py compare artifacts/x/lang-on.jsonl artifacts/x/lang-off.jsonl \\
      --tokenizer qwen38-tokenizer.json --vocab 151643

Standard library only, except `compare --tokenizer`, which needs the `tokenizers` package.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

PROMPTS = {
    "en": "Write a detailed essay of about 600 words on how lighthouses were built and kept in the nineteenth century, "
          "covering the engineering, the keepers' daily routine, and why most were automated later.",
    "zh": "请用中文写一篇约八百字的文章，介绍十九世纪灯塔是如何建造和维护的，包括工程技术、守塔人的日常工作，"
          "以及后来大多数灯塔为什么改为自动化。",
    "code": "Write a Python module that implements an LRU cache with TTL expiry, with docstrings and a short unittest "
            "suite. Return only the code.",
}


def cmd_run(a):
    model = a.model or json.loads(urllib.request.urlopen(a.url + "/v1/models", timeout=30).read())["data"][0]["id"]
    with open(a.out, "a", encoding="utf-8") as f:
        for rep in range(a.repeats):
            for lang, prompt in PROMPTS.items():
                body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                                   "max_tokens": a.max_tokens, "temperature": 0, "stream": False, "think": False,
                                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
                req = urllib.request.Request(a.url + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                t0 = time.time()
                res = json.loads(urllib.request.urlopen(req, timeout=a.timeout).read())
                dt = time.time() - t0
                usage = res.get("usage", {})
                text = res["choices"][0]["message"].get("content") or ""
                n = usage.get("completion_tokens") or 0
                rec = {"lang": lang, "rep": rep, "seconds": round(dt, 3), "completion_tokens": n,
                       "tok_s": round(n / dt, 1), "chars": len(text), "text": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps({k: v for k, v in rec.items() if k != "text"}, ensure_ascii=False), flush=True)


def cmd_stats(a):
    stats = [l for l in open(a.log, errors="replace") if "Speculative decoding stats for" in l]
    recs = [json.loads(l) for l in open(a.runs, encoding="utf-8")]
    if len(stats) < len(recs):
        sys.exit(f"the log holds {len(stats)} speculative-stats lines for {len(recs)} requests")
    agg = {}
    for rec, line in zip(recs, stats[-len(recs):]):
        m = re.search(r"drafted=(\d+) accepted=(\d+)", line)
        drafted, accepted = int(m.group(1)), int(m.group(2))
        g = agg.setdefault(rec["lang"], [0, 0, 0, 0.0])
        g[0] += drafted
        g[1] += accepted
        g[2] += rec["completion_tokens"] or 0
        g[3] += rec["seconds"]
        print(f"{rec['lang']:5s} rep {rec['rep']}: drafted {drafted:4d} accepted {accepted:4d} "
              f"({100.0 * accepted / max(1, drafted):5.1f}%)  {rec['completion_tokens']} tokens in {rec['seconds']:.2f} s")
    for lang, (drafted, accepted, n, s) in agg.items():
        print(f"== {lang:5s}: acceptance {100.0 * accepted / max(1, drafted):5.1f}% ({accepted}/{drafted}), "
              f"{n / max(s, 1e-9):6.1f} tok/s over {n} tokens")


def cmd_compare(a):
    x = [json.loads(l) for l in open(a.first, encoding="utf-8")]
    y = [json.loads(l) for l in open(a.second, encoding="utf-8")]
    tok = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
    same = 0
    for p, q in zip(x, y):
        ok = p.get("text") == q.get("text")
        same += ok
        line = f"{p['lang']:5s} rep {p['rep']}: {'identical' if ok else 'DIFFERENT'} ({p['chars']} / {q['chars']} chars)"
        if tok is not None and p.get("text"):
            ids = tok.encode(p["text"], add_special_tokens=False).ids
            line += f"; {sum(1 for i in ids if i >= a.vocab)} of {len(ids)} tokens at or past {a.vocab}"
        print(line)
    print(f"{same} of {min(len(x), len(y))} answers identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send the prompts, append answers to --out")
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", help="model id (default: the first of GET /v1/models)")
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--max-tokens", type=int, default=700)
    r.add_argument("--timeout", type=int, default=600)
    s = sub.add_parser("stats", help="acceptance per language from a server log")
    s.add_argument("--log", required=True)
    s.add_argument("--runs", required=True)
    c = sub.add_parser("compare", help="two runs' texts, request by request")
    c.add_argument("first")
    c.add_argument("second")
    c.add_argument("--tokenizer")
    c.add_argument("--vocab", type=int, default=151643)
    a = ap.parse_args()
    {"run": cmd_run, "stats": cmd_stats, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
