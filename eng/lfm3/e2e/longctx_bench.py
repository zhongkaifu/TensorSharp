#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Prefill and decode speed after a long real-text prompt, TensorSharp vs llama-server, interleaved.

    longctx_bench.py --ts http://127.0.0.1:5738 --ts-log ts.log --llama http://127.0.0.1:5732 \
        --haystack-dir docs --lengths 4096,16384,32768 --rounds 2 --out bench.jsonl

Each round sends, for every length, the same prompt (repo prose cut to the length with llama-server's
/tokenize, opened by a nonce of the round and the length, so no prefix cache serves it: with the round's nonce
alone, llama-server's slot reused the previous, shorter prompt of the round and its prefill timed only the
new tokens) to TensorSharp and then to llama-server
(A/B, then the next round), greedy, ignore_eos, max_tokens --decode. Prefill time and decode rate come
from llama-server's "timings" and TensorSharp's chat.complete log line (ttftMs = prefill plus the first
token, tokensPerSec). The machine may be shared: read the spread across rounds before any claim.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY  # noqa: E402
from needle import corpus, count_tokens, ts_timings  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--ts-log", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--haystack-dir", required=True)
    ap.add_argument("--lengths", default="4096,16384,32768")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ts, ll = Client(args.ts), Client(args.llama)
    text = corpus(args.haystack_dir)
    rows = []
    with open(args.out, "a", encoding="utf-8") as f:
        for rnd in range(args.rounds):
            for n_tok in [int(x) for x in args.lengths.split(",")]:
                chars = n_tok * 4
                prompt = None
                for _ in range(6):
                    prompt = f"Round {rnd}, {n_tok} tokens, document:\n" + text[:chars] + "\n\nSummarize the document above."
                    have = count_tokens(args.llama, prompt)
                    if abs(have - (n_tok - 30)) <= 16:
                        break
                    chars = int(chars * (n_tok - 30) / have)
                msgs = [{"role": "user", "content": prompt}]
                for name, cl in (("ts", ts), ("llama", ll)):
                    offset = os.path.getsize(args.ts_log)
                    res = cl.chat(msgs, sampling=GREEDY, max_tokens=args.decode, ignore_eos=True)
                    pt = (res.get("usage") or {}).get("prompt_tokens")
                    if name == "llama":
                        t = res.get("raw", {}).get("timings") or {}
                        # prompt_n: the tokens it prefilled (less than the prompt where a cached prefix served the rest).
                        tim = {"prefill_s": t.get("prompt_ms", 0) / 1000, "prefilled": t.get("prompt_n"),
                               "decode_tps": t.get("predicted_per_second")}
                    else:
                        tim = ts_timings(args.ts_log, offset, pt)
                    rec = {"engine": name, "round": rnd, "target": n_tok, "prompt_tokens": pt,
                           "completion_tokens": (res.get("usage") or {}).get("completion_tokens"), **tim}
                    rows.append(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(json.dumps(rec), flush=True)
    for n_tok in sorted({r["target"] for r in rows}):
        for name in ("ts", "llama"):
            sel = [r for r in rows if r["engine"] == name and r["target"] == n_tok and r.get("prefill_s")]
            if sel:
                pp = [(r.get("prefilled") or r["prompt_tokens"]) / r["prefill_s"] for r in sel]
                tg = [r["decode_tps"] for r in sel if r.get("decode_tps")]
                print(f"{name:6s} {n_tok:>7}: prefill {statistics.median(pp):8.0f} tok/s (runs {[round(x) for x in pp]}) "
                      f"decode {statistics.median(tg):6.1f} tok/s (runs {[round(x, 1) for x in tg]})")


if __name__ == "__main__":
    main()
