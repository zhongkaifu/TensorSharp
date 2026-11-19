#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Thinking-off replies cut after a stray </think>, against a real model on TensorSharp.Server.Host.

    thinkoff_cut.py --url http://127.0.0.1:5731 --data test.jsonl [--ids 15,38] [--want 3] [--tries 40] --out cut.jsonl

LFM3-1.7B with thinking off sometimes reasons past the prompt's closed <think></think> and closes the block itself
(docs/models/lfm3.md, Reasoning). For GSM8K questions (gsm8k.py's prompt, greedy) this finds replies that did so and
finished, then asks again with max_tokens set to cut the answer after the close, non-streaming and streaming, and
checks what the client gets:

  finished, non-streaming   the reasoning apart (reasoning_content), the answer as content
  cut, non-streaming        finish_reason "length", no reasoning_content, the reasoning and </think> in the content,
                            the same text as the cut stream
  cut, streaming            the same content

Greedy decoding of one request at a time repeats the first reply's tokens, so the cut lands where it was aimed; a cut
that lands before the close is reported as missed, not as a pass. It stops after --want exercised questions or --tries
questions asked (from --ids, else from the start of --data). Works with and without the server's sub-agent loop
(ts-server.sh passes --no-multi-agent; start the host without it to exercise the loop). Exit status 1 when a check fails,
2 when no question was exercised.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY  # noqa: E402
from gsm8k import PROMPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--want", type=int, default=3)
    ap.add_argument("--tries", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cl = Client(args.url)
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    ids = [int(x) for x in args.ids.split(",") if x.strip()] or list(range(len(rows)))
    exercised = failed = asked = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for qid in ids:
            if exercised >= args.want or asked >= args.tries:
                break
            asked += 1
            msgs = [{"role": "user", "content": PROMPT.format(question=rows[qid]["question"])}]
            full = cl.chat(msgs, thinking=False, sampling=GREEDY, max_tokens=args.max_tokens)
            n = (full.get("usage") or {}).get("completion_tokens") or 0
            rec = {"id": qid, "finish": full["finish_reason"], "completion_tokens": n,
                   "reasoning_chars": len(full["reasoning"]), "content_chars": len(full["content"])}
            if full["status"] != 200 or not full["reasoning"] or full["finish_reason"] != "stop" or "</think>" in full["content"]:
                rec["exercised"] = False
                f.write(json.dumps(rec) + "\n")
                print(f"id={qid} finish={full['finish_reason']} reasoning={len(full['reasoning'])} chars: not a finished stray close")
                continue
            # Cut half-way through the answer after the close (its share of the reply's characters, in tokens).
            answer_tokens = max(2, round(n * len(full["content"]) / max(1, len(full["reasoning"]) + len(full["content"]))))
            cut = n - max(1, answer_tokens // 2)
            a = cl.chat(msgs, thinking=False, sampling=GREEDY, max_tokens=cut)
            # The next turn sends the cut reply back as the client got it: the server recognises its own turn and
            # continues the KV cache through it (the first prompt and the generated tokens; the last one sampled at
            # max_tokens may never have been fed through the model). Asked before the stream below: a second,
            # identical conversation in between is a different reuse case.
            first_prompt = (a.get("usage") or {}).get("prompt_tokens") or 0
            nxt = cl.chat(msgs + [{"role": "assistant", "content": a["content"]},
                                  {"role": "user", "content": "Repeat only the final number."}],
                          thinking=False, sampling=GREEDY, max_tokens=16)
            cached = (((nxt.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
            b = cl.chat(msgs, thinking=False, stream=True, sampling=GREEDY, max_tokens=cut)
            landed = "</think>" in a["content"] or "</think>" in b["content"]
            rec["next_turn"] = {"first_prompt": first_prompt, "cut": cut, "cached": cached,
                                "prompt": (nxt.get("usage") or {}).get("prompt_tokens")}
            checks = {
                "next turn: cache continued through the cut reply": cached >= first_prompt + cut - 1,
                "finished: reasoning apart": bool(full["reasoning"]) and "</think>" not in full["content"],
                "cut: finish_reason length": a["finish_reason"] == "length" and b["finish_reason"] == "length",
                "cut: no reasoning_content": not a["reasoning"],
                "cut: reasoning and </think> in the content": a["content"].lstrip().startswith(full["reasoning"].strip()[:200])
                                                              and "</think>" in a["content"],
                "cut: non-stream == stream": a["content"] == b["content"],
            }
            rec.update({"exercised": landed, "cut_at": cut, "checks": checks, "cut_content_tail": a["content"][-300:],
                        "cut_stream_tail": b["content"][-300:], "cut_reasoning": a["reasoning"][:300]})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if not landed:
                print(f"id={qid} cut at {cut}/{n}: landed before the close (missed, not counted)")
                continue
            exercised += 1
            bad = [k for k, ok in checks.items() if not ok]
            failed += bool(bad)
            print(f"id={qid} cut at {cut}/{n} tokens: {'OK' if not bad else 'FAIL ' + ', '.join(bad)}")
    print(json.dumps({"asked": asked, "exercised": exercised, "failed": failed}))
    sys.exit(1 if failed else 2 if exercised == 0 else 0)


if __name__ == "__main__":
    main()
