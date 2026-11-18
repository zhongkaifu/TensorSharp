#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Needle-in-a-haystack retrieval at long context, the same prompts for every server.

    needle.py --url http://127.0.0.1:5731 --haystack-dir docs --lengths 8192,32768,65536,131072 \
              --depths 0.1,0.5,0.9 --out ts-needle.jsonl

The haystack is real prose: the Markdown files under --haystack-dir, concatenated in sorted order and
cut to the target length (characters per token calibrated on the server's own prompt_tokens, then
trimmed until the prompt fits). One needle sentence (a passphrase) goes in at each depth fraction;
the question asks for it after the haystack. Each request is greedy, thinking off, non-streaming, and
asks for the passphrase and then the sentence it came from, so the reply also exercises decoding after
the long prompt. --tokenize-url (a llama-server on the same GGUF; the tokenizers agree) sizes every
prompt to its target exactly; without it the characters per token are calibrated once. Records
prompt_tokens, whether the reply holds the passphrase, and the prefill / decode timings: llama-server's
own "timings", or TensorSharp's chat.complete log line (ttftMs, tokensPerSec) with --server-log.
A fresh nonce at the top of every prompt keeps a prefix cache from serving it.
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY  # noqa: E402

NEEDLE = "Remember this: the secret passphrase for the archive vault is \"{code}\"."
QUESTION = ("\n\nAnswer from the document above. What is the secret passphrase for the archive vault? "
            "Give the passphrase on the first line, then quote the sentence that contains it.")


def corpus(directory):
    parts = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)):
        with open(path, encoding="utf-8", errors="replace") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def build(text, chars, depth, code, nonce):
    hay = text[:chars]
    at = hay.rfind("\n", 0, int(len(hay) * depth)) + 1
    return (f"Document {nonce}:\n" + hay[:at] + NEEDLE.format(code=code) + "\n" + hay[at:] + QUESTION)


def count_tokens(url, text):
    req = urllib.request.Request(url.rstrip("/") + "/tokenize", json.dumps({"content": text}).encode(),
                                 {"Content-Type": "application/json"})
    return len(json.loads(urllib.request.urlopen(req, timeout=600).read())["tokens"])


def ts_timings(log_path, offset, prompt_tokens):
    """The last chat.complete line written after offset for a prompt of this length."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        lines = re.findall(r"chat\.complete tokens=(\d+) promptTokens=(\d+) .*?ttftMs=(\d+) elapsedMs=([\d.]+) "
                           r"tokensPerSec=([\d.]+)", f.read())
    for t in reversed(lines):
        if int(t[1]) == prompt_tokens or prompt_tokens is None:
            return {"prefill_s": int(t[2]) / 1000, "decode_tps": float(t[4])}
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--haystack-dir", required=True)
    ap.add_argument("--lengths", default="8192,32768,65536,131072")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenize-url", help="llama-server URL whose /tokenize sizes each prompt exactly")
    ap.add_argument("--server-log", help="TensorSharp server log, for prefill / decode timings")
    args = ap.parse_args()
    cl = Client(args.url)
    text = corpus(args.haystack_dir)
    if args.tokenize_url:
        # Sized by the tokenizer endpoint: no 10K-token probe prompt on the server under test (minutes on a slow lane).
        cpt = 40000 / max(1, count_tokens(args.tokenize_url, text[:40000]))
    else:
        probe = cl.chat([{"role": "user", "content": text[:40000]}], sampling=GREEDY, max_tokens=1)
        cpt = 40000 / max(1, (probe["usage"] or {}).get("prompt_tokens", 10000))
    print(f"corpus {len(text)} chars, {cpt:.2f} chars/token", flush=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for n_tok in [int(x) for x in args.lengths.split(",")]:
            for depth in [float(x) for x in args.depths.split(",")]:
                code = f"amber-{n_tok % 9973}-{int(depth * 100)}-kestrel"
                chars = int((n_tok - 200) * cpt)
                if chars > len(text):
                    text = (text + "\n\n") * (chars // len(text) + 1)
                prompt = build(text, chars, depth, code, f"{n_tok}-{depth}")
                if args.tokenize_url:
                    # The chat framing adds about 20 tokens; aim the content 30 short of the target.
                    for _ in range(6):
                        have = count_tokens(args.tokenize_url, prompt)
                        if abs(have - (n_tok - 30)) <= 16:
                            break
                        chars = int(chars * (n_tok - 30) / have)
                        if chars > len(text):
                            text = (text + "\n\n") * (chars // len(text) + 1)
                        prompt = build(text, chars, depth, code, f"{n_tok}-{depth}")
                offset = os.path.getsize(args.server_log) if args.server_log else 0
                res = cl.chat([{"role": "user", "content": prompt}], sampling=GREEDY, max_tokens=args.max_tokens)
                pt = (res.get("usage") or {}).get("prompt_tokens")
                ct = (res.get("usage") or {}).get("completion_tokens")
                timings = {}
                raw_t = (res.get("raw") or {}).get("timings")
                if raw_t:
                    timings = {"prefill_s": raw_t.get("prompt_ms", 0) / 1000, "decode_tps": raw_t.get("predicted_per_second")}
                elif args.server_log:
                    timings = ts_timings(args.server_log, offset, pt)
                rec = {"target": n_tok, "depth": depth, "prompt_tokens": pt, "found": code in res["content"],
                       "quoted": "archive vault" in res["content"].split("\n", 1)[-1], "total": round(res["total"], 2),
                       "completion_tokens": ct, **timings, "status": res["status"],
                       "error": res.get("error"), "reply": res["content"], "id": f"{n_tok}@{depth}", "correct": code in res["content"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{n_tok:>7} depth {depth:.2f}: prompt {pt} found={rec['found']} prefill={timings.get('prefill_s') or 0:.1f}s "
                      f"decode={timings.get('decode_tps') or 0:.1f} tok/s total={res['total']:.1f}s {res['content'][:70]!r} "
                      f"{res.get('error') or ''}", flush=True)


if __name__ == "__main__":
    main()
