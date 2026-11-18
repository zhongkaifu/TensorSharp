#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Prompt-token parity of chat requests: TensorSharp's render against llama-server's and Hugging Face's.

Each step writes one JSONL file of {"id", "tokens"} for the same request file; `diff` compares them.

    # the requests: GSM8K questions as eng/lfm3/e2e/gsm8k.py sends them (thinking off unless --think)
    prompt-parity.py requests --gsm8k test.jsonl --n 200 --out req.jsonl
    # TensorSharp: the server's renderer and tokenizer (KVCachePromptRenderer over the GGUF template)
    dotnet eng/lfm3/Lfm3Probe/bin/Release/net10.0/Lfm3Probe.dll chat LFM3-1.7B-Q4_K_M.gguf req.jsonl ts.jsonl
    # llama.cpp: /apply-template (the code path /v1/chat/completions renders with) and /tokenize, special tokens parsed
    prompt-parity.py llama --url http://127.0.0.1:18431 --requests req.jsonl --out llama.jsonl
    # Hugging Face: tokenizer.apply_chat_template from the checkpoint directory (needs transformers; run it with
    # the Python that has it, e.g. ~/work/venvs/vllm-cpu/bin/python -I)
    prompt-parity.py hf --model-dir hf-1.7b-full --requests req.jsonl --out hf.jsonl
    prompt-parity.py diff ts.jsonl llama.jsonl hf.jsonl

`diff` prints, per pair, how many requests have identical token ids and the first differing position of each
that does not (with the tokens around it); its exit code is 1 when any pair differs. The llama step pins
chat_template_kwargs.enable_thinking from the request, as eng/lfm3/chat-template-fixtures.py does.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e"))


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_requests(args):
    from gsm8k import PROMPT  # noqa: E402  (the e2e harness's exact prompt)
    rows = read_jsonl(args.gsm8k)[args.offset:args.offset + args.n]
    write_jsonl(args.out, [{"id": args.offset + i,
                            "messages": [{"role": "user", "content": PROMPT.format(question=row["question"])}],
                            "enable_thinking": bool(args.think)} for i, row in enumerate(rows)])
    print(f"{len(rows)} requests -> {args.out}")


def post(url, path, body):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def cmd_llama(args):
    out = []
    for r in read_jsonl(args.requests):
        prompt = post(args.url, "/apply-template", {
            "messages": r["messages"],
            "chat_template_kwargs": {"enable_thinking": r.get("enable_thinking", False)}})["prompt"]
        # The rendered text carries the BOS as "<|startoftext|>": special tokens parsed, none added.
        tokens = post(args.url, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
        out.append({"id": r["id"], "tokens": tokens})
    write_jsonl(args.out, out)
    print(f"{len(out)} llama.cpp renders -> {args.out}")


def cmd_hf(args):
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    out = []
    for r in read_jsonl(args.requests):
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True,
                                      enable_thinking=r.get("enable_thinking", False))
        if hasattr(ids, "keys"):   # transformers 5 returns a BatchEncoding
            ids = ids["input_ids"]
        out.append({"id": r["id"], "tokens": list(ids)})
    write_jsonl(args.out, out)
    print(f"{len(out)} Hugging Face renders -> {args.out}")


def cmd_diff(args):
    # Lfm3Probe writes the request's id as its JSON text ("0"), the Python steps as given (0).
    files = {p: {int(r["id"]): r["tokens"] for r in read_jsonl(p)} for p in args.files}
    names = list(files)
    differs = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = files[names[i]], files[names[j]]
            ids = sorted(set(a) & set(b))
            same = [k for k in ids if a[k] == b[k]]
            print(f"{os.path.basename(names[i])} vs {os.path.basename(names[j])}: {len(same)}/{len(ids)} identical "
                  f"(only in one: {len(set(a) ^ set(b))})")
            for k in ids:
                if a[k] == b[k]:
                    continue
                differs = True
                x, y = a[k], b[k]
                at = next((p for p in range(min(len(x), len(y))) if x[p] != y[p]), min(len(x), len(y)))
                print(f"  id {k}: lengths {len(x)} / {len(y)}, first difference at {at}: "
                      f"{x[max(0, at - 3):at + 3]} / {y[max(0, at - 3):at + 3]}")
    lengths = [len(t) for t in files[names[0]].values()]
    if lengths:
        print(f"prompt lengths ({os.path.basename(names[0])}): {min(lengths)}-{max(lengths)} tokens, "
              f"{sum(lengths)} in all")
    sys.exit(1 if differs else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("requests")
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--think", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("llama")
    p.add_argument("--url", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("hf")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("diff")
    p.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"requests": cmd_requests, "llama": cmd_llama, "hf": cmd_hf, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    main()
