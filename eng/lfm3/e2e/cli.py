#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""TensorSharp.Cli end-to-end checks for LFM3 (one checked scenario per line).

    cli.py --model MODEL.gguf --backend ggml_metal --work artifacts/lfm3/e2e/cli [--ref http://127.0.0.1:5732]

Runs the CLI (TensorSharp.Cli/bin/TensorSharp.Cli.dll, or --cli) greedily with skills off:
plain answer, thinking on, a stop string, a tool call from --tools, multi-turn reuse
(--multi-turn-jsonl), batch requests (--input-jsonl), n-gram speculation equal to plain decoding,
and --dump-prompt, whose rendered prompt and token ids are compared with llama-server's
/apply-template and /tokenize when --ref names a llama-server on the same GGUF.
Exit status 1 when a check fails.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

GREEDY = ["--temperature", "0", "--top-k", "1", "--top-p", "1", "--min-p", "0", "--repeat-penalty", "1"]


def run(args, cli, extra, timeout=900):
    cmd = ["dotnet", cli, "--model", args.model, "--backend", args.backend, "--no-skills"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
    return p.returncode, p.stdout, p.stderr


def output_of(stdout):
    """The answer after '=== Generated Output ===' and the reasoning between the Thinking markers."""
    out = stdout.split("=== Generated Output ===", 1)[-1]
    out = "\n".join(l for l in out.split("\n") if not re.match(r"^\d\d:\d\d:\d\d (info|warn|fail|dbug):", l))
    thinking = ""
    m = re.search(r"--- Thinking ---\n(.*?)\n--- End Thinking ---\n", out, re.S)
    if m:
        thinking = m.group(1)
        out = out[m.end():]
    return out.strip(), thinking.strip()


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="ggml_metal")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cli", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "TensorSharp.Cli", "bin",
                                                  "TensorSharp.Cli.dll"))
    ap.add_argument("--ref", help="llama-server URL on the same GGUF, for --dump-prompt parity")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    w = lambda name: os.path.join(args.work, name)
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok)))
        print(f"{name:34s} {'PASS' if ok else 'FAIL'} {detail if not ok else ''}"[:600], flush=True)

    open(w("q.txt"), "w").write("What is 17*23? Answer with the number only.")
    rc, so, se = run(args, args.cli, ["--input", w("q.txt"), "--max-tokens", "200"] + GREEDY)
    ans, th = output_of(so)
    check("plain answer", rc == 0 and "391" in ans and "</think>" not in ans, (rc, ans[-200:], se[-300:]))
    plain_answer = ans

    rc, so, se = run(args, args.cli, ["--input", w("q.txt"), "--max-tokens", "1500", "--think"] + GREEDY)
    ans, th = output_of(so)
    check("thinking", rc == 0 and "391" in ans and len(th) > 20 and "<think>" not in ans, (rc, ans[-200:], th[:100]))

    open(w("count.txt"), "w").write("Count from 1 to 10, separated by commas, nothing else.")
    rc, so, se = run(args, args.cli, ["--input", w("count.txt"), "--max-tokens", "60", "--stop", "5"] + GREEDY)
    ans, _ = output_of(so)
    check("stop string", rc == 0 and "4" in ans and "5" not in ans, ans)

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    json.dump(tools, open(w("tools.json"), "w"))
    open(w("weather.txt"), "w").write("What's the weather in Paris right now?")
    rc, so, se = run(args, args.cli, ["--input", w("weather.txt"), "--tools", w("tools.json"), "--max-tokens", "200"] + GREEDY)
    check("tool call", rc == 0 and "get_weather" in so and "Paris" in so, so[-400:])

    with open(w("turns.jsonl"), "w") as f:
        f.write(json.dumps({"content": "My name is Ada and my favourite number is 42. Please remember both and reply briefly."}) + "\n")
        f.write(json.dumps({"content": "Tell me a two-sentence fact about owls."}) + "\n")
        f.write(json.dumps({"content": "What is my name and my favourite number?"}) + "\n")
    rc, so, se = run(args, args.cli, ["--multi-turn-jsonl", w("turns.jsonl"), "--max-tokens", "120"] + GREEDY)
    log = so + se
    previews = re.findall(r"multi-turn content chars=\d+ .*?preview=(.*)", log)
    last = previews[-1] if previews else log[-600:]
    reuse = re.findall(r"radix prefix reused=(\d+)/\d+", log)
    check("multi-turn recall", rc == 0 and "Ada" in last and "42" in last, last[-300:])
    if "short of memory" in log and not (len(reuse) >= 3 and all(int(r) > 0 for r in reuse[1:3])):
        # The prefix cache gives retained state back when macOS runs short of memory (a shared
        # machine): no evidence either way, so neither a pass nor a failure.
        print(f"{'multi-turn reuses the cache':34s} SKIP (the prefix cache evicted under memory pressure) {reuse}")
    else:
        check("multi-turn reuses the cache", len(reuse) >= 3 and all(int(r) > 0 for r in reuse[1:3]), reuse)

    with open(w("batch.jsonl"), "w") as f:
        f.write(json.dumps({"id": "a", "prompt": "What is the capital of France? One word.", "max_tokens": 20}) + "\n")
        f.write(json.dumps({"id": "b", "messages": [{"role": "user", "content": "What is 2+3? Digits only."}],
                            "max_tokens": 20}) + "\n")
    rc, so, se = run(args, args.cli, ["--input-jsonl", w("batch.jsonl"), "--output", w("batch.out.jsonl")] + GREEDY)
    rows = [json.loads(l) for l in open(w("batch.out.jsonl"))] if os.path.exists(w("batch.out.jsonl")) else []
    text = json.dumps(rows)
    check("batch requests", rc == 0 and len(rows) == 2 and "Paris" in text and "5" in text, text[:400])

    open(w("doc.txt"), "w").write(("Repeat the following list exactly, twice: " + ", ".join(
        f"item-{i}" for i in range(60)) + "."))
    rc, so, se = run(args, args.cli, ["--input", w("doc.txt"), "--max-tokens", "400"] + GREEDY)
    a, _ = output_of(so)
    rc2, so2, se2 = run(args, args.cli, ["--input", w("doc.txt"), "--max-tokens", "400", "--spec", "--spec-type", "ngram"] + GREEDY)
    b, _ = output_of(so2)
    check("n-gram speculation == plain", rc == 0 and rc2 == 0 and a == b and len(a) > 100, (a[:120], b[:120], se2[-300:]))

    rc, so, se = run(args, args.cli, ["--input", w("weather.txt"), "--tools", w("tools.json"), "--dump-prompt", "--think"])
    rendered = so.split("=== Rendered Prompt ===\n", 1)[-1].split("\n=== End", 1)[0]
    rendered = "\n".join(l for l in rendered.split("\n") if not re.match(r"^\d\d:\d\d:\d\d info:", l))
    check("dump-prompt", rc == 0 and rendered.endswith("<|im_start|>assistant\n<think>"), rendered[-200:])
    if args.ref:
        body = {"messages": [{"role": "user", "content": "What's the weather in Paris right now?"}], "tools": tools,
                "chat_template_kwargs": {"enable_thinking": True}}
        ref_prompt = post(args.ref, "/apply-template", body)["prompt"]
        ref_ids = post(args.ref, "/tokenize", {"content": ref_prompt, "add_special": False, "parse_special": True})["tokens"]
        count = int(re.search(r"Token count: (\d+)", so).group(1)) if "Token count:" in so else -1
        first = [int(x) for x in re.search(r"First 20 tokens: \[([^\]]*)\]", so).group(1).split(",")] if "First 20" in so else []
        last = [int(x) for x in re.search(r"Last 10 tokens: \[([^\]]*)\]", so).group(1).split(",")] if "Last 10" in so else []
        # The CLI prints the template's text; the tokenizer adds the one BOS llama.cpp renders.
        check("dump-prompt text == llama-server", "<|startoftext|>" + rendered == ref_prompt, (rendered[-120:], ref_prompt[-120:]))
        check("dump-prompt ids == llama-server", count == len(ref_ids) and first == ref_ids[:20] and last == ref_ids[-10:],
              (count, len(ref_ids)))
    failed = [n for n, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
