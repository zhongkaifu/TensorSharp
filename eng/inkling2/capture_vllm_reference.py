"""Capture reference logprobs for Inkling 2 Mini from a vLLM server (official checkpoint).

    python -I eng/inkling2/capture_vllm_reference.py --base http://127.0.0.1:8102 --out artifacts/inkling2/vllm-reference.json

For each prompt it records the token ids (vLLM's own tokenizer and, for chat
prompts, its chat template), the top-5 logprobs of every prompt token after the
first, and a 32-token greedy continuation with its top-5 logprobs. TensorSharp's
port is checked against these by teacher forcing the same ids. Standard library only.
"""

from __future__ import annotations

import argparse
import http.client
import json
import urllib.parse

RAW_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1905, Albert Einstein published four papers that",
    "Translate to French: The weather is lovely today.\nFrench:",
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
    "我们今天去公园散步，天气",
    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox",
]

CHAT_PROMPTS = [
    ([{"role": "user", "content": "What is 17 * 23? Reply with the number only."}], {"reasoning_effort": "none"}),
    ([{"role": "user", "content": "Write a haiku about rain."}], {"reasoning_effort": "low"}),
    ([{"role": "system", "content": "You are a terse assistant."},
      {"role": "user", "content": "Name three prime numbers."}], {"reasoning_effort": "none"}),
    ([{"role": "user", "content": "Explain recursion in one sentence."}], None),
]


def post(base: str, path: str, body: dict) -> dict:
    url = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=600)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    data = response.read()
    if response.status != 200:
        raise RuntimeError(f"{path} -> {response.status}: {data[:400]!r}")
    return json.loads(data)


def top(entry) -> list:
    """vLLM's {token_id: {logprob, rank, decoded_token}} as a list sorted by logprob."""
    if not entry:
        return []
    items = [(int(k), v["logprob"], v.get("decoded_token")) for k, v in entry.items()]
    return sorted(items, key=lambda item: -item[1])


def capture(base: str, model: str, ids: list[int]) -> dict:
    reply = post(base, "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": 32, "temperature": 0,
        "logprobs": 5, "prompt_logprobs": 5, "return_token_ids": True,
    })
    choice = reply["choices"][0]
    prompt_lp = [top(e) for e in (choice.get("prompt_logprobs") or [])]
    gen = choice.get("logprobs") or {}
    generated_ids = choice.get("token_ids")
    gen_top = []
    for entry in gen.get("top_logprobs") or []:
        gen_top.append(sorted(((t, lp) for t, lp in entry.items()), key=lambda item: -item[1]))
    return {"prompt_ids": ids, "prompt_top5": prompt_lp, "completion": choice["text"],
            "completion_ids": generated_ids, "completion_tokens": gen.get("tokens"),
            "completion_top5_text": gen_top}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8102")
    parser.add_argument("--model", default="inkling-2-mini")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = []
    for text in RAW_PROMPTS:
        ids = post(args.base, "/tokenize", {"model": args.model, "prompt": text, "add_special_tokens": True})["tokens"]
        cases.append({"kind": "raw", "text": text, **capture(args.base, args.model, ids)})
    for messages, kwargs in CHAT_PROMPTS:
        body = {"model": args.model, "messages": messages, "add_generation_prompt": True}
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        tokenized = post(args.base, "/tokenize", body)
        cases.append({"kind": "chat", "messages": messages, "chat_template_kwargs": kwargs,
                      **capture(args.base, args.model, tokenized["tokens"])})
    meta = {"server": "vLLM inkling-2-mini pre-release 7cb5dae", "checkpoint": "thinkingmachines/Inkling-2-Mini@ed40e7da",
            "kv_cache_dtype": "fp8_e4m3", "speculative": "dspark (greedy output is target-exact)", "cases": cases}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)
    for case in cases:
        print(case["kind"], len(case["prompt_ids"]), repr(case["completion"][:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
