"""Transformers goldens for the Raon-1 nano parity tests (InferenceWeb.Tests/RaonParityTests.cs).

Runs KRAFTON/Raon-1-nano's own modeling code (trust_remote_code) in float32 on the CPU and records, per chat prompt,
the rendered prompt (apply_chat_template, enable_thinking=False), its token ids, a greedy continuation and the top
log-probabilities of every generated position (ten at the first, four after it), plus HF tokenizer ids for a small
text corpus.

    python eng/validation/raon-hf-goldens.py --model <hf snapshot dir> --gguf <raon-1-nano-Q8_0.gguf> \
        --out InferenceWeb.Tests/Fixtures/Raon/hf_goldens.json
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("capital", "What is the capital of France? Answer in one word."),
    ("math", "Compute 17 * 23 and explain the steps briefly."),
    ("korean", "한국의 수도는 어디인가요? 한 문장으로 답해 주세요."),
    ("code", "Write a Python function that returns the n-th Fibonacci number."),
    ("story", "Tell me a two-sentence story about a lighthouse keeper."),
]

TOKENIZER_CORPUS = [
    "Hello, world!",
    "  leading spaces and\ttabs\n\nnewlines  ",
    "Numbers: 1234567890 3.14159 -42 1e10",
    "한국어 문장과 English가 섞인 텍스트입니다.",
    "日本語と中文的混合文本。",
    "def f(x):\n    return x ** 2  # square\n",
    "Emoji 🚀🔥 and accents: café naïve résumé",
    "<think>tags</think> stay text? <|im_end|>",
    "URL https://example.com/path?q=1&r=2",
    "ＦＵＬＬＷＩＤＴＨ and composed e\u0301 forms",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True, help="the GGUF the C# tests load (hashed into the fixture)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 8)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    records = []
    for name, text in PROMPTS:
        messages = [{"role": "user", "content": text}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        generated, steps = [], []
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            for i in range(args.tokens):
                lp = torch.log_softmax(logits, -1)
                k = 10 if i == 0 else 4
                top = torch.topk(lp, k)
                choice = int(top.indices[0])
                generated.append(choice)
                steps.append({"logprob": float(lp[choice]),
                              "top": [[int(t), float(v)] for v, t in zip(top.values, top.indices)]})
                if choice in eos:
                    break
                out = model(input_ids=torch.tensor([[choice]]), past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1].float()
        records.append({"name": name, "user": text, "prompt": prompt, "prompt_tokens": ids,
                        "generated_tokens": generated, "generated_text": tok.decode(generated), "steps": steps})
        print(f"{name}: {len(ids)} prompt tokens -> {tok.decode(generated)!r}", file=sys.stderr)

    corpus = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in TOKENIZER_CORPUS]
    doc = {
        "reference": {"transformers": transformers.__version__, "device": "cpu", "dtype": "float32"},
        "model": {"file": os.path.basename(args.gguf), "sha256": sha256(args.gguf), "hf": "KRAFTON/Raon-1-nano"},
        "torch": torch.__version__,
        "records": records,
        "tokenizer": corpus,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
