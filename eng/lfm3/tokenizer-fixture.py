#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""
Regenerate InferenceWeb.Tests/Fixtures/Lfm3Tokenizer/llamacpp-reference.json: the token ids llama.cpp's
llama-tokenize gives every case below for an LFM3 GGUF (tokenizer.ggml.pre "lfm2": the LLAMA3 split with
ignore_merges), with special tokens parsed and no BOS added - what TensorSharp's Encode(text, addSpecial:
false) must reproduce (Lfm3TokenizerParityTests).

    python3 -I eng/lfm3/tokenizer-fixture.py \
        --llama-tokenize /path/to/llama.cpp/build/bin/llama-tokenize \
        --model /path/to/LFM3-1.7B-Q8_0.gguf \
        --out InferenceWeb.Tests/Fixtures/Lfm3Tokenizer/llamacpp-reference.json

Every LFM3 GGUF (1.7B and Omni-2B, any quant) carries the same vocabulary and merges.
"""
import argparse
import json
import os
import subprocess
import tempfile

CASES = [
    # digits: LLAMA3 keeps runs of up to three
    "1", "12", "123", "1234", "12345", "1234567890", "3.14159", "2026-11-16", "v1.2.3", "$1,234,567.89",
    "INV-472", "silver-4821", "0x7fffffff", "1e-30", "x² + y³ = z⁴", "٣٤٥ ٦٧", "१२३४",
    # contractions and case
    "I'm here, you're there, they've gone, we'll see, he'd say, it's OK", "I'M HERE, YOU'RE THERE", "don't Don't DON'T",
    # whitespace runs
    "a  b   c    d", "  leading", "trailing  ", "\ttab\tseparated\t", "line1\nline2\n\nline4", "crlf\r\nline\r\n",
    "    indented code", "\n\n\n", " \n \n", "a b c　d",
    # CJK and other scripts
    "你好，世界！今天天气很好。", "日本語のテキストです。カタカナとひらがな。", "한국어 텍스트입니다.", "Привет, мир! Как дела?",
    "مرحبا بالعالم", "नमस्ते दुनिया", "Γειά σου Κόσμε", "שלום עולם", "ไทย ภาษา", "Tiếng Việt có dấu",
    "naïve café résumé façade", "é combining acute", "Ångström Å",
    # emoji and supplementary plane
    "Hello 👋 world 🌍!", "👨‍👩‍👧‍👦 family", "🇺🇸🇯🇵 flags", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙", "𠀀𠀁 rare CJK",
    # code
    "def f(x):\n    return x * 2  # double\n", "for (int i = 0; i < n; i++) { sum += a[i]; }",
    "if (a != b && c >= d) return -1;", "console.log(`template ${value}`);", "#include <stdio.h>\nint main(void){}",
    "SELECT * FROM t WHERE id = 42;", "<div class=\"x\">&amp;</div>", "path/to/file.txt", "https://example.com/a?b=c&d=e",
    "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}",
    # punctuation runs
    "Wait... what?!", "---===---", "!!!???", "a--b", "(((nested)))", "«quotes» “curly” ‘single’",
    # special tokens and the chat framing
    "<|startoftext|><|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "<|vision_start|><|image|><|vision_end|>Describe this.", "text<|endoftext|>more", "<think></think>Answer: 42",
    # mixed
    "The quick brown fox jumps over the lazy dog. 1234 quick-brown_fox.",
    "Mixed中文English混合text123数字", "Zürich, München und Köln", "C'est l'été à Paris!",
]


def tokenize(binary, model, text):
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(text.encode("utf-8"))
        name = f.name
    try:
        out = subprocess.run(
            [binary, "-m", model, "-f", name, "--ids", "--no-bos", "--no-escape", "--log-disable"],
            check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(name)
    line = out.strip().splitlines()[-1]
    return json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-tokenize", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cases = [{"text": t, "ids": tokenize(a.llama_tokenize, a.model, t)} for t in CASES]
    doc = {
        "source": "llama.cpp llama-tokenize (parse_special, --no-bos, --no-escape) over an LFM3 GGUF "
                  "(tokenizer.ggml.pre lfm2). Regenerate with eng/lfm3/tokenizer-fixture.py.",
        "cases": cases,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("{\n\"source\": " + json.dumps(doc["source"]) + ",\n\"cases\": [\n")
        f.write(",\n".join(json.dumps(c, ensure_ascii=False) for c in cases))
        f.write("\n]\n}\n")
    print(f"{len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
