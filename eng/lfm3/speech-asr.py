#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Round-trip intelligibility of speech output: transcribe WAVs with LFM3-Omni's ASR (llama.cpp's llama-mtmd-cli, the
model card's "Perform ASR in <Language>." system prompt, greedy) and score each transcript against the text that was
spoken: word error rate (English, French, ...) or character error rate (Chinese, Japanese, Korean).

    speech-asr.py LLAMA_BIN_DIR LM.gguf MMPROJ.gguf MANIFEST.tsv OUT.json

MANIFEST.tsv lines: id <TAB> language code <TAB> wav path <TAB> reference text. Run with `python3 -I` (stdlib only).
Text is normalized before scoring: case folded, punctuation dropped, English numbers 0-999 written as words, so "80"
and "eighty" agree.
"""
import json
import os
import re
import subprocess
import sys
import unicodedata

LANGUAGES = {"en": "English", "en-us": "English", "en-gb": "English", "zh": "Chinese", "fr": "French", "de": "German",
             "es": "Spanish", "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "nl": "Dutch"}
CHARACTER_LANGUAGES = {"zh", "ja", "ko"}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    rest = n % 100
    return ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))


def normalize(text, lang):
    text = unicodedata.normalize("NFKC", text).lower()
    if lang.startswith("en"):
        text = re.sub(r"\d+", lambda m: number_words(int(m.group())) if int(m.group()) < 1000 else m.group(), text)
        text = text.replace("-", " ")
    text = "".join(c if not unicodedata.category(c).startswith("P") else " " for c in text)
    if lang in CHARACTER_LANGUAGES:
        return [c for c in text if not c.isspace()]
    return text.split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def transcribe(llama_bin, lm, mmproj, wav, lang):
    system = "Perform ASR in %s." % LANGUAGES.get(lang, "English")
    out = subprocess.run([os.path.join(llama_bin, "llama-mtmd-cli"), "-m", lm, "--mmproj", mmproj, "--audio", wav, "-sys", system,
                          "-p", "<__media__>", "--temp", "0", "-n", "256", "-ngl", "99"],
                         capture_output=True, text=True, timeout=600)
    text = out.stdout
    marker = text.rfind("<think></think>")
    if marker >= 0:
        text = text[marker + len("<think></think>"):]
    return text.strip()


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    llama_bin, lm, mmproj, manifest, out_path = sys.argv[1:]
    rows = []
    for line in open(manifest, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        ident, lang, wav, reference = line.rstrip("\n").split("\t", 3)
        hypothesis = transcribe(llama_bin, lm, mmproj, wav, lang)
        ref, hyp = normalize(reference, lang), normalize(hypothesis, lang)
        errors = edit_distance(ref, hyp)
        rate = errors / max(1, len(ref))
        unit = "CER" if lang in CHARACTER_LANGUAGES else "WER"
        rows.append({"id": ident, "language": lang, "wav": wav, "reference": reference, "asr": hypothesis,
                     "unit": unit, "errors": errors, "length": len(ref), "rate": rate})
        print("%-28s %s %5.1f%%  %s" % (ident, unit, 100 * rate, hypothesis), flush=True)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
