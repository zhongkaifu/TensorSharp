#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""IFEval-style instruction-following spot check: prompts whose constraints a program can verify.

    ifeval.py --url http://127.0.0.1:5731 --out ts-ifeval.jsonl [--think]
    compare.py ts-ifeval.jsonl llama-ifeval.jsonl --show 5

Modelled on IFEval's verifiable instruction kinds (Zhou et al. 2023: length, case, keywords, format,
start/end, paragraphs, placeholders, JSON, language). A prompt is correct when every one of its checks
holds (IFEval's "prompt-level strict" accuracy). Greedy decoding, thinking off unless --think.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oaiclient import Client, GREEDY, wilson  # noqa: E402


def words(t):
    return re.findall(r"\b[\w'-]+\b", t)


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def bullets(t):
    return [l for l in t.splitlines() if re.match(r"^\s*[\*\-•]\s+", l)]


def is_json(t):
    t = t.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        json.loads(t)
        return True
    except json.JSONDecodeError:
        return False


def json_obj(t):
    t = t.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


CASES = [
    ("words-max-50", "Describe the water cycle in at most 50 words.", [lambda t: len(words(t)) <= 50]),
    ("words-min-150", "Write an essay of at least 150 words about why libraries matter.", [lambda t: len(words(t)) >= 150]),
    ("lowercase", "Explain what a black hole is. Your entire response must be in lowercase letters; no capital letters at all.",
     [lambda t: t == t.lower() and any(c.isalpha() for c in t)]),
    ("uppercase", "Give me three tips for staying focused. Your entire response must be in English and in ALL CAPITAL LETTERS.",
     [lambda t: t == t.upper() and any(c.isalpha() for c in t)]),
    ("bullets-4", "List exactly 4 benefits of regular exercise. Use markdown bullet points such as:\n* This is a point.\n"
                  "Do not write anything other than the 4 bullet points.", [lambda t: len(bullets(t)) == 4]),
    ("bullets-3", "Give exactly 3 bullet points (markdown '* ' bullets) about the planet Mars.", [lambda t: len(bullets(t)) == 3]),
    ("keywords", "Write a short paragraph about autumn that includes the words 'lantern' and 'cinnamon'.",
     [lambda t: "lantern" in t.lower(), lambda t: "cinnamon" in t.lower()]),
    ("forbidden-word", "Describe the ocean in two sentences without using the word 'blue' or 'water'.",
     [lambda t: "blue" not in t.lower(), lambda t: "water" not in t.lower()]),
    ("no-commas", "Write a short story about a cat in under 80 words without using any commas.",
     [lambda t: "," not in t, lambda t: len(words(t)) < 80]),
    ("json-format", "Give me information about the Eiffel Tower (name, city, height_m) as JSON. "
                    "Wrap the whole output in JSON format and output nothing else.",
     [is_json, lambda t: isinstance(json_obj(t), dict) and {"name", "city"} <= set(json_obj(t))]),
    ("json-list", "Return a JSON array of the first five prime numbers and nothing else.",
     [lambda t: json_obj(t) == [2, 3, 5, 7, 11]]),
    ("end-phrase", "Write a two-sentence thank-you note to a teacher. Finish your response with this exact phrase: "
                   "Is there anything else I can help with?",
     [lambda t: t.strip().endswith("Is there anything else I can help with?")]),
    ("start-word", "Write a haiku about rain. Your response must start with the word 'Softly'.",
     [lambda t: t.strip().lstrip("*#\"' ").startswith("Softly")]),
    ("title", "Write a short poem about friendship. Include a title wrapped in double angular brackets, like <<my title>>.",
     [lambda t: re.search(r"<<[^<>]+>>", t) is not None]),
    ("postscript", "Write a short email inviting a colleague to lunch. At the end of your response, add a postscript "
                   "starting with P.S.", [lambda t: re.search(r"P\.S\.", t) is not None]),
    ("paragraphs-3", "Write 3 paragraphs about the history of tea. Separate the paragraphs with the markdown divider ***",
     [lambda t: len([p for p in t.split("***") if p.strip()]) == 3]),
    ("placeholders", "Write a template for a job offer letter. Include at least 3 placeholders in square brackets, "
                     "such as [name].", [lambda t: len(re.findall(r"\[[^\[\]]+\]", t)) >= 3]),
    ("highlight", "Explain photosynthesis briefly and highlight at least 2 sections with markdown, i.e. *highlighted section*.",
     [lambda t: len(re.findall(r"\*[^\n\*]+\*", t)) >= 2]),
    ("quotation", "Wrap your entire response with double quotation marks. Tell me one fun fact about octopuses.",
     [lambda t: len(t.strip()) > 2 and t.strip()[0] == '"' and t.strip()[-1] == '"']),
    ("chinese", "请用中文回答：什么是人工智能？用两三句话说明。",
     [lambda t: len(re.findall(r"[一-鿿]", t)) >= 10, lambda t: len(re.findall(r"[A-Za-z]", t)) < 20]),
    ("french", "Answer only in French: what are two good habits for sleeping well?",
     [lambda t: any(w in t.lower() for w in (" le ", " la ", " les ", " des ", " de ")),
      lambda t: " the " not in t.lower()]),
    ("sentences-2", "In exactly two sentences, explain why the sky appears blue.",
     [lambda t: len(sentences(t)) == 2]),
    ("one-word", "What is the chemical symbol for gold? Answer with one word only.",
     [lambda t: len(words(t)) == 1, lambda t: "au" in t.lower()]),
    ("number-only", "How many legs does a spider have? Reply with just the number in digits.",
     [lambda t: re.fullmatch(r"\s*8\.?\s*", t) is not None]),
    ("repeat-request", "First repeat the request below word for word without change, then give your answer. "
                       "Request: Name one primary color.",
     [lambda t: t.strip().startswith("Name one primary color.")]),
    ("two-responses", "Give two different slogans for a bakery. Separate the two responses with six asterisks: ******",
     [lambda t: len([p for p in t.split("******") if p.strip()]) == 2]),
    ("keyword-freq", "Write a short paragraph about coffee in which the word 'aroma' appears at least 3 times.",
     [lambda t: len(re.findall(r"\baroma\b", t.lower())) >= 3]),
    ("letter-freq", "Write a sentence about zebras in which the letter 'z' appears at least 4 times.",
     [lambda t: t.lower().count("z") >= 4]),
    ("sections", "Write a short guide to brewing green tea with 2 sections. Mark the beginning of each section with "
                 "'SECTION X' (SECTION 1, SECTION 2).",
     [lambda t: "SECTION 1" in t and "SECTION 2" in t and "SECTION 3" not in t]),
    ("no-markdown-heading", "Explain recursion to a child in under 60 words, as plain text without any markdown.",
     [lambda t: "#" not in t and "**" not in t, lambda t: len(words(t)) < 60]),
    ("capital-words", "Write a sentence about space travel in which at least 3 words are written in ALL CAPS.",
     [lambda t: len([w for w in words(t) if len(w) > 1 and w.isupper()]) >= 3]),
    ("end-question", "Write one sentence about the moon and end your whole response with a question mark.",
     [lambda t: t.strip().endswith("?")]),
    ("yes-no-option", "Is Python a programming language? Your answer must be exactly one of: 'My answer is yes.', "
                      "'My answer is no.', 'My answer is maybe.'",
     [lambda t: t.strip() in ("My answer is yes.", "My answer is no.", "My answer is maybe.")]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sampling", default="greedy",
                    help="greedy (the default), server (send no sampling fields: the server's defaults), "
                         "or a JSON object of request fields")
    args = ap.parse_args()
    cl = Client(args.url)
    sampling = GREEDY if args.sampling == "greedy" else None if args.sampling == "server" else json.loads(args.sampling)
    n = k = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for cid, prompt, checks in CASES:
            res = cl.chat([{"role": "user", "content": prompt}], thinking=args.think, sampling=sampling,
                          max_tokens=args.max_tokens if not args.think else max(4096, args.max_tokens))
            text = res["content"]
            results = []
            for c in checks:
                try:
                    results.append(bool(c(text)))
                except Exception:
                    results.append(False)
            ok = res["status"] == 200 and all(results) and "</think>" not in text
            n += 1
            k += ok
            f.write(json.dumps({"id": cid, "correct": ok, "checks": results, "reply": text, "finish": res["finish_reason"],
                                "reasoning": res["reasoning"]}, ensure_ascii=False) + "\n")
            print(f"{cid:20s} {'OK ' if ok else 'BAD'} {results} {text[:90]!r}", flush=True)
    lo, hi = wilson(k, n)
    print(json.dumps({"n": n, "correct": k, "acc": k / n, "ci95": [lo, hi]}))


if __name__ == "__main__":
    main()
