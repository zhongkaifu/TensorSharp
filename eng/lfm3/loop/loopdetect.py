#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Mechanical repetition-loop detector for chat replies (standard library only).

A reply "loops" when the model writes the same sentence (or line) over and over, or when its
text ends in an exactly periodic run. Three signals, each reported:

- units: the reply split into sentences and lines (Latin and CJK punctuation), each normalized
  (case, whitespace, markdown bullets and numbering, punctuation dropped). A unit of at least
  MIN_UNIT_CHARS normalized characters that occurs LOOP_UNIT_COUNT times or more, or LOOP_RUN
  times in a row, is a loop. With digits folded to '#' as well, a counting loop ("Step 4 ...",
  "Step 5 ..." with the same words) is caught at FOLDED_FACTOR times those counts: a short
  schedule ("Week 1: 10 minutes", "Week 2: 15 minutes", ...) is a list, not a loop. Code blocks
  are left out of this test, because a code line may legitimately repeat.
- tail: the reply's last characters (whitespace collapsed) are exactly periodic with a period of
  2-400 characters, repeated at least TAIL_REPEATS times and covering at least TAIL_MIN_CHARS
  characters (with digits folded: FOLDED_FACTOR times the repeats). This catches loops inside code
  and loops with no sentence breaks.
- finish: the server ended the reply with finish reason "repetition" (TensorSharp's guard).

`analyze(text, finish_reason)` returns a dict with `loop` (bool), `why` (list), the largest unit
count and run, the tail period/repeats, and `dup_frac`: the share of the reply's characters that
sit in a repeated unit's second and later copies (a graded measure, not used for the flag).

Run as a script on a JSONL of results ({"text": ..., "finish_reason": ...} per line) to print
the flagged rows, or `--selftest` to check the detector on fixed examples.
"""
import json
import re
import sys
import unicodedata

MIN_UNIT_CHARS = 12      # normalized characters (a CJK character counts 2)
LOOP_UNIT_COUNT = 4      # the same unit this many times anywhere in the reply
LOOP_RUN = 3             # or this many times in a row
TAIL_REPEATS = 4
TAIL_MIN_CHARS = 120
TAIL_MAX_PERIOD = 400
FOLDED_FACTOR = 2        # digit-folded matches need twice the repeats

_FENCE = re.compile(r"```.*?(?:```|$)", re.S)
_SPLIT = re.compile(r"(?<=[.!?;])\s+|[。！？；\n]+")
_BULLET = re.compile(r"^\s*(?:[-*+•·>#]+|\(?\d+[.)、:]|[a-zA-Z][.)])\s*")
_DIGITS = re.compile(r"\d+(?:[.,:]\d+)*")
_PUNCT_WS = re.compile(r"[\s\W_]+", re.U)


def _width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def normalize_unit(s, fold_digits=False):
    s = _BULLET.sub("", s.strip())
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = s.lower()
    if fold_digits:
        s = _DIGITS.sub("#", s)
    s = _PUNCT_WS.sub(" ", s).strip()
    return s


def units(text, fold_digits=False):
    prose = _FENCE.sub("\n", text)
    out = []
    for part in _SPLIT.split(prose):
        if not part:
            continue
        n = normalize_unit(part, fold_digits)
        if _width(n.replace(" ", "").replace("#", "")) >= MIN_UNIT_CHARS:
            out.append((n, len(part)))
    return out


def tail_period(text, fold_digits=False):
    """(period, repeats) of the longest exactly periodic tail, digits folded; (0, 0) if none.

    For each period p the tail is periodic over the longest suffix where every character equals
    the one p before it, so a run cut mid-copy (max_tokens, the guard) still counts."""
    t = re.sub(r"\s+", " ", _DIGITS.sub("#", text) if fold_digits else text).rstrip()
    n = len(t)
    if n < TAIL_MIN_CHARS:
        return 0, 0
    best = (0, 0)
    for p in range(2, min(TAIL_MAX_PERIOD, n // TAIL_REPEATS) + 1):
        i = n - 1
        while i >= p and t[i] == t[i - p]:
            i -= 1
        span = (n - 1 - i) + p          # characters covered by the periodic run
        k = span // p
        if k >= TAIL_REPEATS and span >= TAIL_MIN_CHARS and span > best[0] * best[1]:
            best = (p, k)
    return best


def _unit_stats(us):
    counts = {}
    for n, _ in us:
        counts[n] = counts.get(n, 0) + 1
    max_count, max_unit = 0, ""
    for n, c in counts.items():
        if c > max_count:
            max_count, max_unit = c, n
    run, max_run, run_unit = 1, 1 if us else 0, us[0][0] if us else ""
    for i in range(1, len(us)):
        run = run + 1 if us[i][0] == us[i - 1][0] else 1
        if run > max_run:
            max_run, run_unit = run, us[i][0]
    return max_count, max_unit, max_run, run_unit


def analyze(text, finish_reason=None):
    text = text or ""
    us = units(text)
    max_count, max_unit, max_run, run_unit = _unit_stats(us)
    fmax_count, fmax_unit, fmax_run, frun_unit = _unit_stats(units(text, fold_digits=True))
    seen, dup_chars = set(), 0
    for n, length in us:
        if n in seen:
            dup_chars += length
        seen.add(n)
    period, repeats = tail_period(text)
    fperiod, frepeats = tail_period(text, fold_digits=True)
    why = []
    if max_count >= LOOP_UNIT_COUNT:
        why.append(f"unit x{max_count}: {max_unit[:60]}")
    elif fmax_count >= FOLDED_FACTOR * LOOP_UNIT_COUNT:
        why.append(f"unit x{fmax_count} (digits folded): {fmax_unit[:60]}")
    if max_run >= LOOP_RUN:
        why.append(f"run x{max_run}: {run_unit[:60]}")
    elif fmax_run >= FOLDED_FACTOR * LOOP_RUN:
        why.append(f"run x{fmax_run} (digits folded): {frun_unit[:60]}")
    if period:
        why.append(f"tail period {period} x{repeats}")
    elif fperiod and frepeats >= FOLDED_FACTOR * TAIL_REPEATS:
        why.append(f"tail period {fperiod} x{frepeats} (digits folded)")
    if finish_reason == "repetition":
        why.append("finish repetition")
    return {
        "loop": bool(why),
        "why": why,
        "max_unit_count": max(max_count, fmax_count // FOLDED_FACTOR),
        "max_run": max(max_run, fmax_run // FOLDED_FACTOR),
        "tail_period": period or fperiod,
        "tail_repeats": repeats or frepeats,
        "dup_frac": round(dup_chars / max(1, len(text)), 4),
        "chars": len(text),
    }


def _selftest():
    loop_en = "Here is the answer. " + "The quick brown fox jumps over the lazy dog. " * 6
    loop_zh = "答案如下。" + "这是一个非常重要的问题，我们需要认真对待。" * 5
    count = "Let me compute. " + " ".join(f"3+8=11? Let me check again." for _ in range(5))
    tail = "x = [" + "1, 2, " * 80
    ok_list = "\n".join(f"{i}. Tip number {i}: {w} every day." for i, w in enumerate(
        ["exercise", "read", "sleep well", "drink water", "plan ahead", "take breaks"], 1))
    ok_plan = "\n".join(f"**Day {d}:**\n- Morning: visit {p}\n- Evening: dinner at {r}" for d, p, r in [
        (1, "Senso-ji temple", "an izakaya"), (2, "Meiji Shrine", "a sushi bar"), (3, "the Tsukiji market", "a ramen shop")])
    ok_code = "```python\nfor i in range(3):\n    print('hello world from the loop body')\n    print('hello world from the loop body')\n    print('hello world from the loop body')\n    print('hello world from the loop body')\n```\nDone."
    schedule = "Plan:\n" + "\n".join(f"- Week {w}: {m} minutes, 3x/week" for w, m in [(1, 10), (2, 15), (3, 20), (4, 25)])
    counting = "Let me count. " + " ".join(f"Step {i}: add one more apple to the basket." for i in range(1, 12))
    cases = [(loop_en, True), (loop_zh, True), (count, True), (tail, True), (schedule, False), (counting, True),
             (ok_list, False), (ok_plan, False), (ok_code, False), ("Hello! How can I help?", False)]
    bad = 0
    for text, want in cases:
        got = analyze(text)
        if got["loop"] != want:
            bad += 1
            print("FAIL", want, got, repr(text[:80]))
    print("selftest", "ok" if not bad else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    for path in sys.argv[1:]:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            r = analyze(row.get("text", ""), row.get("finish_reason"))
            if r["loop"]:
                print(row.get("id"), r["why"])
