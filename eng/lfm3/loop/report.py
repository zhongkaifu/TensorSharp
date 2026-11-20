#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Loop rate and answer quality per run of the repetition-study prompt set (eng/lfm3/loop/run.py output).

  python3 eng/lfm3/loop/report.py [--markdown] [--examples N] RUN.jsonl ...

Per file (its rows' `setting`) and language: turns, loops (loopdetect: a sentence repeated, an exactly periodic
tail, or the guard's finish reason "repetition"), cases with any looping turn, replies that hit max_tokens,
thinking-block loops, task checks passed (CHECKS below: fixed facts each answer must state, the reply's language,
and a non-empty answer), mean output tokens, and median decode tok/s.

The task checks are deliberately coarse (a fact or keyword per case, in either language): they tell a reply that
answers from one that drifts or loops, not a good answer from a better one.
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loopdetect  # noqa: E402

CJK = re.compile(r"[一-鿿]")

# case id -> list of alternatives groups; every group must match (any alternative in a group).
CHECKS = {
    "en-math": [[r"8:05|20:05|8\.05"]],
    "zh-math": [[r"8:05|20:05|8点05|8点5分|晚上8点|20点05|20点5分|八点零五|8時05"]],
    "en-translate": [[r"博物馆"], [r"退款|退票|退"]],
    "zh-translate": [[r"(?i)museum"], [r"(?i)refund"]],
    "en-multi2#0": [[r"(?i)canberra"]],
    "zh-multi2#0": [[r"堪培拉|(?i:canberra)"]],
    "en-img": [[r"(?i)zebra"]],
    "zh-img": [[r"斑马|(?i:zebra)"]],
    "en-img2": [[r"26\.46"]],
    "zh-img2": [[r"26\.46"]],
    "en-aud": [[r"(?i)snow"]],
    "zh-aud": [[r"雪|(?i:snow)"]],
    "en-aud2": [[r"(?i)canoe"], [r"(?i)planks"]],
    "zh-aud2": [[r"(?i)canoe|独木舟|皮划艇"]],
    "en-code": [[r"def "], [r"(?i)csv"]],
    "zh-code": [[r"def "], [r"(?i)csv"]],
    "en-code2": [[r"(?i)insert"], [r"(?i)search"]],
    "zh-code2": [[r"(?i)insert"], [r"(?i)search|find"]],
    "en-summary": [[r"(?i)britain|british"]],
    "zh-summary": [[r"英国"]],
    "en-compare": [[r"(?i)reliab|connection"]],
    "zh-compare": [[r"可靠|连接"]],
}


def check(row):
    """(passed, reason): the case's facts, its language, and a non-empty answer."""
    text = row.get("text") or ""
    if row.get("error"):
        return False, "error"
    if len(text.strip()) < 2:
        return False, "empty"
    key = f"{row['id']}#{row['turn']}"
    groups = CHECKS.get(key) or (CHECKS.get(row["id"]) if row["turn"] == 0 else None) or []
    for g in groups:
        if not any(re.search(p, text) for p in g):
            return False, f"missing {g[0]}"
    if row["lang"] == "zh" and row["cat"] not in ("translate", "code") and not CJK.search(text):
        return False, "not Chinese"
    if row["lang"] == "en" and row["cat"] != "translate":
        cjk = len(CJK.findall(text))
        if cjk > 0.2 * max(1, len(text)):
            return False, "not English"
    return True, ""


def summarize(rows):
    turns = len(rows)
    loops = sum(1 for r in rows if r.get("loop"))
    cases = {}
    for r in rows:
        cases.setdefault(r["id"], []).append(r)
    case_loops = sum(1 for rs in cases.values() if any(r.get("loop") for r in rs))
    think_loops = sum(1 for r in rows if r.get("thinking_loop"))
    hit_max = sum(1 for r in rows if r.get("finish_reason") == "length" or r.get("truncated"))
    guard = sum(1 for r in rows if r.get("finish_reason") == "repetition")
    ok = sum(1 for r in rows if check(r)[0])
    toks = [r["tokens"] for r in rows if isinstance(r.get("tokens"), (int, float))]
    tps = [r["tok_s"] for r in rows if isinstance(r.get("tok_s"), (int, float)) and r["tok_s"] > 0]
    return {
        "turns": turns, "loops": loops, "cases": len(cases), "case_loops": case_loops,
        "think_loops": think_loops, "hit_max": hit_max, "guard": guard, "quality": ok,
        "mean_tokens": round(statistics.mean(toks), 1) if toks else None,
        "tok_s": round(statistics.median(tps), 1) if tps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--examples", type=int, default=0, help="print the first N looping replies of each run")
    ap.add_argument("--by", default="lang", choices=["lang", "cat", "none"])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    table = []
    for path in a.files:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not rows:
            continue
        # The flags are recomputed, so runs recorded with an older detector are judged alike.
        for r in rows:
            det = loopdetect.analyze(r.get("text") or "", r.get("finish_reason"))
            r["loop"], r["why"] = det["loop"], det["why"]
            if r.get("thinking"):
                r["thinking_loop"] = loopdetect.analyze(r["thinking"])["loop"]
        setting = rows[0].get("setting") or path
        groups = {"all": rows}
        if a.by != "none":
            for r in rows:
                groups.setdefault(r.get(a.by, "?"), []).append(r)
        for g, rs in groups.items():
            table.append((setting, g, summarize(rs)))
        if a.examples:
            shown = 0
            for r in rows:
                if r.get("loop") and shown < a.examples:
                    shown += 1
                    print(f"--- {setting} {r['id']}#{r['turn']} {r.get('why')} tokens={r.get('tokens')}")
                    print((r.get("text") or "")[:700].replace("\n", "\\n"))
    hdr = ["setting", "group", "turns", "loops", "loop%", "cases looping", "think loops", "guard", "hit max",
           "quality", "mean tok", "tok/s"]
    lines = []
    for s, g, m in table:
        lines.append([s, g, m["turns"], m["loops"], f"{100 * m['loops'] / max(1, m['turns']):.0f}%",
                      f"{m['case_loops']}/{m['cases']}", m["think_loops"], m["guard"], m["hit_max"],
                      f"{m['quality']}/{m['turns']}", m["mean_tokens"], m["tok_s"]])
    if a.markdown:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for l in lines:
            print("| " + " | ".join(str(x) for x in l) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *lines)]
        for l in [hdr] + lines:
            print("  ".join(str(x).ljust(n) for x, n in zip(l, w)))


if __name__ == "__main__":
    sys.exit(main())
