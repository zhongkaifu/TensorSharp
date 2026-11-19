#!/usr/bin/env python3
"""Read eng/longturn.py runs: per request, where the time to first token went (issue #376).

  python3 eng/longturn-report.py RUN_DIR [RUN_DIR...]          # Markdown on stdout
  python3 eng/longturn-report.py RUN_DIR --tsv                  # one TSV row a request, for diffs and spreadsheets
  python3 eng/longturn-report.py RUN_DIR --json artifacts/longturn/report.json
  python3 eng/longturn-report.py --server-log S.log --client-log C.jsonl

A run directory holds server.log (the server's output with a UTC HH:MM:SS.mmm stamp on every line:
`... 2>&1 | python3 -u eng/longturn.py stamp > server.log`, the server run with TS_SCHED_STEP_LOG=1) and client.jsonl
from eng/longturn.py (or client.log in the older '<stamp> <size> <A|B|C> <promptTokens> <secs> s' form). Client and
server must share a clock (one machine): requests are paired by response id, then by time, then by order.

Per request it prints, from the server's lines:
  ttft      chat.complete's ttftMs (server side, from chat.start)
  reuse     kvReused of promptTokens
  tail      chat.complete after the first token (A: how long a 1-token request takes to finish)
  radix     the "Radix prompt reuse" line after chat.start
  admit     chat.start to the start of the request's first step (B: its admission gap)
  first     the first step: idle gap before it, its ms and rows (P1044: 1,044 prompt rows)
  prompt    prompt steps: count, ms summed, idle gaps summed
  decode    decode steps: count, median and p90 ms, tokens a step
  spec      the "Speculative decoding stats" line: verify steps, acceptance, and per verify step the draft, verify,
            commit and catch-up ms (the line's perVerifyMs section where it has one; on a runtime whose line has no
            commit phase, the shared runtime's verify still holds the draft)
  error     the request's failure (HTTP status, the exception and where it was thrown)
and the notes the server logged while the request ran ([cuda...], [prefix-...] and warning lines, plus --note REGEX).
A size's summary row puts A, B and C side by side: the issue's table. Nothing is written but stdout and --json.
Standard library only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import sys

STAMP = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d{3}) ?(.*)$")
STEP = re.compile(r"\[step\] t=(\d\d:\d\d:\d\d\.\d{3}) gap=([\d.]+) ms=([\d.]+) (\S+)(.*)$")
STEP_ENTRY = re.compile(r"(\S+):([PD])(\d+)")
KV = re.compile(r"(\w+)=([^\s|]+)")
DEFAULT_NOTES = r"^\[(cuda[\w-]*|prefix-[\w-]+|slot[\w-]*|kv[\w-]*|q35-cuda)\]|WARNING"


def tod(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_tod(text: str) -> float:
    h, m, rest = text.split(":")
    s, ms = rest.split(".")
    return tod(h, m, s, ms)


class Clock:
    """Times of day (seconds) made monotonic across midnight: a stamp more than 12 h behind the last one is a day later."""

    def __init__(self, day0_offset: float = 0.0):
        self.offset = day0_offset
        self.last = None

    def __call__(self, seconds: float) -> float:
        if self.last is not None and seconds + self.offset < self.last - 43200:
            self.offset += 86400
        t = seconds + self.offset
        self.last = t
        return t

    def near(self, seconds: float, anchor: float) -> float:
        """seconds (a time of day) placed on the day that puts it closest to anchor, without advancing the clock."""
        t = seconds + 86400 * round((anchor - seconds) / 86400)
        return t


# ----------------------------------------------------------------------------------------------------------------------
# client logs


def read_client(path: str) -> tuple[list[dict], float | None]:
    """Client requests (size, round, step, sent, done in seconds since midnight UTC of the first request's day)."""
    reqs = []
    if path.endswith(".jsonl"):
        day0 = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sent = datetime.datetime.strptime(r["sent"], "%Y-%m-%dT%H:%M:%S.%fZ")
                done = datetime.datetime.strptime(r["done"], "%Y-%m-%dT%H:%M:%S.%fZ")
                if day0 is None:
                    day0 = datetime.datetime(sent.year, sent.month, sent.day)
                r["t_sent"] = (sent - day0).total_seconds()
                r["t_done"] = (done - day0).total_seconds()
                reqs.append(r)
        return reqs, (reqs[0]["t_sent"] if reqs else None)
    clock = Clock()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"^(\d\d:\d\d:\d\d\.\d{3}) (\d+) ([ABC]) (\d+) ([\d.]+) s", line)
            if m:
                done = clock(parse_tod(m.group(1)))
                wall = float(m.group(5))
                reqs.append({"size": int(m.group(2)), "round": None, "step": m.group(3), "prompt_tokens": int(m.group(4)),
                             "wall_s": wall, "t_done": done, "t_sent": done - wall, "status": 200, "id": None, "error": None})
                continue
            # The issue's script dies on a failed request with urllib's traceback: the request after the last one logged.
            m = re.match(r"^(\d\d:\d\d:\d\d\.\d{3}) urllib\.error\.HTTPError: HTTP Error (\d+): ?(.*)$", line.rstrip())
            if m and reqs:
                last = reqs[-1]
                # After a C it is the next size's A, whose size the log does not say.
                size, step = (last["size"], {"A": "B", "B": "C"}[last["step"]]) if last["step"] != "C" else (None, "A")
                reqs.append({"size": size, "round": None, "step": step, "prompt_tokens": None, "wall_s": None,
                             "t_done": clock(parse_tod(m.group(1))), "t_sent": None, "status": int(m.group(2)),
                             "id": None, "error": f"HTTP {m.group(2)}: {m.group(3)}"})
    # Rounds: a size's A/B/C sequences in order.
    seen = {}
    for r in reqs:
        if r["step"] == "A":
            seen[r["size"]] = seen.get(r["size"], -1) + 1
        r["round"] = max(0, seen.get(r["size"], 0))
    return reqs, (reqs[0]["t_sent"] if reqs else None)


# ----------------------------------------------------------------------------------------------------------------------
# server logs


def new_request(t: float, warm: bool, max_tokens: int | None) -> dict:
    return {"t_start": t, "warm": warm, "max_tokens": max_tokens, "response_id": None, "rid": None, "t_radix": None,
            "radix": None, "steps": [], "spec": None, "spec_line": None, "complete": None, "t_complete": None,
            "error": None, "error_site": None, "http_error": None, "notes": []}


def read_server(path: str, anchor: float | None, note_patterns: list[str]) -> dict:
    """Requests (chat.start .. chat.complete), steps and notes of a stamped server log."""
    notes_re = re.compile("|".join(f"(?:{p})" for p in note_patterns)) if note_patterns else None
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()
    first = next((STAMP.match(l) for l in raw if STAMP.match(l)), None)
    offset = 0.0
    if first is not None and anchor is not None and tod(*first.group(1, 2, 3, 4)) - anchor > 43200:
        offset = -86400.0   # the server started the day before the first request
    clock = Clock(offset)
    reqs, steps, pending_id, engine = [], [], None, False
    cur = None                 # the client request being served (warm-ups are skipped)
    t = None
    in_fail = None             # the request a fail: entry belongs to, while its exception lines follow
    fail_frames: list[str] = []
    for line in raw:
        m = STAMP.match(line)
        if m:
            t = clock(tod(*m.group(1, 2, 3, 4)))
            s = m.group(5)
        else:
            s = line
        if t is None:
            continue
        body = s.strip()
        if "on the direct-CUDA engine" in body:
            engine = True
        if in_fail is not None:
            # A fail: entry is its message, the exception's type and message, then its frames.
            fm = re.match(r"^at (\S+)", body)
            em = re.match(r"^([\w.]+(?:Exception|Error)): (.*)$", body)
            if em:
                if in_fail["error"] is None:
                    in_fail["error"] = f"{em.group(1).split('.')[-1]}: {em.group(2)}"[:200]
                continue
            if fm:
                fail_frames.append(fm.group(1))
                continue
            if body.startswith("---"):
                continue
            # The entry ends with its frames, or at a line that is not part of a logger entry; its message lines are
            # read below like any other line ("HTTP POST ... threw" is one).
            if fail_frames or not s.startswith(" "):
                if fail_frames and in_fail["error_site"] is None:
                    in_fail["error_site"] = frames_site(fail_frames)
                in_fail, fail_frames = None, []
        if body.startswith("fail:"):
            in_fail = cur if cur is not None else (reqs[-1] if reqs else None)
            fail_frames = []
            continue
        if "/v1/chat/completions request: id=" in body:
            pending_id = re.search(r"request: id=(\S+)", body).group(1)
            continue
        if "chat.start arch=" in body:
            warm = "origin=prefix-cache-warmup" in body
            mt = re.search(r"maxTokens=(\d+)", body)
            r = new_request(t, warm, int(mt.group(1)) if mt else None)
            if not warm:
                r["response_id"], pending_id = pending_id, None
                reqs.append(r)
                cur = r
            continue
        if "chat.complete tokens=" in body or "chat.cancelled" in body:
            if "origin=prefix-cache-warmup" in body or cur is None:
                continue
            cur["complete"] = dict(KV.findall(body.split(" assistantOutput=")[0]))
            if "chat.cancelled" in body:
                cur["complete"]["cancelled"] = "1"
            cur["t_complete"] = t
            cur = None
            continue
        if "HTTP POST /v1/chat/completions threw" in body:
            target = cur if cur is not None else (reqs[-1] if reqs else None)
            if target is not None:
                target["http_error"] = body[: 200]
                if target["t_complete"] is None:
                    target["t_complete"] = t
                if target is cur:
                    cur = None
            continue
        sm = STEP.search(body)
        if sm:
            end = clock.near(parse_tod(sm.group(1)), t)
            ms = float(sm.group(3))
            entries = [(e.group(1), e.group(2), int(e.group(3))) for e in STEP_ENTRY.finditer(sm.group(5))]
            steps.append({"t_end": end, "t_begin": end - ms / 1000.0, "gap": float(sm.group(2)), "ms": ms,
                          "path": sm.group(4), "entries": entries})
            continue
        if cur is None:
            continue
        rm = re.search(r"requestId=(chat-[\w-]+)", body)
        if rm and cur["rid"] is None:
            cur["rid"] = rm.group(1)
        rm = re.search(r"Radix prompt reuse for (chat-[\w-]+): (.*)$", body)
        if rm:
            cur["rid"] = cur["rid"] or rm.group(1)
            if cur["t_radix"] is None:
                cur["t_radix"], cur["radix"] = t, rm.group(2)[:160]
            continue
        sm = re.search(r"Speculative decoding stats for (chat-[\w-]+): (.*)$", body)
        if sm:
            cur["spec_line"] = sm.group(2)
            cur["spec"] = parse_spec(sm.group(2))
            continue
        if notes_re is not None and notes_re.search(body):
            cur["notes"].append((t, body[:240]))
    if in_fail is not None and fail_frames and in_fail["error_site"] is None:
        in_fail["error_site"] = frames_site(fail_frames)
    attribute_steps(reqs, steps)
    return {"requests": reqs, "steps": steps, "engine": engine}


def frames_site(frames: list[str]) -> str:
    """Where an exception was thrown: up to six of its model and runtime frames, outermost first, by method name, and the
    innermost frame of all."""
    def name(fr):
        return re.sub(r"\(.*$", "", fr).split(".")[-1]
    ours = []
    for fr in frames:
        if fr.startswith(("TensorSharp.Models.", "TensorSharp.Runtime.")) and name(fr) not in ours:
            ours.append(name(fr))
    site = " -> ".join(reversed(ours[:6]))
    inner = next((name(fr) for fr in frames if name(fr) not in ("ThrowOnError", "ThrowIfFailed")), "")
    if inner and inner not in ours:
        site = f"{site} (in {inner})" if site else inner
    return site


def parse_spec(text: str) -> dict:
    """The sections of a 'Speculative decoding stats' line: totals, phaseMs, perVerifyMs, governor."""
    sections = [p.strip() for p in text.split("|")]
    out = {"totals": dict(KV.findall(sections[0]))}
    for sec in sections[1:]:
        head = sec.split(" ", 1)[0]
        out[head] = {k: v.rstrip("%") for k, v in KV.findall(sec)}
    acc = re.search(r"acceptance=([\d.]+) ?%", sections[0])
    if acc:
        out["totals"]["acceptance"] = acc.group(1)
    return out


def attribute_steps(reqs: list[dict], steps: list[dict]) -> None:
    """A step belongs to a request when it carries the request's id suffix (the step log prints the last four
    characters of ids longer than nine); a request whose id never appeared takes the steps inside its window."""
    for r in reqs:
        rid = r["rid"]
        suffix = (rid[-4:] if len(rid) > 9 else rid) if rid else None
        end = r["t_complete"]
        for st in steps:
            if st["t_end"] < r["t_start"] - 0.001:
                continue
            if end is not None and st["t_begin"] > end + 0.001:
                break
            mine = [e for e in st["entries"] if suffix is not None and e[0] == suffix]
            if suffix is None and end is not None:
                mine = st["entries"]
            if mine:
                r["steps"].append({**st, "mine": mine})


# ----------------------------------------------------------------------------------------------------------------------
# pairing and metrics


def pair(client: list[dict], server: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """(client, server) per request: by response id, then by time (chat.start inside the request's send..done), then
    in order."""
    used, out = set(), []
    by_id = {s["response_id"]: i for i, s in enumerate(server) if s["response_id"]}
    for c in client:
        idx = by_id.get(c.get("id")) if c.get("id") else None
        if idx is None:
            for i, s in enumerate(server):
                if i in used:
                    continue
                if c.get("t_sent") is not None and c["t_sent"] - 1.0 <= s["t_start"] <= c["t_done"] + 1.0:
                    idx = i
                    break
        if idx is None:
            idx = next((i for i in range(len(server)) if i not in used and not any(j > i for j in used)), None)
        if idx is not None and idx not in used:
            used.add(idx)
            out.append((c, server[idx]))
        else:
            out.append((c, None))
    for i, s in enumerate(server):
        if i not in used:
            out.append((None, s))
    return out


def fnum(v, nd=0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if nd else round(x)


def metrics(c: dict | None, s: dict | None) -> dict:
    m = {"size": c.get("size") if c else None, "round": c.get("round") if c else None, "step": c["step"] if c else "?",
         "client_s": c.get("wall_s") if c else None, "status": c.get("status") if c else None}
    err = c.get("error") if c else None
    if s is None:
        m["error"] = err or "no chat.start on the server"
        return m
    k = s["complete"] or {}
    # A request that never completed: its prompt and reuse from the radix line ("47013/48057 tokens ...").
    rad = re.match(r"(\d+)/(\d+) tokens", s["radix"] or "")
    m.update(prompt=fnum(k.get("promptTokens")) or (c or {}).get("prompt_tokens") or (int(rad.group(2)) if rad else None),
             reused=fnum(k.get("kvReused")) if k else (int(rad.group(1)) if rad else None),
             tokens=fnum(k.get("tokens")), ttft_ms=fnum(k.get("ttftMs")), elapsed_ms=fnum(k.get("elapsedMs")),
             tok_s=fnum(k.get("tokensPerSec"), 1), finish=k.get("finishReason"))
    if m["ttft_ms"] is not None and s["t_complete"] is not None:
        m["tail_ms"] = round((s["t_complete"] - (s["t_start"] + m["ttft_ms"] / 1000.0)) * 1000)
    if s["t_radix"] is not None:
        m["radix_ms"] = round((s["t_radix"] - s["t_start"]) * 1000)
    steps = s["steps"]
    if steps:
        f = steps[0]
        m["admit_ms"] = round((f["t_begin"] - s["t_start"]) * 1000)
        m["first_gap_ms"], m["first_ms"] = round(f["gap"], 1), round(f["ms"], 1)
        m["first_rows"] = " ".join(f"{kind}{n}" for _, kind, n in f["mine"])
        pre = [st for st in steps if any(kind == "P" for _, kind, _ in st["mine"])]
        dec = [st for st in steps if all(kind == "D" for _, kind, _ in st["mine"])]
        if pre:
            m["prompt_steps"] = len(pre)
            m["prompt_ms"] = round(sum(st["ms"] for st in pre))
            m["prompt_gap_ms"] = round(sum(st["gap"] for st in pre))
            m["prompt_max_gap_ms"] = round(max(st["gap"] for st in pre), 1)
        if dec:
            ms = sorted(st["ms"] for st in dec)
            rows = sum(n for st in dec for _, _, n in st["mine"])
            m["decode_steps"] = len(dec)
            m["decode_ms_med"] = round(statistics.median(ms), 1)
            m["decode_ms_p90"] = round(ms[min(len(ms) - 1, int(len(ms) * 0.9))], 1)
            m["decode_gap_ms"] = round(sum(st["gap"] for st in dec))
            m["tokens_per_decode_step"] = round(rows / len(dec), 2)
    if m.get("prompt") is not None and m.get("reused") is not None and m.get("ttft_ms"):
        fresh = m["prompt"] - m["reused"]
        if fresh > 1000:
            m["prefill_tok_s"] = round(fresh / (m["ttft_ms"] / 1000.0))
    sp = s["spec"]
    if sp:
        tot, ph = sp["totals"], sp.get("phaseMs", {})
        n = fnum(tot.get("verifySteps")) or 0
        m["verify_steps"] = n
        m["acceptance"] = tot.get("acceptance")
        per = sp.get("perVerifyMs", {})
        if n:
            for key in ("draft", "verify", "commit", "catchUp"):
                if key in per:
                    m[f"{key}_ms"] = fnum(per[key], 1)
                elif key in ph:
                    m[f"{key}_ms"] = round(float(ph[key]) / n, 1)
    if err or s["error"] or s["http_error"]:
        parts = [p for p in (err, s["error"], s["error_site"]) if p]
        m["error"] = "; ".join(parts) or s["http_error"]
    m["notes"] = [(round((t - s["t_start"]) * 1000), text) for t, text in s["notes"]]
    return m


# ----------------------------------------------------------------------------------------------------------------------
# output


COLUMNS = ["size", "round", "step", "prompt", "reused", "tokens", "ttft_ms", "tail_ms", "radix_ms", "admit_ms",
           "first_gap_ms", "first_ms", "first_rows", "prompt_steps", "prompt_ms", "prompt_gap_ms", "prefill_tok_s",
           "decode_steps", "decode_ms_med", "decode_ms_p90", "tokens_per_decode_step", "tok_s", "verify_steps",
           "acceptance", "draft_ms", "verify_ms", "commit_ms", "catchUp_ms", "client_s", "error"]


def cell(v) -> str:
    return "" if v is None else str(v)


def load_run(server_log: str, client_log: str | None, notes: list[str]) -> dict:
    client, anchor = read_client(client_log) if client_log and os.path.exists(client_log) else ([], None)
    server = read_server(server_log, anchor, notes)
    rows = [metrics(c, s) for c, s in pair(client, server["requests"])]
    return {"rows": rows, "engine": server["engine"], "requests": len(server["requests"]), "client": len(client)}


def run_header(run_dir: str | None) -> list[str]:
    out = []
    for name in ("run.json", "env.txt"):
        p = os.path.join(run_dir, name) if run_dir else None
        if p and os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if name == "run.json":
                try:
                    meta = json.loads(text)
                    text = " ".join(f"{k}={meta[k]}" for k in ("model", "sizes", "rounds", "sampling", "corpus_rev",
                                                               "corpus_sha256") if k in meta)
                except ValueError:
                    pass
            out.extend(f"    {l}" for l in text.splitlines()[:6])
    return out


def markdown(name: str, run: dict, header: list[str]) -> str:
    rows = run["rows"]
    lines = [f"## {name}", ""]
    if header:
        lines += header + [""]
    lines.append(f"{run['client']} client requests, {run['requests']} on the server"
                 + (" (direct-CUDA engine: verify excludes the draft)" if run["engine"] else ""))
    lines.append("")
    # The issue's table: one row a size and round.
    groups = {}
    for r in rows:
        if r["size"] is not None:
            groups.setdefault((r["size"], r["round"]), {})[r["step"]] = r
    if groups:
        lines += ["| size r | A ttft s | A tail ms | B ttft s | B admit ms | B first step ms | C ttft s | C reuse | C tok/s "
                  "| verify / draft / commit ms a verify step | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (size, rnd), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            a, b, c = g.get("A", {}), g.get("B", {}), g.get("C", {})
            def s(x): return "" if x.get("ttft_ms") is None else f"{x['ttft_ms'] / 1000:.2f}"
            spec = " / ".join(cell(c.get(k)) for k in ("verify_ms", "draft_ms", "commit_ms")) if c.get("verify_ms") else ""
            errs = "; ".join(f"{k}: {x['error']}" for k, x in (("A", a), ("B", b), ("C", c)) if x.get("error"))
            missing = [k for k, x in (("A", a), ("B", b), ("C", c)) if not x]
            if missing and not errs:
                errs = "not reached: " + ",".join(missing)
            elif missing:
                errs += "; not reached: " + ",".join(missing)
            lines.append(f"| {size} r{rnd} | {s(a)} | {cell(a.get('tail_ms'))} | {s(b)} | {cell(b.get('admit_ms'))} "
                         f"| {cell(b.get('first_ms'))} | {s(c)} | {cell(c.get('reused'))} | {cell(c.get('tok_s'))} "
                         f"| {spec} | {errs} |")
        lines.append("")
    outside = [r for r in rows if r["size"] is None]
    if outside:
        lines += [f"Not in the table (size unknown): {len(outside)} request(s), "
                  + "; ".join(f"{r['step']} prompt {cell(r.get('prompt')) or '?'}: {r.get('error') or 'no error'}"
                              for r in outside), ""]
    keep = [c for c in COLUMNS if c != "error" and any(r.get(c) is not None for r in rows)]
    lines += ["| " + " | ".join(keep) + " |", "|" + "---|" * len(keep)]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c)) for c in keep) + " |")
    for r in rows:
        extra = []
        if r.get("error"):
            extra.append(f"error: {r['error']}")
        extra += [f"+{dt} ms {text}" for dt, text in r.get("notes", [])[:12]]
        if extra:
            lines.append(f"\n{cell(r['size']) or '?'} r{cell(r['round']) or '?'} {r['step']} "
                         f"(prompt {cell(r.get('prompt')) or '?'}):")
            lines += [f"    {x}" for x in extra]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (server.log and client.jsonl or client.log)")
    ap.add_argument("--server-log", help="a stamped server log (instead of a run directory)")
    ap.add_argument("--client-log", help="its client.jsonl or client.log")
    ap.add_argument("--tsv", action="store_true", help="one TSV row a request instead of Markdown")
    ap.add_argument("--json", help="also write the rows as JSON to this file (keep it under the ignored artifacts/)")
    ap.add_argument("--note", action="append", default=[],
                    help="a regex of more server lines to list under the request they appeared in (repeatable)")
    args = ap.parse_args(argv)
    targets = []
    for d in args.runs:
        client = next((os.path.join(d, n) for n in ("client.jsonl", "client.log") if os.path.exists(os.path.join(d, n))), None)
        targets.append((d.rstrip("/"), os.path.join(d, "server.log"), client, d))
    if args.server_log:
        targets.append((args.server_log, args.server_log, args.client_log, None))
    if not targets:
        ap.error("give run directories or --server-log")
    notes = [DEFAULT_NOTES] + args.note
    results = {}
    for name, server_log, client_log, run_dir in targets:
        run = load_run(server_log, client_log, notes)
        results[name] = run
        if args.tsv:
            if len(results) == 1:
                print("\t".join(["run"] + COLUMNS))
            for r in run["rows"]:
                print("\t".join([os.path.basename(name)] + [cell(r.get(c)) for c in COLUMNS]))
        else:
            print(markdown(name, run, run_header(run_dir)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
