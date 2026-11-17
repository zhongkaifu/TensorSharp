#!/usr/bin/env python3
"""Run N multi-turn conversations against a running TensorSharp.Server.Host at once
and check that concurrent serving keeps every conversation's state its own.

    parallel-multiturn-webui.py --url http://127.0.0.1:5100 --conversations 4 \
        --think on --max-tokens 1200 --out artifacts/parallel-multiturn/deepseek-tp6

Up to 16 conversations. With --greedy, a --sequential run is the control for a concurrent
one: pass its --out as --compare-with and the summary counts the turns whose answers differ
(informational: batched decoding agrees with solo decoding only up to near-ties).

Each conversation is its own Web UI session (POST /api/sessions) and posts its whole
history every turn, as index.html does (POST /api/chat, SSE; assistant turns carry
their `thinking`). Turns run in LOCKSTEP by default: every conversation sends turn k
together, so the engine has N requests in flight and has to batch them.

  turn 1  "Remember this secret code ...: <CODE_i>. Now introduce <TOPIC_i> ..."
  turn 2  "continue"
  turn 3  "What was the secret code ...? Reply with only the code."

A conversation FAILS when a turn errors or ends without a done frame, a reply
loops (a 1-60 character unit repeated back to back --min-repeats times) or its tail
loses almost all distinct 8-grams, or the last answer does not contain its own code
or contains another conversation's code (cross-sequence leakage or a corrupt reused
cache). Per-turn prompt and reused token counts come from the done frame, so the
report shows what KV reuse each turn actually got under concurrency.

`--stagger S` starts conversation i S*i seconds after the first and lets turns drift
(no lockstep): conversations then arrive while others sit between turns, which is when
a server that makes room for a new request by evicting a retained conversation shows
it (that conversation's next turn reuses nothing). Check with parallel-reuse-check.py.

`--plain` sends `skills: []`, `skills_discovery: false`, `multi_agent: false`, so
only the engine is exercised; without it the server's own defaults apply. Pair a
run with the server log: "Batched fused decode accepted N sequences in one graph"
(token-batched decode engaged) and the "Radix prompt reuse ..." admission lines.
Exit status 1 if any conversation fails.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

TOPICS = ["Final Fantasy VII", "The Legend of Zelda: Ocarina of Time", "Chrono Trigger", "Half-Life 2",
          "Super Mario 64", "Metal Gear Solid", "Portal", "StarCraft",
          "Tetris", "Minecraft", "Pac-Man", "The Elder Scrolls V: Skyrim",
          "Street Fighter II", "Civilization IV", "Doom (1993)", "Myst"]
CODES = ["ORCHID-742", "FALCON-318", "GLACIER-905", "COBALT-261", "MERIDIAN-477", "SAFFRON-683",
         "TUNDRA-529", "VORTEX-146", "LANTERN-833", "PRAIRIE-207", "QUARTZ-594", "HARBOR-368",
         "EMBER-715", "NEBULA-452", "CANYON-189", "SPRUCE-926"]


def post_json(url, body=None, timeout=60):
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def stream_chat(base, payload, timeout):
    req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    thinking, text, done, frames, first = [], [], None, 0, None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if not line.startswith("data: "):
                continue
            frames += 1
            data = json.loads(line[6:])
            if data.get("error"):
                raise RuntimeError(data["error"])
            if data.get("thinking"):
                thinking.append(data["thinking"])
                first = first or time.time()
            if data.get("token"):
                text.append(data["token"])
                first = first or time.time()
            if data.get("replace") is not None:
                text = [data["replace"]]
            if data.get("done"):
                done = data
    return "".join(thinking), "".join(text), done, frames, time.time() - t0, (first - t0) if first else None


def longest_loop(s, min_repeats):
    best = None
    for m in re.compile(r"(.{1,60}?)\1{%d,}" % (min_repeats - 1), re.S).finditer(s):
        unit = m.group(1)
        if not unit.strip():
            continue
        if best is None or len(m.group(0)) > best[2]:
            best = (unit, len(m.group(0)) // len(unit), len(m.group(0)), m.start())
    return best


def tail_diversity(s, n=8, tail=600):
    t = s[-tail:]
    grams = [t[i:i + n] for i in range(max(0, len(t) - n + 1))]
    return len(set(grams)) / max(1, len(grams)) if grams else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5100")
    ap.add_argument("--conversations", type=int, default=4)
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--plain", action="store_true", help="no skills, no discovery, no multi-agent")
    ap.add_argument("--system-file", help="a system message every conversation shares (cross-request reuse)")
    ap.add_argument("--sequential", action="store_true", help="one conversation at a time (the solo control)")
    ap.add_argument("--no-lockstep", action="store_true", help="let conversations drift instead of turn barriers")
    ap.add_argument("--stagger", type=float, default=0.0,
                    help="start conversation i this many seconds times i after the first (implies --no-lockstep)")
    ap.add_argument("--greedy", action="store_true",
                    help="send temperature 0, top-k 1, no repetition penalty (the server must let requests "
                         "set sampling: --sampling-precedence request), so --sequential runs are a control "
                         "for the concurrent ones; compare with --compare-with")
    ap.add_argument("--compare-with", help="a --sequential --greedy run's --out directory: count the turns whose "
                                           "answers differ from it (informational; batched decoding agrees with "
                                           "solo decoding up to near-ties)")
    ap.add_argument("--min-repeats", type=int, default=12)
    ap.add_argument("--min-tail-diversity", type=float, default=0.25)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = args.conversations
    if not 1 <= n <= len(TOPICS):
        ap.error(f"--conversations must be 1..{len(TOPICS)}")
    os.makedirs(args.out, exist_ok=True)
    system = open(args.system_file, encoding="utf-8").read() if args.system_file else None
    turns = [lambda i: (f"Remember this secret code for later: {CODES[i]}. Now please introduce "
                        f"{TOPICS[i]} in about 120 words."),
             lambda i: "continue",
             lambda i: "What was the secret code I asked you to remember? Reply with only the code."]
    if args.stagger < 0:
        ap.error("--stagger must not be negative")
    barrier = None if (args.sequential or args.no_lockstep or args.stagger > 0) else threading.Barrier(n)
    results = [None] * n
    lock = threading.Lock()

    def converse(i):
        if args.stagger > 0 and not args.sequential:
            time.sleep(args.stagger * i)
        rec = {"conversation": i, "topic": TOPICS[i], "code": CODES[i], "turns": [], "failures": []}
        history = [{"role": "system", "content": system}] if system else []
        session = None
        try:
            session = post_json(args.url + "/api/sessions").get("sessionId")
        except Exception as e:
            rec["failures"].append(f"session: {e}")
        for k, make in enumerate(turns):
            if barrier is not None:
                try:
                    barrier.wait(timeout=args.timeout)
                except threading.BrokenBarrierError:
                    rec["failures"].append(f"turn {k + 1}: barrier broken")
                    break
            history.append({"role": "user", "content": make(i)})
            payload = {"messages": history, "maxTokens": args.max_tokens, "think": args.think == "on",
                       "sessionId": session, "newChat": k == 0}
            if args.plain:
                payload.update({"skills": [], "skills_discovery": False, "multi_agent": False})
            if args.greedy:
                payload.update({"temperature": 0, "topK": 1, "topP": 1, "minP": 0, "repeatPenalty": 1})
            t = {"turn": k + 1, "startedAt": round(time.time(), 3)}
            try:
                thinking, text, done, frames, elapsed, ttft = stream_chat(args.url, payload, args.timeout)
            except Exception as e:
                t["error"] = f"{type(e).__name__}: {e}"
                rec["failures"].append(f"turn {k + 1}: {t['error']}")
                rec["turns"].append(t)
                if barrier is not None:
                    barrier.abort()
                break
            done = done or {}
            whole = thinking + text
            loop = longest_loop(whole, args.min_repeats)
            div = tail_diversity(whole)
            t.update({"promptTokens": done.get("promptTokens"), "kvReusedTokens": done.get("kvReusedTokens"),
                      "tokenCount": done.get("tokenCount"), "tokPerSec": done.get("tokPerSec"),
                      "truncated": done.get("truncated"), "elapsedSec": round(elapsed, 1),
                      "ttftSec": round(ttft, 2) if ttft else None, "frames": frames,
                      "thinkingChars": len(thinking), "answerChars": len(text), "tailDiversity": round(div, 3),
                      "loop": None if loop is None else {"unit": loop[0], "count": loop[1], "chars": loop[2]},
                      "hasDone": bool(done), "answer": text, "thinking": thinking})
            if not done:
                rec["failures"].append(f"turn {k + 1}: no done frame")
            if loop is not None or div < args.min_tail_diversity:
                rec["failures"].append(f"turn {k + 1}: degenerate repetition {t['loop']} diversity {div:.3f}")
            if not text.strip():
                rec["failures"].append(f"turn {k + 1}: empty answer")
            rec["turns"].append(t)
            message = {"role": "assistant", "content": text}
            if thinking:
                message["thinking"] = thinking
            history.append(message)
        last = rec["turns"][-1]["answer"] if rec["turns"] and "answer" in rec["turns"][-1] else ""
        if len(rec["turns"]) == len(turns):
            if CODES[i].casefold() not in last.casefold():
                rec["failures"].append(f"secret: answer {last[-120:]!r} does not contain {CODES[i]}")
            leaked = [c for j, c in enumerate(CODES[:n]) if j != i and c.casefold() in last.casefold()]
            if leaked:
                rec["failures"].append(f"secret: answer contains other conversations' codes {leaked}")
        rec["ok"] = not rec["failures"]
        with lock:
            results[i] = rec
            with open(os.path.join(args.out, f"conversation{i}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=1)
            brief = [{k: t.get(k) for k in ("turn", "promptTokens", "kvReusedTokens", "tokenCount", "elapsedSec", "error")}
                     for t in rec["turns"]]
            print(json.dumps({"conversation": i, "ok": rec["ok"], "turns": brief, "failures": rec["failures"]},
                             ensure_ascii=False), flush=True)

    t0 = time.time()
    if args.sequential:
        for i in range(n):
            converse(i)
    else:
        threads = [threading.Thread(target=converse, args=(i,)) for i in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    summary = {"configuration": {k: v for k, v in vars(args).items()}, "wallSec": round(time.time() - t0, 1),
               "conversations": n, "passed": sum(1 for r in results if r and r["ok"]),
               "perTurn": []}
    for k in range(len(turns)):
        rows = [r["turns"][k] for r in results if r and len(r["turns"]) > k and "promptTokens" in r["turns"][k]]
        summary["perTurn"].append({
            "turn": k + 1, "requests": len(rows),
            "promptTokens": [t["promptTokens"] for t in rows],
            "kvReusedTokens": [t["kvReusedTokens"] for t in rows],
            "tokenCount": [t["tokenCount"] for t in rows],
            "tokPerSec": [t["tokPerSec"] for t in rows]})
    if args.compare_with:
        same = differ = 0
        first_diff = []
        for i in range(n):
            try:
                with open(os.path.join(args.compare_with, f"conversation{i}.json"), encoding="utf-8") as f:
                    ref = json.load(f)
            except OSError:
                continue
            mine = results[i] or {"turns": []}
            for k, (a, b) in enumerate(zip(mine["turns"], ref["turns"])):
                if "answer" not in a or "answer" not in b:
                    continue
                if a["answer"] == b["answer"] and a.get("thinking") == b.get("thinking"):
                    same += 1
                else:
                    differ += 1
                    x, y = a.get("thinking", "") + a["answer"], b.get("thinking", "") + b["answer"]
                    at = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                    first_diff.append({"conversation": i, "turn": k + 1, "char": at,
                                       "this": x[max(0, at - 40):at + 40], "control": y[max(0, at - 40):at + 40]})
                    break   # later turns build on a different history
        summary["againstControl"] = {"sameTurns": same, "differentTurns": differ, "firstDifferences": first_diff}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "configuration"}, ensure_ascii=False))
    print(f"{summary['passed']}/{n} conversations passed")
    return 0 if summary["passed"] == n else 1


if __name__ == "__main__":
    sys.exit(main())
