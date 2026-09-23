#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""End-to-end validation and benchmark for TensorSharp sub-agents.

Drives a RUNNING TensorSharp.Server (started with --code-exec --sub-agents) through
/v1/chat/completions, one request at a time, and reads the server's own log for what
happened inside each request: which agents were spawned and how they finished, how
their results were delivered, and how much of every prompt the KV cache reused.

Each scenario can run in two modes over the SAME server (so the tool block, the
system prompt and the prefix cache are identical):

  parallel  the prompt asks for sub-agents, one per task
  solo      the prompt asks the model to do the same tasks itself, one after another

Comparing the two is the benchmark: same tasks, same model, same tool surface; the only
difference is whether the work runs as concurrent sequences on the engine.

Usage (server already listening):
  python3 eng/validation/sub-agents-e2e.py --base-url http://127.0.0.1:5101 \
      --server-log path/to/server.log --model-label gemma-4-E2B \
      --scenarios compute3,write4,files2,guard,fork --modes parallel,solo --repeat 3 \
      --out artifacts/subagents/e2e/results-e2b.json

Generated reports belong in ignored directories (artifacts/ or docs/validation/).
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

TASKS_COMPUTE3 = [
    ("the sum of the integers from 1 to 1000", "500500"),
    ("the number of prime numbers below 10000", "1229"),
    ("the 50th Fibonacci number, with F(1)=1 and F(2)=1", "12586269025"),
]

TOPICS_WRITE4 = ["photosynthesis", "plate tectonics", "the water cycle", "the human immune system"]


def scenario_prompts(name, mode):
    """(prompt, verifier) for one scenario and mode. The verifier takes the answer text
    and the parsed log facts and returns a list of failed checks (empty = pass)."""
    if name == "compute3":
        tasks = "\n".join(f"{i + 1}. Compute {t} with a python3 program." for i, (t, _) in enumerate(TASKS_COMPUTE3))
        if mode == "parallel":
            prompt = ("Use three sub-agents working in parallel, one per task below. Wait for all three, then "
                      "reply with the three numbers, one per line.\n" + tasks)
        else:
            prompt = ("Do the three tasks below yourself, one after another. Do not start sub-agents. Then "
                      "reply with the three numbers, one per line.\n" + tasks)

        def verify(answer, facts):
            return [f"missing {v}" for _, v in TASKS_COMPUTE3 if v not in answer.replace(",", "")]
        return prompt, verify

    if name == "write4":
        topics = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(TOPICS_WRITE4))
        if mode == "parallel":
            prompt = ("I need four short factual paragraphs of about 120 words each, one per topic below. Use "
                      "four sub-agents in parallel, one per topic, each writing its paragraph. Then reply with "
                      "the four paragraphs, each under its topic as a heading.\n" + topics)
        else:
            prompt = ("I need four short factual paragraphs of about 120 words each, one per topic below. Write "
                      "them yourself, without starting sub-agents, each under its topic as a heading.\n" + topics)

        def verify(answer, facts):
            low = answer.lower()
            failed = [f"missing topic '{t}'" for t in ("photosynth", "tecton", "water", "immun") if t not in low]
            words = len(answer.split())
            if words < 300:
                failed.append(f"only {words} words")
            return failed
        return prompt, verify

    if name == "files2":
        if mode == "parallel":
            prompt = ("Please delegate to two sub-agents in parallel. Sub-agent A: create a file named a.txt "
                      "containing exactly the word alpha. Sub-agent B: create a file named b.txt containing "
                      "exactly the word beta. When both are done, read both files yourself and tell me what "
                      "each one contains.")
        else:
            prompt = ("Create a file named a.txt containing exactly the word alpha, and a file named b.txt "
                      "containing exactly the word beta. Do not start sub-agents. Then read both files and "
                      "tell me what each one contains.")

        def verify(answer, facts):
            low = answer.lower()
            failed = [f"missing {w}" for w in ("alpha", "beta") if w not in low]
            if re.search(r"(not|n't) (been )?(created|found|exist)", low):
                failed.append("answer reports a missing file")
            return failed
        return prompt, verify

    if name == "guard":
        # The model is told NOT to wait: the host's end-of-turn guard has to collect the
        # results itself and give the model a round to use them.
        prompt = ("Start two sub-agents: one computes 2**20 with a python3 program, the other computes 3**13 "
                  "with a python3 program. Do not call wait_agent. After starting them, reply to me with the "
                  "results you have.")

        def verify(answer, facts):
            failed = [f"missing {v}" for v in ("1048576", "1594323") if v not in answer.replace(",", "")]
            if facts["spawned"] < 2:
                failed.append(f"spawned {facts['spawned']} agents")
            return failed
        return prompt, verify

    if name == "fork":
        prompt = ("The secret code for this conversation is 4817. Start exactly one sub-agent with "
                  "fork_context set to true, and ask it to multiply the secret code by 3 with a python3 "
                  "program. Do not write the code number in its message: it can see this conversation. "
                  "Wait for it, then reply with its result.")

        def verify(answer, facts):
            failed = [] if "14451" in answer.replace(",", "") else ["missing 14451"]
            if facts["forked"] < 1:
                failed.append("no forked agent")
            return failed
        return prompt, verify

    raise SystemExit(f"unknown scenario {name}")


def loaded_model(base_url):
    """The id the server lists for its loaded model; requests must name it."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=30) as response:
        models = json.loads(response.read()).get("data", [])
    if not models:
        raise SystemExit("the server lists no loaded model")
    return models[0]["id"]


def post_chat(base_url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - start


KV = re.compile(r"chat\.complete tokens=(\d+) promptTokens=(\d+) kvReused=(\d+).*?ttftMs=(\d+).*?tokensPerSec=([\d.]+)")
ROUND = re.compile(r"agents\.round id=(\S+) fork=(\S+) promptTokens=(\d+) kvReused=(\d+) evalTokens=(\d+) ttftMs=(\d+) tokensPerSec=([\d.]+)")
SPAWN = re.compile(r"agents\.spawn id=(\S+) .*?fork=(\S+)")
FINISH = re.compile(r"agents\.finish id=(\S+) status=(\S+)(?: turns=(\d+) rounds=(\d+) toolCalls=(\d+) ms=(\d+))?")
DELIVER = re.compile(r"agents\.deliver id=(\S+) via=(\S+)")


def read_log_slice(path, start):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start)
        return handle.read().splitlines()


def log_size(path):
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def parse_facts(lines):
    facts = {"spawned": 0, "forked": 0, "completed": 0, "failed": 0, "closed": 0,
             "delivered_wait": 0, "delivered_notification": 0, "collect_before_answer": 0,
             "limit_hits": 0, "agent_ms": [], "agent_rounds": [], "agent_tool_calls": [],
             "all_prompt": 0, "all_reused": 0, "all_eval": 0,
             "child_prompt": 0, "child_reused": 0, "child_eval": 0, "child_first_round_reuse": [],
             "rounds_total": 0}
    seen_child = set()
    for line in lines:
        if (m := KV.search(line)):
            facts["rounds_total"] += 1
            facts["all_eval"] += int(m.group(1))
            facts["all_prompt"] += int(m.group(2))
            facts["all_reused"] += int(m.group(3))
        if (m := ROUND.search(line)):
            agent, prompt, reused, evald = m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5))
            facts["child_prompt"] += prompt
            facts["child_reused"] += reused
            facts["child_eval"] += evald
            if agent not in seen_child:
                seen_child.add(agent)
                facts["child_first_round_reuse"].append(round(100.0 * reused / max(prompt, 1), 1))
        if (m := SPAWN.search(line)):
            facts["spawned"] += 1
            if m.group(2).lower() == "true":
                facts["forked"] += 1
        if (m := FINISH.search(line)):
            status = m.group(2)
            if status == "completed":
                facts["completed"] += 1
            elif status == "errored":
                facts["failed"] += 1
            if m.group(6):
                facts["agent_ms"].append(int(m.group(6)))
                facts["agent_rounds"].append(int(m.group(4)))
                facts["agent_tool_calls"].append(int(m.group(5)))
        if (m := DELIVER.search(line)):
            facts["delivered_wait" if m.group(2) == "wait_agent" else "delivered_notification"] += 1
        if "agents.collect-before-answer" in line:
            facts["collect_before_answer"] += 1
        if "agents.limit" in line:
            facts["limit_hits"] += 1
        if "agents.close" in line:
            facts["closed"] += 1
    parent_prompt = facts["all_prompt"] - facts["child_prompt"]
    parent_reused = facts["all_reused"] - facts["child_reused"]
    facts["parent_reuse_pct"] = round(100.0 * parent_reused / parent_prompt, 1) if parent_prompt > 0 else None
    facts["child_reuse_pct"] = (round(100.0 * facts["child_reused"] / facts["child_prompt"], 1)
                                if facts["child_prompt"] > 0 else None)
    facts["all_reuse_pct"] = (round(100.0 * facts["all_reused"] / facts["all_prompt"], 1)
                              if facts["all_prompt"] > 0 else None)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:5101")
    parser.add_argument("--server-log", help="the server's console log file, for per-request facts")
    parser.add_argument("--model-label", default="model")
    parser.add_argument("--scenarios", default="compute3,write4,files2,guard,fork")
    parser.add_argument("--modes", default="parallel,solo")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out", help="write rows + summary as JSON here")
    args = parser.parse_args()

    model = loaded_model(args.base_url)
    rows = []
    for scenario in [s for s in args.scenarios.split(",") if s]:
        for mode in [m for m in args.modes.split(",") if m]:
            if scenario in ("guard", "fork") and mode == "solo":
                continue  # these scenarios exist only to exercise sub-agent mechanics
            for attempt in range(1, args.repeat + 1):
                prompt, verify = scenario_prompts(scenario, mode)
                before = log_size(args.server_log)
                error = None
                try:
                    payload, seconds = post_chat(args.base_url, model, prompt, args.max_tokens, args.timeout)
                    answer = payload["choices"][0]["message"].get("content") or ""
                    usage = payload.get("usage", {})
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as ex:
                    payload, seconds, answer, usage, error = None, 0.0, "", {}, str(ex)
                time.sleep(0.5)  # let the server flush the request's last log lines
                facts = parse_facts(read_log_slice(args.server_log, before))
                failures = [f"request failed: {error}"] if error else verify(answer, facts)
                row = {"model": args.model_label, "scenario": scenario, "mode": mode, "attempt": attempt,
                       "seconds": round(seconds, 2), "pass": not failures, "failures": failures,
                       "answer": answer[:4000], "usage": usage, "facts": facts}
                rows.append(row)
                print(f"{args.model_label:>14} {scenario:>9} {mode:>8} #{attempt}  {seconds:7.1f}s  "
                      f"{'PASS' if not failures else 'FAIL ' + '; '.join(failures)}  "
                      f"agents={facts['spawned']} done={facts['completed']} rounds={facts['rounds_total']} "
                      f"reuse(parent/child)={facts['parent_reuse_pct']}/{facts['child_reuse_pct']}",
                      flush=True)

    summary = []
    keys = sorted({(r["scenario"], r["mode"]) for r in rows})
    for scenario, mode in keys:
        group = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
        times = [r["seconds"] for r in group if not r["failures"] or r["seconds"] > 0]
        summary.append({
            "scenario": scenario, "mode": mode, "runs": len(group),
            "passed": sum(1 for r in group if r["pass"]),
            "median_seconds": round(statistics.median(times), 2) if times else None,
            "min_seconds": round(min(times), 2) if times else None,
            "median_rounds": statistics.median([r["facts"]["rounds_total"] for r in group]) if group else None,
            "median_parent_reuse_pct": _median([r["facts"]["parent_reuse_pct"] for r in group]),
            "median_child_reuse_pct": _median([r["facts"]["child_reuse_pct"] for r in group]),
            "child_first_round_reuse_pct": _median(
                [v for r in group for v in r["facts"]["child_first_round_reuse"]]),
        })
    print("\nsummary")
    for item in summary:
        print("  " + json.dumps(item))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"rows": rows, "summary": summary}, handle, indent=2)
    return 0 if all(r["pass"] for r in rows) else 1


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


if __name__ == "__main__":
    sys.exit(main())
