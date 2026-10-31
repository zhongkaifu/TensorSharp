#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Interleaved A/B harness for the direct-CUDA prefill work (--backend cuda).

Every measurement runs one engine process at a time, in an interleaved arm sequence (ABBA ABBA),
so slow drift of the machine (thermals, page cache, other tenants) lands on both arms. An A/A
control (two copies of the base build, sequence ACCA...) measures the noise band the gates scale
with; without one, the timing gates are NOT RUN.

Subcommands (each run writes JSON Lines under --out with a fresh run id, then a report; an older
file of the same kind is moved aside to <name>.prev-<run id>, so a report never mixes runs):

  bench   prefill tok/s at --lengths and decode tok/s, from TensorSharp.Cli --benchmark. Run 1 of
          every process is warm-up and never gated; prefill still ramping inside a process is
          detected and only the settled tail is used. Each process contributes the MEDIAN of its
          used runs; an arm is the median of its process medians.
  conc    G2c: decode tok/s at concurrency --levels through TensorSharp.Server (streaming
          /v1/chat/completions, tokens from the final chunk's usage), over the part of the wave
          after every request has its first token.
  spec    G2b: speculative decoding (--spec-args) through the server: acceptance from the
          server's "Speculative decoding stats" lines AFTER the warm-up request, decode tok/s from
          the client.
  parity  G3b (and G3-exact): eng/ForcedLogitProbe teacher-forced logits against a reference
          (default ggml_cuda, run once or reused with --reference-dir). Real-text cases only
          (--cases cases.json with pinned token ids, or --prompts DIR); --synthetic-prompts is a
          smoke run whose verdict never gates.
  smoke   G3c: greedy decoding (default 128 tokens) of the same cases on the reference, A and B.
  report  recompute the verdicts from existing JSON Lines.
  selftest  the statistics, parsers and gate rules on canned data (no GPU, no build).

Gates (docs/validation/direct-cuda-prefill/design/PLAN.md, section 4, with the Stage 0 rules):
  G1   prefill (--role gate): pooled B/A >= 1 + max(3%, 2 x A/A noise) at every gated length and
       every paired cycle ratio > 1. Needs --aa-report; NOT RUN without it, or when the A/A band
       is above 3% (STOP: find the cause first), or when an arm has fewer processes/samples than
       the protocol asks (>= 4 processes and >= 16 timed samples per arm at <= 4096 tokens, >= 2
       processes above). --expected-gain LO,HI flags a gain below the package's range.
  G1c  prefill (--role control): B/A >= 1 - max(1%, A/A noise).
  G1'  fallback build (--fallback-arm P, the committed-PTX-only B'): P/A >= 1 - max(1%, noise),
       i.e. not slower than A beyond the noise band.
  G2   decode: B/A >= 1 - max(1%, A/A decode noise); with that noise above 1%, at least 4 ABBA
       cycles are required (else NOT RUN).
  G2b  speculative: acceptance(B) >= acceptance(A) - 0.02 and decode B/A >= 1 - max(1%, noise).
  G2c  concurrent decode: B/A >= 1 - max(1%, noise) at every level.
  G3b  parity, per case and per metric (step-0 relative L2, step-0 KL, median relative L2,
       median KL over the teacher-forced rows): B <= max(1.10 x A, floor) with the floors from
       --parity-floors (Stage 0's ggml-vs-ggml distances); top-1 mismatches B <= A + floor flips.
       Both arms must have finished every case (complete metrics, exit 0/1, the reference's row
       count, all rows finite), or the gate is ERROR.
  G3x  bit-exact packages (--bit-exact): every logits row of B byte-identical to A's.
  G3c  greedy smoke: --bit-exact: B's tokens and logits identical to A's; otherwise B's first
       divergence from the reference no earlier than A's on at least 2 of 3 prompts. The decoded
       text is in the report for the coherence check, which a human makes.
  G6   VRAM: B's nvidia-smi peak <= A's peak + --vram-budget-mib (default 0).
A skipped or failed measurement is reported as NOT RUN / ERROR, never as a pass.

Typical use on a VM (claim the VM first, see eng/validation/README-direct-cuda-prefill.md):

  H=eng/validation/direct-cuda-prefill-ab.py
  python3 -I $H bench --aa --out OUT/aa --model M.gguf --arm A=/root/ts-base --arm C=/root/ts-base2 \\
      --base A --candidate C --sequence ACCAACCA --lengths 512
  python3 -I $H bench --out OUT/bench --model M.gguf --arm A=/root/ts-base --arm B=/root/ts-head \\
      --sequence ABBAABBA --lengths 512,2048 --aa-report OUT/aa/report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

RUN_RE = re.compile(r"benchmark run (\d+)/(\d+): prefillMs=(\S+) prefillTps=(\S+) decodeMs=(\S+) decodeTps=(\S+)")
SPEC_RE = re.compile(r"Speculative decoding stats for (\S+): drafted=(\d+) accepted=(\d+)")

# PLAN section 4 / Stage 0 constants.
G1_MIN_GAIN = 0.03          # G1: at least +3% ...
AA_STOP = 0.03              # ... and an A/A band above 3% stops every timing gate
CONTROL_SLACK = 0.01        # controls, B' and decode: >= 1 - max(1%, noise)
DECODE_MIN_CYCLES_NOISY = 4 # G2 with decode noise above 1%: at least 4 ABBA cycles
PARITY_RATIO = 1.10         # G3: <= max(1.10 x A, floor)
SPEC_ACCEPT_SLACK = 0.02
SMOKE_MIN_SHARE = 2 / 3     # G3c: at least 2 of 3 prompts
PARITY_METRICS = ("step0_max_relative_l2", "step0_max_kl", "median_relative_l2", "median_kl")


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------

def median(values):
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else None


def settled_runs(values: list, warmup: int = 1, ramp_tolerance: float = 0.03) -> tuple[list, bool]:
    """The timed runs of one process: drop `warmup` leading runs, then, if the rest still ramps
    (the first half's median is more than `ramp_tolerance` below the second half's), keep only the
    second half. Returns (runs used, ramped)."""
    timed = [v for v in values[warmup:] if v is not None and v == v]
    if len(timed) < 4:
        return timed, False
    half = len(timed) // 2
    first, second = statistics.median(timed[:half]), statistics.median(timed[half:])
    if second > 0 and first < (1 - ramp_tolerance) * second:
        return timed[half:], True
    return timed, False


def cycles(sequence: str) -> list[list[int]]:
    """Split an interleaved arm sequence into paired cycles of 2 x arms processes when every such
    chunk holds each arm twice (ABBA -> [[0,1,2,3]]; ABBAABBA -> two cycles), else of one process
    per arm (ABC -> [[0,1,2]]). A trailing partial chunk is its own cycle."""
    k = len(set(sequence))
    size = 2 * k
    chunks = [sequence[i:i + size] for i in range(0, len(sequence), size)]
    if not all(len(c) == size and all(c.count(a) == 2 for a in set(sequence)) for c in chunks):
        size = k
    return [list(range(i, min(i + size, len(sequence)))) for i in range(0, len(sequence), size)]


def full_cycles(sequence: str, base: str, cand: str) -> int:
    """Number of paired cycles that hold both arms (each at least once)."""
    return sum(1 for c in cycles(sequence) if base in {sequence[i] for i in c} and cand in {sequence[i] for i in c})


def process_median(rec, key="samples"):
    return median(rec.get(key) or [])


def pooled(records, arm, key="samples"):
    """PLAN section 4: each process contributes the median of its used runs; the arm statistic is
    the median of those process medians. Returns (statistic, processes, timed samples)."""
    meds = [process_median(r, key) for r in records if r["arm"] == arm]
    meds = [m for m in meds if m is not None]
    samples = sum(len(r.get(key) or []) for r in records if r["arm"] == arm)
    return median(meds), len(meds), samples


def cycle_ratios(records, sequence, base, cand, key="samples"):
    """Per paired cycle: the mean of the candidate's process medians over the base's."""
    ratios = []
    by_index = {r["process_index"]: r for r in records}
    for cyc in cycles(sequence):
        a = [process_median(by_index[i], key) for i in cyc if i in by_index and by_index[i]["arm"] == base]
        b = [process_median(by_index[i], key) for i in cyc if i in by_index and by_index[i]["arm"] == cand]
        a, b = [x for x in a if x], [x for x in b if x]
        if a and b:
            ratios.append(statistics.mean(b) / statistics.mean(a))
    return ratios


def aa_noise(records, sequence, base, copy, key="samples"):
    """Relative difference of the two copies' pooled medians plus the largest per-cycle deviation
    (PLAN.md section 4)."""
    a, _, _ = pooled(records, base, key)
    c, _, _ = pooled(records, copy, key)
    if not a or not c:
        return None
    ratios = cycle_ratios(records, sequence, base, copy, key)
    worst = max((abs(r - 1) for r in ratios), default=0.0)
    return abs(c / a - 1) + worst


# ---------------------------------------------------------------------------------------------
# run bookkeeping: a fresh run id per invocation, older JSON Lines moved aside
# ---------------------------------------------------------------------------------------------

def new_run_id():
    return time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"


def start_jsonl(path, run_id):
    """A new run never appends to an older run's file: it is renamed to <path>.prev-<run id>."""
    if os.path.exists(path):
        aside = f"{path}.prev-{run_id}"
        os.replace(path, aside)
        print(f"[harness] moved the previous {os.path.basename(path)} to {os.path.basename(aside)}", flush=True)


def latest_run(records):
    """The records of the newest run in a JSON Lines file (a file holds one run; a hand-merged
    file is reported, and only its last run is used)."""
    if not records:
        return [], []
    last = records[-1].get("run_id")
    others = sorted({str(r.get("run_id")) for r in records if r.get("run_id") != last})
    return [r for r in records if r.get("run_id") == last], others


# ---------------------------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------------------------

@dataclass
class Arm:
    letter: str
    tree: str

    @property
    def cli(self):
        return os.path.join(self.tree, "TensorSharp.Cli", "bin", "TensorSharp.Cli.dll")

    @property
    def server(self):
        return os.path.join(self.tree, "TensorSharp.Server.Host", "bin", "TensorSharp.Server.Host.dll")

    def probe(self):
        hits = sorted(glob.glob(os.path.join(self.tree, "eng", "ForcedLogitProbe", "bin", "**", "ForcedLogitProbe.dll"), recursive=True))
        return hits[-1] if hits else None


def parse_arms(values) -> dict:
    arms = {}
    for v in values or []:
        letter, _, tree = v.partition("=")
        if len(letter) != 1 or not letter.isalpha() or not tree:
            raise SystemExit(f"--arm takes LETTER=TREE, got {v!r}")
        arms[letter] = Arm(letter, os.path.abspath(tree))
    return arms


def base_env(args) -> dict:
    env = dict(os.environ)
    env.update({
        "TENSORSHARP_GGML_NO_UPDATE": "1", "TS_PREFILL_WARMUP": "0", "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
    })
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def warm_page_cache(paths):
    """Read the model files once before a block (MooseFS/cgroup stalls otherwise land on whichever
    arm loads first)."""
    for p in paths:
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                while f.read(64 << 20):
                    pass


class VramSampler:
    """nvidia-smi memory.used every 250 ms while a process runs; peak per GPU index."""

    def __init__(self, path):
        self.path, self.proc = path, None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
                                         stdout=open(self.path, "w"), stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def peak(self):
        return vram_peak(self.path)


def vram_peak(path):
    peak = {}
    try:
        for line in open(path):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[1].isdigit():
                peak[parts[0]] = max(peak.get(parts[0], 0), int(parts[1]))
    except OSError:
        pass
    return peak


def run_logged(cmd, env, log_path, timeout, dry_run):
    with open(log_path + ".cmd", "w") as f:
        f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    if dry_run:
        print("DRY:", " ".join(shlex.quote(c) for c in cmd))
        return None, 0.0
    t0 = time.time()
    with open(log_path, "w") as log, VramSampler(log_path + ".vram"):
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, time.time() - t0


def parse_bench(text: str):
    runs = []
    for m in RUN_RE.finditer(text):
        def f(x):
            try:
                v = float(x)
                return v if v == v else None
            except ValueError:
                return None
        runs.append({"run": int(m.group(1)), "prefill_ms": f(m.group(3)), "prefill_tps": f(m.group(4)),
                     "decode_ms": f(m.group(5)), "decode_tps": f(m.group(6))})
    return runs


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def parse_ints(text):
    return [int(x) for x in str(text).split(",") if x.strip()]


def check_sequence(sequence, arms):
    missing = sorted(set(sequence) - set(arms))
    if missing:
        raise SystemExit(f"--sequence uses arms with no --arm: {missing}")


# ---------------------------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------------------------

def cmd_bench(args):
    arms = parse_arms(args.arm)
    check_sequence(args.sequence, arms)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "bench.jsonl")
    run_id = new_run_id()
    start_jsonl(out, run_id)
    env = base_env(args)
    jobs = [("prefill", n) for n in parse_ints(args.lengths)]
    if args.decode > 0:
        jobs.append(("decode", args.decode_prefill))
    for kind, n in jobs:
        ctx = args.max_context or (16384 if n > 4096 else 4096)
        env_job = dict(env, MAX_CONTEXT=str(ctx))
        if not args.no_warm_cache:
            warm_page_cache([args.model])
        for i, letter in enumerate(args.sequence):
            arm = arms[letter]
            runs = args.runs_decode if kind == "decode" else (args.runs_long if n > 4096 else args.runs)
            cmd = [args.dotnet, arm.cli, "--model", args.model, "--backend", args.backend, "--benchmark",
                   "--bench-prefill", str(n), "--bench-decode", str(args.decode if kind == "decode" else 0),
                   "--bench-runs", str(runs), "--bench-random-tokens"] + shlex.split(args.extra or "")
            log = os.path.join(args.out, f"{kind}{n}-p{i:02d}-{letter}.log")
            code, secs = run_logged(cmd, env_job, log, args.timeout, args.dry_run)
            text = open(log, errors="replace").read() if os.path.exists(log) else ""
            parsed = parse_bench(text)
            key = "prefill_tps" if kind == "prefill" else "decode_tps"
            used, ramped = settled_runs([r[key] for r in parsed], warmup=args.warmup_runs)
            rec = {"run_id": run_id, "kind": kind, "n": n, "arm": letter, "tree": arm.tree, "process_index": i,
                   "sequence": args.sequence, "exit": code, "seconds": round(secs, 1), "runs": parsed, "samples": used,
                   "process_median": median(used), "ramped": ramped, "vram_peak_mib": vram_peak(log + ".vram"),
                   "log": log, "model": args.model, "backend": args.backend, "aa": bool(args.aa)}
            append(out, rec)
            print(f"[{kind} {n}] p{i} {letter}: exit={code} {key}={['%.1f' % v for v in used]} ramped={ramped}", flush=True)
    write_report(args)


# ---------------------------------------------------------------------------------------------
# server: conc (G2c) and spec (G2b)
# ---------------------------------------------------------------------------------------------

class Server:
    def __init__(self, args, arm, port, log, extra_args, max_running):
        self.args, self.arm, self.port, self.log = args, arm, port, log
        self.extra_args, self.max_running, self.proc = extra_args, max_running, None

    def __enter__(self):
        env = base_env(self.args)
        env.update({"MAX_CONTEXT": str(self.args.max_context or 2048), "TS_SCHED_MAX_RUNNING_SEQS": str(self.max_running)})
        cmd = [self.args.dotnet, self.arm.server, "--model", self.args.model, "--backend", self.args.backend,
               "--host", "127.0.0.1", "--port", str(self.port), "--no-webui", "--no-skills", "--no-multi-agent"]
        if self.args.no_prefix_cache:
            cmd.append("--no-prefix-cache")
        cmd += shlex.split(self.extra_args or "")
        with open(self.log + ".cmd", "w") as f:
            f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
        if self.args.dry_run:
            print("DRY:", " ".join(shlex.quote(c) for c in cmd))
            return self
        self.proc = subprocess.Popen(cmd, env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + self.args.timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during startup ({self.proc.returncode}); see {self.log}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:
                    if b'"id"' in r.read():
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise RuntimeError(f"server not ready after {self.args.timeout}s; see {self.log}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()


PROMPTS = [
    "Write a detailed, step-by-step explanation of how a hash map handles collisions, with examples.",
    "Describe the water cycle for a curious ten-year-old, covering evaporation, condensation and rain.",
    "Explain, in plain prose, why the sky is blue and why sunsets are red. Be thorough.",
    "List and explain ten practical tips for writing maintainable Python code.",
]


def parse_sse(lines):
    """(content delta arrival times, completion tokens from the final usage, request id) of one
    streamed chat completion. A delta can carry several tokens, so token counts come from usage."""
    times, usage, rid = [], None, None
    for raw in lines:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        rid = rid or chunk.get("id")
        if isinstance(chunk.get("usage"), dict) and chunk["usage"].get("completion_tokens") is not None:
            usage = int(chunk["usage"]["completion_tokens"])
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                times.append(time.time() if isinstance(raw, bytes) else float(len(times)))
    return times, usage, rid


def stream_one(port, prompt, max_tokens, timeout):
    """One streaming chat completion; returns (delta times, completion tokens, error)."""
    body = json.dumps({"model": "m", "stream": True, "temperature": 0, "max_tokens": max_tokens,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            times, usage, _ = parse_sse(r)
        return times, usage, None
    except Exception as ex:  # noqa: BLE001 - reported, never swallowed into a pass
        return [], None, repr(ex)


def tokens_per_delta(times, usage):
    """Tokens one content delta stands for (usage / deltas; 1 when the server sent no usage)."""
    return (usage / len(times)) if usage and times else 1.0


def wave_rate(results):
    """Decode tok/s of a concurrent wave: tokens delivered after every request has its first token,
    over the rest of the wave. results: [(times, usage, error)]."""
    errors = [e for _, _, e in results if e]
    if errors or any(not t for t, _, _ in results):
        return {"error": errors[:3] or ["no tokens"]}
    start, end = max(t[0] for t, _, _ in results), max(t[-1] for t, _, _ in results)
    after = sum(tokens_per_delta(t, u) * sum(1 for x in t if x > start) for t, u, _ in results)
    total = sum(u if u else len(t) for t, u, _ in results)
    return {"tokens": total, "decode_tps": after / (end - start) if end > start else None,
            "usage_missing": sum(1 for _, u, _ in results if not u)}


def wave(port, n, max_tokens, timeout, salt):
    results = [None] * n

    def worker(i):
        prompt = f"[{salt}-{i}] " + PROMPTS[i % len(PROMPTS)]
        results[i] = stream_one(port, prompt, max_tokens, timeout)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec = wave_rate(results)
    rec.update({"n": n, "wall": time.time() - t0})
    if "error" not in rec:
        rec["ttft_max"] = max(t[0] for t, _, _ in results) - t0
    return rec


def single_rate(times, usage):
    """Decode tok/s of one request: the tokens after the first delta over the time they took."""
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    per = tokens_per_delta(times, usage)
    return per * (len(times) - 1) / (times[-1] - times[0])


def cmd_conc(args):
    arms = parse_arms(args.arm)
    check_sequence(args.sequence, arms)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "conc.jsonl")
    run_id = new_run_id()
    start_jsonl(out, run_id)
    levels = parse_ints(args.levels)
    if not args.no_warm_cache:
        warm_page_cache([args.model])
    for i, letter in enumerate(args.sequence):
        arm = arms[letter]
        log = os.path.join(args.out, f"conc-p{i:02d}-{letter}.server.log")
        rec = {"run_id": run_id, "kind": "conc", "arm": letter, "tree": arm.tree, "process_index": i,
               "sequence": args.sequence, "levels": {}}
        try:
            with Server(args, arm, args.port, log, args.server_args, max(levels)):
                if not args.dry_run:
                    wave(args.port, 2, 16, args.timeout, f"warm{i}")      # warm-up wave, never gated
                    for n in levels:
                        res = [wave(args.port, n, args.max_tokens, args.timeout, f"p{i}r{r}") for r in range(args.repeats)]
                        rec["levels"][str(n)] = res
                        print(f"[conc] p{i} {letter} n={n}: " + ", ".join(
                            f"{r.get('decode_tps') or 0:.1f}" if not r.get("error") else f"ERR {r['error']}" for r in res), flush=True)
        except RuntimeError as ex:
            rec["error"] = str(ex)
            print(f"[conc] p{i} {letter}: {ex}", flush=True)
        append(out, rec)
    write_report(args)


def spec_counts(text):
    drafted = accepted = 0
    for m in SPEC_RE.finditer(text):
        drafted += int(m.group(2))
        accepted += int(m.group(3))
    return drafted, accepted


def wait_for_stats_line(log, timeout=15.0):
    """After the warm-up request: wait until its stats line is in the log (it is written as the
    request completes, possibly just after the client sees the last token), then return the log
    size, so the measured requests' acceptance excludes the warm-up's."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if SPEC_RE.search(open(log, errors="replace").read()):
                break
        except OSError:
            pass
        time.sleep(0.5)
    try:
        return os.path.getsize(log)
    except OSError:
        return 0


def cmd_spec(args):
    arms = parse_arms(args.arm)
    check_sequence(args.sequence, arms)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "spec.jsonl")
    run_id = new_run_id()
    start_jsonl(out, run_id)
    if not args.no_warm_cache:
        warm_page_cache([args.model])
    for i, letter in enumerate(args.sequence):
        arm = arms[letter]
        log = os.path.join(args.out, f"spec-p{i:02d}-{letter}.server.log")
        rec = {"run_id": run_id, "kind": "spec", "arm": letter, "tree": arm.tree, "process_index": i, "sequence": args.sequence}
        offset = 0
        try:
            with Server(args, arm, args.port, log, (args.server_args or "") + " " + args.spec_args, 1):
                if not args.dry_run:
                    wave(args.port, 1, 16, args.timeout, f"warm{i}")
                    offset = wait_for_stats_line(log)
                    rates = []
                    for r in range(args.repeats):
                        for j, prompt in enumerate(PROMPTS):
                            times, usage, err = stream_one(args.port, f"[spec{i}-{r}-{j}] " + prompt, args.max_tokens, args.timeout)
                            rate = None if err else single_rate(times, usage)
                            if rate is None:
                                rec.setdefault("errors", []).append(err or "too few tokens")
                                continue
                            rates.append(rate)
                    rec["decode_tps"] = rates
        except RuntimeError as ex:
            rec["error"] = str(ex)
        text = ""
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                f.seek(offset)
                text = f.read()
        rec["drafted"], rec["accepted"] = spec_counts(text)
        rec["stats_offset"] = offset
        rec["acceptance"] = rec["accepted"] / rec["drafted"] if rec["drafted"] else None
        print(f"[spec] p{i} {letter}: acceptance={rec['acceptance']} tps={['%.1f' % v for v in rec.get('decode_tps') or []]}", flush=True)
        append(out, rec)
    write_report(args)


# ---------------------------------------------------------------------------------------------
# parity (G3b, G3-exact) and greedy smoke (G3c)
# ---------------------------------------------------------------------------------------------

# Smoke only (--synthetic-prompts): a few sentences repeated to the target length. Real-text
# gating uses --cases (Stage 0's pinned cases.json) or --prompts (code/prose/chat text files).
PARITY_TEXT = {
    "code": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n    i = j = 0\n    out = []\n"
            "    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
            "        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n\n",
    "prose": "The river had been rising for three days when the ferryman finally agreed to take the last "
             "passengers across. Nobody spoke on the crossing; the water was the colour of strong tea, "
             "and the old boat groaned each time the current caught it broadside. ",
    "chat": "User: Can you explain how photosynthesis turns sunlight into chemical energy?\nAssistant: "
            "Plants capture light with chlorophyll, split water, and use the energy to build sugars from "
            "carbon dioxide. Let me walk through the two stages in detail.\n",
}


def build_cases(args):
    """(cases, source label, gate-grade?)."""
    if getattr(args, "cases", None):
        cases = json.load(open(args.cases))
        for c in cases:
            c.pop("generated_tokens", None)
        return cases, f"cases {args.cases}", True
    lengths = parse_ints(args.lengths)
    if getattr(args, "prompts", None):
        texts = {}
        for path in sorted(glob.glob(os.path.join(args.prompts, "*.txt"))):
            texts[os.path.splitext(os.path.basename(path))[0]] = open(path, encoding="utf-8").read()
        if not texts:
            raise SystemExit(f"--prompts {args.prompts}: no *.txt")
        return [{"name": f"{k}{n}", "prompt": v, "target_tokens": n} for n in lengths for k, v in texts.items()], \
            f"prompts {args.prompts}", True
    if getattr(args, "synthetic_prompts", False):
        return [{"name": f"{k}{n}", "prompt": v, "target_tokens": n} for n in lengths for k, v in PARITY_TEXT.items()], \
            "built-in repeated sentences (smoke only)", False
    raise SystemExit("parity/smoke need real-text cases: --cases cases.json (pinned token ids, e.g. Stage 0's) "
                     "or --prompts DIR (*.txt); --synthetic-prompts runs a smoke test that never gates")


def f32_rows(path, vocab):
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    return size // (4 * vocab) if vocab and size % (4 * vocab) == 0 else None


def validate_reference(ref_dir):
    """A reference directory is usable when its metrics.json is complete, cases.json lists every
    case's forced tokens, and each <case>.f32 holds exactly that many rows of one vocabulary width.
    Returns (problems, {case: rows})."""
    problems, rows = [], {}
    metrics_path, cases_path = os.path.join(ref_dir, "metrics.json"), os.path.join(ref_dir, "cases.json")
    if not os.path.exists(cases_path):
        return [f"{ref_dir}: no cases.json"], rows
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))
        if m.get("complete") is False:
            problems.append(f"{ref_dir}: metrics.json is incomplete ({m.get('cases_done')}/{m.get('cases_total')} cases)")
        if m.get("all_finite") is False:
            problems.append(f"{ref_dir}: the reference has non-finite logits")
    cases = json.load(open(cases_path))
    vocab = None
    for c in cases:
        gen = c.get("generated_tokens") or []
        if not gen:
            problems.append(f"{ref_dir}: case {c.get('name')} has no generated_tokens")
            continue
        f32 = os.path.join(ref_dir, c["name"] + ".f32")
        try:
            size = os.path.getsize(f32)
        except OSError:
            problems.append(f"{ref_dir}: missing {c['name']}.f32")
            continue
        if size % (4 * len(gen)):
            problems.append(f"{ref_dir}: {c['name']}.f32 is not {len(gen)} whole rows")
            continue
        width = size // (4 * len(gen))
        if vocab is not None and width != vocab:
            problems.append(f"{ref_dir}: {c['name']}.f32 rows are {width} wide, others {vocab}")
        vocab = vocab or width
        rows[c["name"]] = len(gen)
    return problems, rows


def validate_arm_output(d, code, run_id, ref_rows, dry_run=False):
    """Everything that must hold before an arm's parity numbers may be gated. Returns (problems,
    metrics or None)."""
    if dry_run:
        return ["dry run"], None
    problems = []
    if code not in (0, 1):
        problems.append(f"probe exit code {code} (0/1 = finished)")
    path = os.path.join(d, "metrics.json")
    if not os.path.exists(path):
        return problems + ["no metrics.json"], None
    m = json.load(open(path))
    if m.get("complete") is not True:
        problems.append(f"metrics.json incomplete ({m.get('cases_done')}/{m.get('cases_total')} cases)")
    if run_id is not None and m.get("run_id") != run_id:
        problems.append(f"metrics.json is from run {m.get('run_id')!r}, not this run {run_id!r} (stale)")
    if m.get("all_finite") is not True:
        problems.append("non-finite logits")
    summary = m.get("summary") or {}
    if summary.get("non_finite_rows"):
        problems.append(f"{summary['non_finite_rows']} non-finite rows")
    got = {c["name"]: (c.get("summary") or {}).get("rows") for c in m.get("cases", [])}
    if set(got) != set(ref_rows):
        problems.append(f"cases {sorted(got)} != the reference's {sorted(ref_rows)}")
    for name, rows in ref_rows.items():
        if got.get(name) is not None and got[name] != rows:
            problems.append(f"case {name}: {got[name]} rows, reference {rows}")
    if summary.get("rows") != sum(ref_rows.values()):
        problems.append(f"summary.rows {summary.get('rows')} != reference rows {sum(ref_rows.values())}")
    return problems, m


def run_reference(args, arms, cases, run_id, env, greedy_steps=None):
    ref_arm = arms[args.reference_arm or sorted(arms)[0]]
    if getattr(args, "reference_dir", None):
        ref_dir = os.path.abspath(args.reference_dir)
        problems, rows = validate_reference(ref_dir)
        return ref_dir, problems, rows, "reused"
    probe = ref_arm.probe()
    if not probe:
        raise SystemExit(f"ForcedLogitProbe is not built in {ref_arm.tree} (dotnet build eng/ForcedLogitProbe -c Release)")
    ref_dir = os.path.join(args.out, f"reference-{args.reference_backend}")
    shutil.rmtree(ref_dir, ignore_errors=True)
    cases_path = os.path.join(args.out, "cases.in.json")
    json.dump(cases, open(cases_path, "w"), indent=1)
    code, _ = run_logged([args.dotnet, probe, args.model, ref_dir, args.reference_backend, "--cases", cases_path,
                          "--steps", str(greedy_steps or args.steps), "--run-id", run_id], env, ref_dir + ".log", args.timeout, args.dry_run)
    print(f"[{args.cmd}] reference {args.reference_backend}: exit={code}", flush=True)
    if args.dry_run:
        return ref_dir, ["dry run"], {}, "dry run"
    problems, rows = validate_reference(ref_dir)
    if code not in (0, 1):
        problems.insert(0, f"reference probe exit code {code}")
    return ref_dir, problems, rows, f"ran ({args.reference_backend})"


def bytes_equal(a, b):
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            while True:
                x, y = fa.read(1 << 20), fb.read(1 << 20)
                if x != y:
                    return False
                if not x:
                    return True
    except OSError:
        return False


def cmd_parity(args):
    arms = parse_arms(args.arm)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "parity.jsonl")
    run_id = new_run_id()
    start_jsonl(out, run_id)
    env = base_env(args)
    cases, source, gate_grade = build_cases(args)
    lengths = [c.get("target_tokens") or len(c.get("prompt_tokens") or []) for c in cases]
    env["MAX_CONTEXT"] = str(args.max_context or (16384 if max(lengths or [0]) > 4096 else 4096))
    ref_dir, ref_problems, ref_rows, ref_how = run_reference(args, arms, cases, run_id, env)
    append(out, {"run_id": run_id, "kind": "reference", "dir": ref_dir, "how": ref_how, "problems": ref_problems,
                 "rows": ref_rows, "source": source, "gate_grade": gate_grade})
    for letter in sorted(arms):
        arm = arms[letter]
        probe = arm.probe()
        rec = {"run_id": run_id, "kind": "parity", "arm": letter, "tree": arm.tree, "reference": ref_dir,
               "source": source, "gate_grade": gate_grade}
        d = os.path.join(args.out, f"arm-{letter}-{args.backend}")
        rec["dir"] = d
        if not probe:
            rec["error"] = "ForcedLogitProbe not built"
        elif ref_problems:
            rec["error"] = "reference unusable: " + "; ".join(ref_problems)
        else:
            shutil.rmtree(d, ignore_errors=True)   # never read a previous run's files
            code, _ = run_logged([args.dotnet, probe, args.model, d, args.backend, "--reference", ref_dir, "--run-id", run_id],
                                 env, d + ".log", args.timeout, args.dry_run)
            rec["exit"] = code
            problems, metrics = validate_arm_output(d, code, run_id, ref_rows, args.dry_run)
            if problems:
                rec["error"] = "; ".join(problems)
            if metrics:
                rec["summary"] = metrics.get("summary")
                rec["all_finite"] = metrics.get("all_finite")
                rec["cases"] = {c["name"]: c.get("summary") for c in metrics.get("cases", [])}
        append(out, rec)
        print(f"[parity] {letter}: {rec.get('error') or rec.get('summary')}", flush=True)
    write_report(args)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmd_smoke(args):
    arms = parse_arms(args.arm)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "smoke.jsonl")
    run_id = new_run_id()
    start_jsonl(out, run_id)
    env = base_env(args)
    cases, source, gate_grade = build_cases(args)
    env["MAX_CONTEXT"] = str(args.max_context or 4096)
    ref_dir, ref_problems, ref_rows, ref_how = run_reference(args, arms, cases, run_id, env, greedy_steps=args.tokens)
    ref_gen = {}
    if not ref_problems:
        ref_gen = {c["name"]: c["generated_tokens"] for c in json.load(open(os.path.join(ref_dir, "cases.json")))}
    append(out, {"run_id": run_id, "kind": "reference", "dir": ref_dir, "how": ref_how, "problems": ref_problems,
                 "source": source, "gate_grade": gate_grade})
    for letter in sorted(arms):
        arm = arms[letter]
        probe = arm.probe()
        d = os.path.join(args.out, f"smoke-{letter}-{args.backend}")
        rec = {"run_id": run_id, "kind": "smoke", "arm": letter, "tree": arm.tree, "dir": d, "source": source, "gate_grade": gate_grade}
        if not probe or ref_problems:
            rec["error"] = "ForcedLogitProbe not built" if not probe else "reference unusable: " + "; ".join(ref_problems)
        else:
            shutil.rmtree(d, ignore_errors=True)
            code, _ = run_logged([args.dotnet, probe, args.model, d, args.backend, "--cases", os.path.join(ref_dir, "cases.json"),
                                  "--steps", str(args.tokens), "--run-id", run_id], env, d + ".log", args.timeout, args.dry_run)
            rec["exit"] = code
            m_path = os.path.join(d, "metrics.json")
            m = json.load(open(m_path)) if os.path.exists(m_path) else {}
            if args.dry_run:
                rec["error"] = "dry run"
            elif code not in (0, 1) or m.get("complete") is not True or m.get("run_id") != run_id:
                rec["error"] = f"probe did not finish (exit {code}, complete={m.get('complete')}, run_id={m.get('run_id')})"
            else:
                rec["cases"] = {c["name"]: {"tokens": c.get("generated_tokens"), "text": c.get("generated_text"),
                                            "first_divergence": first_divergence(c.get("generated_tokens") or [], ref_gen.get(c["name"], []))}
                                for c in m.get("cases", [])}
        append(out, rec)
        print(f"[smoke] {letter}: {rec.get('error') or {k: v['first_divergence'] for k, v in rec['cases'].items()}}", flush=True)
    write_report(args)


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------

def verdict(ok: Optional[bool]):
    return "NOT RUN" if ok is None else ("PASS" if ok else "FAIL")


def load_aa(path):
    """The A/A control's noise bands: {"prefill": {n: band}, "prefill_max": band, "decode": band}.
    Raises SystemExit for a file that is not an A/A report."""
    if not path:
        return None
    if not os.path.exists(path):
        raise SystemExit(f"--aa-report {path} does not exist")
    rep = json.load(open(path))
    if not rep.get("is_aa"):
        raise SystemExit(f"--aa-report {path} is not an A/A report (run bench --aa)")
    per = {int(k): v for k, v in (rep.get("aa_noise_per_length") or {}).items()}
    return {"prefill": per, "prefill_max": rep.get("aa_noise"), "decode": rep.get("aa_noise_decode"), "path": path}


def noise_for(aa, kind, n):
    if not aa:
        return None
    if kind == "decode":
        return aa.get("decode")
    return aa["prefill"].get(n, aa.get("prefill_max"))


def depth_problem(kind, n, procs_a, procs_b, samples_a, samples_b, args):
    min_procs = args.min_procs if args.min_procs is not None else (2 if n > 4096 else 4)
    min_samples = args.min_samples if args.min_samples is not None else (16 if kind == "prefill" and n <= 4096 else 0)
    if min(procs_a, procs_b) < min_procs:
        return f"{min(procs_a, procs_b)} processes in an arm (protocol: >= {min_procs})"
    if min(samples_a, samples_b) < min_samples:
        return f"{min(samples_a, samples_b)} timed samples in an arm (protocol: >= {min_samples})"
    return None


def bench_gates(bench, args, aa):
    """Gate records for every (kind, n) of a bench run (A/B or control)."""
    gates, rows = [], []
    base, cand = args.base, args.candidate
    seq = bench[0]["sequence"]
    role = getattr(args, "role", "gate")
    lo_hi = parse_range(getattr(args, "expected_gain", None))
    for kind, n in sorted({(r["kind"], r["n"]) for r in bench}):
        recs = [r for r in bench if r["kind"] == kind and r["n"] == n]
        a, pa, sa = pooled(recs, base)
        b, pb, sb = pooled(recs, cand)
        ratios = cycle_ratios(recs, seq, base, cand)
        failed = [r for r in recs if r["exit"] != 0 or not r["samples"]]
        ratio = b / a if a and b else None
        noise = noise_for(aa, kind, n)
        reason, note, ok, threshold = None, None, None, None
        if failed:
            reason = f"{len(failed)} failed process(es)"
        elif ratio is None:
            reason = "no samples"
        elif aa is None:
            reason = "no A/A report (--aa-report): the noise band is unknown"
        elif noise is None:
            reason = f"the A/A report has no {kind} noise band"
        elif noise > AA_STOP:
            reason = f"A/A noise {noise:.1%} > {AA_STOP:.0%}: STOP and find the cause before gating"
        else:
            reason = depth_problem(kind, n, pa, pb, sa, sb, args)
        if kind == "prefill":
            gate = "G1" if role == "gate" else "G1c"
            if noise is not None:
                threshold = (1 + max(G1_MIN_GAIN, 2 * noise)) if role == "gate" else (1 - max(CONTROL_SLACK, noise))
            if reason is None:
                ok = ratio >= threshold and (role != "gate" or all(x > 1 for x in ratios))
                if ok and role == "gate" and lo_hi and ratio - 1 < lo_hi[0]:
                    note = f"gain {ratio - 1:.1%} below the expected {lo_hi[0]:.0%}-{lo_hi[1]:.0%}: explain before merge"
        else:
            gate = "G2"
            if noise is not None:
                threshold = 1 - max(CONTROL_SLACK, noise)
                ncyc = full_cycles(seq, base, cand)
                if reason is None and noise > CONTROL_SLACK and ncyc < DECODE_MIN_CYCLES_NOISY:
                    reason = f"decode noise {noise:.1%} > 1%: needs >= {DECODE_MIN_CYCLES_NOISY} ABBA cycles, has {ncyc}"
            if reason is None:
                ok = ratio >= threshold
        gates.append({"gate": gate, "kind": kind, "n": n, "base": a, "candidate": b, "ratio": ratio,
                      "cycle_ratios": ratios, "threshold": threshold, "noise": noise, "processes": [pa, pb],
                      "samples": [sa, sb], "failed_processes": [r["log"] for r in failed], "reason": reason,
                      "note": note, "verdict": verdict(ok)})
        rows.append((kind, n, a, b, ratio, ratios, sum(1 for r in recs if r.get("ramped")), pa, pb, sa, sb,
                     gate, verdict(ok), reason or note))
        fb = getattr(args, "fallback_arm", None)
        if fb and any(r["arm"] == fb for r in recs) and kind == "prefill":
            p, pp, sp = pooled(recs, fb)
            rp = p / a if a and p else None
            fail_p = [r for r in recs if r["arm"] == fb and (r["exit"] != 0 or not r["samples"])]
            okp, why = None, None
            if fail_p or rp is None:
                why = "failed or missing fallback processes"
            elif noise is None or aa is None:
                why = "no A/A noise band"
            elif noise > AA_STOP:
                why = f"A/A noise {noise:.1%} > {AA_STOP:.0%}"
            else:
                why = depth_problem(kind, n, pa, pp, sa, sp, args)
            thr = 1 - max(CONTROL_SLACK, noise) if noise is not None else None
            if why is None:
                okp = rp >= thr
            gates.append({"gate": "G1'", "kind": kind, "n": n, "arm": fb, "ratio": rp, "threshold": thr, "reason": why,
                          "verdict": verdict(okp)})
        gates.append(vram_gate(recs, base, cand, kind, n, args))
    return gates, rows


def vram_gate(recs, base, cand, kind, n, args):
    def peak(arm):
        vals = [max((r.get("vram_peak_mib") or {}).values(), default=None) for r in recs if r["arm"] == arm]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    pa, pb = peak(base), peak(cand)
    budget = getattr(args, "vram_budget_mib", 0) or 0
    ok = None if pa is None or pb is None else pb <= pa + budget
    return {"gate": "G6", "kind": kind, "n": n, "base_peak_mib": pa, "candidate_peak_mib": pb, "budget_mib": budget,
            "reason": None if ok is not None else "no nvidia-smi samples", "verdict": verdict(ok)}


def parse_range(text):
    if not text:
        return None
    lo, _, hi = str(text).partition(",")
    return float(lo), float(hi or lo)


def load_floors(args):
    """Stage 0 floors: {"step0_max_relative_l2": x, "step0_max_kl": y, "median_relative_l2": z,
    "median_kl": w, "top1_flips": k, "cases": {name: {...}}}. --parity-floor sets the relative-L2
    floors only (back-compat). Returns (floors, calibrated?)."""
    floors = {m: 0.0 for m in PARITY_METRICS}
    floors["top1_flips"] = 0
    floors["cases"] = {}
    calibrated = False
    if getattr(args, "parity_floors", None):
        loaded = json.load(open(args.parity_floors))
        if "models" in loaded:   # eng/validation/direct-cuda-prefill-floors.json: pick one model
            name = getattr(args, "floors_model", None)
            if name not in loaded["models"]:
                raise SystemExit(f"--parity-floors {args.parity_floors} holds {sorted(loaded['models'])}: pass --floors-model NAME")
            loaded = loaded["models"][name]
        # null = not measured: no floor for that metric (the limit is 1.10 x A alone)
        floors.update({k: (v or 0) for k, v in loaded.items() if k != "cases" and not k.startswith("_")})
        floors["cases"] = loaded.get("cases", {})
        calibrated = True
    if getattr(args, "parity_floor", None):
        floors["step0_max_relative_l2"] = max(floors["step0_max_relative_l2"], args.parity_floor)
        floors["median_relative_l2"] = max(floors["median_relative_l2"], args.parity_floor)
        calibrated = True
    if getattr(args, "floor_flips", None) is not None:
        floors["top1_flips"] = args.floor_flips
    return floors, calibrated


def parity_gate(parity, args):
    """G3b (and G3x) from one parity run's records."""
    base, cand = args.base, args.candidate
    byarm = {r["arm"]: r for r in parity if r.get("kind") == "parity"}
    ref = next((r for r in parity if r.get("kind") == "reference"), None)
    ra, rb = byarm.get(base), byarm.get(cand)
    floors, calibrated = load_floors(args)
    out = {"gate": "G3b", "floors": {k: v for k, v in floors.items() if k != "cases"}, "floors_calibrated": calibrated}
    problems = []
    if ref and ref.get("problems"):
        problems.append("reference: " + "; ".join(ref["problems"]))
    for letter, r in ((base, ra), (cand, rb)):
        if r is None:
            problems.append(f"arm {letter}: no record")
        elif r.get("error"):
            problems.append(f"arm {letter}: {r['error']}")
    if problems:
        out.update({"verdict": "ERROR", "reason": "; ".join(problems)})
        return [out]
    if not (ra.get("gate_grade") and rb.get("gate_grade")):
        out.update({"verdict": "NOT RUN", "reason": "synthetic prompts (smoke only): rerun with --cases or --prompts"})
        return [out]
    case_rows, ok = [], True
    for name in sorted(ra["cases"]):
        ca, cb = ra["cases"][name] or {}, (rb["cases"].get(name) or {})
        cf = floors["cases"].get(name, {})
        for m in PARITY_METRICS:
            va, vb = ca.get(m), cb.get(m)
            floor = cf.get(m, floors.get(m, 0.0)) or 0.0
            if va is None or vb is None:
                case_rows.append({"case": name, "metric": m, "base": va, "candidate": vb, "limit": None, "ok": None})
                ok = None if ok is not False else ok
                continue
            limit = max(PARITY_RATIO * va, floor)
            good = vb <= limit
            ok = ok and good if ok is not None else ok
            case_rows.append({"case": name, "metric": m, "base": va, "candidate": vb, "limit": limit, "ok": good})
    fa, fbv = ra["summary"]["top1_mismatches"], rb["summary"]["top1_mismatches"]
    flips_ok = fbv <= fa + int(floors.get("top1_flips") or 0)
    if ok is not None:
        ok = ok and flips_ok
    out.update({"verdict": verdict(ok), "cases": case_rows, "top1": {"base": fa, "candidate": fbv,
                "allowed": fa + int(floors.get("top1_flips") or 0), "ok": flips_ok},
                "reason": None if calibrated else "floors uncalibrated (--parity-floors): the limit is 1.10 x A alone"})
    gates = [out]
    if getattr(args, "bit_exact", False):
        same, missing = [], []
        for name in sorted(ra["cases"]):
            fa_, fb_ = os.path.join(ra["dir"], name + ".f32"), os.path.join(rb["dir"], name + ".f32")
            if not (os.path.exists(fa_) and os.path.exists(fb_)):
                missing.append(name)
            else:
                same.append((name, bytes_equal(fa_, fb_)))
        okx = None if missing else all(s for _, s in same)
        gates.append({"gate": "G3x", "identical_cases": [n for n, s in same if s], "differing_cases": [n for n, s in same if not s],
                      "missing": missing, "verdict": verdict(okx)})
    return gates


def smoke_gate(smoke, args):
    base, cand = args.base, args.candidate
    byarm = {r["arm"]: r for r in smoke if r.get("kind") == "smoke"}
    ref = next((r for r in smoke if r.get("kind") == "reference"), None)
    ra, rb = byarm.get(base), byarm.get(cand)
    problems = []
    if ref and ref.get("problems"):
        problems.append("reference: " + "; ".join(ref["problems"]))
    for letter, r in ((base, ra), (cand, rb)):
        if r is None or r.get("error"):
            problems.append(f"arm {letter}: {(r or {}).get('error', 'no record')}")
    if problems:
        return {"gate": "G3c", "verdict": "ERROR", "reason": "; ".join(problems)}
    if not (ra.get("gate_grade") and rb.get("gate_grade")):
        return {"gate": "G3c", "verdict": "NOT RUN", "reason": "synthetic prompts (smoke only)"}
    names = sorted(ra["cases"])
    if getattr(args, "bit_exact", False):
        diffs = [n for n in names if ra["cases"][n]["tokens"] != (rb["cases"].get(n) or {}).get("tokens")
                 or not bytes_equal(os.path.join(ra["dir"], n + ".f32"), os.path.join(rb["dir"], n + ".f32"))]
        return {"gate": "G3c", "mode": "bit-exact", "differing_cases": diffs, "verdict": verdict(not diffs)}
    inf = float("inf")
    good = [n for n in names if (rb["cases"].get(n, {}).get("first_divergence") or inf) >= (ra["cases"][n]["first_divergence"] or inf)
            or rb["cases"].get(n, {}).get("first_divergence") is None]
    need = math.ceil(SMOKE_MIN_SHARE * len(names) - 1e-9)
    return {"gate": "G3c", "mode": "first divergence from the reference", "cases_no_earlier_than_A": good, "required": need,
            "divergence": {n: [ra["cases"][n]["first_divergence"], rb["cases"].get(n, {}).get("first_divergence")] for n in names},
            "verdict": verdict(len(good) >= need), "note": "text coherence: read the generated text in report.md (a human check)"}


def write_report(args):
    base, cand = args.base, args.candidate
    report = {"base": base, "candidate": cand, "role": getattr(args, "role", "gate"), "gates": [], "is_aa": bool(getattr(args, "aa", False))}
    lines = [f"# direct-CUDA prefill A/B: {base} (base) vs {cand} (candidate), role {report['role']}", ""]
    aa = None if report["is_aa"] else load_aa(getattr(args, "aa_report", None))
    if aa:
        lines += [f"A/A control: {aa['path']}: prefill {aa['prefill']} (max {aa['prefill_max']}), decode {aa['decode']}", ""]
    elif not report["is_aa"]:
        lines += ["No A/A report (--aa-report): every timing gate is NOT RUN.", ""]

    bench, older = latest_run(read_jsonl(os.path.join(args.out, "bench.jsonl")))
    if older:
        lines += [f"NOTE: bench.jsonl also holds older runs {older}; only the latest run is used.", ""]
    if bench:
        seq = bench[0]["sequence"]
        if report["is_aa"]:
            arms_present = sorted({r["arm"] for r in bench})
            copy = cand if cand in arms_present else next((a for a in arms_present if a != base), None)
            per_len = {}
            for n in sorted({r["n"] for r in bench if r["kind"] == "prefill"}):
                recs = [r for r in bench if r["kind"] == "prefill" and r["n"] == n]
                per_len[n] = aa_noise(recs, seq, base, copy) if copy else None
            dec = [r for r in bench if r["kind"] == "decode"]
            values = [v for v in per_len.values() if v is not None]
            report["aa_noise"] = max(values) if values else None
            report["aa_noise_per_length"] = per_len
            report["aa_noise_decode"] = aa_noise(dec, seq, base, copy) if dec and copy else None
            report["aa_failed_processes"] = [r["log"] for r in bench if r["exit"] != 0 or not r["samples"]]
            stop = (report["aa_noise"] or 0) > AA_STOP
            lines += [f"A/A noise band ({base} vs {copy}): " + ", ".join(f"{n}: {v:.2%}" if v is not None else f"{n}: n/a" for n, v in per_len.items())
                      + (f"; decode {report['aa_noise_decode']:.2%}" if report["aa_noise_decode"] is not None else ""),
                      f"STOP: noise above {AA_STOP:.0%}; find the cause before gating anything." if stop else "", ""]
            if report["aa_failed_processes"]:
                lines += [f"{len(report['aa_failed_processes'])} failed A/A processes: the band is not trustworthy.", ""]
        else:
            gates, rows = bench_gates(bench, args, aa)
            report["gates"] += gates
            lines += ["| kind | n | base | cand | ratio | cycle ratios | ramped | procs | samples | gate | note |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for (kind, n, a, b, ratio, ratios, ramped, pa, pb, sa, sb, gate, v, why) in rows:
                lines.append(f"| {kind} | {n} | {fmt(a)} | {fmt(b)} | {fmt(ratio, 3)} | {' '.join(fmt(x, 3) for x in ratios)} | {ramped} | "
                             f"{pa}/{pb} | {sa}/{sb} | {gate} {v} | {why or ''} |")
            lines.append("")

    conc, older = latest_run(read_jsonl(os.path.join(args.out, "conc.jsonl")))
    if conc:
        lines += ["| G2c level | base tok/s | cand tok/s | ratio | gate |", "|---|---|---|---|---|"]
        noise = noise_for(aa, "decode", 0)
        for lvl in sorted({k for r in conc for k in r.get("levels", {})}, key=int):
            def proc_rates(arm):
                return [median([w["decode_tps"] for w in r.get("levels", {}).get(lvl, []) if w.get("decode_tps")])
                        for r in conc if r["arm"] == arm]
            def errs(arm):
                return [w for r in conc if r["arm"] == arm for w in r.get("levels", {}).get(lvl, []) if w.get("error")] + \
                       [r["error"] for r in conc if r["arm"] == arm and r.get("error")]
            a, b = median(proc_rates(base)), median(proc_rates(cand))
            ratio = b / a if a and b else None
            thr = 1 - max(CONTROL_SLACK, noise or 0)
            reason = "errors" if errs(base) or errs(cand) else (None if ratio else "no samples")
            ok = None if reason else ratio >= thr
            report["gates"].append({"gate": "G2c", "level": int(lvl), "base": a, "candidate": b, "ratio": ratio, "threshold": thr,
                                    "noise": noise, "reason": reason, "verdict": verdict(ok)})
            lines.append(f"| {lvl} | {fmt(a)} | {fmt(b)} | {fmt(ratio, 3)} | {verdict(ok)} {reason or ''} |")
        lines.append("")

    spec, older = latest_run(read_jsonl(os.path.join(args.out, "spec.jsonl")))
    if spec:
        def acc(arm):
            d = sum(r.get("drafted", 0) for r in spec if r["arm"] == arm)
            a_ = sum(r.get("accepted", 0) for r in spec if r["arm"] == arm)
            return a_ / d if d else None
        ta = median([median(r.get("decode_tps") or []) for r in spec if r["arm"] == base])
        tb = median([median(r.get("decode_tps") or []) for r in spec if r["arm"] == cand])
        aa_, ab = acc(base), acc(cand)
        noise = noise_for(aa, "decode", 0)
        thr = 1 - max(CONTROL_SLACK, noise or 0)
        errs = [r.get("error") for r in spec if r.get("error")]
        ok = None if errs or None in (ta, tb, aa_, ab) else (ab >= aa_ - SPEC_ACCEPT_SLACK and tb / ta >= thr)
        report["gates"].append({"gate": "G2b", "acceptance_base": aa_, "acceptance_candidate": ab, "tps_base": ta, "tps_candidate": tb,
                                "threshold": thr, "errors": errs, "verdict": verdict(ok)})
        lines += [f"G2b speculative: acceptance {fmt(aa_, 3)} -> {fmt(ab, 3)}, decode {fmt(ta)} -> {fmt(tb)} tok/s: {verdict(ok)}", ""]

    parity, older = latest_run(read_jsonl(os.path.join(args.out, "parity.jsonl")))
    if parity:
        for g in parity_gate(parity, args):
            report["gates"].append(g)
            if g["gate"] == "G3b":
                lines.append(f"G3b parity: {g['verdict']}" + (f" ({g['reason']})" if g.get("reason") else ""))
                for row in g.get("cases", []):
                    lines.append(f"  {row['case']:<14} {row['metric']:<22} A {fmt(row['base'], 6)} B {fmt(row['candidate'], 6)} "
                                 f"limit {fmt(row['limit'], 6)} {'ok' if row['ok'] else ('n/a' if row['ok'] is None else 'OVER')}")
                if g.get("top1"):
                    t = g["top1"]
                    lines.append(f"  top-1 mismatches A {t['base']} B {t['candidate']} (allowed {t['allowed']}): {'ok' if t['ok'] else 'OVER'}")
            else:
                lines.append(f"{g['gate']} bit-exact rows: {g['verdict']} (differing {g.get('differing_cases')}, missing {g.get('missing')})")
        lines.append("")

    smoke, older = latest_run(read_jsonl(os.path.join(args.out, "smoke.jsonl")))
    if smoke:
        g = smoke_gate(smoke, args)
        report["gates"].append(g)
        lines.append(f"G3c greedy smoke: {g['verdict']}" + (f" ({g.get('reason')})" if g.get("reason") else ""))
        for r in smoke:
            if r.get("kind") == "smoke" and r.get("cases"):
                for n, c in sorted(r["cases"].items()):
                    lines.append(f"  {r['arm']} {n}: first divergence {c['first_divergence']}: {(c.get('text') or '')[:200]!r}")
        lines.append("")

    if not report["is_aa"]:
        lines.append("Gates: " + ", ".join(f"{g['gate']}{'@' + str(g.get('n') or g.get('level')) if g.get('n') or g.get('level') else ''}={g['verdict']}"
                                           for g in report["gates"]))
        lines.append("NOT RUN and ERROR rows are not passes.")
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    if not getattr(args, "quiet", False):
        print("\n".join(lines))
    return report


def fmt(v, digits=1):
    return "n/a" if v is None else (f"{v:.{digits}f}" if digits < 4 else f"{v:.{min(digits, 3)}e}")


# ---------------------------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------------------------

def cmd_selftest(_args):
    import tempfile
    log = ("info: benchmark run 1/5: prefillMs=900 prefillTps=569.0 decodeMs=0 decodeTps=NaN msPerTok=NaN\n"
           "info: benchmark run 2/5: prefillMs=850 prefillTps=602.4 decodeMs=0 decodeTps=NaN msPerTok=NaN\n"
           "info: benchmark run 3/5: prefillMs=845 prefillTps=605.9 decodeMs=0 decodeTps=NaN\n")
    runs = parse_bench(log)
    assert [r["prefill_tps"] for r in runs] == [569.0, 602.4, 605.9] and runs[0]["decode_tps"] is None
    assert settled_runs([500, 600, 601, 602, 603], warmup=1) == ([600, 601, 602, 603], False)
    used, ramped = settled_runs([400, 500, 520, 600, 610], warmup=1)
    assert ramped and used == [600, 610], (used, ramped)
    assert cycles("ABBAABBA") == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert cycles("ACCA") == [[0, 1, 2, 3]] and cycles("ABC") == [[0, 1, 2]]
    assert full_cycles("ABBAABBA", "A", "B") == 2 and full_cycles("ABBAABBAABBAABBA", "A", "B") == 4

    # pooled statistic = median of per-process medians, not of all raw samples
    recs = [{"arm": "A", "process_index": 0, "samples": [100, 100, 100, 100, 100]},
            {"arm": "A", "process_index": 1, "samples": [90]},
            {"arm": "A", "process_index": 2, "samples": [95]}]
    assert pooled(recs, "A") == (95, 3, 7), pooled(recs, "A")
    recs = [{"arm": a, "process_index": i, "samples": [v]} for i, (a, v) in enumerate(zip("ABBAABBA", [100, 110, 111, 101, 99, 109, 110, 100]))]
    assert abs(pooled(recs, "B")[0] - 110) < 1e-9
    assert all(r > 1.08 for r in cycle_ratios(recs, "ABBAABBA", "A", "B"))
    aa = [{"arm": a, "process_index": i, "samples": [v]} for i, (a, v) in enumerate(zip("ACCA", [100, 101, 99, 100]))]
    assert 0 <= aa_noise(aa, "ACCA", "A", "C") < 0.02
    m = SPEC_RE.search("Speculative decoding stats for req-1: drafted=40 accepted=31 acceptance=78%")
    assert m and m.group(2) == "40" and m.group(3) == "31"

    # SSE: tokens from usage, not from the number of deltas
    sse = ['data: {"id":"r1","choices":[{"delta":{"content":"ab"}}]}', 'data: {"id":"r1","choices":[{"delta":{"content":"cd"}}]}',
           'data: {"id":"r1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":6}}', "data: [DONE]"]
    times, usage, rid = parse_sse(sse)
    assert len(times) == 2 and usage == 6 and rid == "r1" and tokens_per_delta(times, usage) == 3
    w = wave_rate([([0.0, 1.0, 2.0], 6, None), ([0.5, 1.5, 2.5], 3, None)])
    assert abs(w["decode_tps"] - (2 * 2 + 1 * 2) / 2.0) < 1e-9, w        # after t=0.5: 2 deltas x 2 + 2 deltas x 1, over 2 s
    assert wave_rate([([0.0], 1, None), ([], None, "boom")])["error"] == ["boom"]
    assert abs(single_rate([0.0, 1.0, 2.0], 6) - 2.0) < 1e-9

    class A:  # namespace for args
        pass

    def args_for(out, **kw):
        a = A()
        a.out, a.base, a.candidate, a.aa, a.aa_report, a.role = out, "A", "B", False, None, "gate"
        a.min_procs = a.min_samples = None
        a.expected_gain = a.fallback_arm = a.parity_floors = a.parity_floor = a.floor_flips = a.floors_model = None
        a.vram_budget_mib, a.bit_exact, a.quiet = 0, False, True
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def bench_rec(run, kind, n, arm, i, samples, seq="ABBAABBA", exit_code=0, vram=1000):
        return {"run_id": run, "kind": kind, "n": n, "arm": arm, "process_index": i, "sequence": seq, "exit": exit_code,
                "samples": samples, "log": f"p{i}", "vram_peak_mib": {"0": vram}}

    with tempfile.TemporaryDirectory() as tmp:
        # A/A report
        aa_dir = os.path.join(tmp, "aa")
        os.makedirs(aa_dir)
        for i, (arm, v) in enumerate(zip("ACCAACCA", [100, 100.5, 100.2, 99.8, 100.1, 100.4, 99.9, 100])):
            append(os.path.join(aa_dir, "bench.jsonl"), bench_rec("aa1", "prefill", 512, arm, i, [v] * 4, "ACCAACCA"))
        for i, (arm, v) in enumerate(zip("ACCAACCA", [50, 50.1, 50.0, 49.9, 50.0, 50.2, 49.9, 50.0])):
            append(os.path.join(aa_dir, "bench.jsonl"), bench_rec("aa1", "decode", 512, arm, i, [v] * 2, "ACCAACCA"))
        rep = write_report(args_for(aa_dir, aa=True, candidate="C"))
        assert rep["is_aa"] and rep["aa_noise"] is not None and rep["aa_noise"] < 0.03 and rep["aa_noise_decode"] < 0.01
        aa_path = os.path.join(aa_dir, "report.json")

        # G1: +10% with every cycle > 1 -> PASS only with an A/A report
        ab = os.path.join(tmp, "ab")
        os.makedirs(ab)
        for i, (arm, v) in enumerate(zip("ABBAABBA", [100, 110, 111, 101, 99, 109, 110, 100])):
            append(os.path.join(ab, "bench.jsonl"), bench_rec("r1", "prefill", 512, arm, i, [v] * 4))
        g1 = [g for g in write_report(args_for(ab))["gates"] if g["gate"] == "G1"][0]
        assert g1["verdict"] == "NOT RUN" and "A/A" in g1["reason"], g1
        g1 = [g for g in write_report(args_for(ab, aa_report=aa_path))["gates"] if g["gate"] == "G1"][0]
        assert g1["verdict"] == "PASS", g1
        # control role: the same data passes the control rule too; a 2% drop fails it
        g1c = [g for g in write_report(args_for(ab, aa_report=aa_path, role="control"))["gates"] if g["gate"] == "G1c"][0]
        assert g1c["verdict"] == "PASS"
        # too few processes -> NOT RUN
        thin = os.path.join(tmp, "thin")
        os.makedirs(thin)
        for i, (arm, v) in enumerate(zip("ABBA", [100, 110, 111, 101])):
            append(os.path.join(thin, "bench.jsonl"), bench_rec("r1", "prefill", 512, arm, i, [v] * 4, "ABBA"))
        g = [g for g in write_report(args_for(thin, aa_report=aa_path))["gates"] if g["gate"] == "G1"][0]
        assert g["verdict"] == "NOT RUN" and "processes" in g["reason"], g
        # a noisy A/A (>3%) stops G1
        noisy = os.path.join(tmp, "noisy")
        os.makedirs(noisy)
        for i, (arm, v) in enumerate(zip("ACCAACCA", [100, 106, 104, 99, 100, 103, 105, 100])):
            append(os.path.join(noisy, "bench.jsonl"), bench_rec("aa2", "prefill", 512, arm, i, [v] * 4, "ACCAACCA"))
        write_report(args_for(noisy, aa=True, candidate="C"))
        g = [g for g in write_report(args_for(ab, aa_report=os.path.join(noisy, "report.json")))["gates"] if g["gate"] == "G1"][0]
        assert g["verdict"] == "NOT RUN" and "STOP" in g["reason"], g
        # a failed process -> NOT RUN; G6 compares VRAM peaks
        failed = os.path.join(tmp, "failed")
        os.makedirs(failed)
        for i, (arm, v) in enumerate(zip("ABBAABBA", [100, 110, 111, 101, 99, 109, 110, 100])):
            append(os.path.join(failed, "bench.jsonl"), bench_rec("r1", "prefill", 512, arm, i, [] if i == 5 else [v] * 4,
                                                               exit_code=134 if i == 5 else 0, vram=1200 if arm == "B" else 1000))
        gates = write_report(args_for(failed, aa_report=aa_path))["gates"]
        assert [g for g in gates if g["gate"] == "G1"][0]["verdict"] == "NOT RUN"
        assert [g for g in gates if g["gate"] == "G6"][0]["verdict"] == "FAIL"
        assert [g for g in write_report(args_for(failed, aa_report=aa_path, vram_budget_mib=256))["gates"] if g["gate"] == "G6"][0]["verdict"] == "PASS"
        # an older run in the same file is ignored (latest run only)
        stale = os.path.join(tmp, "stale")
        os.makedirs(stale)
        for i, (arm, v) in enumerate(zip("ABBAABBA", [100, 50, 50, 100, 100, 50, 50, 100])):
            append(os.path.join(stale, "bench.jsonl"), bench_rec("old", "prefill", 512, arm, i, [v] * 4))
        for i, (arm, v) in enumerate(zip("ABBAABBA", [100, 110, 111, 101, 99, 109, 110, 100])):
            append(os.path.join(stale, "bench.jsonl"), bench_rec("new", "prefill", 512, arm, i, [v] * 4))
        assert [g for g in write_report(args_for(stale, aa_report=aa_path))["gates"] if g["gate"] == "G1"][0]["verdict"] == "PASS"
        start_jsonl(os.path.join(stale, "bench.jsonl"), "r9")
        assert not os.path.exists(os.path.join(stale, "bench.jsonl")) and os.path.exists(os.path.join(stale, "bench.jsonl.prev-r9"))
        # G2: decode noise above 1% needs 4 cycles
        dn = os.path.join(tmp, "dn")
        os.makedirs(dn)
        for i, (arm, v) in enumerate(zip("ACCAACCA", [50, 50.4, 50.3, 49.9, 50, 50.4, 50.3, 50])):
            append(os.path.join(dn, "bench.jsonl"), bench_rec("aa3", "decode", 512, arm, i, [v] * 2, "ACCAACCA"))
            append(os.path.join(dn, "bench.jsonl"), bench_rec("aa3", "prefill", 512, arm, i, [100] * 4, "ACCAACCA"))
        write_report(args_for(dn, aa=True, candidate="C"))
        dec = os.path.join(tmp, "dec")
        os.makedirs(dec)
        for i, (arm, v) in enumerate(zip("ABBAABBA", [50, 49.6, 49.7, 50, 50, 49.5, 49.8, 50])):
            append(os.path.join(dec, "bench.jsonl"), bench_rec("r1", "decode", 512, arm, i, [v] * 2))
        dn_noise = json.load(open(os.path.join(dn, "report.json")))["aa_noise_decode"]
        assert 0.01 < dn_noise < 0.03, dn_noise
        g2 = [g for g in write_report(args_for(dec, aa_report=os.path.join(dn, "report.json")))["gates"] if g["gate"] == "G2"][0]
        assert g2["verdict"] == "NOT RUN" and "ABBA cycles" in g2["reason"], g2
        g2 = [g for g in write_report(args_for(dec, aa_report=aa_path))["gates"] if g["gate"] == "G2"][0]
        assert g2["verdict"] == "PASS", g2       # 0.994 >= 1 - max(1%, ~0.3%)

        # G3b: a crashed, incomplete, stale or short candidate is ERROR, never PASS
        ref = os.path.join(tmp, "ref")
        os.makedirs(ref)
        json.dump([{"name": "code512", "prompt_tokens": [1, 2], "generated_tokens": [5, 6, 7]}], open(os.path.join(ref, "cases.json"), "w"))
        open(os.path.join(ref, "code512.f32"), "wb").write(b"\0" * (3 * 4 * 8))
        json.dump({"complete": True, "all_finite": True}, open(os.path.join(ref, "metrics.json"), "w"))
        problems, ref_rows = validate_reference(ref)
        assert not problems and ref_rows == {"code512": 3}, (problems, ref_rows)

        def arm_dir(name, complete=True, run="r1", rows=3, finite=True, cases=("code512",)):
            d = os.path.join(tmp, name)
            os.makedirs(d, exist_ok=True)
            summ = {"rows": rows, "non_finite_rows": 0 if finite else 1, "top1_mismatches": 0, "median_relative_l2": 0.01,
                    "median_kl": 1e-4, "step0_max_relative_l2": 0.01, "step0_max_kl": 1e-4}
            json.dump({"complete": complete, "run_id": run, "all_finite": finite, "summary": summ,
                       "cases": [{"name": c, "summary": dict(summ)} for c in cases]}, open(os.path.join(d, "metrics.json"), "w"))
            return d
        assert validate_arm_output(arm_dir("ok"), 1, "r1", ref_rows)[0] == []
        assert any("exit code" in p for p in validate_arm_output(arm_dir("crash"), 134, "r1", ref_rows)[0])
        assert any("incomplete" in p for p in validate_arm_output(arm_dir("part", complete=False), 1, "r1", ref_rows)[0])
        assert any("stale" in p for p in validate_arm_output(arm_dir("stale2", run="r0"), 0, "r1", ref_rows)[0])
        assert any("rows" in p for p in validate_arm_output(arm_dir("short", rows=2), 0, "r1", ref_rows)[0])
        assert any("non-finite" in p for p in validate_arm_output(arm_dir("nan", finite=False), 0, "r1", ref_rows)[0])
        assert any("cases" in p for p in validate_arm_output(arm_dir("fewer", cases=()), 0, "r1", ref_rows)[0])
        assert validate_arm_output(os.path.join(tmp, "missing"), 1, "r1", ref_rows)[0]

        def prec(arm, rel, kl, flips, error=None, grade=True, d=None):
            s = {"rows": 3, "top1_mismatches": flips, "median_relative_l2": rel, "median_kl": kl, "step0_max_relative_l2": rel, "step0_max_kl": kl}
            r = {"run_id": "r1", "kind": "parity", "arm": arm, "gate_grade": grade, "summary": s, "cases": {"code512": dict(s)}, "dir": d}
            if error:
                r["error"] = error
            return r
        refrec = {"run_id": "r1", "kind": "reference", "problems": []}
        floors = os.path.join(tmp, "floors.json")
        json.dump({"step0_max_relative_l2": 0.05, "step0_max_kl": 1e-2, "median_relative_l2": 0.05, "median_kl": 1e-2, "top1_flips": 1}, open(floors, "w"))
        a = args_for(tmp, parity_floors=floors)
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1), prec("B", 0.04, 2e-3, 2)], a)[0]["verdict"] == "PASS"   # inside the floor, +1 flip
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1), prec("B", 0.06, 2e-3, 1)], a)[0]["verdict"] == "FAIL"   # above floor and 1.1 x A
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1), prec("B", 0.01, 1e-3, 3)], a)[0]["verdict"] == "FAIL"   # 2 extra flips
        assert parity_gate([refrec, prec("A", 0.1, 1e-3, 1), prec("B", 0.105, 1e-3, 1)], a)[0]["verdict"] == "PASS"   # within 1.10 x A
        # the shipped Stage 0 floors file: one model's entry, nulls meaning "no floor"
        shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)), "direct-cuda-prefill-floors.json")
        if os.path.exists(shipped):
            fl, cal = load_floors(args_for(tmp, parity_floors=shipped, floors_model="qwen08"))
            assert cal and fl["step0_max_relative_l2"] == 7.0e-3 and fl["median_kl"] == 0 and fl["top1_flips"] == 1, fl
            try:
                load_floors(args_for(tmp, parity_floors=shipped))
                raise AssertionError("a multi-model floors file needs --floors-model")
            except SystemExit:
                pass
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1), prec("B", 0.0, 0.0, 0, error="probe exit code 134")], a)[0]["verdict"] == "ERROR"
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1)], a)[0]["verdict"] == "ERROR"
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1, grade=False), prec("B", 0.02, 1e-3, 1, grade=False)], a)[0]["verdict"] == "NOT RUN"
        assert parity_gate([{"run_id": "r1", "kind": "reference", "problems": ["incomplete"]}, prec("A", 0.02, 1e-3, 1), prec("B", 0.02, 1e-3, 1)], a)[0]["verdict"] == "ERROR"
        # bit-exact
        da, db, dc = os.path.join(tmp, "xa"), os.path.join(tmp, "xb"), os.path.join(tmp, "xc")
        for d, payload in ((da, b"\1" * 16), (db, b"\1" * 16), (dc, b"\2" * 16)):
            os.makedirs(d)
            open(os.path.join(d, "code512.f32"), "wb").write(payload)
        ax = args_for(tmp, parity_floors=floors, bit_exact=True)
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1, d=da), prec("B", 0.02, 1e-3, 1, d=db)], ax)[1]["verdict"] == "PASS"
        assert parity_gate([refrec, prec("A", 0.02, 1e-3, 1, d=da), prec("B", 0.02, 1e-3, 1, d=dc)], ax)[1]["verdict"] == "FAIL"
        # G3c: first divergence no earlier than A's on 2 of 3 prompts
        def srec(arm, divs):
            return {"run_id": "r1", "kind": "smoke", "arm": arm, "gate_grade": True,
                    "cases": {f"c{i}": {"tokens": [1], "first_divergence": dv} for i, dv in enumerate(divs)}}
        assert smoke_gate([refrec, srec("A", [10, 20, None]), srec("B", [12, 5, None])], a)["verdict"] == "PASS"
        assert smoke_gate([refrec, srec("A", [10, 20, None]), srec("B", [9, 5, 3])], a)["verdict"] == "FAIL"
        assert first_divergence([1, 2, 3], [1, 2, 3]) is None and first_divergence([1, 2, 4], [1, 2, 3]) == 2
    print("selftest: ok")


# ---------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_model=True):
        sp.add_argument("--out", required=True, help="evidence directory (keep it under docs/validation/direct-cuda-prefill/<stage>/)")
        sp.add_argument("--arm", action="append", help="LETTER=TREE (repeat): a built TensorSharp tree per arm")
        sp.add_argument("--base", default="A", help="base arm letter [A]")
        sp.add_argument("--candidate", default="B", help="candidate arm letter [B]")
        sp.add_argument("--sequence", default="ABBAABBA", help="interleaved arm order [ABBAABBA]")
        if needs_model:
            sp.add_argument("--model", required=True)
        sp.add_argument("--backend", default="cuda", help="backend under test [cuda]")
        sp.add_argument("--dotnet", default=os.environ.get("DOTNET", "dotnet"))
        sp.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES [0]")
        sp.add_argument("--env", action="append", help="K=V for every process (repeat)")
        sp.add_argument("--max-context", type=int, default=0, help="MAX_CONTEXT (default: 4096, 16384 above 4096 tokens)")
        sp.add_argument("--timeout", type=int, default=2400)
        sp.add_argument("--no-warm-cache", action="store_true", help="skip reading the GGUF before each block")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--aa-report", help="report.json of an A/A control (bench --aa): its noise bands scale the timing gates; "
                                             "without it they are NOT RUN")
        sp.add_argument("--aa", action="store_true", help="this run IS the A/A control (base vs its copy): report the noise band")
        sp.add_argument("--role", choices=("gate", "control"), default="gate",
                        help="gate model (G1: >= +max(3%%, 2 x noise)) or regression control (G1c: >= -max(1%%, noise)) [gate]")
        sp.add_argument("--fallback-arm", help="arm letter of the committed-PTX-only build B' (G1': not slower than A beyond the noise)")
        sp.add_argument("--expected-gain", help="LO,HI fractions of the package's expected prefill gain (G1 flags a gain below LO)")
        sp.add_argument("--min-procs", type=int, help="processes per arm a gate needs [4; 2 above 4096 tokens]")
        sp.add_argument("--min-samples", type=int, help="timed samples per arm a prefill gate needs [16 at <= 4096 tokens]")
        sp.add_argument("--vram-budget-mib", type=int, default=0, help="G6: the package's declared VRAM budget over A's peak [0]")
        sp.add_argument("--parity-floor", type=float, default=None, help="relative-L2 floor for G3b (use --parity-floors for all metrics)")
        sp.add_argument("--parity-floors", help="JSON of Stage 0 floors: step0_max_relative_l2, step0_max_kl, median_relative_l2, "
                                                "median_kl, top1_flips, and optional per-case overrides under \"cases\"")
        sp.add_argument("--floors-model", help="model entry of a multi-model --parity-floors file (e.g. qwen08, gptoss)")
        sp.add_argument("--floor-flips", type=int, default=None, help="top-1 flips B may add over A (Stage 0: qwen08 1, Lightning 3, Bolt 1)")
        sp.add_argument("--bit-exact", action="store_true", help="package claims bit-exactness: G3x/G3c require B byte-identical to A")

    def cases_args(sp, lengths):
        sp.add_argument("--cases", help="ForcedLogitProbe cases.json with pinned prompt_tokens (e.g. Stage 0's gR/cases.json)")
        sp.add_argument("--prompts", help="directory of *.txt real-text prompts (code, prose, chat), cut to --lengths tokens")
        sp.add_argument("--synthetic-prompts", action="store_true", help="built-in repeated sentences: a smoke run that never gates")
        sp.add_argument("--lengths", default=lengths)
        sp.add_argument("--reference-backend", default="ggml_cuda")
        sp.add_argument("--reference-arm", help="arm whose tree runs the reference [first arm]")
        sp.add_argument("--reference-dir", help="reuse this reference directory (validated) instead of running the reference")

    b = sub.add_parser("bench", help="prefill/decode tok/s via the CLI benchmark")
    common(b)
    b.add_argument("--lengths", default="512,2048,8192")
    b.add_argument("--runs", type=int, default=5, help="runs per process at <= 4096 tokens [5]")
    b.add_argument("--runs-long", type=int, default=3, help="runs per process above 4096 tokens [3]")
    b.add_argument("--runs-decode", type=int, default=3, help="runs per process of the decode job [3]")
    b.add_argument("--warmup-runs", type=int, default=1, help="leading runs per process never gated [1]")
    b.add_argument("--decode", type=int, default=128, help="decode tokens for the G2 job (0: skip) [128]")
    b.add_argument("--decode-prefill", type=int, default=512)
    b.add_argument("--extra", help="extra CLI arguments (quoted)")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("conc", help="G2c concurrent decode via the server")
    common(c)
    c.add_argument("--levels", default="4,32,64")
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--repeats", type=int, default=2)
    c.add_argument("--port", type=int, default=8093)
    c.add_argument("--server-args", default="")
    c.add_argument("--no-prefix-cache", action="store_true")
    c.set_defaults(func=cmd_conc)

    s = sub.add_parser("spec", help="G2b speculative acceptance + tok/s via the server")
    common(s)
    s.add_argument("--spec-args", required=True, help='e.g. "--spec --spec-type ngram --spec-draft 8"')
    s.add_argument("--max-tokens", type=int, default=192)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--port", type=int, default=8094)
    s.add_argument("--server-args", default="")
    s.add_argument("--no-prefix-cache", action="store_true")
    s.set_defaults(func=cmd_spec)

    q = sub.add_parser("parity", help="G3b teacher-forced logits vs a reference backend")
    common(q)
    cases_args(q, "512,2048")
    q.add_argument("--steps", type=int, default=24)
    q.set_defaults(func=cmd_parity)

    g = sub.add_parser("smoke", help="G3c greedy decoding of the cases on the reference, A and B")
    common(g)
    cases_args(g, "512")
    g.add_argument("--tokens", type=int, default=128)
    g.set_defaults(func=cmd_smoke)

    r = sub.add_parser("report", help="recompute verdicts from existing JSON Lines")
    common(r, needs_model=False)
    r.set_defaults(func=write_report)

    t = sub.add_parser("selftest", help="parsers, statistics and gate rules on canned data")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
