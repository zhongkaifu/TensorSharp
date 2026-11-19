#!/usr/bin/env python3
"""Serving benchmark: TensorSharp.Server.Host vs llama-server with one client (OpenAI chat completions, streamed).

For each (concurrency C, prompt length P) it sends R requests (R = max(--min-requests, --waves x C), by default 2 C), C at
a time, each a distinct prompt of P tokens (a unique first line, so no prefix cache can reuse another request's prompt), asking for
exactly --gen tokens (ignore_eos, greedy, no repetition penalty, thinking off). It reports TTFT p50/p95 (request sent -> first streamed token,
content or reasoning), per-request decode tok/s ((tokens - 1) / (last token - first token)), aggregate output tok/s
(all output tokens / the level's wall time) and the server process's peak physical footprint (`footprint`, which counts
Metal buffers) and peak resident set size (`ps -o rss`: the footprint plus the clean pages of mapped files, such as
weights a server reads from their mapping) sampled every 0.25 s.

The servers are started and stopped by this script (one process per engine per concurrency, warmed by two requests):
  serve-bench.py --engine ts    --model MODEL.gguf [--config config/lfm3-1.7b.json] ...
  serve-bench.py --engine llama --model MODEL.gguf ...
Prompts are cut from --corpus to exact token counts with llama-tokenize (the GGUF's tokenizer, the same for both).

usage: serve-bench.py --engine {ts,llama} --model GGUF [--backend B] [--threads N] [--concurrency 1,4,16]
                      [--prompts 128,1024,4096] [--gen 256] [--corpus FILE] [--out DIR] [--port 18080] [--tag NAME]
  --backend  TensorSharp's backend (ggml_metal, ggml_cpu, cpu, mlx); for llama-server, cpu or ggml_cpu keeps every
             layer on the CPU (-ngl 0), anything else offloads them
  --threads  CPU threads: llama-server -t N -tb N; TensorSharp TS_GGML_CPU_THREADS, TS_CPU_THREADS and
             DOTNET_PROCESSOR_COUNT (as eng/lfm3/bench/ts-bench.sh --threads); 0 keeps each engine's defaults
  --threads-batch  prompt-batch threads apart from --threads: llama-server -tb N; TensorSharp ggml_cpu's prompt pool
             (TS_GGML_CPU_THREADS = N, its decode pool TS_GGML_CPU_DECODE_THREADS = --threads)
  --llama-fa llama-server -fa (default on)
  --tag      names this run's files and table rows (default: the engine)
Environment: LLAMA_CPP_DIR (default ~/work/llama.cpp-lfm3) and LLAMA_BUILD (its build directory, default build; e.g.
build-cpu); the TS server is this checkout's Release build of TensorSharp.Server.Host (build it first; TS_SERVER_DLL
overrides it); MAX_CONTEXT is set from the largest prompt.
"""
import argparse
import http.client
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LLAMA = os.environ.get("LLAMA_CPP_DIR", os.path.expanduser("~/work/llama.cpp-lfm3"))
LLAMA_BIN = os.path.join(LLAMA, os.environ.get("LLAMA_BUILD", "build"), "bin")
CPU_BACKENDS = ("cpu", "ggml_cpu")


def tokenize(model, text):
    p = subprocess.run([f"{LLAMA_BIN}/llama-tokenize", "-m", model, "--ids", "--log-disable", "--no-bos", "-p", text],
                       capture_output=True, text=True)
    m = re.search(r"\[([0-9, ]*)\]", p.stdout)
    if not m:
        raise RuntimeError("llama-tokenize: " + p.stderr[-400:])
    return [int(t) for t in m.group(1).split(",") if t.strip()]


def cut(model, corpus, tokens):
    """The longest prefix of the corpus (at a whitespace) holding at most `tokens` tokens, by bisection."""
    lo, hi = 0, len(corpus)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        end = corpus.rfind(" ", 0, mid)
        end = mid if end <= 0 else end
        if len(tokenize(model, corpus[:end])) <= tokens:
            lo = mid
        else:
            hi = mid - 1
    end = corpus.rfind(" ", 0, lo)
    return corpus[:end if end > 0 else lo]


def footprint_bytes(pid):
    try:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    m = re.search(r"Footprint:\s*([0-9.]+)\s*(KB|MB|GB)", out)
    if not m:
        return 0
    return float(m.group(1)) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[m.group(2)]


def rss_bytes(pid):
    """Resident set size: the footprint plus the clean pages of mapped files (weights read from their mapping)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout
        return int(out.strip()) * 1024
    except Exception:
        return 0


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.peak_rss, self.stop = pid, 0, 0, False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, footprint_bytes(self.pid))
            self.peak_rss = max(self.peak_rss, rss_bytes(self.pid))
            time.sleep(0.25)


def request(port, model_name, text, gen, timeout=1200):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": gen,
        # Greedy and no penalty on both engines (config/lfm3-1.7b.json's defaults are the model card's 0.1 / 1.1; the
        # TensorSharp server runs with --sampling-precedence request so these win).
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "stream": True,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]!r}")
    stamps, usage = [], None
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1024)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    stamps.append(time.perf_counter())
    t1 = time.perf_counter()
    conn.close()
    tokens = (usage or {}).get("completion_tokens") or len(stamps)
    return {"ttft": (stamps[0] - t0) if stamps else float("nan"), "start": t0, "end": t1,
            "first": stamps[0] if stamps else t1, "last": stamps[-1] if stamps else t1,
            "tokens": tokens, "chunks": len(stamps), "prompt_tokens": (usage or {}).get("prompt_tokens")}


def pct(values, q):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    f = int(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)


def wait_ready(port, proc, seconds=600):
    t = time.time()
    while time.time() - t < seconds:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/v1/models")
            r = c.getresponse()
            body = r.read()
            c.close()
            if r.status == 200:
                data = json.loads(body).get("data") or []
                return data[0]["id"] if data else "model"
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready")


def start_server(args, concurrency, ctx, log):
    env = dict(os.environ)
    if args.engine == "ts":
        env["MAX_CONTEXT"] = str(ctx)
        env.setdefault("TS_SCHED_MAX_RUNNING_SEQS", str(max(concurrency, 1)))
        if args.threads:
            for k in ("TS_GGML_CPU_THREADS", "TS_CPU_THREADS", "DOTNET_PROCESSOR_COUNT"):
                env[k] = str(args.threads)
        if args.threads_batch:
            # ggml_cpu's prompt pool takes the batch count, its decode pool --threads (as llama-server -tb / -t).
            env["TS_GGML_CPU_THREADS"] = str(args.threads_batch)
            if args.threads:
                env["TS_GGML_CPU_DECODE_THREADS"] = str(args.threads)
        dll = os.environ.get("TS_SERVER_DLL") or os.path.join(ROOT, "TensorSharp.Server.Host", "bin", "Release", "net10.0",
                                                              "TensorSharp.Server.Host.dll")
        if not os.path.exists(dll):
            dll = os.path.join(ROOT, "TensorSharp.Server.Host", "bin", "TensorSharp.Server.Host.dll")
        # --sampling-precedence request: the request's greedy / no-penalty settings below win over the config's model-card
        # sampling (temperature 0.1, top-k 50, repetition penalty 1.1), which `config` precedence would otherwise keep.
        cmd = ["dotnet", dll, "--model", args.model, "--backend", args.backend, "--port", str(args.port),
               "--no-multi-agent", "--no-skills", "--no-webui", "--max-tokens", str(args.gen + 64),
               "--sampling-precedence", "request"]
        if args.config:
            cmd[2:2] = ["--config", args.config]
    else:
        cmd = [f"{LLAMA_BIN}/llama-server", "-m", args.model, "--port", str(args.port), "-np", str(concurrency),
               "-c", str(ctx * concurrency), "-fa", args.llama_fa, "--jinja",
               "-ngl", "0" if args.backend in CPU_BACKENDS else "99", "--no-webui", "-b", "2048", "-ub", "512"]
        if args.threads:
            cmd += ["-t", str(args.threads)]
        if args.threads or args.threads_batch:
            cmd += ["-tb", str(args.threads_batch or args.threads)]
    print("$ " + " ".join(cmd), file=sys.stderr)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["ts", "llama"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None, help="TS: a config file (its model path is overridden by --model)")
    ap.add_argument("--backend", default="ggml_metal", help="TS backend; llama: anything but cpu offloads every layer")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--threads-batch", type=int, default=0)
    ap.add_argument("--llama-fa", default="on", help="llama-server -fa (on, off, auto)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--concurrency", default="1,4,16")
    ap.add_argument("--prompts", default="128,1024,4096")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--min-requests", type=int, default=4)
    ap.add_argument("--waves", type=int, default=2, help="requests per level: max(--min-requests, WAVES x C)")
    ap.add_argument("--corpus", default=os.path.join(ROOT, "docs", "models", "lfm3.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "lfm3", "serve"))
    ap.add_argument("--port", type=int, default=18080)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or args.engine
    levels = [int(c) for c in args.concurrency.split(",")]
    lengths = [int(p) for p in args.prompts.split(",")]

    corpus = open(args.corpus, encoding="utf-8").read()
    while len(corpus) < 40 * max(lengths):
        corpus += "\n" + corpus
    cache = os.path.join(args.out, "prompts.json")
    prompts = json.load(open(cache)) if os.path.exists(cache) else {}
    for n in lengths:
        if str(n) not in prompts:
            # Room for the nonce line and the chat framing: the body is cut a little short of n.
            prompts[str(n)] = cut(args.model, corpus, n - 12)
    json.dump(prompts, open(cache, "w"))

    results = []
    ctx = max(lengths) + args.gen + 128
    for c in levels:
        log = open(os.path.join(args.out, f"{tag}-c{c}.server.log"), "w")
        proc = start_server(args, c, ctx, log)
        try:
            name = wait_ready(args.port, proc)
            for i in range(2):   # warm: kernels, graphs, slots
                request(args.port, name, f"(warm {i})\n" + prompts[str(lengths[0])], 16)
            for n in lengths:
                total = max(args.min_requests, args.waves * c)
                texts = [f"(request {n}-{k})\n" + prompts[str(n)] for k in range(total)]
                out, lock, it = [], threading.Lock(), iter(texts)
                sampler = Sampler(proc.pid)
                sampler.start()

                def worker():
                    while True:
                        with lock:
                            t = next(it, None)
                        if t is None:
                            return
                        r = request(args.port, name, t, args.gen)
                        with lock:
                            out.append(r)
                t0 = time.perf_counter()
                threads = [threading.Thread(target=worker) for _ in range(c)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - t0
                sampler.stop = True
                sampler.join()
                ttft = [r["ttft"] for r in out]
                rates = [(r["chunks"] - 1) / (r["last"] - r["first"]) for r in out if r["chunks"] > 1 and r["last"] > r["first"]]
                tokens = sum(r["tokens"] for r in out)
                row = {
                    "engine": tag, "concurrency": c, "prompt": n, "requests": len(out),
                    "prompt_tokens": statistics.median([r["prompt_tokens"] for r in out if r["prompt_tokens"]] or [0]),
                    "ttft_p50_ms": 1000 * pct(ttft, 0.5), "ttft_p95_ms": 1000 * pct(ttft, 0.95),
                    "decode_tok_s_p50": pct(rates, 0.5), "decode_tok_s_min": min(rates) if rates else float("nan"),
                    "aggregate_tok_s": tokens / wall, "output_tokens": tokens, "wall_s": wall,
                    "peak_footprint_gb": sampler.peak / (1 << 30), "peak_rss_gb": sampler.peak_rss / (1 << 30),
                    "short": sum(1 for r in out if r["tokens"] < args.gen),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                os.killpg(proc.pid, signal.SIGKILL)
            log.close()
    with open(os.path.join(args.out, f"{tag}-results.jsonl"), "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("| engine | C | prompt | TTFT p50 ms | TTFT p95 ms | decode tok/s/req p50 | aggregate tok/s | peak footprint GB | peak RSS GB | short |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['engine']} | {r['concurrency']} | {r['prompt']} | {r['ttft_p50_ms']:.0f} | {r['ttft_p95_ms']:.0f} | "
              f"{r['decode_tok_s_p50']:.1f} | {r['aggregate_tok_s']:.1f} | {r['peak_footprint_gb']:.2f} | "
              f"{r['peak_rss_gb']:.2f} | {r['short']} |")


if __name__ == "__main__":
    sys.exit(main())
