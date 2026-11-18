#!/usr/bin/env python3
"""LFM3-1.7B on the Mac's CPU: vLLM (CPU backend) vs TensorSharp (ggml_cpu, managed cpu) vs llama.cpp (CPU build).

Driven by vllm-cpu-mac.sh (read its header first); every subcommand also runs on its own. Runs under the vLLM venv's
Python (it imports vllm, transformers and torch). Outputs go to --out (never into Git).

  check                       versions, vllm._C and its CPU ops, model files, binaries
  prompts                     the correctness prompts rendered with the HF chat template (thinking off) -> prompts.json
  gen-vllm  --dtype D         vLLM offline greedy generation from prompts.json's token ids (LLM.generate)
  gen-hf    --dtype D         HF transformers greedy generation (the checkpoint's remote code) from the same token ids;
            [--recompute]     with --recompute a full forward of the whole sequence per token, no cache (the reference)
  gen-server ENGINE ...       one server, greedy chat completions of the same messages; token ids where the server
                              returns them (vLLM: return_tokens_as_token_ids, llama-server: logprob ids), otherwise the
                              text re-tokenized with the HF tokenizer (TensorSharp)
  compare                     token identity / first divergence of every gen-*.json against a reference (--ref)
  margins                     at each first divergence, the reference model's logit gap between the two tokens
  arm ENGINE ...              one benchmark arm: start the server, warm it, run each cell with `vllm bench serve`
                              (backend openai-chat, random dataset), sample the server process tree's peak footprint
                              and RSS, stop the server; one JSON per cell
  summary                     medians over rounds (min-max) of every arm/cell in --out

ENGINE options (gen-server, arm): --engine {vllm,ts,llama} --name LABEL --threads T [--threads-batch TB]
  vllm:  --dtype {float16,bfloat16,float32} [--shortconv-fix]   (HF checkpoint --hf)
  ts:    --backend {ggml_cpu,cpu} --gguf FILE      (this checkout's Release TensorSharp.Server.Host)
  llama: --gguf FILE                               (llama-server from LLAMA_BIN, default ~/work/llama.cpp-lfm3/build-cpu/bin)
"""
import argparse
import glob
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
VENV = os.environ.get("VLLM_VENV", os.path.expanduser("~/work/venvs/vllm-cpu"))
LLAMA_BIN = os.environ.get("LLAMA_BIN", os.path.expanduser("~/work/llama.cpp-lfm3/build-cpu/bin"))
MODEL_NAME = "lfm3"
# PYTHONPATH entry of the opt-in short-conv workaround for vLLM on macOS (see its sitecustomize.py); --shortconv-fix
SHORTCONV_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm-mac-shortconv")

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "Explain in two or three sentences why the sky is blue.",
    "List five prime numbers greater than 50 and say how you checked them.",
    "Translate into French and Spanish: 'Good morning, the train to the airport leaves in ten minutes.'",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def loadavg():
    try:
        return subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip(" {}\n")
    except Exception:
        return ""


# ---------------------------------------------------------------------------------------------- servers

def ts_dll():
    for p in [os.path.join(ROOT, "TensorSharp.Server.Host", "bin", "Release", "net10.0", "TensorSharp.Server.Host.dll"),
              os.path.join(ROOT, "TensorSharp.Server.Host", "bin", "TensorSharp.Server.Host.dll")]:
        if os.path.exists(p):
            return p
    raise SystemExit("TensorSharp.Server.Host.dll not found: dotnet build TensorSharp.Server.Host -c Release")


def vllm_env(env, a):
    # gloo resolves the host name at startup; on a Mac whose name does not resolve that waits ~7 minutes
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    if a.shortconv_fix:
        env["PYTHONPATH"] = SHORTCONV_FIX + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")


def server_command(a, concurrency, ctx):
    """(argv, env) of one server. Prefix/prompt caching is off on every engine; one slot per concurrent request."""
    env = dict(os.environ)
    if a.engine == "vllm":
        vllm_env(env, a)
        env["VLLM_CPU_KVCACHE_SPACE"] = str(a.kv_gib)
        # macOS has no thread affinity; vLLM's CPU worker takes its OpenMP width from the bind list's length
        # (OMP_NUM_THREADS = len(list)); torch's intra-op pool and vLLM's own kernels follow it.
        env["VLLM_CPU_OMP_THREADS_BIND"] = f"0-{a.threads - 1}"
        env.pop("OMP_NUM_THREADS", None)
        cmd = [os.path.join(VENV, "bin", "vllm"), "serve", a.hf, "--dtype", a.dtype, "--max-model-len", str(ctx),
               "--no-enable-prefix-caching", "--max-num-seqs", str(concurrency), "--host", "127.0.0.1",
               "--port", str(a.port), "--served-model-name", MODEL_NAME, "--generation-config", "vllm",
               "--disable-uvicorn-access-log"]
    elif a.engine == "ts":
        env["MAX_CONTEXT"] = str(ctx)
        env["TS_SCHED_MAX_RUNNING_SEQS"] = str(concurrency)
        env["TS_GGML_CPU_THREADS" if a.backend == "ggml_cpu" else "TS_CPU_THREADS"] = str(a.threads)
        cmd = ["dotnet", ts_dll(), "--model", a.gguf, "--backend", a.backend, "--host", "127.0.0.1",
               "--port", str(a.port), "--no-multi-agent", "--no-skills", "--no-webui", "--no-prefix-cache",
               "--max-tokens", str(ctx)]
    elif a.engine == "llama":
        cmd = [os.path.join(LLAMA_BIN, "llama-server"), "-m", a.gguf, "--host", "127.0.0.1", "--port", str(a.port),
               "-np", str(concurrency), "-cb", "-c", str(ctx * concurrency), "-fa", "on", "--jinja", "-ngl", "0",
               "-t", str(a.threads), "-tb", str(a.threads_batch or a.threads), "--no-cache-prompt", "-cram", "0",
               "--no-webui", "--alias", MODEL_NAME, "-b", "2048", "-ub", "512"]
    else:
        raise SystemExit("unknown engine " + a.engine)
    return cmd, env


def wait_ready(port, proc, seconds=900):
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
                # the id requests must name (TensorSharp serves only its own model's id)
                data = json.loads(body).get("data") or []
                return time.time() - t, (data[0]["id"] if data else MODEL_NAME)
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready")


def tree_pids(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True).stdout.split()
        except Exception:
            kids = []
        todo += [int(k) for k in kids]
    return out


def footprint_bytes(pids):
    """Sum of `footprint` (dirty + compressed + IOKit, the Activity Monitor 'Memory' figure) over the pids."""
    total = 0.0
    for pid in pids:
        try:
            txt = subprocess.run(["footprint", "-f", "bytes", str(pid)], capture_output=True, text=True, timeout=20).stdout
        except Exception:
            continue
        m = re.search(r"Footprint:\s*([0-9]+)\s*B", txt)
        if m:
            total += float(m.group(1))
    return total


def rss_bytes(pids):
    """Sum of resident set sizes (includes the clean, file-backed pages of mmapped weights, which footprint omits)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)], capture_output=True,
                             text=True, timeout=10).stdout
        return 1024.0 * sum(int(x) for x in out.split())
    except Exception:
        return 0.0


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.peak_rss, self.stop, self.nproc = pid, 0.0, 0.0, False, 0

    def run(self):
        while not self.stop:
            pids = tree_pids(self.pid)
            self.nproc = max(self.nproc, len(pids))
            self.peak = max(self.peak, footprint_bytes(pids))
            self.peak_rss = max(self.peak_rss, rss_bytes(pids))
            time.sleep(0.5)


class Server:
    def __init__(self, a, concurrency, ctx, logpath):
        self.cmd, env = server_command(a, concurrency, ctx)
        self.port = a.port
        self.log = open(logpath, "w")
        self.log.write("$ " + " ".join(self.cmd) + "\n")
        self.log.write("env: " + " ".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith(("VLLM_", "TS_", "MAX_CONTEXT", "OMP_"))) + "\n")
        self.log.flush()
        log("$ " + " ".join(self.cmd))
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.ready_s, self.model_id = wait_ready(self.port, self.proc)
        log(f"server ready in {self.ready_s:.0f}s (pid {self.proc.pid}, model id {self.model_id!r})")

    def stop(self):
        pids = tree_pids(self.proc.pid)
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                pass
        for p in pids:   # vLLM's engine core / workers can outlive the API server's group
            try:
                os.kill(p, signal.SIGKILL)
            except Exception:
                pass
        self.log.close()


def chat(port, body, timeout=1800):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:400]!r}")
    return json.loads(data)


def greedy_body(text, max_tokens, engine, model=MODEL_NAME):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1.0, "top_k": 1, "seed": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if engine == "vllm":
        body.update({"logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True})
    elif engine == "llama":
        body.update({"logprobs": True, "top_logprobs": 0, "cache_prompt": False, "repeat_penalty": 1.0})
    else:
        body.update({"repeat_penalty": 1.0})
    return body


# ---------------------------------------------------------------------------------------------- correctness

def cmd_check(a):
    import torch
    import vllm
    import vllm._C  # noqa: F401  the CPU kernels
    print("python", sys.version.split()[0], "torch", torch.__version__, "vllm", vllm.__version__)
    print("torch parallel:", " ".join(torch.__config__.parallel_info().split()[:12]))
    ops = [n for n in ("cpu_attention_with_kv_cache", "cpu_attn_reshape_and_cache", "rms_norm", "silu_and_mul")
           if hasattr(torch.ops._C, n)]
    print("vllm._C ops present:", ops)
    if len(ops) < 4:
        raise SystemExit("vllm._C lacks CPU ops")
    from vllm.platforms import current_platform
    print("platform:", current_platform.device_name, "supported dtypes:", current_platform.supported_dtypes)
    for f in ["config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"]:
        p = os.path.join(a.hf, f)
        print(f"{p}: {os.path.getsize(p) if os.path.exists(p) else 'MISSING'}")
    print("llama-server:", os.path.exists(os.path.join(LLAMA_BIN, "llama-server")), LLAMA_BIN)
    print("TensorSharp.Server.Host:", ts_dll())


def tokenizer(hf):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf, trust_remote_code=True)


def cmd_prompts(a):
    tok = tokenizer(a.hf)
    items = []
    for i, p in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        items.append({"i": i, "user": p, "text": text, "ids": ids})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"max_tokens": a.max_tokens, "prompts": items}, open(os.path.join(a.out, "prompts.json"), "w"), indent=1)
    print(items[0]["text"])
    print("prompt lengths:", [len(x["ids"]) for x in items])


def load_prompts(out):
    return json.load(open(os.path.join(out, "prompts.json")))


def save_gen(out, name, meta, results):
    json.dump({"name": name, "meta": meta, "results": results}, open(os.path.join(out, f"gen-{name}.json"), "w"), indent=1)
    for r in results:
        log(f"{name} #{r['i']}: {r['text'][:100]!r}")


def cmd_gen_vllm(a):
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(a.kv_gib)
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = f"0-{a.threads - 1}"
    vllm_env(os.environ, a)   # the engine core and worker are spawned processes: they read PYTHONPATH at start
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    P = load_prompts(a.out)
    t0 = time.time()
    llm = LLM(model=a.hf, dtype=a.dtype, max_model_len=2048, enable_prefix_caching=False, max_num_seqs=1,
              generation_config="vllm", seed=0)
    load_s = time.time() - t0
    sp = SamplingParams(temperature=0.0, max_tokens=P["max_tokens"])
    results = []
    t0 = time.time()
    for p in P["prompts"]:
        o = llm.generate([TokensPrompt(prompt_token_ids=p["ids"])], sp, use_tqdm=False)[0]
        c = o.outputs[0]
        results.append({"i": p["i"], "ids": list(c.token_ids), "text": c.text, "finish": c.finish_reason,
                        "prompt_tokens": len(o.prompt_token_ids), "exact_ids": True})
    save_gen(a.out, f"vllm-offline-{a.dtype}" + ("-fix" if a.shortconv_fix else ""),
             {"dtype": a.dtype, "threads": a.threads, "load_s": load_s, "gen_s": time.time() - t0,
              "shortconv_fix": a.shortconv_fix,
              "generated_tokens": sum(len(r["ids"]) for r in results)}, results)


def cmd_gen_hf(a):
    import torch
    from transformers import AutoModelForCausalLM
    P = load_prompts(a.out)
    torch.set_num_threads(a.threads)
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[a.dtype]
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=dt, trust_remote_code=True)
    model.eval()
    load_s = time.time() - t0
    tok = tokenizer(a.hf)
    results = []
    t0 = time.time()
    for p in P["prompts"][: a.count]:
        ids = torch.tensor([p["ids"]])
        if a.recompute:
            # Greedy by full forwards of the whole sequence each step (no cache): the reference that does not depend
            # on the remote code's incremental (conv / sliding-window) cache path.
            new = []
            with torch.no_grad():
                for _ in range(P["max_tokens"]):
                    nxt = int(model(torch.tensor([p["ids"] + new]), use_cache=False).logits[0, -1].argmax())
                    new.append(nxt)
                    if nxt == model.config.eos_token_id:
                        break
        else:
            with torch.no_grad():
                g = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=P["max_tokens"], eos_token_id=model.config.eos_token_id,
                                   pad_token_id=model.config.pad_token_id)
            new = g[0, ids.shape[1]:].tolist()
        results.append({"i": p["i"], "ids": new, "text": tok.decode(new, skip_special_tokens=True),
                        "finish": "stop" if new and new[-1] == model.config.eos_token_id else "length",
                        "prompt_tokens": ids.shape[1], "exact_ids": True})
    save_gen(a.out, f"hf-{a.dtype}" + ("-recompute" if a.recompute else ""),
             {"dtype": a.dtype, "threads": a.threads, "load_s": load_s, "gen_s": time.time() - t0,
              "transformers_remote_code": True, "recompute": a.recompute}, results)


def cmd_gen_server(a):
    P = load_prompts(a.out)
    tok = tokenizer(a.hf)
    srv = Server(a, 1, 4096, os.path.join(a.out, f"gen-{a.name}.server.log"))
    results = []
    try:
        for p in P["prompts"]:
            r = chat(a.port, greedy_body(p["user"], P["max_tokens"], a.engine, srv.model_id))
            ch = r["choices"][0]
            text = ch["message"].get("content") or ""
            reasoning = ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or ""
            ids, exact = None, False
            lp = (ch.get("logprobs") or {}).get("content") or []
            if a.engine == "vllm" and lp:
                ids = [int(e["token"].split(":", 1)[1]) for e in lp]
                exact = True
            elif a.engine == "llama" and lp and all("id" in e for e in lp):
                ids = [int(e["id"]) for e in lp]
                exact = True
            if ids is None:
                ids = tok(text, add_special_tokens=False)["input_ids"]
            # The server strips the end-of-turn token; the offline runs keep it.
            if ch.get("finish_reason") == "stop" and (not ids or ids[-1] != 124900) and not exact:
                ids = ids + [124900]
            results.append({"i": p["i"], "ids": ids, "text": text, "reasoning": reasoning, "finish": ch.get("finish_reason"),
                            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"), "exact_ids": exact})
    finally:
        srv.stop()
    save_gen(a.out, a.name, {"engine": a.engine, "cmd": srv.cmd, "threads": a.threads}, results)


def cmd_compare(a):
    P = load_prompts(a.out)
    gens = {}
    for f in sorted(glob.glob(os.path.join(a.out, "gen-*.json"))):
        g = json.load(open(f))
        gens[g["name"]] = {r["i"]: r for r in g["results"]}
    ref = a.ref if a.ref in gens else next(iter(gens))
    names = [n for n in gens if n != ref]
    lines = [f"Reference: `{ref}`. Each cell: tokens identical before the first divergence / reference length "
             f"(`=` all equal; `*` ids re-tokenized from the text, so a split can differ without a different text).", "",
             "| engine | prompt tokens | " + " | ".join(f"p{p['i']}" for p in P["prompts"]) + " | all equal |",
             "|---|---|" + "---|" * len(P["prompts"]) + "---|"]
    for n in [ref] + names:
        cells, eq = [], 0
        pt = sorted({str(gens[n][i].get("prompt_tokens")) for i in gens[n]})
        for p in P["prompts"]:
            r = gens[n].get(p["i"])
            rr = gens[ref].get(p["i"])
            if r is None or rr is None:
                cells.append("-")
                continue
            x, y = r["ids"], rr["ids"]
            k = 0
            while k < min(len(x), len(y)) and x[k] == y[k]:
                k += 1
            same = (x == y) or (r["text"].strip() == rr["text"].strip())
            eq += same
            star = "" if r.get("exact_ids") else "*"
            cells.append("=" if same else f"{k}/{len(y)}{star}")
        lines.append(f"| {n} | {','.join(pt)} | " + " | ".join(cells) + f" | {eq}/{len(gens[n])} |")
    lines += ["", "Outputs (first 160 characters):", ""]
    for p in P["prompts"]:
        lines.append(f"- p{p['i']} `{p['user'][:60]}`")
        for n in [ref] + names:
            r = gens[n].get(p["i"])
            if r is not None:
                lines.append(f"  - {n} ({r.get('finish')}): {json.dumps(r['text'][:160])}")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "compare.md"), "w").write(txt + "\n")
    print(txt)


def cmd_margins(a):
    """At each engine's first divergence from the reference, the HF float32 logits of the two competing tokens after
    the shared prefix: a gap of a few hundredths is a near tie (summation order decides it), a large gap is a bug."""
    import torch
    from transformers import AutoModelForCausalLM
    P = load_prompts(a.out)
    torch.set_num_threads(a.threads)
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32, trust_remote_code=True).eval()
    tok = tokenizer(a.hf)
    ref = {r["i"]: r for r in json.load(open(os.path.join(a.out, f"gen-{a.ref}.json")))["results"]}
    lines = [f"First divergence from `{a.ref}`: HF float32 logits after the shared prefix (reference token vs the "
             f"engine's token; gap = ref - engine; rank of the engine's token).", "",
             "| engine | prompt | position | reference token | engine token | gap | engine token rank |", "|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(os.path.join(a.out, "gen-*.json"))):
        g = json.load(open(f))
        if g["name"] == a.ref:
            continue
        for r in g["results"]:
            y = ref[r["i"]]["ids"]
            x = r["ids"]
            k = 0
            while k < min(len(x), len(y)) and x[k] == y[k]:
                k += 1
            if k >= min(len(x), len(y)):
                continue
            prompt = P["prompts"][r["i"]]["ids"]
            ids = torch.tensor([prompt + y[:k]])
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            gap = (logits[y[k]] - logits[x[k]]).item()
            rank = int((logits > logits[x[k]]).sum().item()) + 1
            lines.append(f"| {g['name']}{'' if r.get('exact_ids') else '*'} | p{r['i']} | {k} | {tok.decode([y[k]])!r} | "
                         f"{tok.decode([x[k]])!r} | {gap:.3f} | {rank} |")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "margins.md"), "w").write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------------------------- benchmark

def parse_cells(s):
    """'512x128@1,128x256@4' -> [(512, 128, 1), (128, 256, 4)]"""
    cells = []
    for c in s.split(","):
        io, conc = c.split("@")
        i, o = io.split("x")
        cells.append((int(i), int(o), int(conc)))
    return cells


def requests_for(a, conc):
    m = dict(kv.split(":") for kv in a.requests.split(",")) if a.requests else {}
    return int(m.get(str(conc), max(4, 2 * conc)))


def cmd_arm(a):
    os.makedirs(a.out, exist_ok=True)
    cells = parse_cells(a.cells)
    tag = f"{a.name}.r{a.round}"
    # One server per concurrency level (slots = concurrency on every engine), each serving that level's cells.
    for conc in sorted({c for _, _, c in cells}):
        srv = Server(a, conc, a.ctx, os.path.join(a.out, f"{tag}.c{conc}.server.log"))
        try:
            for k in range(2):   # warm: kernels, graphs, slots (not timed)
                chat(a.port, greedy_body(f"(warm {k}) Say hello in five words.", 8, a.engine, srv.model_id))
            for isl, osl, c in [x for x in cells if x[2] == conc]:
                n = requests_for(a, c)
                fn = f"{tag}.i{isl}o{osl}c{c}.json"
                sampler = Sampler(srv.proc.pid)
                sampler.start()
                la0 = loadavg()
                t0 = time.time()
                cmd = [os.path.join(VENV, "bin", "vllm"), "bench", "serve", "--backend", "openai-chat",
                       "--base-url", f"http://127.0.0.1:{a.port}", "--endpoint", "/v1/chat/completions",
                       "--model", srv.model_id, "--tokenizer", a.hf, "--trust-remote-code", "--dataset-name", "random",
                       "--random-input-len", str(isl), "--random-output-len", str(osl), "--random-range-ratio", "0",
                       "--num-prompts", str(n), "--max-concurrency", str(c), "--request-rate", "inf",
                       "--ignore-eos", "--temperature", "0", "--num-warmups", "0",
                       "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": False}, "top_k": 1,
                                                   "top_p": 1.0, "cache_prompt": False}),
                       "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99",
                       "--seed", str(a.seed + 7 * isl + 31 * c), "--disable-tqdm",
                       "--save-result", "--result-dir", a.out, "--result-filename", fn]
                log(f"{tag} i{isl} o{osl} c{c} n{n}")
                p = subprocess.run(cmd, capture_output=True, text=True)
                sampler.stop = True
                sampler.join()
                with open(os.path.join(a.out, fn + ".bench.log"), "w") as f:
                    f.write("$ " + " ".join(cmd) + "\n" + p.stdout + "\n" + p.stderr)
                if p.returncode != 0 or not os.path.exists(os.path.join(a.out, fn)):
                    log(f"bench failed rc={p.returncode}: {p.stderr[-800:]}")
                    continue
                r = json.load(open(os.path.join(a.out, fn)))
                r.update({"arm": a.name, "round": a.round, "engine": a.engine, "isl": isl, "osl": osl, "conc": c,
                          "requests": n, "threads": a.threads, "threads_batch": a.threads_batch,
                          "server_cmd": srv.cmd, "server_ready_s": srv.ready_s,
                          "peak_footprint_gb": sampler.peak / (1 << 30), "peak_rss_gb": sampler.peak_rss / (1 << 30),
                          "server_procs": sampler.nproc,
                          "loadavg_before": la0, "loadavg_after": loadavg(), "cell_wall_s": time.time() - t0})
                r.pop("itls", None)
                r.pop("generated_texts", None)
                json.dump(r, open(os.path.join(a.out, fn), "w"), indent=1)
                log(f"  ttft p50 {r.get('median_ttft_ms', 0):.0f} ms, tpot p50 {r.get('median_tpot_ms', 0):.1f} ms, "
                    f"out {r.get('output_throughput', 0):.1f} tok/s, ok {r.get('completed')}/{n}, "
                    f"peak {r['peak_footprint_gb']:.2f} GB, load {la0}")
        finally:
            srv.stop()


def med_range(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = statistics.median(vals)
    if len(vals) == 1:
        return fmt.format(m) + " (n=1)"
    return f"{fmt.format(m)} [{fmt.format(min(vals))}-{fmt.format(max(vals))}]"


def cmd_summary(a):
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.out, "*.r*.i*o*c*.json"))):
        r = json.load(open(f))
        if "arm" not in r:
            continue
        rows.setdefault((r["isl"], r["osl"], r["conc"]), {}).setdefault(r["arm"], []).append(r)
    order = a.arms.split(",") if a.arms else None
    out = []
    for (isl, osl, c) in sorted(rows, key=lambda k: (k[2], -k[0])):
        arms = rows[(isl, osl, c)]
        names = [n for n in (order or []) if n in arms] + sorted(n for n in arms if n not in (order or []))
        out += [f"### ISL {isl} / OSL {osl}, concurrency {c}", "",
                "| engine | rounds x requests | prompt tok (mean) | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | "
                "output tok/s | req/s | peak footprint GB | peak RSS GB | ok | load (1-min, before) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            rs = sorted(arms[n], key=lambda r: r["round"])
            p99 = lambda r, k: dict((float(p), v) for p, v in r.get(f"percentiles_{k}_ms", [])).get(99.0, r.get(f"p99_{k}_ms"))
            ok = f"{sum(r.get('completed', 0) for r in rs)}/{sum(r['requests'] for r in rs)}"
            loads = ",".join(r["loadavg_before"].split()[0] for r in rs if r.get("loadavg_before"))
            out.append(
                f"| {n} | {len(rs)} x {rs[0]['requests']} | {statistics.mean([r['total_input_tokens'] / max(1, r.get('completed', 1)) for r in rs]):.0f} | "
                f"{med_range([r.get('median_ttft_ms') for r in rs], '{:.0f}')} | {med_range([p99(r, 'ttft') for r in rs], '{:.0f}')} | "
                f"{med_range([r.get('median_tpot_ms') for r in rs], '{:.1f}')} | {med_range([p99(r, 'tpot') for r in rs], '{:.1f}')} | "
                f"{med_range([r.get('output_throughput') for r in rs], '{:.1f}')} | {med_range([r.get('request_throughput') for r in rs], '{:.3f}')} | "
                f"{med_range([r.get('peak_footprint_gb') for r in rs], '{:.2f}')} | "
                f"{med_range([r.get('peak_rss_gb') for r in rs], '{:.2f}')} | {ok} | {loads} |")
        out.append("")
    # At a glance: per concurrency, each shape's TTFT p50 / TPOT p50 / output tok/s (medians over rounds).
    for c in sorted({k[2] for k in rows}):
        shapes = sorted([k for k in rows if k[2] == c], key=lambda k: -k[0])
        present = {n for k in shapes for n in rows[k]}
        names = [n for n in (order or []) if n in present] + sorted(n for n in present if n not in (order or []))
        out += [f"### At a glance, concurrency {c}: TTFT p50 ms / TPOT p50 ms / output tok/s (medians over rounds)", "",
                "| engine | " + " | ".join(f"ISL {k[0]} / OSL {k[1]}" for k in shapes) + " |",
                "|---|" + "---|" * len(shapes)]
        for n in names:
            cells = []
            for k in shapes:
                rs = rows[k].get(n)
                if not rs:
                    cells.append("-")
                    continue
                med = lambda key: statistics.median([r[key] for r in rs if r.get(key) is not None])
                cells.append(f"{med('median_ttft_ms'):.0f} / {med('median_tpot_ms'):.1f} / {med('output_throughput'):.1f}")
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append("")
    txt = "\n".join(out)
    open(os.path.join(a.out, "summary.md"), "w").write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "prompts", "gen-vllm", "gen-hf", "gen-server", "compare", "margins", "arm",
                                    "summary"])
    ap.add_argument("--hf", default=os.path.expanduser("~/work/models/lfm3/hf-1.7b-full"))
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "lfm3", "cpu", "vllm"))
    ap.add_argument("--engine", choices=["vllm", "ts", "llama"])
    ap.add_argument("--name")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--backend", default="ggml_cpu")
    ap.add_argument("--gguf")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--threads-batch", type=int, default=0)
    ap.add_argument("--kv-gib", type=int, default=2)
    ap.add_argument("--shortconv-fix", action="store_true",
                    help="vllm: load vllm-mac-shortconv/sitecustomize.py (the macOS short-conv workaround)")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--count", type=int, default=5, help="gen-hf: prompts to run")
    ap.add_argument("--recompute", action="store_true", help="gen-hf: full forward per token instead of generate()")
    ap.add_argument("--ref", default="hf-float32-recompute")
    ap.add_argument("--cells", default="512x128@1,128x256@1,512x128@4,128x256@4")
    ap.add_argument("--requests", default="", help="per concurrency, e.g. '1:4,4:8' (default max(4, 2C))")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arms", default="", help="summary: arm order")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"check": cmd_check, "prompts": cmd_prompts, "gen-vllm": cmd_gen_vllm, "gen-hf": cmd_gen_hf,
     "gen-server": cmd_gen_server, "compare": cmd_compare, "margins": cmd_margins, "arm": cmd_arm,
     "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
