#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Regenerate the llama.cpp goldens that InferenceWeb.Tests/MuseGlimmerParityTests.cs checks Muse-Glimmer against.

The script starts a llama-server on the given GGUF, asks it for greedy continuations of a fixed prompt set, records the
prompt tokens, the generated tokens and the top log-probabilities of every generated token, and writes
InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/{ref_text,ref_text_long,ref_text_dflash}.json. The prompts are
rendered here, by the same rules as the GGUF's chat template but without its date line (the template prints
strftime_now when the renderer defines it), so a regeneration produces the same prompt text on any day.

    python3 eng/validation/muse-glimmer-llamacpp-goldens.py \
        --llama-cpp ~/work/llama.cpp-lfm3 \
        --model ~/work/models/muse-glimmer-30b/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
        --draft ~/work/models/muse-glimmer-30b/dflash-kquant.gguf \
        --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp

One set per llama.cpp backend, because the ggml backends round differently (the parity tests compare each lane with the
set of its own ggml backend): the Metal set above is in LlamaCpp/, the CUDA and Vulkan sets in LlamaCpp/cuda/ and
LlamaCpp/vulkan/. --build-dir names the llama.cpp build directory (default build) and --backend the label written into
the provenance (default metal):

    python3 eng/validation/muse-glimmer-llamacpp-goldens.py --llama-cpp ~/llama.cpp --build-dir build-vk --backend vulkan \
        --model .../Muse-Glimmer-30B-UD-IQ2_XXS.gguf --draft .../dflash-kquant.gguf --out InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/vulkan

Only the standard library is used. Run it under the Mac job lock and memguard like any real-weight run.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# The short prompt set: English prose, the Chinese prompt that looped in September 2026, code, arithmetic, verse.
PROMPTS = [
    ("refrigerator", "Explain how a refrigerator keeps food cold, in three short paragraphs."),
    ("ff7", "请详细介绍最终幻想7"),
    ("fibonacci", "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring."),
    ("train", "A train travels 180 km in 2 hours and then 120 km in 1.5 hours. What is its average speed for the whole "
              "trip? Show your reasoning."),
    ("haiku", "Write a haiku about the ocean at night."),
]

# The long prompt: the history-of-computing text (artifacts/glimmer-baseline/prompts/p_4k.txt's body, kept in the repo as
# benchmarks/engine_comparison/assets texts) extended until the prompt passes LONG_MIN_TOKENS, which is more than twice the
# 2048-token sliding window and more than legacy's 4352-row ring, so the window, the ring wrap and a KV grow are all crossed.
LONG_MIN_TOKENS = 4650
LONG_QUESTION = ("Summarize the text above in five bullet points, then name the three inventions it treats as the most "
                 "important and explain why.")

N_PREDICT = 128
N_PROBS = 10        # top log-probabilities requested per generated token
KEEP_FIRST = 10     # kept for the first generated token
KEEP_REST = 4       # kept for every later one


def render(user_text):
    """The GGUF chat template's rendering of one user turn with the generation prompt, minus the date line (and minus the
    bos_token text: /tokenize with add_special adds the BOS the vocabulary asks for)."""
    system = ("<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\n\n"
              "Reasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|>")
    return system + "<|start|>user<|message|>" + user_text + "<|eot|><|start|>assistant"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def known_sha(path):
    """The sha256 recorded next to a verified download (<file>.done), else computed."""
    done = path + ".done"
    if os.path.exists(done):
        with open(done) as f:
            return f.read().split()[0]
    return sha256(path)


def post(port, route, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, binary, args, port, log):
        self.port = port
        self.cmd = [binary, "--port", str(port), "--host", "127.0.0.1"] + args
        self.log = open(log, "w")
        self.proc = None

    def __enter__(self):
        print("[goldens] " + " ".join(self.cmd), flush=True)
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} (see {self.log.name})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except (urllib.error.URLError, ConnectionError, json.JSONDecodeError, TimeoutError):
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy in 600 s")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def tokenize(port, text):
    return post(port, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]


def complete(port, tokens, n_probs):
    body = {
        "prompt": tokens, "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
        "cache_prompt": False, "return_tokens": True, "n_probs": n_probs, "post_sampling_probs": False,
        "samplers": ["top_k"],
    }
    return post(port, "/completion", body)


def record(name, user_text, prompt, tokens, res, with_probs):
    out = {
        "name": name,
        "user": user_text,
        "prompt": prompt,
        "prompt_tokens": tokens,
        "generated_tokens": res["tokens"],
        "content": res["content"],
        "stop_type": res.get("stop_type"),
        "timings": {k: res.get("timings", {}).get(k) for k in
                    ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "draft_n", "draft_n_accepted")},
    }
    if with_probs:
        steps = []
        probs = res.get("completion_probabilities") or []
        if len(probs) != len(res["tokens"]):
            raise RuntimeError(f"{name}: {len(probs)} probability entries for {len(res['tokens'])} tokens")
        for i, p in enumerate(probs):
            if p["id"] != res["tokens"][i]:
                raise RuntimeError(f"{name} step {i}: the probabilities are for token {p['id']}, the output is {res['tokens'][i]}")
            keep = KEEP_FIRST if i == 0 else KEEP_REST
            top = [[t["id"], round(t["logprob"], 6)] for t in p["top_logprobs"][:keep]]
            if top and top[0][0] != p["id"]:
                raise RuntimeError(f"{name} step {i}: greedy token {p['id']} is not the top of its own distribution {top[0]}")
            steps.append({"logprob": round(p["logprob"], 6), "top": top})
        out["steps"] = steps
    return out


def long_user_text(port):
    """The engine-comparison corpus, the long text, then the corpus again, paragraph by paragraph until the rendered
    prompt reaches LONG_MIN_TOKENS (the legacy baseline's 4K prompt was built from the same texts), then the question."""
    assets = os.path.join(REPO, "benchmarks", "engine_comparison", "assets")
    paragraphs = []
    for name in ("prefill_corpus.txt", "long_text.txt", "prefill_corpus.txt"):
        with open(os.path.join(assets, name), encoding="utf-8") as f:
            paragraphs += [p.strip() for p in f.read().replace("\r\n", "\n").split("\n\n") if p.strip()]
    text, n = "", 0
    for paragraph in paragraphs:
        text = paragraph if not text else text + "\n\n" + paragraph
        user = text + "\n\n" + LONG_QUESTION
        n = len(tokenize(port, render(user)))
        if n >= LONG_MIN_TOKENS:
            return user, n
    raise RuntimeError(f"the long text reaches only {n} tokens")


def portable(args):
    """Server arguments with file paths reduced to their names (the fixtures name files, not machines)."""
    return [os.path.basename(x) if os.path.isabs(x) else x for x in args]


def write(path, payload):
    """The provenance one key a line, each record compact on a line of its own (the files stay small and diff by record)."""
    items = list(payload.items())
    lines = ["{"]
    for i, (key, value) in enumerate(items):
        if key == "records":
            body = ",\n".join("  " + json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in value)
            text = "[\n" + body + "\n ]"
        else:
            text = json.dumps(value, ensure_ascii=False)
        lines.append(f" {json.dumps(key)}: {text}" + ("," if i < len(items) - 1 else ""))
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[goldens] wrote {path} ({os.path.getsize(path)} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout with a built build/bin/llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", help="dflash GGUF: also write ref_text_dflash.json (llama.cpp's own DFlash path)")
    ap.add_argument("--out", default=os.path.join(REPO, "InferenceWeb.Tests", "Fixtures", "MuseGlimmer", "LlamaCpp"))
    ap.add_argument("--log-dir", default=os.path.join(REPO, "artifacts", "glimmer-goldens"))
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--build-dir", default="build", help="llama.cpp build directory under --llama-cpp (default build)")
    ap.add_argument("--backend", default="metal", help="the llama.cpp build's ggml backend, recorded in the provenance (default metal)")
    a = ap.parse_args()

    binary = os.path.join(a.llama_cpp, a.build_dir, "bin", "llama-server")
    commit = subprocess.check_output(["git", "-C", a.llama_cpp, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", a.llama_cpp, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    version = subprocess.run([binary, "--version"], capture_output=True, text=True).stderr.strip().splitlines()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)

    base_args = ["-m", a.model, "-c", "8192", "-ngl", "99", "-fa", "on", "-np", "1", "--no-webui"]
    provenance = {
        "llama_cpp": {"commit": commit, "dirty": bool(dirty), "version": version, "backend": a.backend,
                      "server_args": portable(base_args[2:])},
        "model": {"file": os.path.basename(a.model), "sha256": known_sha(a.model),
                  "source": "unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39"},
        "sampling": "greedy (temperature 0, top_k 1); logprobs are log-softmax over the whole vocabulary of the raw logits "
                    "(post_sampling_probs false), after the model's logit scale and softcap",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eng/validation/muse-glimmer-llamacpp-goldens.py",
    }

    short, long_rec = [], None
    with Server(binary, base_args, a.port, os.path.join(a.log_dir, "llama-server.log")):
        for name, text in PROMPTS:
            prompt = render(text)
            tokens = tokenize(a.port, prompt)
            res = complete(a.port, tokens, N_PROBS)
            short.append(record(name, text, prompt, tokens, res, with_probs=True))
            print(f"[goldens] {name}: {len(tokens)} prompt tokens, {len(res['tokens'])} generated, stop {res.get('stop_type')}",
                  flush=True)
        user, n = long_user_text(a.port)
        prompt = render(user)
        tokens = tokenize(a.port, prompt)
        res = complete(a.port, tokens, N_PROBS)
        long_rec = record("history-of-computing", user, prompt, tokens, res, with_probs=True)
        print(f"[goldens] long: {len(tokens)} prompt tokens, {len(res['tokens'])} generated", flush=True)

    write(os.path.join(a.out, "ref_text.json"), dict(provenance, records=short))
    write(os.path.join(a.out, "ref_text_long.json"), dict(provenance, records=[long_rec]))

    if a.draft:
        dargs = base_args + ["-md", a.draft, "-ngld", "99", "--spec-type", "draft-dflash", "--spec-draft-n-max", "15"]
        dflash = []
        with Server(binary, dargs, a.port, os.path.join(a.log_dir, "llama-server-dflash.log")):
            for rec in short:
                res = complete(a.port, rec["prompt_tokens"], 0)
                dflash.append(record(rec["name"], rec["user"], rec["prompt"], rec["prompt_tokens"], res, with_probs=False))
                same = res["tokens"] == rec["generated_tokens"]
                t = res.get("timings", {})
                print(f"[goldens] dflash {rec['name']}: identical={same} drafted={t.get('draft_n')} "
                      f"accepted={t.get('draft_n_accepted')}", flush=True)
        payload = dict(provenance, records=dflash)
        payload["llama_cpp"] = dict(provenance["llama_cpp"], server_args=portable(dargs[2:]))
        payload["draft"] = {"file": os.path.basename(a.draft), "sha256": known_sha(a.draft)}
        write(os.path.join(a.out, "ref_text_dflash.json"), payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
