#!/usr/bin/env python3
"""
Inference benchmark matrix driver for TensorSharp / llama.cpp / Ollama.

Runs the same (model, task) combinations across engines, parses
their stdout for prefill/decode timings, and writes a JSON file per
(engine, model, task) into the results directory.

TensorSharp is tested with multiple KV cache data types (f32, f16, q8_0)
to measure the performance impact of KV cache precision.

Usage:
    python3 run_bench.py [--engines tensorsharp_f32,tensorsharp_f16,tensorsharp_q8,llamacpp,ollama]
                        [--models gemma4]
                        [--tasks pp512,tg128,...]
                        [--quick]            # smaller token counts for smoke test
                        [--results <dir>]    # default: ../results
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/Users/ZhongkaiFu/work/TensorSharp")
MODEL_DIR = Path("/Users/ZhongkaiFu/work/model")
DATA_DIR = REPO / "data"
TENSORSHARP_BIN = REPO / "TensorSharp.Cli" / "bin" / "TensorSharp.Cli"

OLLAMA_API = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    short_id: str
    display: str
    gguf: Path
    mmproj: Optional[Path]
    ollama_tag: str
    family: str
    supports_image: bool
    supports_audio: bool
    supports_video: bool

MODELS: dict[str, ModelSpec] = {
    "gemma4": ModelSpec(
        short_id="gemma4",
        display="Gemma 4 E4B (Q8_0, 7.6 GiB)",
        gguf=MODEL_DIR / "gemma-4-E4B-it-Q8_0.gguf",
        mmproj=MODEL_DIR / "gemma-4-mmproj-F16.gguf",
        ollama_tag="ts-gemma4-e4b-q8",
        family="gemma4",
        supports_image=True,
        supports_audio=True,
        supports_video=True,
    ),
}

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    short_id: str
    kind: str  # "synthetic" | "text" | "image" | "audio" | "video"
    description: str
    pp: int = 0
    tg: int = 0
    prompt_file: Optional[str] = None
    max_tokens: int = 64

TASKS: dict[str, TaskSpec] = {
    "pp512": TaskSpec("pp512", "synthetic", "Prefill 512 synthetic tokens", pp=512, tg=0),
    "tg128": TaskSpec("tg128", "synthetic", "Decode 128 tokens after 32-token prefill", pp=32, tg=128),
    "pp2048": TaskSpec("pp2048", "synthetic", "Prefill 2048 synthetic tokens (long context)", pp=2048, tg=0),
    "short_text": TaskSpec("short_text", "text", "Short text prompt (19 tokens) -> 64 generated",
                           prompt_file="short_text.txt", max_tokens=64),
    "long_text": TaskSpec("long_text", "text", "Long text prompt (~1043 tokens) -> 64 generated",
                          prompt_file="long_text.txt", max_tokens=64),
    "image": TaskSpec("image", "image", "Image (apple.png) + question -> 64 generated",
                      prompt_file="image_question.txt", max_tokens=64),
    "audio": TaskSpec("audio", "audio", "Audio (45 s of speech) + question -> 64 generated",
                      prompt_file="audio_question.txt", max_tokens=64),
    "video": TaskSpec("video", "video", "Video (concert clip) + question -> 64 generated",
                      prompt_file="video_question.txt", max_tokens=64),
}

QUICK_OVERRIDES = {
    "pp512": dict(pp=128),
    "tg128": dict(tg=32),
    "pp2048": dict(pp=512),
    "short_text": dict(max_tokens=16),
    "long_text": dict(max_tokens=16),
    "image": dict(max_tokens=16),
    "audio": dict(max_tokens=16),
    "video": dict(max_tokens=16),
}

# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    engine: str
    model: str
    task: str
    ok: bool
    error: str = ""
    prefill_tokens: int = 0
    prefill_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tokens: int = 0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    total_wall_ms: float = 0.0
    model_load_ms: float = 0.0
    output_text: str = ""
    cmd: str = ""
    raw_stdout_tail: str = ""
    kv_cache_dtype: str = ""

# ---------------------------------------------------------------------------
# Shell helper
# ---------------------------------------------------------------------------
def run_cmd(cmd: list[str], timeout_s: float, env: Optional[dict] = None) -> tuple[int, str, str, float]:
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        return proc.returncode, proc.stdout, proc.stderr, elapsed_ms
    except subprocess.TimeoutExpired as ex:
        elapsed_ms = (time.monotonic() - t0) * 1000
        return 124, ex.stdout or "", (ex.stderr or "") + f"\nTIMEOUT after {timeout_s}s", elapsed_ms

# ---------------------------------------------------------------------------
# TensorSharp runners
# ---------------------------------------------------------------------------
TS_BENCH_SUMMARY_RE = re.compile(
    r"benchmark summary:\s*"
    r"bestPrefillMs=(?P<pms>[0-9.]+)\s*"
    r"bestPrefillTps=(?P<pps>[0-9.]+)\s*"
    r"bestDecodeMs=(?P<dms>[0-9.]+)\s*"
    r"bestDecodeTps=(?P<dps>[0-9.]+)"
)
TS_LOAD_RE = re.compile(
    r"Loaded model .*? elapsedMs=(?P<load_ms>[0-9.]+)"
)
TS_INFER_PREFILL_RE = re.compile(
    r"prefill complete:\s*tokens=(?P<tok>\d+)\s*ms=(?P<ms>[0-9.]+)\s*tokensPerSec=(?P<tps>[0-9.]+)"
)
TS_INFER_DECODE_RE = re.compile(
    r"decode complete:\s*tokens=(?P<tok>\d+)\s*ms=(?P<ms>[0-9.]+)\s*tokensPerSec=(?P<tps>[0-9.]+)"
)


def run_tensorsharp(model: ModelSpec, task: TaskSpec, results_dir: Path,
                    kv_cache_dtype: str = "f32") -> BenchResult:
    engine_id = f"tensorsharp_{kv_cache_dtype.replace('_', '')}"
    res = BenchResult(engine=engine_id, model=model.short_id, task=task.short_id,
                      ok=False, kv_cache_dtype=kv_cache_dtype)

    if task.kind == "synthetic":
        runs = 3
        cmd = [
            str(TENSORSHARP_BIN),
            "--model", str(model.gguf),
            "--backend", "ggml_metal",
            "--benchmark",
            "--bench-prefill", str(task.pp if task.pp else 32),
            "--bench-decode", str(task.tg if task.tg else 1),
            "--bench-runs", str(runs),
            "--kv-cache-dtype", kv_cache_dtype,
            "--log-level", "info",
            "--log-file", "off",
        ]
    else:
        prompt_path = ROOT / "prompts" / task.prompt_file
        cmd = [
            str(TENSORSHARP_BIN),
            "--model", str(model.gguf),
            "--backend", "ggml_metal",
            "--input", str(prompt_path),
            "--max-tokens", str(task.max_tokens),
            "--temperature", "0",
            "--warmup-runs", "1",
            "--kv-cache-dtype", kv_cache_dtype,
            "--log-level", "info",
            "--log-file", "off",
        ]
        if task.kind == "image":
            cmd += ["--image", str(DATA_DIR / "apple.png")]
            if model.mmproj is not None:
                cmd += ["--mmproj", str(model.mmproj)]
        elif task.kind == "audio":
            cmd += ["--audio", str(DATA_DIR / "obama_first_45_secs.mp3")]
            if model.mmproj is not None:
                cmd += ["--mmproj", str(model.mmproj)]
        elif task.kind == "video":
            cmd += ["--video", str(DATA_DIR / "concert.mp4")]
            if model.mmproj is not None:
                cmd += ["--mmproj", str(model.mmproj)]

    res.cmd = " ".join(cmd)
    rc, out, err, wall_ms = run_cmd(cmd, timeout_s=900)
    res.total_wall_ms = wall_ms
    combined = (err or "") + "\n" + (out or "")
    res.raw_stdout_tail = "\n".join(combined.splitlines()[-30:])

    if rc != 0:
        res.error = f"exit {rc}"
        return res

    m = TS_LOAD_RE.search(combined)
    if m:
        res.model_load_ms = float(m.group("load_ms"))

    if task.kind == "synthetic":
        m = TS_BENCH_SUMMARY_RE.search(combined)
        if not m:
            res.error = "could not parse benchmark summary"
            return res
        if task.pp:
            res.prefill_tokens = task.pp
            res.prefill_ms = float(m.group("pms"))
            res.prefill_tps = float(m.group("pps"))
        if task.tg:
            res.decode_tokens = task.tg
            res.decode_ms = float(m.group("dms"))
            res.decode_tps = float(m.group("dps"))
        res.ok = True
    else:
        m = TS_INFER_PREFILL_RE.search(combined)
        if m:
            res.prefill_tokens = int(m.group("tok"))
            res.prefill_ms = float(m.group("ms"))
            res.prefill_tps = float(m.group("tps"))
        m = TS_INFER_DECODE_RE.search(combined)
        if m:
            res.decode_tokens = int(m.group("tok"))
            res.decode_ms = float(m.group("ms"))
            res.decode_tps = float(m.group("tps"))
        if res.prefill_ms == 0.0 and res.decode_ms == 0.0:
            res.error = "could not parse prefill/decode metrics"
            return res
        res.ok = True

    return res

# ---------------------------------------------------------------------------
# llama.cpp runners
# ---------------------------------------------------------------------------
LLAMA_BENCH_ROW_RE = re.compile(
    r"^\|.*?\|\s*[\d.]+\s*GiB\s*\|.*?\|.*?\|.*?\|\s*(?P<test>\w+)\s*\|\s*(?P<tps>[0-9.]+)\s*"
)


def parse_llama_bench(out: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in out.splitlines():
        m = LLAMA_BENCH_ROW_RE.match(line)
        if m:
            metrics[m.group("test")] = float(m.group("tps"))
    return metrics


def run_llamacpp(model: ModelSpec, task: TaskSpec, results_dir: Path) -> BenchResult:
    res = BenchResult(engine="llamacpp", model=model.short_id, task=task.short_id, ok=False)

    if task.kind == "synthetic":
        pp = task.pp if task.pp else 0
        tg = task.tg if task.tg else 0
        cmd = [
            "/opt/homebrew/bin/llama-bench",
            "-m", str(model.gguf),
            "-p", str(pp),
            "-n", str(tg),
            "-r", "3",
            "-o", "md",
        ]
        res.cmd = " ".join(cmd)
        rc, out, err, wall_ms = run_cmd(cmd, timeout_s=900)
        res.total_wall_ms = wall_ms
        combined = (out or "") + "\n" + (err or "")
        res.raw_stdout_tail = "\n".join(combined.splitlines()[-20:])
        if rc != 0:
            res.error = f"exit {rc}"
            return res

        metrics = parse_llama_bench(out)
        if not metrics:
            res.error = "could not parse llama-bench output"
            return res

        if pp and f"pp{pp}" in metrics:
            res.prefill_tokens = pp
            res.prefill_tps = metrics[f"pp{pp}"]
            res.prefill_ms = (pp / res.prefill_tps) * 1000 if res.prefill_tps else 0
        if tg and f"tg{tg}" in metrics:
            res.decode_tokens = tg
            res.decode_tps = metrics[f"tg{tg}"]
            res.decode_ms = (tg / res.decode_tps) * 1000 if res.decode_tps else 0
        res.ok = True
        return res

    prompt_path = ROOT / "prompts" / task.prompt_file
    if task.kind == "text":
        cmd = [
            "/opt/homebrew/bin/llama-cli",
            "-m", str(model.gguf),
            "-f", str(prompt_path),
            "-n", str(task.max_tokens),
            "--temp", "0",
            "-st",
            "--no-warmup",
            "--no-display-prompt",
            "-ngl", "999",
            "--jinja",
        ]
    elif task.kind == "image" and model.mmproj is not None:
        cmd = [
            "/opt/homebrew/bin/llama-mtmd-cli",
            "-m", str(model.gguf),
            "--mmproj", str(model.mmproj),
            "--image", str(DATA_DIR / "apple.png"),
            "-p", prompt_path.read_text(),
            "-n", str(task.max_tokens),
            "--temp", "0",
            "-ngl", "999",
            "--jinja",
            "--no-warmup",
        ]
    elif task.kind == "audio" and model.mmproj is not None:
        cmd = [
            "/opt/homebrew/bin/llama-mtmd-cli",
            "-m", str(model.gguf),
            "--mmproj", str(model.mmproj),
            "--audio", str(DATA_DIR / "obama_first_45_secs.mp3"),
            "-p", prompt_path.read_text(),
            "-n", str(task.max_tokens),
            "--temp", "0",
            "-ngl", "999",
            "--jinja",
            "--no-warmup",
        ]
    elif task.kind == "video":
        res.error = "llama.cpp has no video input path"
        res.ok = False
        return res
    else:
        res.error = f"unsupported task kind {task.kind} for llama.cpp"
        return res

    res.cmd = " ".join(cmd)
    rc, out, err, wall_ms = run_cmd(cmd, timeout_s=900)
    res.total_wall_ms = wall_ms
    combined = (out or "") + "\n" + (err or "")
    res.raw_stdout_tail = "\n".join(combined.splitlines()[-40:])
    if rc != 0:
        res.error = f"exit {rc}"
        return res

    pe = re.search(
        r"llama_perf_context_print:\s*prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second",
        combined,
    )
    de = re.search(
        r"llama_perf_context_print:\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([0-9.]+)\s*tokens per second",
        combined,
    )
    if pe:
        res.prefill_ms = float(pe.group(1))
        res.prefill_tokens = int(pe.group(2))
        res.prefill_tps = float(pe.group(3))
    if de:
        res.decode_ms = float(de.group(1))
        res.decode_tokens = int(de.group(2))
        res.decode_tps = float(de.group(3))

    if res.prefill_ms == 0 and res.decode_ms == 0:
        m = re.search(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]", combined)
        if m:
            res.prefill_tps = float(m.group(1))
            res.decode_tps = float(m.group(2))
            res.decode_tokens = task.max_tokens
            if res.prefill_tps > 0:
                res.prefill_ms = -1
            if res.decode_tps > 0:
                res.decode_ms = -1
            res.ok = True
            return res
        res.error = "could not parse llama-cli timings"
        return res

    res.ok = True
    return res

# ---------------------------------------------------------------------------
# Ollama runner (HTTP API)
# ---------------------------------------------------------------------------
def ollama_request(path: str, body: dict, timeout_s: float = 900) -> dict:
    req = urllib.request.Request(
        OLLAMA_API + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_warmup(tag: str):
    try:
        ollama_request("/api/generate", {
            "model": tag,
            "prompt": "ping",
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 1, "temperature": 0},
        }, timeout_s=600)
    except Exception:
        pass


def ollama_unload_all():
    """Force-unload every model the Ollama daemon currently has resident."""
    try:
        with urllib.request.urlopen(OLLAMA_API + "/api/ps", timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        loaded = data.get("models", []) or []
    except Exception:
        return
    for m in loaded:
        name = m.get("name") or m.get("model")
        if not name:
            continue
        try:
            ollama_request("/api/generate", {
                "model": name,
                "prompt": "",
                "keep_alive": 0,
                "stream": False,
            }, timeout_s=30)
        except Exception:
            pass
    time.sleep(2.0)


def run_ollama(model: ModelSpec, task: TaskSpec, results_dir: Path) -> BenchResult:
    res = BenchResult(engine="ollama", model=model.short_id, task=task.short_id, ok=False)

    if task.kind == "synthetic":
        if task.tg and not task.pp:
            prompt = (
                "Continue the following story for as long as you can without stopping. "
                "Be very detailed. The story so far: Once upon a time, in a faraway land,"
            )
            num_predict = task.tg
        elif task.tg and task.pp:
            prompt = (
                "Continue the following story for as long as you can without stopping. "
                "Be very detailed. The story so far: Once upon a time, in a faraway land,"
            )
            num_predict = task.tg
        else:
            pp = task.pp if task.pp else 32
            word = "Lorem "
            prompt = word * max(1, pp // 2)
            num_predict = 1

        body = {
            "model": model.ollama_tag,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "keep_alive": "10m",
            "options": {
                "num_predict": num_predict,
                "temperature": 0,
                "num_ctx": 4096,
                "stop": [],
            },
        }
    else:
        prompt_path = ROOT / "prompts" / task.prompt_file
        prompt_text = prompt_path.read_text()
        body = {
            "model": model.ollama_tag,
            "prompt": prompt_text,
            "stream": False,
            "raw": True,
            "keep_alive": "10m",
            "options": {
                "num_predict": task.max_tokens,
                "temperature": 0,
                "num_ctx": 4096,
            },
        }
        if task.kind == "image":
            import base64
            data = (DATA_DIR / "apple.png").read_bytes()
            body["images"] = [base64.b64encode(data).decode("ascii")]
        elif task.kind == "audio":
            res.error = "Ollama API has no audio input"
            return res
        elif task.kind == "video":
            res.error = "Ollama API has no video input"
            return res

    res.cmd = f"POST /api/generate model={model.ollama_tag} task={task.short_id}"
    t0 = time.monotonic()
    try:
        resp = ollama_request("/api/generate", body, timeout_s=900)
    except Exception as ex:
        res.error = f"ollama request failed: {ex}"
        return res
    res.total_wall_ms = (time.monotonic() - t0) * 1000

    if "prompt_eval_count" in resp and "prompt_eval_duration" in resp:
        res.prefill_tokens = int(resp["prompt_eval_count"])
        res.prefill_ms = float(resp["prompt_eval_duration"]) / 1e6
        if res.prefill_ms > 0:
            res.prefill_tps = res.prefill_tokens / (res.prefill_ms / 1000.0)
    if "eval_count" in resp and "eval_duration" in resp:
        res.decode_tokens = int(resp["eval_count"])
        res.decode_ms = float(resp["eval_duration"]) / 1e6
        if res.decode_ms > 0:
            res.decode_tps = res.decode_tokens / (res.decode_ms / 1000.0)
    res.output_text = (resp.get("response") or "").strip()[:300]
    res.raw_stdout_tail = json.dumps({k: resp.get(k) for k in (
        "model", "done_reason",
        "total_duration", "load_duration",
        "prompt_eval_count", "prompt_eval_duration",
        "eval_count", "eval_duration"
    )}, indent=2)
    if "load_duration" in resp:
        res.model_load_ms = float(resp["load_duration"]) / 1e6
    res.ok = res.prefill_ms > 0 or res.decode_ms > 0
    if not res.ok:
        res.error = "ollama returned no timing"
    return res

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------
# TensorSharp KV cache dtype variants are handled by wrapping run_tensorsharp
# with the appropriate dtype argument.

def _make_ts_runner(kv_dtype: str):
    def runner(model, task, results_dir):
        return run_tensorsharp(model, task, results_dir, kv_cache_dtype=kv_dtype)
    return runner

ENGINES = {
    "tensorsharp_f32": _make_ts_runner("f32"),
    "tensorsharp_f16": _make_ts_runner("f16"),
    "tensorsharp_q80": _make_ts_runner("q8_0"),
    "llamacpp": run_llamacpp,
    "ollama": run_ollama,
}


def applies(engine: str, model: ModelSpec, task: TaskSpec) -> tuple[bool, str]:
    if task.kind == "image" and not model.supports_image:
        return False, f"{model.short_id} does not have a vision projector"
    if task.kind == "audio" and not model.supports_audio:
        return False, f"{model.short_id} does not have an audio encoder"
    if task.kind == "video" and not model.supports_video:
        return False, f"{model.short_id} does not support video"
    if engine == "ollama" and task.kind in ("audio", "video"):
        return False, "Ollama API has no audio/video input"
    if engine == "llamacpp" and task.kind == "video":
        return False, "llama.cpp has no video CLI"
    return True, ""


def is_tensorsharp_engine(engine: str) -> bool:
    return engine.startswith("tensorsharp_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="ollama,llamacpp,tensorsharp_f32,tensorsharp_f16,tensorsharp_q80")
    ap.add_argument("--models", default="gemma4")
    ap.add_argument("--tasks", default="pp512,tg128,pp2048,short_text,long_text,image,audio,video")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a (engine,model,task) if its result file already exists")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.quick:
        for tid, overrides in QUICK_OVERRIDES.items():
            t = TASKS[tid]
            for k, v in overrides.items():
                setattr(t, k, v)

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"# benchmark plan")
    print(f"engines: {engines}")
    print(f"models : {models}")
    print(f"tasks  : {tasks}")
    print(f"out dir: {results_dir}")

    plan: list[tuple[str, ModelSpec, TaskSpec]] = []
    for engine in engines:
        for model_id in models:
            model = MODELS[model_id]
            for task_id in tasks:
                task = TASKS[task_id]
                ok, why = applies(engine, model, task)
                if not ok:
                    print(f"  skip  {engine:18s} {model_id:8s} {task_id:11s}  -> {why}")
                    continue
                plan.append((engine, model, task))
    print(f"\n# {len(plan)} runs scheduled\n")

    last_engine: Optional[str] = None
    for i, (engine, model, task) in enumerate(plan, 1):
        out_file = results_dir / f"{engine}__{model.short_id}__{task.short_id}.json"
        if args.skip_existing and out_file.exists():
            print(f"[{i:3d}/{len(plan)}] {engine:18s} {model.short_id:8s} {task.short_id:11s}  cached -> {out_file.name}")
            continue
        # Evict Ollama models before non-ollama engines to free GPU memory.
        if engine != "ollama" and last_engine != engine:
            ollama_unload_all()
        print(f"[{i:3d}/{len(plan)}] {engine:18s} {model.short_id:8s} {task.short_id:11s}  starting...", flush=True)
        if engine == "ollama":
            ollama_warmup(model.ollama_tag)
        last_engine = engine
        runner = ENGINES[engine]
        t0 = time.monotonic()
        result = runner(model, task, results_dir)
        wall = time.monotonic() - t0
        out_file.write_text(json.dumps(asdict(result), indent=2))
        status = "OK " if result.ok else "FAIL"
        msg = (
            f"          -> {status}  "
            f"prefill={result.prefill_tps:7.1f} t/s  "
            f"decode={result.decode_tps:6.1f} t/s  "
            f"wall={wall:5.1f}s  "
            f"{result.error}"
        )
        print(msg, flush=True)


if __name__ == "__main__":
    main()
