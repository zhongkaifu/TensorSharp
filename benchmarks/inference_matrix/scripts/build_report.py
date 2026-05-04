#!/usr/bin/env python3
"""Aggregate JSON results from run_bench.py into a markdown report."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_PATH = ROOT.parents[1] / "docs" / "inference_benchmark_matrix.md"

ENGINES = ["tensorsharp_f32", "tensorsharp_f16", "tensorsharp_q80", "llamacpp", "ollama"]
ENGINE_LABEL = {
    "tensorsharp_f32": "TensorSharp (F32 KV)",
    "tensorsharp_f16": "TensorSharp (F16 KV)",
    "tensorsharp_q80": "TensorSharp (Q8 KV)",
    "llamacpp": "llama.cpp",
    "ollama": "Ollama",
}

TS_ENGINES = ["tensorsharp_f32", "tensorsharp_f16", "tensorsharp_q80"]
TS_KV_LABEL = {"tensorsharp_f32": "F32", "tensorsharp_f16": "F16", "tensorsharp_q80": "Q8_0"}
BASELINE_ENGINES = ["llamacpp", "ollama"]

MODELS_ORDER = ["gemma4"]
MODEL_LABEL = {
    "gemma4": "Gemma 4 E4B Q8_0 (7.6 GiB, dense)",
}
MODEL_GGUF = {
    "gemma4": "gemma-4-E4B-it-Q8_0.gguf",
}

TASKS_ORDER = ["pp512", "tg128", "pp2048", "short_text", "long_text", "image", "audio", "video"]
TASK_LABEL = {
    "pp512": "Synthetic prefill (512 tok)",
    "tg128": "Synthetic decode (128 tok)",
    "pp2048": "Synthetic prefill (2048 tok)",
    "short_text": "Short text prompt (~32 tok in / 64 tok out)",
    "long_text": "Long text prompt (~1043 tok in / 64 tok out)",
    "image": "Image: apple.png (~282 tok in / 64 tok out)",
    "audio": "Audio: 45 s of speech (~1148 tok in / 64 tok out)",
    "video": "Video: concert clip (frames -> tokens / 64 tok out)",
}

PREFILL_ONLY_TASKS = {"pp512", "pp2048"}


def _safe(v: float, fmt: str = "{:.1f}") -> str:
    if v is None or v <= 0:
        return "n/a"
    return fmt.format(v)


def load_all() -> dict:
    out = {}
    for f in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        engine, model, task = f.stem.split("__")
        out.setdefault(engine, {}).setdefault(model, {})[task] = d
    return out


def engine_versions_block() -> str:
    try:
        llama_v = subprocess.check_output(
            ["llama-cli", "--version"], text=True, stderr=subprocess.STDOUT
        )
        llama_v = next((l for l in llama_v.splitlines() if "version:" in l.lower() or "build" in l.lower()), "").strip()
    except Exception:
        llama_v = ""
    try:
        ollama_v_raw = subprocess.check_output(["ollama", "--version"], text=True).strip()
        m = re.search(r"\d+\.\d+\.\d+", ollama_v_raw)
        ollama_v = m.group(0) if m else ollama_v_raw
    except Exception:
        ollama_v = ""
    try:
        ts_rev = subprocess.check_output(
            ["git", "-C", str(ROOT.parents[1]), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        ts_rev = ""
    try:
        dotnet_v = subprocess.check_output(["dotnet", "--version"], text=True).strip()
    except Exception:
        dotnet_v = ""
    return (
        "| Engine | Version / build |\n"
        "|--------|-----------------|\n"
        f"| TensorSharp | git `{ts_rev}` (this repo), .NET {dotnet_v}, ggml_metal backend |\n"
        f"| llama.cpp   | brew package, {llama_v or 'unknown'} (Metal + BLAS) |\n"
        f"| Ollama      | {ollama_v or 'unknown'} |\n"
    )


def task_short(t: str) -> str:
    return {
        "pp512": "Prefill 512",
        "tg128": "Decode 128",
        "pp2048": "Prefill 2048",
        "short_text": "Short text",
        "long_text": "Long text",
        "image": "Image",
        "audio": "Audio",
        "video": "Video",
    }.get(t, t)


def full_results_table(all_data: dict, model_id: str) -> str:
    """All engines (TS variants + llama.cpp + Ollama) in one table."""
    header_parts = ["Task"]
    for e in ENGINES:
        header_parts.append(f"{ENGINE_LABEL[e]} prefill")
        header_parts.append(f"{ENGINE_LABEL[e]} decode")
    rows = ["| " + " | ".join(header_parts) + " |"]
    rows.append("|------|" + "|".join(["----:"] * (len(ENGINES) * 2)) + "|")

    for t in TASKS_ORDER:
        cells = [task_short(t)]
        for e in ENGINES:
            d = all_data.get(e, {}).get(model_id, {}).get(t)
            if d is None:
                pref, dec = "n/a", "n/a"
            elif not d.get("ok"):
                pref, dec = "fail", "fail"
            else:
                pv = d.get("prefill_tps", 0.0) or 0.0
                dv = d.get("decode_tps", 0.0) or 0.0
                pref = "—" if t == "tg128" and pv <= 0 else _safe(pv)
                dec = "—" if t in PREFILL_ONLY_TASKS else _safe(dv)
            cells.extend([pref, dec])
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def kv_cache_comparison_table(all_data: dict, model_id: str) -> str:
    """TensorSharp-only table comparing KV cache dtypes."""
    rows = []
    rows.append("| Task | F32 prefill | F32 decode | F16 prefill | F16 decode | Q8_0 prefill | Q8_0 decode |")
    rows.append("|------|----:|----:|----:|----:|----:|----:|")
    for t in TASKS_ORDER:
        cells = [task_short(t)]
        for e in TS_ENGINES:
            d = all_data.get(e, {}).get(model_id, {}).get(t)
            if d is None:
                pref, dec = "n/a", "n/a"
            elif not d.get("ok"):
                pref, dec = "fail", "fail"
            else:
                pv = d.get("prefill_tps", 0.0) or 0.0
                dv = d.get("decode_tps", 0.0) or 0.0
                pref = "—" if t == "tg128" and pv <= 0 else _safe(pv)
                dec = "—" if t in PREFILL_ONLY_TASKS else _safe(dv)
            cells.extend([pref, dec])
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def relative_table(all_data: dict, metric: str, ts_engine: str, baseline: str, tasks: list[str]) -> str:
    """TensorSharp (specific KV dtype) vs baseline engine."""
    head = "| KV dtype | " + " | ".join(task_short(t) for t in tasks) + " |"
    sep = "|---------|" + "|".join(["----:"] * len(tasks)) + "|"
    rows = [head, sep]
    for e in TS_ENGINES:
        cells = [TS_KV_LABEL[e]]
        for t in tasks:
            ts = all_data.get(e, {}).get("gemma4", {}).get(t)
            ll = all_data.get(baseline, {}).get("gemma4", {}).get(t)
            if ts is None or ll is None or not ts.get("ok") or not ll.get("ok"):
                cells.append("—")
                continue
            tv = ts.get(metric, 0.0) or 0.0
            lv = ll.get(metric, 0.0) or 0.0
            if lv <= 0 or tv <= 0:
                cells.append("—")
                continue
            cells.append(f"{tv/lv*100:.0f}%")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    all_data = load_all()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = []
    out.append("# TensorSharp inference benchmark matrix\n")
    out.append(
        "Comparison of **TensorSharp**, **llama.cpp** and **Ollama** running the **same**"
        " GGUF files on the **same** machine, across text, prefill, decode and multimodal"
        " (image / audio / video) workloads."
    )
    out.append("")
    out.append("TensorSharp is tested with three KV cache data types (F32, F16, Q8_0)"
               " to measure the performance impact of KV cache precision.")
    out.append("")
    out.append("All three engines were pointed at the same on-disk `.gguf` files."
               " For Ollama this required registering a custom Modelfile (`ts-gemma4-e4b-q8`)"
               " so that no quantisation differences would skew the comparison.")
    out.append("")

    # ---- TL;DR ----
    out.append("## TL;DR\n")
    out.append(
        "Headline numbers on Gemma 4 E4B Q8_0, decode throughput on a real text prompt"
        " (`long_text`, ~1043-token prompt -> 64 tokens generated), in tokens/second:"
    )
    out.append("")
    head = "| TensorSharp (F32 KV) | TensorSharp (F16 KV) | TensorSharp (Q8 KV) | llama.cpp | Ollama |"
    out.append(head)
    out.append("|----:|----:|----:|----:|----:|")
    cells = []
    for e in ENGINES:
        d = all_data.get(e, {}).get("gemma4", {}).get("long_text")
        v = (d or {}).get("decode_tps", 0.0) if d and d.get("ok") else 0.0
        cells.append(_safe(v))
    out.append("| " + " | ".join(cells) + " |")
    out.append("")

    out.append(
        "And prefill throughput on the synthetic 2048-token prompt (`pp2048`):"
    )
    out.append("")
    head = "| TensorSharp (F32 KV) | TensorSharp (F16 KV) | TensorSharp (Q8 KV) | llama.cpp | Ollama |"
    out.append(head)
    out.append("|----:|----:|----:|----:|----:|")
    cells = []
    for e in ENGINES:
        d = all_data.get(e, {}).get("gemma4", {}).get("pp2048")
        v = (d or {}).get("prefill_tps", 0.0) if d and d.get("ok") else 0.0
        cells.append(_safe(v))
    out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # ---- Software ----
    out.append("## 1. Software versions\n")
    out.append(engine_versions_block())
    out.append("")

    # ---- Model ----
    out.append("## 2. Model under test\n")
    out.append("| Short id | GGUF file | Family | Notes |\n|---|---|---|---|")
    out.append(f"| `gemma4` | `{MODEL_GGUF['gemma4']}` | Gemma 4 (8 B dense)"
               f" | Q8_0; vision + audio + video projector available (`gemma-4-mmproj-F16.gguf`) |")
    out.append("")

    # ---- Tasks ----
    out.append("## 3. Tasks\n")
    out.append("| Task | Description |")
    out.append("|------|-------------|")
    for t in TASKS_ORDER:
        out.append(f"| `{t}` | {TASK_LABEL[t]} |")
    out.append("")

    # ---- Methodology ----
    out.append("## 4. Methodology\n")
    out.append(
        "For each `(engine, model, task)` cell we recorded both **prefill throughput**"
        " (tokens/second of the prompt-processing phase) and **decode throughput**"
        " (tokens/second of the autoregressive generation phase). Numbers are taken"
        " from the engines' own internal timers, not from wall-clock around the"
        " process (so model-load and warmup are excluded)."
    )
    out.append("")
    out.append(
        "**Warm-up is excluded for every engine.** On Apple Metal the first prefill"
        " of a given batch shape pays a non-trivial pipeline-JIT cost that is not"
        " representative of steady-state speed."
    )
    out.append("")
    out.append("Per-engine details:")
    out.append("")
    out.append("- **TensorSharp** uses its `--benchmark` mode for the synthetic `pp*`/`tg*`"
               " tasks (3 runs each, best-run is reported). For real text and multimodal"
               " tasks it uses `--warmup-runs 1`. Each TensorSharp variant is run with"
               " a specific `--kv-cache-dtype` (f32, f16, or q8_0) to measure the"
               " impact of KV cache precision on throughput.")
    out.append("- **llama.cpp** uses `llama-bench -p N -n M -r 3` for synthetic,"
               " `llama-cli` for real text (`-st --no-warmup --no-display-prompt --jinja`),"
               " and `llama-mtmd-cli` for image/audio (with `--no-warmup`).")
    out.append("- **Ollama** is driven through its HTTP `/api/generate` endpoint."
               " Before each timed request a 1-token `keep_alive` ping ensures the"
               " model is already resident.")
    out.append("")
    out.append("Sampling is greedy (`temperature=0`) everywhere. The driver source is"
               " `benchmarks/inference_matrix/scripts/run_bench.py` and the raw per-cell"
               " JSON outputs are in `benchmarks/inference_matrix/results/`.")
    out.append("")

    # ---- Full results ----
    out.append("## 5. Full results — Gemma 4 E4B Q8_0\n")
    out.append(
        "All numbers are tokens / second (higher is better). `—` means the metric does"
        " not apply to that task, `n/a` means the engine does not support that"
        " combination, and `fail` means the engine errored out at runtime."
    )
    out.append("")
    out.append(full_results_table(all_data, "gemma4"))
    out.append("")

    # ---- KV cache comparison ----
    out.append("## 6. TensorSharp KV cache dtype comparison\n")
    out.append(
        "How does the KV cache data type affect TensorSharp throughput? All numbers"
        " are tokens / second for Gemma 4 E4B Q8_0."
    )
    out.append("")
    out.append(kv_cache_comparison_table(all_data, "gemma4"))
    out.append("")

    # ---- Relative to llama.cpp ----
    out.append("## 7. TensorSharp vs. llama.cpp (relative throughput)\n")
    out.append(
        "Each cell is `TensorSharp t/s / llama.cpp t/s`. 100% means parity;"
        " >100% means TensorSharp is faster."
    )
    out.append("")
    out.append("**Prefill**\n")
    out.append(relative_table(all_data, "prefill_tps", "", "llamacpp",
                              ["pp512", "pp2048", "long_text", "image", "audio"]))
    out.append("")
    out.append("**Decode**\n")
    out.append(relative_table(all_data, "decode_tps", "", "llamacpp",
                              ["tg128", "short_text", "long_text", "image", "audio"]))
    out.append("")

    # ---- Relative to Ollama ----
    out.append("## 8. TensorSharp vs. Ollama (relative throughput)\n")
    out.append("**Prefill**\n")
    out.append(relative_table(all_data, "prefill_tps", "", "ollama",
                              ["pp512", "pp2048", "long_text", "image"]))
    out.append("")
    out.append("**Decode**\n")
    out.append(relative_table(all_data, "decode_tps", "", "ollama",
                              ["tg128", "short_text", "long_text", "image"]))
    out.append("")

    # ---- Reproducing ----
    out.append("## 9. Reproducing this report\n")
    out.append("```bash")
    out.append("# 1. register the Gemma4 GGUF inside Ollama")
    out.append("cd benchmarks/inference_matrix")
    out.append("ollama create ts-gemma4-e4b-q8 -f modelfiles/Modelfile.gemma4-e4b-q8")
    out.append("")
    out.append("# 2. (re)build the TensorSharp CLI")
    out.append("dotnet build ../../TensorSharp.Cli/TensorSharp.Cli.csproj -c Release")
    out.append("")
    out.append("# 3. run the matrix (one engine at a time keeps the GPU happy)")
    out.append("python3 scripts/run_bench.py --engines ollama")
    out.append("python3 scripts/run_bench.py --engines llamacpp")
    out.append("python3 scripts/run_bench.py --engines tensorsharp_f32")
    out.append("python3 scripts/run_bench.py --engines tensorsharp_f16")
    out.append("python3 scripts/run_bench.py --engines tensorsharp_q80")
    out.append("")
    out.append("# 4. regenerate this report from the JSON results")
    out.append("python3 scripts/build_report.py")
    out.append("```")
    out.append("")
    out.append(f"Driver: `benchmarks/inference_matrix/scripts/run_bench.py`  ")
    out.append(f"Report generator: `benchmarks/inference_matrix/scripts/build_report.py`  ")
    out.append(f"Per-cell raw JSON: `benchmarks/inference_matrix/results/<engine>__<model>__<task>.json`")
    out.append("")

    REPORT_PATH.write_text("\n".join(out))
    print(f"Wrote {REPORT_PATH} ({REPORT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
