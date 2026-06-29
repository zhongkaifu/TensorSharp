#!/usr/bin/env python3
"""
Aggregate the per-cell JSON results into a markdown report + CSV.

Reads results/*.json (written by run_matrix.py) and produces:
  * docs/engine_comparison_report.md  - human-readable tables
  * results/results.csv                - flat machine-readable dump
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _peek_config_arg(argv) -> str | None:
    """`--config PATH` must be honored before `config` imports + loads its JSON."""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


_cfg_arg = _peek_config_arg(sys.argv[1:])
if _cfg_arg:
    os.environ["BENCH_CONFIG"] = _cfg_arg

import config

RESULTS_DIR = config.RESULTS_DIR
REPORT_PATH = config.REPO_ROOT / "docs" / "engine_comparison_report.md"
CSV_PATH = RESULTS_DIR / "results.csv"

# Column order: engine x backend.
COLUMNS = [
    ("tensorsharp", "gpu"), ("tensorsharp", "cpu"),
    ("llamacpp", "gpu"), ("llamacpp", "cpu"),
    ("vllm", "gpu"),
]
COL_LABEL = {
    ("tensorsharp", "gpu"): "TensorSharp · GPU",
    ("tensorsharp", "cpu"): "TensorSharp · CPU",
    ("llamacpp", "gpu"): "llama.cpp · GPU",
    ("llamacpp", "cpu"): "llama.cpp · CPU",
    ("vllm", "gpu"): "vLLM · GPU",
}

# Performance-ratio comparisons: TensorSharp (numerator) vs a reference engine on
# the *same* backend, so the columns stay apples-to-apples. A ratio > 1.0×
# means TensorSharp is faster (for throughput metrics) / lower-latency (TTFT).
RATIO_PAIRS = [
    (("tensorsharp", "gpu"), ("llamacpp", "gpu"), "vs llama.cpp · GPU"),
    (("tensorsharp", "cpu"), ("llamacpp", "cpu"), "vs llama.cpp · CPU"),
    (("tensorsharp", "gpu"), ("vllm", "gpu"), "vs vLLM · GPU"),
]


def load_all() -> dict:
    """Returns (baseline, rows).

    baseline[model][scenario][(engine,backend)] = record   (only mtp-off,
    concurrency-1 cells, so the headline per-engine tables stay apples-to-apples).
    rows = every record (all mtp / concurrency axes), used by the MTP and
    concurrency sections.
    """
    out: dict = {}
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        eng, backend, model, scenario = d["engine"], d["backend"], d["model"], d["scenario"]
        rows.append(d)
        if d.get("mtp", False) or int(d.get("concurrency", 1) or 1) != 1:
            continue  # keep the baseline tables to the single-stream, no-MTP point
        out.setdefault(model, {}).setdefault(scenario, {})[(eng, backend)] = d
    return out, rows


def _cell(rec, metric) -> str:
    if rec is None:
        return "n/a"
    status = rec.get("status")
    if status == "skipped":
        return "—"
    if status != "ok":
        return "fail"
    v = rec.get(metric, 0.0) or 0.0
    return f"{v:.1f}" if v > 0 else "—"


def _present_columns(data: dict) -> list:
    seen = set()
    for scen_map in data.values():
        for col_map in scen_map.values():
            seen.update(col_map.keys())
    return [c for c in COLUMNS if c in seen]


def _scenario_rows(scen_map: dict) -> list:
    """Scenario ids to render, in a stable order: declared scenarios first (in
    registry order), then any extra ids present in the data (e.g. on-the-fly
    `prefill_*` scenarios run against a config that did not declare them)."""
    declared = list(config.SCENARIOS.keys())   # real declared keys only
    ordered = [s for s in declared if s in scen_map]
    extra = sorted(s for s in scen_map if s not in declared)
    return ordered + extra


def metric_table(scen_map: dict, cols: list, metric: str) -> str:
    head = "| Scenario | " + " | ".join(COL_LABEL[c] for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    rows = [head, sep]
    for scenario_id in _scenario_rows(scen_map):
        col_map = scen_map[scenario_id]
        cells = [scenario_id]
        for c in cols:
            cells.append(_cell(col_map.get(c), metric))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _ok_value(rec, metric: str) -> float:
    """The metric value only when the cell actually ran; else 0.0."""
    if not rec or rec.get("status") != "ok":
        return 0.0
    return float(rec.get(metric, 0.0) or 0.0)


def _ratio_val(ts_rec, ref_rec, metric: str, higher_is_better: bool = True) -> float:
    """TensorSharp-relative speedup for one cell, or 0.0 if either side is
    missing/failed/zero. For throughput (higher is better) this is TS/ref; for
    latency like TTFT (lower is better) it is ref/TS — so a value > 1.0× always
    means TensorSharp won."""
    a = _ok_value(ts_rec, metric)
    b = _ok_value(ref_rec, metric)
    if a <= 0 or b <= 0:
        return 0.0
    return (a / b) if higher_is_better else (b / a)


def _fmt_ratio(r: float) -> str:
    return f"{r:.2f}×" if r > 0 else "—"


def _geomean(vals: list) -> float:
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


_RATIO_METRICS = (("decode_tps", True), ("prefill_tps", True), ("ttft_ms", False))


def _present_ratio_pairs(scen_map: dict) -> list:
    """Ratio pairs that yield at least one real number for this model — i.e. some
    scenario where both TensorSharp and the reference actually ran. Drops e.g. an
    all-`—` vLLM column when that endpoint never produced a comparable cell."""
    out = []
    for ts, ref, lbl in RATIO_PAIRS:
        if any(_ratio_val(col_map.get(ts), col_map.get(ref), m, hib) > 0
               for col_map in scen_map.values()
               for m, hib in _RATIO_METRICS):
            out.append((ts, ref, lbl))
    return out


def ratio_table(scen_map: dict, pairs: list, metric: str,
                higher_is_better: bool = True) -> str:
    head = "| Scenario | " + " | ".join(lbl for _, _, lbl in pairs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(pairs)) + "|"
    rows = [head, sep]
    for scenario_id in _scenario_rows(scen_map):
        col_map = scen_map[scenario_id]
        cells = [scenario_id]
        for ts, ref, _ in pairs:
            cells.append(_fmt_ratio(_ratio_val(
                col_map.get(ts), col_map.get(ref), metric, higher_is_better)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def summary_section(data: dict) -> str:
    """Headline geomean speedup of TensorSharp over each reference engine, per
    model, across all scenarios both engines ran. Geomean (not mean) so a single
    runaway scenario can't dominate the per-model number."""
    lines = ["| Model | Comparison | decode | prefill | TTFT |",
             "|---|---|---:|---:|---:|"]
    any_row = False
    for model_id in config.MODELS:
        if model_id not in data:
            continue
        scen_map = data[model_id]
        for ts, ref, lbl in _present_ratio_pairs(scen_map):
            d_ratios, p_ratios, t_ratios = [], [], []
            for col_map in scen_map.values():
                d = _ratio_val(col_map.get(ts), col_map.get(ref), "decode_tps")
                p = _ratio_val(col_map.get(ts), col_map.get(ref), "prefill_tps")
                t = _ratio_val(col_map.get(ts), col_map.get(ref), "ttft_ms",
                               higher_is_better=False)
                if d:
                    d_ratios.append(d)
                if p:
                    p_ratios.append(p)
                if t:
                    t_ratios.append(t)
            dg, pg, tg = _geomean(d_ratios), _geomean(p_ratios), _geomean(t_ratios)
            if dg <= 0 and pg <= 0 and tg <= 0:
                continue
            any_row = True
            lines.append(f"| {config.MODELS[model_id].display} | {lbl} | "
                         f"{_fmt_ratio(dg)} | {_fmt_ratio(pg)} | {_fmt_ratio(tg)} |")
    if not any_row:
        return "_No overlapping TensorSharp / reference cells to compare._"
    return "\n".join(lines)


def versions_block() -> str:
    def _try(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception:
            return ""
    ts_rev = _try(["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"])
    dotnet_v = _try(["dotnet", "--version"])
    gpu = _try(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return (
        "| Component | Version / detail |\n"
        "|---|---|\n"
        f"| TensorSharp | git `{ts_rev}`, .NET {dotnet_v} (backends: ggml_cuda / ggml_cpu) |\n"
        f"| llama.cpp | `{config.LLAMA_SERVER_EXE}` |\n"
        f"| vLLM | endpoint `{config.VLLM_BASE_URL}` (connect-only) |\n"
        f"| GPU | {gpu or 'unknown'} |\n"
    )


def _ok_decode(rec) -> float:
    if not rec or rec.get("status") != "ok":
        return 0.0
    return float(rec.get("decode_tps", 0.0) or 0.0)


def mtp_section(rows: list) -> str:
    """MTP/NextN on-vs-off decode comparison (single-stream, TensorSharp).

    Pairs the concurrency-1 records that share (engine, backend, model, scenario)
    but differ on `mtp`, and reports decode tok/s off → on plus the speedup."""
    # index: (engine, backend, model, scenario) -> {mtp: rec}
    idx: dict = {}
    for r in rows:
        if int(r.get("concurrency", 1) or 1) != 1:
            continue
        key = (r["engine"], r["backend"], r["model"], r["scenario"])
        idx.setdefault(key, {})[bool(r.get("mtp", False))] = r

    paired = [(k, v) for k, v in idx.items() if True in v and False in v]
    if not paired:
        return "_No MTP on/off pairs were run (use `--mtp off,on`)._"

    lines = ["| Engine · Backend · Model | Scenario | decode off | decode on | speedup |",
             "|---|---|---:|---:|---:|"]
    for (eng, backend, model, scenario), v in sorted(paired):
        off = _ok_decode(v[False])
        on = _ok_decode(v[True])
        if off > 0 and on > 0:
            spd = f"{on / off:.2f}×"
        else:
            spd = "—"
        off_s = f"{off:.1f}" if off > 0 else _cell(v[False], "decode_tps")
        on_s = f"{on:.1f}" if on > 0 else _cell(v[True], "decode_tps")
        lines.append(f"| {eng} · {backend} · {model} | {scenario} | {off_s} | {on_s} | {spd} |")
    return "\n".join(lines)


def concurrency_section(rows: list) -> str:
    """Parallel-request scaling: per (engine, backend, model, scenario, mtp),
    show per-request and system-aggregate decode throughput at each concurrency."""
    levels = sorted({int(r.get("concurrency", 1) or 1) for r in rows})
    if levels == [1]:
        return "_No parallel-request cells were run (use `--concurrency 1,4,8`)._"

    # index: (engine, backend, model, scenario, mtp) -> {concurrency: rec}
    idx: dict = {}
    for r in rows:
        key = (r["engine"], r["backend"], r["model"], r["scenario"], bool(r.get("mtp", False)))
        idx.setdefault(key, {})[int(r.get("concurrency", 1) or 1)] = r

    # Only keep series that actually exercise >1 concurrency.
    series = {k: v for k, v in idx.items() if any(c > 1 for c in v)}
    if not series:
        return "_No parallel-request cells were run (use `--concurrency 1,4,8`)._"

    head = ("| Engine · Backend · Model · Scenario | metric | "
            + " | ".join(f"c={c}" for c in levels) + " |")
    sep = "|---|---|" + "|".join(["---:"] * len(levels)) + "|"
    lines = [head, sep]
    for key, by_c in sorted(series.items()):
        eng, backend, model, scenario, mtp = key
        label = f"{eng} · {backend} · {model} · {scenario}" + (" · mtp" if mtp else "")

        def _row(metric_label, metric_key):
            cells = []
            for c in levels:
                rec = by_c.get(c)
                cells.append(_cell(rec, metric_key))
            return f"| {label} | {metric_label} | " + " | ".join(cells) + " |"

        lines.append(_row("decode/req t/s", "decode_tps"))
        lines.append(_row("aggregate t/s", "aggregate_decode_tps"))
    return "\n".join(lines)


def tool_summary(rows: list) -> str:
    fc = [r for r in rows if r["scenario"] == "function_call" and r["status"] == "ok"]
    if not fc:
        return "_No function-call cells were run._"
    lines = ["| Engine · Backend · Model | tool_call emitted |", "|---|:---:|"]
    for r in sorted(fc, key=lambda r: (r["engine"], r["backend"], r["model"])):
        ok = r.get("tool_call_ok")
        mark = "yes" if ok else ("no" if ok is False else "?")
        lines.append(f"| {r['engine']} · {r['backend']} · {r['model']} | {mark} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="JSON benchmark config file (default: benchmark_config.json, "
                         "or set BENCH_CONFIG)")
    ap.add_argument("--results", default=None,
                    help="results directory to aggregate (default from config)")
    args = ap.parse_args()

    global RESULTS_DIR, CSV_PATH
    if args.results:
        RESULTS_DIR = Path(args.results)
        CSV_PATH = RESULTS_DIR / "results.csv"

    data, rows = load_all()
    if not rows:
        print(f"No results found in {RESULTS_DIR}. Run run_matrix.py first.")
        return
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = []
    out.append("# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM\n")
    out.append("Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` "
               "surface, across text / image / audio / video / single-turn / multi-turn / "
               "function-call / structured-output scenarios on GPU and CPU backends.\n")
    out.append("Numbers are tokens/second (higher is better). `—` = not applicable / skipped, "
               "`fail` = errored at runtime, `n/a` = combination never attempted.\n")

    out.append("## Software / hardware\n")
    out.append(versions_block())
    out.append("")

    out.append("## Methodology\n")
    out.append("- Each `(engine, backend, model)` group launches its server once; all of "
               "that group's scenarios run against it, so per-scenario timings exclude "
               "model-load cost.\n"
               "- Metrics come from the **streamed** response: `ttft` is time-to-first-token "
               "(prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and "
               "`decode_tps = (completion_tokens - 1) / (t_last - t_first)`.\n"
               "- DiffusionGemma denoises whole blocks (no token stream), so it is run "
               "non-streaming and its `decode_tps` is wall-clock tokens/second.\n"
               "- Greedy sampling (`temperature=0`); one warmup request per server is discarded.\n"
               "- The headline per-engine tables are the **single-stream, MTP-off** baseline. "
               "MTP on/off and parallel-request scaling are reported in their own sections "
               "below.\n")

    out.append("## Performance ratio — TensorSharp vs reference engines\n")
    out.append("Geomean of TensorSharp's per-scenario speedup over each reference engine on "
               "the **same backend**, across every scenario both engines ran (single-stream, "
               "MTP-off). A value **> 1.0× means TensorSharp is faster** (for decode / prefill "
               "throughput) or lower-latency (for TTFT); `—` = no overlapping cells. "
               "Per-scenario ratios are in each model's section below.\n")
    out.append(summary_section(data))
    out.append("")

    for model_id in config.MODELS:
        if model_id not in data:
            continue
        model = config.MODELS[model_id]
        scen_map = data[model_id]
        cols = _present_columns({model_id: scen_map})
        if not cols:
            continue
        out.append(f"## {model.display}  (`{model_id}`)\n")
        out.append("**Decode throughput (tok/s)**\n")
        out.append(metric_table(scen_map, cols, "decode_tps"))
        out.append("")
        out.append("**Prefill throughput (tok/s)**\n")
        out.append(metric_table(scen_map, cols, "prefill_tps"))
        out.append("")
        out.append("**Time to first token (ms, lower is better)**\n")
        out.append(metric_table(scen_map, cols, "ttft_ms"))
        out.append("")

        pairs = _present_ratio_pairs(scen_map)
        if pairs:
            out.append("**Performance ratio — TensorSharp vs reference "
                       "(> 1.0× = TensorSharp faster)**\n")
            out.append("_Decode throughput_\n")
            out.append(ratio_table(scen_map, pairs, "decode_tps"))
            out.append("")
            out.append("_Prefill throughput_\n")
            out.append(ratio_table(scen_map, pairs, "prefill_tps"))
            out.append("")
            out.append("_Time to first token (latency; > 1.0× = TensorSharp lower)_\n")
            out.append(ratio_table(scen_map, pairs, "ttft_ms", higher_is_better=False))
            out.append("")

    out.append("## MTP / NextN speculative decoding (on vs off)\n")
    out.append("Single-stream decode tok/s with MTP/NextN speculative decoding off vs on "
               "(TensorSharp only). Speedup `< 1.0×` means speculation cost more than it "
               "saved for that cell — expected when the fused full-model decode path is "
               "already the fast path.\n")
    out.append(mtp_section(rows))
    out.append("")

    out.append("## Parallel-request scaling (concurrency)\n")
    out.append("`decode/req` is the mean per-request decode tok/s; `aggregate` is the "
               "system-wide decode throughput (total generated tokens / the wall window "
               "during which any sequence was decoding) when N identical requests are fired "
               "at one server at once.\n")
    out.append(concurrency_section(rows))
    out.append("")

    out.append("## Function-calling correctness\n")
    out.append(tool_summary(rows))
    out.append("")

    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

    # Flat CSV
    fields = ["engine", "backend", "model", "scenario", "mtp", "concurrency",
              "status", "detail", "prompt_tokens", "completion_tokens", "ttft_ms",
              "prefill_tps", "decode_tps", "aggregate_decode_tps", "requests_ok",
              "total_wall_ms", "finish_reason", "tool_call_ok"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
