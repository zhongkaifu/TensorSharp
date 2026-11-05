#!/usr/bin/env python3
"""Score, rank and report a Qwen3.8-27B cross-engine campaign (bilingual report + poster).

  python3 eng/qwen38-engine-scorecard.py CAMPAIGN_DIR --out artifacts/qwen38-scorecard \
      [--context context.json]

CAMPAIGN_DIR holds one directory per engine and context tier, as written by the campaign
driver: <engine>-main/main.jsonl (the short scenarios), <engine>-{64k,128k,256k}/ with the tier's
needle-retrieval TTFT (long_prompt_*), its long session (long_session_*: cold TTFT, multi-turn,
KV-cache reuse, concurrent waves) and longdecode.jsonl (one JSON line per long-context decode
probe), ready.txt (load_s=...), vram.txt and spec.txt ("on ..." or "off": whether speculative
decoding ran; absent = on, except <engine>-nospec). Every metric records which (summary
"spec": on / off / mixed) so the poster can label each result.
TensorSharp is scored as two deployments: tensorsharp (the NInfer .ninfer artifact on the
native CUDA engine, text only) and tensorsharp-gguf (the GGUF + mmproj model on the same
engine). <engine>-nospec holds the same engine with speculative
decoding off (long_generation only): it feeds the speculative-decoding section and is
never scored. Outputs: summary.json, report_en.html, report_zh.html, poster_en.html,
poster_zh.html. Nothing is started; only recorded evidence is read.

Every chart and the poster's test matrix name the task and the measured workload: the
median prompt tokens and generated tokens per request of the recorded cases.

Scoring. Every category has a quality score (100 x passed checks / checks) and/or a
performance score (100 x best / value for latencies, value / best for rates, averaged
over the category's metrics, missing values scored 0). A category with both is their
mean; the overall score is the weighted mean of categories. Performance from an errored
(unanswered) or missing case is not credited. Model files and KV formats differ between engines (see
the report): this ranks served configurations, not kernels in isolation.
"""
from __future__ import annotations

import argparse
import fnmatch
import html
import json
import math
import re
import statistics
from pathlib import Path

ENGINE_NAMES = {"tensorsharp": "TensorSharp-NInfer", "tensorsharp-gguf": "TensorSharp-GGUF", "ignis": "Ignis",
                "ninfer": "NInfer", "llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
TIERS = ("64k", "128k", "256k")
TS_ENGINES = ("tensorsharp", "tensorsharp-gguf")

# (id, en, zh, weight, quality items, perf metrics [(metric id, higher_is_better)]). A quality item is a
# scenario name, or "scenario-glob:check-name-regex" for some checks of the matching scenarios.
SESSION = "long_session_*"
CATEGORIES = [
    ("prefill", "Prefill TTFT (8K / 64K / 128K / 256K)", "预填充首 token（8K / 64K / 128K / 256K）", 1.5,
     ["long_prompt_8k", "long_prompt_63k", "long_prompt_127k", "long_prompt_250k"],
     [("ttft_8k", False), ("ttft_63k", False), ("ttft_127k", False), ("ttft_250k", False)]),
    ("longdecode", "Decode after 64K / 128K / 256K of context", "64K / 128K / 256K 上下文后解码", 1.5, [],
     [("ld_60000", True), ("ld_120000", True), ("ld_250000", True)]),
    ("reuse", "KV-cache reuse and multi-turn (64K / 128K / 256K)", "KV 缓存复用与多轮对话（64K / 128K / 256K）", 1.0,
     [SESSION + ":(turn|reuse|context_length)", "multi_turn"],
     [(f"lreuse_{t}", False) for t in TIERS] + [(f"lmt_{t}", False) for t in TIERS] + [("multiturn_ttft", False)]),
    ("concurrency", "Concurrent requests (64K / 128K / 256K)", "并发请求（64K / 128K / 256K）", 1.0,
     [SESSION + ":par"], [(f"lpar{n}_{t}", True) for t in TIERS for n in (2, 4)]),
    ("decode", "Decode, single request", "单请求解码", 1.0, ["long_generation", "short_chat"],
     [("decode_gen", True), ("ttft_short", False)]),
    ("agent", "Agent tool loop", "智能体工具循环", 1.0, ["agentic_loop"], [("agent_e2e", False)]),
    ("subagent", "Sub-agent fan-out", "子智能体扇出", 1.0, ["subagent_fanout"], [("subagent_e2e", False)]),
    ("tools", "Tool and skill calls", "工具与 Skill 调用", 1.0, ["tool_call_single", "tool_call_parallel", "skill_call"],
     [("tool_single_s", False), ("tool_parallel_s", False), ("skill_e2e", False)]),
    ("json", "JSON generation", "JSON 结构化生成", 1.0, ["json_object", "json_schema"],
     [("json_schema_s", False)]),
    ("code", "Code write / edit / run", "代码编写 / 编辑 / 运行", 1.0, ["code_edit", "code_run"],
     [("code_edit_s", False), ("code_run_e2e", False)]),
    ("image", "Image understanding", "图像理解", 1.0, ["image"], [("image_ttft", False)]),
    ("reasoning", "Answer quality (chat, arithmetic, GSM8K)", "回答质量（问答、计算、GSM8K）", 1.0,
     ["short_chat", "quality_tasks", "quality_gsm8k"], []),
]


def quality_items(m, items):
    """(passed, total) over a category's quality items (see CATEGORIES)."""
    ok = n = 0
    for item in items:
        if ":" in item:
            glob, rx = item.split(":", 1)
            for scenario, name, good in m["_checks"]:
                if fnmatch.fnmatch(scenario, glob) and re.match(rx, name):
                    n += 1
                    ok += 1 if good else 0
        else:
            a, b = m["_quality"].get(item, (0, 0))
            ok, n = ok + a, n + b
    return ok, n


def scored_scenario(scenario):
    """True when some category scores this scenario's checks (older runs carry retired scenarios)."""
    for c in CATEGORIES:
        for item in c[4]:
            glob = item.split(":", 1)[0]
            if fnmatch.fnmatch(scenario, glob):
                return True
    return False


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def load_dir(d: Path):
    cases = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name == "longdecode.jsonl":
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "case":
                    cases.append(rec)
    probes = []
    ld = d / "longdecode.jsonl"
    if ld.exists():
        for line in ld.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("{"):
                probes.append(json.loads(line))
    meta = {}
    rt = d / "ready.txt"
    if rt.exists():
        for line in rt.read_text().splitlines():
            if line.startswith("load_s="):
                meta["load_s"] = float(line.split("=", 1)[1])
    vt = d / "vram.txt"
    if vt.exists():
        vals = [int(x.split()[0]) for x in vt.read_text().split("\n") if x.strip()]
        meta["vram_mib"] = max(vals) if vals else None
    meta["failed"] = (d / "FAILED").exists()
    st = d / "spec.txt"
    if st.exists():
        meta["spec"] = "off" if st.read_text().strip().lower().startswith("off") else "on"
    else:
        meta["spec"] = "off" if d.name.endswith("-nospec") or (d / "LEAN").exists() else "on"
    vt2 = d / "variant.txt"
    if vt2.exists():
        meta["variant"] = vt2.read_text().strip()
    for c in cases:
        c["_spec"] = meta["spec"]
    for p in probes:
        p["_spec"] = meta["spec"]
    return cases, probes, meta


def collect(campaign: Path):
    engines = {}
    for d in sorted(p for p in campaign.iterdir() if p.is_dir()):
        name, _, tier = d.name.rpartition("-")
        if not name:
            continue
        base = name
        if base not in ENGINE_NAMES:
            continue
        e = engines.setdefault(base, {"cases": [], "probes": [], "meta": {}, "companion": [], "nospec": []})
        cases, probes, meta = load_dir(d)
        if tier == "nospec":
            e["nospec"].extend(cases)
            continue
        e["cases"].extend(cases)
        e["probes"].extend(probes)
        e["meta"][tier] = meta
    return engines


def metrics_for(e):
    by = {}
    for c in e["cases"]:
        by.setdefault(c["scenario"], []).append(c)

    def m(scenario, key, answered_only=True):
        # Latency/throughput count for every delivered answer (PASS or FAIL); an ERROR (no answer,
        # refused or unsupported request) earns none. Quality is scored separately from the checks.
        return med([c["metrics"].get(key) for c in by.get(scenario, []) if not answered_only or c.get("verdict") in ("PASS", "FAIL")])

    out = {
        "ttft_8k": m("long_prompt_8k", "ttft_s"),
        "ttft_63k": m("long_prompt_63k", "ttft_s"), "ttft_127k": m("long_prompt_127k", "ttft_s"),
        "ttft_250k": m("long_prompt_250k", "ttft_s"),
        "decode_gen": m("long_generation", "decode_tok_s"), "ttft_short": m("short_chat", "ttft_s"),
        "agent_e2e": m("agentic_loop", "end_to_end_s"),
        "subagent_e2e": m("subagent_fanout", "end_to_end_s"),
        "tool_single_s": m("tool_call_single", "total_s"), "tool_parallel_s": m("tool_call_parallel", "total_s"),
        "skill_e2e": m("skill_call", "end_to_end_s"), "json_schema_s": m("json_schema", "total_s"),
        "code_edit_s": m("code_edit", "total_s"), "code_run_e2e": m("code_run", "end_to_end_s"),
        "image_ttft": m("image", "ttft_s"),
    }
    mt = [c["metrics"].get("turn_ttft_s") or c["metrics"].get("ttft_s") for c in by.get("multi_turn", [])
          if c.get("verdict") in ("PASS", "FAIL")]
    later = [t[1:] for t in mt if isinstance(t, list) and len(t) > 1]
    out["multiturn_ttft"] = med([x for t in later for x in t]) if later else m("multi_turn", "ttft_s")
    out["mt_turn1_ttft"] = med([t[0] for t in mt if isinstance(t, list) and t])
    out["mt_cached"] = med([x for c in by.get("multi_turn", []) for x in (c["metrics"].get("turn_cached_tokens") or [])[1:]])
    for tier in TIERS:
        ss = [c["metrics"] for c in by.get(f"long_session_{tier}", []) if c.get("verdict") in ("PASS", "FAIL")]
        out[f"lcold_{tier}"] = med([x.get("cold_ttft_s") for x in ss])
        out[f"lmt_{tier}"] = med([y for x in ss for y in (x.get("turn_ttft_s") or [])[1:]])
        out[f"lreuse_{tier}"] = med([y for x in ss for y in (x.get("reuse_ttft_s") or [])])
        out[f"lmt_cached_{tier}"] = med([y for x in ss for y in (x.get("turn_cached_tokens") or [])[1:]])
        out[f"lreuse_cached_{tier}"] = med([y for x in ss for y in (x.get("reuse_cached_tokens") or [])])
        out[f"ldoc_{tier}"] = med([x.get("prompt_tokens") for x in ss])
        for n in (2, 4):
            waves = [x.get(f"par{n}") or {} for x in ss]
            out[f"lpar{n}_{tier}"] = med([w.get("aggregate_tok_s") for w in waves])
            out[f"lpar{n}_ttft_{tier}"] = med([w.get("ttft_s") for w in waves])
    gen = [c["metrics"] for c in by.get("long_generation", []) if c.get("verdict") in ("PASS", "FAIL")]
    out["spec_accept"] = med([g.get("acceptance_rate") for g in gen])
    out["spec_tokens_per_step"] = med([g.get("tokens_per_step") for g in gen])
    out["decode_nospec"] = med([c["metrics"].get("decode_tok_s") for c in e.get("nospec", [])
                                if c["scenario"] == "long_generation" and c.get("verdict") in ("PASS", "FAIL")])
    out["spec_speedup"] = (out["decode_gen"] / out["decode_nospec"]
                           if out.get("decode_gen") and out["decode_nospec"] else None)
    for size in (60000, 120000, 250000):
        out[f"ld_{size}"] = med([p.get("decode_tok_s") for p in e["probes"]
                                 if p.get("tag", "").endswith(f"-{size}") and p.get("completion_tokens")])
        out[f"ld_ttft_{size}"] = med([p.get("ttft_s") for p in e["probes"]
                                      if p.get("tag", "").endswith(f"-{size}") and p.get("completion_tokens")])
    # Whether speculative decoding ran for each measurement: on / off / mixed.
    spec = {}
    for mid, (scenario, _, _) in WORKLOADS.items():
        if scenario.startswith("probe:"):
            vals = {p.get("_spec") for p in e["probes"] if p.get("tag", "").endswith("-" + scenario.split(":")[1])
                    and p.get("completion_tokens")}
        else:
            vals = {c.get("_spec") for c in by.get(scenario, []) if c.get("verdict") in ("PASS", "FAIL")}
        vals.discard(None)
        spec[mid] = (vals.pop() if len(vals) == 1 else "mixed") if vals else None
    spec["decode_nospec"] = "off" if out["decode_nospec"] else None
    out["_spec"] = spec
    q = {}
    for scenario, cs in by.items():
        checks = [ch for c in cs for ch in c.get("checks", [])]
        q[scenario] = (sum(1 for ch in checks if ch.get("ok")), len(checks))
    out["_quality"] = q
    out["_checks"] = [(c["scenario"], ch.get("name", ""), bool(ch.get("ok"))) for c in e["cases"] for ch in c.get("checks", [])]
    return out


def score(engines, not_run=None):
    """not_run: {engine: [metric ids]} the campaign could not run for that engine (an
    environment problem on the test machine, not a capability): excluded from that
    engine's score instead of scored 0, and shown as not run."""
    not_run = not_run or {}
    mets = {k: metrics_for(v) for k, v in engines.items()}
    perf_ids = {mid for cat in CATEGORIES for mid, _ in cat[5]}
    best = {}
    for mid in perf_ids:
        hib = next(h for cat in CATEGORIES for m_, h in cat[5] if m_ == mid)
        vals = [m[mid] for m in mets.values() if m.get(mid)]
        best[mid] = (max(vals) if hib else min(vals)) if vals else None
    result = {}
    for k, m in mets.items():
        cats = {}
        for cid, en, zh, w, qs, ps in CATEGORIES:
            n_ok, n = quality_items(m, qs)
            quality = 100.0 * n_ok / n if n else (0.0 if qs else None)
            perf_parts = []
            for mid, hib in ps:
                v, b = m.get(mid), best.get(mid)
                if b is None or mid in not_run.get(k, ()):
                    continue
                perf_parts.append(0.0 if not v else 100.0 * (v / b if hib else b / v))
            perf = sum(perf_parts) / len(perf_parts) if perf_parts else None
            parts = [x for x in (quality, perf) if x is not None]
            cats[cid] = {"quality": quality, "perf": perf, "checks": [n_ok, n],
                         "score": sum(parts) / len(parts) if parts else None}
        wsum = sum(c[3] for c in CATEGORIES if cats[c[0]]["score"] is not None)
        overall = sum(c[3] * cats[c[0]]["score"] for c in CATEGORIES if cats[c[0]]["score"] is not None) / wsum
        qcats = [cats[c[0]]["quality"] for c in CATEGORIES if cats[c[0]]["quality"] is not None]
        pcats = [cats[c[0]]["perf"] for c in CATEGORIES if cats[c[0]]["perf"] is not None]
        allq = [m["_quality"][s] for s in m["_quality"] if scored_scenario(s)]
        result[k] = {"metrics": {a: b for a, b in m.items() if not a.startswith("_")}, "spec": m["_spec"], "categories": cats,
                     "overall": overall, "quality": sum(qcats) / len(qcats) if qcats else None,
                     "performance": sum(pcats) / len(pcats) if pcats else None,
                     "checks": [sum(a for a, _ in allq), sum(b for _, b in allq)],
                     "meta": engines[k]["meta"]}
    order = sorted(result, key=lambda k: -result[k]["overall"])
    for i, k in enumerate(order):
        result[k]["rank"] = i + 1
    return result, order, best


# --------------------------------------------------------------------------- rendering

T = {
    "en": {"title": "Qwen3.8-27B inference engines on one RTX 5090: performance and quality",
           "sub": "TensorSharp vs Ignis, NInfer, llama.cpp and SGLang - same prompts, greedy, thinking off",
           "rank": "Overall ranking", "overall": "Overall", "quality": "Quality", "perf": "Performance",
           "engine": "Engine", "cats": "Category scores (0-100)", "metrics": "Key measurements",
           "method": "Method", "caveats": "Caveats", "ts": "TensorSharp improvements in this round",
           "analysis": "What transfers to Metal, MLX and the other models",
           "checks": "checks passed", "load": "load", "table": "Table view",
           "lower": "lower is better", "higher": "higher is better", "na": "unsupported / not run"},
    "zh": {"title": "单张 RTX 5090 上的 Qwen3.8-27B 推理引擎：性能与质量评测",
           "sub": "TensorSharp 对比 Ignis、NInfer、llama.cpp、SGLang —— 相同提示、贪心解码、关闭思考",
           "rank": "综合排名", "overall": "综合", "quality": "质量", "perf": "性能",
           "engine": "引擎", "cats": "分项得分（0–100）", "metrics": "关键测量值",
           "method": "方法", "caveats": "注意事项", "ts": "本轮 TensorSharp 优化",
           "analysis": "对 Metal、MLX 及其他模型的适用性",
           "checks": "项检查通过", "load": "加载", "table": "表格视图",
           "lower": "越低越好", "higher": "越高越好", "na": "不支持/未测"},
}
TIER_LABEL = {"64k": "64K", "128k": "128K", "256k": "256K"}
METRIC_LABELS = {
    "ttft_8k": ("TTFT 8K prompt (s)", "8K 提示首 token (s)", False),
    "ttft_63k": ("TTFT 64K prompt (s)", "64K 提示首 token (s)", False),
    "ttft_127k": ("TTFT 128K prompt (s)", "128K 提示首 token (s)", False),
    "ttft_250k": ("TTFT 256K prompt (s)", "256K 提示首 token (s)", False),
    "ld_60000": ("Decode after 64K context (tok/s)", "64K 上下文后解码 (tok/s)", True),
    "ld_120000": ("Decode after 128K context (tok/s)", "128K 上下文后解码 (tok/s)", True),
    "ld_250000": ("Decode after 256K context (tok/s)", "256K 上下文后解码 (tok/s)", True),
    **{f"lreuse_{t}": (f"{TIER_LABEL[t]} shared document, new-chat TTFT (s)", f"{TIER_LABEL[t]} 共享文档新会话首 token (s)", False) for t in TIERS},
    **{f"lmt_{t}": (f"{TIER_LABEL[t]} document, turns 2-3 TTFT (s)", f"{TIER_LABEL[t]} 文档多轮对话第 2–3 轮首 token (s)", False) for t in TIERS},
    **{f"lpar{n}_{t}": (f"{TIER_LABEL[t]} document, {n} concurrent chats (tok/s total)", f"{TIER_LABEL[t]} 文档 {n} 路并发总吞吐 (tok/s)", True)
       for t in TIERS for n in (2, 4)},
    "decode_gen": ("Decode, code/prose/copy (tok/s)", "解码：代码/散文/复制 (tok/s)", True),
    "ttft_short": ("Short chat TTFT (s)", "短问答首 token (s)", False),
    "agent_e2e": ("Agent loop end to end (s)", "智能体循环端到端 (s)", False),
    "multiturn_ttft": ("Multi-turn TTFT, turns 2-3 (s)", "多轮对话第 2–3 轮首 token (s)", False),
    "subagent_e2e": ("Sub-agent fan-out end to end (s)", "子智能体扇出端到端 (s)", False),
    "tool_single_s": ("Single tool call (s)", "单次工具调用 (s)", False),
    "tool_parallel_s": ("Parallel tool calls (s)", "并行工具调用 (s)", False),
    "skill_e2e": ("Skill call end to end (s)", "Skill 调用端到端 (s)", False),
    "json_schema_s": ("Schema-constrained JSON (s)", "Schema 约束 JSON (s)", False),
    "code_edit_s": ("Code edit (s)", "代码编辑 (s)", False),
    "code_run_e2e": ("Write-run-answer code loop (s)", "代码编写-运行-作答 (s)", False),
    "image_ttft": ("Image TTFT (s)", "图像首 token (s)", False),
}

# What each measurement runs: (scenario or "probe:<context>", task en, task zh). The
# prompt/output token counts shown with it are measured from the recorded requests.
WORKLOADS = {
    "ttft_8k": ("long_prompt_8k", "find the vault codes hidden at 5-95% depth of the document, answer as JSON",
                "在文档 5%–95% 深度处找出隐藏的保险库代码，以 JSON 作答"),
    "ttft_63k": ("long_prompt_63k", "same needle retrieval, 64K-token document", "同上检索任务，64K token 文档"),
    "ttft_127k": ("long_prompt_127k", "same needle retrieval, 128K-token document", "同上检索任务，128K token 文档"),
    "ttft_250k": ("long_prompt_250k", "same needle retrieval, 256K-token document", "同上检索任务，256K token 文档"),
    "ld_60000": ("probe:60000", "section-by-section summary of a 60K-token technical document (512-token answer)",
                 "对 60K token 技术文档逐节总结（生成 512 token）"),
    "ld_120000": ("probe:120000", "same summary task after 120K tokens of context", "同上总结任务，120K 上下文"),
    "ld_250000": ("probe:250000", "same summary task after 250K tokens of context", "同上总结任务，250K 上下文"),
    **{f"lreuse_{t}": (f"long_session_{t}", f"new chats on a {TIER_LABEL[t]}-tier document another chat already read; TTFT",
                       f"在已被其他会话读过的 {TIER_LABEL[t]} 档文档上开新会话；首 token") for t in TIERS},
    **{f"lmt_{t}": (f"long_session_{t}", f"turns 2-3 of a chat over a {TIER_LABEL[t]}-tier document; TTFT",
                    f"基于 {TIER_LABEL[t]} 档文档的对话第 2–3 轮；首 token") for t in TIERS},
    **{f"lpar{n}_{t}": (f"long_session_{t}", f"{n} new chats at once on the {TIER_LABEL[t]}-tier document: retrieve a code, write ~150 words",
                        f"{n} 个新会话同时基于 {TIER_LABEL[t]} 档文档：检索代码并写约 150 词") for t in TIERS for n in (2, 4)},
    "decode_gen": ("long_generation", "write code, write prose, copy and summarize a passage (greedy, spec decoding on)",
                   "写代码、写散文、复制并总结段落（贪心，开启推测解码）"),
    "ttft_short": ("short_chat", "eight short factual questions", "8 个简短事实问答"),
    "agent_e2e": ("agentic_loop", "tool-using agent: read two files, call a calculator, report the total",
                  "工具型智能体：读取两个文件、调用计算器、给出总额"),
    "multiturn_ttft": ("multi_turn", "3-turn chat over the same records; TTFT of turns 2-3 (prefix reused)",
                       "基于同一记录的 3 轮对话；第 2–3 轮首 token（复用前缀）"),
    "subagent_e2e": ("subagent_fanout", "orchestrator delegates to sub-agents through tool calls and merges their results",
                     "主智能体通过工具调用委派子智能体并汇总结果"),
    "tool_single_s": ("tool_call_single", "one function call with typed arguments", "一次带类型参数的函数调用"),
    "tool_parallel_s": ("tool_call_parallel", "several function calls in one turn", "一轮内多次函数调用"),
    "skill_e2e": ("skill_call", "load a skill, follow it with a tool call, answer", "加载 Skill、按其调用工具并作答"),
    "json_schema_s": ("json_schema", "JSON object constrained by a schema", "受 Schema 约束的 JSON 对象"),
    "code_edit_s": ("code_edit", "edit a function to a spec; the result is unit-tested", "按规格修改函数，并以单元测试验证"),
    "code_run_e2e": ("code_run", "write a program, run it through a tool, answer from its output",
                     "编写程序、经工具运行、根据输出作答"),
    "image_ttft": ("image", "answer questions about an image", "回答关于图片的问题"),
}


def workload_stats(engines):
    """Median prompt tokens, generated tokens per request and requests per case, for each
    measurement, from TensorSharp's recorded requests (any engine's when it has none)."""
    out = {}
    for mid, (scenario, _, _) in WORKLOADS.items():
        stats = None
        for k in ["tensorsharp"] + [x for x in engines if x != "tensorsharp"]:
            e = engines.get(k)
            if not e:
                continue
            if scenario.startswith("probe:"):
                size = scenario.split(":")[1]
                ps = [p for p in e["probes"] if p.get("tag", "").endswith("-" + size) and p.get("completion_tokens")]
                if ps:
                    stats = {"prompt": med([p.get("prompt_tokens") for p in ps]),
                             "output": med([p.get("completion_tokens") for p in ps]), "requests": 1}
            else:
                cases = [c for c in e["cases"] if c["scenario"] == scenario]
                reqs = [q for c in cases for q in c.get("requests", []) if q.get("prompt_tokens")]
                if reqs:
                    stats = {"prompt": med([q["prompt_tokens"] for q in reqs]),
                             "prompt_max": max(q["prompt_tokens"] for q in reqs),
                             "output": med([q.get("completion_tokens") for q in reqs]),
                             "requests": med([len(c.get("requests", [])) for c in cases])}
            if stats:
                break
        out[mid] = stats
    return out


def workload_note(mid, stats, lang):
    """One line: the task, then the measured prompt and generated tokens per request."""
    scenario, en, zh = WORKLOADS.get(mid, (None, "", ""))
    task = zh if lang == "zh" else en
    st = (stats or {}).get(mid)
    if not st:
        return task
    p, o, n = st.get("prompt"), st.get("output"), st.get("requests")
    pmax = st.get("prompt_max")
    ptxt = f"{fmt_tokens(p)}" + (f"–{fmt_tokens(pmax)}" if pmax and p and pmax > 1.2 * p else "")
    if lang == "zh":
        req = f"{n:g} 个请求 · " if n and n > 1 else ""
        return f"{task} · {req}提示 {ptxt} token · 每请求生成 {fmt_tokens(o)} token"
    req = f"{n:g} requests · " if n and n > 1 else ""
    return f"{task} · {req}prompt {ptxt} tok · {fmt_tokens(o)} tok generated per request"


def fmt_tokens(v):
    if v is None:
        return "–"
    return f"{v / 1000:.1f}K" if v >= 10000 else f"{v:,.0f}"


def fmt(v, digits=2):
    if v is None:
        return "–"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    if abs(v) < 0.1:
        return f"{v:.3f}"
    return f"{v:.{digits}f}"


def bar_chart(title, rows, higher, lang, width=560, unit="", note=""):
    """Horizontal bars, one per engine, TensorSharp highlighted; value labels on every bar.
    note: the workload line (task, prompt and generated tokens) under the title."""
    rows = [(n, v) for n, v in rows]
    valid = [v for _, v in rows if v]
    if not valid:
        return ""
    vmax = max(valid)
    bh, gap, left, right = 22, 8, 118, 64
    h = len(rows) * (bh + gap) + 34
    out = [f'<figure class="chart"><figcaption>{html.escape(title)} <span class="hint">· '
           f'{T[lang]["higher"] if higher else T[lang]["lower"]}</span>'
           + (f'<div class="wl">{html.escape(note)}</div>' if note else "") + '</figcaption>'
           f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{html.escape(title)}">']
    plot = width - left - right
    for i, (name, v) in enumerate(rows):
        y = 8 + i * (bh + gap)
        is_ts = name == "TensorSharp"
        out.append(f'<text x="{left - 8}" y="{y + bh * 0.7:.1f}" text-anchor="end" class="lab{" ts" if is_ts else ""}">{html.escape(name)}</text>')
        if v:
            w = max(3.0, plot * v / vmax)
            out.append(f'<rect class="bar{" tsbar" if is_ts else ""}" x="{left}" y="{y}" width="{w:.1f}" height="{bh}" rx="4">'
                       f'<title>{html.escape(name)}: {fmt(v)}{unit}</title></rect>')
            out.append(f'<text x="{left + w + 6:.1f}" y="{y + bh * 0.7:.1f}" class="val">{fmt(v)}</text>')
        else:
            out.append(f'<text x="{left + 6}" y="{y + bh * 0.7:.1f}" class="val muted">{T[lang]["na"]}</text>')
    out.append(f'<line x1="{left}" x2="{left}" y1="4" y2="{h - 26}" class="axis"/></svg></figure>')
    return "".join(out)


CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f2f1ee;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --muted:#8a8984;--series-1:#2a78d6;--neutral-bar:#b9b7b0;--grid:#e3e1db}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
 --surface-2:#242422;--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#8f8e86;--series-1:#3987e5;--neutral-bar:#5d5c57;--grid:#33332f}}
body.viz-root{margin:0;background:var(--surface-1);color:var(--text-primary);font:15px/1.5 -apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 28px 64px}
h1{font-size:28px;margin:0 0 6px} h2{font-size:20px;margin:36px 0 12px} .sub{color:var(--text-secondary);margin:0 0 24px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right} th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600} tr.ts td{font-weight:650}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:18px}
.chart{margin:0;background:var(--surface-2);border-radius:10px;padding:12px 14px} figcaption{font-weight:600;margin-bottom:6px}
.hint{color:var(--muted);font-weight:400;font-size:12.5px} svg{width:100%;height:auto;display:block}
svg .lab{fill:var(--text-secondary);font-size:12.5px} svg .lab.ts{fill:var(--text-primary);font-weight:700}
svg .val{fill:var(--text-primary);font-size:12.5px;font-variant-numeric:tabular-nums} svg .muted{fill:var(--muted)}
svg .bar{fill:var(--neutral-bar)} svg .tsbar{fill:var(--series-1)} svg .axis{stroke:var(--grid)}
.rankrow{display:flex;align-items:center;gap:12px;margin:6px 0} .rankrow .n{width:26px;color:var(--muted);text-align:right}
.rankrow .nm{width:130px;font-weight:600} .rankrow .track{flex:1;background:var(--surface-2);border-radius:6px;height:26px;position:relative}
.rankrow .fill{height:100%;border-radius:6px;background:var(--neutral-bar)} .rankrow.ts .fill{background:var(--series-1)}
.rankrow .sc{width:260px;font-variant-numeric:tabular-nums;color:var(--text-secondary)}
ul.notes li{margin:4px 0;color:var(--text-secondary)} details summary{cursor:pointer;color:var(--text-secondary)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
.tile{background:var(--surface-2);border-radius:12px;padding:14px 16px} .tile .big{font-size:30px;font-weight:750;font-variant-numeric:tabular-nums}
.tile .cap{color:var(--text-secondary);font-size:13px}
.bagrid{display:grid;grid-template-columns:1fr 1fr;gap:10px 28px} .ba .bal{font-weight:600;margin-bottom:3px}
.ba .gain{float:right;color:var(--series-1);font-weight:750} .bab{display:flex;align-items:center;gap:8px;height:20px;margin:3px 0}
.bab span.b0,.bab span.b1{display:block;height:16px;border-radius:4px} .b0{background:var(--neutral-bar)} .b1{background:var(--series-1)}
.bv{font-size:12.5px;color:var(--text-secondary);white-space:nowrap;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}
.wl{font-weight:400;font-size:12.5px;color:var(--text-secondary);margin-top:2px}
table.matrix td,table.matrix th:nth-child(2){text-align:left;vertical-align:top} table.matrix td.num{text-align:right;white-space:nowrap}
"""


SECTION = {
    "en": {"matrix": "Test matrix: what each measurement runs", "task": "Task", "prompt": "Prompt tokens",
           "gen": "Generated tokens / request", "reqs": "Requests", "best": "Best", "metric": "Measurement",
           "spec": "Speculative decoding", "method": "Drafting", "accept": "Draft acceptance",
           "tps": "Tokens / verify step", "on": "Decode, spec on (tok/s)", "off": "Decode, spec off (tok/s)",
           "gain": "Speed-up", "reuse": "Long-context sessions: TTFT, multi-turn, KV-cache reuse, concurrency", "t1": "Cold TTFT (s)",
           "t23": "Turns 2-3 TTFT (s)", "mtc": "Turns 2-3 cached tokens", "doc": "Prompt tokens",
           "warm": "New-chat TTFT (s)", "rc": "New-chat cached tokens", "p2": "2 concurrent (tok/s)", "p4": "4 concurrent (tok/s)",
           "p2t": "2 concurrent TTFT (s)", "p4t": "4 concurrent TTFT (s)", "specc": "Spec decoding",
           "spec_note": "Same three long_generation prompts with the engine's speculative decoding switched off "
                        "(separate run). Acceptance and tokens per step are what the engine reports; blank where it reports neither.",
           "reuse_note": "long_session_<tier>: one document of the tier minus 4K tokens is the system message of every request. "
                         "Chat A turn 1 reads it cold; turns 2-3 continue that chat; chats B and C are new chats on it; then waves of 2 "
                         "and 4 new chats arrive at once, each retrieving a code and writing about 150 words (tok/s = all generated "
                         "tokens over the wave's wall time).",
           "spec_x": "Speculative decoding speed-up (x)"},
    "zh": {"matrix": "测试矩阵：每项测量的具体内容", "task": "任务", "prompt": "提示 token 数",
           "gen": "每请求生成 token 数", "reqs": "请求数", "best": "最佳", "metric": "测量项",
           "spec": "推测解码", "method": "草稿方式", "accept": "草稿接受率",
           "tps": "每次验证步产出 token", "on": "开启推测解码 (tok/s)", "off": "关闭推测解码 (tok/s)",
           "gain": "加速比", "reuse": "长上下文会话：首 token、多轮、KV 缓存复用、并发", "t1": "冷启动首 token (s)",
           "t23": "第 2–3 轮首 token (s)", "mtc": "第 2–3 轮命中缓存 token", "doc": "提示 token",
           "warm": "新会话首 token (s)", "rc": "新会话命中缓存 token", "p2": "2 路并发 (tok/s)", "p4": "4 路并发 (tok/s)",
           "p2t": "2 路并发首 token (s)", "p4t": "4 路并发首 token (s)", "specc": "推测解码",
           "spec_note": "同样三个 long_generation 提示，单独一轮关闭各引擎的推测解码。接受率与每步 token 数取自引擎自身上报；未上报者留空。",
           "reuse_note": "long_session_<档位>：一份（档位减 4K token）的文档作为每个请求的系统消息。会话 A 第 1 轮冷启动读入；第 2–3 轮延续该会话；"
                         "会话 B、C 为基于同一文档的新会话；随后 2 路、4 路新会话同时到达，各自检索一个代码并写约 150 词（tok/s = 全部生成 token / 整波墙钟时间）。",
           "spec_x": "推测解码加速比 (x)"},
}


def matrix_block(result, order, stats, best, lang, mids=None):
    """Every measurement with its task and measured workload, TensorSharp's value and the best."""
    t, sec = T[lang], SECTION[lang]
    li = 1 if lang == "zh" else 0
    rows = []
    for mid in mids or list(METRIC_LABELS):
        label = METRIC_LABELS[mid][li]
        task = WORKLOADS[mid][2 if lang == "zh" else 1]
        st = stats.get(mid) or {}
        p, pmax = st.get("prompt"), st.get("prompt_max")
        ptxt = fmt_tokens(p) + (f"–{fmt_tokens(pmax)}" if pmax and p and pmax > 1.2 * p else "")
        ts = result.get("tensorsharp", {}).get("metrics", {}).get(mid)
        tsg = result.get("tensorsharp-gguf", {}).get("metrics", {}).get(mid)
        b = best.get(mid)
        who = next((ENGINE_NAMES[k] for k in order if result[k]["metrics"].get(mid) == b), "") if b else ""
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(task)}</td><td class='num'>{ptxt}</td>"
                    f"<td class='num'>{fmt_tokens(st.get('output'))}</td><td class='num'>{st.get('requests') or 1:g}</td>"
                    f"<td class='num'>{fmt(ts)}</td><td class='num'>{fmt(tsg)}</td><td class='num'>{fmt(b)} {html.escape(who)}</td></tr>")
    return (f"<h2>{html.escape(sec['matrix'])}</h2><div style='overflow-x:auto'><table class='matrix'><tr>"
            f"<th>{sec['metric']}</th><th>{sec['task']}</th><th>{sec['prompt']}</th><th>{sec['gen']}</th>"
            f"<th>{sec['reqs']}</th><th>TensorSharp-NInfer</th><th>TensorSharp-GGUF</th><th>{sec['best']}</th></tr>{''.join(rows)}</table></div>")


def spec_block(result, order, context, lang, charts=True):
    sec = SECTION[lang]
    methods = context.get("spec_methods", {}).get(lang, {})
    rows = []
    for k in order:
        m = result[k]["metrics"]
        acc = m.get("spec_accept")
        rows.append(f'<tr class="{"ts" if k in TS_ENGINES else ""}"><td>{ENGINE_NAMES[k]}</td>'
                    f'<td style="text-align:left">{html.escape(methods.get(k, ""))}</td>'
                    f'<td>{fmt(100 * acc, 1) + "%" if acc else "–"}</td><td>{fmt(m.get("spec_tokens_per_step"))}</td>'
                    f'<td>{fmt(m.get("decode_gen"))}</td><td>{fmt(m.get("decode_nospec"))}</td>'
                    f'<td>{fmt(m.get("spec_speedup")) + "×" if m.get("spec_speedup") else "–"}</td></tr>')
    out = (f'<h2>{html.escape(sec["spec"])}</h2><p class="sub">{html.escape(sec["spec_note"])}</p>'
           f'<div style="overflow-x:auto"><table><tr><th>{T[lang]["engine"]}</th><th style="text-align:left">{sec["method"]}</th>'
           f'<th>{sec["accept"]}</th><th>{sec["tps"]}</th><th>{sec["on"]}</th><th>{sec["off"]}</th><th>{sec["gain"]}</th></tr>'
           + "".join(rows) + "</table></div>")
    if charts:
        out += '<div class="grid" style="margin-top:14px">' + bar_chart(
            sec["spec_x"], [(ENGINE_NAMES[k], result[k]["metrics"].get("spec_speedup")) for k in order], True, lang,
            note=WORKLOADS["decode_gen"][2 if lang == "zh" else 1]) + "</div>"
    return out


def reuse_block(result, order, stats, lang, charts=True):
    """One table per context tier: the long session's measurements for every engine."""
    sec = SECTION[lang]
    out = [f'<h2>{html.escape(sec["reuse"])}</h2><p class="sub">{html.escape(sec["reuse_note"])}</p>']
    for t in TIERS:
        rows = []
        for k in order:
            m, sp = result[k]["metrics"], result[k].get("spec", {})
            if not any(m.get(f"{x}_{t}") for x in ("lcold", "lmt", "lreuse")):
                continue
            rows.append(f'<tr class="{"ts" if k in TS_ENGINES else ""}"><td>{ENGINE_NAMES[k]}</td>'
                        f'<td>{fmt_tokens(m.get(f"ldoc_{t}"))}</td><td>{fmt(m.get(f"lcold_{t}"))}</td>'
                        f'<td>{fmt(m.get(f"lmt_{t}"), 3)}</td><td>{fmt_tokens(m.get(f"lmt_cached_{t}"))}</td>'
                        f'<td>{fmt(m.get(f"lreuse_{t}"), 3)}</td><td>{fmt_tokens(m.get(f"lreuse_cached_{t}"))}</td>'
                        f'<td>{fmt(m.get(f"lpar2_{t}"))}</td><td>{fmt(m.get(f"lpar2_ttft_{t}"))}</td>'
                        f'<td>{fmt(m.get(f"lpar4_{t}"))}</td><td>{fmt(m.get(f"lpar4_ttft_{t}"))}</td>'
                        f'<td>{html.escape(sp.get(f"lreuse_{t}") or "–")}</td></tr>')
        if rows:
            out.append(f'<h3>{TIER_LABEL[t]}</h3><div style="overflow-x:auto"><table><tr><th>{T[lang]["engine"]}</th>'
                       f'<th>{sec["doc"]}</th><th>{sec["t1"]}</th><th>{sec["t23"]}</th><th>{sec["mtc"]}</th><th>{sec["warm"]}</th>'
                       f'<th>{sec["rc"]}</th><th>{sec["p2"]}</th><th>{sec["p2t"]}</th><th>{sec["p4"]}</th><th>{sec["p4t"]}</th>'
                       f'<th>{sec["specc"]}</th></tr>' + "".join(rows) + "</table></div>")
    return "".join(out)


def ranking_block(result, order, lang):
    t = T[lang]
    rows = []
    for k in order:
        r = result[k]
        name = ENGINE_NAMES[k]
        rows.append(f'<div class="rankrow{" ts" if k in TS_ENGINES else ""}"><span class="n">{r["rank"]}</span>'
                    f'<span class="nm">{html.escape(name)}</span><span class="track"><span class="fill" style="width:{r["overall"]:.1f}%;display:block"></span></span>'
                    f'<span class="sc">{t["overall"]} {r["overall"]:.1f} · {t["quality"]} {fmt(r["quality"],1)} · {t["perf"]} {fmt(r["performance"],1)}</span></div>')
    return "\n".join(rows)


def render_report(result, order, context, lang, stats=None, best=None):
    t = T[lang]
    li = 1 if lang == "zh" else 0
    parts = [f'<!DOCTYPE html><html lang="{"zh-CN" if lang == "zh" else "en"}"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(t["title"])}</title>'
             f'<style>{CSS}</style></head><body class="viz-root"><main>',
             f'<h1>{html.escape(t["title"])}</h1><p class="sub">{html.escape(t["sub"])}</p>']
    summary = context.get("summary", {}).get(lang)
    if summary:
        parts.append("<p>" + html.escape(summary) + "</p>")
    parts.append(f'<h2>{t["rank"]}</h2>' + ranking_block(result, order, lang))
    # category table
    head = "".join(f"<th>{html.escape(c[1 + li])}</th>" for c in CATEGORIES)
    body = []
    for k in order:
        cells = "".join(f'<td>{fmt(result[k]["categories"][c[0]]["score"], 1)}</td>' for c in CATEGORIES)
        ok, n = result[k]["checks"]
        body.append(f'<tr class="{"ts" if k in TS_ENGINES else ""}"><td>{ENGINE_NAMES[k]}</td><td>{result[k]["overall"]:.1f}</td>{cells}<td>{ok}/{n}</td></tr>')
    parts.append(f'<h2>{t["cats"]}</h2><div style="overflow-x:auto"><table><tr><th>{t["engine"]}</th><th>{t["overall"]}</th>{head}'
                 f'<th>{t["checks"]}</th></tr>{"".join(body)}</table></div>')
    # charts
    parts.append(f'<h2>{t["metrics"]}</h2><div class="grid">')
    for mid, (en, zh, hib) in METRIC_LABELS.items():
        rows = [(ENGINE_NAMES[k], result[k]["metrics"].get(mid)) for k in order]
        parts.append(bar_chart(zh if lang == "zh" else en, rows, hib, lang, note=workload_note(mid, stats, lang)))
    parts.append("</div>")
    parts.append(matrix_block(result, order, stats or {}, best or {}, lang))
    parts.append(spec_block(result, order, context, lang))
    parts.append(reuse_block(result, order, stats or {}, lang))
    # full metric table
    mids = list(METRIC_LABELS)
    th = "".join(f"<th>{html.escape(METRIC_LABELS[m][li])}</th>" for m in mids)
    trs = "".join(f'<tr class="{"ts" if k in TS_ENGINES else ""}"><td>{ENGINE_NAMES[k]}</td>' +
                  "".join(f"<td>{fmt(result[k]['metrics'].get(m))}</td>" for m in mids) + "</tr>" for k in order)
    parts.append(f'<details><summary>{t["table"]}</summary><div style="overflow-x:auto"><table><tr><th>{t["engine"]}</th>{th}</tr>{trs}</table></div></details>')
    for key, title in (("ts_improvements", t["ts"]), ("analysis", t["analysis"]), ("method", t["method"]), ("caveats", t["caveats"])):
        items = context.get(key, {}).get(lang) or []
        if items:
            parts.append(f"<h2>{html.escape(title)}</h2><ul class='notes'>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>")
    configs = context.get("configs", {}).get(lang) or {}
    if configs:
        parts.append(f'<h2>{"引擎配置" if lang == "zh" else "Engine configurations"}</h2><table>' +
                     "".join(f"<tr><td>{html.escape(ENGINE_NAMES.get(k, k))}</td><td style='text-align:left'>{html.escape(v)}</td></tr>"
                             for k, v in configs.items()) + "</table>")
    parts.append("</main></body></html>")
    return "".join(parts)


def render_poster(result, order, context, lang, stats=None, best=None):
    t = T[lang]
    ts = result.get("tensorsharp", {})
    tiles = context.get("poster_tiles", {}).get(lang) or []
    tile_html = "".join(f'<div class="tile"><div class="big">{html.escape(a)}</div><div class="cap">{html.escape(b)}</div></div>' for a, b in tiles)
    key = [("ttft_63k", False), ("ld_60000", True), ("decode_gen", True), ("lpar4_64k", True),
           ("agent_e2e", False), ("lreuse_64k", False)]
    charts = "".join(bar_chart(METRIC_LABELS[m][1 if lang == "zh" else 0],
                               [(ENGINE_NAMES[k], result[k]["metrics"].get(m)) for k in order], h, lang, width=520,
                               note=workload_note(m, stats, lang))
                     for m, h in key)
    foot = context.get("poster_footer", {}).get(lang, "")
    ba = context.get("ts_before_after", {}).get(lang) or []
    ba_html = ""
    if ba:
        rows = []
        for label, before, after, unit, higher in ba:
            vmax = max(before, after)
            gain = (after / before) if higher else (before / after)
            rows.append(f'<div class="ba"><div class="bal">{html.escape(label)}<span class="gain">{gain:.1f}×</span></div>'
                        f'<div class="bab"><span class="b0" style="width:{100 * before / vmax:.1f}%"></span><span class="bv">{fmt(before)} {unit}</span></div>'
                        f'<div class="bab"><span class="b1" style="width:{100 * after / vmax:.1f}%"></span><span class="bv">{fmt(after)} {unit}</span></div></div>')
        legend = ("改动前 (PR #325 head)", "改动后") if lang == "zh" else ("before (PR #325 head)", "after")
        ba_html = (f'<h2>{html.escape(t["ts"])}</h2><p class="sub"><span class="sw b0"></span>{legend[0]} &nbsp; <span class="sw b1"></span>{legend[1]}</p>'
                   f'<div class="bagrid">{"".join(rows)}</div>')
    return (f'<!DOCTYPE html><html lang="{"zh-CN" if lang == "zh" else "en"}" data-theme="light"><head><meta charset="utf-8"><title>{html.escape(t["title"])}</title>'
            f'<style>{CSS} body{{width:1400px}} main{{max-width:1340px;padding:40px}} h1{{font-size:38px}} .grid{{grid-template-columns:1fr 1fr}} .rankrow .sc{{width:380px}}</style></head>'
            f'<body class="viz-root"><main><h1>{html.escape(t["title"])}</h1><p class="sub">{html.escape(t["sub"])}</p>'
            f'<div class="tiles">{tile_html}</div><h2>{t["rank"]}</h2>{ranking_block(result, order, lang)}'
            f'<h2>{t["metrics"]}</h2><div class="grid">{charts}</div>'
            f'{spec_block(result, order, context, lang, charts=False)}{reuse_block(result, order, stats or {}, lang, charts=False)}'
            f'{matrix_block(result, order, stats or {}, best or {}, lang)}'
            f'{ba_html}<p class="sub" style="margin-top:22px;font-size:13px">{html.escape(foot)}</p></main></body></html>')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", type=Path, help="JSON with summary/ts_improvements/method/caveats/configs/poster_tiles (per language)")
    a = ap.parse_args()
    engines = collect(a.campaign)
    incomplete = sorted(k for k, e in engines.items()
                        if not any(c["scenario"] == "long_generation" for c in e["cases"]))
    for k in incomplete:
        print(f"excluded (no complete 64K suite): {ENGINE_NAMES[k]}")
        del engines[k]
    if not engines:
        raise SystemExit("no engine results under " + str(a.campaign))
    context = json.loads(a.context.read_text(encoding="utf-8")) if a.context else {}
    result, order, best = score(engines, context.get("not_run"))
    stats = workload_stats(engines)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps({"order": order, "best": best, "engines": result, "workloads": stats,
        "categories": [{"id": c[0], "en": c[1], "zh": c[2], "weight": c[3], "quality": c[4],
                        "perf": [m for m, _ in c[5]]} for c in CATEGORIES]}, indent=1, ensure_ascii=False), encoding="utf-8")
    for lang in ("en", "zh"):
        (a.out / f"report_{lang}.html").write_text(render_report(result, order, context, lang, stats, best), encoding="utf-8")
        (a.out / f"poster_{lang}.html").write_text(render_poster(result, order, context, lang, stats, best), encoding="utf-8")
    for k in order:
        r = result[k]
        print(f"{r['rank']}. {ENGINE_NAMES[k]:14s} overall {r['overall']:.1f} quality {fmt(r['quality'],1)} perf {fmt(r['performance'],1)} checks {r['checks'][0]}/{r['checks'][1]}")


if __name__ == "__main__":
    main()
