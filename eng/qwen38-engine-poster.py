#!/usr/bin/env python3
"""Render the one-page Qwen3.8-27B engine comparison poster (English and Chinese).

  python3 eng/qwen38-engine-poster.py REPORT_DIR --context poster-context.json

REPORT_DIR holds summary.json as written by eng/qwen38-engine-scorecard.py. Every engine ran the
main suite at 1 lane (one request at a time) and each long tier at 1, 2 and 4 lanes; the score uses
the 1-lane runs for single-request measurements and the 2 / 4-lane runs for the waves of 2 / 4
concurrent chats, so every measurement compares engines at the same lane count. The long-context
tables show each lane count's own rows (<metric>@l<N>). The context JSON supplies what the evidence
cannot: date, machine, per-engine setup rows, notes and the reasons behind results that did not run
or ran slowly (per language). Writes REPORT_DIR/poster2_en.html and poster2_zh.html; render them at
1600 px wide.

Encoding: TensorSharp (both deployments: the NInfer .ninfer model and the GGUF model) is
blue, the best value of each measurement gold, every other engine neutral gray; every bar
and row carries its engine name and value, so no identity depends on color. Values carry
whether speculative decoding ran for them ("spec" / "plain", from summary.json "spec"); a
configuration that could not run shows its reason instead of a number.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

NAMES = {"tensorsharp": "TensorSharp · NInfer model", "tensorsharp-gguf": "TensorSharp · GGUF model", "ignis": "Ignis",
         "ninfer": "NInfer", "llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
SHORT = {"tensorsharp": "TS·NInfer", "tensorsharp-gguf": "TS·GGUF", "ignis": "Ignis", "ninfer": "NInfer",
         "llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
TIERS = ("64k", "128k", "256k")
TIER_LABEL = {"64k": "64K", "128k": "128K", "256k": "256K"}
TIER_TTFT = {"64k": "ttft_63k", "128k": "ttft_127k", "256k": "ttft_250k"}
TIER_LD = {"64k": "ld_60000", "128k": "ld_120000", "256k": "ld_250000"}
TS = ("tensorsharp", "tensorsharp-gguf")
NAMES_ZH = {"tensorsharp": "TensorSharp · NInfer 模型", "tensorsharp-gguf": "TensorSharp · GGUF 模型"}


def name(k, lang):
    return NAMES_ZH.get(k, NAMES[k]) if lang == "zh" else NAMES[k]

TXT = {
    "en": {
        "title": "Qwen3.8-27B Inference Benchmark: TensorSharp (NInfer model / GGUF model) vs Six Engines",
        "rank": "Overall Ranking", "rank_sub": "Score 0-100 = weighted mean of {n} categories; each category averages quality (% checks passed) and performance (relative to the best engine)",
        "overall": "Overall", "quality": "Quality", "perf": "Performance", "checks": "checks passed",
        "decode": "Decode Throughput: Speculative vs Plain", "decode_sub": "Single request, tok/s - median of code, prose and passage-copy replies (74-1,959-token prompts, up to 1,024 generated), 3 repeats",
        "plain": "Plain decoding", "spec": "Speculative (DFlash2, 7 drafts / step)", "speedup": "Speed-up",
        "key": "Key Performance Metrics", "scen": "Performance by Task Scenario", "qual": "Quality Comparison: share of automatic checks passed",
        "setup": "Engine Setup", "notes": "Notes", "engine": "Engine", "version": "Version", "weights": "Weights",
        "kv": "KV cache", "drafter": "Drafter", "lower": "lower is better", "higher": "higher is better",
        "tldr": "TL;DR", "na": "n/a", "s": "s", "toks": "tok/s",
        "prompt": "prompt", "gen": "generated", "req": "requests", "total": "Total",
        "long": "Long Context: 64K, 128K and 256K", "long_sub": "Per tier: TTFT = first token of a needle-retrieval question over one new document of the tier; decode = tok/s of a 512-token summary after the document. "
                "Session on one shared document (tier minus 4K tokens, the system message): new chat = TTFT of a new chat on a document another chat has read (KV-cache reuse); "
                "turns 2-3 = TTFT of the follow-up turns of a chat (multi-turn); 2 / 4 concurrent = new chats arriving at once, each retrieving a code and writing ~150 words "
                "(total generated tok/s over the wave; small figure = median TTFT). Bars scale within each column.",
        "cols": {"ttft": "TTFT (s)", "ld": "Decode after (tok/s)", "reuse": "New chat, KV reuse (s)", "mt": "Turns 2-3 (s)",
                 "p2": "2 concurrent (tok/s)", "p4": "4 concurrent (tok/s)", "spec": "Spec decoding"},
        "spec_on": "spec", "spec_off": "plain", "spec_mixed": "mixed", "spec_row_on": "spec on", "spec_row_off": "plain (spec off)",
        "spec_legend": "spec = speculative decoding on (DFlash2, 7 drafts per step) · plain = off · "
                       "max N concurrent = the most requests the server ran at once (its configured slots; more wait in a queue)",
        "lanes": "max {n} concurrent", "ttft_short": "TTFT",
        "cmp": "TensorSharp: GGUF model vs NInfer model", "cmp_sub": "Same engine build and harness; bold = the better of the two",
        "cmp_rows": {"overall": "Overall score", "quality": "Quality (checks passed)", "ttft_63k": "Prefill 64K (s)", "ttft_127k": "Prefill 128K (s)",
                     "decode_gen": "Decode, speculative (tok/s)", "decode_nospec": "Decode, plain (tok/s)", "ld_60000": "Decode after 64K (tok/s)",
                     "ld_120000": "Decode after 128K (tok/s)", "ttft_250k": "Prefill 256K (s)", "ld_250000": "Decode after 256K (tok/s)",
                     "lreuse_64k": "64K new chat, KV reuse (s)", "lmt_128k": "128K turns 2-3 (s)", "lpar4_64k": "64K, 4 concurrent (tok/s)",
                     "lpar2_256k": "256K, 2 concurrent (tok/s)", "agent_e2e": "Agentic tool loop (s)"},
        "gguf": "GGUF model", "ninf": "NInfer model",
        "group": {"1": "Max 1 concurrent request", "2": "Max 2 concurrent requests", "4": "Max 4 concurrent requests"},
        "rank_g": "Overall Ranking", "excluded": "Not ranked",
        "long_note": "Long-context results all use speculative decoding (DFlash2, 7 drafts per step). Each engine ran every tier three times, "
                     "at max 1, 2 and 4 concurrent requests; bars and gold compare engines at the same lane count only. At 1 lane the waves "
                     "of 2 / 4 chats queue. The score takes single-request columns from the 1-lane runs and the 2 / 4-concurrent columns "
                     "from the 2 / 4-lane runs.",
        "notrun": "did not run", "why": "Why",
        "scen_g": "Performance by Task Scenario · one request at a time (1 lane)", "scen_tbl_sub": "Each row ranks the engines on one task (1 = best; gold = best). "
                     "Lower is better for times, higher for tok/s.",
        "task": "Task", "spec_all": "all results: speculative decoding",
    },
    "zh": {
        "title": "Qwen3.8-27B 推理基准：TensorSharp（NInfer / GGUF 模型）对比六款引擎",
        "rank": "综合排名", "rank_sub": "得分 0–100 = {n} 个类别的加权平均；每个类别为质量（检查通过率）与性能（相对最优引擎）的平均",
        "overall": "综合", "quality": "质量", "perf": "性能", "checks": "项检查通过",
        "decode": "解码吞吐：推测解码 vs 普通解码", "decode_sub": "单请求 tok/s —— 代码、散文、段落复制三类回答的中位数（提示 74–1,959 token，最多生成 1,024 token），重复 3 次",
        "plain": "普通解码", "spec": "推测解码（DFlash2，每步 7 个草稿）", "speedup": "加速比",
        "key": "关键性能指标", "scen": "分场景性能排名", "qual": "质量对比：自动检查通过率",
        "setup": "引擎配置", "notes": "说明", "engine": "引擎", "version": "版本", "weights": "权重",
        "kv": "KV 缓存", "drafter": "草稿模型", "lower": "越低越好", "higher": "越高越好",
        "tldr": "要点", "na": "无", "s": "s", "toks": "tok/s",
        "prompt": "提示", "gen": "生成", "req": "个请求", "total": "合计",
        "long": "长上下文：64K、128K 与 256K", "long_sub": "每个档位：首 token = 针对一份该档位新文档的检索问题的首 token 时间；解码 = 读完文档后生成 512 token 总结的 tok/s。"
                "共享文档会话（文档为档位减 4K token，作为系统消息）：新会话 = 在其他会话已读过的文档上开新会话的首 token（KV 缓存复用）；"
                "第 2–3 轮 = 同一会话后续轮次的首 token（多轮对话）；2 / 4 路并发 = 同时到达的新会话，各自检索代码并写约 150 词（整波总生成 tok/s；小字为首 token 中位数）。条形按列内比例绘制。",
        "cols": {"ttft": "首 token (s)", "ld": "上下文后解码 (tok/s)", "reuse": "新会话·KV 复用 (s)", "mt": "第 2–3 轮 (s)",
                 "p2": "2 路并发 (tok/s)", "p4": "4 路并发 (tok/s)", "spec": "推测解码"},
        "spec_on": "推测", "spec_off": "普通", "spec_mixed": "混合", "spec_row_on": "推测解码", "spec_row_off": "普通解码",
        "spec_legend": "推测 = 开启推测解码（DFlash2，每步 7 个草稿）· 普通 = 关闭推测解码 · "
                       "最多 N 路并发 = 服务端同时处理的请求上限（配置的并发槽位，多出的请求排队等待）",
        "lanes": "最多 {n} 路并发", "ttft_short": "首 token",
        "cmp": "TensorSharp：GGUF 模型 vs NInfer 模型", "cmp_sub": "同一引擎构建与测试工具；加粗 = 两者中更优者",
        "cmp_rows": {"overall": "综合得分", "quality": "质量（检查通过）", "ttft_63k": "64K 预填充（s）", "ttft_127k": "128K 预填充（s）",
                     "decode_gen": "推测解码（tok/s）", "decode_nospec": "普通解码（tok/s）", "ld_60000": "64K 后解码（tok/s）",
                     "ld_120000": "128K 后解码（tok/s）", "ttft_250k": "256K 预填充（s）", "ld_250000": "256K 后解码（tok/s）",
                     "lreuse_64k": "64K 新会话·KV 复用（s）", "lmt_128k": "128K 第 2–3 轮（s）", "lpar4_64k": "64K 4 路并发（tok/s）",
                     "lpar2_256k": "256K 2 路并发（tok/s）", "agent_e2e": "智能体工具循环（s）"},
        "gguf": "GGUF 模型", "ninf": "NInfer 模型",
        "group": {"1": "最多 1 路并发", "2": "最多 2 路并发", "4": "最多 4 路并发"},
        "rank_g": "综合排名", "excluded": "未参与排名",
        "long_note": "长上下文结果全部为推测解码（DFlash2，每步 7 个草稿）。每个引擎的每个档位跑三次：最多 1、2、4 路并发；"
                     "条形与金色只在同一并发槽位数内比较。1 路时 2 / 4 个并发会话排队处理。计分时单请求各列取 1 路结果，2 / 4 路并发列取 2 / 4 路配置的结果。",
        "notrun": "未能运行", "why": "原因",
        "scen_g": "分场景性能排名 · 单请求（1 路并发）", "scen_tbl_sub": "每行是一个任务下各引擎的排名（1 = 最优，金色 = 最优）。时间越低越好，tok/s 越高越好。",
        "task": "任务", "spec_all": "全部为推测解码",
    },
}

QUALITY_LABELS = {
    "prefill": ("Long-doc retrieval", "长文档检索"), "decode": ("Chat & generation", "问答与生成"),
    "concurrency": ("Concurrent (long)", "长上下文并发"), "agent": ("Agent loop", "智能体"), "reuse": ("Reuse & multi-turn", "复用/多轮"),
    "subagent": ("Sub-agents", "子智能体"), "tools": ("Tools & skills", "工具/Skill"), "json": ("JSON", "JSON"),
    "code": ("Code", "代码"), "image": ("Image", "图像"), "reasoning": ("Reasoning Q&A", "推理问答"),
}

# (metric, higher_is_better, unit, en title, zh title, en task, zh task)
KEY_METRICS = [
    ("ttft_8k", False, "s", "Prefill: 8K-token prompt", "预填充：8K token 提示", "time to first token, 8.2K-token document", "8.2K token 文档的首 token 时间"),
    ("ttft_short", False, "s", "TTFT: short prompt", "首 token：短提示", "eight short questions", "8 个简短问题"),
    ("multiturn_ttft", False, "s", "TTFT: later turns of a chat", "首 token：多轮对话后续轮次", "turns 2-3 over a 2.9K-token record", "基于 2.9K token 记录的第 2–3 轮"),
    ("agent_e2e", False, "s", "Agentic tool loop, end to end", "智能体工具循环端到端", "read two files, call a calculator, report", "读取两个文件、调用计算器、汇报"),
]

# (metric, higher, unit, en title, zh title)
CARDS = [
    ("subagent_e2e", False, "s", "Sub-agent fan-out, end to end", "子智能体扇出端到端"),
    ("json_schema_s", False, "s", "JSON-schema reply", "Schema 约束 JSON 回复"),
    ("image_ttft", False, "s", "Image question, TTFT", "图像问答首 token"),
    ("code_run_e2e", False, "s", "Write-run-answer code loop", "代码编写-运行-作答"),
    ("skill_e2e", False, "s", "Skill call, end to end", "Skill 调用端到端"),
    ("tool_parallel_s", False, "s", "Parallel tool calls", "并行工具调用"),
    ("tool_single_s", False, "s", "Single tool call", "单次工具调用"),
    ("code_edit_s", False, "s", "Code edit to a spec", "按规格修改代码"),
]

CSS = """
:root{--navy:#13233f;--blue:#2a78d6;--blue-soft:#e3eefb;--gold:#eda100;--gold-soft:#fdf1d3;--gray:#c9c7c0;
 --ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--card:#ffffff;--page:#eef2f8;--rule:#e3e1db;--green:#e8f5ec;--green-ink:#1d6b3a}
*{box-sizing:border-box} body{margin:0;background:var(--page);font:15px/1.4 -apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;color:var(--ink)}
.page{width:1600px;padding:22px}
.hdr{display:grid;grid-template-columns:1fr 600px;gap:22px;background:var(--navy);color:#fff;border-radius:14px;padding:26px 30px}
.hdr h1{font-size:42px;line-height:1.12;margin:0 0 14px;font-weight:800} .hdr .meta{color:#c8d3e6;font-size:18px}
.tldr{border:1.5px solid #4c5f82;border-radius:12px;padding:14px 18px;font-size:17px;line-height:1.45}
.tldr .h{color:#ffd166;font-weight:800;font-size:26px;margin-bottom:4px} .tldr b{color:#ffd166} .tldr .ts{color:#7fb6ff;font-weight:700}
.row{display:grid;gap:18px;margin-top:18px}
.card{background:var(--card);border-radius:14px;padding:16px 18px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.card h2{margin:0 0 2px;font-size:24px} .sub{color:var(--ink2);font-size:13.5px;margin:0 0 10px}
svg{display:block} svg text{font-family:inherit}
.lab{fill:var(--ink2);font-size:14px} .lab.ts{fill:var(--blue);font-weight:700} .val{fill:var(--ink);font-size:13.5px;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:18px;font-size:13.5px;color:var(--ink2);margin:2px 0 6px} .sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
.badge{display:inline-block;min-width:58px;text-align:center;background:var(--green);color:var(--green-ink);border-radius:8px;padding:6px 0;font-weight:700;font-variant-numeric:tabular-nums}
.minis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.mini{border:1px solid var(--rule);border-radius:10px;padding:10px 12px} .mini .t{font-weight:700;font-size:16px} .mini .d{color:var(--ink2);font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.sc{border-radius:12px;padding:12px 14px;border:1px solid var(--rule);background:#fff}
.sc .t{font-weight:700;font-size:16px} .sc .d{color:var(--ink2);font-size:12.5px;min-height:34px;margin:2px 0 6px}
.sc table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.sc td{padding:5px 8px;font-size:15px} .sc td.v{text-align:right;white-space:nowrap}
.sc tr.ts td{background:var(--blue-soft);color:var(--blue);font-weight:700} .sc tr.best td{background:var(--gold-soft);font-weight:700}
.sc tr.ts.best td{background:var(--blue-soft);box-shadow:inset 0 -2px 0 var(--gold)}
.cmp-h{font-size:20px;margin:22px 0 2px} table.cmp{width:100%;border-collapse:collapse;font-size:14.5px}
table.cmp th{text-align:right;padding:4px 10px;font-size:14px} table.cmp th:first-child{text-align:left}
table.cmp td{padding:4px 10px;border-top:1px solid #ecebe6;text-align:right;color:#5f5e5a} table.cmp td:first-child{text-align:left;color:#2c2c2a}
table.cmp td.w{font-weight:800;color:var(--blue)}
.rk{display:grid;grid-template-columns:28px 230px 1fr 330px;align-items:center;gap:10px;margin:7px 0}
.rk .n{color:var(--muted);text-align:right;font-weight:700} .rk .nm{font-weight:700} .rk.ts .nm{color:var(--blue)}
.rk .tr{background:#f2f1ee;height:24px;border-radius:6px;position:relative} .rk .fl{height:100%;border-radius:6px;background:var(--gray)} .rk.ts .fl{background:var(--blue)}
.rk .sv{font-variant-numeric:tabular-nums;font-size:14px;color:var(--ink2)} .rk .sv b{color:var(--ink);font-size:16px}
table.q{width:100%;border-collapse:separate;border-spacing:3px;font-variant-numeric:tabular-nums}
table.q th{font-size:12.5px;color:var(--ink2);font-weight:600;text-align:center;padding:4px} table.q th:first-child{text-align:left}
table.q td{text-align:center;padding:7px 4px;border-radius:5px;font-size:14px} table.q td:first-child{text-align:left;font-weight:700;background:none}
table.q tr.ts td:first-child{color:var(--blue)} table.q td.tot{font-weight:700}
table.st{width:100%;border-collapse:collapse;font-size:13px} table.st th{text-align:left;color:var(--ink2);border-bottom:1.5px solid var(--rule);padding:6px}
table.st td{padding:6px;border-bottom:1px solid var(--rule);vertical-align:top} table.st tr.ts td{background:var(--blue-soft)}
ul.nt{margin:0;padding-left:18px;font-size:13.5px;color:var(--ink2)} ul.nt li{margin:5px 0} ul.nt b{color:var(--ink)}
.foot{display:flex;justify-content:space-between;color:var(--muted);font-size:12.5px;margin-top:12px}
.tg{font-size:10.5px;color:var(--muted);font-weight:400;margin-left:4px;letter-spacing:.02em} .tg.off{color:#a3541c}
.speclegend{display:inline-block;margin-top:10px;font-size:14px;color:#ffd166;border:1px solid #4c5f82;border-radius:8px;padding:4px 10px}
.tier{display:grid;grid-template-columns:86px 1fr;gap:12px;align-items:start;margin-top:12px}
.tier .tl{background:var(--navy);color:#fff;border-radius:10px;font-size:26px;font-weight:800;text-align:center;padding:14px 0}
table.bt{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;table-layout:fixed}
table.bt th{font-size:13px;text-align:left;color:var(--ink2);padding:3px 6px;font-weight:700;vertical-align:bottom}
table.bt th .w{display:block;font-weight:400;font-size:11px;color:var(--muted)}
table.bt td{padding:3px 6px;border-top:1px solid #efeee9;font-size:14px;white-space:nowrap}
table.bt td.nm{font-weight:700;width:150px} table.bt tr.ts td.nm{color:var(--blue)} table.bt tr.ts td{background:#f4f8fe}
table.bt .bar{display:inline-block;height:12px;border-radius:3px;background:var(--gray);vertical-align:-1px;margin-right:6px}
table.bt tr.ts .bar{background:var(--blue)} table.bt td.best{background:var(--gold-soft)!important} table.bt td.best .bar{background:var(--gold)}
table.bt td.best b{color:#7a5300} table.bt .sm{font-size:11px;color:var(--muted);margin-left:4px} table.bt td.na{color:var(--muted);font-size:12.5px}
table.bt td.sp{font-size:12.5px;color:var(--ink2);white-space:normal} table.bt td.sp.off{color:#a3541c;font-weight:700}
table.bt tr.grp td{background:#f2f1ee;font-weight:800;font-size:13px;color:var(--navy);padding:4px 6px}
table.bt td.why{white-space:normal;font-size:12px;color:#7a3d12;line-height:1.25}
table.cmp .sm2{display:block;font-size:11px;font-weight:400;color:var(--muted)}
.exc{margin-top:10px;font-size:13px;color:#7a3d12} .exc ul{margin:4px 0 0;padding-left:18px}
table.sct{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
table.sct th{font-size:13px;color:var(--ink2);text-align:right;padding:4px 6px} table.sct th:first-child{text-align:left} table.sct th.ts{color:var(--blue)}
table.sct td{padding:4px 6px;border-top:1px solid #efeee9;text-align:right;white-space:nowrap} table.sct td.ts{background:var(--blue-soft);color:var(--blue)}
table.sct td.best{background:var(--gold-soft)} table.sct td.best.ts{box-shadow:inset 0 -2px 0 var(--gold)}
table.sct td.tk{text-align:left;font-weight:700;white-space:normal} table.sct td.tk .w{display:block;font-weight:400;font-size:11px;color:var(--muted)}
table.sct .rk2{font-size:11px;color:var(--muted);margin-left:5px} table.sct td.na{color:var(--muted);font-size:12px}
"""


def fmt(v, unit):
    if v is None:
        return None
    if unit == "tok/s":
        return f"{v:,.0f}"
    if v < 0.1:
        return f"{v:.3f}"
    if v < 10:
        return f"{v:.2f}"
    return f"{v:.1f}"


def spec_of(summary, k, metric):
    return (summary["engines"][k].get("spec") or {}).get(metric)


def spec_tag(summary, k, metric, lang, svg=False):
    """'spec' / 'plain' after a value: whether speculative decoding ran for it."""
    sp = spec_of(summary, k, metric)
    if not sp:
        return ""
    text = TXT[lang]["spec_" + sp]
    if svg:
        color = "#a3541c" if sp == "off" else "#8a8984"
        return f'<tspan dx="4" style="font-size:10.5px;font-weight:{700 if sp == "off" else 400};fill:{color}">{text}</tspan>'
    return f'<span class="tg{" off" if sp == "off" else ""}">{text}</span>'


def ranked(summary, metric, higher, not_run):
    rows = []
    for k in summary["order"]:
        if metric in not_run.get(k, []):
            rows.append((k, None))
            continue
        rows.append((k, summary["engines"][k]["metrics"].get(metric)))
    have = sorted([r for r in rows if r[1]], key=lambda r: -r[1] if higher else r[1])
    return have + [r for r in rows if not r[1]]


def workload_text(summary, metric, lang):
    w = (summary.get("workloads") or {}).get(metric)
    t = TXT[lang]
    if not w:
        return ""

    def tok(v):
        return f"{v / 1000:.1f}K" if v and v >= 10000 else f"{v:,.0f}" if v else "?"
    p = tok(w.get("prompt"))
    if w.get("prompt_max") and w.get("prompt") and w["prompt_max"] > 1.2 * w["prompt"]:
        p += "–" + tok(w["prompt_max"])
    n = w.get("requests") or 1
    req = (f"{n:g} {t['req']} · " if lang == "en" else f"{n:g} {t['req']} · ") if n > 1 else ""
    if lang == "zh":
        return f"{req}提示 {p} token · 每请求生成 {tok(w.get('output'))} token"
    return f"{req}prompt {p} tok · {tok(w.get('output'))} tok generated per request"


def decode_chart(summary, lang):
    """Grouped horizontal bars (plain gray, speculative blue) with the speed-up badge per row."""
    t = TXT[lang]
    rows = []
    for k in summary["order"]:
        m = summary["engines"][k]["metrics"]
        rows.append((k, m.get("decode_nospec"), m.get("decode_gen")))
    rows.sort(key=lambda r: -(r[2] or 0))
    vmax = max(max(r[1] or 0, r[2] or 0) for r in rows)
    w, rowh, left, right, badge = 740, 60, 124, 150, 84
    h = len(rows) * rowh + 44
    plot = w - left - right
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">',
           f'<text x="{w - badge / 2 - 6}" y="12" text-anchor="middle" class="lab" style="font-weight:700;font-size:13px">{t["speedup"]}</text>']
    for i, (k, plain, spec) in enumerate(rows):
        y = 20 + i * rowh
        ts = k in TS
        if ts:
            out.append(f'<rect x="0" y="{y - 2}" width="{w}" height="{rowh - 4}" rx="6" fill="var(--blue-soft)"/>')
        out.append(f'<text x="{left - 10}" y="{y + 31}" text-anchor="end" class="lab{" ts" if ts else ""}" style="font-size:15px">{html.escape(SHORT[k] if ts else NAMES[k])}</text>')
        for j, (v, color) in enumerate(((plain, "var(--gray)"), (spec, "var(--blue)"))):
            yy = y + 5 + j * 24
            if v:
                bw = max(3, plot * v / vmax)
                out.append(f'<rect x="{left}" y="{yy}" width="{bw:.1f}" height="18" rx="3" fill="{color}"/>')
                out.append(f'<text x="{left + bw + 6:.1f}" y="{yy + 14}" class="val" style="font-weight:{700 if j == 1 else 400}">{v:,.0f}</text>')
            else:
                out.append(f'<text x="{left + 4}" y="{yy + 14}" class="val" style="fill:#8a8984">{t["na"]}</text>')
        s_txt = f"×{spec / plain:.1f}" if plain and spec else "–"
        bx = w - badge - 6
        out.append(f'<rect x="{bx}" y="{y + 9}" width="{badge}" height="34" rx="8" fill="var(--green)"/>')
        out.append(f'<text x="{bx + badge / 2}" y="{y + 31}" text-anchor="middle" style="fill:var(--green-ink);font-weight:700;font-size:15px">{s_txt}</text>')
    axis_y = 20 + len(rows) * rowh - 4
    out.append(f'<line x1="{left}" x2="{left}" y1="16" y2="{axis_y}" stroke="#e3e1db"/>')
    for tick in (0, 100, 200, 300):
        if tick > vmax * 1.02:
            break
        x = left + plot * tick / vmax
        if x > left + plot - 44:   # keep clear of the unit label at the end of the axis
            continue
        out.append(f'<text x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle" class="lab" style="font-size:12px">{tick}</text>')
    out.append(f'<text x="{left + plot}" y="{axis_y + 16}" text-anchor="end" class="lab" style="font-size:12px">tok/s</text></svg>')
    return "".join(out)


def tint(pct):
    """Sequential single-hue tint for a pass rate (light enough for dark text)."""
    if pct is None:
        return "#f2f1ee"
    a = max(0.0, min(1.0, (pct - 40) / 60))
    r = round(250 - a * (250 - 168)); g = round(250 - a * (250 - 201)); b = round(252 - a * (252 - 240))
    return f"rgb({r},{g},{b})"


def quality_table(summary, lang):
    t = TXT[lang]
    cats = [c for c in summary["categories"] if c["quality"]]
    head = "".join(f'<th>{html.escape(QUALITY_LABELS.get(c["id"], (c["en"], c["zh"]))[1 if lang == "zh" else 0])}</th>' for c in cats)
    rows = []
    for k in summary["order"]:
        e = summary["engines"][k]
        cells = []
        for c in cats:
            cat = e["categories"][c["id"]]
            ok, n = cat["checks"]
            pct = 100.0 * ok / n if n else None
            label = f"{pct:.0f}%" if pct is not None else "–"
            sps = {spec_of(summary, k, mid) for mid in c["perf"]} - {None}
            sp = (sps.pop() if len(sps) == 1 else "mixed") if sps else None
            tg = (f' · <span style="color:{"#a3541c" if sp == "off" else "#52514e"}">{t["spec_" + sp]}</span>') if sp else ""
            cells.append(f'<td style="background:{tint(pct)}" title="{ok}/{n}">{label}<div style="font-size:10.5px;color:#52514e">{ok}/{n}{tg}</div></td>')
        ok, n = e["checks"]
        rows.append(f'<tr class="{"ts" if k in TS else ""}"><td>{html.escape(name(k, lang))}</td>{"".join(cells)}'
                    f'<td class="tot" style="background:{tint(100.0 * ok / n)}">{100.0 * ok / n:.1f}%<div style="font-size:10.5px;color:#52514e">{ok}/{n}</div></td></tr>')
    return f'<table class="q"><tr><th>{t["engine"]}</th>{head}<th>{t["total"]}</th></tr>{"".join(rows)}</table>'


def ranking_block(summary, lang):
    t = TXT[lang]
    out = []
    for k in summary["order"]:
        e = summary["engines"][k]
        ok, n = e["checks"]
        out.append(f'<div class="rk{" ts" if k in TS else ""}"><span class="n">{e["rank"]}</span><span class="nm">{html.escape(name(k, lang))}</span>'
                   f'<span class="tr"><span class="fl" style="width:{e["overall"]:.1f}%;display:block"></span></span>'
                   f'<span class="sv"><b>{e["overall"]:.1f}</b> &nbsp;{t["quality"]} {e["quality"]:.1f} · {t["perf"]} {e["performance"]:.1f} · {ok}/{n}</span></div>')
    return "".join(out)


def ts_compare(summary, lang):
    """The two TensorSharp deployments side by side: score, quality and the headline metrics."""
    t = TXT[lang]
    e = summary["engines"]
    if "tensorsharp-gguf" not in e or "tensorsharp" not in e:
        return ""
    gg, ni = e["tensorsharp-gguf"], e["tensorsharp"]
    rows = [("overall", True), ("quality", True), ("ttft_63k", False), ("ttft_127k", False), ("ttft_250k", False),
            ("lreuse_64k", False), ("lmt_128k", False), ("lpar4_64k", True), ("lpar2_256k", True), ("decode_gen", True),
            ("decode_nospec", True), ("ld_60000", True), ("ld_120000", True), ("ld_250000", True), ("agent_e2e", False)]
    out = [f'<h3 class="cmp-h">{html.escape(t["cmp"])}</h3><p class="sub">{html.escape(t["cmp_sub"])}</p>'
           f'<table class="cmp"><tr><th></th><th class="ts">{t["gguf"]}</th><th class="ts">{t["ninf"]}</th></tr>']
    for key, hib in rows:
        if key == "overall":
            a, b, fa, fb = gg["overall"], ni["overall"], f'{gg["overall"]:.1f}', f'{ni["overall"]:.1f}'
        elif key == "quality":
            a, b = gg["checks"][0] / gg["checks"][1], ni["checks"][0] / ni["checks"][1]
            fa, fb = f'{gg["checks"][0]}/{gg["checks"][1]}', f'{ni["checks"][0]}/{ni["checks"][1]}'
        else:
            a, b = gg["metrics"].get(key), ni["metrics"].get(key)
            unit = "s" if key.startswith(("ttft", "lreuse", "lmt")) or key.endswith("_e2e") else "tok/s"
            fa = (fmt(a, unit) + spec_tag(summary, "tensorsharp-gguf", key, lang)) if a else t["na"]
            fb = (fmt(b, unit) + spec_tag(summary, "tensorsharp", key, lang)) if b else t["na"]
            if not a and not b:
                continue
        wa = a is not None and (b is None or a == b or (a > b) == hib)
        wb = b is not None and (a is None or a == b or (b > a) == hib)
        out.append(f'<tr><td>{html.escape(t["cmp_rows"][key])}</td><td{" class=w" if wa else ""}>{fa}</td><td{" class=w" if wb else ""}>{fb}</td></tr>')
    out.append("</table>")
    return "".join(out)


def reason_text(ctx, lang, engine, mode, lanes, meta):
    """Why a configuration did not run or ran slowly: the curated text from the context, else the campaign's reason."""
    key = f"{engine}-{mode}-l{lanes}"
    curated = (ctx.get(lang, {}).get("reasons") or {}).get(key)
    if curated:
        return curated
    if meta and meta.get("failed"):
        return meta.get("reason") or TXT[lang]["notrun"]
    return None


LANES = ("1", "2", "4")


def tier_table(summary, tier, lang, ctx):
    """One context tier: per lane count (max 1, 2, 4 concurrent), every engine's TTFT, decode, KV reuse,
    multi-turn and concurrency as a bar table (bars and gold compare within the lane count only; blue rows =
    TensorSharp). A configuration that did not run shows its reason; a slow one carries its reason too."""
    t = TXT[lang]
    cols = [(TIER_TTFT[tier], False, "s", "ttft"), (TIER_LD[tier], True, "tok/s", "ld"), (f"lreuse_{tier}", False, "s", "reuse"),
            (f"lmt_{tier}", False, "s", "mt"), (f"lpar2_{tier}", True, "tok/s", "p2"), (f"lpar4_{tier}", True, "tok/s", "p4")]
    eng = summary["engines"]
    head = ['<th style="width:150px"></th>']
    for mid, hib, unit, key in cols:
        head.append(f'<th>{html.escape(t["cols"][key])}<span class="w">{html.escape(workload_text(summary, mid, lang))}'
                    f' · {t["higher"] if hib else t["lower"]}</span></th>')
    head.append(f'<th style="width:250px">{t["why"]}</th>')
    rows = []
    for g in LANES:
        sfx = f"@l{g}"
        present = [k for k in summary["order"] if (eng[k].get("meta") or {}).get(f"{tier}{sfx}")]
        if not present:
            continue
        best = {}
        for mid, hib, _, _ in cols:
            vals = [eng[k]["metrics"].get(mid + sfx) for k in present if eng[k]["metrics"].get(mid + sfx)]
            best[mid] = ((max if hib else min)(vals), max(vals)) if vals else (None, None)
        rows.append(f'<tr class="grp"><td colspan="{len(cols) + 2}">{html.escape(t["group"][g])}</td></tr>')
        for k in present:
            meta = eng[k]["meta"][f"{tier}{sfx}"]
            m = eng[k]["metrics"]
            why = reason_text(ctx, lang, k, tier, g, meta)
            cells = [f'<td class="nm">{html.escape(SHORT[k] if k in TS else NAMES[k])}</td>']
            if meta.get("failed") or not any(m.get(c[0] + sfx) for c in cols):
                cells.append(f'<td class="na" colspan="{len(cols)}">{t["notrun"]}</td>')
                cells.append(f'<td class="why">{html.escape(why or "")}</td>')
                rows.append(f'<tr class="{"ts" if k in TS else ""}">{"".join(cells)}</tr>')
                continue
            for mid, hib, unit, key in cols:
                v = m.get(mid + sfx)
                if not v:
                    cells.append(f'<td class="na">{t["na"]}</td>')
                    continue
                b, vmax = best[mid]
                sub = ""
                if key in ("p2", "p4"):
                    tt = m.get(f"lpar{key[1]}_ttft_{tier}{sfx}")
                    sub = f'<span class="sm">{t["ttft_short"]} {fmt(tt, "s")} s</span>' if tt else ""
                cells.append(f'<td class="{"best" if v == b else ""}"><span class="bar" style="width:{max(3, 46 * v / vmax):.0f}px"></span>'
                             f'<b>{fmt(v, unit)}</b>{sub}</td>')
            cells.append(f'<td class="why">{html.escape(why or "")}</td>')
            rows.append(f'<tr class="{"ts" if k in TS else ""}">{"".join(cells)}</tr>')
    return (f'<div class="tier"><div class="tl">{TIER_LABEL[tier]}</div><table class="bt"><tr>{"".join(head)}</tr>'
            f'{"".join(rows)}</table></div>')


def scen_table(summary, lang):
    """Per-scenario ranking (one request at a time): rows = tasks, columns = engines (overall order), value + rank."""
    t = TXT[lang]
    eng = summary["engines"]
    order = summary["order"]
    items = [(mid, hib, unit, zh if lang == "zh" else en) for mid, hib, unit, en, zh, _, _ in KEY_METRICS] + \
            [(mid, hib, unit, zh if lang == "zh" else en) for mid, hib, unit, en, zh in CARDS]
    head = "".join(f'<th class="{"ts" if k in TS else ""}">{html.escape(SHORT[k])}</th>' for k in order)
    out = [f'<table class="sct"><tr><th>{t["task"]}</th>{head}</tr>']
    for mid, hib, unit, title in items:
        vals = {k: eng[k]["metrics"].get(mid) for k in order}
        have = sorted([k for k in order if vals[k]], key=lambda k: -vals[k] if hib else vals[k])
        if not have:
            continue
        rank = {k: i + 1 for i, k in enumerate(have)}
        cells = []
        for k in order:
            v = vals[k]
            if not v:
                cells.append(f'<td class="na">{t["na"]}</td>')
                continue
            cls = ("best " if rank[k] == 1 else "") + ("ts" if k in TS else "")
            unit_txt = " s" if unit == "s" else ""
            cells.append(f'<td class="{cls.strip()}"><b>{fmt(v, unit)}{unit_txt}</b>{spec_tag(summary, k, mid, lang) if spec_of(summary, k, mid) == "off" else ""}'
                         f'<span class="rk2">#{rank[k]}</span></td>')
        desc = workload_text(summary, mid, lang)
        out.append(f'<tr><td class="tk">{html.escape(title)}<span class="w">{html.escape(desc)} · {t["higher"] if hib else t["lower"]}</span></td>{"".join(cells)}</tr>')
    out.append("</table>")
    return "".join(out)


def excluded_block(ctx, lang):
    ex = ctx.get("excluded") or {}
    if not ex:
        return ""
    items = "".join(f'<li><b>{html.escape(name(k, lang))}</b>: {html.escape(v.get(lang) or "")}</li>' for k, v in ex.items())
    return f'<div class="exc"><b>{TXT[lang]["excluded"]}</b><ul>{items}</ul></div>'


def tldr(summary, lang, tail=""):
    t = TXT[lang]
    o = summary["order"]
    e = summary["engines"]
    n = lambda k: name(k, lang)
    first = o[0]
    ts_bits = [(k, e[k]) for k in TS if k in e and k != first]
    q = lambda k: f'{e[k]["checks"][0]}/{e[k]["checks"][1]}'
    runner = next((k for k in o if k not in TS), None)

    def who(x, hib, unit):
        have = [k for k in o if e[k]["metrics"].get(x)]
        if not have:
            return None
        k = (max if hib else min)(have, key=lambda k: e[k]["metrics"][x])
        v = e[k]["metrics"][x]
        val = f"{v:.0f} tok/s" if unit == "tok/s" else (f"{v:.2f} s" if v < 10 else f"{v:.1f} s")
        return f'{SHORT[k]} <b>{val}</b>'

    parts = []
    for tier in TIERS:
        a, b = who(TIER_TTFT[tier], False, "s"), who(f"lpar4_{tier}", True, "tok/s")
        bits = [x for x in ((("首 token 最快 " if lang == "zh" else "fastest TTFT ") + a) if a else None,
                            (("4 路并发最快 " if lang == "zh" else "4 concurrent ") + b) if b else None) if x]
        if bits:
            parts.append(f"{TIER_LABEL[tier]}" + ("：" if lang == "zh" else ": ") + ("，" if lang == "zh" else ", ").join(bits))
    if lang == "zh":
        head = (f'<span class="ts">{n(first)}</span>' if first in TS else n(first)) + f' 综合 <b>{e[first]["overall"]:.1f}</b> 排名<b>第一</b>（质量 {q(first)}）'
        if first in TS and runner:
            head += f'，领先 {n(runner)}（{e[runner]["overall"]:.1f}）'
        head += "".join(f'；<span class="ts">{"GGUF" if k == "tensorsharp-gguf" else "NInfer"} 模型</span> 第 <b>{v["rank"]}</b>（{v["overall"]:.1f}）' for k, v in ts_bits) + "。"
        return head + "长上下文 —— " + "；".join(parts) + "。" + tail
    head = (f'<span class="ts">{n(first)}</span>' if first in TS else n(first)) + f' ranks <b>#1</b> overall (<b>{e[first]["overall"]:.1f}</b>, {q(first)} checks)'
    if first in TS and runner:
        head += f', ahead of {n(runner)} ({e[runner]["overall"]:.1f})'
    head += "".join(f'; the <span class="ts">{"GGUF" if k == "tensorsharp-gguf" else "NInfer"} model</span> ranks <b>#{v["rank"]}</b> ({v["overall"]:.1f})' for k, v in ts_bits) + ". "
    return head + "Long context &mdash; " + "; ".join(parts) + ". " + tail


def render(summary, ctx, lang):
    t = TXT[lang]
    c = ctx[lang]
    longs = "".join(tier_table(summary, tier, lang, ctx) for tier in TIERS)
    setup_rows = "".join(f'<tr class="{"ts" if k in TS else ""}"><td><b>{html.escape(name(k, lang))}</b></td>'
                         + "".join(f"<td>{html.escape(x)}</td>" for x in c["setup"][k]) + "</tr>"
                         for k in summary["order"] if k in c["setup"])
    notes = "".join(f"<li>{x}</li>" for x in c["notes"])
    others = sum(1 for k in summary["order"] if k not in TS)
    count = {"en": {4: "Four", 5: "Five", 6: "Six", 7: "Seven"}, "zh": {4: "四", 5: "五", 6: "六", 7: "七"}}[lang].get(others, str(others))
    title = t["title"].replace("Six", count) if lang == "en" else t["title"].replace("六", count)
    rank_sub = t["rank_sub"].format(n=len(summary["categories"]))
    return f"""<!DOCTYPE html><html lang="{'zh-CN' if lang == 'zh' else 'en'}"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="page">
<div class="hdr"><div><h1>{html.escape(title)}</h1><div class="meta">{html.escape(c['meta'])}</div><div class="speclegend">{html.escape(t['spec_legend'])}</div></div>
<div class="tldr"><div class="h">{t['tldr']}</div>{tldr(summary, lang, c.get('tldr_tail', ''))}</div></div>
<div class="row" style="grid-template-columns:1fr 1fr">
 <div class="card"><h2>{t['rank_g']}</h2><p class="sub">{html.escape(rank_sub)}</p>{ranking_block(summary, lang)}{excluded_block(ctx, lang)}{ts_compare(summary, lang)}</div>
 <div class="card"><h2>{t['decode']}</h2><p class="sub">{html.escape(c.get('decode_sub', t['decode_sub']))}</p>
  <div class="legend"><span><span class="sw" style="background:var(--gray)"></span>{t['plain']}</span><span><span class="sw" style="background:var(--blue)"></span>{t['spec']}</span></div>
  {decode_chart(summary, lang)}</div>
</div>
<div class="row"><div class="card" style="border:2px solid var(--blue)"><h2>{t['long']}</h2><p class="sub">{html.escape(t['long_note'])} {html.escape(t['long_sub'])}</p>{longs}</div></div>
<div class="row"><div class="card"><h2>{html.escape(t['scen_g'])}</h2><p class="sub">{html.escape(t['scen_tbl_sub'])} {html.escape(c.get('scen_sub', ''))}</p>{scen_table(summary, lang)}</div></div>
<div class="row"><div class="card"><h2>{t['qual']}</h2><p class="sub">{html.escape(c['qual_sub'])}</p>{quality_table(summary, lang)}</div></div>
<div class="row" style="grid-template-columns:1.25fr 1fr">
 <div class="card"><h2>{t['setup']}</h2><p class="sub">{html.escape(c['setup_sub'])}</p><table class="st"><tr><th>{t['engine']}</th><th>{t['version']}</th><th>{t['weights']}</th><th>{t['kv']}</th><th>{t['drafter']}</th></tr>{setup_rows}</table></div>
 <div class="card"><h2>{t['notes']}</h2><ul class="nt">{notes}</ul></div>
</div>
<div class="foot"><span>{html.escape(c['foot_left'])}</span><span>{html.escape(c['foot_right'])}</span></div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--context", type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.report / "summary.json").read_text(encoding="utf-8"))
    ctx = json.loads(a.context.read_text(encoding="utf-8"))
    for lang in ("en", "zh"):
        (a.report / f"poster2_{lang}.html").write_text(render(summary, ctx, lang), encoding="utf-8")
        print(a.report / f"poster2_{lang}.html")


if __name__ == "__main__":
    main()
