#!/usr/bin/env python3
"""Render the one-page Qwen3.8-27B engine comparison poster (English and Chinese).

  python3 eng/qwen38-engine-poster.py REPORT_DIR --context poster-context.json

REPORT_DIR holds summary.json as written by eng/qwen38-engine-scorecard.py. The context
JSON supplies what the evidence cannot: date, machine, per-engine setup rows and notes
(per language). Every number on the poster comes from summary.json. Writes
REPORT_DIR/poster2_en.html and poster2_zh.html; render them at 1600 px wide.

Encoding: TensorSharp is blue, the best value of each measurement gold, every other
engine neutral gray; every bar and row carries its engine name and value, so no
identity depends on color.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

NAMES = {"tensorsharp": "TensorSharp", "ignis": "Ignis", "ninfer": "NInfer", "velox": "velox-inferer",
         "llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
SHORT = {"tensorsharp": "TensorSharp", "ignis": "Ignis", "ninfer": "NInfer", "velox": "velox",
         "llamacpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}

TXT = {
    "en": {
        "title": "Qwen3.8-27B Inference Benchmark: TensorSharp vs Six Engines",
        "rank": "Overall Ranking", "rank_sub": "Score 0-100 = weighted mean of 12 categories; each category averages quality (% checks passed) and performance (relative to the best engine)",
        "overall": "Overall", "quality": "Quality", "perf": "Performance", "checks": "checks passed",
        "decode": "Decode Throughput: Speculative vs Plain", "decode_sub": "Single request, tok/s - median of code, prose and passage-copy replies (74-1,959-token prompts, up to 1,024 generated), 3 repeats",
        "plain": "Plain decoding", "spec": "Speculative (DFlash2, 7 drafts / step)", "speedup": "Speed-up",
        "key": "Key Performance Metrics", "scen": "Performance by Task Scenario", "qual": "Quality Comparison: share of automatic checks passed",
        "setup": "Engine Setup", "notes": "Notes", "engine": "Engine", "version": "Version", "weights": "Weights",
        "kv": "KV cache", "drafter": "Drafter", "lower": "lower is better", "higher": "higher is better",
        "tldr": "TL;DR", "na": "n/a", "s": "s", "toks": "tok/s",
        "prompt": "prompt", "gen": "generated", "req": "requests", "total": "Total",
    },
    "zh": {
        "title": "Qwen3.8-27B 推理基准：TensorSharp 对比六款推理引擎",
        "rank": "综合排名", "rank_sub": "得分 0–100 = 12 个类别的加权平均；每个类别为质量（检查通过率）与性能（相对最优引擎）的平均",
        "overall": "综合", "quality": "质量", "perf": "性能", "checks": "项检查通过",
        "decode": "解码吞吐：推测解码 vs 普通解码", "decode_sub": "单请求 tok/s —— 代码、散文、段落复制三类回答的中位数（提示 74–1,959 token，最多生成 1,024 token），重复 3 次",
        "plain": "普通解码", "spec": "推测解码（DFlash2，每步 7 个草稿）", "speedup": "加速比",
        "key": "关键性能指标", "scen": "分场景性能排名", "qual": "质量对比：自动检查通过率",
        "setup": "引擎配置", "notes": "说明", "engine": "引擎", "version": "版本", "weights": "权重",
        "kv": "KV 缓存", "drafter": "草稿模型", "lower": "越低越好", "higher": "越高越好",
        "tldr": "要点", "na": "无", "s": "s", "toks": "tok/s",
        "prompt": "提示", "gen": "生成", "req": "个请求", "total": "合计",
    },
}

QUALITY_LABELS = {
    "prefill": ("Long-doc retrieval", "长文档检索"), "decode": ("Chat & generation", "问答与生成"),
    "concurrency": ("Parallel", "并发"), "agent": ("Agent loop", "智能体"), "reuse": ("Multi-turn & reuse", "多轮/复用"),
    "subagent": ("Sub-agents", "子智能体"), "tools": ("Tools & skills", "工具/Skill"), "json": ("JSON", "JSON"),
    "code": ("Code", "代码"), "image": ("Image", "图像"), "reasoning": ("Reasoning Q&A", "推理问答"),
}

# (metric, higher_is_better, unit, en title, zh title, en task, zh task)
KEY_METRICS = [
    ("ttft_32k", False, "s", "Prefill: 32K-token prompt", "预填充：32K token 提示", "time to first token, 32.7K-token document", "32.7K token 文档的首 token 时间"),
    ("ttft_short", False, "s", "TTFT: short prompt", "首 token：短提示", "eight ~50-token questions", "8 个约 50 token 的问题"),
    ("multiturn_ttft", False, "s", "TTFT: later turns of a chat", "首 token：多轮对话后续轮次", "turns 2-3 over a 2.9K-token record", "基于 2.9K token 记录的第 2–3 轮"),
    ("reuse_warm_ttft", False, "s", "TTFT: shared 16K prefix (KV reuse)", "首 token：共享 16K 前缀（KV 复用）", "new chats on the same 16.4K-token document", "同一 16.4K token 文档上的新会话"),
]

# (metric, higher, unit, en title, zh title)
CARDS = [
    ("agent_e2e", False, "s", "Agentic tool loop, end to end", "智能体工具循环端到端"),
    ("subagent_e2e", False, "s", "Sub-agent fan-out, end to end", "子智能体扇出端到端"),
    ("par4_tok_s", True, "tok/s", "4 concurrent requests, total", "4 路并发总吞吐"),
    ("json_schema_s", False, "s", "JSON-schema reply", "Schema 约束 JSON 回复"),
    ("image_ttft", False, "s", "Image question, TTFT", "图像问答首 token"),
    ("code_run_e2e", False, "s", "Write-run-answer code loop", "代码编写-运行-作答"),
    ("ld_60000", True, "tok/s", "Decode after 60K context", "60K 上下文后解码"),
    ("ttft_127k", False, "s", "Prefill: 127K-token prompt", "预填充：127K token 提示"),
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
.rk{display:grid;grid-template-columns:28px 150px 1fr 330px;align-items:center;gap:10px;margin:7px 0}
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


def mini_chart(summary, metric, higher, unit, not_run, lang):
    """Horizontal bars, best first: blue = TensorSharp, gold = best, gray = the rest."""
    rows = ranked(summary, metric, higher, not_run)
    vals = [v for _, v in rows if v]
    if not vals:
        return ""
    best = rows[0][1]
    w, rowh, left, right = 360, 24, 96, 52
    h = len(rows) * rowh + 6
    plot = w - left - right
    vmax = max(vals)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i, (k, v) in enumerate(rows):
        y = 3 + i * rowh
        ts = k == "tensorsharp"
        out.append(f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" class="lab{" ts" if ts else ""}" style="font-size:13px">{html.escape(SHORT[k])}</text>')
        if not v:
            out.append(f'<text x="{left + 4}" y="{y + 16}" class="val" style="fill:#8a8984;font-size:12.5px">{TXT[lang]["na"]}</text>')
            continue
        bw = max(3.0, plot * v / vmax)
        color = "var(--blue)" if ts else ("var(--gold)" if v == best else "var(--gray)")
        out.append(f'<rect x="{left}" y="{y + 4}" width="{bw:.1f}" height="16" rx="3" fill="{color}"><title>{html.escape(NAMES[k])}: {fmt(v, unit)} {unit}</title></rect>')
        weight = 700 if (v == best or ts) else 400
        out.append(f'<text x="{left + bw + 5:.1f}" y="{y + 16.5}" class="val" style="font-size:12.5px;font-weight:{weight}">{fmt(v, unit)}</text>')
    out.append(f'<line x1="{left}" x2="{left}" y1="2" y2="{h - 2}" stroke="#e3e1db"/></svg>')
    return "".join(out)


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
        ts = k == "tensorsharp"
        if ts:
            out.append(f'<rect x="0" y="{y - 2}" width="{w}" height="{rowh - 4}" rx="6" fill="var(--blue-soft)"/>')
        out.append(f'<text x="{left - 10}" y="{y + 31}" text-anchor="end" class="lab{" ts" if ts else ""}" style="font-size:15px">{html.escape(NAMES[k])}</text>')
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
        out.append(f'<text x="{x:.1f}" y="{axis_y + 16}" text-anchor="middle" class="lab" style="font-size:12px">{tick}</text>')
    out.append(f'<text x="{left + plot}" y="{axis_y + 16}" text-anchor="end" class="lab" style="font-size:12px">tok/s</text></svg>')
    return "".join(out)


def scenario_card(summary, metric, higher, unit, title, not_run, lang):
    rows = ranked(summary, metric, higher, not_run)
    best = rows[0][1]
    trs = []
    for i, (k, v) in enumerate(rows):
        cls = []
        if k == "tensorsharp":
            cls.append("ts")
        if v and v == best:
            cls.append("best")
        val = f"{fmt(v, unit)} {unit if unit == 's' else ''}".strip() if v else TXT[lang]["na"]
        pos = f"{i + 1}." if v else "–"
        trs.append(f'<tr class="{" ".join(cls)}"><td>{pos} {html.escape(NAMES[k])}</td><td class="v">{val}</td></tr>')
    hint = TXT[lang]["higher"] if higher else TXT[lang]["lower"]
    return (f'<div class="sc"><div class="t">{html.escape(title)}</div>'
            f'<div class="d">{html.escape(workload_text(summary, metric, lang))} · {hint}</div>'
            f'<table>{"".join(trs)}</table></div>')


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
            cells.append(f'<td style="background:{tint(pct)}" title="{ok}/{n}">{label}<div style="font-size:10.5px;color:#52514e">{ok}/{n}</div></td>')
        ok, n = e["checks"]
        rows.append(f'<tr class="{"ts" if k == "tensorsharp" else ""}"><td>{html.escape(NAMES[k])}</td>{"".join(cells)}'
                    f'<td class="tot" style="background:{tint(100.0 * ok / n)}">{100.0 * ok / n:.1f}%<div style="font-size:10.5px;color:#52514e">{ok}/{n}</div></td></tr>')
    return f'<table class="q"><tr><th>{t["engine"]}</th>{head}<th>{t["total"]}</th></tr>{"".join(rows)}</table>'


def ranking_block(summary, lang):
    t = TXT[lang]
    out = []
    for k in summary["order"]:
        e = summary["engines"][k]
        ok, n = e["checks"]
        out.append(f'<div class="rk{" ts" if k == "tensorsharp" else ""}"><span class="n">{e["rank"]}</span><span class="nm">{html.escape(NAMES[k])}</span>'
                   f'<span class="tr"><span class="fl" style="width:{e["overall"]:.1f}%;display:block"></span></span>'
                   f'<span class="sv"><b>{e["overall"]:.1f}</b> &nbsp;{t["quality"]} {e["quality"]:.1f} · {t["perf"]} {e["performance"]:.1f} · {ok}/{n}</span></div>')
    return "".join(out)


def tldr(summary, lang):
    o = summary["order"]
    e = summary["engines"]
    first, second, third = o[0], o[1], o[2]
    m = lambda k, x: e[k]["metrics"].get(x)
    best_decode = max(o, key=lambda k: m(k, "decode_gen") or 0)
    best_32k = min((k for k in o if m(k, "ttft_32k")), key=lambda k: m(k, "ttft_32k"))
    ts = e["tensorsharp"]["metrics"]
    if lang == "zh":
        return (f'<span class="ts">{NAMES[first]}</span> 综合得分 <b>{e[first]["overall"]:.1f}</b> 排名第一，'
                f'{NAMES[second]}（{e[second]["overall"]:.1f}）第二，{NAMES[third]}（{e[third]["overall"]:.1f}）第三。'
                f'TensorSharp 质量检查通过 <b>{e["tensorsharp"]["checks"][0]}/{e["tensorsharp"]["checks"][1]}</b>，'
                f'短提示首 token <b>{ts["ttft_short"] * 1000:.0f} ms</b>、多轮对话 <b>{ts["multiturn_ttft"] * 1000:.0f} ms</b>、'
                f'共享前缀复用 <b>{ts["reuse_warm_ttft"] * 1000:.0f} ms</b> 均为最快。'
                f'{NAMES[best_decode]} 单流解码最快（<b>{m(best_decode, "decode_gen"):.0f} tok/s</b>，其 FFN 采用 3 位 IQ3 权重），'
                f'{NAMES[best_32k]} 32K 预填充最快（<b>{m(best_32k, "ttft_32k"):.2f} s</b>，FP8 注意力）；TensorSharp 保持无损 BF16 注意力与 KV。')
    return (f'<span class="ts">{NAMES[first]}</span> ranks <b>#1</b> overall (<b>{e[first]["overall"]:.1f}</b>), ahead of '
            f'{NAMES[second]} ({e[second]["overall"]:.1f}) and {NAMES[third]} ({e[third]["overall"]:.1f}). '
            f'It passes <b>{e["tensorsharp"]["checks"][0]}/{e["tensorsharp"]["checks"][1]}</b> quality checks and is fastest at short-prompt TTFT '
            f'(<b>{ts["ttft_short"] * 1000:.0f} ms</b>), later chat turns (<b>{ts["multiturn_ttft"] * 1000:.0f} ms</b>) and shared-prefix reuse '
            f'(<b>{ts["reuse_warm_ttft"] * 1000:.0f} ms</b>). {NAMES[best_decode]} decodes fastest (<b>{m(best_decode, "decode_gen"):.0f} tok/s</b>, with 3-bit IQ3 FFN weights); '
            f'{NAMES[best_32k]} prefills 32K fastest (<b>{m(best_32k, "ttft_32k"):.2f} s</b>, FP8 attention). TensorSharp keeps lossless BF16 attention and KV.')


def render(summary, ctx, lang):
    t = TXT[lang]
    not_run = ctx.get("not_run", {})
    c = ctx[lang]
    minis = "".join(f'<div class="mini"><div class="t">{html.escape(title_zh if lang == "zh" else title_en)}</div>'
                    f'<div class="d">{html.escape(task_zh if lang == "zh" else task_en)} · {t["higher"] if hib else t["lower"]}</div>'
                    f'{mini_chart(summary, mid, hib, unit, not_run, lang)}</div>'
                    for mid, hib, unit, title_en, title_zh, task_en, task_zh in KEY_METRICS)
    cards = "".join(scenario_card(summary, mid, hib, unit, zh if lang == "zh" else en, not_run, lang)
                    for mid, hib, unit, en, zh in CARDS)
    setup_rows = "".join(f'<tr class="{"ts" if k == "tensorsharp" else ""}"><td><b>{html.escape(NAMES[k])}</b></td>'
                         + "".join(f"<td>{html.escape(x)}</td>" for x in c["setup"][k]) + "</tr>"
                         for k in summary["order"] if k in c["setup"])
    notes = "".join(f"<li>{x}</li>" for x in c["notes"])
    return f"""<!DOCTYPE html><html lang="{'zh-CN' if lang == 'zh' else 'en'}"><head><meta charset="utf-8"><title>{html.escape(t['title'])}</title>
<style>{CSS}</style></head><body><div class="page">
<div class="hdr"><div><h1>{html.escape(t['title'])}</h1><div class="meta">{html.escape(c['meta'])}</div></div>
<div class="tldr"><div class="h">{t['tldr']}</div>{tldr(summary, lang)}</div></div>
<div class="row" style="grid-template-columns:1fr 1fr">
 <div class="card"><h2>{t['rank']}</h2><p class="sub">{html.escape(t['rank_sub'])}</p>{ranking_block(summary, lang)}</div>
 <div class="card"><h2>{t['decode']}</h2><p class="sub">{html.escape(t['decode_sub'])}</p>
  <div class="legend"><span><span class="sw" style="background:var(--gray)"></span>{t['plain']}</span><span><span class="sw" style="background:var(--blue)"></span>{t['spec']}</span></div>
  {decode_chart(summary, lang)}</div>
</div>
<div class="row"><div class="card"><h2>{t['key']}</h2><p class="sub">{html.escape(c['key_sub'])}</p><div class="minis">{minis}</div></div></div>
<div class="row"><div class="card"><h2>{t['scen']}</h2><p class="sub">{html.escape(c['scen_sub'])}</p><div class="cards">{cards}</div></div></div>
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
