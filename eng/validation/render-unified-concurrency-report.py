#!/usr/bin/env python3
"""Render recorded unified-memory concurrency evidence; never run a model.

Usage: python render-unified-concurrency-report.py <evidence-directory>
The directory contains summary.json, concurrency/report.json, launch-audit.json.
Only report.html is written. Inputs are preserved and SHA-256 linked in the HTML.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def escaped(value):
    return html.escape(str(value), quote=True)


def number(value, digits=2, divisor=1):
    if isinstance(value, bool) or not isinstance(value, (float, int)) \
            or not math.isfinite(value) or value < 0:
        return "未知"
    return f"{value / divisor:,.{digits}f}"


def status(value):
    if value is True:
        return '<span class="pass">通过</span>'
    if value is False:
        return '<span class="fail">未通过</span>'
    return '<span class="unknown">未知／未记录</span>'


def recorded_exit(value):
    if type(value) is not int:
        return "未执行或未记录"
    return f"{status(value == 0)}（退出码 {value}）"


def json_block(value):
    return "<pre>" + escaped(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def table(headers, rows):
    return '<div class="table"><table><thead><tr>' + "".join(
        "<th>" + escaped(h) + "</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join("<td>" + cell + "</td>" for cell in row) + "</tr>" for row in rows
    ) + "</tbody></table></div>"


def linked_rows(summary, report):
    """Do not accidentally show a different run's answers under these metrics."""
    raw_rows = {}
    for mode in ("concurrent", "serial"):
        for row in report.get(mode, []):
            key = (mode, row.get("name"))
            if key in raw_rows:
                raise ValueError(f"Duplicate request identity: {key}")
            raw_rows[key] = row
    linked = []
    seen = set()
    for row in summary.get("requests", []):
        key = (row.get("mode"), row.get("name"))
        if key in seen or key not in raw_rows:
            raise ValueError(f"Unmatched summary request: {key}")
        raw = raw_rows[key]
        if row.get("text") != raw.get("text"):
            raise ValueError(f"Summary and raw answer differ: {key}")
        seen.add(key)
        linked.append((row, raw))
    if seen != raw_rows.keys():
        raise ValueError("Summary omits raw requests")
    return linked


def render(directory):
    directory = Path(directory)
    paths = {"摘要": directory / "summary.json",
             "原始并发验收": directory / "concurrency" / "report.json",
             "启动与附加检查": directory / "launch-audit.json"}
    documents = {name: json.loads(path.read_text(encoding="utf-8-sig")) for name, path in paths.items()}
    summary, report, audit = (documents[key] for key in paths)
    if summary.get("schema_version") != 2:
        raise ValueError("Requires summarize-unified-concurrency.py schema_version 2")
    if summary.get("source_sha256") != digest(paths["原始并发验收"]):
        raise ValueError("Summary source SHA-256 does not match concurrency/report.json")
    for summary_key, report_key in (("acceptance_passed", "passed"),
            ("actual_joint_engagement_verified", "actual_joint_engagement_verified"),
            ("full_interval_continuity_verified", "full_interval_continuity_verified"),
            ("greedy_text_parity", "greedy_text_parity")):
        # Missing source fields are displayed as unknown, never promoted from a
        # legacy summary's default false or a renderer's inferred result.
        if report_key in report and summary.get(summary_key) != report[report_key]:
            raise ValueError(f"Summary/source acceptance mismatch: {report_key}")
    rows = linked_rows(summary, report)
    model = report.get("model") or {}
    warmup = model.get("startupWarmup") or {}
    before, after = audit.get("binariesBefore"), audit.get("binariesAfter")
    identity_equal = before == after if isinstance(before, dict) and before \
        and isinstance(after, dict) and after else None
    parity = report.get("greedy_text_parity")
    parity_text = "未知／未记录" if not isinstance(parity, list) or not parity else "；".join(
        f"配对 {i + 1}：{status(value)}" for i, value in enumerate(parity))
    joint_steps = report.get("fused_joint_steps")
    gaps = report.get("fused_continuity_gaps")
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>统一内存并发验收报表</title><style>',
           'body{font:16px/1.6 system-ui,sans-serif;max-width:1240px;margin:32px auto;padding:0 20px;color:#17202a;background:#fafbfc}',
           'h1{font-size:28px}h2{font-size:21px;margin-top:30px}table{border-collapse:collapse;width:100%;background:white}',
           'th,td{border:1px solid #d5dce3;padding:9px;text-align:left;vertical-align:top}th{background:#edf1f5}',
           '.table{overflow-x:auto}.pass{color:#146c43}.fail{color:#ad2424}.unknown{color:#65605b}',
           'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid #d5dce3;padding:14px}',
           'details{margin:10px 0}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere}a{color:#165eab}',
           '</style></head><body><h1>统一内存并发验收报表</h1>',
           f'<p>记录：{escaped(directory.name)}；模型：{escaped(model.get("loaded", "未知"))}；',
           f'后端：{escaped(model.get("loadedBackend", "未知"))}；统一内存状态：{escaped(model.get("unifiedMemory", "未知"))}。</p>',
           '<p>本页只展示已记录证据，不运行模型、不评分回答事实。Greedy 文本一致不等于事实正确；自然 EOS 也不证明回答完整。',
           '达到 max_tokens 的输出只属于有界测试，不能称完整任务通过。</p>',
           ('<p><strong>共享进程运行范围：</strong>' + escaped(audit["extendedScope"]) +
            '；本进程还有扩展请求，它们复用同一次启动，不是独立冷启动。'
            '本页请求表格只展示自身原始报告，未混合其他请求的性能；资源采样可能还包含短请求、扩展请求及后续检查。</p>')
           if audit.get("extendedScope") else '',
           '<h2>核心文本验收与附加检查</h2>',
           table(["检查", "记录结果"], [
               ["核心并发文本验收", status(report.get("passed"))],
               ["本次请求实际联合逻辑图", status(report.get("actual_joint_engagement_verified"))],
               ["首末已记录联合步之间连续", status(report.get("full_interval_continuity_verified"))],
               ["联合步骤／连续性缺口", number(joint_steps, 0) + "／" + (str(len(gaps)) if isinstance(gaps, list) else "未知")],
               ["并发与串行 Greedy 原文一致", parity_text],
               ["启动器及所有附加检查总结果", status(audit.get("passed"))],
               ["主要短请求 checker", recorded_exit(audit.get("checkerExit"))],
               ["扩展请求 checker", recorded_exit(audit.get("extendedCheckerExit"))],
               ["附加 Prefix checker", recorded_exit(audit.get("prefixCheckerExit"))],
               ["附加 HTTP checker", recorded_exit(audit.get("httpCheckerExit"))],
               ["启动器记录的清理完成", status(audit.get("cleanupComplete"))],
               ["部署二进制前后哈希一致", status(identity_equal)],
           ]),
           '<p>“联合逻辑图”可包含多个 CPU/GPU 分段及逐请求算子；它本身不证明全部算子采用一个批量 kernel。',
           '连续性只检查首末已记录联合步之间的位置；缺少该区间外的可合批资格与执行轨迹，不能证明整个共同生命周期均在合批。',
           '本页不从 SSE 帧交替或帧数推断合批与 token 数量。附加检查失败保留独立结果，不改写核心文本验收记录。</p>',
           '<details><summary>启动器错误与连续性原始证据</summary>' + json_block({
               "wrapper_errors": audit.get("errors"), "continuity_gaps": gaps,
               "fused_evidence_errors": (report.get("fused_evidence") or {}).get("errors")}) + '</details>',
           '<h2>请求时延与吞吐</h2>']
    perf_rows = []
    for row, raw in rows:
        mode = {"concurrent": "并发", "serial": "串行对照"}.get(row.get("mode"), "未知")
        perf_rows.append([escaped(f'{mode} · {row.get("name")}'),
            number(row.get("ttft_seconds"), 3),
            number(row.get("uncached_prompt_tokens"), 0) + "／" + number(row.get("reused_prompt_tokens"), 0),
            number(row.get("attributed_prefill_seconds"), 3) + "／" + number(row.get("attributed_prefill_tokens_per_second")),
            number(row.get("recorded_pipeline_output_tokens"), 0),
            number(row.get("pipeline_tokens_per_first_content_to_terminal_second")),
            number(row.get("attributed_decode_tokens_per_second")),
            number(row.get("attributed_decode_seconds"), 3),
            number(row.get("elapsed_seconds"), 3),
            escaped(row.get("finish_reason") or "未知") + "；截断=" + escaped(row.get("truncated") if row.get("truncated") is not None else "未知")])
    out += [table(["请求", "TTFT 秒", "新 PP／复用 token", "PP 秒／token每秒", "管线输出 token",
                   "管线墙钟 decode token/秒", "归因 decode token/秒", "归因 decode 秒", "请求总秒", "结束范围"], perf_rows),
        '<p>TTFT 从派发到首个非空内容，包含排队和准备。PP 只算未缓存 token；多轮 generation 的 token 与 forward 时长分别求和。',
        '管线墙钟 decode＝记录的管线输出 token ÷ 首内容至终止的墙钟时间；归因 decode＝同一 token 计数 ÷ 归因 forward 时间。',
        '联合 batch 的归因时间在请求间分摊，不能当作每请求墙钟速度。管线计数可能包含思考、工具内容和首个生成 token，不是可见正文 token。',
        '计时缺失显示未知，服务端终止事件的备用速率和内容帧数都不用于填补。</p>',
        '<h2>资源与启动</h2>',
        table(["观察", "值", "口径"], [
            ["进程 RSS 峰值", number(summary.get("process_sampled_peak_rss_bytes"), 3, 1024**3) + " GiB", "采样驻留集，非独占物理归属"],
            ["进程 private 峰值", number(summary.get("process_sampled_peak_private_bytes"), 3, 1024**3) + " GiB", "平台记录的 private；不与 RSS 相加"],
            ["整卡 VRAM 使用峰值", number(summary.get("whole_board_sampled_peak_used_mib"), 1) + " MiB", "整张卡，包含其他进程及驱动；不是本进程 owned VRAM"],
            ["启动至 ready 墙钟", number(audit.get("readySeconds"), 3) + " 秒", "包含加载与启动预热，不是单独模型加载时间"],
            ["启动预热", number(warmup.get("elapsedMilliseconds"), 3, 1000) + " 秒", "完成=" + status(warmup.get("completed")) + "；模式=" + escaped(warmup.get("mode", "未知"))],
        ]),
        '<p>资源峰值覆盖整个运行（加载、预热、并发及串行对照），不是仅解码阶段；未知值不替换成零。',
        'ready 时长包含预热，两者不能相加。采样峰值与账本预算不是同一指标，也不证明操作系统硬限或完整物理归属。</p>',
        '<details><summary>采样数量与预热各阶段</summary>' + json_block({
            "samples": summary.get("resource_sample_counts"), "startup_warmup": warmup}) + '</details>',
        '<h2>完整模型原文</h2>']
    for row, raw in rows:
        label = f'{row.get("mode")} · {row.get("name")} · {row.get("finish_reason") or "未知"}'
        out.append('<details class="answer"><summary>' + escaped(label) + '</summary>' +
            '<p>输入请求（包括显式输出上限）：</p>' + json_block(raw.get("request")) +
            '<p>模型原文（HTML 转义，保留完整字符）：</p><pre class="model-output">' +
            (escaped(raw["text"]) if isinstance(raw.get("text"), str) else "未知／未记录") + '</pre>' +
            '<details><summary>逐轮计时、结束和错误</summary>' + json_block({
                "request_id": raw.get("request_id"), "terminal": raw.get("terminal"),
                "timing_complete": row.get("timing_complete"), "timing_errors": row.get("timing_errors"),
                "generation_timings": row.get("generation_timings")}) + '</details></details>')
    source_rows = [[f'<a href="{escaped(path.relative_to(directory).as_posix())}">{escaped(name)}</a>',
                    '<code>' + digest(path) + '</code>'] for name, path in paths.items()]
    source_rows.append(["报表生成器源码 SHA-256", '<code>' + digest(__file__) + '</code>'])
    out += ['<h2>证据与身份</h2>', table(["文件", "SHA-256"], source_rows),
        '<details><summary>实际启动命令、二进制与工具 pins</summary>' + json_block({
            "command": audit.get("command"), "startedUtc": audit.get("startedUtc"),
            "finishedUtc": audit.get("finishedUtc"), "loadedNative": audit.get("loadedNative"),
            "binariesBefore": before, "binariesAfter": after,
            "tool_pins": {k: v for k, v in audit.items() if k.endswith("Sha")},
            "extendedScope": audit.get("extendedScope"),
            "modelIdentityScope": audit.get("modelIdentityScope")}) + '</details>',
        '<p>生成器仅校验摘要绑定的原始 report SHA 与所展示原文、结果；前后部署身份按启动器记录展示，',
        '不会重新读取模型权重或把历史二进制路径上的现有文件冒充当时加载的模块。</p></body></html>']
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    output = args.directory / "report.html"
    content = render(args.directory)
    output.write_text(content, encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "html_sha256": digest(output),
                      "renderer_sha256": digest(__file__)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
