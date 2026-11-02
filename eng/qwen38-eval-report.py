#!/usr/bin/env python3
"""Generate an offline Chinese evaluation report from actual Qwen3.8 JSONL evidence.

  python3 eng/qwen38-eval-report.py 'docs/validation/run/results/*.jsonl' \\
      --context-json docs/validation/run/context.json --output docs/validation/run/report.html

Inputs may be files, quoted globs or directories. No engine is started, no
benchmark is inferred from a historical claim, and no winning engine is invented.
The HTML and its adjacent summary.json retain failed and unavailable scenarios;
performance is withheld unless every expected repeat passes the quality gate.

Optional context JSON fields: title, summary, hardware, narratives, screenshot, figures,
fixes, tests, missingEngines, limitations, sources, plannedScenarios. Narrative
sections accept a string, string list, or objects with title/text/items. A
screenshot object may include image_path to embed an existing local image.
missingEngines accepts strings or objects with engine/status/reason/mode/label.
Unknown context fields are retained in a collapsed metadata section. Generated
reports belong in ignored docs/validation/ or artifacts/.

measurement_groups maps whole raw-input SHA256 strings to explicit group IDs.
Unmapped inputs are UNSPECIFIED, isolated internally by source SHA; host identity
is never inferred from a label, GPU model, or context narrative.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import glob
import hashlib
from html import escape
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = "qwen38-eval-report/1"
spec = importlib.util.spec_from_file_location("qwen38_engine_report", Path(__file__).with_name("qwen38-engine-report.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

STATUS_NAMES = {"PASS": "通过", "FAIL": "未通过", "INCOMPLETE": "不完整", "UNSUPPORTED": "不支持 / 容量拒绝",
                "NOT_RUN": "未执行", "BUILD_FAILED": "构建失败", "ERROR": "执行错误"}
DEFAULT_LIMITATIONS = [
    "测量组仅由原始输入文件 SHA256 的显式映射确定。不同组的延迟和区间不可合并；UNSPECIFIED 未确认主机，不能据此作跨机排名。",
    "只有全部预期重复完成且所有功能检查、配置的基线检查均通过，才展示该场景的性能。失败、缺失、容量拒绝不算验证通过。",
    "表中性能为中位数。95% 区间按整个重复轮次进行 cluster bootstrap；重复很少时，区间仍不足以证明稳定优势。",
    "本报告不自动宣布赢家。差异小于 5% 不能宣称领先；不同检查点、量化、KV 精度、drafter、上下文容量或活动并发数不能隔离比较引擎本身。",
    "TTFT 是客户端观察到首个非空模型 delta 的时间，含调度、网络、视觉编码和解析等开销；工具调用解析可推迟首个可见 delta。",
    "Prefill 吞吐量是未缓存输入 token / 客户端 TTFT，不是纯 GPU kernel 吞吐量。Decode 是流式时间估计，可能受推测解码成批输出影响；SSE 事件数不会当成 token。",
    "功能测试、随机算术任务和可选 GSM8K 小样本只能检查特定回归，不能证明全面模型质量、无训练数据污染或采样分布等价。",
    "质量题组存在请求错误时只报告请求失败数，答案质量未验证；连接失败留下的原始 0/N 不当成模型答错。",
    "长上下文的多位置检索、顺序和相邻关系检查覆盖特定位置失真；通过这些检查不代表全面长上下文推理质量或 RoPE 数值等价。",
    "原始权重 SHA256、来源 revision、启动参数或设备信息未记录时，配置等价性仍未验证。历史截图或帖子数字仅作线索，不混入新测量。",
]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_inputs(inputs):
    paths, warnings = [], []
    for item in inputs:
        matches = sorted(glob.glob(str(item), recursive=True))
        if not matches:
            warnings.append(f"输入未匹配任何文件：{item}")
        for match in matches:
            path = Path(match)
            found = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
            paths.extend(p.resolve() for p in found if p.is_file())
    return list(dict.fromkeys(paths)), warnings


def measurement_groups(context):
    raw = context.get("measurement_groups", {})
    if not isinstance(raw, dict):
        raise ValueError("measurement_groups must map raw input SHA256 strings to nonempty group IDs")
    result = {}
    for source_sha, group in raw.items():
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha):
            raise ValueError("measurement_groups keys must be 64-character raw input SHA256 strings")
        if (not isinstance(group, str) or not group.strip() or group != group.strip()
                or group == "UNSPECIFIED" or group.startswith("UNSPECIFIED:")):
            raise ValueError("measurement_groups values must be nonempty, trimmed group IDs; UNSPECIFIED is reserved")
        source_sha = source_sha.lower()
        if source_sha in result and result[source_sha] != group:
            raise ValueError(f"conflicting measurement_groups assignments for SHA256 {source_sha}")
        result[source_sha] = group
    return result


def measurement_identity(run):
    return {"measurement_group": run.get("measurement_group", "UNSPECIFIED"),
            "source_sha256": run.get("source_sha256")}


def load_runs(paths, groups=None):
    """Last invocation per group/profile; unknown profiles are isolated by source SHA."""
    groups = measurement_groups({"measurement_groups": groups if groups is not None else {}})
    runs, diagnostics, sources, superseded = {}, [], [], 0
    for path in paths:
        current, recognised = None, 0
        try:
            source_bytes = path.read_bytes()
            lines = source_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            diagnostics.append(f"无法读取 {path}: {error}")
            continue
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        group = groups.get(source_sha, "UNSPECIFIED")
        isolation = group if group != "UNSPECIFIED" else "UNSPECIFIED:" + source_sha
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                diagnostics.append(f"无效 JSON：{path}:{line_no}；此行未用于评估")
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "run" and rec.get("engine"):
                current = (str(rec["engine"]), str(rec.get("mode", "plain")), str(rec.get("label", "")), isolation)
                superseded += current in runs
                runs[current] = {"header": rec, "cases": [], "source": str(path), "header_line": line_no,
                                 "measurement_group": group, "source_sha256": source_sha}
                recognised += 1
            elif rec.get("kind") == "case" and current and rec.get("scenario") is not None:
                header_id = runs[current]["header"].get("run_id")
                if rec.get("run_id") and header_id and rec["run_id"] != header_id:
                    diagnostics.append(f"run_id 不匹配：{path}:{line_no}；此 case 未用于评估")
                    continue
                runs[current]["cases"].append(rec)
        sources.append({"path": str(path), "bytes": len(source_bytes), "sha256": source_sha,
                        "measurement_group": group, "recognised_run_headers": recognised})
        if recognised and group == "UNSPECIFIED":
            diagnostics.append(f"测量组未映射：{path}；UNSPECIFIED 按原始输入 SHA256 隔离，不推断主机。")
        if not recognised:
            diagnostics.append(f"未发现 engine-bench run header：{path}；未将其他日志格式解释成引擎成绩")
    if superseded:
        diagnostics.append(f"相同测量组内 engine/mode/label 有 {superseded} 个较早 invocation 被后读取的 header 替换；未合并不同运行，UNSPECIFIED 按输入 SHA256 隔离。")
    return runs, diagnostics, sources


def is_unsupported(case):
    return (case.get("availability") in ("capacity_rejected", "unsupported", "unavailable")
            or str(case.get("verdict", "")).upper() in ("UNSUPPORTED", "UNAVAILABLE", "SKIP", "SKIPPED"))


def request_availability(cases, expected=None):
    """Transport evidence only; an error-free request is not an answer-quality pass."""
    observed, errors = 0, []
    for case in cases:
        requests = case.get("requests", [])
        if not isinstance(requests, list):
            continue
        observed += len(requests)
        for index, request in enumerate(requests):
            if isinstance(request, dict) and request.get("error"):
                errors.append({"repeat": case.get("repeat"), "request_index": index,
                               "error": str(request["error"])})
    return {"status": "request_error" if errors else "no_recorded_request_error" if observed else "not_run",
            "requests_observed": observed, "requests_with_errors": len(errors),
            "expected_requests": expected, "errors": errors}


def functional_answer_counts(header, cases, scenario):
    """Reuse original question checks/counts; never score a transport failure as an answer.

    This is shared by HTML and figures so both require the same complete repeat,
    question identity and original-count evidence. A real strict 0/N remains 0/N.
    """
    integer = lambda value, minimum=0: type(value) is int and value >= minimum
    repeats = header.get("repeats")
    expected = header.get("quality_tasks" if scenario == "quality_tasks" else "gsm8k_limit")
    planned = repeats * expected if integer(repeats, 1) and integer(expected, 1) else None
    availability = request_availability(cases, planned)
    unavailable = sorted({"UNSUPPORTED" if is_unsupported(case) else str(case.get("verdict", "")).upper()
                          for case in cases if is_unsupported(case)
                          or str(case.get("verdict", "")).upper() in ("ERROR", "INCOMPLETE")})
    result = {"correct": None, "total": None, "qualified": False, "counts_complete": False,
              "unique_questions": None, "repeats_observed": 0, "question_counts_by_repeat": [],
              "confidence_interval": None, "independent_question_samples": None,
              "request_availability": availability, "withheld_reason_code": "incomplete_counts",
              "withheld_reason": None}

    def withhold(reason):
        if availability["requests_with_errors"]:
            result.update(withheld_reason_code="request_error", counts_issue=reason,
                          withheld_reason=f"request_error: {availability['requests_with_errors']}/{availability['requests_observed']} recorded requests failed; answer quality unverified")
        elif unavailable:
            result.update(withheld_reason_code="unavailable_case", counts_issue=reason,
                          withheld_reason=f"unavailable_case: {', '.join(unavailable)}; answer quality unverified")
        else:
            result["withheld_reason"] = reason
        return result

    ids = [case.get("repeat") for case in cases]
    if not (integer(repeats, 1) and len(ids) == repeats and all(integer(i) for i in ids)
            and set(ids) == set(range(repeats))):
        return withhold("question counts require every expected repeat exactly once")
    if not integer(expected, 1):
        return withhold("expected questions per repeat were not recorded")
    identities = []
    for case in sorted(cases, key=lambda case: case["repeat"]):
        m, notes = case.get("metrics", {}), case.get("notes", {})
        n, correct = m.get("n_questions"), m.get("n_correct")
        names = ([f"task{i}" for i in range(expected)] if scenario == "quality_tasks" else notes.get("ids"))
        requests = case.get("requests", [])
        if (not integer(n, 1) or n != expected or not integer(correct) or correct > n
                or not isinstance(names, list) or len(names) != expected
                or any(not isinstance(name, str) for name in names) or len(set(names)) != expected
                or not isinstance(requests, list) or len(requests) != expected
                or any(not isinstance(request, dict) for request in requests)):
            return withhold("incomplete or invalid original question counts/identities")
        checks = [check for check in case.get("checks", []) if check.get("name") in names]
        if (len(checks) != expected or len({check["name"] for check in checks}) != expected
                or any(type(check.get("ok")) is not bool for check in checks)
                or sum(check["ok"] for check in checks) != correct):
            return withhold("original question checks and n_correct/n_questions disagree")
        identities.append(names)
        result["question_counts_by_repeat"].append({"repeat": case["repeat"], "correct": correct,
                                                   "total": n, "original_verdict": case.get("verdict")})
    if scenario == "quality_gsm8k" and any(names != identities[0] for names in identities):
        return withhold("GSM8K question IDs differ across repeats")
    result.update(counts_complete=True, repeats_observed=len(cases), questions_per_repeat=expected,
                  original_answer_counts={"correct": sum(item["correct"] for item in result["question_counts_by_repeat"]),
                                          "total": sum(item["total"] for item in result["question_counts_by_repeat"])})
    if availability["requests_with_errors"] or unavailable:
        return withhold(None)
    result.update(qualified=True, withheld_reason=None, withheld_reason_code=None,
                  **result["original_answer_counts"])
    if scenario == "quality_gsm8k":
        result.update(unique_questions=len(identities[0]), question_ids=identities[0],
                      independent_question_samples=False,
                      sample_note=f"{len(identities[0])} unique questions × {len(cases)} repeats; not {result['total']} independent questions")
    else:
        result["sample_note"] = f"{expected} generated tasks per repeat × {len(cases)} repeats; no CI"
    return result


def empty_row(engine, mode, label, scenario, status="NOT_RUN", reason="未执行此场景", expected=None):
    return {"engine": engine, "mode": mode, "label": label, "scenario": scenario,
            "measurement_group": "UNSPECIFIED", "source_sha256": None,
            "status": status, "reason": reason, "repeats": 0, "expected_repeats": expected,
            "pass": 0, "checks_passed": 0, "checks_total": 0, "quality_rate": None,
            "metrics": {}, "errors": []}


def summarise_run(key, run, min_decode_tokens):
    header, cases = run["header"], run["cases"]
    expected = header.get("repeats")
    valid_expected = isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
    grouped = {}
    for case in cases:
        grouped.setdefault(str(case["scenario"]), []).append(case)
    # Invalid repeat ids are evidence errors, not a reason to crash the report.
    summaries = {}
    for scenario, group in grouped.items():
        try:
            summaries.update(metrics.summarise(group, min_decode_tokens, expected if valid_expected else None))
        except (KeyError, TypeError, ValueError) as error:
            summaries[scenario] = {"quality_gate": "INCOMPLETE", "repeats": len(group), "pass": 0,
                "checks_passed": 0, "checks_total": 0, "expected_repeats": expected,
                "evidence_error": f"无法验证 case 结构：{error}"}
    scenarios = list(dict.fromkeys([str(s) for s in header.get("scenarios", [])] + list(grouped)))
    out = []
    for scenario in scenarios:
        if scenario not in summaries:
            out.append(empty_row(*key[:3], scenario, "INCOMPLETE", "run header 计划了此场景，但没有 case", expected))
            continue
        summary, group = summaries[scenario], grouped[scenario]
        status = summary["quality_gate"] if valid_expected else "INCOMPLETE"
        if group and all(is_unsupported(case) for case in group):
            status = "UNSUPPORTED"
        errors = []
        for case in group:
            errors += [str(error) for error in case.get("errors", [])]
            errors += [str(r["error"]) for r in case.get("requests", []) if r.get("error")]
            errors += [f"{c.get('name', '?')}: {c.get('detail', '')}" for c in case.get("checks", []) if c.get("ok") is not True]
        if summary.get("evidence_error"):
            errors.append(summary["evidence_error"])
        errors = list(dict.fromkeys(errors))
        measured = {name: summary.get(name) for name in metrics.METRICS if finite(summary.get(name))} if status == "PASS" else {}
        if status == "PASS":
            # Per-workflow E2E must not be mistaken for one turn's latency.
            if scenario in ("multi_turn", "agentic_loop") and "end_to_end_s" not in measured:
                e2e = [sum(r["total_s"] for r in case["requests"]) for case in group
                       if case.get("requests") and all(finite(r.get("total_s")) for r in case["requests"])]
                if len(e2e) == len(group):
                    measured["end_to_end_s"] = metrics.med(e2e)
            if "end_to_end_s" in measured:
                e2e = [case.get("metrics", {}).get("end_to_end_s", case.get("metrics", {}).get("wall_s")) for case in group]
                if all(finite(value) for value in e2e):
                    low, high = metrics.repeat_median_ci([[value] for value in e2e])
                    if finite(low):
                        measured["end_to_end_s_ci95_low"], measured["end_to_end_s_ci95_high"] = low, high
        checks_total, checks_passed = summary.get("checks_total", 0), summary.get("checks_passed", 0)
        row = {"engine": key[0], "mode": key[1], "label": key[2], "scenario": scenario,
               "status": status, "reason": "全部预期重复和质量检查通过" if status == "PASS" else
                   ("原始请求被引擎拒绝或记录为不支持" if status == "UNSUPPORTED" else
                    "未记录有效的预期重复次数" if not valid_expected else "存在失败、错误或缺失重复，性能不计入比较"),
               "repeats": summary.get("repeats", len(group)), "expected_repeats": expected,
               "pass": summary.get("pass", 0), "checks_passed": checks_passed, "checks_total": checks_total,
               "quality_rate": checks_passed / checks_total if checks_total and status != "UNSUPPORTED" else None,
               "metrics": measured, "errors": errors[:20], "run_id": header.get("run_id"),
               "source": run["source"]}
        answers = [case.get("metrics", {}) for case in group]
        if scenario in ("quality_tasks", "quality_gsm8k"):
            row["functional_answers"] = functional_answer_counts(header, group, scenario)
            row["request_availability"] = row["functional_answers"]["request_availability"]
        elif status != "UNSUPPORTED" and answers and all(isinstance(item.get("n_questions"), int) and isinstance(item.get("n_correct"), int) for item in answers):
            row["functional_answers"] = {"correct": sum(item["n_correct"] for item in answers),
                                         "total": sum(item["n_questions"] for item in answers)}
        out.append(row)
    for row in out:
        row.update(**measurement_identity(run), source=run["source"], run_id=header.get("run_id"))
    return out


def missing_engines(context):
    entries = context.get("missingEngines", context.get("missing_engines", []))
    if isinstance(entries, dict):
        entries = [{"engine": name, **(value if isinstance(value, dict) else {"reason": value})}
                   for name, value in entries.items()]
    out = []
    for entry in entries or []:
        entry = {"engine": entry} if isinstance(entry, str) else dict(entry)
        engine = entry.get("engine", entry.get("name", "unknown"))
        status = str(entry.get("status", "NOT_RUN")).upper().replace("-", "_")
        if status in ("MISSING", "UNAVAILABLE", "NOT_EXECUTED", "PENDING", "BLOCKED"):
            status = "NOT_RUN"
        if status not in STATUS_NAMES:
            status = "NOT_RUN"
        out.append({**entry, "engine": str(engine), "status": status})
    return out


def build_summary(paths, context, min_decode_tokens=32, initial_warnings=()):
    groups = measurement_groups(context)
    runs, diagnostics, sources = load_runs(paths, groups)
    rows = [row for key, run in runs.items() for row in summarise_run(key, run, min_decode_tokens)]
    missing = []
    for entry in missing_engines(context):
        if any(key[0].lower() == entry["engine"].lower() and
               ("mode" not in entry or str(entry["mode"]) == key[1]) and
               ("label" not in entry or str(entry["label"]) == key[2]) for key in runs):
            diagnostics.append(f"missingEngines 中 {entry['engine']} 的状态与实际 run header 重叠；优先使用原始测量，未额外生成缺失行。")
        else:
            missing.append(entry)
    scenarios = list(dict.fromkeys([row["scenario"] for row in rows] + list(context.get("plannedScenarios", []))))
    for key, run in runs.items():
        present = {row["scenario"] for row in rows if (row["engine"], row["mode"], row["label"]) == key[:3]
                   and row["measurement_group"] == run["measurement_group"] and row["source_sha256"] == run["source_sha256"]}
        for scenario in scenarios:
            if scenario not in present:
                row = empty_row(*key[:3], scenario)
                row.update(**measurement_identity(run), source=run["source"], run_id=run["header"].get("run_id"))
                rows.append(row)
    for entry in missing:
        key = (entry["engine"], str(entry.get("mode", "—")), str(entry.get("label", "")))
        for scenario in scenarios:
            if not any((row["engine"], row["mode"], row["label"], row["scenario"]) == (*key, scenario) for row in rows):
                rows.append(empty_row(*key, scenario, entry["status"], str(entry.get("reason", "无实际运行数据"))))
    return safe_json({"schema": REPORT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "context": context, "sources": sources, "diagnostics": list(initial_warnings) + diagnostics,
        "measurement_groups": groups,
        "selection_policy": "按输入文件和行顺序读取；相同测量组内 engine/mode/label 仅保留最后的 run header 及其 cases。UNSPECIFIED 按原始输入 SHA256 隔离，不合并 invocation 或跨组重复。",
        "method": {"min_decode_tokens": min_decode_tokens, "performance_gate": "all_expected_repeats_and_all_quality_checks",
                   "confidence_intervals": "95% cluster bootstrap by whole repeats, 2000 resamples; no winner inference",
                   "measurement_groups": "explicit raw input SHA256 mapping; unknown inputs isolated by SHA; latency/CI never pooled across runs or groups",
                   "winner_policy": "none; <5% differences do not establish a lead"},
        "runs": [{"engine": key[0], "mode": key[1], "label": key[2], "source": run["source"],
                  **measurement_identity(run), "header_line": run["header_line"], "header": run["header"]} for key, run in runs.items()],
        "rows": rows, "missing_engines": missing, "status_counts": dict(Counter(row["status"] for row in rows)),
        "limitations": DEFAULT_LIMITATIONS})


def esc(value):
    return escape(str(value), quote=True)


def text_block(value):
    """Context is data: escape prose and never interpret it as HTML or JavaScript."""
    if value is None:
        return ""
    if isinstance(value, str):
        return "".join(f"<p>{esc(p).replace(chr(10), '<br>')}</p>" for p in value.split("\n\n") if p)
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{text_block(item)}</li>" for item in value) + "</ul>"
    if isinstance(value, dict):
        title = value.get("title", value.get("name"))
        body = "".join(text_block(value[key]) for key in ("text", "body", "items", "paragraphs") if key in value)
        if not body:
            body = "<dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{text_block(v)}</dd>" for k, v in value.items()
                                      if k not in ("title", "name", "image_path")) + "</dl>"
        return (f"<h3>{esc(title)}</h3>" if title else "") + body
    return f"<p>{esc(value)}</p>"


def metric_cell(row, name, decimals=3):
    value = row["metrics"].get(name)
    if not finite(value):
        return '<td class="number muted">—</td>'
    interval = ""
    low, high = row["metrics"].get(name + "_ci95_low"), row["metrics"].get(name + "_ci95_high")
    if finite(low) and finite(high):
        interval = f'<small class="ci">[{low:.{decimals}f}, {high:.{decimals}f}]</small>'
    return f'<td class="number">{value:,.{decimals}f}{interval}</td>'


def data_table(rows, table_id):
    if not rows:
        return '<p class="empty">未执行：没有此类场景的可核验原始测量。</p>'
    parts = ['<p class="method">仅同测量组内比较延迟和区间；UNSPECIFIED 表示主机未标注，按输入 SHA 隔离。</p>',
             f'<div class="table-wrap"><table id="{esc(table_id)}"><thead><tr>',
             '<th>引擎 / 模式</th><th>测量组</th><th>场景</th><th>状态</th><th>质量检查</th><th>重复通过 / 计划</th>',
             '<th>TTFT · s</th><th>总耗时 · s</th><th>Decode · tok/s</th><th>Prefill · tok/s</th><th>输入 token</th></tr></thead><tbody>']
    for row in rows:
        status, m = row["status"], row["metrics"]
        search = " ".join(str(row.get(key, "")) for key in ("engine", "mode", "label", "scenario", "status", "measurement_group"))
        quality = (f"{row['quality_rate'] * 100:.1f}% <small>{row['checks_passed']}/{row['checks_total']} 检查</small>"
                   if finite(row.get("quality_rate")) else "—")
        if row.get("functional_answers"):
            answer = row["functional_answers"]
            availability = answer.get("request_availability", {})
            if availability.get("requests_with_errors"):
                quality = (f"{availability['requests_with_errors']}/{availability['requests_observed']} 请求失败"
                           "<small>答案质量未验证</small>")
            elif answer.get("qualified", True):
                quality += f"<small>功能题 {answer['correct']}/{answer['total']}</small>"
            elif answer.get("withheld_reason_code") == "unavailable_case":
                quality = "—<small>答案质量未验证</small>"
            else:
                quality = "—<small>题目计数或重复不完整，答案质量未验证</small>"
        planned = row.get("expected_repeats")
        repeats = f"{row['pass']}/{planned}" if isinstance(planned, int) and planned > 0 else "未执行" if not row["repeats"] else f"{row['pass']}/{row['repeats']}（计划缺失）"
        if row["repeats"] and planned and row["repeats"] != planned:
            repeats += f"<small>实际记录 {row['repeats']} 轮</small>"
        latency = "end_to_end_s" if "end_to_end_s" in m else "total_s"
        reason = row["reason"] + ("\n" + "\n".join(row.get("errors", [])[:3]) if row.get("errors") else "")
        parts += [f'<tr data-search="{esc(search.lower())}" data-status="{esc(status)}">',
                  f'<td><strong>{esc(row["engine"])}</strong><small>{esc(row["mode"])}</small>'
                  + (f'<small class="label">{esc(row["label"])}</small>' if row.get("label") else "") + '</td>',
                  f'<td><code>{esc(row.get("measurement_group", "UNSPECIFIED"))}</code>'
                  + (f'<small>input {esc(row["source_sha256"][:12])}</small>' if row.get("source_sha256") else "") + '</td>',
                  f'<td><code>{esc(row["scenario"])}</code></td>',
                  f'<td><span class="badge {esc(status.lower())}" title="{esc(reason)}">{esc(STATUS_NAMES.get(status, status))}</span></td>',
                  f'<td class="number">{quality}</td><td class="number">{repeats}</td>',
                  metric_cell(row, "ttft_s"), metric_cell(row, latency), metric_cell(row, "decode_tok_s", 1),
                  metric_cell(row, "prefill_tok_s", 1), metric_cell(row, "prompt_tokens", 0), '</tr>']
    parts.append('</tbody></table></div>')
    return "".join(parts)


def render_html(summary):
    context, rows = summary["context"], summary["rows"]
    title = str(context.get("title", "TensorSharp · Qwen3.8 27B 评估报告"))
    sections = []
    for key, heading in (("summary", "实测结论"), ("hardware", "评估环境"), ("narratives", "分析与证据"),
                         ("screenshot", "截图线索与历史比较"), ("fixes", "修复与优化"), ("tests", "验证覆盖")):
        if key in context:
            body = text_block(context[key])
            if key == "screenshot" and isinstance(context[key], dict) and context[key].get("image_path"):
                path = Path(context[key]["image_path"])
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower())
                if path.is_file() and mime and path.stat().st_size <= 10 * 1024 * 1024:
                    data = base64.b64encode(path.read_bytes()).decode("ascii")
                    body += f'<details><summary>查看原始截图</summary><img class="screenshot" alt="用户提供的历史性能截图；不计入新测量" src="data:{mime};base64,{data}"></details>'
            sections.append(f'<section><h2>{heading}</h2>{body}</section>')
    figures = []
    for item in context.get("figures", []):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        if path.is_file() and path.suffix.lower() == ".png" and path.stat().st_size <= 10 * 1024 * 1024:
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            figures.append(f'<figure><h3>{esc(item.get("title", path.stem))}</h3>'
                           f'<img class="screenshot" alt="{esc(item.get("title", path.stem))}" src="data:image/png;base64,{payload}">'
                           f'<figcaption class="method">{esc(item.get("caption", ""))}</figcaption></figure>')
    if figures:
        sections.append('<section><h2>可核验图表</h2><p class="method">速度点沿用原始质量门槛；失败和缺失不是零速度。区间不覆盖配置差异、运行顺序或并行CPU活动。</p>' + ''.join(figures) + '</section>')
    no_data = '<div class="notice caution"><strong>本报告没有可核验的引擎运行数据。</strong> 所有性能仍为未执行；不能判断修复收益、质量保持或引擎排名。</div>' if not summary["runs"] else ""
    long_rows = [row for row in rows if re.search(r"(?:128|250)k", row["scenario"], re.I)]
    short_rows = [row for row in rows if row not in long_rows]
    count = summary["status_counts"]
    cards = [("记录的运行配置", len(summary["runs"])), ("全通过场景组", count.get("PASS", 0)),
             ("未通过 / 不完整场景组", sum(count.get(key, 0) for key in ("FAIL", "INCOMPLETE", "ERROR"))),
             ("128K / 250K 全通过场景组", sum(row["status"] == "PASS" for row in long_rows))]
    card_html = "".join(f'<div class="card"><span>{esc(name)}</span><strong>{value if summary["runs"] else "—"}</strong></div>' for name, value in cards)
    notes = ["128K 指 131,072 token；250K 指 256,000 token。实际输入量以服务端 usage 为准。",
             "长上下文任务是否包含随机顺序和相邻关系检查，以折叠配置中的 long_order_quality 和原始 checks 为准。",
             "总耗时：普通场景为单请求中位数；多轮/agent 为整个工作流，多并发为整波请求。下方小字为 95% bootstrap 区间。"]
    missing = "".join(f'<li><strong>{esc(entry["engine"])}</strong> <span class="badge {esc(entry["status"].lower())}">{esc(STATUS_NAMES[entry["status"]])}</span> {esc(entry.get("reason", "无实际运行数据"))}</li>' for entry in summary["missing_engines"])
    failures = "".join(f'<details><summary>{esc(row["engine"])} · {esc(row["mode"])} · {esc(row["scenario"])} · {esc(STATUS_NAMES[row["status"]])}</summary>{text_block(row["reason"])}<pre>{esc(json.dumps(row.get("errors", []), ensure_ascii=False, indent=2))}</pre></details>' for row in rows if row["status"] not in ("PASS", "NOT_RUN"))
    provenance = "".join(f'<details><summary>{esc(run["engine"])} · {esc(run["mode"])} · {esc(run["label"] or "无额外标签")} · run {esc(run["header"].get("run_id", "未记录"))}</summary><pre>{esc(json.dumps(run, ensure_ascii=False, indent=2))}</pre></details>' for run in summary["runs"])
    sources = []
    for item in context.get("sources", []):
        if isinstance(item, dict) and urlparse(str(item.get("url", ""))).scheme in ("http", "https"):
            sources.append(f'<li><a href="{esc(item["url"])}" rel="noreferrer">{esc(item.get("title", item["url"]))}</a>{text_block(item.get("note"))}</li>')
        else:
            sources.append(f'<li>{text_block(item)}</li>')
    diagnostics = text_block(summary["diagnostics"]) if summary["diagnostics"] else '<p>输入解析没有报告异常。</p>'
    limitations = text_block(summary["limitations"]) + text_block(context.get("limitations"))
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title><style>
:root{{--bg:#f2f5fa;--paper:#fff;--ink:#172438;--muted:#64748b;--line:#dce4ef;--accent:#2357c6}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}header{{background:#13243d;color:white;padding:48px max(28px,calc((100vw - 1480px)/2));border-bottom:5px solid #60a5fa}}header .eyebrow{{color:#9ec7ff;font-size:12px;letter-spacing:.16em}}h1{{font-size:clamp(24px,3vw,36px);line-height:1.3;margin:12px 0 16px}}header p{{color:#cbd8ec;max-width:960px}}nav{{display:flex;gap:20px;flex-wrap:wrap}}nav a{{color:#bfd8ff;text-decoration:none;font-size:13px}}main{{max-width:1536px;margin:auto;padding:28px}}section{{background:var(--paper);padding:26px 30px;margin-bottom:22px;border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 14px #1b335008}}h2{{font-size:20px;margin:0 0 16px;letter-spacing:.02em}}h3{{font-size:16px;margin:20px 0 8px}}p{{margin:8px 0 12px}}ul{{padding-left:22px}}li{{margin:8px 0}}li p{{margin:3px 0}}a{{color:var(--accent)}}.notice{{padding:16px 20px;background:#eaf1ff;border:1px solid #cfdefa;border-radius:9px;margin:0 0 22px}}.caution{{background:#fff5dd;border-color:#f3d592}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}}.card{{background:white;padding:18px 24px;border:1px solid var(--line);border-radius:10px}}.card span{{display:block;color:var(--muted);font-size:13px}}.card strong{{font-size:30px;font-weight:650}}.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th{{background:#edf2fa;text-align:left;font-size:12px;color:#46566e;padding:13px 12px}}td{{padding:12px;border-top:1px solid var(--line);vertical-align:top}}tbody tr:hover{{background:#f7faff}}td strong{{font-weight:650}}td small{{display:block;color:var(--muted);font-size:11px;line-height:1.5}}.label{{max-width:180px;white-space:normal;overflow-wrap:anywhere}}.number{{text-align:right;font-variant-numeric:tabular-nums}}.ci{{white-space:nowrap;color:#6a7f9c}}.muted,.empty{{color:var(--muted)}}.empty{{padding:18px;background:#f7f9fd;border-radius:8px}}.badge{{display:inline-block;border-radius:5px;padding:3px 7px;font-size:11px;font-weight:600;background:#edf0f4;color:#5c697c}}.pass{{background:#e3f4ea;color:#18633d}}.fail,.error,.build_failed{{background:#fee7e5;color:#a82e2e}}.incomplete{{background:#fff1cf;color:#86600b}}.unsupported{{background:#eee8fc;color:#644295}}.filters{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}input,select{{border:1px solid #cbd7e6;border-radius:6px;padding:9px 12px;background:white;color:var(--ink);font:inherit}}input{{min-width:260px}}.method{{font-size:13px;color:var(--muted)}}code,pre{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}code{{font-size:12px}}details{{border:1px solid var(--line);border-radius:7px;padding:12px 16px;margin:10px 0}}summary{{cursor:pointer;font-weight:550}}pre{{font-size:12px;background:#f5f7fb;border-radius:6px;padding:15px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}}dt{{font-weight:600;margin-top:8px}}dd{{margin-left:0;color:#42536c}}.screenshot{{max-width:100%;height:auto;border-radius:6px;margin-top:15px}}footer{{text-align:center;color:var(--muted);font-size:12px;padding:18px 0 38px}}tr[hidden]{{display:none}}@media(max-width:800px){{main{{padding:14px}}section{{padding:20px 16px}}.cards{{grid-template-columns:repeat(2,1fr);gap:10px}}header{{padding:32px 20px}}}}@media print{{header{{background:white;color:#172438;padding:20px;border-color:#2357c6}}header p,nav a{{color:#46566e}}main{{padding:0}}section,.card{{box-shadow:none;break-inside:avoid}}.filters{{display:none}}details{{break-inside:avoid}}}}
</style></head><body><header><div class="eyebrow">TENSORSHARP · QWEN3.8 27B · 实测与质量</div><h1>{esc(title)}</h1><p>所有成绩来自输入的原始运行记录。质量失败的性能隐藏，缺失场景保留状态，配置差异与验证局限逐项列出。</p><nav><a href="#performance">性能与质量</a><a href="#long-context">128K / 250K</a><a href="#coverage">覆盖与限制</a><a href="#provenance">来源与复现</a></nav></header><main>{no_data}<div class="cards">{card_html}</div>{''.join(sections)}
<section id="performance"><h2>性能与质量 · 按场景</h2><div class="notice">不同检查点和精度属于不同部署配置，表格不构成纯引擎排名。— 表示未执行、没有可靠计数，或未通过质量门槛。</div><div class="filters"><input id="search" type="search" aria-label="筛选引擎或场景" placeholder="筛选引擎、模式、标签或场景"><select id="status" aria-label="筛选状态"><option value="ALL">全部状态</option>{''.join(f'<option value="{key}">{esc(value)}</option>' for key,value in STATUS_NAMES.items())}</select></div>{data_table(short_rows, 'main-table')}<p class="method">{esc(notes[2])}</p></section>
<section id="long-context"><h2>超长 Prefill 与位置质量 · 128K / 250K</h2><p class="method">{esc(notes[0])} {esc(notes[1])}</p>{data_table(long_rows, 'long-table')}</section>
<section id="coverage"><h2>覆盖、失败与评估局限</h2>{('<h3>未完成的引擎</h3><ul>'+missing+'</ul>') if missing else ''}{failures}<h3>解释范围</h3>{limitations}</section>
<section id="provenance"><h2>来源、版本与复现配置</h2><p class="method">{esc(summary['selection_policy'])} 原始哈希、启动命令、模型和 KV 精度、drafter、采样与质量基线信息在各运行 header 中完整保留。</p>{provenance or '<p class="empty">没有引擎运行 header，不能确认模型与配置。</p>'}<details><summary>原始文件清单与 SHA256</summary><pre>{esc(json.dumps(summary['sources'], ensure_ascii=False, indent=2))}</pre></details><details><summary>输入解析诊断</summary>{diagnostics}</details><details><summary>补充上下文原文</summary><pre>{esc(json.dumps(context, ensure_ascii=False, indent=2))}</pre></details>{('<h3>外部参考</h3><ul>'+''.join(sources)+'</ul>') if sources else ''}<p class="method">完整机器可读结果：<a href="summary.json">summary.json</a>。生成器 {REPORT_VERSION}；{esc(summary['generated_at'])}。</p></section></main><footer>离线 HTML · 无 CDN、无外部脚本 · 原始证据决定结论</footer><script>
const q=document.getElementById('search'),s=document.getElementById('status');function filterRows(){{const text=q.value.toLowerCase();document.querySelectorAll('tr[data-search]').forEach(r=>r.hidden=!r.dataset.search.includes(text)||(s.value!=='ALL'&&r.dataset.status!==s.value));}}q.addEventListener('input',filterRows);s.addEventListener('change',filterRows);
</script></body></html>'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="raw JSONL files, globs or directories")
    ap.add_argument("--output", type=Path, required=True, help="generated offline HTML report")
    ap.add_argument("--context-json", type=Path, help="Chinese analysis, tests, missing engines and limitations")
    ap.add_argument("--min-decode-tokens", type=int, default=32)
    args = ap.parse_args(argv)
    if args.min_decode_tokens < 2:
        ap.error("--min-decode-tokens must be at least 2")
    context = json.loads(args.context_json.read_text(encoding="utf-8")) if args.context_json else {}
    if not isinstance(context, dict):
        ap.error("--context-json must contain a JSON object")
    paths, warnings = collect_inputs(args.inputs)
    try:
        summary = build_summary(paths, context, args.min_decode_tokens, warnings)
    except ValueError as error:
        ap.error(str(error))
    html = render_html(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(args.output), "summary": str(summary_path), "runs": len(summary["runs"]),
                      "statuses": summary["status_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
