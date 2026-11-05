#!/usr/bin/env python3
"""Offline aggregation/PNG+SVG for frozen qwen38-cua-control/1 JSONL.

No server is contacted and no model reply is rescored. Original failed checks
stay failed. Each run/case must be independently complete; separate partial
runs are never stitched together. Matplotlib is imported only by render().

  python3 eng/validation/qwen38-cua-report.py 'artifacts/cua/*.jsonl' \
      --output-dir docs/validation/cua --data-only

Optional --plan is a JSON list of expected engine/mode/label/cache_profile/lanes/
scenario slots. It can make unrun configurations explicit without inventing data.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import fnmatch
from functools import lru_cache
import glob
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()
CONTROL = HERE.with_name("qwen38-cua-control.py")
BENCH = ROOT / "eng/qwen38-engine-bench.py"
VERSION = "qwen38-cua-report/1"
CONTROL_VERSION = "qwen38-cua-control/1"
DEFAULT = ("append", "truncate", "edit")
MEDIA = "edit_media"
GROUP_FIELDS = ("engine", "mode", "label", "cache_profile", "lanes", "scenario")
QUALITY_CHECKS = {"http_success", "single_choice", "stream_complete", "no_hidden_thinking", "one_correct_tool",
                  "typed_exact_arguments", "no_freeform_answer"}
MEASUREMENT_CHECKS = {"actual_server_usage", "finite_client_timings", "ttft_within_request"}
COMMON_FIELDS = ("messages", "tools", "tool_choice", "max_tokens", "temperature", "top_p", "presence_penalty",
                 "frequency_penalty", "seed", "stream", "stream_options")
METRICS = ("ttft_median_s", "request_e2e_median_s", "wave_wall_total_s", "completion_tokens_total")
LIMITATIONS = [
    "公开合成 prescribed-history CUA-like replay，不是原私有工作流或闭环代理成功率。",
    "原始质量/测量FAIL或UNVERIFIED不改判。默认30步与独立media探针6..10步必须逐run、逐case完整；部分run不能相互补齐。",
    "30步是同一会话中的相关顺序观测，不是30个独立样本。无置信区间、显著性或胜负排序；多次run只给整case统计量的描述性中位数。",
    "TTFT是首个非空semantic delta，含工具名称/参数；工具解析器可能缓冲整段输出。工具SSE不是目标token流，不计算decode tok/s。",
    "工具预算固定192不保证输出长度相同；实际server usage、每请求输出量与缓存计数均保留，缺失不以SSE事件数或字符数代替。",
    "prescribed回复可能不满足TensorSharp emitted-prose lineage证明；default-lineage单独诊断，external global仅与显式trusted-single-tenant-shared分组。",
    "common SHA是适配前portable fixture；canonical input只去model，仍保留adapter/tool IDs。完整body/wire/response/source SHA分别记录，不能互相冒充byte identity。",
    "同engine/label的不同fixture、context、协议、实际部署fingerprint分层；同一分层的失败attempt不会因另一attempt通过而被丢弃。",
    "run_end.complete仅表示selected scenarios结束。单独一个场景通过不代表默认3场景×1/2lane全矩阵完成。",
    "本工具只核验已记录数据/原门槛的一致性，不复跑模型、不扩展原质量覆盖；原始全文和SSE在被SHA引用的raw JSONL中。",
]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def sha_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def all_checks(value, required):
    return isinstance(value, dict) and required <= set(value) and all(v is True for v in value.values())


def mapping(value):
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=4)
def frozen_client_module(expected_sha):
    if not sha_value(expected_sha) or file_sha(BENCH) != expected_sha:
        raise ValueError("frozen body builder source SHA differs")
    spec = importlib.util.spec_from_file_location("cua_report_frozen_body_builder", BENCH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    if bench.HARNESS_VERSION != "qwen38-engine-bench/3":
        raise ValueError("unsupported frozen body builder")
    return bench


def original_wire_body(record, header):
    """Rebuild frozen Client.body order lost by canonical JSONL serialization.

    The prepared fixture itself was canonical JSON: its common/nested objects
    therefore entered Client.body in sorted order. Adapters and base fields
    were then inserted by the frozen builder, not by JSONL's final key sort.
    Never contact a server or accept a pair of reported hashes as byte proof.
    """
    body = record["request_body"]
    if (not isinstance(body, dict) or not isinstance(header.get("model"), str) or not header["model"]
            or header.get("engine") not in ("tensorsharp", "velox", "ninfer")):
        raise ValueError("engine/model/body missing for frozen wire reconstruction")
    bench = frozen_client_module(mapping(header.get("sources")).get("frozen_bench"))
    common = json.loads(canonical({k: body[k] for k in COMMON_FIELDS}))
    args = types.SimpleNamespace(engine=header["engine"], model=header["model"], url="http://offline.invalid",
        extra_body=None, metrics_url="", tokenizer=None, temperature=0, top_p=1, sampling_seed=1234)
    client = bench.Client(args) # Nonempty model and tokenizer=None: no discovery or file/network I/O.
    rebuilt = client.body(common["messages"], common["max_tokens"],
        {k: v for k, v in common.items() if k not in ("messages", "max_tokens")})
    if rebuilt != body or canonical(rebuilt) != canonical(body):
        raise ValueError("complete body differs from the frozen engine adapter")
    return rebuilt


def cache_group(profile):
    if not isinstance(profile, str): return "UNVERIFIED"
    return {"default-lineage": "default-lineage-diagnostic", "trusted-single-tenant-shared": "trusted-single-tenant-prefix",
            "engine-native-global": "trusted-single-tenant-prefix"}.get(profile, "UNVERIFIED")


def deployment(header):
    manifest = mapping(header.get("launch_manifest"))
    keys = ("engine", "mode", "context_tokens", "lanes", "argv", "working_directory", "environment", "models",
            "engine_source", "tensorsharp_source", "ggml_source", "configuration_files", "gpu", "host")
    data = {k: manifest.get(k) for k in keys}
    binaries = manifest.get("binaries") if isinstance(manifest.get("binaries"), list) else []
    data["binary_sha256"] = {str(b.get("path", "")): b.get("sha256") for b in binaries if isinstance(b, dict)}
    return digest(data)


def parse(path):
    """Keep every run boundary; malformed source never qualifies an older run."""
    path = Path(path)
    source = {"path": str(path.resolve()), "status": "OK", "diagnostics": [], "aliases": []}
    try:
        raw = path.read_bytes()
    except OSError as error:
        source.update(status="NOT_RUN" if not path.exists() else "ERROR", error=str(error), sha256=None)
        return source, []
    source.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    segments, current = [], None
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        source.update(status="ERROR", error=f"invalid UTF8: {error}")
        return source, []
    for number, line in enumerate(lines, 1):
        if not line.strip(): continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict): raise ValueError("record must be an object")
        except ValueError as error:
            source["status"] = "ERROR"
            source["diagnostics"].append({"line": number, "error": str(error), "excerpt": line[:200]})
            continue
        if record.get("kind") == "run":
            current = {"header": record, "records": [], "line": number, "index": len(segments), "errors": []}
            segments.append(current)
        elif current is None:
            source["status"] = "ERROR"
            source["diagnostics"].append({"line": number, "error": "record before run header"})
        else:
            current["records"].append((number, record))
    if not segments:
        source.update(status="ERROR", error="no CUA run header")
    if source["status"] == "ERROR":
        for segment in segments: segment["errors"].append("source malformed; no segment in this file is eligible")
    return source, segments


def protocol(header):
    name, steps = header.get("protocol"), header.get("steps")
    if name == "default_three_scenarios" and steps == 30 and header.get("standard_30step_protocol") is True:
        return DEFAULT, 30, []
    if (name == "opt_in_6_to_10_step_old_media_probe" and integer(steps, 6) and steps <= 10
            and header.get("standard_30step_protocol") is False):
        return (MEDIA,), steps, []
    return (), steps, ["unknown/nonstandard protocol: default must be30, media must be6..10"]


def header_errors(header):
    scenarios, steps, errors = protocol(header)
    if header.get("schema") != CONTROL_VERSION: errors.append("unknown frozen control schema")
    if header.get("lanes") not in (1, 2) or type(header.get("lanes")) is not int: errors.append("lanes must be integer1/2")
    selected = header.get("selected_scenarios")
    if (not isinstance(selected, list) or not selected or not all(isinstance(s, str) for s in selected) or len(selected) != len(set(selected))
            or not set(selected) <= set(scenarios)):
        errors.append("invalid selected scenarios")
    for key in ("engine", "mode", "label"):
        if not isinstance(header.get(key), str) or not header[key]: errors.append(f"missing {key}")
    if not isinstance(header.get("model"), str) or not header["model"]: errors.append("model routing ID required for exact wire reconstruction")
    if header.get("engine") not in ("tensorsharp", "velox", "ninfer"): errors.append("unknown engine for frozen CUA version")
    profile = header.get("cache_profile_operator_declaration")
    if cache_group(profile) == "UNVERIFIED" or header.get("cache_comparison_group") != cache_group(profile):
        errors.append("unknown/inconsistent cache fairness declaration")
    if header.get("context_capacity_operator_declaration") != 65536: errors.append("CTX65536 declaration missing")
    manifest = header.get("launch_manifest")
    if not isinstance(manifest, dict):
        errors.append("launch manifest missing"); manifest = {}
    client_lanes = header.get("lanes") if type(header.get("lanes")) is int else 99
    if (manifest.get("engine") != header.get("engine") or manifest.get("context_tokens") != 65536
            or not integer(manifest.get("lanes"), 1) or manifest.get("lanes", 0) < client_lanes):
        errors.append("launch engine/context/lanes inconsistent")
    if header.get("engine") == "tensorsharp":
        env = mapping(manifest.get("environment"))
        expected = {"default-lineage": "lineage", "trusted-single-tenant-shared": "shared"}.get(profile)
        actual = env.get("TS_PREFIX_CACHE_ISOLATION") or "lineage"
        actual = actual.strip().lower() if isinstance(actual, str) else actual
        if ("TS_PREFIX_CACHE_ISOLATION" not in env or expected is None or actual != expected):
            errors.append("TS cache declaration differs from actual recorded environment")
    elif profile != "engine-native-global": errors.append("external engine must declare native-global")
    settings = {"temperature": 0, "top_p": 1, "seed": 1234, "thinking": False, "tool_choice": "auto", "max_tokens": 192,
                "warmup": "NONE", "prior_history": "prescribed"}
    if header.get("settings") != settings: errors.append("fixed sampler/budget/history contract differs")
    sources, prepared = header.get("sources"), mapping(header.get("fixture_manifest"))
    if (not isinstance(sources, dict) or not all(sha_value(sources.get(k)) for k in ("tool", "frozen_bench", "media_generator"))
            or sources != prepared.get("sources") or sources.get("tool") != file_sha(CONTROL)
            or sources.get("frozen_bench") != file_sha(BENCH)):
        errors.append("frozen source/fixture source identities inconsistent or unsupported")
    if not sha_value(prepared.get("fixture_sha256")) or not sha_value(header.get("launch_manifest_sha256")):
        errors.append("fixture/launch SHA missing")
    return errors


def request_observation(record, line, header):
    """Retain raw counts/hashes and original verdict; never regrade semantics."""
    q, response = mapping(record.get("quality")), mapping(record.get("response"))
    raw, wire, body = mapping(record.get("raw_observation")), mapping(record.get("raw_transport")), record.get("request_body")
    usage = response.get("raw_usage")
    errors = []
    original_quality = (q.get("verdict") == "PASS" and all_checks(q.get("checks"), QUALITY_CHECKS)
                        and q.get("schema_errors") == [])
    original_measurement = (q.get("measurement_verdict") == "PASS" and all_checks(q.get("measurement_checks"), MEASUREMENT_CHECKS))
    if not (original_quality and original_measurement and q.get("latency_qualified") is True):
        errors.append("original quality/measurement gate not fully PASS")
    if (not isinstance(usage, dict) or not integer(usage.get("prompt_tokens"), 1)
            or not integer(usage.get("completion_tokens"), 1) or usage.get("completion_tokens", 193) > 192
            or usage != raw.get("usage") or usage.get("completion_tokens") != mapping(record.get("output")).get("completion_tokens")
            or response.get("completion_tokens_source") != "usage"
            or response.get("completion_tokens") != usage.get("completion_tokens")
            or response.get("prompt_tokens") != usage.get("prompt_tokens")):
        errors.append("actual server usage missing/invalid/inconsistent; no token fallback")
    raw_usage = raw.get("usage")
    cached = (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens") if isinstance(raw_usage, dict) and isinstance(raw_usage.get("prompt_tokens_details") or {}, dict) else None
    if isinstance(usage, dict) and usage.get("prompt_tokens_details") is not None and not isinstance(usage["prompt_tokens_details"], dict):
        errors.append("cached token details must be object when present")
    if cached is not None and (not integer(cached) or not isinstance(raw_usage, dict)
            or not integer(raw_usage.get("prompt_tokens"), 1) or cached > raw_usage["prompt_tokens"]):
        errors.append("invalid recorded cached token usage")
    if (not positive(response.get("ttft_s")) or not positive(response.get("total_s"))
            or response["ttft_s"] > response["total_s"]): errors.append("invalid client timing")
    if (response.get("error") or raw.get("error") or response.get("http_status") != 200
            or response.get("finish_reason") not in ("stop", "tool_calls", "function_call")
            or wire.get("done") is not True or wire.get("closed_normally") is not True):
        errors.append("request/transport error or incomplete/length stream")
    identity_checks = {}
    if isinstance(body, dict):
        identity_checks = {"full_body": record.get("full_body_sha256") == digest(body),
            "canonical_input": record.get("canonical_input_sha256") == digest({k: v for k, v in body.items() if k != "model"}),
            "portable_common": set(COMMON_FIELDS) <= set(body) and record.get("common_sha256") == digest({k: body[k] for k in COMMON_FIELDS if k in body}),
            "frozen_body_reconstruction": False, "wire_request": False}
        try:
            rebuilt = original_wire_body(record, header)
            identity_checks["frozen_body_reconstruction"] = True
            identity_checks["wire_request"] = (record.get("wire_request_body_sha256") ==
                hashlib.sha256(json.dumps(rebuilt).encode("utf-8")).hexdigest() == wire.get("request_body_sha256"))
        except (KeyError, TypeError, ValueError, OSError, ImportError) as error:
            errors.append("exact frozen wire reconstruction unavailable: " + str(error))
    if not identity_checks or not all(identity_checks.values()): errors.append("request body/input hash inconsistency")
    try:
        payload = base64.b64decode(wire["body_base64"], validate=True)
        if (hashlib.sha256(payload).hexdigest() != wire.get("body_sha256") or len(payload) != wire.get("bytes")):
            errors.append("raw response transport SHA/size differs")
    except (KeyError, TypeError, ValueError): errors.append("raw response transport bytes unavailable/invalid")
    if not sha_value(response.get("response_sha256")): errors.append("normalized response SHA missing")
    return {"source_line": line, "lane": record.get("lane"), "step": record.get("step"),
        "scope": record.get("scope"), "history": record.get("history"), "expected": record.get("expected"),
        "original_quality": q, "original_quality_pass": original_quality, "original_measurement_pass": original_measurement,
        "eligible": not errors, "aggregation_errors": errors, "raw_usage": raw_usage, "normalized_response_usage": usage,
        "server_cached_tokens": cached, "reported_cached_tokens": response.get("cached_tokens"),
        "server_timings": response.get("server_timings"), "raw_server_timings": raw.get("timings"),
        "ttft_s": response.get("ttft_s"), "request_e2e_s": response.get("total_s"),
        "completion_tokens_source": response.get("completion_tokens_source"), "output": record.get("output"),
        "finish_reason": response.get("finish_reason"), "request_error": response.get("error") or raw.get("error"),
        "hashes": {k: record.get(k) for k in ("common_sha256", "full_body_sha256", "canonical_input_sha256",
            "wire_request_body_sha256", "frozen_normalized_request_sha256")}, "request_identity_checks": identity_checks,
        "response_sha256": response.get("response_sha256"), "content_sha256": response.get("content_sha256"),
        "raw_transport_sha256": wire.get("body_sha256"), "raw_transport_bytes": wire.get("bytes"),
        "tool_decode_rate_discarded": response.get("decode_tok_s") is not None, "decode_tok_s": None}


def case_data(segment, scenario, source):
    h, records = segment["header"], segment["records"]
    errors = list(segment["errors"]) + header_errors(h)
    steps, lanes = h.get("steps"), h.get("lanes")
    valid_shape = not protocol(h)[2] and type(lanes) is int and lanes in (1, 2)
    requests = [(line, r) for line, r in records if r.get("kind") == "request" and r.get("scenario") == scenario]
    waves = [(line, r) for line, r in records if r.get("kind") == "wave" and r.get("scenario") == scenario]
    observed = [request_observation(r, line, h) for line, r in requests]
    ids = [(r.get("lane"), r.get("step")) for _, r in requests]
    valid_ids = all(type(a) is int and type(b) is int for a, b in ids)
    expected = {(lane, step) for lane in range(lanes) for step in range(steps)} if valid_shape else set()
    actual = set(ids) if valid_ids else set()
    duplicate_ids = sorted({key for key in ids if ids.count(key) > 1}) if valid_ids else []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    complete = valid_shape and valid_ids and len(ids) == len(expected) and actual == expected
    if not complete: errors.append("request step/lane missing, duplicate, wrong type or out of range")
    if any(type(r.get("lanes")) is not int or r.get("lanes") != lanes for _, r in requests): errors.append("request concurrency field differs")
    wave_ids = [r.get("step") for _, r in waves]
    wave_complete = (valid_shape and all(type(s) is int for s in wave_ids) and len(wave_ids) == steps
                     and set(wave_ids) == set(range(steps)))
    if not wave_complete: errors.append("wave step missing, duplicate or out of range")
    for _, wave in waves:
        if (type(wave.get("lanes")) is not int or wave.get("lanes") != lanes or not positive(wave.get("wall_s"))
                or wave.get("arrival_policy") != "barrier_each_step_no_client_think_time"):
            errors.append("invalid original wave wall/concurrency/arrival policy")
        durations = [r["request_e2e_s"] for r in observed if r["step"] == wave.get("step") and positive(r["request_e2e_s"])]
        if positive(wave.get("wall_s")) and durations and wave["wall_s"] + 1e-6 < max(durations):
            errors.append("wave wall shorter than a contained request duration")
    summaries = [r for _, r in records if r.get("kind") == "case_summary" and r.get("scenario") == scenario]
    summary = summaries[0] if len(summaries) == 1 else None
    if summary is None: errors.append("case_summary missing or duplicate")
    original_pass = complete and all(r["original_quality_pass"] for r in observed)
    original_measure = complete and all(r["original_measurement_pass"] for r in observed)
    original_qualified = original_pass and original_measure and all(mapping(r.get("quality")).get("latency_qualified") is True for _, r in requests)
    if summary:
        required = {"lanes": lanes, "expected_requests": len(expected), "observed_requests": len(requests),
            "complete_request_count": complete, "quality_pass": sum(r["original_quality_pass"] for r in observed),
            "quality_total": len(expected), "measurement_pass": sum(r["original_measurement_pass"] for r in observed),
            "latency_qualified": bool(original_qualified)}
        if any(summary.get(k) != v for k, v in required.items()): errors.append("case_summary contradicts original per-request gates/counts")
    if any(not r["eligible"] for r in observed): errors.append("one or more original requests failed/unverified/inconsistent")
    qualified = not errors and bool(observed)
    metrics = {k: None for k in METRICS}
    if qualified:
        metrics = {"ttft_median_s": statistics.median(r["ttft_s"] for r in observed),
            "request_e2e_median_s": statistics.median(r["request_e2e_s"] for r in observed),
            "wave_wall_total_s": sum(w["wall_s"] for _, w in waves),
            "completion_tokens_total": sum(r["raw_usage"]["completion_tokens"] for r in observed)}
        originals = {"qualified_ttft_median_s": metrics["ttft_median_s"], "qualified_e2e_median_s": metrics["request_e2e_median_s"],
            "qualified_wave_wall_total_s": metrics["wave_wall_total_s"], "qualified_server_completion_tokens": metrics["completion_tokens_total"]}
        if any(not positive(summary.get(k)) or not math.isclose(summary[k], v, rel_tol=1e-9, abs_tol=1e-9) for k, v in originals.items()):
            errors.append("case_summary metric differs from complete raw recomputation"); qualified = False
            metrics = {k: None for k in METRICS}
    status = "QUALIFIED" if qualified else "FAIL" if complete and any(not r["original_quality_pass"] for r in observed) else "INCOMPLETE"
    if complete and original_pass and not original_measure: status = "UNVERIFIED"
    if complete and original_pass and original_measure and any(not r["eligible"] for r in observed): status = "ERROR"
    if segment["errors"] or header_errors(h) or any("contradict" in e or "differs" in e or "unknown" in e for e in errors): status = "ERROR" if not qualified else status
    if not requests: status = "NOT_RUN"
    p = h.get("cache_profile_operator_declaration")
    return {"engine": h.get("engine"), "mode": h.get("mode"), "label": h.get("label"), "cache_profile": p,
        "cache_group": cache_group(p), "lanes": lanes, "scenario": scenario, "protocol": h.get("protocol"), "steps": steps,
        "context_capacity": h.get("context_capacity_operator_declaration"),
        "fixture_sha256": mapping(h.get("fixture_manifest")).get("fixture_sha256"),
        "hardware_fingerprint": digest({k: mapping(h.get("launch_manifest")).get(k) for k in ("gpu", "host")}),
        "deployment_fingerprint": deployment(h), "source": source["path"], "source_sha256": source["sha256"],
        "run_index": segment["index"], "header_line": segment["line"], "attempt_id": segment["attempt_id"],
        "status": status, "qualified": qualified, "errors": list(dict.fromkeys(errors)), "expected_requests": len(expected),
        "observed_requests": len(requests), "request_identity_complete": complete, "missing_ids": missing,
        "duplicate_ids": duplicate_ids, "unexpected_ids": unexpected, "wave_identity_complete": wave_complete,
        "expected_waves": steps, "observed_waves": len(waves), "waves": [dict(w, source_line=line) for line, w in waves],
        "original_quality_pass_count": sum(r["original_quality_pass"] for r in observed),
        "original_measurement_pass_count": sum(r["original_measurement_pass"] for r in observed),
        "original_case_summary": summary, "metrics": metrics, "observations": observed,
        "sample_note": f"{steps} correlated sequential waves per run; not {steps} independent samples; no CI"}


def prepare_segment(segment):
    h, records, errors = segment["header"], segment["records"], segment["errors"]
    selected = h.get("selected_scenarios") if isinstance(h.get("selected_scenarios"), list) else []
    errors.extend(header_errors(h))
    ends = [(line, r) for line, r in records if r.get("kind") == "run_end"]
    if len(ends) != 1 or ends[0][1].get("complete") is not True:
        errors.append("run_end complete missing/duplicate; runs never stitched")
    elif records[-1][1].get("kind") != "run_end": errors.append("records after run_end")
    for line, r in records:
        if r.get("kind") not in ("request", "wave", "case_summary", "run_end"):
            errors.append(f"unknown record kind at line{line}")
        if r.get("kind") in ("request", "wave", "case_summary") and r.get("scenario") not in selected:
            errors.append(f"record outside selected scenarios at line{line}")
    summaries = [r for _, r in records if r.get("kind") == "case_summary"]
    if ends:
        end = ends[0][1]
        if end.get("summaries") != summaries: errors.append("run_end summaries differ from original case summaries")
        expected = bool(summaries) and all(s.get("latency_qualified") is True for s in summaries)
        if end.get("qualified") is not expected: errors.append("run_end qualified contradicts original summaries")
    segment["attempt_id"] = digest({"source_header": h, "records": [r for _, r in records]})
    segment["header_identity"] = digest(h)


def group_cases(cases):
    buckets = defaultdict(list)
    for case in cases:
        key = tuple(canonical(case.get(k)) for k in GROUP_FIELDS + ("protocol", "steps", "context_capacity", "fixture_sha256", "deployment_fingerprint"))
        buckets[key].append(case)
    groups = []
    for key, attempts in buckets.items():
        first = attempts[0]
        qualified = bool(attempts) and all(a["qualified"] for a in attempts)
        metrics = {k: statistics.median(a["metrics"][k] for a in attempts) if qualified else None for k in METRICS}
        groups.append({k: first.get(k) for k in GROUP_FIELDS + ("cache_group", "protocol", "steps", "context_capacity", "fixture_sha256", "deployment_fingerprint", "hardware_fingerprint")} |
            {"qualified": qualified, "status": "QUALIFIED" if qualified else "ERROR" if any(a["status"] == "ERROR" for a in attempts) else "FAIL" if any(a["status"] == "FAIL" for a in attempts) else "NOT_RUN" if all(a["status"] == "NOT_RUN" for a in attempts) else "UNVERIFIED" if any(a["status"] == "UNVERIFIED" for a in attempts) else "INCOMPLETE",
             "run_cases": attempts, "run_case_count": len(attempts), "metrics": metrics,
             "statistic": "median of whole-run case statistics; no resampling of steps; no CI", "decode_tok_s": None})
    return sorted(groups, key=lambda g: tuple(str(g.get(k)) for k in GROUP_FIELDS + ("fixture_sha256", "deployment_fingerprint")))


def add_plan(groups, plan):
    for slot in plan:
        allowed = GROUP_FIELDS + ("protocol", "steps", "context_capacity", "fixture_sha256", "deployment_fingerprint")
        if not isinstance(slot, dict) or not all(k in slot for k in GROUP_FIELDS) or not set(slot) <= set(allowed): raise ValueError("plan slots require engine/mode/label/cache_profile/lanes/scenario and only supported identity fields")
        if not any(all(g.get(k) == v for k, v in slot.items()) for g in groups):
            defaults = {"protocol": "opt_in_6_to_10_step_old_media_probe" if slot["scenario"] == MEDIA else "default_three_scenarios",
                "steps": 6 if slot["scenario"] == MEDIA else 30, "context_capacity": 65536}
            groups.append(dict(defaults, **slot, cache_group=cache_group(slot["cache_profile"]), qualified=False, status="NOT_RUN",
                run_case_count=0, run_cases=[], metrics={k: None for k in METRICS}, decode_tok_s=None,
                errors=["expected plan slot has no recorded run"] ))
    return sorted(groups, key=lambda g: tuple(str(g.get(k)) for k in GROUP_FIELDS + ("fixture_sha256", "deployment_fingerprint")))


def build_data(paths, *, plan=(), include=(), exclude=()):
    sources, segments, seen = [], [], {}
    for path in paths:
        source, parsed = parse(path)
        if source.get("sha256") and source["sha256"] in seen:
            seen[source["sha256"]]["aliases"].append(source["path"]); continue
        sources.append(source)
        if source.get("sha256"): seen[source["sha256"]] = source
        for segment in parsed:
            label = str(segment["header"].get("label", ""))
            if (include and not any(fnmatch.fnmatchcase(label, p) for p in include)) or any(fnmatch.fnmatchcase(label, p) for p in exclude): continue
            prepare_segment(segment); segments.append((source, segment))
    # Detect differing partial snapshots of one logical header; exact duplicate
    # segments are retained as aliases, never counted as independent repeats.
    by_header = defaultdict(set)
    shared_errors = defaultdict(list)
    for _, s in segments: by_header[s["header_identity"]].add(s["attempt_id"])
    for _, s in segments: shared_errors[s["attempt_id"]].extend(s["errors"])
    attempts, cases, dedup, duplicate_attempts = [], [], set(), []
    for source, segment in segments:
        if segment["attempt_id"] in dedup:
            duplicate_attempts.append({"attempt_id": segment["attempt_id"], "source": source["path"], "run_index": segment["index"]})
            continue
        dedup.add(segment["attempt_id"])
        segment["errors"] = list(dict.fromkeys(shared_errors[segment["attempt_id"]]))
        if len(by_header[segment["header_identity"]]) > 1: segment["errors"].append("inconsistent snapshots with identical run header")
        h = segment["header"]
        selected = h.get("selected_scenarios") if isinstance(h.get("selected_scenarios"), list) else []
        attempts.append({"attempt_id": segment["attempt_id"], "source": source["path"], "source_sha256": source["sha256"],
            "run_index": segment["index"], "header_line": segment["line"], "header": h,
            "selected_only_coverage": {"selected": selected, "protocol_scenarios": list(protocol(h)[0]),
                "unselected_NOT_RUN": [s for s in protocol(h)[0] if s not in selected], "full_default_matrix_NOT_inferred": True},
            "errors": segment["errors"]})
        for scenario in selected: cases.append(case_data(segment, scenario, source))
    groups = add_plan(group_cases(cases), plan)
    qualified = sum(g["qualified"] for g in groups)
    source_errors = sum(s["status"] == "ERROR" for s in sources)
    source_not_run = sum(s["status"] == "NOT_RUN" for s in sources)
    diagnostics = sum(len(a["errors"]) for a in attempts)
    if not groups:
        status = "ERROR" if source_errors or diagnostics else "NOT_RUN"
    elif all(g["status"] == "NOT_RUN" for g in groups) and not source_errors:
        status = "NOT_RUN"
    elif qualified == len(groups) and not source_errors and not source_not_run and not diagnostics:
        status = "QUALIFIED_DATA"
    else:
        status = "PARTIAL_OR_WITHHELD"
    summary = {"状态": status, "已记录运行段": len(attempts), "已记录场景组": len(groups), "合格场景组": qualified,
        "未跑场景组": sum(g["status"] == "NOT_RUN" for g in groups), "失败或未完成组": sum(not g["qualified"] and g["status"] != "NOT_RUN" for g in groups),
        "输入ERROR文件": source_errors, "未找到或未跑输入文件": source_not_run, "运行段诊断数": diagnostics,
        "说明": "只显示每run完整且所有原质量/测量门槛通过的场景。30步为相关会话步骤，未计算工具decode速度或显著性；selected场景不代表完整矩阵。",
        "实际性能数值": "见qualified groups；NOT_RUN/FAIL/INCOMPLETE/ERROR均为null，不填零值。"}
    return {"schema": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "tool_sha256": file_sha(HERE), "frozen_control_sha256": file_sha(CONTROL), "plan": list(plan),
        "selection": {"include_labels": list(include), "exclude_labels": list(exclude)},
        "attempts": attempts, "duplicate_attempts": duplicate_attempts, "groups": groups, "summary_zh": summary, "limitations": LIMITATIONS, "context_figures": []}


def render(data, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    facets = defaultdict(list)
    for group in data["groups"]:
        values = [group.get(k) for k in ("protocol", "steps", "context_capacity", "fixture_sha256", "cache_group", "mode", "lanes", "scenario", "hardware_fingerprint")]
        key = tuple(v if isinstance(v, (str, int, float, bool, type(None))) else canonical(v) for v in values)
        facets[key].append(group)
    figures = []
    for index, (key, groups) in enumerate(sorted(facets.items(), key=lambda item: tuple(str(v) for v in item[0]))):
        labels = ["\n".join(textwrap.wrap(f"{g['engine']} | {g['label']} | {g['cache_profile']} | cfg{str(g.get('deployment_fingerprint') or '')[:6]} | nRun={g['run_case_count']}", 48)) for g in groups]
        height = max(3.5, len(groups) * (max(label.count('\n') + 1 for label in labels) * .22 + .35) + 1.8)
        fig, axes = plt.subplots(1, 4, figsize=(18, height), sharey=True)
        titles = ("Semantic TTFT median (s)", "Request E2E median (s)", "Wave wall total / replay (s)", "Actual output tokens / replay")
        for ax, metric, title in zip(axes, METRICS, titles):
            values = []
            for y, g in enumerate(groups):
                value = g["metrics"][metric]
                if g["qualified"] and value is not None:
                    ax.scatter(value, y, s=38, color="#27689a"); values.append(value)
                    ax.annotate(f"{value:.4g}", (value, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
                else: ax.text(.025, y, f"withheld: {g['status']}", transform=ax.get_yaxis_transform(), fontsize=8, color="#875d50")
            ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
            if values: ax.set_xlim(0, max(values) * 1.25)
            else: ax.set_xlim(0, 1); ax.set_xticks([])
        axes[0].set_yticks(range(len(groups)), labels)
        axes[0].set_ylim(len(groups) - .5, -.5) # text-only withheld rows need explicit plot margins
        fig.suptitle(f"CUA-like | {key[4]} | {key[5]} | C={key[6]} | {key[7]} | {key[1]} waves", fontsize=12)
        fig.text(.5, .015, "No CI: sequential steps are correlated. Whole-case statistics; failed/incomplete points withheld. Alphabetical labels, no ranking.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .94))
        slug = lambda value: re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))[:80]
        name = f"cua-{index:02d}-{slug(key[4])}-{slug(key[5])}-c{slug(key[6])}-{slug(key[7])}"
        png, svg = directory / (name + ".png"), directory / (name + ".svg")
        fig.savefig(png, dpi=160); fig.savefig(svg); plt.close(fig)
        figures.append({"title": f"公开CUA-like · {key[7]} · {key[5]} · {key[6]}路 · {key[4]}",
            "image_path": str(png.resolve()), "svg_path": str(svg.resolve()), "png_sha256": file_sha(png), "svg_sha256": file_sha(svg),
            "caption": "原质量与测量门槛逐run全部通过才画点；失败/缺失无0值。步骤相关，无置信区间和工具decode tok/s。缓存隔离组分开，实际输出量并列。"})
    data["context_figures"] = figures


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value): return f"NONFINITE:{value}"
    if isinstance(value, dict): return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe_json(v) for v in value]
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="explicit JSONL files or quoted glob patterns")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--plan", type=Path, help="optional expected group slots; absent slots become NOT_RUN")
    ap.add_argument("--data-only", action="store_true", help="stdlib only; never import matplotlib")
    ap.add_argument("--include-label", action="append", default=[])
    ap.add_argument("--exclude-label", action="append", default=[])
    args = ap.parse_args(argv)
    out = args.output_dir.resolve()
    if not any(out.is_relative_to(ROOT / p) for p in ("artifacts", "docs/validation")):
        ap.error("--output-dir must be in ignored artifacts/ or docs/validation/")
    paths = []
    for pattern in args.inputs: paths.extend(sorted(glob.glob(pattern)) or [pattern])
    plan = json.loads(args.plan.read_text()) if args.plan else []
    if not isinstance(plan, list): ap.error("--plan must be a JSON list")
    data = build_data(paths, plan=plan, include=args.include_label, exclude=args.exclude_label)
    if args.plan: data["plan_source"] = {"path": str(args.plan.resolve()), "sha256": file_sha(args.plan)}
    out.mkdir(parents=True, exist_ok=True)
    if not args.data_only: render(data, out)
    for name, value in (("data.json", data), ("summary-zh.json", data["summary_zh"]), ("context-figures.json", data["context_figures"])):
        (out / name).write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data["summary_zh"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
