#!/usr/bin/env python3
"""Index local startup evidence without turning failed retries into successes.

This is an availability audit, not a quality grader or performance ranking.
Only reads already downloaded evidence; it never contacts a server or GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re


REPORT_PATH = Path(__file__).resolve().parents[1] / "qwen38-engine-report.py"
REPORT_SPEC = importlib.util.spec_from_file_location("qwen38_availability_quality_rules", REPORT_PATH)
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(REPORT)


def identity(path: Path, remote_path=None):
    value = {"path": str(path), "exists_locally": path.is_file(), "sha256": None}
    if remote_path:
        value["remote_path"] = remote_path
    if path.is_file():
        data = path.read_bytes()
        value.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return value


def classify(log):
    line = next((s for s in log.splitlines() if 'ignis.artifact.sidecar_missing' in s), "")
    if line:
        return "ERROR", "启动所需的官方模型来源sidecar缺失；属于安装/文件准备错误，没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if 'the VRAM plan needs ' in s and 'VRAM budget' in s), "")
    if line:
        return "UNSUPPORTED", "该配置的VRAM规划无法容纳权重、工作区、状态及一个完整声明上下文；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "automatic KV capacity requires " in s and " bytes are available after weights" in s), "")
    if line:
        return "UNSUPPORTED", "该配置所需的KV运行时预算和自动预留空间超过权重加载后可用设备内存；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "Device budget cannot hold model weights and required execution resources" in s), "")
    if line:
        return "UNSUPPORTED", "该配置的设备预算无法容纳模型权重和必需执行资源；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "ValueError: To serve at least one request" in s), "")
    if line:
        return "UNSUPPORTED", "该配置的上下文/KV容量准入被拒绝；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "ValueError: Free memory on device" in s), "")
    if line:
        return "UNSUPPORTED", "该配置要求的GPU内存比例超过启动时可用内存；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=" in s), "")
    if line:
        return "UNSUPPORTED", "该配置加载权重并预留运行时及SSM状态后，没有可分配的KV缓存预算；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform plugin is available." in s), "")
    if line:
        return "ERROR", "启动时未能初始化计算设备；属于运行环境错误，不能推断模型、精度或上下文不受支持。", line.strip()
    if "No space left on device" in log:
        return "ERROR", "启动/JIT临时磁盘空间不足；属于环境错误，不代表模型或GPU上下文不支持。", "No space left on device"
    if "Operation not supported" in log and ("ZMQ" in log or "ZeroMQ" in log or "zmq" in log):
        return "ERROR", "运行时临时目录不支持ZeroMQ IPC套接字；属于环境错误。", "ZeroMQ IPC: Operation not supported"
    lines = [s.strip() for s in log.splitlines() if re.search(r"(?:Error|ERROR|error|fatal):", s)]
    return "ERROR", "启动或评估失败；没有足够正式case，不能算通过。", lines[-1] if lines else "See retained log/status."


def effective_flags(argv):
    flags = {}
    names = {"--cpu-offload-gb", "--gpu-memory-utilization", "--max-num-batched-tokens",
             "--max-cudagraph-capture-size", "--kv-cache-dtype", "--max-model-len", "--max-num-seqs"}
    for i, arg in enumerate(argv):
        if arg in ("--enable-prefix-caching", "--no-enable-prefix-caching"):
            flags["--enable-prefix-caching"] = arg == "--enable-prefix-caching"
            continue
        name, separator, value = arg.partition("=")
        if name in names:
            flags[name] = value if separator else argv[i + 1] if i + 1 < len(argv) else None
    return flags


def last_run(path):
    header, cases = None, []
    if not path.is_file():
        return header, cases
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "run":
            header, cases = row, []
        elif row.get("kind") == "case" and header is not None:
            cases.append(row)
    return header, cases


def case_matrix(header, cases):
    """Require the declared scenario × repeat matrix, not merely its size."""
    if not header:
        return False, 0
    scenarios, repeats = header.get("scenarios"), header.get("repeats")
    if (not isinstance(scenarios, list) or not scenarios
            or any(not isinstance(s, str) or not s.strip() for s in scenarios)
            or len(set(scenarios)) != len(scenarios)
            or type(repeats) is not int or repeats <= 0):
        return False, 0
    expected = {(s, r) for s in scenarios for r in range(repeats)}
    actual = []
    for case in cases:
        if not isinstance(case.get("scenario"), str) or type(case.get("repeat")) is not int:
            return False, len(expected)
        if header.get("run_id") is not None and case.get("run_id") != header["run_id"]:
            return False, len(expected)
        if any(key in case and case[key] != header.get(key) for key in ("engine", "mode", "label")):
            return False, len(expected)
        actual.append((case["scenario"], case["repeat"]))
    return len(actual) == len(set(actual)) == len(expected) and set(actual) == expected, len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="local evidence directory containing raw/ and external-engine-*/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engines", default="tensorsharp,llamacpp,vllm,sglang,ninfer,infernix,velox,ignis")
    args = parser.parse_args()
    missing, completed, indexed = [], [], set()
    for status_path in sorted((args.root / "external-engine-status").glob("*.json")):
        status = json.loads(status_path.read_text())
        if "observed_cases" not in status or "expected_cases" not in status:
            continue
        match = re.fullmatch(r"(.+?)-(plain|dflash2)-(.+)", status_path.stem)
        if not match:
            continue
        engine, mode, label = match.groups()
        indexed.add((engine, mode, label))
        manifest_path = args.root / "external-engine-manifests" / status_path.name
        log_path = args.root / "external-engine-logs" / (status_path.stem + ".log")
        raw_path = args.root / "raw" / (status_path.stem + ".jsonl")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        raw_header, raw_cases = last_run(raw_path)
        raw_identity_matches = bool(raw_header) and tuple(raw_header.get(k) for k in ("engine", "mode", "label")) == (engine, mode, label)
        matrix_complete, header_expected = case_matrix(raw_header, raw_cases)
        counts_valid = (type(status["observed_cases"]) is int and status["observed_cases"] >= 0
                        and type(status["expected_cases"]) is int and status["expected_cases"] >= 0)
        count_complete = (counts_valid and matrix_complete and len(raw_cases) == status["observed_cases"]
                          == status["expected_cases"] == header_expected > 0)
        exit_ok = type(status.get("process_exit_code")) is int and status["process_exit_code"] == 0
        env = manifest.get("environment", {})
        item = dict(engine=engine, mode=mode, label=label, scope="this exact label/configuration only",
                    observed_cases=status["observed_cases"], expected_cases=status["expected_cases"],
                    verified_local_raw_cases=len(raw_cases),
                    last_raw_header_identity_matches=raw_identity_matches,
                    last_raw_case_matrix_complete=matrix_complete, header_expected_cases=header_expected,
                    process_exit_code=status.get("process_exit_code"), controller_quality_passed=status.get("quality_passed", False),
                    quality_passed=status.get("quality_passed") is True and exit_ok and raw_identity_matches
                                   and count_complete and all(REPORT.quality_passed(c) for c in raw_cases),
                    verdict_counts=status.get("verdict_counts", {}),
                    artifacts={"status": identity(status_path), "manifest": identity(manifest_path),
                               "log": identity(log_path), "raw": identity(raw_path, status.get("result_file"))},
                    configuration={"context_tokens": manifest.get("context_tokens"), "lanes": manifest.get("lanes"),
                                   "readiness_budget": manifest.get("readiness_budget"),
                                   "explicit_cli_flags": effective_flags(manifest.get("argv", [])),
                                   "environment": {key: {"recorded": key in env, "value": env.get(key)} for key in
                                                   ("TMPDIR", "MAX_JOBS", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_WORKSPACE_BASE")}})
        if not counts_valid:
            item.update(status="INCOMPLETE", quality_passed=False, reason="controller case计数类型无效；不能算通过。")
            missing.append(item)
        elif raw_header and not raw_identity_matches:
            item.update(status="INCOMPLETE", quality_passed=False,
                        reason="本地raw最后一个run header的engine/mode/label与controller不一致；不能合并或算通过。")
            missing.append(item)
        elif status["observed_cases"] > 0 and not count_complete:
            item.update(status="INCOMPLETE", quality_passed=False,
                        reason="最后run的scenario×repeat集合不完整/重复/越界，或与controller计数不符；不能核验或算通过。")
            missing.append(item)
        elif count_complete and exit_ok:
            item.update(status="EXECUTED", reason="完整正式case已执行；质量仍以原始case判定为准，此状态不是quality PASS。")
            completed.append(item)
        elif status["observed_cases"] == 0 and exit_ok:
            item.update(status="NOT_RUN", quality_passed=False, reason="没有正式case；准备或warmups不能算通过。")
            missing.append(item)
        else:
            verdict, reason, evidence = classify(log)
            item.update(status=verdict, quality_passed=False, reason=reason, primary_log_evidence=evidence)
            missing.append(item)
    # A run header/warmups alone supplies no quality or speed result, even when
    # there is no separate controller status JSON (e.g. an empty scenario job).
    observed_modes = set()
    for raw_path in sorted((args.root / "raw").glob("*.jsonl")):
        header, cases = last_run(raw_path)
        if not header:
            continue
        if cases:
            observed_modes.add((header.get("engine"), header.get("mode")))
            continue
        key = tuple(header.get(k) for k in ("engine", "mode", "label"))
        if key in indexed:
            continue
        missing.append(dict(zip(("engine", "mode", "label"), key), status="NOT_RUN", observed_cases=0,
                            quality_passed=False, reason="只有run header/warmups，没有正式case；不能作为通过或性能数据。",
                            artifacts={"raw": identity(raw_path)}, scenarios=header.get("scenarios")))
    for engine in args.engines.split(","):
        for mode in ("plain", "dflash2"):
            if (engine, mode) not in observed_modes:
                missing.append(dict(engine=engine, mode=mode, status="NOT_RUN", observed_cases=0,
                                    quality_passed=False, scope="local formal-case evidence availability",
                                    reason="本地尚无该engine/mode的正式case；不能算通过，也不代表引擎不支持。", artifacts={}))
    output = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
                  policy="UNSUPPORTED means only the recorded configuration's capacity refusal, not a whole-engine impossibility. ERROR is an environment/startup error. NOT_RUN has no formal cases. Never overwrite an earlier failed label with a retry; EXECUTED is not quality PASS.",
                  missingEngines=missing, executedProfiles=completed,
                  preparationEvidence=[identity(p) for p in sorted((args.root / "external-engine-status").glob("*cpu-precompile*.json"))],
                  orchestrationEvidence=[identity(p) for p in sorted((args.root / "external-engine-status").glob("*startup-extension*.json"))],
                  configurationEvidence=[identity(p) for p in sorted((args.root / "external-engine-status").glob("*source-evidence.json"))],
                  quality_rule_source=identity(REPORT_PATH),
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "failed_or_not_run_attempts": len(missing), "executed_profiles": len(completed)}))


if __name__ == "__main__":
    main()
