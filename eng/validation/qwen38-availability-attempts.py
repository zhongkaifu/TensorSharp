#!/usr/bin/env python3
"""Index local startup evidence without turning failed retries into successes.

This is an availability audit, not a quality grader or performance ranking.
Only reads already downloaded evidence; it never contacts a server or GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def identity(path: Path, remote_path=None):
    value = {"path": str(path), "exists_locally": path.is_file(), "sha256": None}
    if remote_path:
        value["remote_path"] = remote_path
    if path.is_file():
        data = path.read_bytes()
        value.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return value


def classify(log):
    line = next((s for s in log.splitlines() if "ValueError: To serve at least one request" in s), "")
    if line:
        return "UNSUPPORTED", "该配置的上下文/KV容量准入被拒绝；没有执行正式请求。", line.strip()
    line = next((s for s in log.splitlines() if "ValueError: Free memory on device" in s), "")
    if line:
        return "UNSUPPORTED", "该配置要求的GPU内存比例超过启动时可用内存；没有执行正式请求。", line.strip()
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
        name, separator, value = arg.partition("=")
        if name in names:
            flags[name] = value if separator else argv[i + 1] if i + 1 < len(argv) else None
    return flags


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
        env = manifest.get("environment", {})
        item = dict(engine=engine, mode=mode, label=label, scope="this exact label/configuration only",
                    observed_cases=status["observed_cases"], expected_cases=status["expected_cases"],
                    process_exit_code=status.get("process_exit_code"), quality_passed=status.get("quality_passed", False),
                    verdict_counts=status.get("verdict_counts", {}),
                    artifacts={"status": identity(status_path), "manifest": identity(manifest_path),
                               "log": identity(log_path), "raw": identity(raw_path, status.get("result_file"))},
                    configuration={"context_tokens": manifest.get("context_tokens"), "lanes": manifest.get("lanes"),
                                   "explicit_cli_flags": effective_flags(manifest.get("argv", [])),
                                   "environment": {key: {"recorded": key in env, "value": env.get(key)} for key in
                                                   ("TMPDIR", "MAX_JOBS", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_WORKSPACE_BASE")}})
        if status["observed_cases"] == status["expected_cases"] > 0 and status.get("process_exit_code") == 0:
            item.update(status="EXECUTED", reason="完整正式case已执行；质量仍以原始case判定为准，此状态不是quality PASS。")
            completed.append(item)
        else:
            verdict, reason, evidence = classify(log)
            item.update(status=verdict, reason=reason, primary_log_evidence=evidence)
            missing.append(item)
    # A run header/warmups alone supplies no quality or speed result, even when
    # there is no separate controller status JSON (e.g. an empty scenario job).
    observed_modes = set()
    for raw_path in sorted((args.root / "raw").glob("*.jsonl")):
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        headers = [r for r in rows if r.get("kind") == "run"]
        if not headers:
            continue
        header = headers[-1]
        if any(r.get("kind") == "case" for r in rows):
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
                  tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "failed_or_not_run_attempts": len(missing), "executed_profiles": len(completed)}))


if __name__ == "__main__":
    main()
