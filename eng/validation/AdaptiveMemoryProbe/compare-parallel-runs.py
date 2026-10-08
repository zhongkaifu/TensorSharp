#!/usr/bin/env python3
"""Compare isolated serial/parallel-K processes with one unchanged native library.

Inputs are run-bounded-probe.py execution.json files, not bare model reports.
Requires serial/parallel/parallel/serial, two processes and three measured
requests per arm, successful exits, capture disabled and identical token paths.
Numerical raw hashes may differ; qualify them separately with compare-captures.py.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

FLAG = "TS_GGML_Q8_PARALLEL_VECTOR"
MARKER = "[q8-f32] Experimental parallel-K F32 vector selected (N=1); N>1 retains K-ordered arithmetic."


def require(value, message):
    if not value:
        raise ValueError(message)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def utc(value):
    # System.Text.Json preserves DateTime's 100 ns ticks. Python 3.10 accepts
    # at most microseconds; truncate only the sub-microsecond remainder.
    value = re.sub(r"(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d$)", r"\1", value)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load(execution_path, arm):
    execution = json.loads(execution_path.read_text(encoding="utf-8-sig"))
    require(execution.get("complete") is True and execution.get("timed_out") is False
            and execution.get("exit_code") == 0 and not execution.get("error"), "Process did not exit successfully")
    command = execution.get("command", [])
    require(isinstance(command, list) and command.count("--output") == 1, "Execution has no unique probe output")
    report_path = Path(command[command.index("--output") + 1]) / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    require(report.get("Executed") is True and not report.get("Error") and report.get("NativeShutdown") is True,
            "Model or native shutdown failed")
    require(type(execution.get("pid")) is int and execution["pid"] == report.get("ProcessId"),
            "Exit evidence belongs to a different process")
    require(utc(execution["started_utc"]) <= utc(report["StartedUtc"])
            <= utc(report["FinishedUtc"]) <= utc(execution["finished_utc"]), "Report times are outside the process lifetime")
    require(report.get("CaptureLogits") is False and report.get("LogitCapture") is None,
            "Correctness capture cannot count as quiet throughput")
    require(isinstance(report.get("RequestedOptions"), dict), "Missing original probe options")
    env = report.get("Environment", {})
    require(env.get(FLAG) == ("0" if arm == "serial" else "1"), "Experimental flag selection differs from the assigned arm")
    log_path = execution_path.parent / "process.log"
    log = log_path.read_text(encoding="utf-8-sig", errors="replace")
    require(log.count(MARKER) == (0 if arm == "serial" else 1), "Actual native experimental-selection log is missing or inconsistent")
    native = report.get("Native", [])
    require(len(native) == 1 and is_hash(native[0].get("Sha256")) and native[0].get("FileName"),
            "Expected one actual loaded native library identity")
    require(is_hash(report.get("ModelSha256")) and is_hash(report.get("ModelsAssemblySha256"))
            and is_hash(report.get("ProbeAssemblySha256")) and report.get("ManagedAssembliesSha256")
            and all(is_hash(value) for value in report["ManagedAssembliesSha256"].values()), "Incomplete managed or checkpoint identities")
    require(report.get("ProbeAssemblyPath") and any(Path(item).resolve() == Path(report["ProbeAssemblyPath"]).resolve()
            for item in command if isinstance(item, str) and item.lower().endswith(".dll")),
            "Execution command did not launch the recorded probe assembly")
    binary_directory = Path(report["ProbeAssemblyPath"]).parent.resolve()
    require(Path(native[0]["FileName"]).parent.resolve() == binary_directory, "Native library did not load from the isolated probe directory")
    geometry = report.get("ModelGeometry", {})
    require(geometry.get("Architecture") and all(type(geometry.get(key)) is int and geometry[key] > 0 for key in
            ("HiddenSize", "NumLayers", "NumHeads", "NumKVHeads", "Vocabulary", "Context")), "Invalid model geometry")
    steps, repeats = report.get("Steps"), report.get("Repeats")
    require(type(steps) is int and steps >= 2 and repeats == 3, "Require three complete measured requests per process")
    valid_tokens = lambda ids: isinstance(ids, list) and all(type(x) is int and 0 <= x < geometry["Vocabulary"] for x in ids)
    prompt = report.get("Prompt")
    require(valid_tokens(prompt) and prompt and len(prompt) + steps <= geometry["Context"], "Invalid prompt/context")
    teacher = report.get("Teacher")
    if report.get("Generation") == "teacher-forced":
        require(valid_tokens(teacher) and len(teacher) == steps and is_hash(report.get("TeacherSha256")), "Invalid teacher history")
    else:
        require(report.get("Generation") == "raw-greedy" and teacher is None, "Unknown conditioning")
    rows = report.get("Records", [])
    require([row.get("Run") for row in rows] == [-1, 0, 1, 2]
            and all(row.get("Warmup") is (i == 0) for i, row in enumerate(rows)), "Missing excluded warmup or complete measured requests")
    for row in rows:
        require(row.get("PromptTokens") == len(prompt) and row.get("DecodeCalls") == steps - 1
                and valid_tokens(row.get("Generated")) and len(row["Generated"]) == steps
                and row.get("Consumed") == (teacher if teacher is not None else row["Generated"])[:-1]
                and is_hash(row.get("LogitsSha256")), "Incomplete token history or raw-logit identity")
        for ms_key, rate_key, count in (("PrefillMilliseconds", "PrefillTokensPerSecond", len(prompt)),
                                       ("DecodeMilliseconds", "DecodeTokensPerSecond", steps - 1)):
            duration, rate = row.get(ms_key), row.get(rate_key)
            require(isinstance(duration, (float, int)) and math.isfinite(duration) and duration > 0
                    and isinstance(rate, (float, int)) and math.isfinite(rate) and rate > 0
                    and math.isclose(rate, count * 1000 / duration, rel_tol=1e-10), "Timing denominator or rate is invalid")
    if report.get("Mode") == "adaptive":
        pools = report.get("AfterDispose")
        require(isinstance(pools, list) and pools and all(p.get("Reserved") == 0 and p.get("Committed") == 0 for p in pools),
                "Adaptive charged owners remain")
    else:
        require(report.get("Mode") == "resident", "Unknown placement mode")
    return {"arm": arm, "report": report, "execution": execution, "directory": str(binary_directory),
            "report_path": str(report_path), "execution_path": str(execution_path),
            "execution_sha256": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()}


def compare(serial, parallel):
    result = {"ComparableForTiming": False, "Errors": [], "Measurements": {}, "Executions": [],
              "Qualification": "Matching completed workloads, not a numerical or semantic PASS. Full-logit hashes may differ; a separate fixed-history numerical gate is mandatory. Idle hardware and thermal/file-cache controls remain external evidence."}
    try:
        require(len(serial) == len(parallel) == 2, "Require exactly two fresh processes per arm")
        paths = [*serial, *parallel]
        require(len({str(path.resolve()) for path in paths}) == 4, "Execution evidence was reused")
        runs = [load(path, arm) for arm, paths in (("serial", serial), ("parallel", parallel)) for path in paths]
        require(len({run["execution"]["pid"] for run in runs}) == 4, "Process IDs must be distinct")
        ordered = sorted(runs, key=lambda run: utc(run["execution"]["started_utc"]))
        require([run["arm"] for run in ordered] == ["serial", "parallel", "parallel", "serial"], "Expected serial/parallel/parallel/serial order")
        require(all(utc(a["execution"]["finished_utc"]) <= utc(b["execution"]["started_utc"])
                    for a, b in zip(ordered, ordered[1:])), "Measured processes overlapped")
        directories = {arm: {run["directory"] for run in runs if run["arm"] == arm} for arm in ("serial", "parallel")}
        require(all(len(paths) == 1 for paths in directories.values()) and directories["serial"].isdisjoint(directories["parallel"]),
                "Use one distinct isolated binary directory per arm")
        baseline = runs[0]["report"]
        identity_keys = ("ModelSha256", "ModelBytes", "ModelsAssemblySha256", "ProbeAssemblySha256", "ManagedAssembliesSha256",
                         "ModelGeometry", "Prompt", "Mode", "Generation", "Teacher", "Steps", "Repeats")
        token_path = [(row["Generated"], row["Consumed"]) for row in baseline["Records"]]
        baseline_env = {key: value for key, value in baseline["Environment"].items() if key != FLAG}
        baseline_options = {key: value for key, value in baseline["RequestedOptions"].items() if key not in ("--output", "--teacher")}
        for run in runs:
            report = run["report"]
            require(all(report.get(key) == baseline.get(key) for key in identity_keys), "Model/managed identity, geometry or conditioning differs")
            require(report["Native"][0]["Sha256"] == baseline["Native"][0]["Sha256"], "Native libraries differ; this comparison isolates only the experiment flag")
            require({key: value for key, value in report["Environment"].items() if key != FLAG} == baseline_env,
                    "Other recorded runtime knobs differ")
            require({key: value for key, value in report["RequestedOptions"].items() if key not in ("--output", "--teacher")} == baseline_options,
                    "Other requested probe settings differ")
            require([(row["Generated"], row["Consumed"]) for row in report["Records"]] == token_path,
                    "Argmax or consumed histories differ")
            result["Executions"].append({key: value for key, value in run.items() if key not in ("report", "execution")})
        for arm in ("serial", "parallel"):
            rows = [row for run in runs if run["arm"] == arm for row in run["report"]["Records"] if not row["Warmup"]]
            result["Measurements"][arm] = {key: {"Median": statistics.median(row[key] for row in rows),
                "Minimum": min(row[key] for row in rows), "Maximum": max(row[key] for row in rows), "Samples": len(rows)}
                for key in ("PrefillTokensPerSecond", "DecodeTokensPerSecond")}
        result["ParallelToSerialRatio"] = {key: result["Measurements"]["parallel"][key]["Median"] / result["Measurements"]["serial"][key]["Median"]
                                            for key in ("PrefillTokensPerSecond", "DecodeTokensPerSecond")}
        result["RawLogitHashes"] = {arm: [[row["LogitsSha256"] for row in run["report"]["Records"]]
                                        for run in runs if run["arm"] == arm] for arm in ("serial", "parallel")}
        result["ComparableForTiming"] = True
    except Exception as error:
        result["Errors"].append(str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", nargs=2, type=Path, required=True)
    parser.add_argument("--parallel", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.serial, args.parallel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ComparableForTiming"] else 1)
