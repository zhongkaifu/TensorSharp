#!/usr/bin/env python3
"""Compare repeated DirectCudaModelProbe processes, with exact output gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


FILE_IDENTITY_FIELDS = ("model_files", "model_file_count", "model_total_file_bytes", "model_file_identity_incomplete")


def checkpoint_files(report):
    present = [key in report for key in FILE_IDENTITY_FIELDS]
    if not any(present):
        return None  # Legacy reports recorded only the selected input file.
    if not all(present) or report["model_file_identity_incomplete"] is not False:
        raise ValueError("Incomplete checkpoint shard identity")
    files = report["model_files"]
    if not isinstance(files, list) or not files or len(files) != report["model_file_count"]:
        raise ValueError("Invalid checkpoint shard count")
    for file in files:
        if (not isinstance(file, dict) or not isinstance(file.get("path"), str) or not file["path"]
                or type(file.get("bytes")) is not int or file["bytes"] <= 0
                or not isinstance(file.get("last_write_utc"), str) or not file["last_write_utc"]):
            raise ValueError("Invalid checkpoint shard identity")
    if len({file["path"] for file in files}) != len(files):
        raise ValueError("Duplicate checkpoint shard path")
    if sum(file["bytes"] for file in files) != report["model_total_file_bytes"]:
        raise ValueError("Checkpoint total bytes do not match its shards")
    if report.get("model_declared_file_count", len(files)) != len(files):
        raise ValueError("Checkpoint paths do not cover the declared shard count")
    # Generated fixtures are rewritten in each process and are already required
    # to have equal complete payload hashes; their paths and mtimes may differ.
    return [file["bytes"] for file in files] if report.get("synthetic") else files


def validate(report):
    if report.get("passed") is not True or report.get("repeated_final_logits_exact") is not True:
        raise ValueError("Probe failed or repeated final logits changed")
    if report.get("quality_checked") and report.get("quality_passed") is not True:
        raise ValueError("Semantic checks failed")
    checkpoint_files(report)
    logits = report["logits"]
    path = Path(logits["path"])
    if logits["format"] != "little-endian-float32" or logits["rows"] <= 0 or logits["columns"] <= 0:
        raise ValueError("Invalid logit shape/format")
    if path.stat().st_size != logits["rows"] * logits["columns"] * 4 or digest(path) != logits["sha256"]:
        raise ValueError("Logit evidence is incomplete or its checksum changed")
    measured = [r for r in report["runs"] if r["warmup"] is False]
    if not measured:
        raise ValueError("No measured iterations")
    for row in report["runs"]:
        if row.get("prefill_tokens") != len(report["prompt_tokens"]) or row.get("decode_tokens") != len(report["forced_tokens"]):
            raise ValueError("Timing token counts do not match the workload")
        if not row["warmup"] and row["final_logit_sha256"] != logits["final_logit_sha256"]:
            raise ValueError("Timed and captured final logits disagree")
        for name in ("prefill_ms", "decode_ms"):
            if not math.isfinite(row[name]) or row[name] <= 0:
                raise ValueError("Invalid timing")
    return measured


def compare(before, after, max_regression=5):
    if not before or len(before) != len(after):
        raise ValueError("Supply equal nonempty process counts")
    reference = before[0]
    reference_files = checkpoint_files(reference)
    identities = ("backend", "architecture", "model_bytes", "max_context", "prompt_tokens", "forced_tokens", "quality_checked",
                  "synthetic", "dimensions", "source_dimensions", "moe_cpu_layers_requested", "synthetic_model_sha256")
    signatures = ("managed_assemblies_sha256", "ptx_sha256", "native_sha256")
    times = {arm: {name: [] for name in ("prefill_ms", "decode_ms", "total_ms")} for arm in ("before", "after")}
    for arm, reports in (("before", before), ("after", after)):
        for report in reports:
            measured = validate(report)
            for key in identities:
                if report.get(key) != reference.get(key):
                    raise ValueError(f"Workload/identity mismatch: {key}")
            if checkpoint_files(report) != reference_files:
                raise ValueError("Checkpoint shard identity mismatch")
            if not report.get("synthetic"):
                for key in ("model_path", "model_last_write_utc"):
                    if report.get(key) != reference.get(key):
                        raise ValueError(f"Checkpoint identity mismatch: {key}")
            flag = "TENSORSHARP_CUDA_MOE_FUSION" if report["backend"] == "Cuda" else "TS_Q4E_PREFILL_COMBINE"
            expected_flag = "0" if arm == "before" else "1"
            if report.get("environment", {}).get(flag) != expected_flag:
                raise ValueError(f"Expected explicit {flag}={expected_flag} in {arm} arm")
            # The runner may set both feature switches per arm. Only the switch
            # for this backend affects execution; permit the other when it agrees.
            feature_flags = ("TENSORSHARP_CUDA_MOE_FUSION", "TS_Q4E_PREFILL_COMBINE")
            for feature_flag in feature_flags:
                value = report.get("environment", {}).get(feature_flag)
                if value is not None and value != expected_flag:
                    raise ValueError(f"Unexpected {feature_flag}={value} in {arm} arm")
            controls = lambda r: {key: value for key, value in r.get("environment", {}).items() if key not in feature_flags}
            if controls(report) != controls(reference):
                raise ValueError("Other captured environment controls changed")
            if report["backend"] == "Cuda" and report.get("fusion", {}).get("kernel_available") is not True:
                raise ValueError("The loaded PTX does not support the fused CUDA kernel")
            if report["backend"] == "Cuda" and report.get("fusion", {}).get("process_enabled") is not (arm == "after"):
                raise ValueError("Effective CUDA fusion policy does not match the selected arm")
            if len(measured) != sum(row["warmup"] is False for row in reference["runs"]):
                raise ValueError("Measured iteration count changed")
            # This runner is for switches in the same binary, not unrelated builds.
            for key in signatures:
                if sorted(report[key].values()) != sorted(reference[key].values()):
                    raise ValueError(f"Binary identity mismatch: {key}")
            for key in ("rows", "columns", "sha256"):
                if report["logits"][key] != reference["logits"][key]:
                    raise ValueError(f"Full-model logits differ: {key}")
            # The unchanged implementation has a startup capture-path hash
            # difference. Preserve this sequence and require corresponding
            # warmups to match across both arms, without inferring its cause.
            warmup_hashes = lambda r: [x["final_logit_sha256"] for x in r["runs"] if x["warmup"]]
            if warmup_hashes(report) != warmup_hashes(reference):
                raise ValueError("Warmup count or corresponding final logits changed")
            semantic = lambda r: [(q["name"], q["prompt_tokens"], q["generated_tokens"], q["eos"], q["passed"]) for q in r["quality"]]
            if semantic(report) != semantic(reference):
                raise ValueError("Semantic cases or generated token streams changed")
            for name in ("prefill_ms", "decode_ms"):
                times[arm][name].append(statistics.median(r[name] for r in measured))
            times[arm]["total_ms"].append(statistics.median(r["prefill_ms"] + r["decode_ms"] for r in measured))
    metrics = {}
    for name in times["before"]:
        b, a = (statistics.median(times[arm][name]) for arm in ("before", "after"))
        metrics[name] = {"before": b, "after": a, "speedup_percent": (b / a - 1) * 100,
                         "latency_change_percent": (a / b - 1) * 100,
                         "before_process_medians": times["before"][name], "after_process_medians": times["after"][name]}
    passed = all(row["latency_change_percent"] <= max_regression for row in metrics.values())
    limitations = ["Medians summarize these workloads on this machine; small changes may be measurement noise.",
                  "Kernel allocation accounting and driver/OS memory snapshots are reported separately by the probes."]
    if reference_files is None:
        limitations.append("Legacy probe reports identify only the selected input file. Split-checkpoint identity and total size require separately verified shard provenance.")
    return {"passed": passed, "exact_full_logits": True, "exact_quality_tokens": True, "exact_corresponding_warmup_logits": True,
            "quality_checked": reference["quality_checked"], "process_pairs": len(before),
            "max_regression_percent": max_regression, "metrics": metrics,
            "complete_checkpoint_file_identity_checked": reference_files is not None,
            "limitations": limitations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", nargs="+", type=Path, required=True)
    p.add_argument("--after", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-regression-percent", type=float, default=5)
    args = p.parse_args()
    try:
        result = compare([json.loads(x.read_text(encoding="utf-8-sig")) for x in args.before],
                         [json.loads(x.read_text(encoding="utf-8-sig")) for x in args.after], args.max_regression_percent)
    except (ValueError, KeyError, OSError) as error:
        result = {"passed": False, "failure": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
