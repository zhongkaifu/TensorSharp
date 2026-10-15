#!/usr/bin/env python3
"""Summarize an owned-server acceptance run without treating overlap as batching.

Input is the directory produced by unified-qwen-server-acceptance.ps1. Outputs
are generated evidence and belong under artifacts/ or docs/validation/.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def whole_number(value):
    return type(value) is int and value >= 0


def measured_sum(rows, key):
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(whole_number(value) for value in values) else None


def rate(tokens, seconds):
    return tokens / seconds if tokens is not None and isinstance(seconds, (int, float)) \
        and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds > 0 else None


def timing_summary(generations, source_errors):
    """Sum root-session generations; never combine one round with HTTP totals."""
    errors = list(source_errors)
    identities = [row.get("internal_request_id") for row in generations]
    if any(not isinstance(value, str) or not value for value in identities) \
            or len(set(identities)) != len(identities):
        errors.append("Missing or duplicate internal generation identity")
    prompt = measured_sum(generations, "prompt_tokens")
    reuse = measured_sum(generations, "kv_reused_tokens")
    pipeline = measured_sum(generations, "recorded_pipeline_tokens")
    engine = measured_sum(generations, "engine_output_tokens")
    pp_ns = measured_sum(generations, "attributed_prefill_ns")
    decode_ns = measured_sum(generations, "attributed_decode_ns")
    valid_prompt = bool(generations) and all(
        whole_number(row.get("prompt_tokens")) and whole_number(row.get("kv_reused_tokens"))
        and row["kv_reused_tokens"] <= row["prompt_tokens"] for row in generations)
    if generations and not valid_prompt:
        errors.append("Invalid prompt/reuse count in a generation")
    complete = bool(generations) and not errors and valid_prompt \
        and all(row.get("engine_completion_available") is True for row in generations) \
        and all(value is not None for value in (pipeline, engine, pp_ns, decode_ns))
    # Missing/invalid completion in any round invalidates full-turn timing,
    # even if the last round and the terminal's fallback rate look plausible.
    if not complete:
        pp_ns = decode_ns = None
    if errors:
        pipeline = engine = None
    reasons = [row.get("finish_reason") for row in generations]
    return {
        "generation_count": len(generations),
        "internal_request_ids": identities,
        "generation_finish_reasons": reasons,
        "generation_timings": generations,
        "timing_complete": complete, "timing_errors": errors,
        "recorded_pipeline_output_tokens": pipeline,
        "engine_output_tokens": engine,
        "finish_reason": reasons[-1] if reasons else None,
        "uncached_prompt_tokens": prompt - reuse if valid_prompt and not errors else None,
        "reused_prompt_tokens": reuse if valid_prompt and not errors else None,
        "attributed_prefill_seconds": pp_ns / 1e9 if pp_ns is not None else None,
        "attributed_decode_seconds": decode_ns / 1e9 if decode_ns is not None else None,
    }


def sampled_peak(samples, key):
    values = [row.get(key) for row in samples]
    measured = [value for value in values if whole_number(value)]
    return max(measured, default=None), len(measured)


def summarize(directory, case_directory="concurrency"):
    directory = Path(directory)
    if case_directory not in ("concurrency", "three-chat", "serial-three"):
        raise ValueError("Unknown HTTP evidence case directory")
    source = directory / case_directory / "report.json"
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    timing = {}
    timing_evidence = data.get("generation_timings", {})
    for generation in timing_evidence.get("rows", []):
        timing.setdefault(generation["http_request_id"], []).append(generation)
    rows = []
    for mode in ("concurrent", "serial"):
        for request in data.get(mode, []):
            t = timing_summary(timing.get(request.get("request_id"), []), timing_evidence.get("errors", []))
            phase = request.get("phase_observations", {})
            terminal = request.get("terminal") or {}
            wall = phase.get("first_content_to_terminal_seconds")
            rows.append({
                "mode": mode, "name": request.get("name"),
                "prompt": request.get("prompt", request.get("request", {}).get("messages", [])),
                "ttft_seconds": request.get("ttft_seconds", phase.get("dispatch_to_first_content_seconds")),
                "elapsed_seconds": request.get("elapsed_seconds"),
                **t,
                "terminal_reported_tokens": terminal.get("tokenCount"),
                "terminal_reported_tokens_per_second": terminal.get("tokPerSec"),
                "terminal_rate_basis": "server-reported; may fall back to wall time when decode timing is unavailable",
                "truncated": terminal.get("truncated"),
                "attributed_prefill_tokens_per_second": rate(t["uncached_prompt_tokens"], t["attributed_prefill_seconds"]),
                "pipeline_tokens_per_first_content_to_terminal_second": rate(t["recorded_pipeline_output_tokens"], wall),
                "attributed_decode_tokens_per_second": rate(t["recorded_pipeline_output_tokens"], t["attributed_decode_seconds"]),
                "text": request.get("text"), "transport_error": terminal.get("error"),
            })
    samples = [json.loads(line) for line in (directory / "memory-samples.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    gpu_mib = []
    for sample in samples:
        for line in (sample.get("board_gpu") or "").splitlines():
            fields = [x.strip() for x in line.split(",")]
            if len(fields) == 4:
                try:
                    value = float(fields[1])
                    if math.isfinite(value) and value >= 0:
                        gpu_mib.append(value)
                except ValueError:
                    pass
    rss, rss_samples = sampled_peak(samples, "rss")
    private, private_samples = sampled_peak(samples, "private")
    native = data.get("native") if case_directory == "three-chat" else None
    three_chat_joint = bool(native and native.get("passed") is True
                            and native.get("n3_steps", 0) > 0
                            and native.get("n2_sibling_steps", 0) > 0)
    joint_steps = data.get("fused_joint_steps")
    if joint_steps is None and three_chat_joint:
        joint_steps = native["n3_steps"] + native["n2_sibling_steps"]
    return {
        "schema_version": 2,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "acceptance_passed": data.get("passed"),
        "actual_joint_engagement_verified": data.get("actual_joint_engagement_verified", three_chat_joint),
        "full_interval_continuity_verified": data.get("full_interval_continuity_verified", False),
        "joint_steps": joint_steps,
        "three_chat_native_evidence": native,
        "continuity_gaps": data.get("fused_continuity_gaps"),
        "greedy_text_parity": data.get("greedy_text_parity"),
        "startup_warmup": data.get("model", {}).get("startupWarmup"),
        "requests": rows,
        "process_sampled_peak_rss_bytes": rss,
        "process_sampled_peak_private_bytes": private,
        "whole_board_sampled_peak_used_mib": max(gpu_mib, default=None),
        "resource_sample_counts": {"total": len(samples), "rss_measured": rss_samples,
            "private_measured": private_samples, "board_measurements": len(gpu_mib),
            "samples_with_error": sum(bool(row.get("error")) for row in samples)},
        "scopes": [
            "TTFT is dispatch to first nonempty content frame; it includes queue and preparation.",
            "Prefill sums uncached tokens and attributed model-forward durations across all recorded root-session generations of each HTTP request; repeated context is not counted as newly prefetched work.",
            "Pipeline output counts include reasoning/tool tokens and the first generated token; they are not visible-answer token counts. Pipeline wall throughput uses first-content to terminal time and is not isolated kernel throughput.",
            "Attributed decode rates use summed recorded pipeline tokens divided by summed forward durations, never the terminal's potentially wall-time fallback rate. Any missing generation completion leaves full-turn attributed timings unavailable.",
            "Attributed decode time apportions shared batch time between requests and is not wall-clock per-request throughput. Per-generation data is retained; child sessions were excluded by the source timing verifier.",
            "Unknown/error-only resource samples remain null rather than zero; peaks use only valid observations and expose sample counts.",
            "Memory peaks cover the whole run, including load, warmup and serial controls; GPU usage is whole-board, not process-owned physical attribution.",
            "Native joint logical graphs may contain multiple CPU/GPU segments. Continuity does not establish output quality or complete answers.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-directory", choices=("concurrency", "three-chat", "serial-three"), default="concurrency")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.directory, args.case_directory), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
