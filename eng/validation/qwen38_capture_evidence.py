"""Completion and identity checks shared by diagnostic vocabulary comparators."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def validate(index_path, report_path):
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    require(report.get("passed") is True and report.get("run_complete") is True and not report.get("error"),
            "Model report did not complete successfully")
    cleanup = report.get("cleanup", {})
    require(all(cleanup.get(name) is True for name in
                ("model_disposed", "cache_cleared", "reuse_released", "native_shutdown"))
            and not cleanup.get("errors") and cleanup.get("retained_model_owner") is False
            and cleanup.get("retained_scope_owner") is False,
            "Model report does not prove complete physical cleanup")
    if report.get("device_budget_bytes") is not None:
        require(cleanup.get("scope_detached") is True, "Shared budget scope did not detach")
    geometry = report.get("model_geometry", {})
    require(isinstance(geometry, dict) and isinstance(geometry.get("architecture"), str)
            and bool(geometry["architecture"]) and all(type(geometry.get(key)) is int and geometry[key] > 0
                for key in ("hidden_size", "layers", "heads", "kv_heads", "vocabulary", "context_limit"))
            and geometry.get("kv_dtype") == "f16", "Missing or invalid model geometry")
    vocabulary = geometry["vocabulary"]
    require(is_hash(report.get("native_sha256")) and report.get("managed_assemblies_sha256")
            and all(is_hash(value) for value in report["managed_assemblies_sha256"].values()),
            "Missing actual native/managed identities")
    if report.get("synthetic") is True:
        require(is_hash(report.get("model_sha256")) and type(report.get("model_bytes")) is int
                and report["model_bytes"] > 0, "Synthetic checkpoint hash/size is absent")
        checkpoint = [(report["model_sha256"].lower(), report["model_bytes"])]
    else:
        require(report.get("synthetic") is False, "Checkpoint kind is absent")
        identity = report.get("checkpoint_identity") or {}
        require(identity.get("current_metadata_checked") is True and is_hash(identity.get("manifest_sha256")),
                "A verified checkpoint identity manifest is required for real weights")
        require(identity.get("model_path") and report.get("model_path")
                and Path(identity["model_path"]).resolve() == Path(report["model_path"]).resolve(),
                "Checkpoint identity is not bound to the loaded model path")
        files = identity.get("files", [])
        require(files and all(is_hash(item.get("sha256")) and type(item.get("bytes")) is int
                              and item["bytes"] > 0 for item in files), "Incomplete checkpoint shard identities")
        checkpoint = [(item["sha256"].lower(), item["bytes"]) for item in files]
    captures = report.get("logit_captures") or {}
    require(index.get("format") == captures.get("format") == "f32le", "Expected f32le captures")
    require(Path(captures.get("index_path") or "").resolve() == index_path.resolve()
            and Path(captures.get("data_path") or "").resolve() == Path(index["data_path"]).resolve(),
            "Capture paths are not bound to the final model report")
    rows = index.get("rows", [])
    require(rows and rows == captures.get("rows"), "Final model report and capture rows differ")
    options = report.get("requested_options", {})
    iterations, warmups = int(options.get("iterations", 5)), int(options.get("warmup", 2))
    require(iterations > 0 and warmups >= 0, "Invalid requested iteration counts")
    runs = report.get("runs", [])
    expected_iterations = list(range(-warmups, iterations))
    require(all(type(run.get("iteration")) is int for run in runs)
            and [run["iteration"] for run in runs] == expected_iterations, "Model iterations are incomplete")
    prompt = report.get("prompt_tokens")
    require(isinstance(prompt, list) and prompt and all(type(token) is int and 0 <= token < vocabulary for token in prompt),
            "Invalid model prompt history")
    expected_rows = []
    offset = 0
    position = 0
    for run in runs:
        iteration = run["iteration"]
        require(run.get("warmup") is (iteration < 0) and run.get("prefill_tokens") == len(prompt),
                "Model run metadata differs from the prompt")
        count = run.get("decode_tokens")
        require(type(count) is int and count >= 0, "Invalid completed decode count")
        if report.get("decode_mode") == "teacher-forced":
            consumed = report.get("forced_tokens")
            require(isinstance(consumed, list) and len(consumed) == count == int(options.get("decode-tokens", 64)),
                    "Incomplete forced-token execution")
        else:
            require(report.get("decode_mode") == "greedy", "Unknown generation mode")
            generated = run.get("generated_tokens")
            require(isinstance(generated, list) and len(generated) == count + 1,
                    "Greedy generated/consumed token counts differ")
            require(len(generated) <= int(options.get("decode-tokens", 64))
                    and run.get("finish_reason") in ("eos", "length")
                    and (run["finish_reason"] != "length" or len(generated) == int(options.get("decode-tokens", 64))),
                    "Greedy run terminated before its requested boundary")
            consumed = generated[:count]
        require(all(type(token) is int and 0 <= token < vocabulary for token in consumed),
                "Invalid consumed token history")
        for step in range(count + 1):
            require(position < len(rows), "Capture is missing a completed model row")
            row = rows[position]
            require(type(row.get("iteration")) is int and row["iteration"] == iteration
                    and row.get("warmup") is (iteration < 0)
                    and row.get("stage") == ("prefill" if step == 0 else "decode")
                    and row.get("input_tokens") == prompt + consumed[:step]
                    and type(row.get("elements")) is int and row["elements"] == vocabulary
                    and type(row.get("byte_offset")) is int and row["byte_offset"] == offset
                    and is_hash(row.get("sha256")), "Capture histories, dimensions or row layout are incomplete")
            offset += vocabulary * 4
            position += 1
            expected_rows.append(row)
        require(rows[position - 1]["sha256"] == run.get("final_logit_sha256"),
                "Last capture does not match the completed model prediction")
    require(position == len(rows) and Path(index["data_path"]).stat().st_size == offset,
            "Capture contains missing, overlapping or extra data")
    # Validate every completed iteration, including warmups that a particular
    # comparison may not select. A metadata-only index cannot prove a completed
    # capture when the backing bytes are corrupt or contain nonfinite values.
    with Path(index["data_path"]).open("rb") as stream:
        for row in rows:
            data = stream.read(vocabulary * 4)
            require(hashlib.sha256(data).hexdigest() == row["sha256"], "Capture row bytes/hash differ")
            require(all(math.isfinite(value) for value, in struct.iter_unpack("<f", data)),
                    "Capture contains nonfinite values")
    return {"checkpoint": checkpoint, "geometry": geometry, "decode_mode": report["decode_mode"],
            "prompt_tokens": prompt, "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "native_sha256": report["native_sha256"],
            "managed_assemblies_sha256": report["managed_assemblies_sha256"],
            "validated_rows": len(expected_rows)}
