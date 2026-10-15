#!/usr/bin/env python3
"""Bounded same-server greedy controls for the three-chat cancellation regression.

Checks text prefixes only. Truncated control output does not establish full-answer
or factual accuracy; the actual concurrent answers remain in the input report.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
path = Path(__file__).with_name("unified-qwen-concurrency-bench.py")
spec = importlib.util.spec_from_file_location("control", path)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def prefix_control_gate(actual, serial, actual_eos, serial_eos):
    a, b = actual.get("text", ""), serial.get("text", "")
    chars = min(len(a), len(b))
    matched = bool(chars) and a[:chars] == b[:chars]
    complete = bool(actual_eos and serial_eos and a == b)
    # A shorter EOS answer cannot be certified by matching the start of a
    # longer control. The bounded control must reach the identical EOS too.
    need_complete = actual_eos and (serial_eos or len(b) >= len(a))
    return dict(compared_characters=chars, complete_answer_text_equal=complete,
        actual_natural_eos=actual_eos, serial_natural_eos=serial_eos,
        passed=matched and (complete or not need_complete) and serial.get("completed") is True
            and not serial.get("error") and not serial.get("cleanup_error"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrent-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--server-log", required=True)
    args = parser.parse_args()
    if not 32 <= args.max_tokens <= 2048: parser.error("Invalid bounded output")
    lexical = Path(os.path.abspath(args.output))
    if ".." in Path(args.output).parts or not any(lexical.is_relative_to(ROOT / n) for n in ("artifacts", "docs/validation")):
        parser.error("Evidence must be ignored")
    output = lexical.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(args.concurrent_report).read_bytes()
    concurrent = json.loads(raw.decode("utf-8-sig"))
    rows = concurrent["concurrent"]
    if len(rows) != 3: raise ValueError("Expected three current chats")
    client = control.Client(args.url, output, 600)
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(), args=vars(args), passed=False,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        concurrent_report_sha256=hashlib.sha256(raw).hexdigest(),
        scope="Same-server bounded greedy text prefix controls. Does not establish full-logit parity, complete-answer parity, fact accuracy or a performance comparison with other engines.")
    try:
        result["model"] = client.json("GET", "/api/models")
        if result["model"].get("unifiedMemory") is not True: raise ValueError("No unified session")
        result["serial"] = [client.stream("serial-" + str(i), row["prompt"], args.max_tokens) for i, row in enumerate(rows)]
        checker_spec = importlib.util.spec_from_file_location("timings", control.FUSED_CHECKER)
        checker = importlib.util.module_from_spec(checker_spec); checker_spec.loader.exec_module(checker)
        for attempt in range(10):
            logs, pin = checker.read_input(args.server_log, checker.MAX_LOG_BYTES)
            try:
                records = [json.loads(line) for line in logs.decode("utf-8-sig").splitlines() if line.strip()]
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                if attempt == 9: raise
                time.sleep(.1)
        (output / "server-log.jsonl").write_bytes(logs)
        result["server_log"] = pin
        result["generation_timings"] = checker.generation_timing_evidence(result, records)
        if not result["generation_timings"]["all_current_requests_measured"]:
            raise ValueError("Serial generation timings are missing or invalid")
        def eos(timings, request):
            matching = [r for r in timings.get("rows", []) if r.get("http_request_id") == request.get("request_id")]
            return bool(matching) and all(r.get("finish_reason") == "eos" and r.get("engine_completion_available") is True for r in matching)
        result["checks"] = []
        for actual, serial in zip(rows, result["serial"]):
            result["checks"].append(dict(prompt=actual["prompt"], **prefix_control_gate(actual, serial,
                eos(concurrent.get("generation_timings", {}), actual), eos(result["generation_timings"], serial))))
        result["passed"] = all(c["passed"] for c in result["checks"])
    except Exception as error: result["error"] = f"{type(error).__name__}: {error}"
    control.save(output / "report.json", result)
    print(json.dumps({k: result.get(k) for k in ("passed", "checks", "error")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
