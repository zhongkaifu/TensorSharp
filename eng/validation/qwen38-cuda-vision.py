#!/usr/bin/env python3
"""Reproduce the Qwen3.8 CUDA TP image turn and check narrow banner OCR.

Run separately with --mode interactive (original thinking turn, then retained
image follow-up) and --mode ocr (greedy JSONL request in a fresh process).
Evidence belongs in ignored artifacts/ or docs/validation/. GPU memory is
sampled board usage, including other processes; it is not an allocation peak.
These checks do not establish broad visual quality or comparative throughput.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
TURN = re.compile(r"\[turn complete: tokens=(\d+) prefillMs=([\d.]+) decodeMs=([\d.]+) "
                  r"tps=([\d.]+) ttftMs=(\d+) reason=(\S+) kvPlan=([^\]\s]+)[^\]]*\]")
ERROR = re.compile(r"\[error\]|\bfail:\s|Step failed for sequence|Interactive turn failed|"
                   r"Cannot grow TensorSharp CUDA matmul scratch|CUDA error:|out of memory", re.I)
LOG_LINE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+(?:trce|dbug|info|warn|fail|crit):.*$", re.M)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command, timeout=15):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "error": str(error)}


def git_identity(directory):
    revision = command_output(["git", "-C", str(directory), "rev-parse", "HEAD"])
    status = command_output(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"])
    return {"path": str(directory), "revision": revision.get("stdout"), "status": status,
            "clean": revision.get("returncode") == status.get("returncode") == 0 and not status["stdout"]}


def file_identity(path, hash_content=True):
    stat = path.stat()
    result = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_content:
        result["sha256"] = sha256(path)
    return result


def interactive_result(log, expected_text):
    clean = ANSI.sub("", log)
    turns = []
    boundary = 0
    for match in TURN.finditer(clean):
        preceding = clean[boundary:match.start()]
        boundary = match.end()
        # Only inspect the answer, never startup banners, paths or reasoning.
        answer = preceding.rsplit("Assistant: ", 1)[-1] if "Assistant: " in preceding else ""
        if "[thinking]" in answer:
            answer = answer.rsplit("[answer]", 1)[-1] if "[answer]" in answer else ""
        answer = LOG_LINE.sub("", answer).strip()
        tokens, prefill, decode, tps, ttft, reason, kv_plan = match.groups()
        checks = {"nonempty_answer": bool(answer), "expected_text_in_answer": expected_text.casefold() in answer.casefold(),
                  "generated_tokens": int(tokens) > 0, "completed": reason in ("stop", "eos", "stop_sequence")}
        turns.append({"answer": answer, "tokens": int(tokens), "prefill_ms": float(prefill),
                      "decode_ms": float(decode), "decode_tokens_per_second": float(tps), "ttft_ms": int(ttft),
                      "finish_reason": reason, "kv_plan": kv_plan, "checks": checks, "passed": all(checks.values())})
    errors = ERROR.findall(clean)
    return {"passed": len(turns) == 2 and all(turn["passed"] for turn in turns) and not errors,
            "expected_turns": 2, "turns": turns, "error_markers": errors}


def ocr_result(path, expected_text, budget):
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        return {"passed": False, "error": str(error)}
    if len(records) != 1 or not isinstance(records[0], dict) or records[0].get("id") != "banner-ocr":
        return {"passed": False, "error": "Expected exactly one banner-ocr result", "records": records}
    row = records[0]
    raw = row.get("output", "")
    if not isinstance(raw, str):
        return {"passed": False, "error": "OCR output must be text", "result": row}
    answer = raw.split("</think>")[-1].strip()
    tokens = row.get("tokens_generated", 0)
    checks = {"no_error": "error" not in row, "nonempty_answer": bool(answer),
              "expected_text_only": answer.strip(" \t\r\n`*\"'.").casefold() == expected_text.casefold(),
              "within_completion_budget": type(tokens) is int and 0 < tokens < budget,
              "no_unclosed_thinking_or_control_tokens": "<think>" not in answer and not re.search(r"<\|[^>]*\|>", answer)}
    return {"passed": all(checks.values()), "checks": checks, "answer": answer, "result": row,
            "limitation": "CLI JSONL omits finish_reason; token count below budget and closed output are checked."}


class GpuSampler:
    def __init__(self, path, interval):
        self.path, self.interval = path, interval
        self.stop = threading.Event()
        self.samples, self.errors = [], []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        result = command_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], timeout=10)
        if result.get("returncode") != 0:
            self.errors.append(result)
            return
        try:
            devices = []
            for row in csv.reader(result["stdout"].splitlines()):
                index, uuid, name, used, total, utilization = (value.strip() for value in row)
                devices.append({"index": int(index), "uuid": uuid, "name": name,
                                "memory_used_mib": float(used), "memory_total_mib": float(total),
                                "utilization_percent": float(utilization) if utilization.isdigit() else None})
            record = {"utc": datetime.now(timezone.utc).isoformat(), "devices": devices}
            self.samples.append(record)
            with self.path.open("a") as output:
                output.write(json.dumps(record) + "\n")
        except (ValueError, OSError) as error:
            self.errors.append({"error": str(error)})

    def run(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def summary(self):
        peaks = {}
        for sample in self.samples:
            for device in sample["devices"]:
                key = device["uuid"]
                if key not in peaks or device["memory_used_mib"] > peaks[key]["memory_used_mib"]:
                    peaks[key] = device
        return {"sample_count": len(self.samples), "interval_seconds": self.interval,
                "baseline": self.samples[0] if self.samples else None, "peak_board_usage": list(peaks.values()),
                "errors": self.errors, "limitation": "Sampled board usage includes other processes and may miss brief peaks."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "model", "mmproj", "image", "report-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("interactive", "ocr"), default="interactive")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--ocr-max-tokens", type=int, default=256)
    parser.add_argument("--expected-text", default="TensorSharp")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--sample-interval", type=float, default=1)
    parser.add_argument("--ggml-dir", type=Path, default=ROOT / "ExternalProjects/ggml")
    args = parser.parse_args()
    if min(args.tp, args.max_tokens, args.ocr_max_tokens, args.timeout, args.sample_interval) <= 0:
        parser.error("TP, budgets, timeout and sample interval must be positive")
    if not args.expected_text.strip():
        parser.error("--expected-text cannot be empty")
    for name in ("cli", "model", "mmproj", "image", "ggml_dir", "report_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if not any(args.report_dir.is_relative_to(ROOT / directory) for directory in ("artifacts", "docs/validation")):
        parser.error("--report-dir must be under this checkout's ignored artifacts/ or docs/validation/")
    for path in (args.cli, args.model, args.mmproj, args.image):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / "report.json"
    if report_path.exists():
        parser.error("Use a fresh --report-dir to preserve previous evidence")
    command = [str(args.cli), "--backend", "ggml_cuda", "--model", str(args.model), "--mmproj", str(args.mmproj),
               "--tp", str(args.tp), "--log-dir", str(args.report_dir / "logs")]
    if args.mode == "interactive":
        command += ["--interactive", "--think", "--max-tokens", str(args.max_tokens)]
        standard_input = f"/image {args.image}\nwhat is this?\nWhat is the largest text written in the image I uploaded? Reply only with that text.\n/exit\n"
    else:
        requests = args.report_dir / "requests.jsonl"
        requests.write_text(json.dumps({"id": "banner-ocr", "messages": [{"role": "user", "content": "Read the largest text in this image. Reply only with that text."}],
                                        "images": [str(args.image)], "temperature": 0, "max_tokens": args.ocr_max_tokens}) + "\n")
        command += ["--input-jsonl", str(requests), "--output", str(args.report_dir / "answers.jsonl")]
        standard_input = ""
    (args.report_dir / "stdin.txt").write_text(standard_input)
    shards = sorted(args.model.parent.glob(re.sub(r"-\d{5}-of-\d{5}\.gguf$", "-*-of-*.gguf", args.model.name)))
    report = {"mode": args.mode, "command": command, "cwd": str(args.cli.parent), "run_complete": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "source": git_identity(ROOT),
              "ggml": git_identity(args.ggml_dir), "cli": file_identity(args.cli), "image": file_identity(args.image),
              "projector": file_identity(args.mmproj), "model_shards": [file_identity(path, False) for path in shards],
              "harness_sha256": sha256(Path(__file__)), "environment": {key: value for key, value in os.environ.items()
                 if key.startswith(("TS_Q4E_", "TS_TP_", "TS_GGML_", "TS_SCHED_", "GGML_"))
                 or key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "TS_N_CPU_MOE", "TS_CPU_MOE")},
              "scope": "Original thinking banner turn, retained image follow-up, or greedy OCR in a separate invocation. No broad visual-quality or comparative performance claim.",
              "limitations": ["Model shards are identified by path, size and mtime, not content hash.", "Interactive sampling uses CLI defaults and is not deterministic.", "Wall time includes model loading; turn prefill/decode times are reported by the CLI."]}
    library = args.cli.parent / "libGgmlOps.so"
    report["native_library"] = file_identity(library.resolve()) if library.is_file() else None
    report["managed_binaries"] = [file_identity(path) for path in (
        args.cli.parent / "TensorSharp.Cli.dll", args.cli.parent / "TensorSharp.Models.dll",
        args.cli.parent / "TensorSharp.Backends.GGML.dll", args.cli.parent / "TensorSharp.Runtime.dll") if path.is_file()]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    sampler = GpuSampler(args.report_dir / "gpu.jsonl", args.sample_interval)
    sampler.sample()
    sampler.thread.start()
    started = time.monotonic()
    try:
        with (args.report_dir / "cli.log").open("w") as log:
            process = subprocess.Popen(command, cwd=args.cli.parent, stdin=subprocess.PIPE, stdout=log,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                process.communicate(standard_input, timeout=args.timeout)
                report["timed_out"] = False
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                report["timed_out"] = True
            report["returncode"] = process.returncode
    except OSError as error:
        report.update(returncode=None, launch_error=str(error), timed_out=False)
    finally:
        report["wall_seconds"] = time.monotonic() - started
        sampler.stop.set()
        sampler.thread.join()
        sampler.sample()
    log_text = (args.report_dir / "cli.log").read_text(errors="replace")
    report["validation"] = interactive_result(log_text, args.expected_text) if args.mode == "interactive" else ocr_result(
        args.report_dir / "answers.jsonl", args.expected_text, args.ocr_max_tokens)
    report["error_markers"] = ERROR.findall(log_text)
    report["gpu"] = sampler.summary()
    report["ggml_after"] = git_identity(args.ggml_dir)
    report["run_complete"] = True
    report["passed"] = (report["returncode"] == 0 and not report["timed_out"] and not report["error_markers"]
                        and report["validation"]["passed"] and report["ggml"]["clean"] and report["ggml_after"]["clean"]
                        and report["ggml"]["revision"] == report["ggml_after"]["revision"])
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(report_path), "wall_seconds": report["wall_seconds"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
