#!/usr/bin/env python3
"""Sequential Qwen Metal performance evidence; stdlib only, no builds/downloads.

Default is a dry run. Pass --execute to launch owned processes. CLI run 1 is
process-first, later runs are graph-warm; OS/file/shader caches are uncontrolled.
CLI prefill is model timing, never HTTP TTFT. HTTP TTFT is first nonempty content
delta; post-first-delta throughput is a transport estimate, never a kernel rate.
Generated evidence must stay in ignored artifacts/ or docs/validation/.
Engine options default to None: use the engine defaults or the chosen variant's
--config file. Explicit --env and --variant-env settings are recorded.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import http.client as http_client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = frozenset("""MAX_CONTEXT KV_CACHE_DTYPE TS_PREFILL_CHUNK TS_CPU_MOE_THREADS
TS_GGML_CPU_THREADS TS_N_CPU_MOE TS_CPU_MOE TS_SPEC TS_SPEC_DRAFT
TS_HOST_MOE_DEVICE_MIN_BATCH
TS_GGML_NODE_PROFILE TS_GGML_PHASE_TIMING TS_GGML_LOG_DEBUG TS_METAL_GRAPH_OPTIMIZE
GGML_METAL_FUSION_DISABLE GGML_METAL_GRAPH_DEBUG GGML_METAL_DEBUG
TS_PREFIX_CACHE_MODE TS_PREFIX_CACHE_ISOLATION TS_PREFIX_CHECKPOINTS_MAX
TS_PREFILL_WARMUP TS_PREFILL_WARMUP_LEN TS_DUMP_LOGITS""".split())
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Infinity|-Infinity"
RUN = re.compile(r"benchmark run (\d+)/(\d+): (.*)")
PAIR = re.compile(r"(\w+)=([^\s]+)")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(argv, cwd):
    run = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
    return {"exit_code": run.returncode, "output": run.stdout.strip()}


def parse_runs(log, expected, decode_tokens=None):
    rows = []
    for match in RUN.finditer(log):
        row = {"run": int(match[1]), "runs": int(match[2]),
               "cache_label": "process-first; OS caches uncontrolled" if match[1] == "1" else "graph-warm; new seeded random token IDs"}
        for key, value in PAIR.findall(match[3]):
            if re.fullmatch(NUMBER, value):
                number = float(value)
                if not math.isfinite(number):
                    # CLI intentionally prints NaN rates for prefill-only runs.
                    # Their elapsed times must still be finite; no decode rate
                    # is accepted when zero decode tokens were requested.
                    if decode_tokens == 0 and math.isnan(number) and key in ("decodeTps", "decodeModelTps", "msPerTok"):
                        row[key] = None
                        continue
                    raise ValueError(f"benchmark {key} is not finite")
                row[key] = number
            else:
                row[key] = value
        for required in ("prefillMs", "prefillTps", "decodeMs", "decodeTps", "decodeModelMs", "decodeMode"):
            if required not in row:
                raise ValueError(f"benchmark row lacks {required}")
        for key in ("prefillMs", "prefillTps", "decodeMs", "decodeTps", "decodeModelMs", "decodeModelTps", "msPerTok", "greedySampleMs"):
            if key not in row or (row[key] is None and decode_tokens == 0 and key in ("decodeTps", "decodeModelTps", "msPerTok")):
                continue
            value = row[key]
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"benchmark {key} is not a nonnegative number")
            if value == 0 and (key in ("prefillMs", "prefillTps")
                    or (decode_tokens != 0 and key in ("decodeMs", "decodeTps", "decodeModelMs", "decodeModelTps", "msPerTok"))):
                raise ValueError(f"benchmark {key} must be positive")
        rows.append(row)
    if [r["run"] for r in rows] != list(range(1, expected + 1)) or any(r["runs"] != expected for r in rows):
        raise ValueError(f"expected exactly {expected} sequential benchmark rows, found {len(rows)}")
    return rows


def read_logits(path):
    raw = path.read_bytes()
    meta = json.loads(Path(str(path) + ".json").read_text())
    if (len(raw) < 8 or len(raw) % 4 or meta.get("format") != "f32le"
            or meta.get("semantics") != "first-generated-token"
            or meta.get("vector_sha256") != hashlib.sha256(raw).hexdigest()
            or meta.get("count") != len(raw) // 4):
        raise ValueError("invalid/stale first-token vector or sidecar")
    values = array.array("f", raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite first-token logits")
    return values, meta


def compare_logits(first, second):
    a, ma = read_logits(first)
    b, mb = read_logits(second)
    for key in ("count", "prompt_tokens", "prompt_fingerprint", "media_spans", "prefix_reused_tokens"):
        if ma.get(key) != mb.get(key) or (key == "prompt_fingerprint" and not ma.get(key)):
            raise ValueError(f"logit row identity differs or is missing: {key}")
    delta = [float(y) - float(x) for x, y in zip(a, b)]
    top_a = sorted(range(len(a)), key=a.__getitem__, reverse=True)[:5]
    top_b = sorted(range(len(b)), key=b.__getitem__, reverse=True)[:5]
    return {"count": len(a), "bit_identical": first.read_bytes() == second.read_bytes(),
            "max_abs": max(map(abs, delta)),
            "relative_l2": math.sqrt(math.fsum(d*d for d in delta) / max(1e-300, math.fsum(float(x)*x for x in a))),
            "top1_equal": top_a[0] == top_b[0], "baseline_top5": top_a, "candidate_top5": top_b,
            "baseline_margin": float(a[top_a[0]]) - a[top_a[1]],
            "candidate_margin": float(b[top_b[0]]) - b[top_b[1]],
            "scope": "Full first-generated-token vector for identical prompt; not full decode/model-quality equivalence"}


def pairs(items):
    result = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value or key in result:
            raise ValueError("use unique NAME=VALUE entries")
        result[key] = value
    return result


def environment(args, label, dump):
    # Explicit engine configuration: an inherited profiling/fusion/speculation
    # knob must not silently contaminate an otherwise identical A/B.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TS_", "TENSORSHARP_", "GGML_"))
           and key not in ("MAX_CONTEXT", "KV_CACHE_DTYPE")}
    env.update(TS_DUMP_LOGITS=str(dump), DOTNET_CLI_TELEMETRY_OPTOUT="1")
    for setting, value in (("MAX_CONTEXT", args.context), ("TS_PREFILL_CHUNK", args.chunk),
                           ("TS_CPU_MOE_THREADS", args.threads), ("TS_GGML_CPU_THREADS", args.threads)):
        if value is not None:
            env[setting] = str(value)
    settings = dict(args.environment, **getattr(args, "variant_environment", {}).get(label, {}))
    for key, value in settings.items():
        if key not in ENV_KEYS or key == "TS_DUMP_LOGITS":
            raise ValueError(f"--env is restricted to benchmark settings (not {key})")
        env[key] = value
    return env


def variant_config(args, label):
    """Return both the literal argv and the identity of this variant's config."""
    path = getattr(args, "variant_configs", {}).get(label)
    if path is None:
        return [], None
    return ["--config", str(path)], {"file": str(path), "sha256": sha(path)}


def model_options(args):
    return ["--n-cpu-moe", str(args.cpu_moe)] if args.cpu_moe is not None else []


def identity(root, dll):
    required = (dll, dll.parent / "libGgmlOps.dylib")
    if any(not path.is_file() for path in required):
        raise ValueError(f"missing DLL/native library for {root}")
    # Models/Runtime/Chat can change independently of the entry point. Identify
    # all TensorSharp managed dependencies actually shipped beside this binary.
    files = set(required).union(dll.parent.glob("TensorSharp*.dll"))
    binaries = {str(path.relative_to(root)): sha(path) for path in sorted(files) if path.is_file()}
    ggml = root / "ExternalProjects/ggml"
    metadata = {"source": capture(["git", "rev-parse", "HEAD"], root),
                "source_changes": capture(["git", "status", "--porcelain"], root),
                "ggml": capture(["git", "rev-parse", "HEAD"], ggml),
                "ggml_changes": capture(["git", "status", "--porcelain"], ggml)}
    for name, result in metadata.items():
        if result["exit_code"] != 0 or (name in ("source", "ggml") and not re.fullmatch(r"[0-9a-fA-F]{40}", result["output"])):
            raise ValueError(f"Git provenance unavailable: {name}")
    return dict(metadata, binaries_sha256=binaries)


def stop_owned(process, child_pid=None):
    # Only the process group created by this harness; never a discovered PID.
    if process.poll() is None:
        # HTTP's tiny launcher records its own PID before exec(dotnet). Signal
        # that exact child first, letting /usr/bin/time reap it and print RSS.
        if child_pid is not None:
            try:
                if os.getpgid(child_pid) == process.pid:
                    os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def cli(args, report):
    references = {}
    for ordinal, label in enumerate(args.order):
        root = args.variants[label]
        dll = root / "TensorSharp.Cli/bin/TensorSharp.Cli.dll"
        for prefill, decode in args.cases:
            directory = args.output / f"{ordinal:02}-{label}-pp{prefill}-tg{decode}"
            dump = directory / "logits.f32"
            env = environment(args, label, dump)
            config_argv, config_identity = variant_config(args, label)
            argv = [args.dotnet, str(dll), "--model", str(args.model), "--backend", "ggml_metal"] + model_options(args) + config_argv + [
                    "--benchmark", "--bench-prefill", str(prefill),
                    "--bench-decode", str(decode), "--bench-runs", str(args.runs),
                    "--bench-random-tokens" if args.token_stream == "random" else "--bench-fixed-tokens",
                    "--bench-chunked"]
            row = {"label": label, "ordinal": ordinal, "prefill_tokens": prefill, "decode_tokens": decode,
                   "argv": argv, "config": config_identity,
                   "environment": {k: env.get(k) for k in sorted(ENV_KEYS)}, "evidence": str(directory)}
            report["processes"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if not args.execute:
                continue
            directory.mkdir()
            row["identity"] = identity(root, dll)
            started = time.perf_counter()
            with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                process = subprocess.Popen(["/usr/bin/time", "-l"] + argv, cwd=root, env=env,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    row["exit_code"] = process.wait(timeout=args.timeout)
                finally:
                    stop_owned(process)
            row["process_wall_ms"] = (time.perf_counter() - started) * 1000
            log = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
            row["placement_lines"] = [line for line in log.splitlines() if "[moe-offload]" in line or "Metal device" in line]
            peak = re.search(r"(\d+)\s+maximum resident set size", log)
            row["peak_process_rss_bytes"] = int(peak[1]) if peak else None
            row["errors"] = []
            if row["exit_code"] != 0:
                row["errors"].append("process failed")
            try:
                row["runs"] = parse_runs(log, args.runs, decode)
                if args.token_stream == "fixed":
                    for run in row["runs"]:
                        run["cache_label"] = ("process-first after standard warmup; OS caches uncontrolled"
                            if run["run"] == 1 else "graph-warm; identical fixed token IDs")
                _, meta = read_logits(dump)
                row["first_token_sidecar"] = meta
                correctness = re.search(r"benchmark sampled tokens \(untimed correctness\): prefillTopToken=(\d+) decode=([\d,]*)", log)
                if not correctness:
                    raise ValueError("missing untimed greedy correctness chain")
                row["untimed_correctness"] = {"prefill_top_token": int(correctness[1]),
                    "decode_tokens": [int(token) for token in correctness[2].split(",") if token]}
                key = (prefill, decode)
                if key in references:
                    first_dump, first_chain = references[key]
                    row["logit_comparison"] = compare_logits(first_dump, dump)
                    row["greedy_chain_equal"] = first_chain == row["untimed_correctness"]
                else:
                    references[key] = dump, row["untimed_correctness"]
            except (OSError, ValueError, KeyError) as error:
                row["errors"].append(str(error))
            save(args.output / "report.json", report)
            if row["errors"]:
                raise ValueError(f"{label}: {row['errors']}")


def request_sse(port, model, messages, max_tokens, timeout):
    body = {"model": model, "messages": messages, "temperature": 0, "repeat_penalty": 1,
            "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
            "think": False, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
            "skills": [], "skills_discovery": False, "multi_agent": False}
    result = {"request": body, "answer": "", "reasoning": "", "ttft_ms": None, "content_delta_count": 0,
              "finish_reason": None, "usage": None, "done": False, "events": [], "errors": []}
    connection = http_client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    start = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200 or "text/event-stream" not in response.getheader("Content-Type", ""):
            raise ValueError(f"HTTP/SSE failure {response.status}: {response.read(1024)!r}")
        for line in response:
            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > timeout * 1000:
                raise TimeoutError("SSE deadline exceeded")
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                result["done"] = True
                break
            if not raw:
                continue
            event = json.loads(raw)
            result["events"].append({"elapsed_ms": elapsed, "data": event})
            if event.get("error"):
                result["errors"].append(str(event["error"]))
            if event.get("usage"):
                result["usage"] = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                if content and result["ttft_ms"] is None:
                    result["ttft_ms"] = elapsed
                if content:
                    result["content_delta_count"] += 1
                result["answer"] += content
                result["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
    finally:
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        connection.close()
    if not result["done"] or not result["finish_reason"] or result["ttft_ms"] is None or not result["usage"]:
        result["errors"].append("incomplete response, content timing or usage")
    if result["reasoning"]:
        result["errors"].append("unexpected reasoning output")
    result["post_first_delta_transport_tokens_per_second"] = None
    completion = (result["usage"] or {}).get("completion_tokens")
    details = (result["usage"] or {}).get("prompt_tokens_details") or {}
    result["cached_tokens"] = details.get("cached_tokens")
    prompt_tokens = (result["usage"] or {}).get("prompt_tokens")
    result["prefilled_tokens"] = prompt_tokens - result["cached_tokens"] if type(prompt_tokens) is int and type(result["cached_tokens"]) is int else None
    if result["content_delta_count"] > 1 and type(completion) is int and completion > 1:
        result["post_first_delta_transport_tokens_per_second"] = (completion - 1) * 1000 / max(.001, result["e2e_ms"] - result["ttft_ms"])
    return result


def json_values_equal(actual, expected):
    """Compare parsed JSON trees without Python's bool/int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_values_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            json_values_equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def http(args, report):
    references = {}
    first_dump = None
    prompts = [
        ("arithmetic", "Only output the integer result: 123 + 456.", "579"),
        ("chinese", "请用三句话说明自行车的刹车、齿轮和平衡分别有什么作用。", None),
        ("json", 'Output only this exact JSON object, without Markdown: {"answer":42,"ok":true}', {"answer": 42, "ok": True}),
    ]
    for ordinal, label in enumerate(args.order):
        root = args.variants[label]
        dll = root / "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll"
        directory = args.output / f"{ordinal:02}-{label}-http"
        env = environment(args, label, directory / "logits.f32")
        config_argv, config_identity = variant_config(args, label)
        argv = [args.dotnet, str(dll), "--model", str(args.model), "--backend", "ggml_metal"] + model_options(args) + config_argv + [
                "--host", "127.0.0.1", "--port", str(args.port), "--no-webui"]
        process_row = {"label": label, "ordinal": ordinal, "argv": argv, "config": config_identity,
                       "environment": {k: env.get(k) for k in sorted(ENV_KEYS)}, "requests": []}
        report["processes"].append(process_row)
        print(json.dumps(process_row, ensure_ascii=False), flush=True)
        if not args.execute:
            continue
        # Reserve-check loopback before launch. Never reuse an existing service.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        directory.mkdir()
        process_row["identity"] = identity(root, dll)
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            pid_path = directory / "server.pid"
            launcher = "import os,sys;open(sys.argv[1],'w').write(str(os.getpid()));os.execvp(sys.argv[2],sys.argv[2:])"
            process = subprocess.Popen(["/usr/bin/time", "-l", sys.executable, "-c", launcher, str(pid_path)] + argv, cwd=root, env=env,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                deadline = time.monotonic() + args.timeout
                while True:
                    if process.poll() is not None:
                        raise ValueError("owned server exited before readiness")
                    try:
                        connection = http_client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                        connection.request("GET", "/v1/models")
                        response = connection.getresponse()
                        data = json.loads(response.read())
                        connection.close()
                        if response.status == 200 and data.get("data"):
                            model = data["data"][0]["id"]
                            break
                    except (OSError, ValueError, http_client.HTTPException):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("owned server readiness deadline exceeded")
                    time.sleep(.5)
                for repeat in range(args.runs):
                    for name, prompt, expected in prompts:
                        result = request_sse(args.port, model, [{"role": "user", "content": prompt}], args.max_tokens, args.timeout)
                        result.update(scenario=name, repeat=repeat,
                            cache_label="first request of scenario in process; OS caches uncontrolled" if repeat == 0 else "repeated request; inspect cached token usage")
                        if isinstance(expected, str) and result["answer"].strip() != expected:
                            result["errors"].append("incorrect arithmetic answer")
                        if isinstance(expected, dict):
                            try:
                                if not json_values_equal(json.loads(result["answer"]), expected):
                                    raise ValueError("JSON value differs")
                            except ValueError:
                                result["errors"].append("incorrect JSON answer")
                        if not result["answer"].strip():
                            result["errors"].append("empty answer")
                        if result["finish_reason"] != "stop":
                            result["errors"].append("did not finish normally")
                        key = (name, repeat)
                        if key in references:
                            reference = references[key]
                            result["reference_answer_equal"] = result["answer"] == reference["answer"]
                            result["reference_usage_equal"] = result["usage"] == reference["usage"]
                        else:
                            references[key] = result
                        process_row["requests"].append(result)
                        save(args.output / "report.json", report)
                        print(json.dumps({k: result.get(k) for k in ("scenario", "repeat", "ttft_ms", "e2e_ms", "finish_reason", "errors")}), flush=True)
            finally:
                child_pid = None
                try:
                    child_pid = int(pid_path.read_text())
                    if child_pid <= 0:
                        child_pid = None
                except (ValueError, OSError):
                    pass
                finally:
                    # A launcher interrupted while writing its PID must still
                    # release the group owned by this harness.
                    stop_owned(process, child_pid)
        memory = re.search(r"(\d+)\s+maximum resident set size", (directory / "stderr.log").read_text())
        process_row["peak_process_rss_bytes"] = int(memory[1]) if memory else None
        log = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
        process_row["placement_lines"] = [line for line in log.splitlines() if "[moe-offload]" in line or "Metal device" in line]
        process_row["exit_code_after_owned_shutdown"] = process.returncode
        # The dump is finalized by its first sampled request. Compare only after
        # this owned server has exited, so no partially written sidecar is read.
        dump = directory / "logits.f32"
        process_row["errors"] = []
        try:
            _, process_row["first_token_sidecar"] = read_logits(dump)
            if first_dump is None:
                first_dump = dump
            else:
                process_row["logit_comparison"] = compare_logits(first_dump, dump)
        except (OSError, ValueError, KeyError) as error:
            process_row["errors"].append(str(error))
        save(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cli", "http", "compare-logits"))
    parser.add_argument("--variant", action="append", default=[], help="LABEL=checkout-root; repeat")
    parser.add_argument("--variant-config", action="append", default=[], help="LABEL=configuration.json; optional per variant")
    parser.add_argument("--order", help="Comma-separated labels, e.g baseline,candidate,candidate,baseline")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", default="128:64,512:128")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--token-stream", choices=("random", "fixed"), default="random",
                        help="Random IDs exercise new expert pages; fixed IDs repeat the same inference workload")
    parser.add_argument("--context", type=int, default=None, help="Override MAX_CONTEXT; default uses engine/config")
    parser.add_argument("--chunk", type=int, default=None, help="Override TS_PREFILL_CHUNK; default uses engine/config")
    parser.add_argument("--cpu-moe", type=int, default=None, help="Override host expert layers; default uses engine/config")
    parser.add_argument("--threads", type=int, default=None, help="Override expert and graph CPU threads; default uses engine/config")
    parser.add_argument("--env", action="append", default=[], help="Allowlisted benchmark KEY=VALUE")
    parser.add_argument("--variant-env", action="append", default=[],
                        help="Allowlisted per-variant LABEL.KEY=VALUE; overrides --env")
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--port", type=int, default=18779)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--execute", action="store_true", help="Actually launch processes; default prints plan")
    parser.add_argument("--first", type=Path, help="First raw F32 vector for compare-logits")
    parser.add_argument("--second", type=Path, help="Second raw F32 vector for compare-logits")
    args = parser.parse_args()
    if args.mode == "compare-logits":
        if not args.first or not args.second:
            parser.error("compare-logits requires --first and --second")
        print(json.dumps(compare_logits(args.first, args.second), indent=2, allow_nan=False))
        return
    args.variants = {key: Path(value).resolve() for key, value in pairs(args.variant).items()}
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) for label in args.variants):
        parser.error("variant labels must contain only letters, digits, dots, underscores or hyphens")
    args.environment = pairs(args.env)
    args.variant_configs = {label: Path(path).resolve(strict=True) for label, path in pairs(args.variant_config).items()}
    if set(args.variant_configs) - args.variants.keys():
        parser.error("variant-config must name a configured variant")
    args.variant_environment = {}
    for key, value in pairs(args.variant_env).items():
        label, dot, setting = key.partition(".")
        if not dot or label not in args.variants or setting not in ENV_KEYS or setting == "TS_DUMP_LOGITS":
            parser.error("variant-env requires a configured label and allowlisted benchmark setting")
        args.variant_environment.setdefault(label, {})[setting] = value
    if not args.variants or not args.model or not args.output:
        parser.error("cli/http require --variant, --model and --output")
    args.order = args.order.split(",") if args.order else list(args.variants)
    if not args.order or set(args.order) - args.variants.keys():
        parser.error("order contains unconfigured variants")
    args.cases = [tuple(int(n) for n in item.split(":")) for item in args.cases.split(",")]
    if (any(len(c) != 2 or c[0] < 1 or c[1] < 0 for c in args.cases)
            or min(args.runs, args.timeout, args.max_tokens) <= 0
            or any(value is not None and value <= 0 for value in (args.context, args.chunk, args.threads))
            or (args.cpu_moe is not None and args.cpu_moe < 0)):
        parser.error("invalid nonpositive settings or pp:tg cases")
    args.output = args.output.resolve()
    if not any(args.output.is_relative_to(ROOT / directory) for directory in ("artifacts", "docs/validation")):
        parser.error("output must be within this checkout's ignored artifacts/ or docs/validation/")
    if args.execute:
        args.model = args.model.resolve(strict=True)
        args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "mode": args.mode, "execute": args.execute,
              "harness_sha256": sha(__file__), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "cache_condition": "OS/file/shader caches were not flushed. No physical-cold claim.",
              "environment_condition": "Inherited TS_*, TENSORSHARP_*, GGML_*, MAX_CONTEXT and KV_CACHE_DTYPE are removed. Engine defaults/config remain unless explicitly overridden by options or allowlisted env settings. TS_DUMP_LOGITS is armed for evidence.",
              "memory_condition": "macOS /usr/bin/time -l peak process RSS bytes only; not total unified memory or total GPU memory",
              "cli_condition": "Run1 follows standard warmup and includes new-shape setup. Random runs may page different experts; fixed runs repeat IDs. Process wall/RSS includes untimed greedy correctness chain.",
              "token_stream": args.token_stream,
              "http_condition": "TTFT is first nonempty SSE content, including queue/preprocessing/prefill/transport. Chinese prose quality needs human review. Repeated prefix-cache usage must be compared.",
              "quality_condition": "Successful evidence collection is not a broad quality acceptance. Inspect full-logit errors, top1, exact greedy chain and prose answers separately.",
              "model_file": {"name": args.model.name, "size_bytes": args.model.stat().st_size if args.execute else None},
              "complete": False, "processes": []}
    try:
        (cli if args.mode == "cli" else http)(args, report)
        report["complete"] = bool(args.execute)
        report["passed"] = report["complete"] and all(not p.get("errors") and all(not r["errors"] for r in p.get("requests", [])) for p in report["processes"])
        if args.mode == "cli":
            report["quality_same_as_first_configuration"] = all(p.get("greedy_chain_equal", True)
                and p.get("logit_comparison", {}).get("top1_equal", True) for p in report["processes"])
        else:
            results = [r for p in report["processes"] for r in p.get("requests", [])]
            report["quality_summary"] = {"requests": len(results),
                "failed_requests": sum(bool(r["errors"]) for r in results),
                "exact_reference_answer_mismatches": sum(r.get("reference_answer_equal") is False for r in results),
                "reference_usage_mismatches": sum(r.get("reference_usage_equal") is False for r in results),
                "first_logit_vectors_validated": sum("first_token_sidecar" in p for p in report["processes"]),
                "first_logit_vectors_compared": sum("logit_comparison" in p for p in report["processes"]),
                "first_logit_bit_mismatches": sum(not p["logit_comparison"]["bit_identical"] for p in report["processes"] if "logit_comparison" in p),
                "first_logit_all_bit_identical": all(p["logit_comparison"]["bit_identical"] for p in report["processes"] if "logit_comparison" in p)
                    if any("logit_comparison" in p for p in report["processes"]) else None,
                "scope": "Arithmetic/JSON exact checks; deterministic full output equality is recorded, Chinese semantic quality requires review"}
    except Exception as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if args.execute:
            save(args.output / "report.json", report)
    if args.execute and not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
