#!/usr/bin/env python3
"""Serial, real-model HTTP quality/compute/memory smoke coverage on Windows/Linux.

Each invocation owns one fresh server; requests reset logical KV with prefix
reuse disabled. First occurrence and two subsequent repetitions are separate.
No generated code is executed. Evidence must remain in ignored directories.
Requires psutil. This is not a cold-storage or independent-engine benchmark.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
_cgroup_spec = importlib.util.spec_from_file_location("linux_cgroup_telemetry", HERE / "linux-cgroup-telemetry.py")
CGROUP = importlib.util.module_from_spec(_cgroup_spec)
_cgroup_spec.loader.exec_module(CGROUP)
PROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
NATIVE_NAME = "GgmlOps.dll" if sys.platform == "win32" else "libGgmlOps.so"
PROMPTS = {
    "squares": "Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.",
    "tool_json": 'Available tool: get_weather(city, unit), where unit is "celsius" or "fahrenheit". User asks: What is the weather in Hangzhou in Celsius? Return only one JSON object with exactly the keys "name" and "arguments", containing the tool name and its required arguments. Do not answer the weather question.',
    "code": "Write a Python function square(x) that returns x*x. Return only the function, without Markdown.",
    "long_extract": "Read these records. " + " ".join(f"record_{i:03d}={i * 7 + 3};" for i in range(60)) + " Return only the integer value of record_037.",
    "image_ocr": "Read the four black digits in this image. Return only those four digits, without explanation.",
}


def backend_environment(settings, overrides, inherited):
    """Keep backend and OpenMP settings explicit in reproducible evidence."""
    settings = dict(settings)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"(?:TS_|GGML_|NCCL_|OMP_|GOMP_)[A-Z0-9_]+", key) or key in settings:
            raise ValueError("Require a unique explicit backend KEY=VALUE setting")
        settings[key] = value
    env = {k: v for k, v in inherited.items()
           if not k.startswith(("TS_", "TENSORSHARP_", "GGML_", "NCCL_", "OMP_", "GOMP_"))}
    env.update(settings)
    return env, settings


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(path))


def checkpoint_files(model):
    """Require all siblings of a split GGUF, rather than identify only shard 1."""
    model = Path(model).resolve()
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", model.name)
    if not match:
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", model.name):
            raise ValueError("Start a split checkpoint at shard 1")
        return [model]
    count = int(match[2])
    if not 1 <= count <= 99999:
        raise ValueError("Invalid GGUF shard count")
    files = [model.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    if any(not path.is_file() for path in files):
        raise ValueError("Missing GGUF shard")
    return files


def verified_identities(paths, manifest=None):
    if manifest is None:
        return [identity(path) for path in paths]
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if data.get("status") != "verified":
        raise ValueError("Require a completed whole-file download verification")
    entries = {}
    for entry in data.get("shards", []):
        path = str(Path(entry["path"]).resolve())
        if path in entries:
            raise ValueError("Duplicate verified file")
        entries[path] = entry
    result = []
    for path in paths:
        path = Path(path).resolve()
        entry = entries.get(str(path), {})
        digest = entry.get("actual_sha256", "")
        if (entry.get("verified") is not True or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes")):
            raise ValueError("File missing from verification or identity/size changed: " + str(path))
        result.append(dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                           sha256=digest, hash_source="prior_whole_file_verification", verification_manifest_sha256=sha(manifest)))
    return result


def native_module_paths(pid):
    if sys.platform == "win32":
        modules = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).Modules | Where-Object ModuleName -eq 'GgmlOps.dll' | Select-Object -ExpandProperty FileName"], check=True, capture_output=True, text=True, timeout=30, creationflags=PROCESS_FLAGS)
        return modules.stdout.strip().splitlines()
    paths = set()
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == NATIVE_NAME:
            paths.add(fields[5])
    return sorted(paths)


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON key")
    return result


def quality(case, text, complete):
    """Independent exact task checks; never repair fences or partial answers."""
    if not complete or not isinstance(text, str) or not text.strip():
        return False
    answer = text.strip()
    if case == "squares":
        return bool(re.fullmatch(r"\d+(?:\s*,\s*\d+){19}", answer)) and [int(v) for v in answer.split(",")] == [i*i for i in range(1, 21)]
    if case == "tool_json":
        try:
            return json.loads(answer, object_pairs_hook=unique_object) == {"name": "get_weather", "arguments": {"city": "Hangzhou", "unit": "celsius"}}
        except (ValueError, TypeError):
            return False
    if case == "long_extract":
        return answer == "262"
    if case == "image_ocr":
        return answer == "4821"
    if case == "code":
        # Compare syntax to the requested pure expression; no eval/exec or
        # normalization of fences, imports, annotations or extra statements.
        try:
            tree = ast.parse(answer)
            expected = ast.parse("def square(x):\n    return x*x")
            return ast.dump(tree, include_attributes=False) == ast.dump(expected, include_attributes=False)
        except SyntaxError:
            return False
    raise ValueError("Unknown case")


def metrics(response):
    for field in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration", "total_duration"):
        value = response.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError("Missing/nonpositive timing or count: " + field)
    if response.get("prompt_cache_hit_tokens") != 0:
        raise ValueError("Expected an uncached prompt")
    prompt, decode, total = (response[k] for k in ("prompt_eval_duration", "eval_duration", "total_duration"))
    if total < prompt + decode:
        raise ValueError("Compute durations exceed total duration")
    return dict(prefill_tps=response["prompt_eval_count"] * 1e9 / prompt,
                decode_tps=response["eval_count"] * 1e9 / decode,
                prefill_ms=prompt / 1e6, decode_ms=decode / 1e6, total_ms=total / 1e6,
                prompt_tokens=response["prompt_eval_count"], generated_tokens=response["eval_count"])


def post(port, body, timeout, evidence):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    evidence["request"] = body
    started = time.monotonic()
    try:
        connection.request("POST", "/api/chat/ollama", json.dumps(body).encode(), {"Content-Type": "application/json"})
        response = connection.getresponse()
        evidence["http_status"] = response.status
        evidence["raw_response"] = response.read().decode("utf-8")
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        evidence["response"] = json.loads(evidence["raw_response"])
        return evidence["response"]
    finally:
        evidence["http_wall_ms"] = (time.monotonic() - started) * 1000
        connection.close()


class Sampler:
    def __init__(self, path):
        self.path, self.process = path, None
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        row = dict(unix_ns=time.time_ns(), available_host_bytes=psutil.virtual_memory().available)
        if self.process is not None:
            try:
                row["memory"] = self.process.memory_info()._asdict()
                row["io"] = self.process.io_counters()._asdict()
                row["cpu"] = self.process.cpu_times()._asdict()
            except psutil.NoSuchProcess:
                pass
        if sys.platform == "linux":
            try:
                row["cgroup"] = CGROUP.capture(self.process.pid if self.process is not None else None)
            except (OSError, ValueError) as error:
                # Do not discard RSS/GPU samples when a process exits between
                # observations, or quietly interpret unknown limits as zero.
                row["cgroup"] = {"errors": [str(error)]}
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, creationflags=PROCESS_FLAGS)
        if gpu.returncode != 0:
            raise RuntimeError(gpu.stderr)
        row["gpu"] = [dict(zip(("index", "used_mib", "total_mib", "utilization"), [float(x.strip()) for x in line.split(",")])) for line in gpu.stdout.splitlines()]
        self.rows.append(row)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")

    def run(self):
        while not self.stop.wait(1):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))


def summarize(records):
    result = {}
    for case in sorted({r["case"] for r in records}):
        rows = [r for r in records if r["case"] == case]
        timed = [r for r in rows if "metrics" in r]
        warm = [r for r in timed if r["repetition"] > 0]
        result[case] = dict(quality_passes=sum(r.get("quality_passed") is True for r in rows), requests=len(rows),
            first=next((r["metrics"] for r in timed if r["repetition"] == 0), None),
            warm_samples=len(warm), warm={key: dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
                for key in ("prefill_tps", "decode_tps") if (values := [r["metrics"][key] for r in warm])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--image", type=Path, help="Existing shared card-0.png, containing 4821")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=52175)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", default="squares,tool_json,code,long_extract")
    parser.add_argument("--cpu-moe", action="store_true")
    parser.add_argument("--n-cpu-moe", type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--devices", default="0")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--layer-split", type=int)
    placement.add_argument("--tp", type=int)
    parser.add_argument("--env", action="append", default=[], help="Explicit TS_/GGML_/NCCL_ KEY=VALUE override, retained in evidence")
    parser.add_argument("--checkpoint-manifest", type=Path, help="Completed download-validated-model report; reuse its prior full hashes and recheck size, without rehashing each run")
    parser.add_argument("--hash-companion", action="store_true", help="Hash the entire companion now instead of using the checkpoint manifest (for a locally prepared vision GGUF); makes no publisher integrity claim")
    parser.add_argument("--tool-roundtrip", action="store_true")
    parser.add_argument("--json-schema", action="store_true", help="Separate constrained OpenAI JSON request; never waives raw-format failures")
    parser.add_argument("--expert-cache-mb", type=int, default=0)
    args = parser.parse_args()
    cases = args.cases.split(",")
    out = args.output.resolve()
    if sys.platform not in ("win32", "linux") or not math.isfinite(args.timeout) or args.timeout <= 0 or args.repetitions < 1 or args.max_tokens < 1 or args.context < 1 or args.expert_cache_mb < 0 or args.threads < 1:
        parser.error("Windows/Linux and positive limits required")
    if not re.fullmatch(r"\d+(?:,\d+)*", args.devices) or len(set(args.devices.split(','))) != len(args.devices.split(',')):
        parser.error("Require unique CUDA device ordinals")
    if any(value is not None and not 1 <= value <= len(args.devices.split(',')) for value in (args.layer_split, args.tp)):
        parser.error("Parallel size must fit the visible device list")
    if args.n_cpu_moe is not None and (args.n_cpu_moe < 0 or args.cpu_moe):
        parser.error("Use either --cpu-moe or a nonnegative --n-cpu-moe")
    if len(cases) != len(set(cases)) or any(c not in PROMPTS for c in cases):
        parser.error("Require unique known cases")
    if "image_ocr" in cases and not (args.image and args.mmproj):
        parser.error("Image case requires the card and matching projector")
    if not any(out.is_relative_to(ROOT / base) for base in ("artifacts", "docs/validation")):
        parser.error("Evidence must be ignored under artifacts/ or docs/validation/")
    out.mkdir(parents=True, exist_ok=False)
    # Keep the executed checker with the evidence even if its source evolves
    # while a later model or independent reference is being investigated.
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out / "linux-cgroup-telemetry.py").write_bytes((HERE / "linux-cgroup-telemetry.py").read_bytes())
    (out / "validate-qwen38-tool-calls.py").write_bytes((HERE / "validate-qwen38-tool-calls.py").read_bytes())
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Port is occupied; leave other servers alone")
    spec = importlib.util.spec_from_file_location("tool_http", HERE / "validate-qwen38-tool-calls.py")
    http = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(http)
    settings = dict(CUDA_VISIBLE_DEVICES=args.devices, MAX_CONTEXT=str(args.context), KV_CACHE_DTYPE="f16",
                    TS_GGML_CPU_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads), TS_HOST_MOE_EXPERT_CACHE_MB=str(args.expert_cache_mb))
    try:
        env, settings = backend_environment(settings, args.env, os.environ)
    except ValueError as error:
        parser.error(str(error))
    command = ["dotnet", str(args.server.resolve()), "--model", str(args.model.resolve()), "--mmproj", str(args.mmproj.resolve()) if args.mmproj else "none",
               "--backend", "ggml_cuda", "--host", "127.0.0.1", "--port", str(args.port), "--no-spec", "--no-prefix-cache", "--no-multi-agent",
               "--temperature", "0", "--top-k", "0", "--top-p", "1", "--repeat-penalty", "1", "--seed", "17"]
    if args.cpu_moe:
        command += ["--cpu-moe", "--cpu-moe-threads", str(args.threads)]
    if args.n_cpu_moe is not None:
        command += ["--n-cpu-moe", str(args.n_cpu_moe), "--cpu-moe-threads", str(args.threads)]
    for key in ("layer_split", "tp"):
        if getattr(args, key) is not None:
            command += ["--" + key.replace('_', '-'), str(getattr(args, key))]
    shards = verified_identities(checkpoint_files(args.model), args.checkpoint_manifest)
    report = dict(schema_version=1, started_unix=time.time(), command=command, environment=settings, records=[], tool_roundtrips=[],
                  execution_plan=dict(cases=cases, repetitions=args.repetitions, expected_requests=len(cases)*args.repetitions,
                                      context=args.context, max_tokens=args.max_tokens, tool_roundtrip=args.tool_roundtrip, json_schema=args.json_schema),
                  run_complete=False, passed=False, checkpoint=shards[0], checkpoint_shards=shards,
                  companion=verified_identities([args.mmproj], None if args.hash_companion else args.checkpoint_manifest)[0] if args.mmproj else None, image=identity(args.image) if args.image else None,
                  platform=sys.platform, native_name=NATIVE_NAME,
                  binaries={p.name: identity(p) for p in args.server.parent.iterdir() if p.is_file() and ((p.suffix == '.dll' and p.name.startswith("TensorSharp")) or p.name == NATIVE_NAME)},
                  harness_sha256=sha(__file__), tool_harness_sha256=sha(HERE / "validate-qwen38-tool-calls.py"),
                  cgroup_sampler_sha256=sha(HERE / "linux-cgroup-telemetry.py"),
                  limitations=["SHA identifies local bytes; no independent publisher integrity claim without a matching manifest.",
                      "Only the first server request is process-first; each case's first occurrence can use earlier warmed resources.",
                      "Model hash reads warm the OS cache; no cold-SSD claim. Server startup includes its own kernel warmup.",
                      "With --checkpoint-manifest, hashes are prior download verification plus current size checks; no repeated full-file integrity scan is claimed.",
                      "Prefill duration is model forwards only, excluding media encoding/rendering/queueing; total latency includes them.",
                      "Decode numerator is emitted token count; duration includes final EOS computation, following the server API. Not a fixed-token kernel benchmark.",
                      "Sampled board VRAM includes desktop/other processes; process RSS is neither a RAM quota nor physical SSD traffic.",
                      "No independent numerical oracle or equal-resource baseline; narrow task acceptance only.",
                      "Server termination by harness does not validate graceful native/budget cleanup."])
    def save():
        report["summary"] = summarize(report["records"])
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    sampler, server = Sampler(out / "memory.jsonl"), None
    try:
        sampler.sample()
        with (out / "server.log").open("w", encoding="utf-8") as log:
            startup_started = time.monotonic()
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=PROCESS_FLAGS)
            report["server_pid"] = server.pid
            sampler.process = psutil.Process(server.pid)
            sampler.thread.start()
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + args.timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server exited during startup")
                try:
                    model = http.hosted_model(url, 2)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Startup deadline exceeded")
                    time.sleep(1)
            paths = native_module_paths(server.pid)
            if len(paths) != 1:
                raise RuntimeError("Cannot identify loaded native library")
            report["loaded_native"] = identity(paths[0])
            if report["loaded_native"]["sha256"] != report["binaries"][NATIVE_NAME]["sha256"]:
                raise RuntimeError("Unexpected native library loaded")
            report["startup_ms"] = (time.monotonic() - startup_started) * 1000
            for case in cases:
                for repetition in range(args.repetitions):
                    row = dict(case=case, repetition=repetition, quality_passed=False)
                    report["records"].append(row)
                    message = dict(role="user", content=PROMPTS[case])
                    if case == "image_ocr":
                        message["images"] = [base64.b64encode(args.image.read_bytes()).decode()]
                    body = dict(model=model, messages=[message], stream=False, think=False, skills=[], skills_discovery=False, multi_agent=False,
                                options=dict(temperature=0, top_k=0, top_p=1, min_p=0, repeat_penalty=1, presence_penalty=0, frequency_penalty=0,
                                             seed=17, num_predict=args.max_tokens, stop=[]))
                    save()
                    try:
                        response = post(args.port, body, args.timeout, row)
                        row["complete"] = response.get("done") is True and response.get("done_reason") == "stop" and not response.get("error")
                        row["quality_passed"] = quality(case, response.get("message", {}).get("content"), row["complete"])
                        row["metrics"] = metrics(response)
                    except (TimeoutError, OSError) as error:
                        row["error"] = str(error)
                        raise  # Do not overlap a still-running timed-out GPU request.
                    except Exception as error:
                        row["error"] = str(error)
                    save()
                    print(json.dumps({k: row.get(k) for k in ("case", "repetition", "quality_passed", "error", "metrics")}), flush=True)
            if args.tool_roundtrip:
                for stream in (False, True):
                    result = http.run_case(SimpleNamespace(url=url, model=model, timeout=args.timeout, max_tokens=args.max_tokens), "weather", stream, False)
                    report["tool_roundtrips"].append(result)
                    save()
                    print(json.dumps(dict(tool_stream=stream, status=result["status"], error=result.get("error"))), flush=True)
                    if any(t.get("http_status") is None for t in result["turns"]):
                        raise RuntimeError("Incomplete tool transport; stop before another request")
            if args.json_schema:
                evidence = dict(passed=False)
                report["json_schema"] = evidence
                schema = dict(type="object", properties=dict(name=dict(type="string", enum=["get_weather"]),
                    arguments=dict(type="object", properties=dict(city=dict(type="string"), unit=dict(type="string", enum=["celsius", "fahrenheit"])),
                                   required=["city", "unit"], additionalProperties=False)), required=["name", "arguments"], additionalProperties=False)
                try:
                    message, finish, usage = http.request(url, dict(model=model, messages=[dict(role="user", content=PROMPTS["tool_json"])],
                        temperature=0, top_k=0, top_p=1, repetition_penalty=1, seed=17, max_tokens=args.max_tokens,
                        stream=False, think=False, skills=[], skills_discovery=False, multi_agent=False,
                        response_format=dict(type="json_schema", json_schema=dict(name="weather_call", strict=True, schema=schema))), args.timeout, evidence)
                    evidence["passed"] = quality("tool_json", message.get("content"), finish == "stop") and bool(usage)
                except (TimeoutError, OSError):
                    raise
                except Exception as error:
                    evidence["error"] = str(error)
                save()
                print(json.dumps(dict(json_schema_passed=evidence["passed"], error=evidence.get("error"))), flush=True)
            report["run_complete"] = True
            report["passed"] = all(r.get("quality_passed") and "metrics" in r and not r.get("error") for r in report["records"]) and all(r["status"] == "ok" for r in report["tool_roundtrips"]) and (not args.json_schema or report["json_schema"]["passed"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        # Finish sampling while the process still exists. A teardown race can
        # otherwise turn the final /proc read into a misleading sample error.
        sampler.stop.set()
        if sampler.thread.is_alive():
            sampler.thread.join(timeout=15)
        if server is not None:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
            report["server_exit"] = server.returncode
        report["memory"] = dict(samples=len(sampler.rows), errors=sampler.errors,
            peak_rss_bytes=max((r.get("memory", {}).get("rss", 0) for r in sampler.rows), default=0),
            peak_private_bytes=max((r.get("memory", {}).get("private", 0) for r in sampler.rows), default=0),
            peak_board_mib=max((d["used_mib"] for r in sampler.rows for d in r["gpu"]), default=0),
            peak_all_boards_mib=max((sum(d["used_mib"] for d in r["gpu"]) for r in sampler.rows), default=0),
            peak_per_board_mib={str(int(i)): max(d['used_mib'] for r in sampler.rows for d in r['gpu'] if d['index'] == i)
                                for i in {d['index'] for r in sampler.rows for d in r['gpu']}},
            peak_sampled_cgroup_bytes=max((int(r['cgroup']['memory.current']) for r in sampler.rows if r.get('cgroup', {}).get('memory.current', '').isdigit()), default=None))
        report["finished_unix"] = time.time()
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
