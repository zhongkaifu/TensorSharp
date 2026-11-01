#!/usr/bin/env python3
"""Windows-local functional check: existing native IChatClient, then an owned HTTP host.

No builds, downloads, model-payload hashing, external endpoints, or math-answer
oracle. Output directory must be new. Only processes started here are stopped.
Native context1088 accommodates the existing sample's >1024-token reserve;
HTTP context256 remains bounded. These are functional, not timing comparisons.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
from http import client as http_client
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def native_module(pid):
    """Inspect only the owned Windows process; do not infer DLL choice from PATH."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = kernel.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION | VM_READ
    if not handle:
        return None
    try:
        modules = (wintypes.HMODULE * 2048)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
            return None
        require(needed.value <= ctypes.sizeof(modules), "Process module inventory exceeded its bound")
        matches = []
        for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            name = ctypes.create_unicode_buffer(32768)
            if psapi.GetModuleFileNameExW(handle, module, name, len(name)) and Path(name.value).name.lower() == "ggmlops.dll":
                matches.append(str(Path(name.value).resolve()))
        require(len(matches) <= 1, "More than one GgmlOps.dll loaded")
        return matches[0] if matches else None
    finally:
        kernel.CloseHandle(handle)


def check_usage(value, maximum):
    require(isinstance(value, dict), "Missing token usage")
    for name in ("prompt_tokens", "completion_tokens"):
        require(type(value.get(name)) is int and value[name] > 0, "Expected positive integer " + name)
    require(value["completion_tokens"] <= maximum, "Completion exceeded the requested token cap")
    require(all(not isinstance(v, (int, float)) or math.isfinite(v) for v in value.values()), "Nonfinite usage value")
    if "total_tokens" in value:
        require(value["total_tokens"] == value["prompt_tokens"] + value["completion_tokens"], "Inconsistent total token usage")


def decode_json(raw):
    def reject(value):
        raise ValueError("Nonfinite JSON number: " + value)
    return json.loads(raw, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="First GGUF shard, or a complete single-file model")
    parser.add_argument("--native-dll", type=Path, required=True, help="Staged TensorSharp.NativeChat.dll")
    parser.add_argument("--server-dll", type=Path, required=True, help="Staged TensorSharp.Server.Host.dll")
    parser.add_argument("--native-sha256", required=True, help="Required GgmlOps.dll hash, identical in both app directories")
    parser.add_argument("--out", type=Path, required=True, help="New directory; prefer D: on the MSI")
    parser.add_argument("--expected-model-id", help="Defaults to the selected GGUF filename without .gguf")
    parser.add_argument("--architecture", default="qwen4exp")
    parser.add_argument("--model-hash-evidence", type=Path, help="Optional previously completed full-shard hash evidence; only size/mtime rechecked here")
    parser.add_argument("--native-context", type=int, default=1088)
    parser.add_argument("--http-context", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, choices=(8, 16), default=16)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--load-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--prompt", default="Explain the water cycle in one short sentence.")
    args = parser.parse_args()
    require(os.name == "nt", "This harness inspects Windows process modules and is Windows-only")
    require(0 < args.port < 65536 and args.load_timeout > 0 and args.request_timeout > 0, "Invalid port/deadline")
    require(args.native_context > 1024 + args.max_tokens, "Existing native sample requires context >1024+max_tokens")
    require(args.http_context > args.max_tokens and args.prompt.strip(), "Invalid HTTP context or empty prompt")
    args.model, args.native_dll, args.server_dll, args.out = [p.resolve() for p in (args.model, args.native_dll, args.server_dll, args.out)]
    require(args.model.is_file(), "Existing local model entry point required")
    require(not args.out.exists(), "Refusing to overwrite evidence")
    expected_hash = args.native_sha256.upper()
    require(re.fullmatch(r"[0-9A-F]{64}", expected_hash), "Invalid native SHA256")
    expected_id = args.expected_model_id or args.model.stem
    allowed_response_models = {expected_id, args.model.name}
    provenance = {}
    for label, app in (("native", args.native_dll), ("server", args.server_dll)):
        require(app.is_file(), "Missing staged application: " + str(app))
        native = app.parent / "GgmlOps.dll"
        require(native.is_file() and file_hash(native) == expected_hash, "Wrong staged native DLL: " + str(native))
        provenance[label] = {"app": str(app), "appSha256": file_hash(app), "nativePath": str(native), "nativeSha256": expected_hash}
    model_evidence = {"entryPoint": str(args.model), "entryPointBytes": args.model.stat().st_size,
                      "payloadHashesRevalidatedByThisRun": False}
    if args.model_hash_evidence:
        evidence = decode_json(args.model_hash_evidence.read_bytes())
        require(evidence.get("complete") is True and evidence.get("files"), "Incomplete model hash evidence")
        require(evidence["files"][0]["FileName"] == args.model.name, "Hash evidence uses another model entry point")
        for file in evidence["files"]:
            require(Path(file["FileName"]).name == file["FileName"], "Invalid shard filename in prior evidence")
            path = args.model.parent / file["FileName"]
            stat = path.stat()
            require(stat.st_size == file["Bytes"] and stat.st_mtime_ns == file["modifiedNs"], "Model changed since the full payload hash pass: " + str(path))
        model_evidence.update(priorEvidencePath=str(args.model_hash_evidence.resolve()), priorEvidenceSha256=file_hash(args.model_hash_evidence), priorFullPayloadHashes=evidence)
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith(("TS_", "TENSORSHARP_", "GGML_", "OPENAI_")) or name in ("MAX_CONTEXT", "KV_CACHE_DTYPE", "MAX_TOKENS", "ASPNETCORE_URLS", "HOST", "PORT"):
            del env[name]
    fixed = dict(TS_CPU_MOE="1", TS_N_CPU_MOE="0", TS_CPU_MOE_THREADS="0", TS_GGML_CPU_THREADS="8",
                 TS_GGUF_PREFAULT="0", TS_HOST_MOE_EXPERT_CACHE_MB="0", TS_Q4E_RETAINED_CACHE_MB="0",
                 TS_SPEC="0", KV_CACHE_DTYPE="f16", TS_HOST_MOE_METAL_OVERLAP="0",
                 TENSORSHARP_TP_DEGREE="1", TENSORSHARP_LAYER_SPLIT_DEGREE="1", TS_SCHED_MAX_RUNNING_SEQS="1")
    env.update(fixed)
    args.out.mkdir(parents=True)
    report = {"startedUtc": utc(), "complete": False, "passed": None, "model": model_evidence,
              "expectedArchitecture": args.architecture, "expectedModelId": expected_id, "allowedResponseModelIds": sorted(allowed_response_models),
              "applicationProvenance": provenance, "fixedEnvironment": fixed, "nativeContext": args.native_context,
              "httpContext": args.http_context, "maxOutputTokens": args.max_tokens,
              "contextDifferenceReason": "Existing native sample reserves1024 tokens; HTTP can use256. Functional compatibility only; no timing comparison.",
              "placementDifferenceReason": "The existing native sample does not initialize MoeCpuOffloadConfig from environment, so record its automatic placement. The HTTP host receives --cpu-moe explicitly and must keep every expert layer on the host.",
              "validatesLogitFiniteness": False, "validatesMathematicalAnswer": False,
              "notRun": ["HTTP .NET SDK example", "cancellation behavior", "performance comparison", "vision/projector"], "cases": []}

    def persist():
        (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def stage(name):
        report["currentCase"] = name
        report["currentCaseStartedUtc"] = utc()
        persist()
        print(name, report["currentCaseStartedUtc"], flush=True)

    def child_env(app, context):
        result = env.copy()
        result.update(MAX_CONTEXT=str(context), TS_SCHED_NUM_BLOCKS=str((context + 255) // 256))
        cuda = str(Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")) / "bin/x64")
        result["PATH"] = str(app.parent) + os.pathsep + cuda + os.pathsep + result.get("PATH", "")
        return result

    def observe(proc, label, row):
        if row.get("loadedNativeModule"):
            return
        module = native_module(proc.pid)
        if module:
            require(Path(module) == Path(provenance[label]["nativePath"]), "Loaded native module differs from staged application")
            require(file_hash(Path(module)) == expected_hash, "Loaded native DLL hash differs")
            row["loadedNativeModule"] = {"path": module, "sha256": expected_hash, "observedUtc": utc()}
            persist()

    def offload_evidence(text, require_all_host):
        lines = [line for line in text.splitlines() if line.startswith("[moe-offload]")]
        if args.architecture == "qwen4exp":
            match = re.search(r"\[moe-offload\] qwen4exp .*routed experts of (\d+) of (\d+) layers run on the host", text)
            require(match and 0 <= int(match[1]) <= int(match[2]) and int(match[2]) > 0,
                    "Qwen4Exp did not report valid expert placement")
            host_layers, total_layers = int(match[1]), int(match[2])
            require(not require_all_host or host_layers == total_layers,
                    "HTTP --cpu-moe did not keep every expert layer on the host")
            return {"allExpertLayersOnHostVerified": host_layers == total_layers,
                    "hostExpertLayers": host_layers, "acceleratorExpertLayers": total_layers - host_layers,
                    "totalExpertLayers": total_layers, "allHostRequired": require_all_host, "lines": lines}
        return {"allExpertLayersOnHostVerified": None, "lines": lines,
                "note": "Offload logs retained; this harness has no architecture-specific placement parser for this model."}

    def stop_owned(proc, row):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        row["exitCode"] = proc.returncode
        row["stoppedUtc"] = utc()

    def http(route, label, body=None, timeout=None):
        duration = timeout or args.request_timeout
        deadline = time.monotonic() + duration
        conn = http_client.HTTPConnection("127.0.0.1", args.port, timeout=duration)
        try:
            data = None if body is None else json.dumps(body).encode("utf-8")
            if body is not None:
                (args.out / (label + ".request.json")).write_bytes(data)
            conn.request("GET" if body is None else "POST", route, data, {"Content-Type": "application/json"})
            response = conn.getresponse()
            (args.out / (label + ".headers.json")).write_text(json.dumps({"status": response.status, "headers": list(response.getheaders())}, indent=2))
            parts, count = [], 0
            with (args.out / (label + ".response.txt")).open("wb") as capture:
                while True:
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "HTTP response exceeded its deadline")
                    if conn.sock:
                        conn.sock.settimeout(remaining)
                    part = response.read1(65536)
                    if not part:
                        break
                    capture.write(part)
                    capture.flush()  # Keep partial responses if a later read fails.
                    parts.append(part)
                    count += len(part)
                    require(count <= 4 * 1024 * 1024, "Unexpectedly large HTTP response")
            raw = b"".join(parts)
            require(response.status == 200, f"HTTP {response.status}; see {label}.response.txt")
            return raw, response.getheader("Content-Type", "")
        finally:
            conn.close()

    try:
        stage("native-IChatClient-mixed")
        command = ["dotnet", str(args.native_dll), "--model", str(args.model), "--backend", "ggml_cuda", "--mode", "mixed",
                   "--prompt", args.prompt, "--follow-up", "Name one stage of that cycle.", "--max-tokens", str(args.max_tokens),
                   "--context-tokens", str(args.native_context), "--timeout-seconds", str(args.request_timeout)]
        row = {"case": "native-IChatClient-mixed", "command": command, "context": args.native_context,
               "responseEvidence": "native.stdout.log retains complete output, including engine diagnostics; sample itself enforces nonempty assistant text"}
        report["cases"].append(row)
        with (args.out / "native.stdout.log").open("wb") as stdout, (args.out / "native.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=args.native_dll.parent, env=child_env(args.native_dll, args.native_context),
                                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            row["ownedPid"] = proc.pid
            deadline = time.monotonic() + args.load_timeout + 2 * args.request_timeout
            try:
                while proc.poll() is None:
                    observe(proc, "native", row)
                    require(time.monotonic() < deadline, "Native sample exceeded process deadline")
                    time.sleep(0.25)
                require(proc.returncode == 0, "Native sample failed; inspect full logs")
                require(row.get("loadedNativeModule"), "Could not verify the native sample's loaded GgmlOps.dll")
            finally:
                stop_owned(proc, row)
                persist()
        stderr = (args.out / "native.stderr.log").read_text(encoding="utf-8", errors="replace")
        expected_loaded = f"model={args.model.name}; architecture={args.architecture}; backend=ggml_cuda; context_tokens={args.native_context}"
        require(expected_loaded in stderr, "Native sample did not confirm exact model/architecture/backend/context")
        metrics = [line for line in stderr.splitlines() if line.startswith("[TensorSharp IChatClient]")]
        require(len(metrics) == 2, "Expected streaming and non-streaming native terminal metrics")
        for line, mode in zip(metrics, ("stream", "response")):
            require(f"mode={mode};" in line and re.search(r"; finish=(stop|length);", line, re.I), "Native completion lacks an accepted finish reason")
            require(re.search(r"; native_finish=(eos|stop_sequence|stop|max_tokens|thinking_budget|repetition);", line), "Native generation reported an error/unknown terminal reason")
            for name in ("prompt_tokens", "generated_tokens"):
                match = re.search(r"; " + name + r"=(\d+)", line)
                require(match and int(match[1]) > 0, "Native completion missing positive " + name)
                if name == "generated_tokens":
                    require(int(match[1]) <= args.max_tokens, "Native output exceeded token cap")
        native_output = (args.out / "native.stdout.log").read_text(encoding="utf-8", errors="replace")
        row.update(passed=True, terminalMetrics=metrics, actualBackend="ggml_cuda", hostOffload=offload_evidence(native_output + "\n" + stderr, False))
        persist()

        stage("owned-HTTP-server")
        with socket.socket() as available:
            available.bind(("127.0.0.1", args.port))  # Fail rather than attach to any existing service.
        command = ["dotnet", str(args.server_dll), "--model", str(args.model), "--mmproj", "none", "--backend", "ggml_cuda",
                   "--host", "127.0.0.1", "--port", str(args.port), "--max-tokens", str(args.max_tokens), "--temperature", "0",
                   "--cpu-moe", "--no-skills", "--no-multi-agent", "--no-prefix-cache", "--no-webui"]
        server_row = {"case": "owned-HTTP-server", "command": command, "context": args.http_context}
        report["cases"].append(server_row)
        with (args.out / "server.stdout.log").open("wb") as stdout, (args.out / "server.stderr.log").open("wb") as stderr:
            proc = subprocess.Popen(command, cwd=args.server_dll.parent, env=child_env(args.server_dll, args.http_context),
                                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            server_row["ownedPid"] = proc.pid
            try:
                persist()
                deadline = time.monotonic() + args.load_timeout
                while True:
                    require(proc.poll() is None, "Owned server exited during startup")
                    observe(proc, "server", server_row)
                    try:
                        state_raw, _ = http("/api/models", "server-state", timeout=2)
                        state = decode_json(state_raw)
                        if state.get("loaded"):
                            break
                    except (OSError, http_client.HTTPException):
                        pass
                    require(time.monotonic() < deadline, "Owned server readiness deadline exceeded")
                    time.sleep(0.5)
                require(state.get("loaded") == args.model.name and state.get("architecture") == args.architecture,
                        "HTTP host loaded a different model/architecture")
                require(state.get("loadedBackend") == "ggml_cuda" and state.get("contextTokens") == args.http_context,
                        "HTTP host backend/context differs from requested settings")
                require(server_row.get("loadedNativeModule"), "Could not verify the HTTP host's loaded GgmlOps.dll")
                host_log = "\n".join((args.out / filename).read_text(encoding="utf-8", errors="replace") for filename in ("server.stdout.log", "server.stderr.log"))
                server_row["hostOffload"] = offload_evidence(host_log, True)
                discovery_raw, _ = http("/v1/models", "models")
                ids = [item.get("id") for item in decode_json(discovery_raw).get("data", [])]
                require(ids == [expected_id], "Discovery must advertise exactly the expected model ID")
                server_row.update(passed=True, actualBackend=state["loadedBackend"], discoveryIds=ids)
                for streaming in (False, True):
                    label = "http-sse" if streaming else "http-json"
                    stage(label)
                    body = {"model": expected_id, "messages": [{"role": "user", "content": args.prompt}],
                            "max_tokens": args.max_tokens, "temperature": 0, "seed": 1, "think": False, "stream": streaming}
                    if streaming:
                        body["stream_options"] = {"include_usage": True}
                    raw, content_type = http("/v1/chat/completions", label, body)
                    if not streaming:
                        response = decode_json(raw)
                        require(response.get("model") in allowed_response_models and "error" not in response, "Wrong model or error in JSON completion")
                        require(len(response.get("choices", [])) == 1, "Expected one completion choice")
                        choice = response["choices"][0]
                        require(choice.get("finish_reason") in ("stop", "length"), "Missing/error JSON finish reason")
                        text = choice.get("message", {}).get("content")
                        usage = response.get("usage")
                        finishes = [choice["finish_reason"]]
                    else:
                        require(content_type.split(";")[0].strip() == "text/event-stream", "Expected SSE content type")
                        pieces, finishes, models = [], [], []
                        usage, done = None, False
                        for frame in re.split(r"\r?\n\r?\n", raw.decode("utf-8")):
                            data = "\n".join(line[5:].lstrip(" ") for line in frame.splitlines() if line.startswith("data:"))
                            if not data:
                                continue
                            require(not done, "SSE emitted data after [DONE]")
                            if data == "[DONE]":
                                done = True
                                continue
                            chunk = decode_json(data)
                            require("error" not in chunk, "Error payload inside SSE")
                            if "model" in chunk:
                                models.append(chunk["model"])
                            if chunk.get("usage") is not None:
                                usage = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                require(choice.get("index", 0) == 0, "Unexpected SSE choice index")
                                value = choice.get("delta", {}).get("content")
                                if isinstance(value, str):
                                    pieces.append(value)
                                if choice.get("finish_reason") is not None:
                                    finishes.append(choice["finish_reason"])
                        require(done and finishes and all(f in ("stop", "length") for f in finishes), "Missing [DONE] or valid SSE finish reason")
                        require(models and all(model in allowed_response_models for model in models), "Wrong/missing SSE model identity")
                        text = "".join(pieces)
                    require(isinstance(text, str) and text.strip(), "Completion contains no assistant text")
                    check_usage(usage, args.max_tokens)
                    report["cases"].append({"case": label, "passed": True, "assistantText": text, "usage": usage, "finishReasons": finishes,
                                            "sseDone": True if streaming else None, "fullResponse": label + ".response.txt"})
                    require(proc.poll() is None, "Owned server exited after generation")
                    persist()
            finally:
                stop_owned(proc, server_row)
                persist()
        report.update(complete=True, passed=True, completedUtc=utc())
    except Exception as error:
        report.update(passed=False, failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
