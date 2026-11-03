"""Pinned prebuilt native sentinels and native/HTTP client semantic parity.

No build, restore, arbitrary filter, model discovery or external endpoint is
accepted here. Source commits are operator declarations; hashes identify the
deployment, not independently attested compiler inputs or mapped modules.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import time
import xml.etree.ElementTree as ET

from .client_live import PROMPT, FOLLOW_UP, load_oracle
from .hardware import _runtime_evidence
from .processes import run_process, containment_failure_evidence
from .reports import file_hash, runtime_identity, source_identity
from .scenarios import ScenarioFailure, ScenarioUntested, _require, _server, backend_evidence, _json_loads

NATIVE_TESTS = (
    "InferenceWeb.Tests.CudaBackendTests.CudaPrefillGraphCache_CaptureReplayAndAbort_BehaveCorrectly",
    "InferenceWeb.Tests.CudaBackendTests.CudaQuantizedWeight_CanRunAfterHostCopyReleased",
)
NATIVE_BACKENDS = {"ggml_cpu", "ggml_cuda", "ggml_metal", "ggml_vulkan"}
RECEIPT_PREFIX = "[TensorSharp regression] "
SOURCE_FILES = {
    "native_unit": ("InferenceWeb.Tests/CudaBackendTests.cs", "InferenceWeb.Tests/GatedFacts.cs",
                    "TensorSharp.Backends.Cuda/CudaGraphCapture.cs", "TensorSharp.Backends.Cuda/CudaProfileCounters.cs"),
    "client_live": ("examples/tensorsharp-native-chat/Program.cs", "examples/tensorsharp-native-chat/TensorSharpChatClient.cs",
                    "examples/dotnet-local-chat/Program.cs", "eng/tests/dotnet-local-chat-e2e.py"),
}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(name + " requires exactly " + str(sorted(expected)))


def validate_profile(profile):
    kind = profile.get("scenario")
    if kind not in SOURCE_FILES:
        raise ValueError("unknown extended profile scenario")
    _keys(profile.get("models"), (), "models (primary is implicit; native_unit never loads it)")
    _keys(profile.get("parameters"), {"source_commit"}, "parameters")
    commit = profile["parameters"]["source_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("source_commit must be the declared full 40-hex build revision")
    roles = {"test_assembly": "InferenceWeb.Tests.dll"} if kind == "native_unit" else {
        "native_client": "TensorSharp.NativeChat.dll", "http_client": "TensorSharp.LocalChat.dll"}
    _keys(profile.get("fixtures"), roles, "fixtures")
    for role, filename in roles.items():
        fixture = profile["fixtures"][role]
        _keys(fixture, {"path", "sha256"}, "fixture " + role)
        if not isinstance(fixture["path"], str) or Path(fixture["path"]).name != filename:
            raise ValueError(role + " must be the fixed prebuilt " + filename)
        if not isinstance(fixture["sha256"], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", fixture["sha256"]):
            raise ValueError(role + " requires a SHA256 pin")
    expected = profile.get("expected")
    if kind == "native_unit":
        _keys(expected, {"required_tests", "max_cpu_fallback_calls"}, "native_unit expected")
        tests = expected["required_tests"]
        if (not isinstance(tests, list) or len(tests) != len(NATIVE_TESTS)
                or not all(isinstance(name, str) for name in tests) or set(tests) != set(NATIVE_TESTS)):
            raise ValueError("required_tests must be exactly the reviewed two CUDA device sentinels")
        if type(expected["max_cpu_fallback_calls"]) is not int or expected["max_cpu_fallback_calls"] != 0:
            raise ValueError("max_cpu_fallback_calls must be integer zero")
    else:
        _keys(expected, {"answers", "require_normal_finish"}, "client_live expected")
        if expected["answers"] != ["4", "12"] or expected["require_normal_finish"] is not True:
            raise ValueError("client_live requires fixed two-turn answers ['4','12'] and normal completion")


def required_options(profile):
    validate_profile(profile)
    return set()


def availability(case, context):
    profile = context.get("scenario_profile")
    try:
        validate_profile(profile or {})
    except ValueError as error:
        return False, str(error)
    if context.get("backend") != profile["backend"]:
        return False, "profile backend differs from the requested execution backend"
    if profile["scenario"] == "native_unit" and profile["backend"] != "cuda":
        return False, "native_unit currently proves direct CUDA synthetic device sentinels only; other native backends use native_ctest"
    if profile["scenario"] == "client_live":
        if profile["backend"] not in NATIVE_BACKENDS:
            return False, "native IChatClient example supports only explicit ggml_cpu/cuda/metal/vulkan backends"
        if not context.get("server_command") or isinstance(context["server_command"], str):
            return False, "provide a prebuilt local --server for the owned HTTP control"
        if context.get("context_tokens", 2048) != 2048:
            return False, "client_live fixed fixture requires context_tokens=2048"
        if not context.get("preparing_model") and not Path(context.get("model_path") or "").is_file():
            return False, "client_live requires the verified local primary model"
    if not shutil.which("dotnet"):
        return False, "dotnet is required; this adapter never builds or restores"
    for role, fixture in profile["fixtures"].items():
        path = Path(fixture["path"])
        if not path.is_file():
            return False, role + " pinned prebuilt assembly is missing"
        for suffix in (".runtimeconfig.json", ".deps.json"):
            if not path.with_suffix(suffix).is_file():
                return False, role + " deployment lacks adjacent " + suffix
    return True, "fixed prebuilt deployment present; runtime/backend/oracles still require execution"


def _deployment(context, result):
    profile = context["scenario_profile"]
    result["evidence"]["deployment"] = evidence = {
        "declared_build_source_commit": profile["parameters"]["source_commit"],
        "harness_source": source_identity(context["repo"]), "assemblies": [], "sidecars": [],
        "source_evidence": [],
        "limitation": "Build source revision is operator-declared. File hashes identify deployment candidates, not mapped modules or verified compiler inputs."}
    directories = set()
    for role, fixture in profile["fixtures"].items():
        path = Path(fixture["path"])
        size = path.stat().st_size
        _require(result, role + " bounded assembly", 0 < size <= 512 * 1024 ** 2, "1..512MiB", size)
        digest = file_hash(path)
        _require(result, role + " immutable assembly identity", digest.lower() == fixture["sha256"].lower(), fixture["sha256"], digest)
        evidence["assemblies"].append({"role": role, "path": str(path), "sha256": digest, "size_bytes": size})
        directories.add(str(path.parent))
        for suffix in (".runtimeconfig.json", ".deps.json"):
            sidecar = path.with_suffix(suffix)
            if not sidecar.is_file() or not 0 < sidecar.stat().st_size <= 8 * 1024 ** 2:
                raise ScenarioFailure("missing or unbounded deployment sidecar: " + str(sidecar))
            evidence["sidecars"].append({"path": str(sidecar), "sha256": file_hash(sidecar)})
    runtime = _runtime_evidence(sorted(directories))
    candidates = [Path(path) for paths in runtime["files"].values() for path in paths]
    if len(candidates) > 512 or any(path.stat().st_size > 512 * 1024 ** 2 for path in candidates) or sum(path.stat().st_size for path in candidates) > 2 * 1024 ** 3:
        raise ScenarioFailure("deployment candidate inventory exceeds 512 files / 512MiB each / 2GiB total")
    evidence["runtime_candidates"] = runtime_identity({"runtime": runtime})
    for relative in SOURCE_FILES[profile["scenario"]]:
        path = Path(context["repo"]) / relative
        if path.is_file():
            evidence["source_evidence"].append({"path": relative, "sha256": file_hash(path)})
    return {role: ["dotnet", fixture["path"]] for role, fixture in profile["fixtures"].items()}


def _healthy(item, result, stage):
    result["processes"].append({key: value for key, value in item.items() if key not in {"stdout", "stderr"}})
    _require(result, stage + " complete owned process", item.get("exit_code") == 0 and not item.get("timed_out")
             and not item.get("cancelled") and not item.get("output_truncated")
             and containment_failure_evidence(item) is None and item.get("teardown", {}).get("verified") is True,
             "exit0; complete output; verified containment", {key: item.get(key) for key in
                 ("exit_code", "timed_out", "cancelled", "output_truncated", "teardown")})


def read_native_trx(path):
    """Use TestMethod FQNs, not display-name substrings or process exit alone."""
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024 ** 2:
        raise ScenarioFailure("native sentinel TRX is missing, empty or exceeds 4MiB")
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ScenarioFailure("TRX document types/entities are forbidden")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ScenarioFailure("malformed native sentinel TRX") from error
    definitions = {}
    for unit in root.findall(".//{*}TestDefinitions/{*}UnitTest"):
        method = unit.find("{*}TestMethod")
        if method is None or unit.get("id") in definitions:
            raise ScenarioFailure("TRX test definitions are missing or duplicated")
        definitions[unit.get("id")] = method.get("className", "").split(",", 1)[0] + "." + method.get("name", "")
    rows, output = [], []
    for node in root.findall(".//{*}Results/{*}UnitTestResult"):
        fqn = definitions.get(node.get("testId"))
        if fqn not in NATIVE_TESTS or any(row["id"] == fqn for row in rows):
            raise ScenarioFailure("TRX includes unexpected or duplicate native tests")
        outcome = node.get("outcome")
        status = "pass" if outcome == "Passed" else "skip" if outcome == "NotExecuted" else "fail"
        rows.append({"id": fqn, "status": status, "outcome": outcome, "duration": node.get("duration"),
                     "message": "\n".join(node.itertext())[:4096]})
    if {row["id"] for row in rows} != set(NATIVE_TESTS):
        raise ScenarioFailure("TRX did not report both required CUDA sentinels")
    counter = root.find(".//{*}Counters")
    if counter is None:
        raise ScenarioFailure("TRX counters are missing")
    try:
        counts = {key: int(counter.get(key, "0")) for key in ("total", "executed", "passed", "failed", "notExecuted", "error", "timeout", "aborted", "inconclusive", "notRunnable")}
    except ValueError as error:
        raise ScenarioFailure("invalid TRX counters") from error
    if any(value < 0 for value in counts.values()) or counts["total"] != len(rows) or counts["passed"] != sum(r["status"] == "pass" for r in rows):
        raise ScenarioFailure("TRX counters contradict per-test outcomes")
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] in {"StdOut", "StdErr"} and node.text:
            output.append(node.text)
    return rows, counts, "\n".join(output)


def cuda_device_proof(text):
    """Require actual graph launches and a complete, empty fallback counter block."""
    if re.search(r"(?i)(?:element-by-element CPU fallback|slow CPU fallback|fall(?:ing)?\s*back\s+to\s+(?:the\s+)?CPU)", text):
        raise ScenarioFailure("CUDA sentinel explicitly used a CPU fallback")
    blocks = re.findall(r"\[cuda-profile\] CPU fallback ops:\s*\r?\n(.*?)\[cuda-profile\] host<->device syncs:", text, re.S)
    calls = []
    for block in blocks:
        for line in block.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*(\S+)\s+(\d+)\s+calls\s+[0-9.]+\s+ms\s*", line)
            if match is None:
                raise ScenarioFailure("unparseable CUDA fallback counter row")
            calls.append({"op": match[1], "calls": int(match[2])})
    if any(row["calls"] > 0 for row in calls):
        raise ScenarioFailure("CUDA sentinel fallback counter is nonzero: " + str(calls))
    captured = bool(re.search(r"\[cuda-graph\] captured test\|shape \(\d+ owned blocks\)", text))
    replayed = bool(re.search(r"\[cuda-graph\] replay test\|shape \(total [1-9]\d*\)", text))
    proof = {"captured": captured, "replayed": replayed, "fallback_counter_blocks": len(blocks), "fallback_calls": calls,
             "scope": "Two direct CUDA synthetic sentinels; graph launch plus host-copy-released quantized matmul/gather. No model/backend-wide coverage."}
    if not captured or not replayed or not blocks:
        raise ScenarioUntested("required CUDA graph/fallback telemetry absent; a passing test exit is insufficient")
    return proof


def _native_unit(context, result, commands):
    root = Path(context["artifact_dir"]) / "native-sentinel-results"
    root.mkdir(parents=True, exist_ok=False)
    trx = root / "sentinels.trx"
    test_filter = "(" + "|".join("FullyQualifiedName=" + name for name in NATIVE_TESTS) + ")&Category!=Bench&Requires!=Models"
    argv = ["dotnet", "vstest", commands["test_assembly"][-1], "--TestCaseFilter:" + test_filter,
            "--Logger:trx;LogFileName=sentinels.trx", "--ResultsDirectory:" + str(root), "--Logger:console;verbosity=detailed"]
    env = dict(context.get("env") or {})
    env.update(TS_TEST_GGML_BACKEND="cpu", TS_CUDA_PROFILE="1", TS_CUDA_PREFILL_GRAPH_LOG="1")
    result["inputs"].update(test_filter=test_filter, model_free=True, environment_overrides=env)
    item = run_process(argv, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="native-cuda-sentinels",
                       timeout_seconds=context.get("timeout_seconds", 600), env=env)
    # Record per-test outcomes even when VSTest exits nonzero; never discard failure TRX.
    try:
        rows, counts, output = read_native_trx(trx)
        result["evidence"]["native_tests"] = {"path": str(trx), "sha256": file_hash(trx), "tests": rows, "counts": counts}
    except (ScenarioFailure, OSError):
        result["processes"].append({key: value for key, value in item.items() if key not in {"stdout", "stderr"}})
        raise
    _healthy(item, result, "native CUDA sentinels")
    _require(result, "native CUDA tests have no failed outcomes", not any(row["status"] == "fail" for row in rows)
             and not any(counts[key] for key in ("failed", "error", "timeout", "aborted", "inconclusive", "notRunnable")),
             "no failed tests or failure counters", {"tests": rows, "counts": counts})
    if any(row["status"] == "skip" for row in rows):
        raise ScenarioUntested("CUDA sentinel tests were skipped; required device qualification did not execute")
    _require(result, "both exact CUDA sentinel tests passed", counts["executed"] == counts["passed"] == 2
             and not any(counts[key] for key in ("failed", "error", "timeout", "aborted", "inconclusive", "notRunnable", "notExecuted")),
             "2 executed and passed; all other counters zero", counts)
    result["evidence"]["native_backend"] = cuda_device_proof(item.get("stdout", "") + "\n" + item.get("stderr", "") + "\n" + output)
    _require(result, "positive CUDA graph launch and zero CPU fallback counter", True)


def validate_native_client(item, *, mode, model, backend, oracle):
    text = item.get("stdout", "") + "\n" + item.get("stderr", "")
    observed = backend_evidence(text, backend)
    if not observed["verified"]:
        raise ScenarioFailure("native client lacks matching allocated backend evidence: " + str(observed))
    receipts = []
    for line in item.get("stderr", "").splitlines():
        if line.startswith(RECEIPT_PREFIX):
            receipts.append(_json_loads(line[len(RECEIPT_PREFIX):]))
    if not receipts:
        raise ScenarioUntested("native client lacks --regression-json receipts; rebuild this example without native builds")
    if len(receipts) != 2:
        raise ScenarioFailure("native client must emit exactly two completed-turn receipts")
    for turn, (receipt, expected) in enumerate(zip(receipts, ("4", "12")), 1):
        if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
                or type(receipt.get("turn")) is not int or receipt["turn"] != turn or receipt.get("mode") != mode
                or receipt.get("model") != model or receipt.get("backend") != backend
                or not isinstance(receipt.get("architecture"), str) or not receipt["architecture"]):
            raise ScenarioFailure("native turn identity/order/backend mismatch")
        if str(receipt.get("finish", "")).lower() != "stop" or receipt.get("native_finish") not in {"eos", "stop_sequence", "stop"}:
            raise ScenarioFailure("native client did not finish normally")
        prompt, generated, cached = (receipt.get(key) for key in ("prompt_tokens", "generated_tokens", "kv_cache_reused_tokens"))
        if (type(prompt) is not int or not 0 < prompt <= 2048 or type(generated) is not int or not 0 < generated <= 64
                or type(cached) is not int or not 0 <= cached <= prompt):
            raise ScenarioFailure("native client token usage invalid or unbounded")
        if not isinstance(receipt.get("text"), str):
            raise ScenarioFailure("native client receipt text missing")
        try:
            oracle.answer_text(receipt["text"], expected)
        except AssertionError as error:
            raise ScenarioFailure(str(error)) from error
    if receipts[0]["architecture"] != receipts[1]["architecture"]:
        raise ScenarioFailure("native model architecture changed across turns")
    return receipts


def _client_live(context, result, commands):
    deadline = time.monotonic() + context.get("timeout_seconds", 600)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise ScenarioFailure("client_live total scenario deadline exceeded")
        return value
    result["inputs"].update(prompt=PROMPT, follow_up=FOLLOW_UP, expected_answers=["4", "12"], native_seed=1,
                            native_temperature=0, native_max_tokens=64, context_tokens=2048)
    evidence = result["evidence"]["native_client"] = {"invocations": [],
        "limitations": ["Semantic parity, not exact tokens/logits: native seed1 and HTTP seed42 are distinct APIs.",
                        "No native live cancellation, abandonment, deadline recovery or cache-hit qualification.",
                        "Receipts report the full native response.Text; unrelated startup logs cannot satisfy the answer oracle."]}
    oracle = load_oracle(context["repo"])
    for mode in ("stream", "response"):
        argv = commands["native_client"] + ["--model", str(context["model_path"]), "--backend", context["backend"],
                "--prompt", PROMPT, "--follow-up", FOLLOW_UP, "--mode", mode, "--max-tokens", "64",
                "--context-tokens", "2048", "--timeout-seconds", str(max(1, min(65536, int(remaining())))), "--regression-json"]
        item = run_process(argv, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="native-client-" + mode,
                           timeout_seconds=remaining(), env=context.get("env"))
        _healthy(item, result, "native IChatClient " + mode)
        receipts = validate_native_client(item, mode=mode, model=Path(context["model_path"]).name,
                                          backend=context["backend"], oracle=oracle)
        evidence["invocations"].append({"mode": mode, "receipts": receipts})
        _require(result, "native " + mode + " strict complete two-turn answer/identity/usage", True, ["4", "12"], [r["text"] for r in receipts])
    # The native children are gone before the server loads the same primary model.
    server_context = dict(context, http_client_command=commands["http_client"], timeout_seconds=remaining())
    _server({"id": "client_http_live"}, server_context, result)
    invocations = result["evidence"].get("http_client", {}).get("invocations", [])
    _require(result, "native/HTTP SDK semantic conversation parity", len(invocations) == 4 and all(
        row.get("passed") and [answer.strip().rstrip(".") for answer in row.get("answers", [])] == ["4", "12"] for row in invocations),
        "all 2 native modes and all 4 HTTP SDK modes produce full answers4/12", invocations)
    # The narrower HTTP adapter's limitations remain scoped to its own calls.
    evidence["counts"] = {"native_processes": 2, "native_turns": 4, "http_sdk_processes": 4, "http_sdk_turns": 8}


def run(case, context, result):
    validate_profile(context["scenario_profile"])
    ready, reason = availability(case, context)
    if not ready:
        raise ScenarioUntested(reason)
    commands = _deployment(context, result)
    if context["scenario_profile"]["scenario"] == "native_unit":
        _native_unit(context, result, commands)
    else:
        _client_live(context, result, commands)
