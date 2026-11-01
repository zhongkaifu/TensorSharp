"""Typed, opt-in scaling/learned-head/text-LoRA execution profiles.

No caller-supplied shell command, model download, build or node discovery occurs.
Unavailable prerequisites never qualify a requested mechanism. Device placement,
actual learned work and the LoRA oracle are checked independently of process exit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import socket
import time
import uuid

from .processes import ManagedProcess, run_process

SCENARIOS = frozenset({"multi_gpu", "distributed", "speculation", "lora"})
_GATES = {"max_absolute_error", "max_rmse", "max_relative_l2", "min_cosine", "require_same_argmax"}
_SOURCES = {
    "multi_gpu": ("eng/validation/DeepSeek41DistributedProbe/Program.cs", "eng/validation/compare-deepseek41-distributed.py",
                  "eng/validation/native_release_evidence.py"),
    "distributed": ("eng/validation/DeepSeek41DistributedProbe/Program.cs", "eng/validation/compare-deepseek41-distributed.py"),
    "speculation": ("benchmarks/AgentTurnBench/Program.cs", "benchmarks/AgentTurnBench/compare.py",
                    "eng/validation/validate-speculative-rows.py"),
    "lora": ("TensorSharp.Server/Endpoints/LoraEndpoints.cs", "TensorSharp.Server/Hosting/TextLoraHostOptions.cs",
             "TensorSharp.Models/ModelBase.cs", "InferenceWeb.Tests/TextLoraModelParityTests.cs"),
}


def _keys(value, required, optional=(), label="object"):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(label + " has missing or unknown fields")


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")


def _number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be finite in {low}..{high}")


def _gates(expected, extra=()):
    _keys(expected, _GATES | set(extra), label="expected")
    for key in ("max_absolute_error", "max_rmse"):
        _number(expected[key], 0, 1, key)
    _number(expected["max_relative_l2"], 0, 0.001, "max_relative_l2")
    _number(expected["min_cosine"], 0.999999, 1, "min_cosine")
    if expected["require_same_argmax"] is not True:
        raise ValueError("require_same_argmax must be true")


def validate_profile(profile):
    """Pure schema check: return None or raise ValueError; no execution/I/O."""
    _keys(profile, {"id", "scenario", "preset", "backend", "parameters", "expected", "resources"},
          {"models", "fixtures", "profile_source"}, "scaling profile")
    kind = profile["scenario"]
    if kind not in SCENARIOS:
        raise ValueError("unknown scaling scenario")
    params, expected = profile["parameters"], profile["expected"]
    roles, fixtures = profile.get("models", {}), profile.get("fixtures", {})
    if not isinstance(roles, dict) or not isinstance(fixtures, dict):
        raise ValueError("models and fixtures must be objects")
    if "primary" in roles and roles["primary"] != profile["preset"]:
        raise ValueError("explicit primary role must match profile preset")
    roles = {key: value for key, value in roles.items() if key != "primary"}
    if kind == "distributed":
        _keys(fixtures, {"probe", "tokens", "known_hosts"}, label="distributed fixtures")
        if roles:
            raise ValueError("distributed uses the implicit primary model only")
        from .scaling_remote import validate_remote_profile
        validate_remote_profile(profile)
        _integer(params["chunk"], 1, 512, "chunk")
        _integer(params["decode_count"], 1, 64, "decode_count")
        _gates(expected, {"weights_id"})
        return
    if kind == "multi_gpu":
        _keys(params, {"mode", "devices", "chunk", "decode_count"}, label="multi_gpu parameters")
        if params["mode"] not in {"tp", "layer"}:
            raise ValueError("mode must be tp or layer")
        devices = params["devices"]
        if not isinstance(devices, list) or not 2 <= len(devices) <= 8 or any(type(x) is not int or x < 0 for x in devices) or len(set(devices)) != len(devices):
            raise ValueError("devices must name 2..8 distinct nonnegative GPU indices")
        _integer(params["chunk"], 1, 512, "chunk")
        _integer(params["decode_count"], 1, 64, "decode_count")
        _keys(fixtures, {"probe", "tokens"}, label="multi_gpu fixtures")
        _gates(expected)
        if profile["backend"] != "ggml_cuda":
            raise ValueError("multi_gpu currently requires source-attested ggml_cuda placement")
    elif kind == "speculation":
        _keys(params, {"head_kind", "draft_tokens"}, label="speculation parameters")
        if params["head_kind"] not in {"dflash", "pertoken", "block"}:
            raise ValueError("head_kind must match observed dflash, pertoken or block")
        _integer(params["draft_tokens"], 1, 16, "draft_tokens")
        _keys(expected, {"require_token_parity", "require_acceptance"}, label="speculation expected")
        if any(value is not True for value in expected.values()):
            raise ValueError("learned speculation requires exact token parity and active acceptance")
        _keys(fixtures, set(), label="speculation fixtures")
        if set(roles) - {"draft"}:
            raise ValueError("speculation accepts only the optional verified draft model role")
    else:
        _keys(params, {"adapter_name", "scale", "max_tokens"}, label="lora parameters")
        if not isinstance(params["adapter_name"], str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", params["adapter_name"]):
            raise ValueError("adapter_name must be a safe name of at most64 characters")
        _number(params["scale"], -8, 8, "scale")
        if params["scale"] == 0:
            raise ValueError("LoRA scale must be nonzero")
        _integer(params["max_tokens"], 1, 128, "max_tokens")
        _keys(fixtures, {"adapter", "prompt"}, label="lora fixtures")
        _keys(expected, {"base_text", "adapted_text"}, label="lora expected")
        if any(not isinstance(x, str) or not x or len(x) > 16384 for x in expected.values()) or expected["base_text"] == expected["adapted_text"]:
            raise ValueError("LoRA requires distinct nonempty bounded exact base/adapted text oracles")
    if kind != "speculation" and roles:
        raise ValueError(kind + " uses the implicit primary model only")


def required_options(profile):
    """External drafters must use the explicitly pinned draft role."""
    return set()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, limit=4 * 1024 * 1024):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= limit:
        raise ValueError("missing, linked or oversized JSON evidence: " + str(path))
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=bad, object_pairs_hook=pairs)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON metric")
        if isinstance(item, dict):
            for x in item.values(): finite(x)
        elif isinstance(item, list):
            for x in item: finite(x)
    finite(value)
    return value


def _fixture(profile, name, limit=512 * 1024 * 1024):
    descriptor = profile.get("fixtures", {}).get(name)
    _keys(descriptor, {"path", "sha256"}, {"bytes"}, "fixture " + name)
    if not re.fullmatch(r"[a-f0-9]{64}", descriptor["sha256"]):
        raise ValueError("fixture requires lowercase SHA256")
    path = Path(descriptor["path"]).resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("missing or oversized " + name + " fixture")
    return path


def _launcher(argv, expected):
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError("provide the prebuilt " + expected + " binary")
    dotnet = Path(argv[0]).name.lower() in {"dotnet", "dotnet.exe"}
    if len(argv) != (2 if dotnet else 1):
        raise ValueError("binary command must not include additional arguments")
    binary = Path(argv[-1])
    if binary.name.lower() not in {expected.lower(), expected.lower() + ".exe", expected.lower() + ".dll"} or (dotnet and binary.suffix.lower() != ".dll"):
        raise ValueError("profile requires fixed entrypoint " + expected)
    if not binary.is_file():
        raise ValueError("prebuilt entrypoint is missing: " + str(binary))
    executable = shutil.which(argv[0]) if dotnet else str(binary.resolve())
    if not executable:
        raise ValueError("dotnet toolchain is unavailable")
    return [executable, str(binary.resolve())] if dotnet else [executable]


def _probe_command(profile):
    binary = _fixture(profile, "probe")
    return _launcher(["dotnet", str(binary)] if binary.suffix.lower() == ".dll" else [str(binary)], "DeepSeek41DistributedProbe")


def _module(repo, path):
    spec = importlib.util.spec_from_file_location("regression_scaling_" + Path(path).stem.replace("-", "_"), Path(repo) / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def availability(case, context):
    try:
        profile = context.get("scenario_profile")
        if not profile:
            return False, "explicit typed scenario profile is required"
        validate_profile(profile)
        kind = profile["scenario"]
        if (case.get("adapter") or case.get("kind") or case["id"]) != kind or context.get("backend") != profile["backend"]:
            return False, "profile scenario/backend differs from selected case"
        if context.get("context_tokens", 2048) != 2048:
            return False, "scaling profiles require the unchanged2048-token context contract"
        if any(not (Path(context["repo"]) / source).is_file() for source in _SOURCES[kind]):
            return False, "selected checkout is missing the fixed executor/validator sources"
        if not context.get("preparing_model"):
            primary = context.get("profile_models", {}).get("primary", {})
            if not primary.get("identity") or not Path(primary.get("model_path", "")).is_file():
                return False, "verified primary model identity and local weights are required"
            if any(not isinstance(item, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", ""))) for item in primary["identity"]):
                return False, "primary identity must include a computed SHA256 for every selected file"
            for name in profile.get("models", {}):
                role = context.get("profile_models", {}).get(name, {})
                if not role.get("identity") or not Path(role.get("model_path", "")).is_file():
                    return False, "verified model role is unavailable: " + name
        if kind in {"multi_gpu", "distributed"}:
            _probe_command(profile)
            tokens = _json(_fixture(profile, "tokens", 64 * 1024), 64 * 1024)
            if not isinstance(tokens, list) or not profile["parameters"]["decode_count"] + 2 <= len(tokens) <= 1024 or any(type(x) is not int or not 0 <= x <= 2**31-1 for x in tokens):
                return False, "tokens must contain decode_count+2..1024 explicit nonnegative int32 history IDs"
            if kind == "distributed":
                from .scaling_remote import availability as remote_available
                return remote_available(profile, context)
            available = {gpu["index"] for gpu in context.get("hardware", {}).get("gpus", [])}
            if not set(profile["parameters"]["devices"]) <= available:
                return False, "requested local GPU indices are not present; no device substitution"
        elif kind == "speculation":
            _launcher(context.get("benchmark_command"), "AgentTurnBench")
        else:
            family = str(context.get("model_family") or context.get("family") or
                         context.get("profile_models", {}).get("primary", {}).get("preset", {}).get("family", "")).replace("_", "-").lower()
            if family not in {"mistral3", "hunyuan-dense", "nemotron-h", "nemotron-h-moe", "nemotron-h-omni"}:
                return False, "text LoRA is source-admitted only for mistral3, hunyuan-dense and nemotron_h families"
            if context["backend"] not in {"cpu", "ggml_cpu", "ggml_cuda", "cuda"}:
                return False, "text LoRA supports cpu, ggml_cpu, ggml_cuda and cuda; no backend fallback"
            _launcher(context.get("server_command"), "TensorSharp.Server.Host")
            if _fixture(profile, "adapter").suffix.lower() != ".safetensors":
                return False, "text LoRA currently requires a verified safetensors adapter"
            prompt = _fixture(profile, "prompt", 16 * 1024).read_text(encoding="utf-8")
            if not prompt.strip():
                return False, "LoRA prompt fixture is empty"
        return True, "typed executor prerequisites present; runtime assertions still required"
    except (KeyError, OSError, ValueError, TypeError, ImportError) as error:
        return False, str(error)


def _require(result, name, condition, expected=None, actual=None):
    from .scenarios import ScenarioFailure
    result["assertions"].append({"name": name, "passed": bool(condition), "expected": expected, "actual": actual})
    if not condition:
        raise ScenarioFailure(name)


def _process(command, context, result, root, stage, env, check_exit=True):
    item = run_process(command, cwd=context["repo"], artifact_dir=root, stage=stage,
                       timeout_seconds=context.get("timeout_seconds", 600), env=env)
    result["processes"].append(item)
    _require(result, stage + " verified process containment", item.get("teardown", {}).get("verified") is True, True, item.get("teardown"))
    _require(result, stage + " no timeout or cancellation", not item.get("timed_out") and not item.get("cancelled"))
    _require(result, stage + " complete capture", item.get("output_truncated") is False)
    text = item.get("stdout", "") + "\n" + item.get("stderr", "")
    _require(result, stage + " no OOM", re.search(r"out.of.memory|CUDA_ERROR_OUT_OF_MEMORY|std::bad_alloc", text, re.I) is None)
    if check_exit:
        _require(result, stage + " process success", item.get("exit_code") == 0, 0, item.get("exit_code"))
    return item, text


def _weights_id(context):
    identity = context["profile_models"]["primary"]["identity"]
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _model_snapshot(context):
    """Cheap mutation detection complements the preparation manifest's full hashes."""
    snapshot = {}
    for role in context.get("profile_models", {}).values():
        for item in role.get("identity", []):
            path = Path(item["path"])
            stat = path.stat()
            snapshot[str(path.resolve())] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return snapshot


def _probe_report(report, profile, result, label, weights_id):
    _require(result, label + " supported probe schema", type(report.get("schema_version")) is int and report["schema_version"] == 1)
    tokens = _json(_fixture(profile, "tokens"))
    params = profile["parameters"]
    prefix = tokens[:-params["decode_count"]]
    tail = tokens[-params["decode_count"]:]
    expected = [("reset", []), ("prefill", prefix)] + [("decode", [token]) for token in tail]
    expected += [("reset", [])] + [("chunk", prefix[i:i + params["chunk"]]) for i in range(0, len(prefix), params["chunk"])]
    expected += [("refill", [token]) for token in tail] + [("reset", []), ("reset-prefill", prefix)]
    steps = report.get("steps", [])
    _require(result, label + " complete fixed operation stream", [(step.get("operation"), step.get("tokens")) for step in steps] == expected,
             expected, [(step.get("operation"), step.get("tokens")) for step in steps])
    _require(result, label + " weights and token source identity", report.get("weights_id") == weights_id
             and report.get("tokens_sha256") == profile["fixtures"]["tokens"]["sha256"] and report.get("tokens") == tokens)
    vocab = report.get("vocab_size")
    _require(result, label + " bounded vocabulary", type(vocab) is int and 1 <= vocab <= 1048576)
    _require(result, label + " history IDs within vocabulary", all(token < vocab for token in tokens))
    for step in steps:
        if step["operation"] == "reset":
            _require(result, label + " reset has no vector", step.get("logits") is None)
        else:
            _require(result, label + " forward vector dimensions and time", step.get("count") == vocab
                     and isinstance(step.get("logits"), str) and bool(step["logits"])
                     and type(step.get("seconds")) in (int, float) and math.isfinite(step["seconds"]) and step["seconds"] > 0)


def _probe(profile, context, result, root, label, devices, weights_id):
    params = profile["parameters"]
    output = root / label
    command = _probe_command(profile) + ["--model", context["profile_models"]["primary"]["model_path"],
        "--tokens", str(_fixture(profile, "tokens")), "--weights-id", weights_id, "--backend", context["backend"],
        "--output", str(output), "--chunk", str(params["chunk"]), "--decode-count", str(params["decode_count"])]
    if len(devices) > 1:
        command += ["--tp" if params["mode"] == "tp" else "--layer-split", str(len(devices))]
    env = dict(context.get("env") or {}, CUDA_VISIBLE_DEVICES=",".join(map(str, devices)), MAX_CONTEXT="2048", KV_CACHE_DTYPE="f16")
    before = _model_snapshot(context)
    _require(result, label + " model snapshot matches scenario start", before == result["evidence"].get("model_file_snapshot", before))
    item, text = _process(command, context, result, root, label, env)
    _require(result, label + " prepared model files unchanged", _model_snapshot(context) == before)
    path = output / "probe.json"
    report = _json(path)
    expected_topology = {"mode": "layer_split" if len(devices) > 1 and params["mode"] == "layer" else "tensor_parallel",
                         "local_degree": len(devices), "global_degree": len(devices), "rank_offset": 0, "node_count": 1}
    _require(result, label + " completed and disposed", report.get("status") == "completed" and report.get("disposed") is True)
    _require(result, label + " exact topology", report.get("topology") == expected_topology, expected_topology, report.get("topology"))
    _probe_report(report, profile, result, label, weights_id)
    placement = _module(context["repo"], "eng/validation/native_release_evidence.py").placement_evidence(
        text, profile["preset"], {"gpus": len(devices), "layer_split": len(devices) > 1 and params["mode"] == "layer", "kv": "f16"})
    result["evidence"][label + "_placement"] = placement
    _require(result, label + " effective backend/device placement", placement.get("status") == "passed", "observed completed placement", placement)
    return path


def _compare(reference, candidate, profile, context, result):
    gates = {key: profile["expected"][key] for key in _GATES}
    proof = _module(context["repo"], "eng/validation/compare-deepseek41-distributed.py").compare(reference, candidate, **gates)
    result["evidence"]["numerical_comparison"] = proof
    _require(result, "all teacher-forced prefill/decode/reset/refill logits agree", proof["status"] == "passed", gates, proof)


def _learned(profile, context, result, root):
    from .scenarios import backend_evidence
    params = profile["parameters"]
    rows_path = root / "learned-rows.json"
    command = _launcher(context.get("benchmark_command"), "AgentTurnBench") + ["--model", context["profile_models"]["primary"]["model_path"],
        "--backend", context["backend"], "--scenarios", "spec", "--spec-only", "--spec-candidates", "auto",
        "--spec-new", "64", "--spec-file", "128", "--warmup", "0", "--measure-passes", "1", "--out", str(rows_path)]
    draft = context.get("profile_models", {}).get("draft")
    if draft:
        command += ["--draft-model", draft["model_path"]]
    env = {key: value for key, value in (context.get("env") or {}).items() if not re.search(r"SPEC|DRAFT|DFLASH|DSPARK|NGRAM|MTP", key, re.I)}
    env.update(MAX_CONTEXT="2048", TS_SPEC_DRAFT=str(params["draft_tokens"]), TS_SCHED_NUM_BLOCKS="16", TS_SCHED_BLOCK_SIZE="128")
    before = _model_snapshot(context)
    item, text = _process(command, context, result, root, "learned", env, check_exit=False)
    _require(result, "learned model and draft files unchanged", _model_snapshot(context) == before)
    observed = backend_evidence(text, context["backend"])
    _require(result, "learned actual backend", observed["verified"], context["backend"], observed)
    heads = re.findall(r"\[agent-turn-bench\] loaded [^\r\n]+\bdrafter=(\w+)", text)
    _require(result, "actual learned head kind", heads == [params["head_kind"]], params["head_kind"], heads)
    _require(result, "drafter attachment did not degrade", re.search(r"draft head NOT attached|not attached:|speculation.*(?:refused|disabled)", text, re.I) is None)
    _json(rows_path)
    rows = _module(context["repo"], "benchmarks/AgentTurnBench/compare.py").load_rows(rows_path)
    _require(result, "exact plain and learned row inventory", set(rows) == {("spec", "plain greedy"), ("spec", "draft head (auto)")})
    plain, learned = rows[("spec", "plain greedy")], rows[("spec", "draft head (auto)")]
    for label, row in (("plain", plain), ("learned", learned)):
        _require(result, label + " bounded complete inference", isinstance(row.get("Tokens"), list) and 1 <= len(row["Tokens"]) <= 64
                 and row.get("Finish") in {"eos", "max_tokens"} and (row["Finish"] != "max_tokens" or row.get("OutTokens") == 64)
                 and type(row.get("Prompt")) is int and 0 < row["Prompt"] <= 1984)
        for key in ("Drafted", "Accepted", "VerifySteps", "PlainSteps", "Rollbacks"):
            _require(result, label + " valid " + key, type(row.get(key)) is int and row[key] >= 0)
        _require(result, label + " internally consistent work", row["Accepted"] <= row["Drafted"] and row["Rollbacks"] <= row["VerifySteps"])
        _require(result, label + " positive executed timing", row.get("TtftMs", 0) > 0
                 and (row.get("Fresh", 0) == 0 or row.get("PrefillTps", 0) > 0)
                 and (row.get("OutTokens", 0) <= 1 or row.get("DecodeTps", 0) > 0))
    _require(result, "plain baseline is nonspeculative", all(plain[key] == 0 for key in ("Drafted", "Accepted", "VerifySteps", "Rollbacks")))
    _require(result, "plain/learned prompt and cache identity", all(plain.get(key) == learned.get(key) for key in ("Prompt", "Reused", "Fresh")))
    proof = _module(context["repo"], "eng/validation/validate-speculative-rows.py").validate(list(rows.values()))
    result["evidence"].update(learned_parity=proof, rows_path=str(rows_path), rows_sha256=_hash(rows_path), head_kind=heads[0])
    result["inputs"]["sampling"] = {"temperature": 0, "top_k": 0, "top_p": 1, "repetition_penalty": 1,
                                    "seed": -1, "source": "AgentTurnBench SamplingConfig.Greedy"}
    _require(result, "exact greedy learned token and finish parity", proof.get("greedy_tokens_equal") and proof.get("finish_equal"), True, proof)
    _require(result, "learned process success", item.get("exit_code") == 0, 0, item.get("exit_code"))
    if not all(learned[key] > 0 for key in ("Drafted", "Accepted", "VerifySteps")):
        from .scenarios import ScenarioUntested
        raise ScenarioUntested("learned head did not exercise drafting, acceptance and verification; parity alone does not qualify it")
    _require(result, "active learned speculation oracle", proof["status"] == "passed", "passed", proof)


def _lora_requests(http, model, profile, result, adapter_source, adapter_dir):
    from .scenarios import validate_chat, _expect_http_error
    params, expected = profile["parameters"], profile["expected"]
    prompt = _fixture(profile, "prompt", 16384).read_text(encoding="utf-8")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
               "seed": 42, "max_tokens": params["max_tokens"], "think": False, "stream": False}
    _require(result, "owned server starts with no LoRA", http.json("/v1/loras").get("data") == [])
    def chat(body):
        return validate_chat(http.json("/v1/chat/completions", body), expected_model=model, max_tokens=params["max_tokens"])
    before = chat(payload)
    _require(result, "exact baseline oracle", before == expected["base_text"], expected["base_text"], before)
    destination = adapter_dir / "verified-adapter.safetensors"
    shutil.copyfile(adapter_source, destination)
    _require(result, "staged adapter SHA256", _hash(destination) == profile["fixtures"]["adapter"]["sha256"])
    name = params["adapter_name"]
    loaded = http.json("/v1/loras", {"name": name, "path": destination.name, "scale": params["scale"]})
    _require(result, "adapter actually loaded modules", loaded.get("id") == name and type(loaded.get("modules")) is int and loaded["modules"] > 0)
    listed = http.json("/v1/loras").get("data", [])
    _require(result, "loaded adapter is applicable", len(listed) == 1 and listed[0].get("id") == name and listed[0].get("applicable") is True)
    adapted = chat(dict(payload, lora=name))
    _require(result, "exact adapted oracle differs from baseline", adapted == expected["adapted_text"] and adapted != before,
             expected["adapted_text"], adapted)
    _require(result, "per-request baseline remains unmodified", chat(payload) == before)
    _expect_http_error(http, result, "/v1/loras", {"name": "invalid-scale", "path": destination.name, "scale": 1e60}, status=400, param="scale")
    removed = http.json("/v1/loras/" + name, method="DELETE")
    _require(result, "adapter unload acknowledged", removed.get("removed") is True)
    _require(result, "no adapter remains after unload", http.json("/v1/loras").get("data") == [])
    after = chat(payload)
    _require(result, "unload restores exact baseline", after == before, before, after)
    _expect_http_error(http, result, "/v1/chat/completions", dict(payload, lora=name), status=400)
    result["evidence"]["lora_oracle"] = {"before": before, "adapted": adapted, "after": after,
        "scope": "Explicit text adapter effect and request isolation; not image/video LoRA or merged-weight numerical parity."}


def _lora(profile, context, result, root):
    from .scenarios import _Http, _base_args, _hosted_response_model, _attest_offload
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = root / "loras"
    directory.mkdir()
    command = _launcher(context.get("server_command"), "TensorSharp.Server.Host") + _base_args(context, "server_chat")
    command += ["--host", "127.0.0.1", "--port", str(port), "--no-webui", "--no-multi-agent", "--loras-dir", str(directory)]
    process = ManagedProcess(command, cwd=context["repo"], artifact_dir=root, stage="lora-server", env=context.get("env"))
    http = _Http(port, time.monotonic() + context.get("timeout_seconds", 600), result)
    try:
        while True:
            if process.process.poll() is not None:
                raise ValueError("owned LoRA server exited before readiness")
            http.remaining()
            stdout, _ = process.stdout.read()
            stderr, _ = process.stderr.read()
            if "Now listening on: " + http.base in stdout + stderr:
                break
            time.sleep(0.1)
        _attest_offload(context, result, "lora-startup", stdout + "\n" + stderr)
        identity = http.json("/api/models")
        _require(result, "LoRA loaded model/backend identity", identity.get("loadedBackend") == context["backend"] and identity.get("loaded") == Path(context["model_path"]).name,
                 {"backend": context["backend"], "model": Path(context["model_path"]).name}, identity)
        architecture = str(identity.get("architecture", "")).replace("_", "-").lower()
        _require(result, "actual text LoRA family admitted", architecture in {"mistral3", "hunyuan-dense", "nemotron-h", "nemotron-h-moe", "nemotron-h-omni"},
                 "source-admitted text LoRA architecture", architecture)
        model = _hosted_response_model(identity["loaded"], http.json("/v1/models").get("data", []), result)
        _lora_requests(http, model, profile, result, _fixture(profile, "adapter"), directory)
    finally:
        process.stop()
        item = process.snapshot()
        result["processes"].append(item)
        _require(result, "LoRA server tree teardown verified", item.get("teardown", {}).get("verified") is True, True, item.get("teardown"))
        _attest_offload(context, result, "lora-final", item.get("stdout", "") + "\n" + item.get("stderr", ""))


def run(case, context, result):
    """Mutate the standard result; dispatcher owns exception/status persistence."""
    from .scenarios import ScenarioUntested
    ready, reason = availability(case, context)
    if not ready:
        raise ScenarioUntested(reason)
    profile = context["scenario_profile"]
    kind = profile["scenario"]
    root = Path(context["artifact_dir"]) / (kind + "-" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    for name, descriptor in profile.get("fixtures", {}).items():
        _require(result, "fixture SHA256 " + name, _hash(_fixture(profile, name)) == descriptor["sha256"], descriptor["sha256"])
    result["inputs"].update(profile_id=profile["id"], parameters=profile["parameters"], expected=profile["expected"],
                            models=context.get("profile_models"), fixtures=profile.get("fixtures", {}))
    sources = ["eng/regression/scaling_profiles.py", *_SOURCES[kind]]
    if kind == "distributed":
        sources.append("eng/regression/scaling_remote.py")
    result["evidence"].update(artifact_dir=str(root), sources=[{"path": source, "sha256": _hash(Path(context["repo"]) / source)} for source in sources])
    result["evidence"]["model_file_snapshot"] = _model_snapshot(context)
    if kind == "multi_gpu":
        devices = profile["parameters"]["devices"]
        weights_id = _weights_id(context)
        reference = _probe(profile, context, result, root, "single-device", devices[:1], weights_id)
        candidate = _probe(profile, context, result, root, "multi-device", devices, weights_id)
        _compare(reference, candidate, profile, context, result)
    elif kind == "speculation":
        _learned(profile, context, result, root)
    elif kind == "lora":
        _lora(profile, context, result, root)
    else:
        from .scaling_remote import run_remote_nodes, RemoteUnavailable
        reference = _probe(profile, context, result, root, "single-device", profile["parameters"]["devices"][:1], profile["expected"]["weights_id"])
        try:
            receipts = run_remote_nodes(profile, dict(context, artifact_dir=str(root)), result)
        except RemoteUnavailable as error:
            raise ScenarioUntested(str(error)) from error
        result["evidence"]["remote_nodes"] = receipts
        driver_path = Path(receipts[0]["local_probe_path"])
        _probe_report(_json(driver_path), profile, result, "distributed-driver", profile["expected"]["weights_id"])
        _compare(reference, driver_path, profile, context, result)
    _require(result, "model identities unchanged throughout profile", _model_snapshot(context) == result["evidence"]["model_file_snapshot"])
