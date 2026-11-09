"""Bounded n-gram speculative decoding parity using a prebuilt AgentTurnBench.

This is n-gram execution coverage, never learned-drafter or performance coverage.
No build, restore, download, external tool execution, or backend fallback occurs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

from .processes import redact, run_process, secret_values

_BACKENDS = {"cpu", "cuda", "mlx", "ggml_cpu", "ggml_cuda", "ggml_metal", "ggml_vulkan"}
_SOURCES = ("benchmarks/AgentTurnBench/Program.cs", "benchmarks/AgentTurnBench/compare.py",
            "eng/validation/validate-speculative-rows.py", "TensorSharp.Runtime/SamplingConfig.cs",
            "TensorSharp.Runtime/Scheduling/InferenceEngine.cs",
            "TensorSharp.Runtime/Scheduling/BatchExecutor.cs",
            "TensorSharp.Runtime/Speculative/SpeculatorRegistry.cs",
            "TensorSharp.Models/Models/Gemma4/Gemma4Model.Speculative.cs",
            "TensorSharp.Models/Models/MuseSpark/MuseSparkModel.Speculative.cs",
            "TensorSharp.Models/Models/MiniMaxM3/MiniMaxM3Model.Speculative.cs")
_FAMILY_BACKENDS = {
    "gemma4": {"cuda", "ggml_cpu", "ggml_cuda", "ggml_metal", "ggml_vulkan"},
    "muse-spark": {"ggml_cpu", "ggml_cuda", "ggml_metal", "ggml_vulkan"},
    "minimax-m3": {"ggml_cpu", "ggml_cuda"},
}
# The bounded engine request terminates normally with EOS or its token cap.
# InferenceEngine.cs emits these exact values; repetition, abort and error are
# not successful completion of this fixture, and protocol aliases do not apply.
_FINISH = {"eos", "max_tokens"}
_COUNTERS = ("Drafted", "Accepted", "VerifySteps", "PlainSteps", "Rollbacks")
_MAX_ROWS_BYTES = 4 * 1024 * 1024


def _command(context):
    argv = context.get("benchmark_command")
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(x, str) or not x for x in argv):
        raise ValueError("provide benchmark_command as a prebuilt executable argv list")
    # Do not permit caller-supplied flags to smuggle a drafter, another scenario,
    # or --spec-engine into what is recorded as a plain-versus-ngram test.
    dotnet = Path(argv[0]).name.lower() in {"dotnet", "dotnet.exe"}
    if len(argv) != (2 if dotnet else 1) or (dotnet and Path(argv[1]).suffix.lower() != ".dll"):
        raise ValueError("benchmark_command must contain only an executable, or dotnet and one prebuilt DLL")
    executable = shutil.which(argv[0]) or (str(Path(argv[0]).resolve()) if Path(argv[0]).is_file() else None)
    if not executable:
        raise ValueError("prebuilt benchmark executable is missing: " + argv[0])
    command = [executable]
    if dotnet:
        dll = Path(argv[1]).resolve()
        if not dll.is_file():
            raise ValueError("prebuilt AgentTurnBench DLL is missing: " + str(dll))
        command.append(str(dll))
    return command


def availability(context):
    """Read-only prerequisite check; this never builds or probes a device."""
    try:
        _command(context)
        if context.get("backend") not in _BACKENDS:
            return False, "unsupported benchmark backend; no fallback will run"
        family = str(context.get("family") or context.get("model_family") or "").lower().replace("_", "-")
        if family not in _FAMILY_BACKENDS:
            return False, "n-gram verification requires a source-verified family: gemma4, muse-spark or minimax-m3"
        if context["backend"] not in _FAMILY_BACKENDS[family]:
            return False, "n-gram verification is not supported for " + family + " on " + context["backend"]
        if not context.get("preparing_model") and not Path(context.get("model_path") or "").is_file():
            return False, "a verified local model is required"
        repo = Path(context["repo"])
        if any(not (repo / path).is_file() for path in _SOURCES):
            return False, "AgentTurnBench fixture/validator sources are missing from the selected checkout"
        timeout = context.get("timeout_seconds", 600)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            return False, "timeout_seconds must be finite and positive"
    except (KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)
    return True, ""


def _hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module(repo, relative, name):
    spec = importlib.util.spec_from_file_location(name, Path(repo) / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(v) for v in value]
    return value


def _assert(result, name, passed, expected=None, actual=None):
    result["assertions"].append(dict(name=name, passed=bool(passed), expected=expected, actual=actual))
    if not passed:
        raise ValueError(name)


def _inspect_rows(repo, path):
    """Reuse repository row validation, with stricter suite identity/shape gates."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("benchmark did not produce fresh regular rows.json")
    if not 0 < path.stat().st_size <= _MAX_ROWS_BYTES:
        raise ValueError("benchmark rows.json is empty or exceeds the 4 MiB bound")
    # Reject NaN/Infinity anywhere, including optional metrics unused by the
    # repository comparator. Avoid letting a serializer hide invalid evidence.
    def reject(value):
        raise ValueError("nonfinite JSON number: " + value)
    raw = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)
    def finite_numbers(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite benchmark metric")
        if isinstance(value, dict):
            for item in value.values(): finite_numbers(item)
        elif isinstance(value, list):
            for item in value: finite_numbers(item)
    finite_numbers(raw)
    comparator = _module(repo, _SOURCES[1], "regression_ngram_rows")
    indexed = comparator.load_rows(path)
    expected = {("spec", "plain greedy"), ("spec", "ngram")}
    if set(indexed) != expected or len(indexed) != 2:
        raise ValueError("expected exactly plain greedy and ngram rows for spec; no extra scenarios/draft head")
    plain, candidate = indexed[("spec", "plain greedy")], indexed[("spec", "ngram")]
    for label, row in (("plain", plain), ("ngram", candidate)):
        if not row["Tokens"] or len(row["Tokens"]) > 64:
            raise ValueError(label + " must produce 1..64 token IDs")
        if not 0 < row["Prompt"] <= 2048 - 64:
            raise ValueError(label + " prompt must leave the 64-token decode reserve within the 2048-token context")
        if row["Finish"] not in _FINISH:
            raise ValueError(label + " has an unsupported terminal reason: " + row["Finish"])
        if row["Finish"] == "max_tokens" and row["OutTokens"] != 64:
            raise ValueError(label + " hit max_tokens before its fixed 64-token budget")
        if row["TtftMs"] <= 0 or (row["Fresh"] > 0 and row["PrefillTps"] <= 0):
            raise ValueError(label + " must record positive timing/throughput for executed fresh prefill")
        if row["OutTokens"] > 1 and row["DecodeTps"] <= 0:
            raise ValueError(label + " must record positive decode throughput for multiple emitted tokens")
        for key in _COUNTERS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(label + " has a missing/invalid counter: " + key)
        if row["Accepted"] > row["Drafted"]:
            raise ValueError(label + " Accepted exceeds Drafted")
    validator = _module(repo, _SOURCES[2], "regression_ngram_parity")
    proof = validator.validate(list(indexed.values()), scenario="spec", label="ngram", plain_label="plain greedy")
    # validate() includes activation failure in its failures. Activation alone
    # is an honest untested outcome, so keep that distinct from parity failure.
    proof["activated"] = all(candidate[key] > 0 for key in ("Drafted", "Accepted", "VerifySteps"))
    proof["baseline_counters"] = {key: plain[key] for key in _COUNTERS}
    proof["baseline_clean"] = all(plain[key] == 0 for key in ("Drafted", "Accepted", "VerifySteps", "Rollbacks"))
    proof["shape"] = {key: {"plain": plain[key], "ngram": candidate[key]} for key in ("Prompt", "Reused", "Fresh")}
    proof["shape_equal"] = all(plain[key] == candidate[key] for key in ("Prompt", "Reused", "Fresh"))
    proof["rows_sha256"] = _hash(path)
    proof["tokens"] = {"plain": plain["Tokens"], "ngram": candidate["Tokens"]}
    proof["finish"] = {"plain": plain["Finish"], "ngram": candidate["Finish"]}
    return proof


def run_ngram(case, context):
    """Return the run_scenario result shape; save cancellation evidence, then rethrow."""
    # Runtime import permits the central scenarios dispatcher to import this module.
    from .scenarios import backend_evidence
    started = time.monotonic()
    root = Path(context["artifact_dir"])
    root.mkdir(parents=True, exist_ok=True)
    result = dict(id=case.get("id", "speculation_ngram"), kind="speculation_ngram", status="untested", reason="",
                  assertions=[], inputs={}, evidence={}, processes=[])
    interrupted = False
    run_dir = None
    try:
        ready, reason = availability(context)
        if not ready:
            result["reason"] = reason
        else:
            # Never remove/reuse evidence left by another attempt, even when the
            # caller accidentally supplies the same artifact_dir twice.
            run_dir = root / ("ngram-" + uuid.uuid4().hex)
            run_dir.mkdir(exist_ok=False)
            rows = run_dir / "rows.json"
            repo = Path(context["repo"]).resolve()
            model = Path(context["model_path"]).resolve()
            launcher = _command(context)
            command = launcher + ["--model", str(model), "--backend", context["backend"],
                "--scenarios", "spec", "--spec-only", "--spec-candidates", "ngram", "--spec-new", "64",
                "--spec-file", "128", "--warmup", "0", "--measure-passes", "1", "--out", str(rows)]
            # run_process already scrubs inherited TS_* settings. Do not let
            # context overrides restore a drafter or alter speculative policy.
            env = {k: v for k, v in context.get("env", {}).items()
                   if not re.search(r"SPEC|DRAFT|DFLASH|DSPARK|NGRAM|MTP", k, re.I)}
            # Exact parity is the contract: verify each row with the one-token kernels.
            # (ggml_metal verifies Qwen3.8 Flash Next rows as one batched forward by
            # default, which is faster and not bit-identical to plain decode.)
            env.update(MAX_CONTEXT="2048", TS_SCHED_NUM_BLOCKS="16", TS_SCHED_BLOCK_SIZE="128",
                       TS_Q4E_VERIFY_ROW_KERNELS="1")
            result["inputs"] = {"scenario": "spec", "candidate": "ngram", "decode_token_cap": 64,
                "family": context.get("family") or context.get("model_family"),
                "code_fixture_target_tokens": 128, "fixture": "Corpus.CodeText plus Corpus.AgentSystemPrompt in AgentTurnBench/Program.cs",
                "sampling": {"source": "SamplingConfig.Greedy", "temperature": 0, "top_k": 0,
                             "top_p": 1, "repetition_penalty": 1, "seed": -1,
                             "seed_note": "AgentTurnBench uses greedy argmax; it exposes no seed option. No random sampling is requested."},
                "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": context.get("model_sha256")},
                "model_hash_note": "Full model/shard integrity belongs to the suite preparation manifest; this adapter does not rehash weights.",
                "source_revision": context.get("source_revision"),
                "sources": [{"path": path, "sha256": _hash(repo / path)} for path in _SOURCES],
                "bounded_environment": {key: env[key] for key in ("MAX_CONTEXT", "TS_SCHED_NUM_BLOCKS", "TS_SCHED_BLOCK_SIZE", "TS_Q4E_VERIFY_ROW_KERNELS")}}
            result["evidence"].update(artifact_dir=str(run_dir), rows_path=str(rows), reproduce_argv=command,
                scope="Within-model plain versus n-gram token parity and active verification; no learned-head or performance coverage.",
                binary={"path": launcher[-1], "sha256": _hash(launcher[-1])})
            item = run_process(command, cwd=repo, artifact_dir=run_dir, stage="ngram",
                               timeout_seconds=context.get("timeout_seconds", 600), env=env)
            result["processes"].append(item)
            observed = backend_evidence(item.get("stdout", "") + "\n" + item.get("stderr", ""), context["backend"])
            result["evidence"]["backend"] = observed
            # Capture any token divergence before enforcing process exit, since
            # AgentTurnBench itself exits nonzero on detected disagreement.
            if rows.is_file():
                result["evidence"]["parity"] = _inspect_rows(repo, rows)
            _assert(result, "owned process tree teardown verified", item.get("teardown", {}).get("verified") is True,
                    True, item.get("teardown"))
            _assert(result, "no timeout", item.get("timed_out") is False, False, item.get("timed_out"))
            _assert(result, "no cancellation", not item.get("cancelled", False), False, item.get("cancelled", False))
            _assert(result, "benchmark exit", item.get("exit_code") == 0, 0, item.get("exit_code"))
            _assert(result, "complete captured output", item.get("output_truncated") is False, False, item.get("output_truncated"))
            oom = re.search(r"out[ -]?of[ -]?memory|CUDA_ERROR_OUT_OF_MEMORY|std::bad_alloc",
                            item.get("stdout", "") + "\n" + item.get("stderr", ""), re.I)
            _assert(result, "no out-of-memory diagnostic", oom is None, None, oom.group(0) if oom else None)
            _assert(result, "actual backend matches request without fallback", observed["verified"], context["backend"], observed)
            _assert(result, "fresh complete row artifact", "parity" in result["evidence"], "two fresh rows", str(rows))
            proof = result["evidence"]["parity"]
            _assert(result, "plain baseline has no speculative work", proof["baseline_clean"], "zero draft/accept/verify/rollback", proof["baseline_counters"])
            _assert(result, "matched prompt and cache shapes", proof["shape_equal"], "equal Prompt/Reused/Fresh", proof["shape"])
            _assert(result, "strict greedy token and finish parity", proof["greedy_tokens_equal"] and proof["finish_equal"],
                    "identical nonempty token IDs and Finish", {"first_divergence": proof["first_divergence"], "finish": proof["finish"]})
            if not proof["activated"]:
                result["reason"] = "n-gram verification/acceptance was not positively exercised; parity alone is not coverage"
            else:
                _assert(result, "n-gram draft acceptance and verification exercised", True, "positive Drafted/Accepted/VerifySteps", proof["counters"])
                result.update(status="pass", reason="active n-gram execution preserves plain greedy token IDs and finish")
    except KeyboardInterrupt:
        interrupted = True
        result["reason"] = "cancelled; scenario remains untested"
        receipt = run_dir / "ngram.process.json" if run_dir else None
        if receipt and receipt.is_file():
            try:
                item = json.loads(receipt.read_text(encoding="utf-8"))
                result["processes"].append(item)
                if item.get("teardown", {}).get("verified") is not True:
                    result.update(status="fail", reason="cancelled; owned process teardown is unverified")
            except (OSError, ValueError):
                result.update(status="fail", reason="cancelled; process teardown receipt could not be read")
        else:
            result["evidence"]["cancellation"] = "process receipt unavailable; teardown outcome is unknown"
            result["status"] = "fail"
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        result.update(status="fail", reason=type(error).__name__ + ": " + str(error))
    finally:
        result["duration_seconds"] = round(time.monotonic() - started, 4)
        encoded = redact(json.dumps(_finite(result), ensure_ascii=False, allow_nan=False, indent=2),
                         secret_values() + secret_values(context.get("env") or {}))
        result = json.loads(encoded)
        (root / "scenario.json").write_text(encoded + "\n", encoding="utf-8")
    if interrupted:
        raise KeyboardInterrupt
    return result
