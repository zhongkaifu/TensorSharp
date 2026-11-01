#!/usr/bin/env python3
"""Strict five-case Metal MoE overlap qualification. Dry-run unless --execute.

Uses one already-built candidate probe/native binary for off-reference then
on-teacher-forced execution. Never builds, downloads, overwrites an old output
directory, or accepts the probe's 1e-3 numerical gate as bitwise equivalence.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "eng/ForcedLogitProbe/Program.cs").is_file())
FEATURE_ENV = "TS_HOST_MOE_METAL_OVERLAP"
EXPECTED_CASES = {"chinese", "english", "arithmetic", "code", "json"}
MAX_STEPS = 128
FIXED_ENV = {
    "TS_CPU_MOE_THREADS": "12", "TS_GGML_CPU_THREADS": "12",
    "MAX_CONTEXT": "8192", "KV_CACHE_DTYPE": "f16", "TS_PREFILL_CHUNK": "512",
    "TS_SPEC": "0", "TS_HOST_MOE_DEVICE_MIN_BATCH": "128",
    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
}


def default_cases():
    questions = [
        ("chinese", "请用三句话说明自行车的刹车、齿轮和平衡分别有什么作用。"),
        ("english", "Explain how evaporation, condensation, and rain form the water cycle in three sentences."),
        ("arithmetic", "Calculate 17*23 and explain your calculation briefly."),
        ("code", "Write a C# function that computes the greatest common divisor using Euclid's algorithm."),
        ("json", "Output a JSON object containing the integers 1 through 12 and their squares. No Markdown."),
    ]
    # These are direct-model plaintext probes, deliberately independent of a
    # family's chat template. This gate checks numerical switch equivalence;
    # semantic conversational quality belongs to a separate chat workload.
    return [{"name": name, "prompt": question} for name, question in questions]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*arguments):
    run = subprocess.run(["git"] + list(arguments), cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=15, check=True)
    return run.stdout


def identity(probe):
    native = probe.parent / "libGgmlOps.dylib"
    require(probe.is_file() and native.is_file(), "Probe DLL and native library must exist together; this tool never builds")
    revision = git("rev-parse", "HEAD").decode().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision), "Source revision unavailable")
    dependency = REPO / "ExternalProjects/ggml"
    ggml = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dependency, capture_output=True,
                          text=True, check=True, timeout=15).stdout.strip()
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=dependency, capture_output=True,
                           text=True, check=True, timeout=15).stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40}", ggml) and not clean, "Upstream ggml must be identifiable and unchanged")
    binaries = set(probe.parent.glob("TensorSharp*.dll")) | {probe, native}
    untracked = {}
    for name in git("ls-files", "--others", "--exclude-standard").decode().splitlines():
        path = REPO / name
        if path.suffix.lower() in {".cs", ".cpp", ".h", ".mm", ".m", ".csproj", ".json"} and path.is_file():
            require(path.stat().st_size < 16 * 1024 * 1024, "Unexpected large untracked source; commit/inspect it before qualification")
            untracked[name] = digest(path)
    return {"source_revision": revision, "source_status": git("status", "--porcelain").decode(),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
            "untracked_source_sha256": untracked, "ggml_revision": ggml, "ggml_status": clean,
            "binary_files_sha256": {str(path): digest(path) for path in sorted(binaries)}}


def model_identity(path):
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    paths = [path]
    if match:
        require(int(match[2]) == 1 and 1 <= int(match[3]) <= 16, "Supply the first GGUF shard")
        paths = [path.with_name(f"{match[1]}-{index:05d}-of-{int(match[3]):05d}.gguf")
                 for index in range(1, int(match[3]) + 1)]
    return [{"name": file.name, "bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns}
            for file in paths]


def environment(feature, enabled, host_layers):
    require(feature == FEATURE_ENV, "Only the implemented overlap switch is supported")
    result = {key: value for key, value in os.environ.items()
              if not key.startswith(("TS_", "TENSORSHARP_", "GGML_"))
              and key not in {"MAX_CONTEXT", "KV_CACHE_DTYPE"}}
    result.update(FIXED_ENV)
    result["TS_N_CPU_MOE"] = str(host_layers)
    result[feature] = str(enabled)
    return result


def overlap_engagement(log, enabled, host_layers, expect_path, expected_async_calls=None):
    # The backend reports and resets counters. Reject-detail lines share the
    # marker but are not summaries; multiple reset summaries are disjoint.
    lines = [line for line in log.splitlines()
             if "[HOSTMOE-METAL-OVERLAP]" in line and "mode=" in line and "considered=" in line]
    if not lines and (not enabled or (expect_path == "fallback" and host_layers == 0)):
        return {"mode": int(enabled), "summary_reported": False, "async": 0, "serial": 0}
    require(lines, "Missing actual overlap engagement summary")
    counters = ("considered", "ineligible", "eligible", "async", "serial", "rejected", "submit_failures", "drain_failures", "fast_fallbacks")
    values = {key: 0 for key in counters}
    for line in lines:
        summary = {key: (float(value) if "." in value else int(value))
                   for key, value in re.findall(r"([a-z_]+)=([0-9]+(?:\.[0-9]+)?)", line)}
        require(type(summary.get("mode")) is int and summary["mode"] == int(enabled), "Wrong overlap execution mode")
        for key in counters:
            require(type(summary.get(key)) is int, "Missing/malformed overlap counter: " + key)
            values[key] += summary[key]
    require(values["submit_failures"] == values["drain_failures"] == 0, "Overlap submission/drain failed")
    if enabled and expect_path == "active":
        require(host_layers > 0, "Active path requires positive host layers")
        require(values["rejected"] == values["fast_fallbacks"] == 0, "Active overlap rejected work or used a fast fallback")
        require(values["async"] > 0 and values["eligible"] == values["async"] and values["serial"] == 0,
                "Candidate did not execute the eligible work asynchronously")
        if expected_async_calls is not None:
            require(values["async"] == expected_async_calls, "Overlap engagement does not cover every expected decode layer")
    elif enabled:
        require(values["async"] == values["serial"] == values["eligible"] == 0, "Fallback unexpectedly performed overlap")
        if host_layers > 0:
            require(values["considered"] > 0 and values["ineligible"] + values["rejected"] > 0,
                    "Fallback host layers were not observed by the overlap driver")
    else:
        require(values["async"] == values["serial"] == values["eligible"] == 0, "Off reference unexpectedly engaged overlap")
    return dict(values, mode=int(enabled), summary_reported=True, raw_summaries=lines)


def stop_owned(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def validate_vocab_size(value):
    require(type(value) is int and value > 1, "vocab_size must be an actual positive integer greater than one")
    return value


def read_metrics(directory, run_id, steps, forced, host_layers, expect_path, expected_vocab=None):
    path = directory / "metrics.json"
    data = json.loads(path.read_text())
    require(data.get("complete") is True and data.get("all_finite") is True,
            "Probe did not finish with finite logits")
    require(data.get("cases_done") == data.get("cases_total") == 5, "Incomplete/wrong case count")
    require(data.get("run_id") == run_id and data.get("backend") == "ggml_metal", "Wrong/stale run identity")
    require(data.get("max_context") == 8192 and data.get("cpu_moe_threads") == 12,
            "Probe did not apply fixed context/expert threads")
    vocab = validate_vocab_size(data.get("vocab_size"))
    require(expected_vocab is None or vocab == expected_vocab, "Off/on vocab_size differs")
    require(data.get("cpu_moe") == (f"first {host_layers} layer(s)" if host_layers else None),
            "Probe did not apply the requested host-layer configuration")
    require(data.get("qualified") is True if forced else data.get("qualified") is None,
            "Probe numerical gate failed or reference/forced mode differs")
    cases = {case["name"]: case for case in data.get("cases", [])}
    require(set(cases) == EXPECTED_CASES and len(data["cases"]) == 5, "Wrong/duplicate case names")
    for name, case in cases.items():
        rows = case.get("metrics", [])
        require(case.get("steps") == steps and len(rows) == steps, "Wrong number of captured rows: " + name)
        require([row.get("step") for row in rows] == list(range(steps)), "Missing/duplicate/out-of-order rows: " + name)
        require(all(row.get("finite") is True for row in rows), "Nonfinite row: " + name)
        if forced:
            require(all(row.get("top1_equal") is True and row.get("reference_top") == row.get("candidate_top")
                        and type(row.get("reference_top")) is int and type(row.get("candidate_top")) is int
                        and 0 <= row["reference_top"] < vocab
                        for row in rows), "Top1 differs: " + name)
        else:
            tokens = case.get("generated_tokens")
            require(isinstance(tokens, list) and len(tokens) == steps
                    and all(type(token) is int and 0 <= token < vocab for token in tokens), "Bad reference token history: " + name)
            require(tokens == [row["top"] for row in rows], "Reference token history does not match its argmax rows")
    log = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
    require("ggml_metal" in log.lower(), "Actual Metal backend not established by startup logs")
    placement = [line for line in log.splitlines() if "moe" in line.lower()
                 and any(term in line.lower() for term in ("host", "cpu", "offload"))]
    observed_host_counts = []
    for line in placement:
        match = re.search(r"routed experts of\s+(\d+)\s+of\s+(\d+)\s+layers", line)
        if match:
            observed_host_counts.append(int(match[1]))
    require(not observed_host_counts or set(observed_host_counts) == {host_layers}, "Actual logged host-layer placement differs")
    if expect_path == "active":
        require(all(type(case.get("prompt_tokens")) is int and case["prompt_tokens"] > 1 for case in cases.values()),
                "Active suite requires multi-token prefills so decode engagement has an unambiguous count")
    expected_calls = (steps - 1) * len(cases) * host_layers
    engagement = overlap_engagement(log, forced, host_layers, expect_path, expected_calls if forced else None)
    return data, cases, placement, engagement, vocab


def finite_row_argmax(raw, vocab):
    values = array.array("f", raw)
    if sys.byteorder != "little":
        values.byteswap()
    require(len(values) == vocab, "Wrong raw vocabulary width")
    top, best = 0, values[0]
    for index, value in enumerate(values):
        require(math.isfinite(value), "Nonfinite raw logit row")
        if value > best:
            top, best = index, value
    return top


def compare_vectors(first, second, steps, vocab, expected_reference_tops=None):
    validate_vocab_size(vocab)
    row_bytes = vocab * 4
    expected_bytes = row_bytes * steps
    require(first.stat().st_size == second.stat().st_size == expected_bytes, "Wrong full-logit file size")
    a_sha, b_sha = hashlib.sha256(), hashlib.sha256()
    different_rows, top1_different_rows = [], []
    require(expected_reference_tops is None or len(expected_reference_tops) == steps,
            "Wrong expected argmax history length")
    with first.open("rb") as a, second.open("rb") as b:
        for row in range(steps):
            x, y = a.read(row_bytes), b.read(row_bytes)
            require(len(x) == len(y) == row_bytes, "Truncated logit row")
            a_sha.update(x)
            b_sha.update(y)
            top_a = finite_row_argmax(x, vocab)
            top_b = top_a if x == y else finite_row_argmax(y, vocab)
            if expected_reference_tops is not None:
                require(top_a == expected_reference_tops[row], "Raw reference argmax disagrees with probe metadata")
            if top_a != top_b:
                top1_different_rows.append(row)
            if x != y:
                different_rows.append(row)
        require(not a.read(1) and not b.read(1), "Extra logit rows")
    return {"bytes_per_file": expected_bytes, "rows": steps, "vocab": vocab,
            "reference_sha256": a_sha.hexdigest(), "candidate_sha256": b_sha.hexdigest(),
            "different_rows": different_rows, "bit_identical": not different_rows,
            "raw_rows_finite": True, "raw_top1_different_rows": top1_different_rows,
            "raw_argmax_metadata_checked": expected_reference_tops is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing first GGUF shard; never copied/downloaded")
    parser.add_argument("--probe", type=Path, default=REPO / "eng/ForcedLogitProbe/bin/Release/net10.0/ForcedLogitProbe.dll")
    parser.add_argument("--cases", type=Path, help="Optional five-case JSON; default suite is embedded and public")
    parser.add_argument("--output", type=Path, required=True, help="Fresh nonexistent directory under this checkout's ignored artifacts")
    parser.add_argument("--feature-env", choices=(FEATURE_ENV,), default=FEATURE_ENV,
                        help="Implemented overlap switch; set0forreference,1forcandidate")
    parser.add_argument("--host-layers", type=int, required=True, help="Requested CPU expert layers, including0for dense fallback")
    parser.add_argument("--expect-path", choices=("active", "fallback"), required=True)
    parser.add_argument("--steps", type=int, default=128, help="2..128 captured rows per case; default128")
    parser.add_argument("--timeout", type=float, default=1800, help="Per-process timeout in seconds")
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--execute", action="store_true", help="Launch qualification processes; the default only prints the plan")
    args = parser.parse_args()
    require(2 <= args.steps <= MAX_STEPS and args.timeout > 0, "Steps must be2..128 and timeout positive")
    require(args.host_layers >= 0 and (args.expect_path != "active" or args.host_layers > 0),
            "Host layers must be nonnegative; active requires positive host layers")
    require(args.feature_env == FEATURE_ENV, "Supply the implemented overlap switch")
    require(args.feature_env not in FIXED_ENV, "Feature switch cannot replace a fixed engine setting")
    args.output = args.output.resolve()
    require(args.output.is_relative_to(REPO / "artifacts") and not args.output.exists(), "Output must be a fresh nonexistent ignored artifacts directory")
    case_bytes = (args.cases.resolve(strict=True).read_bytes() if args.cases
                  else (json.dumps(default_cases(), ensure_ascii=False, indent=2) + "\n").encode())
    cases = json.loads(case_bytes)
    require(isinstance(cases, list) and len(cases) == 5 and {case.get("name") for case in cases} == EXPECTED_CASES,
            "Suite must contain exactly the five required cases")
    require(all(re.fullmatch(r"[a-z_]+", case["name"]) and isinstance(case.get("prompt"), str)
                and case["prompt"] and not case.get("generated_tokens") for case in cases), "Invalid suite case")
    plan = {"execute": args.execute, "candidate_probe": str(args.probe.resolve()), "feature_switch": args.feature_env,
            "sequence": ["off greedy reference", "on teacher-forced candidate"], "steps_per_case": args.steps,
            "expected_rows_per_side": len(cases) * args.steps, "host_layers": args.host_layers, "expected_path": args.expect_path,
            "expected_vector_bytes_both_sides": None, "vector_storage_formula": "2*case_count*steps*vocab_size*4; vocab_size comes from actual probe metrics",
            "storage_scope": "Local generated float32 evidence can exceed1GB for128-row/large-vocabulary suites; no model download/copy. Timings are not performance evidence.",
            "cases_sha256": hashlib.sha256(case_bytes).hexdigest(), "fixed_environment": dict(FIXED_ENV, TS_N_CPU_MOE=str(args.host_layers)),
            "output": str(args.output), "acceptance": "All5full-vocab files byte-identical; every requested row independentlyfinite/top1-identical; matching forced token histories and dynamic vocab_size; actual active/fallback path proven; tight1e-3 gate alone is insufficient"}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return 0
    args.probe = args.probe.resolve(strict=True)
    args.model = args.model.resolve(strict=True)
    initial_identity = identity(args.probe)
    initial_model = model_identity(args.model)
    ancestor = args.output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free > 128 * 1024 * 1024, "Insufficient free space before vocabulary is known")
    args.output.mkdir(parents=True, exist_ok=False)
    reference, candidate = args.output / "reference-off", args.output / "candidate-on"
    reference.mkdir()
    candidate.mkdir()
    snapshot = args.output / "cases-input.json"
    snapshot.write_bytes(case_bytes)
    manifest = dict(plan, schema_version=1, runner_sha256=digest(Path(__file__)),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    input_identity=initial_identity, model_shards_stat=initial_model,
                    complete=False, passed=False, runs=[], vector_comparisons={})
    manifest_path = args.output / "manifest.json"
    save(manifest_path, manifest)
    vocab = None
    try:
        for enabled, directory in [(0, reference), (1, candidate)]:
            require(not any(directory.iterdir()), "Refusing nonempty phase directory")
            require(identity(args.probe) == initial_identity, "Candidate source/binaries changed between phases")
            require(model_identity(args.model) == initial_model, "Model shards changed")
            run_id = uuid.uuid4().hex
            argv = [args.dotnet, str(args.probe), str(args.model), str(directory), "--backend", "ggml_metal",
                    "--steps", str(args.steps), "--cases", str(snapshot), "--run-id", run_id]
            if enabled:
                argv += ["--reference", str(reference)]
            run = {"feature_enabled": bool(enabled), "run_id": run_id, "argv": argv,
                   "environment": dict(FIXED_ENV, TS_N_CPU_MOE=str(args.host_layers), **{args.feature_env: str(enabled)}), "complete": False}
            manifest["runs"].append(run)
            save(manifest_path, manifest)
            print(f"Starting{' candidate' if enabled else ' reference'}: {run_id}", flush=True)
            started = time.monotonic()
            with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
                process = subprocess.Popen(argv, cwd=REPO, env=environment(args.feature_env, enabled, args.host_layers),
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    run["exit_code"] = process.wait(timeout=args.timeout)
                finally:
                    stop_owned(process)
            run["wall_seconds_quality_only"] = time.monotonic() - started
            require(run["exit_code"] == 0, "Probe process failed; inspect owned stdout/stderr")
            require(identity(args.probe) == initial_identity, "Candidate source/binaries changed during phase")
            metrics, by_name, placement, engagement, actual_vocab = read_metrics(
                directory, run_id, args.steps, bool(enabled), args.host_layers, args.expect_path, vocab)
            vocab = actual_vocab
            manifest["vocab_size"] = vocab
            manifest["expected_vector_bytes_both_sides"] = 2 * len(cases) * args.steps * vocab * 4
            if not enabled:
                require(shutil.disk_usage(args.output).free > len(cases) * args.steps * vocab * 4 + 128 * 1024 * 1024,
                        "Insufficient space for candidate vectors at the actual vocabulary width")
            stored_cases = json.loads((directory / "cases.json").read_text())
            histories = {case["name"]: case["generated_tokens"] for case in stored_cases}
            require(set(histories) == EXPECTED_CASES and all(len(tokens) == args.steps for tokens in histories.values()),
                    "Stored forced token histories are incomplete")
            log = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
            run.update(complete=True, all_finite=metrics["all_finite"], probe_qualified=metrics["qualified"],
                       cases_done=metrics["cases_done"], cases_total=metrics["cases_total"],
                       metrics_sha256=digest(directory / "metrics.json"), cases_sha256=digest(directory / "cases.json"),
                       placement_lines=placement, step_tokens=histories, token_history_source="cases.json",
                       reported_settings={key: metrics.get(key) for key in ("backend", "max_context", "vocab_size", "cpu_moe", "cpu_moe_threads")},
                       metal_device_log_lines=[line for line in log.splitlines()
                                               if "ggml_metal" in line.lower() and "gpu name" in line.lower()],
                       overlap_engagement=engagement,
                       feature_log_lines=[line for line in log.splitlines() if "overlap" in line.lower()])
            save(manifest_path, manifest)
        ref_cases = json.loads((reference / "cases.json").read_text())
        on_cases = json.loads((candidate / "cases.json").read_text())
        require(ref_cases == on_cases, "Forced prompt/generated token histories differ")
        _, off, _, _, _ = read_metrics(reference, manifest["runs"][0]["run_id"], args.steps, False, args.host_layers, args.expect_path, vocab)
        _, on, _, _, _ = read_metrics(candidate, manifest["runs"][1]["run_id"], args.steps, True, args.host_layers, args.expect_path, vocab)
        top1_rows = 0
        for name in sorted(EXPECTED_CASES):
            comparison = compare_vectors(reference / (name + ".f32"), candidate / (name + ".f32"),
                                         args.steps, vocab, off[name]["generated_tokens"])
            rows = on[name]["metrics"]
            require([row["reference_top"] for row in rows] == off[name]["generated_tokens"], "Forced reference argmax history differs")
            top1_rows += sum(row["top1_equal"] is True for row in rows)
            manifest["vector_comparisons"][name] = comparison
        manifest["rows_top1_exact"] = top1_rows
        manifest["all_vector_files_bit_identical"] = all(item["bit_identical"] for item in manifest["vector_comparisons"].values())
        manifest["complete"] = True
        manifest["all_requested_rows_qualified"] = top1_rows == len(cases) * args.steps
        manifest["passed"] = manifest["all_requested_rows_qualified"] and manifest["all_vector_files_bit_identical"]
        require(identity(args.probe) == initial_identity and model_identity(args.model) == initial_model,
                "Source, binaries or model identity changed")
        save(manifest_path, manifest)
        print(json.dumps({key: manifest[key] for key in ["complete", "passed", "all_requested_rows_qualified", "rows_top1_exact", "vocab_size", "all_vector_files_bit_identical"]}), flush=True)
        return 0 if manifest["passed"] else 1
    except BaseException as error:
        manifest["passed"] = False
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        save(manifest_path, manifest)
        raise


if __name__ == "__main__":
    def cancelled(signum, frame):
        raise KeyboardInterrupt(f"Runner interrupted by signal{signum}")
    signal.signal(signal.SIGTERM, cancelled)
    sys.exit(main())
