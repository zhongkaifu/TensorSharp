#!/usr/bin/env python3
"""Run matched, EOS-complete trained Qwen3.8 smoke checks in fresh processes.

TensorSharp renders the checkpoint's no-thinking chat template once. The exact
exported token IDs are fed to its cache arm and Strata. Reports retain binary
identity, full TensorSharp logits, engine-specific timing denominators, output
IDs, semantic checks, GPU telemetry, and failures. This is a small acceptance
suite, not a broad language-quality evaluation or a cross-engine exactness claim.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expert_cache_validation", HERE / "qwen4exp-expert-cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

CASES = {
    "math": "Compute 17 + 25. Reply with only the integer.",
    "extraction": "Inventory: cedar=3, maple=7, pine=2. Return only the names whose quantity is at least 3, comma-separated in the original order.",
    "code": "Write a Python function square(x) that returns x*x. Return only the function, without Markdown.",
    "squares": "Return the squares of the integers 1 through 20, in order, as comma-separated integers. Return only the list.",
    "tool_json": 'Available tool: get_weather(city, unit), where unit is "celsius" or "fahrenheit". User asks: What is the weather in Hangzhou in Celsius? Return only one JSON object with exactly the keys "name" and "arguments", containing the tool name and its required arguments. Do not answer the weather question.',
}


def semantic_check(case, text, complete):
    """Check the requested result without executing generated code."""
    if not complete or not text.strip():
        return {"passed": False, "reason": "missing EOS or empty answer"}
    if case == "math":
        passed = text.strip() == "42"
    elif case == "extraction":
        passed = re.fullmatch(r"cedar\s*,\s*maple", text.strip(), re.IGNORECASE) is not None
    elif case == "squares":
        answer = text.strip()
        passed = bool(re.fullmatch(r"\d+(?:\s*,\s*\d+){19}", answer)) and [int(value.strip()) for value in answer.split(",")] == [value * value for value in range(1, 21)]
    elif case == "tool_json":
        def unique_object(pairs):
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("Duplicate JSON keys")
            return result
        try:
            passed = json.loads(text, object_pairs_hook=unique_object) == {
                "name": "get_weather", "arguments": {"city": "Hangzhou", "unit": "celsius"}}
        except (ValueError, TypeError):
            passed = False
    elif case == "code":
        source = text.strip()
        # The task explicitly forbids Markdown. Parsing the original answer
        # prevents a formatting failure from becoming a pass after repair.
        try:
            tree = ast.parse(source)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            function = functions[0] if len(tree.body) == len(functions) == 1 else None
            body = function.body if function else []
            # A docstring is harmless; every executable statement must be the return.
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            expression = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
            passed = bool(function and function.name == "square" and len(function.args.args) == 1
                          and function.args.args[0].arg == "x" and not function.args.vararg and not function.args.kwarg
                          and not function.args.kwonlyargs and not function.decorator_list
                          and isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult)
                          and isinstance(expression.left, ast.Name) and expression.left.id == "x"
                          and isinstance(expression.right, ast.Name) and expression.right.id == "x")
        except SyntaxError:
            passed = False
    else:
        raise ValueError("Unknown semantic case: " + case)
    return {"passed": passed, "reason": None if passed else "answer does not satisfy the case's independent semantic check"}


def parse_strata_log(log, expected_prompt, max_new):
    prompts = re.findall(r"^prompt\s*:\s*([0-9 \t]+)$", log, re.MULTILINE)
    outputs = re.findall(r"^output\s*:\s*([0-9 \t]+)$", log, re.MULTILINE)
    if len(prompts) != 1 or len(outputs) != 1:
        raise ValueError("Missing or ambiguous final Strata prompt/output IDs")
    prompt, output = [int(x) for x in prompts[0].split()], [int(x) for x in outputs[0].split()]
    if prompt != expected_prompt or not 0 < len(output) <= max_new:
        raise ValueError("Strata prompt IDs differ or generated token count is invalid")
    decode = re.findall(r"^decode\s+(\d+) tokens in ([0-9.]+) ms\s*->\s*([0-9.]+) tok/s", log, re.MULTILINE)
    prefill = re.findall(r"^prefill\s+(\d+) tokens in ([0-9.]+) ms\s*->\s*([0-9.]+) tok/s\s*\(time to first token ([0-9.]+) ms\)", log, re.MULTILINE)
    if len(decode) != 1 or int(decode[0][0]) != len(output) or len(prefill) != 1 or int(prefill[0][0]) != len(prompt) - 1:
        raise ValueError("Missing or inconsistent Strata timing denominators")
    return {"prompt_tokens": prompt, "generated_tokens": output, "decode_tokens": len(output),
            "decode_ms": float(decode[0][1]), "decode_tps": float(decode[0][2]),
            "prefill_tokens": int(prefill[0][0]), "prefill_ms": float(prefill[0][1]),
            "prefill_tps": float(prefill[0][2]), "time_to_first_token_ms": float(prefill[0][3]),
            "decode_denominator": "all generated tokens including the first",
            "prefill_denominator": "prompt tokens excluding the final prompt token"}


def completed_suite(cases, expected_cases, expected_arms, repetitions):
    if set(cases) != set(expected_cases):
        return False
    for case in cases.values():
        if len(case.get("runs", [])) != repetitions:
            return False
        for run in case["runs"]:
            arms = run.get("arms", {})
            if set(arms) != set(expected_arms) or not all(
                    arm.get("passed") is True and arm.get("complete") is True
                    and arm.get("semantic_check", {}).get("passed") is True for arm in arms.values()):
                return False
    return True


def engine_order(cache_budgets, repetition, initial_cache_budget=0, strata_first=False):
    order = ["tensor-cache0"] + [f"tensor-cache{budget}" for budget in cache_budgets] + ["strata"]
    initial = f"tensor-cache{initial_cache_budget}"
    if initial not in order:
        raise ValueError("Initial cache budget must be zero or one of the tested budgets")
    shift = (order.index("strata" if strata_first else initial) + repetition) % len(order)
    return order[shift:] + order[:shift]


def parse_strata_scalar_execution(log, capacity, generated_count):
    rounds = re.findall(r"^speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+)", log, re.MULTILINE)
    windows = re.findall(r"^window sizes\s+(.+)$", log, re.MULTILINE)
    if len(rounds) != 1 or len(windows) != 1:
        raise ValueError("Missing verifier scalar execution evidence")
    count, configured, accepted, offered = map(int, rounds[0])
    pairs = re.findall(r"T(\d+):(\d+)", windows[0])
    histogram = {int(width): int(value) for width, value in pairs}
    if len(pairs) != len(histogram) or configured != capacity or count != generated_count or accepted or offered:
        raise ValueError("Strata used drafts or inconsistent verifier capacity/rounds")
    if set(histogram) != set(range(1, capacity + 1)) or histogram[1] != count or any(histogram[width] for width in range(2, capacity + 1)):
        raise ValueError("Strata did not execute exclusively single-row target windows")
    return {"verifier_capacity": capacity, "observed_target_window": 1, "observed_window_counts": histogram,
            "rounds": count, "drafts_accepted": accepted, "drafts_offered": offered,
            "algorithm_matched_to_tensorsharp_scalar_greedy": True}


def verified_download(report_path, checkpoint_paths):
    downloaded = json.loads(Path(report_path).read_text(encoding="utf-8"))
    shards = downloaded.get("shards", [])
    expected = {str(Path(path).resolve()).casefold(): Path(path) for path in checkpoint_paths}
    if downloaded.get("status") != "verified" or len(shards) != len(expected) or not shards:
        raise ValueError("Download report does not verify every requested checkpoint shard")
    seen = set()
    for shard in shards:
        key = str(Path(shard.get("path", "")).resolve()).casefold()
        digest = shard.get("sha256", "")
        if key not in expected or key in seen or shard.get("verified") is not True or shard.get("actual_sha256") != digest or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Download verification has missing, duplicate or mismatched publisher hashes/paths")
        if expected[key].stat().st_size != shard.get("bytes"):
            raise ValueError("Verified checkpoint shard size changed")
        seen.add(key)
    return {"report_path": str(Path(report_path).resolve()), "report_sha256": cache.sha(report_path),
            "status": downloaded["status"], "shards": shards,
            "scope": "publisher hashes verified by the downloader; this comparison checks paths/sizes and does not rehash weights"}


def parse_strata_final_logits(path, vocab, prompt_count, generated_count, max_new):
    """Strata's stride=1 dump contains a row for every consumed position.

    Its header reserves max_new rows even if EOS stops earlier. Accept only the
    exact actual payload shape implied by the printed completed output, then
    select the known final prediction position. No positional guessing.
    """
    size = Path(path).stat().st_size
    with Path(path).open("rb") as source:
        header = source.read(8)
        if len(header) != 8:
            raise ValueError("Truncated Strata logits header")
        n_vocab, planned_rows = struct.unpack("<ii", header)
        rows = prompt_count - 1 + generated_count
        if n_vocab != vocab or planned_rows != prompt_count - 1 + max_new or size != 8 + rows * vocab * 4:
            raise ValueError("Strata dump does not identify the expected complete stride=1 positions")
        source.seek(8 + (rows - 1) * vocab * 4)
        final = list(struct.unpack("<" + "f" * vocab, source.read(vocab * 4)))
    if not all(math.isfinite(value) for value in final):
        raise ValueError("Strata final logits contain nonfinite values")
    return {"final_logits": final, "prediction_position": rows - 1, "actual_rows": rows,
            "header_planned_rows": planned_rows, "sha256": cache.sha(path)}


def load_strata_tokenizer(root, pack):
    # Use Strata's own tokenizer implementation and extracted GGUF metadata.
    sys.path.insert(0, str(root / "tools"))
    from strata_tokenizer import Tokenizer
    directory = pack / "tokenizer"
    config = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * config["vocab_size"]
    for token, index in vocab.items():
        if not 0 <= index < len(tokens) or tokens[index] is not None:
            raise ValueError("Strata vocabulary IDs are not unique and dense")
        tokens[index] = token
    if any(token is None for token in tokens):
        raise ValueError("Strata vocabulary has missing IDs")
    tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text(encoding="utf-8").splitlines(),
                          json.loads((directory / "token_type.json").read_text(encoding="utf-8")),
                          config["pre"], config["special_ids"])
    eos = {value for key, value in config["special_ids"].items()
           if key.rsplit(".", 1)[-1] in ("eos_token_id", "eot_token_id", "eom_token_id")}
    if not eos:
        raise ValueError("Extracted checkpoint tokenizer does not declare EOS IDs")
    return tokenizer, eos, {path.name: cache.sha(path) for path in directory.iterdir() if path.is_file()}


def windows_peak_working_set(pid):
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        return int(counters.PeakWorkingSetSize) if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else None
    finally:
        kernel.CloseHandle(handle)


def run_process(command, cwd, environment, directory, timeout):
    directory.mkdir(parents=True)
    samples, peaks, stopped = [], [], threading.Event()
    started = time.monotonic()
    evidence = {"command": command, "environment": {key: value for key, value in environment.items()
                if key.startswith(("TS_HOST_MOE_", "MAX_CONTEXT", "KV_CACHE_DTYPE"))}, "passed": False}
    with (directory / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        def sample():
            while not stopped.is_set():
                samples.append(cache.gpu_sample())
                peak = windows_peak_working_set(process.pid)
                if peak is not None:
                    peaks.append(peak)
                stopped.wait(1)
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            evidence["exit_code"] = process.wait(timeout=timeout)
            evidence["passed"] = evidence["exit_code"] == 0
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            evidence.update(exit_code=process.returncode, failure="timeout", timeout_seconds=timeout)
        finally:
            evidence["whole_process_seconds"] = time.monotonic() - started
            stopped.set()
            thread.join(timeout=15)
    evidence["sampled_peak_working_set_bytes"] = max(peaks) if peaks else None
    (directory / "gpu.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    (directory / "run.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def ts_environment(budget):
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("TS_SPEC", "TS_MTP", "TS_Q4E_EXPERT_CACHE_")) or key in (
                "TS_CPU_MOE", "TS_N_CPU_MOE", "TS_DSV4_DSPARK", "TS_QWEN35_DFLASH", "TS_MUSE_GLIMMER_DFLASH", "TS_NEMOTRON_DFLASH"):
            environment.pop(key)
    environment.update(TS_HOST_MOE_EXPERT_CACHE_MB=str(budget), TS_HOST_MOE_EXPERT_CACHE_LAYERS="48",
                       TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS="1", TS_HOST_MOE_EXPERT_CACHE_BRIDGE="1",
                       TS_HOST_MOE_DECODE="1", TS_HOST_MOE_PIN_MAX_MB="0", TS_HOST_MOE_PIN="0")
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="first GGUF shard; sibling shards are loaded automatically")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--strata-root", type=Path, default=Path("C:/Works/Strata"))
    parser.add_argument("--strata-dependency", type=Path)
    parser.add_argument("--verified-download-report", type=Path,
                        help="verified downloader report for exactly these checkpoint paths, sizes and publisher SHA-256 values")
    parser.add_argument("--probe", type=Path, default=HERE / "Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll")
    parser.add_argument("--cache-mb", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--resident-budget-gib", type=float, default=8)
    parser.add_argument("--strata-expert-profile", type=Path,
                        help="static expert profile required for resident CPU experts; defaults to Strata's shipped data/expert-profile.bin")
    parser.add_argument("--strata-spec-window", type=int, default=2, choices=(0, 2, 3, 4, 5, 6, 7, 8),
                        help="native verifier capacity (runtime is explicitly capped to one target row); 0 requests a compatible non-verifier scalar path")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--repetitions", type=int, default=1, help="fresh processes per engine/case; rotated arm order after the first")
    parser.add_argument("--initial-cache-mb", type=int, default=0,
                        help="TensorSharp cache budget to run first; zero or one of --cache-mb, for balancing supplemental runs")
    parser.add_argument("--strata-first", action="store_true",
                        help="run Strata first using previously exported --prompt-tokens-file IDs")
    parser.add_argument("--prompt-tokens-file", type=Path,
                        help="reuse exact GGUF-rendered prompt IDs for one selected case")
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument("--dump-strata-logits", action="store_true",
                        help="opt-in stride=1 diagnostic I/O; results then include that I/O and must not be used as quiet throughput")
    args = parser.parse_args()
    if min(*args.cache_mb, args.context, args.max_new, args.repetitions, args.timeout) <= 0 or args.resident_budget_gib < 0 or len(set(args.cache_mb)) != len(args.cache_mb):
        parser.error("budgets, workloads, repetitions and timeouts must be positive")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must be unique")
    if args.initial_cache_mb not in [0] + args.cache_mb:
        parser.error("--initial-cache-mb must be zero or one of --cache-mb")
    if args.strata_first and (not args.prompt_tokens_file or args.initial_cache_mb):
        parser.error("--strata-first requires --prompt-tokens-file and the default --initial-cache-mb")
    if args.prompt_tokens_file and len(args.cases) != 1:
        parser.error("--prompt-tokens-file requires exactly one --cases value")
    root = HERE.parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "passed": False, "language_quality_validated": False, "strata_parity_validated": False,
              "cases": {}, "comparisons": {}, "runner_sha256": cache.sha(__file__), "settings": vars(args).copy(),
              "limitations": ["Deterministic semantic smoke cases do not establish general language quality.",
                  "Strata's pack converts dense projections to BF16; cross-engine bit-exact logits are not expected.",
                  "Stock Strata native IQ experts use a verifier with capacity >=2, explicitly capped to one target row without MTP or suffix drafts; observed window counts must confirm scalar execution.",
                  "Decode and prefill denominators differ between engines; compare their labelled timings and whole-process wall time.",
                  "GPU telemetry includes the desktop and all processes; sampled working set can miss short peaks.",
                  "Checkpoint shards are identified by path/size and metadata; this tool does not hash 75 GB of weights on every arm.",
                  "Cold mapping/page faults and arm order affect whole-process and first-token latency."]}
    report["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["settings"].items()}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for required in (args.model, args.strata, args.probe, args.pack / "native_experts.txt", args.pack / "conversions.json"):
            if not required.exists():
                raise ValueError("Unavailable required model/engine/pack: " + str(required))
        expert_profile = args.strata_expert_profile or args.strata_root / "data/expert-profile.bin"
        if args.resident_budget_gib > 0 and not expert_profile.is_file():
            raise ValueError("Strata resident CPU experts require an available static --strata-expert-profile")
        if expert_profile.is_file():
            report["strata_expert_profile"] = {"path": str(expert_profile.resolve()), "bytes": expert_profile.stat().st_size,
                "sha256": cache.sha(expert_profile), "source": "explicit CLI selection" if args.strata_expert_profile else "Strata stock data/expert-profile.bin"}
        report.update(strata_binary_sha256=cache.sha(args.strata), probe_assembly_sha256=cache.sha(args.probe),
                      source_revision=cache.git(root, "rev-parse", "HEAD"), source_status=cache.git(root, "status", "--short"),
                      strata_revision=cache.git(args.strata_root, "rev-parse", "HEAD"),
                      strata_status=cache.git(args.strata_root, "status", "--porcelain"),
                      ggml_revision=cache.git(root / "ExternalProjects/ggml", "rev-parse", "HEAD"),
                      ggml_status=cache.git(root / "ExternalProjects/ggml", "status", "--porcelain"))
        report["execution_modes"] = {
            "tensorsharp": {"generation": "scalar greedy", "mtp": False, "prompt_lookup_drafts": False},
            "strata": {"verifier_capacity": args.strata_spec_window, "target_window_cap": 1,
                       "mtp": False, "mtp_pack_supplied": False, "oracle_supplied": False, "suffix_draft": 0,
                       "window_histogram_diagnostic_probability": 1,
                       "prefill": "auto", "generation": "greedy", "algorithm_match_requires_observed_single_row_windows": True}}
        if report["ggml_status"] or report["strata_status"]:
            raise ValueError("Comparison requires unchanged upstream ggml and Strata source checkouts")
        if args.strata_dependency:
            report.update(strata_dependency_revision=cache.git(args.strata_dependency, "rev-parse", "HEAD"),
                          strata_dependency_status=cache.git(args.strata_dependency, "status", "--porcelain"))
            if report["strata_dependency_status"]:
                raise ValueError("Strata dependency checkout is modified")
        conversions = json.loads((args.pack / "conversions.json").read_text(encoding="utf-8"))
        report["pack"] = {"path": str(args.pack.resolve()), "conversions": conversions,
                          "small_files_sha256": {p.name: cache.sha(p) for p in args.pack.iterdir() if p.is_file() and p.stat().st_size < 16 * 1024 * 1024},
                          "large_files": [{"name": p.name, "bytes": p.stat().st_size} for p in args.pack.iterdir() if p.is_file() and p.stat().st_size >= 16 * 1024 * 1024]}
        checkpoint_paths = sorted(args.model.parent.glob(args.model.name.split("-00001-of-")[0] + "*.gguf")) if "-00001-of-" in args.model.name else [args.model]
        report["checkpoint_shards"] = [{"path": str(p.resolve()), "bytes": p.stat().st_size} for p in checkpoint_paths]
        if args.verified_download_report:
            report["verified_download"] = verified_download(args.verified_download_report, checkpoint_paths)
        if sorted((row["name"], row["size"]) for row in conversions["source_shards"]) != sorted((p.name, p.stat().st_size) for p in checkpoint_paths):
            raise ValueError("Strata pack conversion provenance identifies different checkpoint shards")
        tokenizer, eos, tokenizer_hashes = load_strata_tokenizer(args.strata_root, args.pack)
        report["tokenizer_sha256"], report["eos_token_ids"] = tokenizer_hashes, sorted(eos)
        if args.prompt_tokens_file:
            seed_prompt = [int(value) for value in re.split(r"[,\s]+", args.prompt_tokens_file.read_text().strip())]
            if not seed_prompt or len(seed_prompt) + args.max_new > args.context or any(value < 0 or value >= len(tokenizer.tokens) for value in seed_prompt):
                raise ValueError("Provided prompt IDs are empty, out of vocabulary or exceed the context")
            if CASES[args.cases[0]] not in tokenizer.decode(seed_prompt):
                raise ValueError("Provided prompt IDs do not contain the selected case's prompt")
            report["prompt_tokens_file_sha256"] = cache.sha(args.prompt_tokens_file)
        binary_identity = None
        for case in args.cases:
            report["cases"][case] = {"prompt": CASES[case], "runs": []}
            for repetition in range(args.repetitions):
                directory = output / case / str(repetition)
                directory.mkdir(parents=True)
                prompt_file, tokens_file = directory / "prompt.txt", directory / "prompt.ids"
                prompt_file.write_text(CASES[case], encoding="utf-8")
                arms, prompt_ids = {}, None
                # The initial TensorSharp arm exports the GGUF-rendered prompt.
                # Later repetitions rotate engine ordering using those same IDs.
                order = engine_order(args.cache_mb, repetition, args.initial_cache_mb, args.strata_first)
                if args.prompt_tokens_file:
                    tokens_file.write_bytes(args.prompt_tokens_file.read_bytes())
                    prompt_ids = seed_prompt
                if repetition and (output / case / "0/prompt.ids").exists():
                    tokens_file.write_bytes((output / case / "0/prompt.ids").read_bytes())
                    prompt_ids = [int(v) for v in re.split(r"[,\s]+", tokens_file.read_text().strip())]
                for arm in order:
                    print(f"Running {case} repetition {repetition + 1}: {arm}", flush=True)
                    arm_directory = directory / arm
                    if arm.startswith("tensor"):
                        budget = int(arm.removeprefix("tensor-cache"))
                        command = ["dotnet", str(args.probe.resolve()), "--output", str((arm_directory / "probe.json").resolve()),
                                   "--model", str(args.model.resolve()), "--placement", "host", "--backend", "ggml_cuda",
                                   "--generation", "greedy", "--decode-tokens", str(args.max_new), "--max-context", str(args.context),
                                   "--warmup", "0", "--iterations", "1", "--thinking", "false", "--require-cache", "1" if budget else "0"]
                        command += ["--tokens-file", str(tokens_file.resolve())] if prompt_ids else [
                            "--prompt-file", str(prompt_file.resolve()), "--prompt-tokens-output", str(tokens_file.resolve())]
                        evidence = run_process(command, root, ts_environment(budget), arm_directory, args.timeout)
                        arms[arm] = evidence
                        report["cases"][case]["runs"] = report["cases"][case]["runs"][:repetition] + [{"repetition": repetition, "arms": arms}]
                        save()
                        if not evidence["passed"] or not (arm_directory / "probe.json").exists():
                            arms[arm] = evidence
                            raise ValueError(f"{case}/{repetition}/{arm}: incomplete process ({evidence.get('failure', evidence.get('exit_code'))})")
                        data = json.loads((arm_directory / "probe.json").read_text())
                        if data.get("passed") is not True or len(data.get("runs", [])) != 1 or data["synthetic"]:
                            raise ValueError("Missing completed trained TensorSharp run")
                        prompt_ids = prompt_ids or data["prompt_tokens"]
                        if data["prompt_tokens"] != prompt_ids:
                            raise ValueError("TensorSharp arms received different prompt tokens")
                        identity = (data["native_sha256"], data["managed_assemblies_sha256"])
                        if binary_identity is not None and identity != binary_identity:
                            raise ValueError("Mapped TensorSharp native/managed binaries differ between arms")
                        binary_identity = identity
                        if cache.sha(data["native_path"]) != data["native_sha256"]:
                            raise ValueError("Mapped native binary changed during comparison")
                        for path, digest in data["managed_assemblies_sha256"].items():
                            if cache.sha(path) != digest:
                                raise ValueError("Loaded managed binary changed during comparison")
                        row = data["runs"][0]
                        ids = data["generated_tokens"]
                        if not ids or not all(0 <= token < len(tokenizer.tokens) for token in ids):
                            raise ValueError("Invalid or absent TensorSharp output IDs")
                        complete = row["finish_reason"] == "eos" and ids[-1] in eos
                        decoded = tokenizer.decode(ids[:-1] if complete else ids)
                        if data["selected_text"] != decoded:
                            raise ValueError("TensorSharp and Strata tokenizer decoders disagree on identical output IDs")
                        if not data["final_logits"] or len(data["final_logits"]) != len(tokenizer.tokens) or not all(math.isfinite(v) for v in data["final_logits"]):
                            raise ValueError("Incomplete/nonfinite full TensorSharp vocabulary logits")
                        evidence.update(row, prompt_tokens=prompt_ids, generated_tokens=ids, selected_text=decoded,
                                        complete=complete, model_load_ms=data["model_load_ms"], cache_stats=data["cache_stats"],
                                        process_peak_working_set_bytes=data["process_peak_working_set_bytes"],
                                        native_sha256=data["native_sha256"], managed_assemblies_sha256=data["managed_assemblies_sha256"],
                                        final_logit_sha256=row["final_logit_sha256"], final_logits=data["final_logits"],
                                        time_to_first_token_ms=row["prefill_ms"],
                                        decode_denominator="Forward calls after the first generated token; excludes that first token",
                                        prefill_denominator="all prompt tokens", ttft_scope="model-loaded prefill; excludes load")
                    else:
                        dump_path = arm_directory / "logits.bin"
                        # The standalone executable enters generate directly.
                        command = [str(args.strata.resolve()), "--pack", str(args.pack.resolve()), "--native", str(args.model.resolve()),
                                   "--tokens-file", str(tokens_file.resolve()), "--max-context", str(args.context), "--max-new", str(args.max_new),
                                   "--kv", "fp16", "--expert-cache", "auto", "--expert-cache-per-layer", "--resident-budget-gib", str(args.resident_budget_gib),
                                   "--prefill", "auto", "--greedy", "--stop-eos"]
                        command += ["--suffix-draft", "0"]
                        if args.strata_spec_window:
                            command += ["--spec", str(args.strata_spec_window), "--mtp-max-t", "1", "--spec-min-p", "1"]
                        if args.dump_strata_logits:
                            command += ["--dump-logits", str(dump_path.resolve())]
                        if expert_profile.is_file():
                            command += ["--expert-profile", str(expert_profile.resolve())]
                        evidence = run_process(command, args.strata_root, os.environ.copy(), arm_directory, args.timeout)
                        arms[arm] = evidence
                        report["cases"][case]["runs"] = report["cases"][case]["runs"][:repetition] + [{"repetition": repetition, "arms": arms}]
                        save()
                        if not evidence["passed"]:
                            arms[arm] = evidence
                            raise ValueError(f"{case}/{repetition}/strata: incomplete process ({evidence.get('failure', evidence.get('exit_code'))})")
                        strata_log = (arm_directory / "process.log").read_text(encoding="utf-8", errors="replace")
                        evidence.update(parse_strata_log(strata_log, prompt_ids, args.max_new))
                        if args.strata_spec_window:
                            evidence["observed_execution"] = parse_strata_scalar_execution(strata_log, args.strata_spec_window, len(evidence["generated_tokens"]))
                        ids = evidence["generated_tokens"]
                        if not all(0 <= token < len(tokenizer.tokens) for token in ids):
                            raise ValueError("Strata output IDs exceed checkpoint vocabulary")
                        complete = ids[-1] in eos
                        evidence.update(complete=complete, selected_text=tokenizer.decode(ids[:-1] if complete else ids),
                                        finish_reason="eos" if complete else "length", ttft_scope="engine-reported prefill and first target token; includes resident-expert setup inside generation, excludes earlier pack/cache initialization",
                                        execution_mode=report["execution_modes"]["strata"])
                        if args.dump_strata_logits:
                            try:
                                evidence["logits_dump"] = parse_strata_final_logits(dump_path, len(tokenizer.tokens), len(prompt_ids), len(ids), args.max_new)
                            except (OSError, ValueError) as error:
                                evidence["logits_dump"] = {"available": False, "reason": str(error)}
                    evidence["semantic_check"] = semantic_check(case, evidence["selected_text"], evidence["complete"])
                    evidence["passed"] = evidence["passed"] and evidence["semantic_check"]["passed"]
                    arms[arm] = evidence
                    report["cases"][case]["runs"] = report["cases"][case]["runs"][:repetition] + [{"repetition": repetition, "arms": arms}]
                    save()
                def compare_ids(left, right):
                    a, b = left["generated_tokens"], right["generated_tokens"]
                    return {"same_output_ids": a == b, "first_id_divergence": next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)) if len(a) != len(b) else None),
                            "both_semantic_checks_pass": left["semantic_check"]["passed"] and right["semantic_check"]["passed"]}
                key = f"{case}/{repetition}"
                report["comparisons"][key] = {}
                for budget in args.cache_mb:
                    cached = arms[f"tensor-cache{budget}"]
                    comparison = {"cache_vs_disabled": compare_ids(cached, arms["tensor-cache0"]),
                                  "cache_vs_strata": compare_ids(cached, arms["strata"]), "cross_engine_bit_exact_claim": False}
                    report["comparisons"][key][str(budget)] = comparison
                    if cached["generated_tokens"] == arms["tensor-cache0"]["generated_tokens"]:
                        comparison["cache_final_logits_same_context"] = cache.difference(
                            [cached["final_logits"]], [arms["tensor-cache0"]["final_logits"]])
                    dumped = arms["strata"].get("logits_dump", {})
                    if "final_logits" in dumped and cached["generated_tokens"] == arms["strata"]["generated_tokens"]:
                        comparison["strata_final_logits_same_context"] = cache.difference(
                            [cached["final_logits"]], [dumped["final_logits"]])
                save()
        report["passed"] = completed_suite(report["cases"], args.cases,
            ["tensor-cache0", "strata"] + [f"tensor-cache{budget}" for budget in args.cache_mb], args.repetitions)
        report["language_quality_validated"] = False
        # This labels only semantic smoke parity. A performance acceptance claim
        # needs repeated matched throughput/latency and memory evidence.
        report["semantic_smoke_parity_validated"] = report["passed"]
        report["strata_parity_validated"] = False
        expected_arms = ["tensor-cache0", "strata"] + [f"tensor-cache{budget}" for budget in args.cache_mb]
        report["timing_summary"] = {}
        for arm in expected_arms:
            rows = [run["arms"][arm] for case in report["cases"].values() for run in case["runs"]]
            report["timing_summary"][arm] = {"completed_runs": len(rows),
                **{"median_" + metric: statistics.median(row[metric] for row in rows) for metric in (
                    "whole_process_seconds", "prefill_ms", "decode_ms", "decode_tps", "time_to_first_token_ms")},
                "decode_denominator": rows[0]["decode_denominator"],
                "scope": "aggregate of the selected semantic cases; output lengths can differ"}
        if cache.sha(args.strata) != report["strata_binary_sha256"] or cache.git(root / "ExternalProjects/ggml", "status", "--porcelain"):
            raise ValueError("Engine binary or upstream checkout changed during comparison")
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as error:
        report.update(passed=False, language_quality_validated=False, semantic_smoke_parity_validated=False, error=str(error))
    save()
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json"), "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
