#!/usr/bin/env python3
"""Compare scalar Qwen expert execution using identical consumed token sequences.

Quality must be checked separately with normal generation. Strata prefills N-1
prompt tokens, then consumes the last prompt token in its scalar verifier. Give
TensorSharp that same N-1 prompt and force [last_prompt] + continuation. Strata
emits continuation + sentinel, so both engines consume precisely the same IDs
and perform the same number of prefill and decode target evaluations. The final
sentinel is emitted but never consumed. No upstream engine is modified.

The four fresh processes run ABBA or BAAB. OS page-cache state is not cleared;
these are not guaranteed cold-storage runs. Placement knobs do not establish
equal total physical RAM/VRAM limits. No output-quality or bit-exactness claim
is derived from following supplied tokens.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("strata_semantic_compare", HERE / "qwen38-strata-compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def read_ids(path):
    values = Path(path).read_text(encoding="utf-8-sig").strip()
    ids = [int(value) for value in re.split(r"[,\s]+", values)] if values else []
    if not ids or any(value < 0 for value in ids):
        raise ValueError("Token files must contain nonnegative integer IDs")
    return ids


def conditioning(prompt, continuation, sentinel=None):
    if len(prompt) < 2 or not continuation:
        raise ValueError("A nonempty batched prefix and continuation are required")
    sentinel = continuation[-1] if sentinel is None else sentinel
    if any(type(token) is not int or token < 0 for token in list(prompt) + list(continuation) + [sentinel]):
        raise ValueError("Invalid token IDs")
    return {"strata_prompt": list(prompt), "tensor_prompt": list(prompt[:-1]),
            "tensor_forced": [prompt[-1]] + list(continuation),
            "strata_follow": list(continuation) + [sentinel],
            "consumed_tokens": list(prompt) + list(continuation),
            "prefill_tokens": len(prompt) - 1, "decode_tokens": len(continuation) + 1}


def validate_tensor(data, plan):
    if data.get("passed") is not True or data.get("synthetic") is not False or data.get("decode_mode") != "teacher-forced":
        raise ValueError("Missing completed trained teacher-forced TensorSharp run")
    if data.get("prompt_tokens") != plan["tensor_prompt"] or data.get("forced_tokens") != plan["tensor_forced"]:
        raise ValueError("TensorSharp consumed a different trajectory")
    runs = data.get("runs", [])
    if len(runs) != 1 or runs[0].get("warmup") is not False:
        raise ValueError("Expected one measured TensorSharp run without warmup")
    row = runs[0]
    if row.get("finish_reason") != "teacher-forced" or row.get("prefill_tokens") != plan["prefill_tokens"] or row.get("decode_tokens") != plan["decode_tokens"]:
        raise ValueError("TensorSharp timing denominators do not match the consumed trajectory")
    if len(row.get("decode_step_ms", [])) != plan["decode_tokens"] or row.get("logit_rows") != plan["decode_tokens"] + 1:
        raise ValueError("Missing target evaluations or complete step timings")
    if any(not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0 for ms in row["decode_step_ms"]):
        raise ValueError("Invalid target step timings")
    logits = data.get("final_logits", [])
    if not logits or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in logits):
        raise ValueError("Missing or nonfinite target logits")
    return checked_timings(row, plan)


def validate_strata(log, plan):
    row = compare.parse_strata_log(log, plan["strata_prompt"], len(plan["strata_follow"]))
    if row["generated_tokens"] != plan["strata_follow"]:
        raise ValueError("Strata did not follow the exact supplied continuation")
    row["observed_execution"] = compare.parse_strata_scalar_execution(log, 2, plan["decode_tokens"])
    follow = re.findall(r"^follow\s+(\d+) of (\d+) emitted tokens differ from the argmax$", log, re.MULTILINE)
    if len(follow) != 1 or int(follow[0][1]) != plan["decode_tokens"] or not 0 <= int(follow[0][0]) <= plan["decode_tokens"]:
        raise ValueError("Missing or inconsistent forced-emission evidence")
    row["forced_argmax_disagreements"] = int(follow[0][0])
    return checked_timings(row, plan)


def checked_timings(row, plan):
    row = dict(row)
    for stage in ("prefill", "decode"):
        if row.get(stage + "_tokens") != plan[stage + "_tokens"]:
            raise ValueError("Different stage token count")
        ms = row.get(stage + "_ms")
        if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            raise ValueError("Missing or nonpositive stage time")
        # Recompute from the exact count instead of trusting rounded printed TPS.
        row[stage + "_tps"] = 1000 * plan[stage + "_tokens"] / ms
    row["combined_target_ms"] = row["prefill_ms"] + row["decode_ms"]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "model", "pack", "strata", "strata-dependency", "prompt-tokens-file", "continuation-tokens-file"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--strata-root", type=Path, default=Path("C:/Works/Strata"))
    p.add_argument("--strata-dependency-manifest", type=Path)
    p.add_argument("--probe", type=Path, default=HERE / "Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll")
    p.add_argument("--cache-mb", type=int, default=8192)
    p.add_argument("--resident-budget-gib", type=float, default=8)
    p.add_argument("--strata-vram-reserve-mib", type=int, default=2048)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--sentinel", type=int)
    p.add_argument("--order", choices=("ABBA", "BAAB"), default="ABBA", help="A=TensorSharp, B=Strata")
    p.add_argument("--timeout", type=float, default=1200)
    args = p.parse_args()
    if min(args.cache_mb, args.resident_budget_gib, args.cpu_threads, args.timeout) <= 0 or not all(math.isfinite(value) for value in (args.resident_budget_gib, args.timeout)) or min(args.context, args.strata_vram_reserve_mib) < 0:
        p.error("Positive budgets/workers/timeouts and nonnegative context/reserve required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "passed": False, "language_quality_validated": False,
              "physical_budget_matched": False, "cross_engine_bit_exact_claim": False,
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "runs": [], "runner_sha256": compare.cache.sha(__file__),
              "limitations": ["Forced continuation is not answer quality; qualify normal generation separately.",
                  "Placement controls are not common total physical hard limits; physical_budget_matched remains false.",
                  "Dense BF16 pack rounding changes arithmetic; source/pack provenance is retained.",
                  "OS file caches are not cleared; fresh process does not imply cold storage.",
                  "Equal consumed tokens and target calls do not equalize engine initialization or first-token timer scope.",
                  "GPU samples include other applications; sampled host working set is not full resource ownership."]}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for path in (args.model, args.probe, args.strata, args.pack / "native_experts.txt", args.pack / "conversions.json", args.strata_root / "data/expert-profile.bin"):
            if not path.is_file():
                raise ValueError("Missing model, pack or built executable: " + str(path))
        source = HERE.parents[1]
        report["sources"] = {}
        for key, path in (("strata", args.strata_root), ("tensorsharp_ggml", source / "ExternalProjects/ggml")):
            row = {"revision": compare.cache.git(path, "rev-parse", "HEAD"), "status": compare.cache.git(path, "status", "--porcelain")}
            if row["status"]:
                raise ValueError("Modified upstream checkout: " + str(path))
            report["sources"][key] = row
        pin = (args.strata_root / "third_party/ggml/VERSION.txt").read_text().split()[0]
        report["sources"]["strata_dependency"] = compare.dependency_identity(
            args.strata_dependency, pin, args.strata_dependency_manifest)
        report["source_revision"] = compare.cache.git(source, "rev-parse", "HEAD")
        report["source_status"] = compare.cache.git(source, "status", "--short")
        plan = conditioning(read_ids(args.prompt_tokens_file), read_ids(args.continuation_tokens_file), args.sentinel)
        tokenizer, eos, tokenizer_hashes = compare.load_strata_tokenizer(args.strata_root, args.pack)
        if any(token >= len(tokenizer.tokens) for token in plan["consumed_tokens"] + plan["strata_follow"]):
            raise ValueError("A supplied token exceeds the checkpoint vocabulary")
        context = args.context or len(plan["strata_prompt"]) + len(plan["strata_follow"])
        if context < len(plan["strata_prompt"]) + len(plan["strata_follow"]):
            raise ValueError("Context must fit the prompt and complete emitted continuation")
        report.update(conditioning=plan, context=context, tokenizer_sha256=tokenizer_hashes, eos_ids=sorted(eos))
        conversions = json.loads((args.pack / "conversions.json").read_text(encoding="utf-8"))
        shards = sorted(args.model.parent.glob(args.model.name.split("-00001-of-")[0] + "*.gguf")) if "-00001-of-" in args.model.name else [args.model]
        if sorted((row["name"], row["size"]) for row in conversions["source_shards"]) != sorted((path.name, path.stat().st_size) for path in shards):
            raise ValueError("Pack provenance identifies different checkpoint shards")
        report["conversions"] = conversions
        report["checkpoint_shards"] = [{"path": str(path.resolve()), "bytes": path.stat().st_size} for path in shards]
        report["binary_sha256"] = {str(path.resolve()): compare.cache.sha(path) for path in (args.probe, args.strata)}
        input_paths = {}
        for key in ("strata_prompt", "tensor_prompt", "tensor_forced", "strata_follow"):
            path = output / (key + ".ids")
            path.write_text(",".join(map(str, plan[key])) + "\n", encoding="utf-8")
            input_paths[key] = str(path)
        tensor_identity = None
        for index, arm in enumerate(args.order):
            directory = output / f"{index}-{arm}"
            if arm == "A":
                command = ["dotnet", str(args.probe.resolve()), "--model", str(args.model.resolve()),
                           "--output", str(directory / "probe.json"), "--placement", "host", "--backend", "ggml_cuda",
                           "--tokens-file", input_paths["tensor_prompt"], "--forced-tokens-file", input_paths["tensor_forced"],
                           "--decode-tokens", str(plan["decode_tokens"]), "--max-context", str(context),
                           "--generation", "teacher-forced", "--warmup", "0", "--iterations", "1", "--require-cache", "1"]
                row = compare.run_process(command, source, compare.ts_environment(args.cache_mb, args.cpu_threads), directory, args.timeout)
            else:
                command = [str(args.strata.resolve()), "--pack", str(args.pack.resolve()), "--native", str(args.model.resolve()),
                           "--tokens-file", input_paths["strata_prompt"], "--spec-follow", input_paths["strata_follow"],
                           "--max-context", str(context), "--max-new", str(plan["decode_tokens"]), "--kv", "fp16",
                           "--prefill", "auto", "--greedy", "--expert-cache", "auto", "--expert-cache-per-layer",
                           "--expert-profile", str((args.strata_root / "data/expert-profile.bin").resolve()),
                           "--suffix-draft", "0", "--spec", "2", "--mtp-max-t", "1", "--spec-min-p", "1"]
                command += compare.strata_resource_arguments(args.resident_budget_gib, args.cpu_threads, args.strata_vram_reserve_mib)
                row = compare.run_process(command, args.strata_root, compare.strata_environment(), directory, args.timeout)
            report["runs"].append(dict(row, arm=arm, trajectory_validated=False))
            save()
            if not row["passed"]:
                raise ValueError(f"Incomplete {arm} process: {row.get('failure', row.get('exit_code'))}")
            if arm == "A":
                data = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
                timings = validate_tensor(data, plan)
                identity = (data["native_sha256"], data["managed_assemblies_sha256"])
                if tensor_identity is not None and tensor_identity != identity:
                    raise ValueError("TensorSharp loaded different binaries between arms")
                tensor_identity = identity
                loaded = dict(data["managed_assemblies_sha256"])
                loaded[data["native_path"]] = data["native_sha256"]
                if any(compare.cache.sha(path).lower() != sha.lower() for path, sha in loaded.items()):
                    raise ValueError("A loaded TensorSharp binary changed")
                timings.update(native_sha256=data["native_sha256"], managed_assemblies_sha256=data["managed_assemblies_sha256"],
                               model_load_ms=data["model_load_ms"], cache_stats=data["cache_stats"],
                               process_peak_working_set_bytes=data["process_peak_working_set_bytes"])
            else:
                timings = validate_strata((directory / "process.log").read_text(encoding="utf-8", errors="replace"), plan)
            report["runs"][-1].update(timings)
            report["runs"][-1]["trajectory_validated"] = True
            save()
        if any(compare.cache.sha(path) != sha for path, sha in report["binary_sha256"].items()):
            raise ValueError("A benchmark executable changed")
        report["medians"] = {arm: {key: statistics.median(row[key] for row in report["runs"] if row["arm"] == arm)
                                  for key in ("prefill_ms", "decode_ms", "prefill_tps", "decode_tps", "combined_target_ms", "whole_process_seconds")}
                             for arm in ("A", "B")}
        report["passed"] = True
        save()
        return 0
    except Exception as error:
        report.update(passed=False, failure=str(error))
        save()
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
