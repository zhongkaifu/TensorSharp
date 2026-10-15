#!/usr/bin/env python3
"""Fresh Linux native/component and managed GLM qualification, never benchmarks.

Use --plan on any OS to inspect arguments without loading a native library.
Execution requires a successful pinned build receipt, Linux ELF binaries, and a
new D/E output directory. Native source/build directories are never modified.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def cpu_cases():
    cases = [("cache-identity-cpu", "GgmlOpsHostMoeCacheIdentityTest", []),
             ("glm-host-mapping-cpu", "GgmlOpsGlmHostMappingTest", []),
             ("config-cpu", "GgmlOpsHostMoeCacheConfigTest", []),
             ("policy-cpu", "GgmlOpsHostMoeCachePolicyTest", []),
             ("pipeline-config-cpu", "GgmlOpsHostMoeFilePipelineConfigTest", []),
             ("pipeline-plan-cpu", "GgmlOpsHostMoeFilePipelinePlanTest", []),
             ("glm-evidence-cpu", "GgmlOpsGlmBatchedExecutionTest", []),
             ("glm-vendor-types-cpu", "GgmlOpsGlmGraphBudgetTest", []),
             ("glm-expert-boundary-cpu", "GgmlOpsGlmExpertBoundaryTest", []),
             ("expert-cpu", "GgmlOpsHostMoeExpertCacheTest", ["--cpu"]),
             ("file-batch-cpu", "GgmlOpsHostMoeFileBatchTest", ["--pipeline-scope", "0"]),
             ("mixed-file-batch-cpu", "GgmlOpsHostMoeFileBatchTest", ["--mixed", "--pipeline-scope", "0"])]
    return cases


def cuda_cases():
    expert, file, parallel = ("GgmlOpsHostMoeExpertCacheTest", "GgmlOpsHostMoeFileBatchTest",
                              "GgmlOpsHostMoeFileBatchParallelTest")
    return [("ordinary-q8", expert, []), ("arena-q8", expert, ["--arena"]),
            ("arena-iq1", expert, ["--arena", "--quantization", "iq1_m"]),
            ("arena-iq2", expert, ["--arena", "--quantization", "iq2_xxs"]),
            ("arena-file-q8", expert, ["--arena", "--file-read"]),
            ("arena-file-iq1", expert, ["--arena", "--file-read", "--quantization", "iq1_m"]),
            ("pipeline-off-q8", file, ["--cuda", "--pipeline-scope", "0"]),
            ("pipeline-on-q8", file, ["--cuda", "--pipeline-scope", "1"]),
            ("pipeline-on-mixed", file, ["--cuda", "--mixed", "--pipeline-scope", "1"]),
            ("pipeline-parallel-q8", parallel, ["--cuda", "--pipeline-scope", "1"]),
            ("pipeline-parallel-mixed", parallel, ["--cuda", "--mixed", "--pipeline-scope", "1"]),
            ("capacity-growth-q8", expert, ["--capacity-growth"]),
            ("capacity-growth-iq1", expert, ["--capacity-growth", "--quantization", "iq1_m"]),
            ("capacity-growth-iq2", expert, ["--capacity-growth", "--quantization", "iq2_xxs"]),
            ("bulk-union-q8", expert, ["--bulk-union"]),
            ("bulk-union-iq1", expert, ["--bulk-union", "--quantization", "iq1_m"]),
            ("bulk-union-iq2", expert, ["--bulk-union", "--quantization", "iq2_xxs"]),
            ("scalar-large-file-q8", expert, ["--glm-scalar-file-window"]),
            ("scalar-large-file-iq1", expert, ["--glm-scalar-file-window", "--quantization", "iq1_m"]),
            ("pooled-largest-quota-q8", expert, ["--pooled-quota"]),
            ("pooled-largest-quota-iq1", expert, ["--pooled-quota", "--quantization", "iq1_m"])]


def glm_cases(quantized_source):
    quant = "--quantized-source=" + str(quantized_source)
    cases = [("production-cpu", "production", "CPU", []),
             ("production-cuda", "production", "CUDA", []),
             ("hooks-cpu-faults", "hooks", "CPU", ["--dispatch-faults"]),
             ("hooks-cuda-faults", "hooks", "CUDA", ["--dispatch-faults"])]
    for mode in ("production", "hooks"):
        faults = ["--dispatch-faults"] if mode == "hooks" else []
        for clamp in (False, True):
            cases.append((mode + "-q8-" + ("clamp" if clamp else "zero"), mode, "CUDA",
                          ["--selected-experts"] + faults + (["--swiglu-clamp"] if clamp else [])))
        for clamp in (False, True):
            cases.append((mode + "-iq-" + ("clamp" if clamp else "checkpoint"), mode, "CUDA",
                          ["--selected-experts", quant] + faults + (["--swiglu-clamp"] if clamp else [])))
        for rows in (8, 9, 10, 16, 128, 256, 384, 512, 1024):
            for kind in ("q8", "iq"):
                cases.append((mode + "-" + kind + "-rows" + str(rows), mode, "CUDA",
                              ["--selected-experts"] + ([quant] if kind == "iq" else [])
                              + ["--prefill-rows=" + str(rows)] + faults))
    for mode in ("production", "hooks"):
        for kind in ("q8", "iq"):
            for rows in (512, 809):
                cases.append((f"{mode}-{kind}-bf16-sensitive-rows{rows}", mode, "CUDA",
                              ["--selected-experts", "--bf16-sensitive", f"--prefill-rows={rows}"]
                              + ([quant] if kind == "iq" else [])
                              + (["--dispatch-faults"] if mode == "hooks" else [])))
    return cases


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require_elf(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise RuntimeError("Linux qualification requires an actual ELF executable/library: " + str(path))
    return path


def native_pins(root):
    return {str(p.relative_to(root)): sha(p) for branch in ("production", "components")
            for p in (root / branch).iterdir() if p.is_file()
            and (p.name == "libGgmlOps.so" or p.name.startswith("GgmlOps"))}


def verify_native_build(root):
    receipt = read_json(root / "build-summary.json")
    if (receipt.get("passed") is not True or receipt.get("production_no_test_hooks") is not True
            or receipt.get("before") != receipt.get("after")
            or receipt.get("libraries_before") != receipt.get("libraries_after")):
        raise RuntimeError("Successful unchanged-source/dependency build evidence is required.")
    for field, directory in (("production_artifacts", "production"), ("hooks_artifacts", "components")):
        expected = receipt.get(field, {})
        if "libGgmlOps.so" not in expected:
            raise RuntimeError("Missing pinned native artifacts: " + field)
        for name, value in expected.items():
            if Path(name).name != name or sha(root / directory / name).lower() != value.lower():
                raise RuntimeError("Native artifact no longer matches its build receipt: " + name)
    return sha(root / "build-summary.json")


def abi_evidence(native, environment):
    # ELF loaders can cache their search path before Python changes os.environ.
    # A fresh child sees LD_LIBRARY_PATH and CUDA visibility before dlopen.
    native = require_elf(native)
    expected = {"TSGgml_GlmUnifiedBudgetVersion": 2, "TSGgml_GlmSelectedExpertAdapterVersion": 7,
                "TSGgml_HostMoeFilePipelineVersion": 2, "TSGgml_Qwen4ExpArenaBudgetKind": 5,
                "TSGgml_HostMoeExpertCacheBudgetKind": 4}
    code = ("import ctypes,json,sys; lib=ctypes.CDLL(sys.argv[1]); "
            "names=json.loads(sys.argv[2]); "
            "print(json.dumps({name:getattr(lib,name)() for name in names}))")
    actual = json.loads(subprocess.check_output([sys.executable, "-c", code, str(native), json.dumps(expected)],
                                               env=environment, text=True, timeout=30))
    if actual != expected:
        raise RuntimeError("Native capability mismatch: " + repr(actual))
    return {"path": str(native), "sha256": sha(native), "capabilities": actual}


def validate_glm_report(actual, comparison, native, native_sha, backend, flags):
    """Fail closed; report booleans alone do not certify complete execution."""
    if not actual or not comparison:
        raise RuntimeError("GLM report and full-logit comparison must both exist.")
    if (actual.get("passed") is not True or actual.get("full_logits_bit_exact") is not True
            or actual.get("shared_graph_owner") is not True or actual.get("backend") != backend
            or comparison.get("passed") is not True or comparison.get("differing_elements") != 0
            or comparison.get("max_abs") != 0 or comparison.get("rmse") != 0):
        raise RuntimeError("Full-logit/owner/backend acceptance failed.")
    elements = actual.get("logit_elements", 0)
    if (type(elements) is not int or elements <= 0 or comparison.get("logit_elements") != elements
            or not actual.get("logits_sha256") or actual["logits_sha256"] != comparison.get("logits_sha256")
            or actual["logits_sha256"] != comparison.get("reference_sha256")
            or actual.get("checks", 0) <= 0):
        raise RuntimeError("Missing or contradictory complete-logit evidence.")
    selected, faults = "--selected-experts" in flags, "--dispatch-faults" in flags
    if actual.get("selected_experts_tested") is not selected or actual.get("dispatch_faults_tested") is not faults:
        raise RuntimeError("Selected-expert/fault coverage does not match requested mode.")
    bf16 = '--bf16-sensitive' in flags
    if (actual.get('bf16_sensitive_projections', False) is not bf16
            or type(actual.get('bf16_sensitive_tensor_count', 0)) is not int
            or actual.get('bf16_sensitive_tensor_count', 0) != (18 if bf16 else 0)
            or comparison.get('bf16_sensitive_projections', False) is not bf16
            or type(comparison.get('bf16_sensitive_tensor_count', 0)) is not int
            or comparison.get('bf16_sensitive_tensor_count', 0) != (18 if bf16 else 0)):
        raise RuntimeError('Requested actual BF16 recurrent/mixing fixture tensor coverage is missing or contradictory.')
    if selected and faults:
        id_failures = [row for row in actual.get('executions', []) if 'selected_ids_download_failure' in row]
        if len(id_failures) != 1 or id_failures[0].get('selected_ids_download_failure') is not True \
                or type(id_failures[0].get('stage')) is not int or id_failures[0]['stage'] != 4 \
                or type(id_failures[0].get('kind')) is not int or id_failures[0]['kind'] != 1 \
                or id_failures[0].get('cache_unchanged') is not True or id_failures[0].get('logits_unpublished') is not True \
                or id_failures[0].get('heads_unpublished') is not True:
            raise RuntimeError('Selected-ID reported download error must stop before expert cache mutation or logits publication.')
        drain_failures = [row for row in actual.get('executions', []) if 'selected_final_drain_failure' in row]
        expected_seams = 3 if bf16 or any(f.startswith('--quantized-source=') for f in flags) else 2
        if len(drain_failures) != 1:
            raise RuntimeError('Missing or duplicated last selected-seam reported drain failure proof.')
        drain = drain_failures[0]
        expected_counts = {'stage': 5, 'kind': 1, 'participating_rows': 3, 'expert_seams': expected_seams,
                           'completed_helper_rows': 3 * expected_seams}
        if (any(type(drain.get(key)) is not int or drain[key] != value for key, value in expected_counts.items())
                or any(drain.get(key) is not True for key in (
                    'selected_final_drain_failure', 'successful_graphs_unchanged', 'logits_unpublished',
                    'heads_unpublished', 'slots_fenced', 'reset_recovered_n3', 'primary_departure_recovered_n2'))):
            raise RuntimeError('Reported final drain error must follow every N3 expert helper, fence all slots, '
                               'reject logits/heads/success publication and recover reset N3/departure N2.')
    quant = next((f.split("=", 1)[1] for f in flags if f.startswith("--quantized-source=")), None)
    rows = next((int(f.split("=", 1)[1]) for f in flags if f.startswith("--prefill-rows=")), None)
    if actual.get("prefill_rows") != rows or actual.get("quantized_source") != quant:
        raise RuntimeError("Requested prefill/quantized source was not exercised.")
    clamp = 0.0001 if "--swiglu-clamp" in flags else 10 if quant is not None else 0
    if actual.get("swiglu_clamp") != clamp or actual.get("expert_groups") != (3 if quant else 1):
        raise RuntimeError("Fixture clamp/layout coverage differs from requested mode.")
    loaded = actual.get("native", [])
    if (len(loaded) != 1 or Path(loaded[0].get("path", "")).resolve() != Path(native).resolve()
            or loaded[0].get("sha256", "").lower() != native_sha.lower()):
        raise RuntimeError("GLM loaded an unexpected native module.")
    final = actual.get("final_budget", [])
    if not final or any(p.get("Committed") != 0 or p.get("Reserved") != 0 for p in final):
        raise RuntimeError("Covered physical allocation ownership did not settle.")
    executions = actual.get("executions", [])
    joint = [e["actual"] for e in executions if "actual" in e and e["actual"].get("Rows") == 3]
    departed = [e["row_departure"] for e in executions if "row_departure" in e]
    # Probe Program.cs executes primary-departure N2 only inside its selected
    # expert branch. Plain shared-graph cases still require actual N3 evidence.
    if len(joint) < 4 or (selected and not departed) or any(d.get("Rows") != 2 for d in departed):
        raise RuntimeError("Missing actual N3 or primary-departure N2 evidence.")
    for execution in joint + departed:
        width = execution["Rows"]
        before, after = execution.get("PositionsBefore", []), execution.get("PositionsAfter", [])
        if (execution.get("ExecutionId", 0) <= 0 or execution.get("Nodes", 0) <= 0
                or len(before) != width or len(after) != width
                or execution.get("BackendSplits", 0) <= 0
                or len(set(execution.get("Slots", []))) != width
                or any(b + 1 != a for b, a in zip(before, after))):
            raise RuntimeError("Invalid native graph identity/slot-head advancement.")
    if selected and not any(e.get("minimum_cache_bytes", 0) > 0 for e in executions):
        raise RuntimeError("Selected tier did not measure its exact complete-union minimum.")


def run_case(command, environment, log, timeout):
    start = time.monotonic()
    with log.open("xb") as stream:
        try:
            code = subprocess.run(command, env=environment, cwd=log.parent, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    skipped = bool(re.search(r"(?im)^\s*(?:SKIP(?:PED)?|Skipping)\b", log.read_text(errors="replace")))
    return {"command": command, "exit_code": code, "seconds": time.monotonic() - start,
            "skipped_or_unavailable": skipped, "log": str(log), "passed": code == 0 and not skipped}


def annotate_requested_native_coverage(name, case, log_text):
    # Native16 guards file_read_checks with _WIN32. On Linux these two commands
    # otherwise exit zero after ordinary arena checks without testing the
    # requested registered-file extents. Require that subtest's actual marker;
    # a future portable implementation can qualify without relaxing this gate.
    if name in ("arena-file-q8", "arena-file-iq1"):
        marker = "PASS: exact file extents, cached reuse, out-of-order staging, partial-read recovery and source release"
        exercised = marker in log_text
        case["registered_file_extents_exercised"] = exercised
        if not exercised:
            case.update(passed=False, skipped_or_unavailable=True,
                        coverage_limitation="Requested file_read_checks was not exercised; Native16 Linux compiles its _WIN32-only invocation out, despite exit-zero ordinary arena checks.")
    return case


def prepare_managed(source, manifest, output, native_root, expected_probe):
    record = read_json(manifest)
    expected = record.get("managed_before", {})
    if (record.get("passed") is not True or expected != record.get("managed_after") or len(expected) != 24
            or expected.get("GlmGraphBudgetProbe.dll", "").lower() != expected_probe.lower()):
        raise RuntimeError("The frozen 24-file managed dependency closure/probe pin is missing.")
    source_files = {p.name for p in source.iterdir() if p.is_file()}
    if source_files != set(expected):
        raise RuntimeError("Managed source file closure changed.")
    for name, value in expected.items():
        if Path(name).name != name or sha(source / name).lower() != value.lower():
            raise RuntimeError("Frozen managed source changed: " + name)
    copied = {}
    for mode, branch in (("production", "production"), ("hooks", "components")):
        deployment = output / (mode + "-bin")
        deployment.mkdir()
        for name, value in expected.items():
            shutil.copy2(source / name, deployment / name)
            if sha(deployment / name).lower() != value.lower():
                raise RuntimeError("Managed deployment copy mismatch: " + name)
        shutil.copy2(native_root / branch / "libGgmlOps.so", deployment / "libGgmlOps.so")
        if sha(deployment / "libGgmlOps.so") != sha(native_root / branch / "libGgmlOps.so"):
            raise RuntimeError("Native deployment copy mismatch: " + mode)
        copied[mode] = {p.name: sha(p) for p in deployment.iterdir() if p.is_file()}
    return {"source": str(source), "source_manifest": str(manifest), "source_manifest_sha256": sha(manifest),
            "source_files": expected, "deployments": copied}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "cuda", "glm"), required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, default=Path("/usr/local/cuda-12.6"))
    parser.add_argument("--dotnet", type=Path, default=Path("/usr/bin/dotnet"))
    parser.add_argument("--managed-source", type=Path)
    parser.add_argument("--managed-manifest", type=Path)
    parser.add_argument("--probe-sha256")
    parser.add_argument("--quantized-source", type=Path)
    parser.add_argument("--only-case", action="append", default=[],
                        help="Execute only these explicitly named cases; no whole-matrix claim.")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "glm" and not all((args.managed_source, args.managed_manifest,
                                      args.probe_sha256, args.quantized_source)):
        parser.error("GLM requires managed source/manifest, explicit probe SHA256 and quantized source.")
    cases = {"cpu": cpu_cases, "cuda": cuda_cases}.get(args.mode, lambda: glm_cases(args.quantized_source))()
    if args.only_case:
        if len(set(args.only_case)) != len(args.only_case) or set(args.only_case) - {c[0] for c in cases}:
            parser.error("Each --only-case must name a distinct case in the selected matrix.")
        cases = [c for c in cases if c[0] in args.only_case]
    if args.plan:
        print(json.dumps({"mode": args.mode, "cases": cases, "count": len(cases), "executed": False}, indent=2))
        return 0
    if sys.platform != "linux":
        raise RuntimeError("Qualification execution requires Linux; Windows interop is not accepted.")
    output, root = args.output.resolve(), args.native_root.resolve()
    if (output.exists() or output.is_relative_to(root)
            or not any(output.is_relative_to(Path(d)) for d in ("/mnt/e", "/mnt/d"))):
        raise RuntimeError("Use a fresh output on D/E outside the frozen native root.")
    output.mkdir(parents=True)
    report = {"utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(), "mode": args.mode,
              "passed": False, "expected_cases": len(cases), "cases": [], "quiet_performance_claimed": False,
              "scope": "Fresh Linux native/synthetic correctness qualification only; no published-model semantics, OS hard-cap, MTP/TP or throughput claim.",
              "selected_case_filter": args.only_case, "runner_sha256": sha(Path(__file__))}
    before = None
    try:
        for name in ("tmp", "cache", "cache/cuda"):
            (output / name).mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TS_", "TENSORSHARP_", "GGML_"))}
        env.update(TMPDIR=str(output / "tmp"), TMP=str(output / "tmp"), TEMP=str(output / "tmp"),
                   XDG_CACHE_HOME=str(output / "cache"), CUDA_CACHE_PATH=str(output / "cache/cuda"),
                   LD_LIBRARY_PATH="/usr/lib/wsl/lib:" + str(args.toolkit / "lib64"),
                   PATH=str(args.toolkit / "bin") + ":" + env["PATH"],
                   CUDA_VISIBLE_DEVICES="-1" if args.mode == "cpu" else "0", OMP_NUM_THREADS="8",
                   TS_CPU_MOE_THREADS="8", TS_GGML_CPU_THREADS="8")
        report["environment"] = {k: v for k, v in env.items() if k.startswith(("TS_", "CUDA_", "OMP_"))
                                 or k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "LD_LIBRARY_PATH")}
        report["build_receipt_sha256"] = verify_native_build(root)
        before = report["native_before"] = native_pins(root)
        report["production_public_abi"] = abi_evidence(root / "production/libGgmlOps.so", env)
        if args.mode != "glm":
            for name, target, flags in cases:
                executable = require_elf(root / "components" / target)
                case_env = env.copy()
                case_env["LD_LIBRARY_PATH"] = str(root / "components") + ":" + env["LD_LIBRARY_PATH"]
                if args.mode == "cuda" and name.startswith("pipeline-"):
                    case_env["TS_HOST_MOE_FILE_PIPELINE"] = "1" if "-off-" in name else "0"
                case = run_case([str(executable)] + flags, case_env, output / (name + ".log"),
                                180 if args.mode == "cpu" else 300)
                annotate_requested_native_coverage(name, case,
                                                   (output / (name + ".log")).read_text(errors="replace"))
                case.update(name=name, executable_sha256=sha(executable),
                            explicit_pipeline_legacy_environment=case_env.get("TS_HOST_MOE_FILE_PIPELINE"))
                report["cases"].append(case)
                (output / "progress.json").write_text(json.dumps(report, indent=2))
        else:
            dotnet = require_elf(args.dotnet)
            report["dotnet"] = {"path": str(dotnet), "version": subprocess.check_output([str(dotnet), "--version"], env=env, text=True).strip()}
            report["managed_copy"] = prepare_managed(args.managed_source, args.managed_manifest,
                                                      output, root, args.probe_sha256)
            report["quantized_source_metadata"] = {"path": str(args.quantized_source),
                "size": args.quantized_source.stat().st_size, "mtime_ns": args.quantized_source.stat().st_mtime_ns}
            for name, mode, backend, flags in cases:
                deployment = output / (mode + "-bin")
                native = deployment / "libGgmlOps.so"
                case_env = env.copy()
                case_env.update(CUDA_VISIBLE_DEVICES="0" if backend == "CUDA" else "-1",
                                OMP_NUM_THREADS="2", TS_CPU_MOE_THREADS="2", TS_GGML_CPU_THREADS="2",
                                LD_LIBRARY_PATH=str(deployment) + ":" + env["LD_LIBRARY_PATH"])
                folder = output / name
                command = [str(dotnet), str(deployment / "GlmGraphBudgetProbe.dll"), backend,
                           str(folder), "--shared-graphs"] + flags
                case = run_case(command, case_env, output / (name + ".log"), 180)
                case["name"] = name
                try:
                    actual, compare = read_json(folder / "report.json"), read_json(folder / "logit-comparison.json")
                    validate_glm_report(actual, compare, native, sha(native), backend, flags)
                    case.update(checks=actual["checks"], logit_elements=actual["logit_elements"],
                                full_logits_bit_exact=True, report=str(folder / "report.json"),
                                comparison=str(folder / "logit-comparison.json"))
                except Exception as error:
                    case.update(passed=False, acceptance_error=repr(error))
                report["cases"].append(case)
                (output / "progress.json").write_text(json.dumps(report, indent=2))
            for mode, expected in report["managed_copy"]["deployments"].items():
                actual = {p.name: sha(p) for p in (output / (mode + "-bin")).iterdir() if p.is_file()}
                if actual != expected:
                    raise RuntimeError("Managed/native deployment changed during execution: " + mode)
        report["passed"] = len(report["cases"]) == len(cases) and all(c["passed"] for c in report["cases"])
    except BaseException as error:
        report["error"] = repr(error)
    finally:
        if before is not None:
            try:
                report["native_after"] = native_pins(root)
                report["passed"] = report["passed"] and report["native_after"] == before
            except BaseException as error:
                report.update(passed=False, final_pin_error=repr(error))
        report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["actual_passed_cases"] = sum(c.get("passed") is True for c in report["cases"])
        report["platform_unexercised_or_skipped_cases"] = sum(c.get("skipped_or_unavailable") is True for c in report["cases"])
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "failed": [c["name"] for c in report["cases"] if not c["passed"]],
                      "actual_passed_cases": report["actual_passed_cases"],
                      "platform_unexercised_or_skipped_cases": report["platform_unexercised_or_skipped_cases"],
                      "error": report.get("error"), "summary": str(output / "summary.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
