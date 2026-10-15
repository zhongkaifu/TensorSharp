#!/usr/bin/env python3
"""Recheck a completed GLM32 run without replacing its original receipt.

This is a parser revalidation, not a native fix or a new model execution.
All original commands, logs, complete-logit comparisons, deployments and native
bookends remain pinned. The selected-only N2 check follows the Probe's branch.
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys

source = Path(__file__).with_name("linux-native-qualification.py")
spec = importlib.util.spec_from_file_location("qualification", source)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Revalidate under the same Linux path/ELF environment.")
    output, previous, root = args.output.resolve(), args.previous.resolve(), args.native_root.resolve()
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(previous.parent):
        raise RuntimeError("Use a new separate D/E output; preserve the original receipts.")
    if not any(output.is_relative_to(Path(d)) for d in ("/mnt/d", "/mnt/e")):
        raise RuntimeError("Revalidation output must be on D/E.")
    output.mkdir(parents=True)
    report = dict(passed=False, utc_start=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  revalidation_only=True, native_fix_claimed=False, new_execution_claimed=False,
                  original_summary=str(previous), original_summary_sha256=tool.sha(previous),
                  parser_sha256=tool.sha(source), revalidation_tool_sha256=tool.sha(Path(__file__)), cases=[])
    try:
        old = tool.read_json(previous)
        if old.get("mode") != "glm" or old.get("expected_cases") != 32 or len(old.get("cases", [])) != 32:
            raise RuntimeError("A completed original 32-case GLM receipt is required.")
        report["original_passed"] = old.get("passed")
        report["original_parser_errors"] = {c["name"]: c["acceptance_error"] for c in old["cases"] if "acceptance_error" in c}
        if old.get("native_before") != old.get("native_after") or tool.native_pins(root) != old["native_after"]:
            raise RuntimeError("Original/current native binary bookends differ.")
        if tool.verify_native_build(root) != old["build_receipt_sha256"]:
            raise RuntimeError("The qualified native build receipt changed.")
        quant = old["quantized_source_metadata"]
        quant_path = Path(quant["path"])
        if quant_path.stat().st_size != quant["size"] or quant_path.stat().st_mtime_ns != quant["mtime_ns"]:
            raise RuntimeError("Original quantized source metadata changed.")
        managed = old["managed_copy"]
        if tool.sha(Path(managed["source_manifest"])) != managed["source_manifest_sha256"]:
            raise RuntimeError("Frozen managed manifest changed.")
        inputs = [previous, source, Path(__file__), Path(managed["source_manifest"])]
        for mode, expected in managed["deployments"].items():
            deployment = previous.parent / (mode + "-bin")
            if {p.name: tool.sha(p) for p in deployment.iterdir() if p.is_file()} != expected:
                raise RuntimeError("Original deployment closure changed: " + mode)
            inputs.extend(deployment / name for name in expected)
        expected_cases = tool.glm_cases(quant_path)
        if [c["name"] for c in old["cases"]] != [c[0] for c in expected_cases]:
            raise RuntimeError("Original case order/coverage differs from the complete matrix.")
        for case in old["cases"]:
            folder = previous.parent / case["name"]
            inputs.extend([Path(case["log"]), folder / "report.json", folder / "logit-comparison.json"])
        report["inputs_before"] = {str(p): tool.sha(p) for p in inputs}
        for case, (name, mode, backend, flags) in zip(old["cases"], expected_cases):
            folder = previous.parent / name
            native = previous.parent / (mode + "-bin/libGgmlOps.so")
            command = [old["dotnet"]["path"], str(previous.parent / (mode + "-bin/GlmGraphBudgetProbe.dll")),
                       backend, str(folder), "--shared-graphs"] + flags
            if case["command"] != command or case.get("exit_code") != 0 or case.get("skipped_or_unavailable"):
                raise RuntimeError("Original actual command/exit/coverage gate failed: " + name)
            log, actual_path, compare_path = Path(case["log"]), folder / "report.json", folder / "logit-comparison.json"
            actual, compare = tool.read_json(actual_path), tool.read_json(compare_path)
            tool.validate_glm_report(actual, compare, native, tool.sha(native), backend, flags)
            report["cases"].append(dict(name=name, passed=True, command=command,
                actual_execution_seconds=case["seconds"], checks=actual["checks"],
                logit_elements=actual["logit_elements"], full_logits_bit_exact=True,
                report=str(actual_path), comparison=str(compare_path), log=str(log)))
        report["inputs_after"] = {str(p): tool.sha(p) for p in inputs}
        report["native_after"] = tool.native_pins(root)
        report["passed"] = (report["inputs_before"] == report["inputs_after"]
                            and report["native_after"] == old["native_after"] and len(report["cases"]) == 32)
    except Exception as error:
        report["error"] = repr(error)
    report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "error": report.get("error"), "summary": str(output / "summary.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
