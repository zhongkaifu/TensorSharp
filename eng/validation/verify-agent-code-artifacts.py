#!/usr/bin/env python3
"""Independently test generated Python artifacts, using a Linux sandbox by default.

The workflow report supplies server-advertised artifact URLs. This verifier reads
their retained copies and checks the last advertised source version for each
successful code workflow. Model-written JSON assertions alone are insufficient.
--sandbox-off permits functional checks in an isolated validation workspace on
hosts without user namespaces; the report explicitly records unconfined execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.parse


CONTRACTS = {
    "code_generation_run": ("sum_numbers.py", "sum_numbers", [0, 1, 2, 37, 100]),
    "code_edit_run": ("parity.py", "parity", [-3, 0, 4, 7, -2, 11]),
}


def execution_command(stage, bwrap, sandbox_off, python_executable, platform_name=None):
    """Never silently substitute unconstrained execution for an unavailable sandbox."""
    platform_name = sys.platform if platform_name is None else platform_name
    if sandbox_off:
        return [python_executable, "-I", "verify.py"]
    if platform_name != "linux":
        raise RuntimeError("This verifier's confined mode requires Linux bwrap. Use --sandbox-off only for explicitly unconfined functional validation.")
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
               "--ro-bind", "/usr", "/usr"]
    for library in ("/lib", "/lib64"):
        if Path(library).exists():
            command += ["--ro-bind", library, library]
    return command + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                      "--bind", str(stage), "/work", "--chdir", "/work",
                      "/usr/bin/python3", "-I", "/work/verify.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-report", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bwrap", default="/usr/local/bin/bwrap")
    parser.add_argument("--python", default=sys.executable, help="Python executable for --sandbox-off, including Windows")
    parser.add_argument("--sandbox-off", action="store_true",
                        help="Explicitly run functional checks without OS sandbox isolation")
    parser.add_argument("--diagnose-incomplete", action="store_true",
                        help="Check retained code even after a failed workflow; original failure still prevents an overall pass")
    args = parser.parse_args()
    workflow = json.loads(args.workflow_report.read_text(encoding="utf-8-sig"))
    store = args.artifact_store.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"workflow_report_sha256": hashlib.sha256(args.workflow_report.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sandbox": None if args.sandbox_off else args.bwrap,
              "execution_mode": "unconfined" if args.sandbox_off else "sandbox",
              "platform": sys.platform, "python": args.python,
              "diagnose_incomplete": args.diagnose_incomplete,
              "cases": [], "run_complete": False}
    for case in workflow["cases"]:
        if case["scenario"] not in CONTRACTS:
            continue
        filename, function, inputs = CONTRACTS[case["scenario"]]
        entry = {"scenario": case["scenario"], "trial": case["trial"], "status": "fail", "inputs": inputs,
                 "original_workflow_status": case["status"], "function_checks_passed": False}
        report["cases"].append(entry)
        try:
            if case["status"] != "ok" and not args.diagnose_incomplete:
                raise ValueError("Original workflow did not pass")
            sources = [item for event in case["events"] for item in (event.get("files") or [])
                       if Path(urllib.parse.urlsplit(item.get("url", "")).path).name == filename]
            if not sources:
                raise ValueError("No source artifact was advertised")
            url = sources[-1]["url"]
            decoded = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
            prefix = "/api/code/artifacts/"
            if not decoded.startswith(prefix):
                raise ValueError("Unexpected artifact route")
            source = (store / decoded[len(prefix):]).resolve(strict=True)
            if not source.is_relative_to(store) or not source.is_file():
                raise ValueError("Source escapes artifact store or is not a regular file")
            entry.update(artifact_url=url, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            expected = [n * (n + 1) // 2 if function == "sum_numbers" else ("even" if n % 2 == 0 else "odd")
                        for n in inputs]
            marker = "TENSORSHARP_INDEPENDENT_CODE_CHECK="
            verifier = ("import json,runpy\n"
                        "namespace=runpy.run_path('module.py',run_name='release_verification_module')\n"
                        f"function=namespace[{function!r}]\n"
                        f"inputs={inputs!r}\nexpected={expected!r}\n"
                        "actual=[function(n) for n in inputs]\n"
                        "assert actual==expected,(actual,expected)\n"
                        "assert all(type(a) is type(e) for a,e in zip(actual,expected)),actual\n"
                        f"print({marker!r}+json.dumps(actual,separators=(',',':')))\n")
            with tempfile.TemporaryDirectory(prefix="verify-agent-", dir=args.output.parent) as temporary:
                stage = Path(temporary)
                shutil.copyfile(source, stage / "module.py")
                (stage / "verify.py").write_text(verifier, encoding="utf-8")
                command = execution_command(stage, args.bwrap, args.sandbox_off, args.python)
                environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
                result = subprocess.run(command, cwd=stage, capture_output=True, text=True, timeout=10,
                                        encoding="utf-8", errors="replace", env=environment,
                                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
                entry.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                             expected=expected, command=command)
                proof = marker + json.dumps(expected, separators=(",", ":"))
                if result.returncode != 0 or proof not in result.stdout.splitlines():
                    raise RuntimeError("Independent function checks failed")
            entry["function_checks_passed"] = True
            if case["status"] == "ok":
                entry["status"] = "ok"
            else:
                entry["detail"] = "Retained function checks passed, but the original workflow did not pass"
        except Exception as error:
            entry["detail"] = str(error)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(case['status'] == 'ok' for case in report['cases'])}/{len(report['cases'])} independent code checks passed")
    return int(not report["cases"] or any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
