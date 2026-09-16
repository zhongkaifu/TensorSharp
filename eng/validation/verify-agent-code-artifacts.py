#!/usr/bin/env python3
"""Independently test generated Python artifacts inside a required Linux sandbox.

The workflow report supplies server-advertised artifact URLs. This verifier reads
their retained copies and checks the last advertised source version for each
successful code workflow. Model-written JSON assertions alone are insufficient.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.parse


CONTRACTS = {
    "code_generation_run": ("sum_numbers.py", "sum_numbers", [0, 1, 2, 37, 100]),
    "code_edit_run": ("parity.py", "parity", [-3, 0, 4, 7, -2, 11]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-report", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bwrap", default="/usr/local/bin/bwrap")
    args = parser.parse_args()
    workflow = json.loads(args.workflow_report.read_text())
    store = args.artifact_store.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"workflow_report_sha256": hashlib.sha256(args.workflow_report.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "sandbox": args.bwrap, "cases": [], "run_complete": False}
    for case in workflow["cases"]:
        if case["scenario"] not in CONTRACTS:
            continue
        filename, function, inputs = CONTRACTS[case["scenario"]]
        entry = {"scenario": case["scenario"], "trial": case["trial"], "status": "fail", "inputs": inputs}
        report["cases"].append(entry)
        try:
            if case["status"] != "ok":
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
                        "namespace=runpy.run_path('/work/module.py',run_name='release_verification_module')\n"
                        f"function=namespace[{function!r}]\n"
                        f"inputs={inputs!r}\nexpected={expected!r}\n"
                        "actual=[function(n) for n in inputs]\n"
                        "assert actual==expected,(actual,expected)\n"
                        "assert all(type(a) is type(e) for a,e in zip(actual,expected)),actual\n"
                        f"print({marker!r}+json.dumps(actual,separators=(',',':')))\n")
            with tempfile.TemporaryDirectory(prefix="verify-agent-", dir=args.output.parent) as temporary:
                stage = Path(temporary)
                shutil.copyfile(source, stage / "module.py")
                (stage / "verify.py").write_text(verifier)
                command = [args.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
                           "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
                           "--ro-bind", "/usr", "/usr"]
                for library in ("/lib", "/lib64"):
                    if Path(library).exists():
                        command += ["--ro-bind", library, library]
                command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                            "--bind", str(stage), "/work", "--chdir", "/work",
                            "/usr/bin/python3", "-I", "/work/verify.py"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                entry.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                             expected=expected, command=command)
                proof = marker + json.dumps(expected, separators=(",", ":"))
                if result.returncode != 0 or proof not in result.stdout.splitlines():
                    raise RuntimeError("Independent sandboxed function checks failed")
            entry["status"] = "ok"
        except Exception as error:
            entry["detail"] = str(error)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{sum(case['status'] == 'ok' for case in report['cases'])}/{len(report['cases'])} independent code checks passed")
    return int(not report["cases"] or any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
