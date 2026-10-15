#!/usr/bin/env python3
"""Exercise real server skill discovery, built-in file tools and shell execution.

Unlike client-simulated tool fixtures, this retains the Web UI SSE tool events
and independently downloads result artifacts. By default, a host without a
working OS sandbox cannot pass execution cases. --sandbox-off explicitly tests
functional execution on a host started with its sandbox disabled and records
that limitation; it makes no sandbox validation claim.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from server_parallel_skills import HttpClient, collect_artifacts, create_session, delete_session, iter_sse

CASES = {
    "skill_selection": {"prompt": "Use the available release verification skill to look up packet cedar-4821. Return only its verification string.",
                        "expected": "ts-release-audit-f70d932b", "tools": ["skills_read"]},
    "skill_run": {"prompt": "Use release-validation to read the release verification record for packet cedar-4821. Return only the verification string.",
                  "skills": ["release-validation"], "expected": "ts-release-audit-f70d932b", "tools": ["skills_read"]},
    "skill_script_run": {"prompt": "Use release-validation to compute the input checksum for cedar-4821 using its bundled script. Return the script stdout only. If the sandbox refuses execution, say so.",
                         "skills": ["release-validation"], "expected": hashlib.sha256(b"cedar-4821").hexdigest(), "tools": ["skills_run"], "execution": True},
    "shell_run": {"prompt": "Use the shell tool to execute printf '%s\\n' release-shell-4821 in the request workspace. Return exactly the stdout. Do not claim execution if the tool refuses.",
                  "expected": "release-shell-4821", "tools": ["shell"], "execution": True},
    "code_generation_run": {"prompt": "Create sum_numbers.py in the request workspace using write_file. Its function sum_numbers(n) must return the sum of the integers 1 through n. Use shell to run the program for n=37 and verify n=0 gives 0 and n=1 gives 1. Save result.json with exactly {\"n\":37,\"sum\":703,\"tests_passed\":true}. Return the artifact download link. Do not claim a test ran if shell refuses.",
                            "tools": ["write_file", "shell"], "artifact": {"n": 37, "sum": 703, "tests_passed": True}, "execution": True},
    "code_edit_run": {"prompt": "Use write_file to create parity.py containing def parity(n): return 'even'. Use read_file to inspect it, then apply_patch to fix this single file to return 'even' for even integers and 'odd' for odd integers. Use shell to run checks for -3,0,4,7. Write result.json containing exactly {\"outputs\":[\"odd\",\"even\",\"even\",\"odd\"],\"tests_passed\":true}. Return its download link. Do not claim execution if a tool refuses.",
                      "tools": ["write_file", "read_file", "apply_patch", "shell"], "artifact": {"outputs": ["odd", "even", "even", "odd"], "tests_passed": True}, "execution": True},
}


def case_spec(name, trial, distinct_inputs=False, target_shell="posix"):
    spec = copy.deepcopy(CASES[name])
    if target_shell not in ("posix", "powershell", "cmd"):
        raise ValueError("Unsupported target shell: " + target_shell)
    if name == "shell_run" and target_shell != "posix":
        command = "Write-Output 'release-shell-4821'" if target_shell == "powershell" else "echo release-shell-4821"
        spec["prompt"] = (f"Use the shell tool to execute {command} in the request workspace. "
                          "Return exactly the stdout. Do not claim execution if the tool refuses.")
    if spec.get("execution") and target_shell != "posix":
        spec["prompt"] += (f" The execution host is Windows and its shell is {target_shell}. "
                           "Use Windows-compatible commands; do not use POSIX shell heredocs.")
    if not distinct_inputs:
        return spec
    marker = "probe-" + hashlib.sha256(f"{name}:{trial}".encode()).hexdigest()[:12]
    if name == "skill_script_run":
        spec["prompt"] = spec["prompt"].replace("cedar-4821", marker)
        spec["expected"] = hashlib.sha256(marker.encode()).hexdigest()
    elif name == "shell_run":
        spec["prompt"] = spec["prompt"].replace(spec["expected"], marker)
        spec["expected"] = marker
    elif "artifact" in spec:
        original = json.dumps(spec["artifact"], separators=(",", ":"))
        spec["artifact"]["probe"] = marker
        replacement = json.dumps(spec["artifact"], separators=(",", ":"))
        if original not in spec["prompt"]:
            raise ValueError("Artifact fixture is absent from its prompt")
        spec["prompt"] = spec["prompt"].replace(original, replacement)
    spec["prompt"] = f"[release validation {marker}]\n" + spec["prompt"]
    return spec


def validate_result_artifact(client, artifacts, expected, result):
    # Each write publishes an immutable snapshot with a new URL. The task asks
    # for the completed file, so an earlier draft must not replace the final
    # version in the oracle. Keep the version history for review.
    versions = [artifact for artifact in artifacts.values() if Path(artifact.name).name == "result.json"]
    if not versions:
        raise RuntimeError("No result.json artifact was advertised")
    result["result_artifact_versions"] = [{"name": item.name, "url": item.url} for item in versions]
    artifact = versions[-1]
    status, mime, data = client.download(artifact.url, 60, 1024 * 1024)
    parsed = json.loads(data)
    result["artifacts"].append({"name": artifact.name, "url": artifact.url, "http_status": status,
        "sha256": hashlib.sha256(data).hexdigest(), "content": parsed})
    if status != 200 or parsed != expected:
        raise RuntimeError("Downloaded final result artifact failed validation")


def run(client, name, trial, timeout, sandbox_available, distinct_inputs=False, thinking=False, sandbox_off=False,
        target_shell="posix"):
    spec = case_spec(name, trial, distinct_inputs, target_shell)
    session = create_session(client, 30)
    result = {"scenario": name, "trial": trial, "session": session, "status": "fail", "events": [], "artifacts": []}
    result["expected"] = spec.get("artifact", spec.get("expected"))
    artifacts, answer, final_answer, connection = {}, "", "", None
    started = time.monotonic()
    try:
        body = {"messages": [{"role": "user", "content": spec["prompt"]}], "sessionId": session,
                "newChat": False, "maxTokens": 4096, "temperature": 0, "think": thinking,
                "tools": [], "skills_discovery": True}
        if "skills" in spec:
            body["skills"] = spec["skills"]
        result["request"] = body
        connection, response = client.open_sse("/api/chat", body, timeout)
        result["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(response.read().decode("utf-8", "replace"))
        terminal = None
        for event in iter_sse(response, started + timeout):
            result["events"].append(event)
            collect_artifacts(artifacts, event)
            if event.get("skill_step"):
                # SkillChatLoop forwards narration from tool-calling rounds too.
                # WebUiChatService emits each completed skill_step before the
                # next round's content, so only text after the final step is the
                # final answer. Keep the combined text and events as evidence.
                final_answer = ""
            if isinstance(event.get("replace"), str):
                answer = event["replace"]
                final_answer = event["replace"]
            elif isinstance(event.get("token"), str):
                answer += event["token"]
                final_answer += event["token"]
            if event.get("done"):
                terminal = event
                break
        result["answer"] = answer
        result["final_answer"] = final_answer
        steps = [event for event in result["events"] if event.get("skill_step")]
        succeeded = {event["skill_step"] for event in steps if event.get("ok")}
        result["successful_tools"] = sorted(succeeded)
        if spec.get("execution") and not sandbox_available and not sandbox_off:
            result["status"] = "blocked"
            result["detail"] = "Host denies user namespace creation; a required OS sandbox is unavailable. Successful unconfined execution would not satisfy this case."
            result["refusal_events"] = [event for event in steps if not event.get("ok")]
            return result
        if terminal is None or terminal.get("error") or terminal.get("aborted") or terminal.get("truncated"):
            raise RuntimeError("Missing or failed terminal event: " + str(terminal))
        missing = set(spec["tools"]) - succeeded
        if missing:
            raise RuntimeError("Required successful tool events absent: " + ", ".join(sorted(missing)))
        if "expected" in spec and final_answer.strip() != spec["expected"]:
            raise RuntimeError("Final answer differs from the independent expected value")
        if "artifact" in spec:
            validate_result_artifact(client, artifacts, spec["artifact"], result)
        result["status"] = "ok"
    except Exception as error:
        result["detail"] = str(error)
    finally:
        if connection:
            connection.close()
        result["wall_seconds"] = time.monotonic() - started
        warning = delete_session(client, session, 30)
        if warning:
            result["cleanup_warning"] = warning
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", default=",".join(CASES))
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--target-shell", choices=("posix", "powershell", "cmd"), default="posix",
                        help="Shell used by the server, independent of this client's operating system")
    sandbox = parser.add_mutually_exclusive_group()
    sandbox.add_argument("--sandbox-unavailable", action="store_true")
    sandbox.add_argument("--sandbox-off", action="store_true",
                         help="Validate functional execution on a host explicitly configured with sandbox=off; does not validate isolation")
    parser.add_argument("--thinking", action="store_true",
                        help="Exercise the same workflow and quality checks with thinking enabled")
    parser.add_argument("--distinct-inputs", action="store_true",
                        help="Give concurrent execution requests different expected outputs to detect crossed sessions")
    args = parser.parse_args()
    client = HttpClient(args.url, 30)
    status, skills = client.json_request("GET", "/api/skills", None, 30)
    report = {"format_version": 1, "started_at_unix": time.time(), "skills_http_status": status,
              "skills": skills, "sandbox_available": False if args.sandbox_unavailable else (None if args.sandbox_off else True),
              "execution_mode": "unconfined" if args.sandbox_off else "sandbox",
              "distinct_inputs": args.distinct_inputs,
              "target_shell": args.target_shell,
              "thinking": args.thinking,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "run_complete": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.scenarios.split(","):
        if name not in CASES:
            parser.error("Unknown scenario: " + name)
        for degree in [int(item) for item in args.concurrency.split(",")]:
            with ThreadPoolExecutor(max_workers=degree) as pool:
                futures = [pool.submit(run, client, name, f"c{degree}-i{index}", args.timeout,
                                       not args.sandbox_unavailable and not args.sandbox_off,
                                       args.distinct_inputs, args.thinking, args.sandbox_off, args.target_shell) for index in range(degree)]
                cases = [future.result() for future in futures]
            for case in cases:
                case["concurrency"] = degree
            report["cases"].extend(cases)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(name, degree, [case["status"] for case in cases], flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
