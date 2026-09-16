#!/usr/bin/env python3
"""Exercise real server skill discovery, built-in file tools and shell execution.

Unlike client-simulated tool fixtures, this retains the Web UI SSE tool events
and independently downloads result artifacts. A host without a working OS
sandbox cannot pass execution cases; refusal evidence is reported as blocked.
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
    "code_edit_run": {"prompt": "Use write_file to create parity.py containing def parity(n): return 'even'. Use read_file to inspect it, then edit_file to fix it to return 'even' for even integers and 'odd' for odd integers. Use shell to run checks for -3,0,4,7. Write result.json containing exactly {\"outputs\":[\"odd\",\"even\",\"even\",\"odd\"],\"tests_passed\":true}. Return its download link. Do not claim execution if a tool refuses.",
                      "tools": ["write_file", "read_file", "edit_file", "shell"], "artifact": {"outputs": ["odd", "even", "even", "odd"], "tests_passed": True}, "execution": True},
}


def case_spec(name, trial, distinct_inputs=False):
    spec = copy.deepcopy(CASES[name])
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


def run(client, name, trial, timeout, sandbox_available, distinct_inputs=False, thinking=False):
    spec = case_spec(name, trial, distinct_inputs)
    session = create_session(client, 30)
    result = {"scenario": name, "trial": trial, "session": session, "status": "fail", "events": [], "artifacts": []}
    result["expected"] = spec.get("artifact", spec.get("expected"))
    artifacts, answer, connection = {}, "", None
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
            if isinstance(event.get("replace"), str):
                answer = event["replace"]
            elif isinstance(event.get("token"), str):
                answer += event["token"]
            if event.get("done"):
                terminal = event
                break
        result["answer"] = answer
        steps = [event for event in result["events"] if event.get("skill_step")]
        succeeded = {event["skill_step"] for event in steps if event.get("ok")}
        result["successful_tools"] = sorted(succeeded)
        if spec.get("execution") and not sandbox_available:
            result["status"] = "blocked"
            result["detail"] = "Host denies user namespace creation; a required OS sandbox is unavailable. Successful unconfined execution would not satisfy this case."
            result["refusal_events"] = [event for event in steps if not event.get("ok")]
            return result
        if terminal is None or terminal.get("error") or terminal.get("aborted") or terminal.get("truncated"):
            raise RuntimeError("Missing or failed terminal event: " + str(terminal))
        missing = set(spec["tools"]) - succeeded
        if missing:
            raise RuntimeError("Required successful tool events absent: " + ", ".join(sorted(missing)))
        if "expected" in spec and answer.strip() != spec["expected"]:
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
    parser.add_argument("--sandbox-unavailable", action="store_true")
    parser.add_argument("--thinking", action="store_true",
                        help="Exercise the same workflow and quality checks with thinking enabled")
    parser.add_argument("--distinct-inputs", action="store_true",
                        help="Give concurrent execution requests different expected outputs to detect crossed sessions")
    args = parser.parse_args()
    client = HttpClient(args.url, 30)
    status, skills = client.json_request("GET", "/api/skills", None, 30)
    report = {"format_version": 1, "started_at_unix": time.time(), "skills_http_status": status,
              "skills": skills, "sandbox_available": not args.sandbox_unavailable,
              "distinct_inputs": args.distinct_inputs,
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
                                       not args.sandbox_unavailable, args.distinct_inputs, args.thinking) for index in range(degree)]
                cases = [future.result() for future in futures]
            for case in cases:
                case["concurrency"] = degree
            report["cases"].extend(cases)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(name, degree, [case["status"] for case in cases], flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
