#!/usr/bin/env python3
"""Run the unchanged Qwen3.8 quality protocols against one already-running server.

PLAN is default: no HTTP requests, server launch, GPU initialization or package
installation. --execute invokes the byte-pinned existing benchmark or complete
program protocol. The operator owns the server, exclusive GPU lease and runtime
provenance. This wrapper never changes the frozen scorers or starts/stops a VM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eng/qwen38-engine-bench.py"
PROGRAM = ROOT / "eng/validation/qwen38-code-program-quality.py"
PINS = {"bench": "4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c",
        "program": "4093b80c7401454344b67b2a4e421314b4a263024d79cd30eefe09f67bfae3cb"}
SHORT = ("short_chat", "long_prompt_4k", "long_prompt_16k", "long_prompt_32k", "json_object",
         "json_schema", "tool_call_single", "tool_call_parallel", "agentic_loop", "multi_turn", "image",
         "long_generation", "parallel_2", "parallel_4", "quality_tasks", "quality_gsm8k")
SUITES = {"short": SHORT, "text": tuple(x for x in SHORT if x != "image"),
          "long": ("long_prompt_128k", "long_prompt_250k")}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh(path):
    need(path.is_absolute() and path == path.resolve(), "Canonical absolute output directory required")
    need(not path.exists() and not path.is_symlink(), "Fresh output directory required")
    parts = path.parts
    need("artifacts" in parts or any(parts[i:i + 2] == ("docs", "validation") for i in range(len(parts) - 1)),
         "Generated evidence must be under ignored artifacts/ or docs/validation/")
    need(all(not p.is_symlink() and not p.is_file() for p in path.parents), "Output parent alias/non-directory refused")


def build_command(args):
    if args.suite == "program":
        return [args.python, str(PROGRAM), "--execute", "--out", str(args.out / "program"),
                "--url", args.url, "--model", args.model, "--label", args.label,
                "--nonce-seed", "20261102", "--timeout", "1800", "--manifest", str(args.manifest)]
    return [args.python, str(BENCH), "--engine", args.engine, "--mode", "plain", "--url", args.url,
            "--model", args.model, "--label", args.label, "--out", str(args.out / "raw.jsonl"),
            "--scenarios", ",".join(SUITES[args.suite]), "--repeats", "3", "--quality-tasks", "8",
            "--gsm8k-limit", "8", "--answer-tokens", "512", "--decode-tokens", "1024",
            "--temperature", "0", "--top-p", "1", "--sampling-seed", "1234", "--nonce-seed", "20261101",
            "--warmup", "3", "--long-order-quality", "--keep-text", "65536", "--timeout", "1800",
            "--tokenizer", str(args.tokenizer), "--manifest", str(args.manifest)]


def observed_output(args):
    if args.suite == "program":
        file = args.out / "program/campaign.json"
        need(file.is_file(), "Program protocol did not produce its campaign")
        data = json.loads(file.read_text())
        return {"source": str(file), "sha256": digest(file), "complete": data.get("complete") is True,
                "qualityPassed": data.get("quality_passed") is True, "verdictCounts": data.get("verdict_counts"),
                "programs": len(data.get("records", [])), "plannedFunctionalCases": data.get("planned_functional_cases")}
    file = args.out / "raw.jsonl"
    need(file.is_file(), "Frozen benchmark did not produce raw JSONL")
    rows = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
    headers = [x for x in rows if x.get("kind") == "run"]
    cases = [x for x in rows if x.get("kind") == "case"]
    need(len(headers) == 1, "Expected exactly one frozen benchmark run header")
    expected = {(scenario, repeat) for scenario in SUITES[args.suite] for repeat in range(3)}
    identities = [(x["scenario"], x["repeat"]) for x in cases]
    complete = len(identities) == len(set(identities)) and set(identities) == expected
    return {"source": str(file), "sha256": digest(file), "complete": complete,
            "cases": len(cases), "expectedCases": len(expected),
            "requests": sum(len(x.get("requests", [])) for x in cases),
            "qualityPassed": complete and all(x.get("verdict") == "PASS" for x in cases),
            "verdictCounts": {key: sum(x.get("verdict") == key for x in cases) for key in ("PASS", "FAIL", "ERROR")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--engine", choices=("tensorsharp", "ignis"), required=True)
    parser.add_argument("--suite", choices=("short", "text", "long", "program"), default="text")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--known-artifact-sha", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        fresh(args.out)
        endpoint = urlparse(args.url)
        need(endpoint.scheme == "http" and endpoint.hostname in ("127.0.0.1", "localhost", "::1")
             and endpoint.port is not None and endpoint.path in ("", "/") and not endpoint.query and not endpoint.fragment
             and endpoint.username is None and endpoint.password is None, "Explicit loopback HTTP server URL required")
        need(len(args.known_artifact_sha) == 64 and all(c in "0123456789abcdefABCDEF" for c in args.known_artifact_sha), "Invalid supplied artifact SHA")
        for key, path in (("artifact", args.artifact), ("tokenizer", args.tokenizer), ("manifest", args.manifest)):
            need(path.is_absolute(), "Absolute input path required: " + key)
            if args.execute:
                need(path.is_file(), "Missing execution input: " + str(path))
        for key, path in (("bench", BENCH), ("program", PROGRAM)):
            need(digest(path) == PINS[key], "Frozen protocol bytes changed: " + key)
        command = build_command(args)
        state = {"schema": "qwen38-ninfer-body-evaluation/1", "complete": False,
                 "status": "PLAN_ONLY_NO_HTTP", "engine": args.engine, "suite": args.suite,
                 "artifactPath": str(args.artifact), "knownArtifactSha256": args.known_artifact_sha.lower(),
                 "wholeArtifactSha256VerifiedByThisTool": False, "serverIdentityVerifiedByThisTool": False,
                 "serverManifest": str(args.manifest), "tokenizer": str(args.tokenizer), "command": command,
                 "protocolPins": PINS, "qualityPassed": False, "performanceQualified": False,
                 "limitations": ["Operator must hold an exclusive GPU lease and bind the actual server/source/runtime/device/artifact identities.",
                                 "This client uses the existing frozen scorers; no error, unsupported image case or unexecuted program input is upgraded.",
                                 "Text suite selects fifteen original scenarios; it does not claim vision/draft/CUA coverage.",
                                 "Cross-engine adapter fields and templates may differ. Preserve full protocol records and do not claim identical wire bytes.",
                                 "Small public repeated tasks do not establish universal quality equivalence or engine leadership."]}
        args.out.mkdir(parents=True)
        result = args.out / "evaluation.json"
        def save():
            result.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if args.execute:
            stat = args.artifact.stat()
            state.update(status="RUNNING_NOT_PASS", artifactBefore={"bytes": stat.st_size, "mtimeNs": stat.st_mtime_ns},
                         serverManifestSha256=digest(args.manifest), tokenizerSha256=digest(args.tokenizer))
            save()
            with (args.out / "stdout.log").open("w") as stdout, (args.out / "stderr.log").open("w") as stderr:
                child = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
            state["subprocessExitCode"] = child.returncode
            state["observed"] = observed_output(args)
            after = args.artifact.stat()
            state["artifactAfter"] = {"bytes": after.st_size, "mtimeNs": after.st_mtime_ns}
            need(state["artifactBefore"] == state["artifactAfter"], "Artifact stat changed during evaluation")
            need(digest(args.manifest) == state["serverManifestSha256"], "Server manifest changed during evaluation")
            need(digest(args.tokenizer) == state["tokenizerSha256"], "Tokenizer bytes changed during evaluation")
            for key, path in (("bench", BENCH), ("program", PROGRAM)):
                need(digest(path) == PINS[key], "Frozen protocol changed during evaluation")
            state.update(status="EXECUTED_COMPLETE" if state["observed"]["complete"] else "INCOMPLETE",
                         complete=state["observed"]["complete"], qualityPassed=state["observed"]["qualityPassed"])
            save()
            print(json.dumps({"output": str(result), "complete": state["complete"], "qualityPassed": state["qualityPassed"], "exitCode": child.returncode}))
            return child.returncode if child.returncode != 0 else (0 if state["complete"] else 1)
        save()
        print(json.dumps(state, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print("Evaluation refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
