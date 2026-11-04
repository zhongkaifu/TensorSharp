#!/usr/bin/env python3
"""Offline frozen API no-loss audit. No HTTP, CUDA, VM or generated-code execution.

Reconstructs every text request/check with the pinned original benchmark and
revalidates bounded program worker outputs with its unchanged original scorer.
An improvement never cancels a lost truth. Changed inputs are not comparable.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eng/qwen38-engine-bench.py"
PROGRAM = ROOT / "eng/validation/qwen38-code-program-quality.py"
BENCH_SHA = "4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c"
PROGRAM_SHA = "4093b80c7401454344b67b2a4e421314b4a263024d79cd30eefe09f67bfae3cb"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def unique(items):
    result = {}
    for key, value in items:
        need(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def regular(path):
    path = Path(os.path.abspath(path))
    need(path.is_file() and not path.is_symlink(), "Regular evidence file required: " + str(path))
    return path


class Evidence:
    def __init__(self):
        self.pins = {}

    def read(self, path):
        path = regular(path)
        data = path.read_bytes()
        pin = {"path": str(path), "bytes": len(data), "sha256": sha(data)}
        need(str(path) not in self.pins or self.pins[str(path)] == pin, "Input changed during audit")
        self.pins[str(path)] = pin
        return data

    def load(self, path):
        return parse(self.read(path))

    def verify(self):
        for pin in list(self.pins.values()):
            self.read(pin["path"])


def module(name, path, expected, evidence):
    need(sha(evidence.read(path)) == expected, "Frozen scorer bytes differ: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def denied(*args, **kwargs):
    raise RuntimeError("Network/process execution is forbidden in this offline audit")


def frozen_tools(evidence):
    bench = module("ninfer_api_frozen_bench", BENCH, BENCH_SHA, evidence)
    program = module("ninfer_api_frozen_program", PROGRAM, PROGRAM_SHA, evidence)
    bench.http_json = bench.http_text = bench.stream_chat = denied
    bench.scrape_metrics = lambda *args, **kwargs: None
    program.query = program.evaluate_program = program.run_case = denied
    haystack = bench.load_haystack(None)
    evidence.read(bench.GSM8K_FIXTURE)
    return bench, program, haystack


def audit_text(directory, bench, haystack, evidence):
    directory = Path(directory)
    text = directory / "text" if (directory / "text/raw.jsonl").is_file() else directory
    raw = evidence.read(text / "raw.jsonl")
    rows = [parse(line) for line in raw.splitlines() if line.strip()]
    need(rows and rows[0].get("kind") == "run", "Missing text run header")
    header = rows[0]
    need(header["harness_sha256"] == BENCH_SHA and header["harness"] == "qwen38-engine-bench/3", "Text protocol identity differs")
    need(header["haystack"]["sha256"] == haystack["sha256"] and header["haystack"]["files"] == haystack["files"], "Haystack identity differs")
    need(header["quality_fixture"]["sha256"] == sha(evidence.read(bench.GSM8K_FIXTURE)), "GSM fixture identity differs")
    cases = {}
    for record in rows[1:]:
        key = (record["scenario"], record["repeat"])
        need(type(key[1]) is int and key not in cases, "Duplicate/invalid text case identity")
        cases[key] = record
    expected = {(scenario, repeat) for scenario in header["scenarios"] for repeat in range(header["repeats"])}
    need(header["repeats"] == 3 and len(cases) == 45 and set(cases) == expected, "Complete original45-case matrix required")
    bodies = {}

    class Replay(bench.Client):
        def __init__(self, record):
            self.args = SimpleNamespace(engine=header["engine"], model=header["model"], url=header["url"],
                extra_body=json.dumps(header["extra_body"]), metrics_url="", tokenizer=None,
                temperature=header["sampling"]["temperature"], top_p=header["sampling"]["top_p"],
                sampling_seed=header["sampling"]["seed"], answer_tokens=header["answer_tokens"],
                decode_tokens=header["decode_tokens"], nonce_seed=header["nonce_seed"],
                quality_tasks=header["quality_tasks"], gsm8k_limit=header["gsm8k_limit"],
                chars_per_token=header["chars_per_token"], long_order_quality=header["long_order_quality"],
                long_prompt_tolerance=header["long_prompt_tolerance"], keep_text=65536,
                timeout=1800, agent_max_turns=8, server_log=None)
            self.engine = header["engine"]; self.url = header["url"]; self.model = header["model"]
            self.extra = header["extra_body"]; self.metrics_url = ""; self.tokenizer = object()
            self.record = record; self.responses = record["requests"]; self.bodies = {}; self.lock = threading.Lock()
            self.by_sha = {r["request_sha256"]: i for i, r in enumerate(self.responses)}
            need(len(self.by_sha) == len(self.responses), "Duplicate per-case request identity")

        def issue(self, body):
            identity = sha(canonical(bench.request_identity(body)))
            need(identity in self.by_sha, "Reconstructed request SHA absent: " + self.record["scenario"])
            index = self.by_sha[identity]
            with self.lock:
                need(index not in self.bodies, "Duplicate replay request")
                self.bodies[index] = copy.deepcopy(body)
            return copy.deepcopy(self.responses[index])

        def chat(self, messages, max_tokens, extra=None):
            return self.issue(self.body(messages, max_tokens, extra))

        def finish(self, observation):
            return observation["record"]

    for key, record in cases.items():
        need(record["kind"] == "case" and record["run_id"] == header["run_id"], "Case/run identity differs")
        client = Replay(record)
        bench.stream_chat = lambda url, body, timeout: {"record": client.issue(body)}

        def recorded_fit(render, prose, target, count_tokens):
            count = record["metrics"]["prompt_chars"] - len(render(""))
            need(0 <= count <= len(prose), "Invalid original long-prompt boundary")
            prompt = render(prose[:count])
            need(len(prompt) == record["metrics"]["prompt_chars"] and sha(prompt.encode()) == record["notes"]["prompt_sha256"], "Reconstructed prompt bytes differ")
            return prompt, record["metrics"]["user_prompt_tokens"]

        bench.fit_prompt_tokens = recorded_fit
        replayed = bench.dispatch(record["scenario"])(bench.Ctx(client.args, client, haystack), record["repeat"])
        need(strict_equal(replayed["checks"], record["checks"]) and replayed["verdict"] == record["verdict"]
             and replayed["pass"] is record["pass"], "Unchanged original text scorer disagrees: " + str(key))
        need("notes" not in replayed or strict_equal(replayed["notes"], record["notes"]), "Reconstructed expected/trace notes differ")
        need(len(client.bodies) == len(record["requests"]), "Incomplete reconstructed request matrix")
        bodies[key] = [client.bodies[i] for i in range(len(client.bodies))]
        for request in record["requests"]:
            content = request.get("content") or ""
            need(sha(content.encode()) == request["content_sha256"], "Text content bytes differ")
            normalized = {"content": content, "reasoning": request.get("reasoning") or "", "tool_calls": [
                {"name": call["name"], "arguments": call["parsed"] if call["parsed"] is not None else call["arguments"]}
                for call in request["tool_calls"]]}
            need(sha(canonical(normalized)) == request["response_sha256"], "Text normalized response hash differs")
    manifest_path = directory / "manifest.json"
    manifest = evidence.load(manifest_path)
    need(sha(evidence.read(manifest_path)) == header["manifest_file"]["sha256"] and strict_equal(manifest, header["manifest"]), "Text manifest raw/structured identity differs")
    evaluation = evidence.load(text / "evaluation.json")
    need(evaluation["complete"] is True and evaluation["status"] == "EXECUTED_COMPLETE"
         and evaluation["observed"]["complete"] is True and evaluation["observed"]["sha256"] == sha(raw), "Text campaign is not finalized")
    verdicts = collections.Counter(r["verdict"] for r in cases.values())
    need(evaluation["observed"]["cases"] == 45 and evaluation["observed"]["requests"] == sum(len(r["requests"]) for r in cases.values()), "Final text counts differ")
    need(all(evaluation["observed"]["verdictCounts"].get(k, 0) == verdicts.get(k, 0) for k in ("PASS", "FAIL", "ERROR")), "Final text verdict totals differ")
    expected_quality = all(r["pass"] for r in cases.values())
    need(evaluation["qualityPassed"] is expected_quality and evaluation["observed"]["qualityPassed"] is expected_quality,
         "Final text quality flag contradicts original case verdicts")
    return {"header": header, "cases": cases, "bodies": bodies, "manifest": manifest, "evaluation": evaluation,
            "summary": {"cases": len(cases), "requests": sum(len(r["requests"]) for r in cases.values()),
                "originalChecks": sum(len(r["checks"]) for r in cases.values()), "oldTrueChecks": sum(c["ok"] for r in cases.values() for c in r["checks"]),
                "verdicts": dict(verdicts), "qualityPassed": evaluation["qualityPassed"]}}


def checks_by_name(checks):
    result = {}
    for check in checks:
        need(isinstance(check.get("name"), str) and type(check.get("ok")) is bool and check["name"] not in result, "Invalid/duplicate check identity")
        result[check["name"]] = check
    return result


def truth(old, new, comparable=True):
    need(type(old) is bool and type(new) is bool, "Truth flags must be bool")
    return "not_comparable" if not comparable else ("correct" if old else "wrong") + "_to_" + ("correct" if new else "wrong")


def compare_text(a, b, bench):
    for name in ("harness_sha256", "scenarios", "repeats", "answer_tokens", "decode_tokens", "nonce_seed", "sampling", "tokenizer_sha256", "haystack", "quality_tasks", "gsm8k_limit", "long_order_quality", "long_prompt_tolerance"):
        need(strict_equal(a["header"][name], b["header"][name]), "Cross-profile text protocol differs: " + name)
    requests = []; checks = []; arithmetic = []; matched = {}
    need(a["cases"].keys() == b["cases"].keys(), "Text case matrix differs")
    for key, left in a["cases"].items():
        right = b["cases"][key]; old_requests = left["requests"]; new_requests = right["requests"]
        same = len(old_requests) == len(new_requests)
        for i in range(max(len(old_requests), len(new_requests))):
            old = old_requests[i] if i < len(old_requests) else None; new = new_requests[i] if i < len(new_requests) else None
            identity = old is not None and new is not None and strict_equal(bench.request_identity(a["bodies"][key][i]), bench.request_identity(b["bodies"][key][i]))
            tokens = identity and type(old.get("prompt_tokens")) is int and type(new.get("prompt_tokens")) is int and old["prompt_tokens"] == new["prompt_tokens"]
            same &= identity and tokens
            requests.append({"scenario": key[0], "repeat": key[1], "index": i, "sameCanonicalLogicalRequest": identity,
                "samePromptTokenCount": tokens, "baselineRequestSha256": old.get("request_sha256") if old else None,
                "candidateRequestSha256": new.get("request_sha256") if new else None,
                "baselinePromptTokens": old.get("prompt_tokens") if old else None, "candidatePromptTokens": new.get("prompt_tokens") if new else None,
                "baselineCachedTokens": old.get("cached_tokens") if old else None, "candidateCachedTokens": new.get("cached_tokens") if new else None,
                "responseEqual": old is not None and new is not None and old["response_sha256"] == new["response_sha256"]})
        matched[key] = same
        old_checks = checks_by_name(left["checks"]); new_checks = checks_by_name(right["checks"])
        for name, old in old_checks.items():
            new = new_checks.get(name)
            checks.append({"scenario": key[0], "repeat": key[1], "name": name, "baselineCorrect": old["ok"],
                "candidateCorrect": new is not None and new["ok"], "candidateCheckPresent": new is not None,
                "transition": truth(old["ok"], new is not None and new["ok"], same), "baseline": old, "candidate": new})
        if key[0] == "quality_tasks":
            need(strict_equal(left["notes"]["expected"], right["notes"]["expected"]), "Arithmetic expected fixture changed")
            for i, expected in enumerate(left["notes"]["expected"]):
                values = []
                for rows in (old_requests, new_requests):
                    try:
                        # Use the unchanged task scorer's content parser; the
                        # evidence container itself still rejects duplicate keys.
                        value = json.loads(rows[i]["content"])
                    except (ValueError, TypeError, KeyError, IndexError):
                        value = None
                    values.append(value if isinstance(value, dict) else {})
                for field, answer in expected.items():
                    x, y = values[0].get(field), values[1].get(field)
                    ac, bc = strict_equal(x, answer), strict_equal(y, answer)
                    arithmetic.append({"repeat": key[1], "taskIndex": i, "field": field, "expected": answer,
                        "baselineValue": x, "candidateValue": y, "baselineCorrect": ac, "candidateCorrect": bc,
                        "transition": truth(ac, bc, same)})
    result = {"comparable": all(matched.values()), "requests": requests, "originalChecks": checks, "arithmeticFields": arithmetic}
    for name, rows in (("original", checks), ("arithmetic", arithmetic)):
        result[name + "OldTrue"] = sum(r["baselineCorrect"] for r in rows)
        result[name + "Transitions"] = dict(collections.Counter(r["transition"] for r in rows))
        result[name + "LostTruths"] = [r for r in rows if r["baselineCorrect"] and r["transition"] != "correct_to_correct"]
        result[name + "Gains"] = [r for r in rows if r["transition"] == "wrong_to_correct"]
    result["noLossPassed"] = result["comparable"] and not result["originalLostTruths"] and not result["arithmeticLostTruths"]
    return result


def program_path(directory):
    for relative in ("program-posix/campaign.json", "program/campaign.json", "program/program/campaign.json"):
        path = Path(directory) / relative
        if path.is_file():
            return path
    raise ValueError("A complete original program campaign is required: " + str(directory))


def audit_program(directory, tool, evidence, manifest):
    path = program_path(directory); campaign = evidence.load(path)
    need(campaign.get("complete") is True and campaign.get("status") == "EXECUTED_COMPLETE" and campaign.get("identities_unchanged") is True,
         "Incomplete program campaign cannot establish no loss or quality rank")
    need(campaign["tool_sha256"] == PROGRAM_SHA and strict_equal(campaign["tasks"], tool.TASKS)
         and campaign["suite_sha256"] == sha(tool.encoded(tool.TASKS)) and campaign["repeats"] == tool.REPEATS
         and campaign["max_tokens"] == tool.MAX_TOKENS, "Frozen program suite/protocol differs")
    need(strict_equal(campaign["server_manifest"]["value"], manifest)
         and campaign["server_manifest"]["sha256"] == sha(evidence.read(Path(directory) / "manifest.json")), "Program server manifest differs")
    tasks = {task["id"]: task for task in tool.TASKS}; records = {}; executed = {}
    for row in campaign["records"]:
        key = (row["task"], row["repeat"])
        need(type(key[1]) is int and key not in records and key[0] in tasks, "Duplicate/invalid program identity")
        cases = tool.validate_comparison_record(row, tasks[key[0]], campaign["model"], campaign["nonce_seed"])
        request = row["request"]; raw = request.get("raw_response")
        if isinstance(raw, str):
            need(sha(raw.encode()) == request["raw_response_sha256"] and len(raw.encode()) == request["raw_response_bytes"], "Program raw response identity differs")
        if request.get("error") is None:
            response = parse(raw); need(strict_equal(response, request["response"]), "Program decoded raw response differs")
            choice = response["choices"][0]; message = choice["message"]
            need(message["content"] == request["content"] and choice["finish_reason"] == request["finish_reason"], "Program content/finish metadata differs")
            normalized = {"content": message["content"], "reasoning": message.get("reasoning_content", message.get("reasoning", "")), "tool_calls": message.get("tool_calls", [])}
            need(sha(tool.encoded(normalized)) == request["response_sha256"], "Program normalized response hash differs")
            evaluation = row["evaluation"]
            # Replay the original pre-execution scorer, including legitimate
            # length/syntax failures. Never repair or execute a generated module.
            replayed = {"finish_reason_stop": request["finish_reason"] == "stop", "complete_module": False, "closed_AST_policy": False}
            policy = None
            if not replayed["finish_reason_stop"]:
                policy = "incomplete/length-limited response; no repair or program execution"
            else:
                try:
                    source = tool.extract_program(request["content"]); tree = tool.ast.parse(source)
                    replayed["complete_module"] = bool(tree.body)
                    if not tree.body:
                        raise tool.Rejected("empty complete module")
                    tool.validate_program(source); replayed["closed_AST_policy"] = True
                except (ValueError, SyntaxError, RecursionError) as error:
                    policy = type(error).__name__ + ": " + str(error)
            need(all(evaluation["checks"][k] is v for k, v in replayed.items()), "Unchanged completion/AST policy scorer differs")
            if policy is not None:
                need(policy == evaluation.get("error"), "Original policy rejection differs")
            else:
                source_file = path.parent / (key[0] + "-" + str(key[1])) / "program.py"
                need(evidence.read(source_file) == source.encode() and evaluation["program_sha256"] == sha(source.encode()), "Persisted generated source differs")
        for case_id, case in cases.items():
            folder = path.parent / (key[0] + "-" + str(key[1])) / ("case-%02d" % case_id)
            for filename, field in (("stdout.txt", "stdout"), ("stderr.txt", "stderr")):
                data = case[field].encode(); need(evidence.read(folder / filename) == data and sha(data) == case[field + "_sha256"] and len(data) == case[field + "_bytes"], "Persisted worker output differs")
            need(strict_equal(evidence.load(folder / "worker.json"), case["worker"]), "Persisted worker metadata differs")
            raw_input = evidence.read(folder / "stdin.json")
            need(sha(raw_input) == case["input_sha256"] and strict_equal(parse(raw_input), case["input"]), "Persisted worker input differs")
        records[key] = row; executed[key] = cases
    need(set(records) == {(task["id"], repeat) for task in tool.TASKS for repeat in range(tool.REPEATS)}, "Complete30-program matrix required")
    verdicts = collections.Counter(row["verdict"] for row in records.values())
    need(all(campaign["verdict_counts"].get(k, 0) == verdicts.get(k, 0) for k in ("PASS", "FAIL", "ERROR")), "Final program verdict totals differ")
    need(campaign["quality_passed"] is all(row["passed"] for row in records.values()), "Final program quality flag contradicts actual program verdicts")
    return {"campaign": campaign, "records": records, "executed": executed,
            "summary": {"programs": len(records), "verdicts": dict(verdicts), "functionalExecuted": sum(len(v) for v in executed.values()),
                "functionalOldTrue": sum(c["passed"] for rows in executed.values() for c in rows.values()),
                "functionalNotExecuted": campaign["planned_functional_cases"] - sum(len(v) for v in executed.values()),
                "qualityPassed": campaign["quality_passed"]}}


def compare_program(a, b, tool):
    need(a["records"].keys() == b["records"].keys(), "Program matrix differs")
    need(a["campaign"]["nonce_seed"] == b["campaign"]["nonce_seed"], "Program nonce differs")
    programs = []; checks = []; functional = []
    tasks = {task["id"]: task for task in tool.TASKS}
    for key, left in a["records"].items():
        right = b["records"][key]
        # The frozen program identity includes the model name. Require its exact
        # original wire body/hash; do not downgrade to shared-prompt-only parity.
        same = strict_equal(left["request"]["request_body"], right["request"]["request_body"])
        programs.append({"task": key[0], "repeat": key[1], "sameWireRequest": same,
            "baselinePassed": left["passed"], "candidatePassed": right["passed"], "transition": truth(left["passed"], right["passed"], same)})
        for name, old, new in (("completeGeneration", left["request"].get("http_status") == 200 and left["request"].get("error") is None and left["request"].get("raw_response_complete") is True and left["request"].get("finish_reason") == "stop",
                right["request"].get("http_status") == 200 and right["request"].get("error") is None and right["request"].get("raw_response_complete") is True and right["request"].get("finish_reason") == "stop"),):
            checks.append({"task": key[0], "repeat": key[1], "name": name, "baselineCorrect": old, "candidateCorrect": new, "transition": truth(old, new, same)})
        old_checks = (left.get("evaluation") or {}).get("checks", {}); new_checks = (right.get("evaluation") or {}).get("checks", {})
        for name, old in old_checks.items():
            new = new_checks.get(name) is True
            checks.append({"task": key[0], "repeat": key[1], "name": name, "baselineCorrect": old, "candidateCorrect": new, "transition": truth(old, new, same)})
        for fixture in tasks[key[0]]["cases"]:
            old = a["executed"][key].get(fixture["id"]); new = b["executed"][key].get(fixture["id"])
            ac, bc = old is not None and old["passed"], new is not None and new["passed"]
            functional.append({"task": key[0], "repeat": key[1], "caseId": fixture["id"], "input": fixture["input"], "expected": fixture["expected"],
                "baselineCorrect": ac, "candidateCorrect": bc, "baselineStatus": old["status"] if old else "NOT_EXECUTED",
                "candidateStatus": new["status"] if new else "NOT_EXECUTED", "transition": truth(ac, bc, same)})
    result = {"comparable": all(r["sameWireRequest"] for r in programs), "programs": programs, "checks": checks, "functionalCases": functional,
              "functionalOldTrue": sum(r["baselineCorrect"] for r in functional)}
    result["lostFunctionalTruths"] = [r for r in functional if r["baselineCorrect"] and r["transition"] != "correct_to_correct"]
    result["lostChecks"] = [r for r in checks if r["baselineCorrect"] and r["transition"] != "correct_to_correct"]
    result["lostPrograms"] = [r for r in programs if r["baselinePassed"] and r["transition"] != "correct_to_correct"]
    result["functionalTransitions"] = dict(collections.Counter(r["transition"] for r in functional))
    result["noLossPassed"] = result["comparable"] and not any(result[k] for k in ("lostFunctionalTruths", "lostChecks", "lostPrograms"))
    return result


def fresh_output(path):
    path = Path(os.path.abspath(path))
    need(not path.exists() and not path.is_symlink() and not Path(str(path) + ".sha256").exists(), "Fresh audit and SHA output required")
    parts = path.parts
    need("artifacts" in parts or any(parts[i:i + 2] == ("docs", "validation") for i in range(len(parts) - 1)), "Ignored evidence output required")
    need(all(not ancestor.is_symlink() for ancestor in (path, *path.parents)), "Symlink output ancestry refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def audit(baseline, candidate, scope, evidence):
    bench, program, haystack = frozen_tools(evidence)
    old = audit_text(baseline, bench, haystack, evidence); new = audit_text(candidate, bench, haystack, evidence)
    text = compare_text(old, new, bench)
    result = {"text": text, "baselineText": old["summary"], "candidateText": new["summary"],
        "recordedRuntimeAndWeightScope": {"baseline": old["manifest"], "candidate": new["manifest"],
            "sameDeclaredKnownArtifactSha256": isinstance(old["evaluation"].get("knownArtifactSha256"), str)
                and len(old["evaluation"]["knownArtifactSha256"]) == 64
                and old["evaluation"]["knownArtifactSha256"] == new["evaluation"].get("knownArtifactSha256"),
            "fullArtifactSha256RecomputedByThisAudit": False,
            "promptTokenIdsRecomputedByThisAudit": False}, "program": {"status": "NOT_RUN_EXPLICIT_TEXT_ONLY"}}
    if scope == "text-program":
        pa = audit_program(baseline, program, evidence, old["manifest"]); pb = audit_program(candidate, program, evidence, new["manifest"])
        result.update(program=compare_program(pa, pb, program), baselineProgram=pa["summary"], candidateProgram=pb["summary"])
        for directory, profile, pp in ((baseline, old, pa), (candidate, new, pb)):
            completed = Path(directory) / "completed-campaign.json"
            if completed.is_file():
                outer = evidence.load(completed)
                need(outer["complete"] is True and strict_equal(outer["textCompletedObserved"], profile["evaluation"])
                     and strict_equal(outer["programCompletedObserved"], pp["campaign"]), "Outer snapshot does not match finalized inner campaigns")
    result["requestedScopeNoLossPassed"] = text["noLossPassed"] and (scope == "text" or result["program"]["noLossPassed"])
    result["fullTextAndProgramNoLossPassed"] = result["requestedScopeNoLossPassed"] if scope == "text-program" else None
    return result


def self_test():
    need(not strict_equal(True, 1) and not strict_equal(["2"], [2]), "JSON type distinctions lost")
    need(truth(True, False) == "correct_to_wrong" and truth(True, True, False) == "not_comparable", "Truth transition policy differs")
    for data in ('{"x":1,"x":2}', '{"x":NaN}'):
        try:
            parse(data)
        except ValueError:
            pass
        else:
            raise ValueError("Malformed JSON accepted")
    return {"CPU": "PASS", "contracts": 6, "GPU": "NOT_RUN", "generatedCodeExecution": "NONE"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path); parser.add_argument("--candidate", type=Path)
    parser.add_argument("--scope", choices=("text-program", "text"), default="text-program")
    parser.add_argument("--output", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if any(x is not None for x in (args.baseline, args.candidate, args.output)):
            parser.error("--self-test cannot be combined with evidence/output arguments")
        print(json.dumps(self_test())); return 0
    if args.baseline is None or args.candidate is None or args.output is None:
        parser.error("--baseline, --candidate and --output are required")
    output = fresh_output(args.output); evidence = Evidence()
    result = {"schema": "ninfer-api-frozen-no-loss-audit/1", "auditComplete": False, "scope": args.scope,
        "baseline": str(args.baseline.absolute()), "candidate": str(args.candidate.absolute()), "VM": "NONE", "GPU": "NONE",
        "generatedCodeExecution": "NONE", "source": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).read_bytes())},
        "limits": ["No gains cancel a lost truth. Missing/corrupted/incomplete evidence or changed inputs never establish no loss.",
            "Text canonical identity is the unchanged benchmark definition: model names and generated transport call IDs are normalized; every generation setting and message remains bound. Program wire identity includes the model name.",
            "Original scorers run offline against stored complete responses and original bounded worker stdout; no program is executed again and no score is rewritten.",
            "Declared runtime/weight configurations are recorded, not independently loaded or device-verified. Identical prompt counts do not prove identical token IDs.",
            "Explicit text-only scope does not establish program no loss. No overall quality/performance/engine ranking follows, including from an incomplete Ignis campaign."]}
    status = 1
    try:
        result.update(audit(args.baseline.absolute(), args.candidate.absolute(), args.scope, evidence))
        evidence.verify(); result["auditComplete"] = True
        status = 0 if result["requestedScopeNoLossPassed"] else 2
    except Exception as error:
        result["error"] = str(error)
    result["inputFilesRehashed"] = list(evidence.pins.values())
    raw = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        stream.write(raw)
    with Path(str(output) + ".sha256").open("x") as stream:
        stream.write(sha(raw) + "  " + output.name + "\n")
    print(json.dumps({"auditComplete": result["auditComplete"], "scope": args.scope,
        "requestedScopeNoLossPassed": result.get("requestedScopeNoLossPassed"), "output": str(output), "sha256": sha(raw), "error": result.get("error")}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
