"""Bounded Jev contract/quality screen using an already owned loopback server.

The caller must verify its server's loaded model/backend before calling this helper.
It owns the listener, deadline, redirect policy, response limits and teardown. This
module never starts a server, follows a URL or executes a model-generated tool.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.error


_FIXTURE_PATH = "TensorSharp.TestMatrix/Inputs/jev/decisions.json"
_VALIDATOR_PATH = "eng/jev-benchmark.py"
_REPEAT_TOLERANCE = 1e-5


def load_contract(repo):
    """Reuse the repository's typed answer validator without its HTTP/process CLI."""
    path = Path(repo) / _VALIDATOR_PATH
    spec = importlib.util.spec_from_file_location("tensorsharp_jev_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError("Jev contract validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixtures(repo):
    path = Path(repo) / _FIXTURE_PATH
    raw = path.read_bytes()
    suite = json.loads(raw)
    cases, schemas = suite.get("cases"), suite.get("schemas")
    if not isinstance(cases, list) or not cases or not isinstance(schemas, dict):
        raise ValueError("Jev fixture suite is empty or malformed")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Jev fixture case IDs must be nonempty and unique")
    for case in cases:
        schema = schemas.get(case.get("schema"))
        if not isinstance(schema, dict) or not schema or set(case.get("expected", {})) != set(schema):
            raise ValueError("Jev fixture expected keys do not match its question schema: " + case["id"])
        if "state" not in case:
            raise ValueError("Jev fixture state is missing: " + case["id"])
    return suite, hashlib.sha256(raw).hexdigest()


def _assert(result, name, passed, expected=None, actual=None):
    result["assertions"].append({"name": name, "passed": bool(passed), "expected": expected, "actual": actual})
    return bool(passed)


def validate_decisions(contract, payload, body, expected, loaded_model):
    """Check served identity, seeded read controls and independent gold decisions."""
    decisions = contract.validate_response(payload, {"status": 200, "body": body, "error": None}, expected)
    if body["model"] != loaded_model:
        raise ValueError("Jev response model differs from verified loaded model")
    diagnostics = body.get("diagnostics", {})
    samples = diagnostics.get("samples", {})
    if (diagnostics.get("engine") != "tensorsharp" or type(diagnostics.get("steps")) is not int or diagnostics.get("steps") != 1 or
            type(diagnostics.get("seed")) is not int or diagnostics.get("seed") != 42 or samples.get("policy") != "fixed" or
            type(samples.get("n")) is not int or samples.get("n") != 1):
        raise ValueError("Jev diagnostics do not verify native one-step seed=42 samples=1 execution")
    if diagnostics.get("probability_semantics") != "conditional_label_softmax_temperature_1":
        raise ValueError("Jev probability semantics missing or changed")
    wrong = [item["question"] for item in decisions if item["correct"] is not True]
    if wrong:
        raise ValueError("Jev fixture decision disagrees with expected label: " + ", ".join(wrong))
    return decisions


def _malformed_requests(model):
    return [
        ("missing_state", {"model": model, "questions": {"x": {"type": "noul", "instructions": "yes?"}}}),
        ("empty_questions", {"model": model, "state": "test", "questions": {}}),
        ("unknown_question_type", {"model": model, "state": "test", "questions": {"x": {"type": "unknown"}}}),
        ("one_choice", {"model": model, "state": "test", "questions": {"x": {"type": "choice", "criteria": {"only": "only"}}}}),
    ]


def run_decisions(http, model, context, result):
    """Run every fixed fixture, seeded replay and malformed requests sequentially.

    Raises ValueError on a failed assertion. Full response/request capture belongs
    to the supplied owned-server HTTP client; per-case outcomes and fixture hashes
    are retained here even when another fixture fails.
    """
    repo = Path(context["repo"])
    suite, fixture_hash = load_fixtures(repo)
    contract = load_contract(repo)
    loaded_model = Path(context["model_path"]).name
    evidence = result["evidence"]["jev"] = {
        "fixture_path": _FIXTURE_PATH, "fixture_sha256": fixture_hash,
        "validator_path": _VALIDATOR_PATH,
        "validator_sha256": hashlib.sha256((repo / _VALIDATOR_PATH).read_bytes()).hexdigest(),
        "seed": 42, "samples": 1, "cases": [], "malformed_requests": [],
        "repeat_absolute_tolerance": _REPEAT_TOLERANCE,
        "limitations": ["Small synthetic decision fixtures do not establish general accuracy or probability calibration.",
                        "No attachment, ASR, concurrent isolation or performance coverage is claimed."],
    }
    result["inputs"]["jev_fixture_ids"] = [case["id"] for case in suite["cases"]]
    failures, first = [], None
    for case in suite["cases"]:
        payload = contract.make_payload(case, suite["schemas"][case["schema"]], model, 1, 0)
        row = {"id": case["id"], "request_sha256": contract.hash_value(payload), "expected": case["expected"], "passed": False}
        evidence["cases"].append(row)
        try:
            body = http.json("/v1/systemone", payload)
            decisions = validate_decisions(contract, payload, body, case["expected"], loaded_model)
            row.update(passed=True, decisions=decisions)
            if first is None:
                first = (case, payload, decisions)
        except (ValueError, TypeError, KeyError, OSError) as error:
            row["error"] = str(error)
            failures.append(case["id"] + ": " + str(error))
        _assert(result, "Jev fixture " + case["id"], row["passed"], case["expected"], row.get("decisions") or row.get("error"))

    if first is not None:
        case, payload, previous = first
        replay = {"case": case["id"], "passed": False, "request_sha256": contract.hash_value(payload)}
        evidence["seeded_replay"] = replay
        try:
            current = validate_decisions(contract, payload, http.json("/v1/systemone", payload), case["expected"], loaded_model)
            delta = max(abs(a - b) for old, new in zip(previous, current)
                        for a, b in zip(old["probabilities"], new["probabilities"]))
            replay.update(max_absolute_probability_delta=delta, passed=delta <= _REPEAT_TOLERANCE)
            if not replay["passed"]:
                failures.append("seeded replay probability delta exceeds tolerance")
        except (ValueError, TypeError, KeyError, OSError) as error:
            replay["error"] = str(error)
            failures.append("seeded replay: " + str(error))
        _assert(result, "Jev seeded probability replay", replay["passed"], "maximum absolute probability delta <= 1e-5", replay)
    else:
        _assert(result, "Jev seeded probability replay", False, "one valid fixture required", "no valid fixture")

    for name, payload in _malformed_requests(model):
        status, detail, error_type = 200, None, None
        error_offset = len(result["evidence"].get("http_errors", []))
        try:
            http.json("/v1/systemone", payload)
        except urllib.error.HTTPError as error:
            status = error.code
            captured = result["evidence"].get("http_errors", [])[error_offset:]
            try:
                if len(captured) != 1 or captured[0].get("route") != "/v1/systemone" or captured[0].get("status") != status:
                    raise ValueError("missing current-request HTTP error evidence")
                body = json.loads(captured[0]["body"])
                fields = body.get("error") if isinstance(body, dict) else None
                if not isinstance(fields, dict) or not isinstance(fields.get("message"), str) or not fields["message"].strip():
                    raise ValueError("Jev validation error envelope/message missing")
                error_type = fields.get("type")
            except (ValueError, KeyError, TypeError) as validation_error:
                detail = str(validation_error)
        except (ValueError, TypeError, KeyError, OSError) as error:
            status, detail = None, str(error)
        # These are valid JSON with invalid Jev fields. JevRequest.Parse throws
        # JevValidationException; JevAdapter maps exactly to this 422 error type.
        # HTTP 400 is reserved for malformed JSON, which these fixtures are not.
        passed = status == 422 and error_type == "validation_error" and detail is None
        evidence["malformed_requests"].append({"id": name, "status": status, "type": error_type, "passed": passed, "error": detail})
        _assert(result, "Jev rejects " + name, passed, {"status": 422, "type": "validation_error"},
                {"status": status, "type": error_type, "error": detail})
        if not passed:
            failures.append(name + ": expected HTTP 422 validation_error, got " + str(status) + " " + str(error_type) + (": " + detail if detail else ""))
    evidence["counts"] = {"fixtures": len(suite["cases"]), "passed_fixtures": sum(row["passed"] for row in evidence["cases"]),
                          "decisions": sum(len(case["expected"]) for case in suite["cases"]), "failures": len(failures)}
    if failures:
        raise ValueError("Jev regression assertions failed: " + "; ".join(failures))
