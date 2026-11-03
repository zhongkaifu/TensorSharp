"""Inline Jev text/document fixture checks; no vision tower or ASR is used.

The owned server wrapper verifies model/backend and supplies bounded, redirect-free
HTTP plus scenario-local upload storage. Only SHA-pinned repository fixture bytes
are submitted; uploaded-reference and remote service paths are not exercised.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import urllib.error

from .jev import load_contract, validate_decisions


MANIFEST = "InferenceWeb.Tests/Fixtures/JevAttachments/manifest.json"
_FIELDS = ("files", "documents", "videos", "audios")
_TEXT_KINDS = {"text", "pdf", "document"}


def _json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate fixture/response key: " + key)
            result[key] = item
        return result
    def reject(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(value, object_pairs_hook=pairs, parse_constant=reject)


def load_text_fixtures(repo):
    path = Path(repo) / MANIFEST
    raw = path.read_bytes()
    manifest = _json(raw)
    if manifest.get("version") != 1 or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("invalid Jev attachment manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Jev attachment file identities are absent")
    identifiers = [case.get("id") for case in manifest["cases"]]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Jev attachment case IDs must be nonempty and unique")
    selected, excluded, verified = [], [], {}
    root = path.parent.resolve()
    for case in manifest["cases"]:
        if not isinstance(case.get("requires"), list):
            raise ValueError("Jev attachment fixture must explicitly declare requirements")
        if case["requires"]:
            excluded.append({"id": case["id"], "requires": case["requires"], "status": "untested",
                             "reason": "outside text/document adapter; vision and ASR are separate coverage"})
            continue
        if not isinstance(case.get("questions"), dict) or not case["questions"] or set(case.get("expected", {})) != set(case["questions"]):
            raise ValueError("Jev attachment expected question keys differ")
        attachments = case.get("attachments")
        if not isinstance(attachments, list) or len(attachments) > 8:
            raise ValueError("Jev attachment fixture list is invalid")
        for attachment in attachments:
            name = attachment.get("path")
            if attachment.get("kind") not in _TEXT_KINDS or attachment.get("field") not in {"files", "documents"}:
                raise ValueError("text-only fixture would exercise an undeclared media requirement")
            if not isinstance(name, str) or name not in files or Path(name).name != name or "/" in name or "\\" in name or ":" in name:
                raise ValueError("Jev fixture must name a pinned bare filename")
            source = (root / name).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Jev fixture resolves outside its repository fixture directory")
            if name in verified:
                continue
            identity = files[name]
            size = identity.get("bytes")
            if type(size) is not int or not 0 < size <= 1024 * 1024 or not 0 < source.stat().st_size <= 1024 * 1024:
                raise ValueError("Jev fixture size mismatch or capture budget exceeded: " + name)
            data = source.read_bytes()
            source_bytes, source_digest = len(data), hashlib.sha256(data).hexdigest()
            normalization = "none"
            # Git autocrlf changes checked-out text files on Windows. Submit only
            # the published canonical fixture bytes, and record both identities.
            # Binary/office/PDF bytes never take this path; arbitrary edits still
            # fail the exact canonical size+hash check below.
            if source_digest != identity.get("sha256") and attachment["kind"] == "text" and b"\r\n" in data:
                data = data.replace(b"\r\n", b"\n")
                normalization = "git_checkout_crlf_to_lf"
            if len(data) != size:
                raise ValueError("Jev fixture size mismatch: " + name)
            digest = hashlib.sha256(data).hexdigest()
            if digest != identity.get("sha256"):
                raise ValueError("Jev fixture SHA256 mismatch: " + name)
            verified[name] = {"sha256": digest, "bytes": len(data), "data": data,
                              "source_sha256": source_digest, "source_bytes": source_bytes, "normalization": normalization}
        selected.append(case)
    if not selected or not any(case["attachments"] for case in selected):
        raise ValueError("no text/document attachment cases selected")
    if sum(item["bytes"] for item in verified.values()) > 1024 * 1024:
        raise ValueError("combined Jev text fixture bytes exceed bounded adapter budget")
    return selected, verified, excluded, hashlib.sha256(raw).hexdigest()


def make_payload(case, model, files):
    payload = {"model": model, "seed": 42, "samples": 1,
               "state": copy.deepcopy(case.get("state", {"task": "Evaluate only the attached source material."})),
               "questions": copy.deepcopy(case["questions"])}
    for attachment in case["attachments"]:
        name = attachment["path"]
        payload.setdefault(attachment["field"], []).append({"name": name, "data": base64.b64encode(files[name]["data"]).decode("ascii")})
    return payload


def validate_preparation(case, response):
    diagnostics = response.get("diagnostics", {})
    expected = sorted(case["attachments"], key=lambda item: _FIELDS.index(item["field"]))
    actual = diagnostics.get("attachments")
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError("attachment diagnostics missing or attachment count differs")
    if type(diagnostics.get("images")) is not int or diagnostics["images"] != 0:
        raise ValueError("text/document fixture unexpectedly used image spans")
    for wanted, got in zip(expected, actual):
        if not isinstance(got, dict) or got.get("name") != wanted["path"] or got.get("kind") != wanted["kind"]:
            raise ValueError("attachment name/kind/order differs from fixture")
        if type(got.get("textCharacters")) is not int or got["textCharacters"] <= 0:
            raise ValueError("text/document attachment produced no extracted text")
        if type(got.get("imageCount")) is not int or got["imageCount"] != 0 or got.get("sampled") is not False:
            raise ValueError("text/document attachment used image sampling")
        if type(got.get("cacheHit")) is not bool:
            raise ValueError("attachment cache-hit observation missing")
    timing = diagnostics.get("timing", {})
    for key in ("preprocessing_ms", "inference_ms", "total_ms"):
        value = timing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("missing/invalid attachment timing: " + key)
    if timing["total_ms"] + 1 < timing["preprocessing_ms"] + timing["inference_ms"]:
        raise ValueError("attachment total timing excludes preparation/inference")
    return {"attachments": actual, "timing": timing}


def _assert(result, name, passed, expected=None, actual=None):
    result["assertions"].append({"name": name, "passed": bool(passed), "expected": expected, "actual": actual})


def _post(http, payload, result):
    offset = len(result["evidence"].get("responses", []))
    body = http.json("/v1/systemone", payload)
    captured = result["evidence"].get("responses", [])[offset:]
    if len(captured) != 1 or captured[0].get("route") != "/v1/systemone" or captured[0].get("status") != 200:
        raise ValueError("Jev attachment request has no current HTTP 200 response evidence")
    return body


def invalid_payloads(model):
    base = {"model": model, "state": "Synthetic validation fixture", "questions": {"q": {"type": "noul", "instructions": "Is the ticket urgent?"}}}
    for name, change in (
        ("malformed-base64", {"files": [{"name": "ticket.txt", "data": "%%%"}]}),
        ("ambiguous-source", {"files": [{"name": "ticket.txt", "data": "eA==", "file": "ticket.txt"}]}),
        ("path-traversal", {"files": [{"file": "../ticket.txt"}]}),
        ("unsupported-extension", {"files": [{"name": "fixture.exe", "data": "eA=="}]}),
        ("wrong-modality", {"videos": [{"name": "ticket.txt", "data": "eA=="}]}),
        ("too-many-attachments", {"files": [{"name": "ticket.txt", "data": "eA=="}] * 9}),
        ("invalid-utf8", {"files": [{"name": "invalid.txt", "data": "/w=="}]}),
        ("binary-control-character", {"files": [{"name": "binary.txt", "data": "aABp"}]}),
    ):
        yield name, {**copy.deepcopy(base), **change}


def _invalid(http, payload, result):
    offset = len(result["evidence"].get("http_errors", []))
    try:
        http.json("/v1/systemone", payload)
    except urllib.error.HTTPError as error:
        captured = result["evidence"].get("http_errors", [])[offset:]
        if error.code != 422 or len(captured) != 1 or captured[0].get("status") != 422 or captured[0].get("route") != "/v1/systemone":
            raise ValueError("invalid attachment requires current HTTP 422 evidence") from error
        body = _json(captured[0].get("body", ""))
        fields = body.get("error") if isinstance(body, dict) else None
        if not isinstance(fields, dict) or fields.get("type") != "validation_error" or not isinstance(fields.get("message"), str) or not fields["message"].strip():
            raise ValueError("invalid attachment requires typed validation_error with a message") from error
        return {"status": 422, "type": fields["type"], "message": fields["message"]}
    raise ValueError("invalid attachment was accepted")


def run_text_attachments(http, model, context, result):
    cases, files, excluded, manifest_hash = load_text_fixtures(context["repo"])
    contract = load_contract(context["repo"])
    evidence = result["evidence"]["jev_text_attachments"] = {
        "manifest": MANIFEST, "manifest_sha256": manifest_hash,
        "files": {name: {key: value for key, value in item.items() if key != "data"} for name, item in files.items()},
        "mode": "inline", "seed": 42, "samples": 1, "cases": [], "excluded_cases": excluded, "invalid_requests": [],
        "upload_directory": str(Path(context["artifact_dir"]).resolve() / "uploads"),
        "limitations": ["Text and office/PDF text extraction only; no vision, video, ASR, macros or formula recalculation.",
                        "No upload-reference transport, concurrent isolation, representative quality or performance claim."]}
    result["inputs"]["jev_text_attachment_cases"] = [case["id"] for case in cases]
    loaded_model = Path(context["model_path"]).name
    failures, repeat = [], None
    for case in cases:
        payload = make_payload(case, model, files)
        row = {"id": case["id"], "passed": False, "request_sha256": contract.hash_value(payload), "expected": case["expected"]}
        evidence["cases"].append(row)
        try:
            body = _post(http, payload, result)
            row["decisions"] = validate_decisions(contract, payload, body, case["expected"], loaded_model)
            row["preparation"] = validate_preparation(case, body)
            row["passed"] = True
            if repeat is None and case["attachments"]:
                repeat = case, payload, row
        except (ValueError, KeyError, TypeError, OSError) as error:
            row["error"] = str(error)
            failures.append(case["id"] + ": " + str(error))
        _assert(result, "Jev attachment " + case["id"], row["passed"], case["expected"], row.get("decisions") if row["passed"] else row.get("error"))
    replay = evidence["cache_replay"] = {"passed": False}
    if repeat:
        case, payload, baseline = repeat
        replay.update(case=case["id"], request_sha256=contract.hash_value(payload))
        try:
            body = _post(http, payload, result)
            decisions = validate_decisions(contract, payload, body, case["expected"], loaded_model)
            preparation = validate_preparation(case, body)
            if not all(item["cacheHit"] is True for item in preparation["attachments"]):
                raise ValueError("no positive attachment extraction cache hit on exact replay")
            delta = max(abs(a - b) for first, second in zip(baseline["decisions"], decisions) for a, b in zip(first["probabilities"], second["probabilities"]))
            if delta > 1e-5:
                raise ValueError("cached attachment decision probabilities changed beyond 1e-5")
            replay.update(passed=True, preparation=preparation, max_absolute_probability_delta=delta)
        except (ValueError, KeyError, TypeError, OSError) as error:
            replay["error"] = str(error)
    else:
        replay["error"] = "no valid attachment baseline; cache behavior not exercised"
    if not replay["passed"]:
        failures.append("cache replay: " + replay["error"])
    _assert(result, "Jev exact attachment replay has positive cache hit and stable decisions", replay["passed"], "all cacheHit=true; probability delta<=1e-5", replay)
    for name, payload in invalid_payloads(model):
        row = {"id": name, "passed": False}
        evidence["invalid_requests"].append(row)
        try:
            row.update(_invalid(http, payload, result), passed=True)
        except (ValueError, KeyError, TypeError, OSError) as error:
            row["error"] = str(error)
            failures.append(name + ": " + str(error))
        _assert(result, "Jev attachment rejects " + name, row["passed"], "HTTP 422 validation_error", row)
    evidence["counts"] = {"fixtures": len(cases), "passed_fixtures": sum(row["passed"] for row in evidence["cases"]),
                          "decisions": sum(len(case["expected"]) for case in cases), "invalid_requests": 8,
                          "excluded_media_cases": len(excluded), "failures": len(failures)}
    if failures:
        raise ValueError("Jev text/document assertions failed: " + "; ".join(failures))
