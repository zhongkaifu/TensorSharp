"""Typed, fixture-pinned grounded retrieval and Jev media profile executors.

Only owned loopback hosts are contacted. Profiles select catalog models and
literal independent oracles, never commands, remote servers, or generated tools.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import socket
import time
import urllib.error
import uuid
import wave

from .processes import ManagedProcess
from .scenarios import (ScenarioFailure, ScenarioUntested, _Http, _base_args, _require, _json_loads,
                        _hosted_response_model, _attest_offload, _attest_offload_architecture,
                        backend_evidence, validate_chat)
from .knowledge_retrieval import _capture_json, _read, host_environment
from .jev import load_contract, validate_decisions

SCENARIOS = {"retrieval", "jev_media"}
DOCUMENT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".html", ".pdf", ".docx", ".xlsx", ".pptx"}
MEDIA_SUFFIXES = {"image": {".png"}, "video": {".mp4", ".webm"}, "audio": {".wav"}}


def required_options(profile):
    validate_profile(profile)
    return {"mmproj"} if profile["scenario"] == "jev_media" and profile["parameters"]["media_kind"] in {"image", "video"} else set()


def _keys(value, required, optional=(), name="object"):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(name + " requires keys " + str(sorted(required)) + "; optional " + str(sorted(optional)))


def _text(value, limit, name):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(name + " must be a nonempty bounded string")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def validate_profile(profile):
    """Validate typed subcontracts during planning; do not inspect model payloads."""
    scenario = profile.get("scenario")
    if scenario not in SCENARIOS:
        raise ValueError("unknown knowledge profile scenario")
    params, expected = profile.get("parameters"), profile.get("expected")
    if scenario == "retrieval":
        _keys(params, {"embedding_backend", "question", "search_query"}, {"max_tokens"}, "retrieval parameters")
        _keys(expected, {"answer_contains", "source_excerpt"}, name="retrieval expected")
        _keys(profile.get("models"), {"embedding"}, name="retrieval models")
        _keys(profile.get("fixtures"), {"document"}, name="retrieval fixtures")
        if params["embedding_backend"] != "cpu":
            raise ValueError("retrieval sidecar qualification requires embedding_backend='cpu'")
        for key, limit in (("question", 1024), ("search_query", 512)):
            _text(params[key], limit, key)
        for key, limit in (("answer_contains", 256), ("source_excerpt", 1024)):
            _text(expected[key], limit, key)
        if expected["answer_contains"].casefold() in (params["question"] + " " + params["search_query"]).casefold():
            raise ValueError("grounded answer oracle must not be supplied in the question/query")
        _integer(params.get("max_tokens", 384), 128, 512, "max_tokens")
    else:
        _keys(params, {"media_kind", "state", "questions"}, name="Jev media parameters")
        _keys(expected, {"decisions", "control_decisions", "image_count"}, name="Jev media expected")
        _keys(profile.get("models"), set(), name="Jev models")
        _keys(profile.get("fixtures"), {"media"}, name="Jev fixtures")
        kind = params["media_kind"]
        if kind not in MEDIA_SUFFIXES:
            raise ValueError("media_kind must be image, video or audio")
        _text(params["state"], 2048, "state")
        questions = params["questions"]
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 8:
            raise ValueError("Jev profile requires 1..8 labeled questions")
        for gold in (expected["decisions"], expected["control_decisions"]):
            if not isinstance(gold, dict) or set(gold) != set(questions):
                raise ValueError("Jev gold keys must match question ownership")
        for name, question in questions.items():
            if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
                raise ValueError("Jev question ID must be a short bare identifier")
            _keys(question, {"type", "instructions"}, {"criteria"}, "Jev question")
            _text(question["instructions"], 1024, "question instructions")
            kind_q, criteria = question["type"], question.get("criteria")
            golds = [expected["decisions"][name], expected["control_decisions"][name]]
            if kind_q == "noul":
                if criteria is not None or any(type(gold) is not bool for gold in golds):
                    raise ValueError("profile noul requires boolean oracles and no criteria")
            elif kind_q in {"choice", "score"}:
                if (kind_q == "choice" and not isinstance(criteria, dict)) or (kind_q == "score" and not isinstance(criteria, list)) or not 2 <= len(criteria) <= 10:
                    raise ValueError("choice/score require 2..10 named/ordered criteria")
                if kind_q == "choice":
                    for key, value in criteria.items():
                        _text(key, 64, "choice label")
                        _text(value, 256, "choice description")
                    if any(not isinstance(gold, str) or gold not in criteria for gold in golds):
                        raise ValueError("choice oracle is outside declared criteria")
                else:
                    for value in criteria: _text(value, 256, "score level")
                    for gold in golds: _integer(gold, 0, len(criteria) - 1, "score oracle")
            else:
                raise ValueError("Jev question type must be noul, choice or score")
        if expected["decisions"] == expected["control_decisions"]:
            raise ValueError("media contrast oracle requires a gold decision different from no-media control")
        low, high = (0, 0) if kind == "audio" else (1, 1) if kind == "image" else (1, 4)
        _integer(expected["image_count"], low, high, "image_count")
    return True


def _fixture(profile, *, read=False):
    scenario = profile["scenario"]
    key = "document" if scenario == "retrieval" else "media"
    descriptor = profile["fixtures"][key]
    _keys(descriptor, {"path", "sha256"}, name="fixture descriptor")
    if not isinstance(descriptor["sha256"], str) or re.fullmatch(r"[a-fA-F0-9]{64}", descriptor["sha256"]) is None:
        raise ValueError("fixture requires a SHA-256 identity")
    path = Path(descriptor["path"])
    if not path.is_file():
        raise FileNotFoundError("required approved fixture is missing: " + str(path))
    limit = 48 * 1024 if scenario == "retrieval" else 2 * 1024 * 1024
    extensions = DOCUMENT_SUFFIXES if scenario == "retrieval" else MEDIA_SUFFIXES[profile["parameters"]["media_kind"]]
    if path.suffix.lower() not in extensions or not 0 < path.stat().st_size <= limit:
        raise ValueError(f"fixture must use an admitted extension and contain 1..{limit} bytes")
    if not read:
        return path
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != descriptor["sha256"].lower():
        raise ValueError("fixture SHA/size changed since profile approval")
    return path, data


def availability(case, context):
    profile = context.get("scenario_profile")
    if not profile:
        return False, "an explicit typed scenario profile and independent fixture oracle are required"
    validate_profile(profile)
    if case.get("id") not in {profile["scenario"], "profile::" + profile["id"]}:
        raise ValueError("profile scenario differs from selected adapter")
    command = context.get("server_command")
    if not isinstance(command, (list, tuple)) or not command:
        return False, "a prebuilt owned Server.Host binary is required"
    if context.get("model_options", {}).get("embeddings") is True:
        return False, "profile primary must be a chat/decision model, not an embedding-only host"
    if context.get("context_tokens", 2048) < 2048:
        return False, "grounded/media profile fixture needs at least 2048 context tokens"
    try:
        _fixture(profile)
    except FileNotFoundError as error:
        return False, str(error)
    if profile["scenario"] == "retrieval":
        embedding = (context.get("profile_models") or {}).get("embedding")
        if not embedding or embedding.get("options", {}).get("embeddings") is not True:
            return False, "a verified embedding catalog role configured for embeddings is required"
        if not context.get("preparing_model") and not Path(embedding["model_path"]).is_file():
            return False, "verified embedding role weights are missing"
    else:
        if profile["parameters"]["media_kind"] == "audio":
            return False, "audio remains untested: source requires an ASR callback/HTTP companion; no owned local transcription executor is configured"
        if profile["parameters"]["media_kind"] == "video" and not shutil.which("ffprobe"):
            return False, "bounded Jev video preflight requires local ffprobe on PATH; no codec download"
        if context.get("model_family") != "diffusion-gemma":
            return False, "Jev inference is source-admitted only for DiffusionGemma"
        projector = context.get("model_options", {}).get("mmproj")
        if not projector or str(projector).lower() == "none" or (not context.get("preparing_model") and not Path(projector).is_file()):
            return False, "Jev image/video requires a verified local vision shard"
    return True, "typed profile, fixture and model prerequisites present; runtime qualification pending"


def _preflight_media(profile, context, result):
    # Reuse the owned, SHA-pinned bounded decoder used by embedding profiles.
    # PNG: complete decode/CRC <=512px each side. MP4/WebM: owned ffprobe,
    # one <=512px video stream, 1..32 decoded frames and <=10seconds, <=30s
    # process deadline; missing/unknown geometry cannot reach a model load.
    from .image_profiles import _media_part
    from .image_generation import UnsupportedPngFormat
    transformed = {**profile, "parameters": {"modality": profile["parameters"]["media_kind"]}}
    try:
        _media_part(transformed, context, result)
    except UnsupportedPngFormat as error:
        result["evidence"]["unsupported_media_encoding"] = {"reason": str(error), "fixture": profile["fixtures"]["media"]}
        raise ScenarioUntested(str(error)) from error


def _json(http, route, payload=None, method=None):
    started = time.monotonic()
    offset = len(http.result["inputs"].get("requests", []))
    try:
        with http.open(route, payload, method) as response:
            return _capture_json(response, route, http.result, started)
    finally:
        # Retain fixture bytes/hash and independent oracles, not repeated multi-MB
        # base64 copies in the report, including HTTP and transport failures.
        # Do not mutate the live payload needed for deterministic replay.
        for request in http.result["inputs"].get("requests", [])[offset:]:
            if request.get("route") != "/v1/systemone" or not isinstance(request.get("body"), dict):
                continue
            captured = copy.deepcopy(request["body"])
            for field in ("files", "videos", "audios"):
                for item in captured.get(field, []):
                    if isinstance(item.get("data"), str):
                        encoded = item.pop("data")
                        item["inline_data_capture"] = {"encoded_bytes": len(encoded), "sha256_of_base64": hashlib.sha256(encoded.encode()).hexdigest()}
            request["body"] = captured


def _negative(http, route, payload, status, result, *, error_type=None, method=None):
    offset = len(result["evidence"].get("http_errors", []))
    try:
        _json(http, route, payload, method)
    except urllib.error.HTTPError as error:
        records = result["evidence"].get("http_errors", [])[offset:]
        body = _json_loads(records[-1]["body"]) if len(records) == 1 else {}
        details = body.get("error") if isinstance(body, dict) else None
        typed = (isinstance(details, dict) and details.get("type") == error_type
                 and isinstance(details.get("message"), str) and bool(details["message"])) if error_type else isinstance(details, str) and bool(details)
        _require(result, "typed negative request " + route, error.code == status and typed,
                 {"status": status, "type": error_type}, {"status": error.code, "body": body})
        return
    _require(result, "typed negative request " + route, False, status, "accepted")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _upload_document(http, cid, path, data):
    name = "profile-document" + path.suffix.lower()
    boundary = "regression-" + uuid.uuid4().hex
    raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
           'Content-Type: application/octet-stream\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
    route = "/api/knowledge/collections/" + cid + "/documents"
    with http.open(route, method="POST", headers={"Content-Type": "multipart/form-data; boundary=" + boundary}, raw_body=raw) as response:
        value = _capture_json(response, route, http.result, time.monotonic())
    docs = value.get("documents", [])
    _require(http.result, "single approved source imported", isinstance(docs, list) and len(docs) == 1)
    return docs[0], name


def _grounded(http, model, context, profile, path, data, result):
    result["stage"] = "retrieval_index"
    params, expected = profile["parameters"], profile["expected"]
    embedding = context["profile_models"]["embedding"]
    digest = _file_hash(embedding["model_path"])
    embedding_id = "gguf:sha256:" + digest + ":mean-parts-v1:" + hashlib.sha256(b"\0").hexdigest().upper()
    proof = result["evidence"]["grounded_retrieval"] = {"embedding_model_sha256": digest,
        "embedding_model_id": embedding_id, "embedding_backend_requested": "CPU", "collections": [],
        "limitations": ["CPU sidecar backend has source/config evidence, no public runtime backend telemetry.",
                        "One approved source, no reranking, OCR, remote providers, model-generated tools or tenant authorization claim.",
                        "Source formats exercise managed extraction only; image layout, macros and formula recalculation are outside this profile."]}
    _require(result, "fresh profile library", _json(http, "/api/knowledge/collections") == {"collections": []})
    for suffix in ("selected", "unrelated"):
        collection = _json(http, "/api/knowledge/collections", {"name": "profile-" + suffix}).get("collection", {})
        cid = collection.get("id", "")
        _require(result, "owned collection identity", isinstance(cid, str) and re.fullmatch(r"[a-f0-9]{32}", cid) is not None and cid not in proof["collections"])
        proof["collections"].append(cid)
    cid, unrelated = proof["collections"]
    doc, name = _upload_document(http, cid, path, data)
    did = doc.get("id", "")
    _require(result, "indexed approved document", re.fullmatch(r"[a-f0-9]{32}", did or "") is not None
        and doc.get("collectionId") == cid and doc.get("fileName") == name and doc.get("status") == "Ready"
        and type(doc.get("chunkCount")) is int and doc["chunkCount"] > 0 and doc.get("version") == 1
        and doc.get("sizeBytes") == len(data) and doc.get("contentHash") == hashlib.sha256(data).hexdigest()
        and doc.get("embeddingModelId") == embedding_id, "Ready source and exact fixture/embedding identities", doc)
    proof["document"] = doc
    route = f"/api/knowledge/collections/{cid}/documents/{did}/source?version=1"
    with http.open(route) as response:
        original = _read(response)
    _require(result, "original approved bytes preserved", original == data, hashlib.sha256(data).hexdigest(), hashlib.sha256(original).hexdigest())
    result["stage"] = "retrieval_search"
    hits = _json(http, "/api/knowledge/search", {"query": params["search_query"], "collectionIds": [cid], "limit": 8, "maxCharacters": 4096}).get("results", [])
    _require(result, "retrieved independent answer and source oracle", bool(hits)
        and any(expected["source_excerpt"] in row.get("text", "") and expected["answer_contains"] in row.get("text", "") for row in hits), expected, hits)
    for hit in hits:
        _require(result, "retrieved source scoped and finite", hit.get("collectionId") == cid and hit.get("documentId") == did
                 and hit.get("sourceUrl") == route and hit.get("status") == "Ready" and hit.get("truncated") is False
                 and type(hit.get("score")) in (int, float) and math.isfinite(hit["score"]) and hit["score"] > 0,
                 {"collection": cid, "document": did, "sourceUrl": route}, hit)
    reference = hits[0]["reference"]
    read = _json(http, "/api/knowledge/read", {"reference": reference, "collectionIds": [cid], "maxCharacters": 4096, "adjacentChunks": 0}).get("results", [])
    _require(result, "scoped read preserves retrieved passage", len(read) == 1 and read[0].get("reference") == reference and read[0].get("text") == hits[0]["text"])
    prompt = params["question"] + "\nUse only selected sources. If they do not contain the answer, say UNKNOWN. Cite the exact source URL when answering."
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 42,
               "max_tokens": params.get("max_tokens", 384), "think": False, "stream": False, "skills": [],
               "skills_discovery": False, "tool_choice": "none"}
    for scope_name, scope in (("selected", [cid]), ("empty", []), ("unrelated", [unrelated])):
        result["stage"] = "grounded_chat_" + scope_name
        reply = _json(http, "/v1/chat/completions", {**payload, "knowledgeCollections": scope})
        answer = validate_chat(reply, None, expected_model=model, max_tokens=payload["max_tokens"])
        sources = reply.get("knowledge_sources", [])
        proof.setdefault("answers", []).append({"scope": scope_name, "answer": answer, "sources": sources,
            "citation_repairs": reply.get("knowledge_citation_repairs")})
        if scope_name != "selected":
            _require(result, scope_name + " scope cannot supply the fixture secret", expected["answer_contains"].casefold() not in answer.casefold() and sources == [], "no oracle answer or sources (case-insensitive)", {"answer": answer, "sources": sources})
            continue
        _require(result, "grounded answer literal", expected["answer_contains"] in answer, expected["answer_contains"], answer)
        _require(result, "authoritative source provenance present", isinstance(sources, list) and bool(sources), "nonempty knowledge_sources", sources)
        for source in sources:
            chunk = source.get("chunkIndex")
            _require(result, "authoritative citation is the approved source", source.get("collectionId") == cid and source.get("documentId") == did
                and source.get("fileName") == name and source.get("version") == 1 and source.get("sourceUrl") == route
                and type(chunk) is int and 0 <= chunk < doc["chunkCount"] and source.get("reference") == f"kb://{cid}/{did}@1/{chunk}", route, source)
        citations = re.findall(r"(?:https?://[^\s/]+)?/api/knowledge/collections/[^\s)\]\"<>`]+", answer)
        citations = [url.rstrip(".,;") for url in citations]
        _require(result, "final answer cites only approved resolvable source URLs", bool(citations) and all(url in {route, http.base + route} for url in citations), route, citations)
        links = re.findall(r"\]\((https?://[^\s)]+)\)", answer)
        _require(result, "no invented external citation link", all(url == http.base + route for url in links), http.base + route, links)
        _require(result, "grounded request executed no host tools", reply.get("knowledge_tool_steps") == [] and reply.get("knowledge_tool_steps_truncated") is False,
                 [], reply.get("knowledge_tool_steps"))
        repairs = reply.get("knowledge_citation_repairs")
        _require(result, "citation repair accounting retained", type(repairs) is int and repairs >= 0, ">=0 integer", repairs)
    _negative(http, "/api/knowledge/read", {"reference": reference, "collectionIds": [unrelated]}, 404, result)
    _negative(http, "/v1/chat/completions", {**payload, "knowledgeCollections": "all"}, 400, result, error_type="invalid_request_error")
    for collection in proof["collections"]:
        _require(result, "owned profile collection removed", _json(http, "/api/knowledge/collections/" + collection, method="DELETE") == {"deleted": True})
    _negative(http, route, None, 404, result)
    _require(result, "profile library empty after cleanup", _json(http, "/api/knowledge/collections") == {"collections": []})
    proof["completed"] = True
    result["stage"] = "retrieval_complete"


def _media_preparation(body, kind, name, count, *, replay=False):
    diagnostics = body.get("diagnostics", {})
    attachments = diagnostics.get("attachments")
    if type(diagnostics.get("images")) is not int or diagnostics["images"] != count or not isinstance(attachments, list) or len(attachments) != 1:
        raise ValueError("Jev positive media/image preparation evidence missing")
    row = attachments[0]
    if (row.get("name") != name or row.get("kind") != kind or type(row.get("imageCount")) is not int
            or row["imageCount"] != count or type(row.get("textCharacters")) is not int or row["textCharacters"] != 0
            or row.get("sampled") is not (kind == "video") or type(row.get("cacheHit")) is not bool
            or (replay and row["cacheHit"] is not True)):
        raise ValueError("Jev media name/kind/count/sampling/cache diagnostics differ from fixture")
    if kind == "video" and (not isinstance(row.get("warning"), str) or "sampled" not in row["warning"].lower()):
        raise ValueError("video sampling coverage warning missing")
    timing = diagnostics.get("timing", {})
    for key in ("preprocessing_ms", "inference_ms", "total_ms"):
        value = timing.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid media timing " + key)
    if timing["total_ms"] + 1 < timing["preprocessing_ms"] + timing["inference_ms"]:
        raise ValueError("media timing omits preprocessing/inference")
    return {"attachments": attachments, "images": count, "timing": timing}


def _media(http, model, context, profile, path, data, result):
    result["stage"] = "jev_media_control"
    params, expected = profile["parameters"], profile["expected"]
    kind, name = params["media_kind"], "profile-media" + path.suffix.lower()
    contract = load_contract(context["repo"])
    base = {"model": model, "seed": 42, "samples": 1, "state": params["state"], "questions": params["questions"]}
    proof = result["evidence"]["jev_media_profile"] = {"kind": kind, "fixture_sha256": hashlib.sha256(data).hexdigest(),
        "limitations": ["One approved contrast fixture, not general perception quality or calibration.",
                        "Video uses at most four sampled frames; unsampled motion and soundtrack are not analyzed.",
                        "Audio recognition remains untested; only absent-ASR rejection is checked."]}
    control = _json(http, "/v1/systemone", base)
    proof["without_media"] = validate_decisions(contract, base, control, expected["control_decisions"], model)
    _require(result, "no-media control has no injected images", control.get("diagnostics", {}).get("images") == 0
             and control.get("diagnostics", {}).get("attachments") == [])
    payload = {**base, "videos" if kind == "video" else "files": [{"name": name, "data": base64.b64encode(data).decode("ascii")}]}
    previous = None
    for index in range(2):
        result["stage"] = "jev_media_replay" if index else "jev_media_inference"
        body = _json(http, "/v1/systemone", payload)
        decisions = validate_decisions(contract, payload, body, expected["decisions"], model)
        preparation = _media_preparation(body, kind, name, expected["image_count"], replay=index == 1)
        proof.setdefault("media", []).append({"decisions": decisions, "preparation": preparation})
        _require(result, "media-dependent labeled decisions and preparation", True, expected["decisions"], decisions)
        if previous is not None:
            delta = max(abs(a - b) for left, right in zip(previous, decisions) for a, b in zip(left["probabilities"], right["probabilities"]))
            _require(result, "cached media probabilities stable", delta <= 1e-5, "<=1e-5", delta)
        previous = decisions
    for changes in ({"files": [{"name": "invalid.png", "data": "%%%"}]},
                    {"images": ["file:///never-open-a-local-file.png"]},
                    {"videos": [{"name": "invalid.txt", "data": "eA=="}]}):
        _negative(http, "/v1/systemone", {**base, **changes}, 422, result, error_type="validation_error")
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(b"\0\0" * 160)
    _negative(http, "/v1/systemone", {**base, "audios": [{"name": "missing-asr.wav", "data": base64.b64encode(audio.getvalue()).decode("ascii")}]},
              503, result, error_type="model_unavailable")
    proof["completed"] = True
    result["stage"] = "jev_media_complete"


def run(case, context, result):
    """Mutate standard evidence; caller persists aggregate status/reproduction."""
    ready, reason = availability(case, context)
    if not ready:
        raise ScenarioUntested(reason)
    profile = context["scenario_profile"]
    path, data = _fixture(profile, read=True)
    if profile["scenario"] == "jev_media":
        result["stage"] = "media_fixture_preflight"
        _preflight_media(profile, context, result)
    result["inputs"]["profile_oracle"] = {"id": profile["id"], "scenario": profile["scenario"],
        "parameters": profile["parameters"], "expected": profile["expected"], "fixture": {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    media = profile["scenario"] == "jev_media"
    command = list(context["server_command"]) + _base_args(context, "cli_image" if media else "retrieval")
    command += ["--host", "127.0.0.1", "--port", str(port), "--no-webui", "--no-multi-agent"]
    env = dict(context.get("env") or {})
    # Inherited or injected ASR providers are never contacted by a profile.
    for key in tuple(env):
        if key.startswith("TS_JEV_TRANSCRIPTION") or key.startswith("TS_KNOWLEDGE_"):
            env.pop(key)
    if not media:
        env.update(host_environment({**context, "model_path": context["profile_models"]["embedding"]["model_path"]}))
    else:
        env["TS_JEV_MAX_BODY_MB"] = "8"
    process = ManagedProcess(command, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="profile-server", env=env)
    result["stage"] = "profile_server_startup"
    http = _Http(port, time.monotonic() + context.get("timeout_seconds", 600), result)
    result["evidence"]["base_url"] = http.base
    try:
        while True:
            if process.process.poll() is not None:
                raise ScenarioFailure("profile server exited before readiness")
            http.remaining()
            out, _ = process.stdout.read(); err, _ = process.stderr.read()
            if "Now listening on: " + http.base in out + err:
                break
            time.sleep(min(.1, http.remaining()))
        identity = _json(http, "/api/models")
        _require(result, "profile primary backend identity", identity.get("loadedBackend") == context["backend"], context["backend"], identity)
        _require(result, "profile primary model identity", identity.get("loaded") == Path(context["model_path"]).name,
                 Path(context["model_path"]).name, identity.get("loaded"))
        _attest_offload(context, result, "profile-startup", out + "\n" + err)
        _attest_offload_architecture(context, result, "profile-model", [identity.get("architecture")], "/api/models architecture")
        if media:
            _require(result, "Jev loaded vision companion", identity.get("visionReady") is True
                     and identity.get("loadedMmProj") == Path(context["model_options"]["mmproj"]).name,
                     Path(context["model_options"]["mmproj"]).name, identity)
        model = _hosted_response_model(identity["loaded"], _json(http, "/v1/models").get("data", []), result)
        (_media if media else _grounded)(http, model, context, profile, path, data, result)
    finally:
        process.stop()
        item = process.snapshot()
        result["processes"].append({key: value for key, value in item.items() if key not in {"stdout", "stderr"}})
        result["evidence"]["profile_server_shutdown"] = item.get("teardown")
        _require(result, "owned profile processes stopped", item.get("teardown", {}).get("verified") is True,
                 "verified empty owned process container", item.get("teardown"))
        logs = item.get("stdout", "") + "\n" + item.get("stderr", "")
        backend = backend_evidence(logs, context["backend"])
        _require(result, "no profile backend fallback", backend["fallback"] is None, "no device/backend switch", backend)
        _attest_offload(context, result, "profile-final", logs)
