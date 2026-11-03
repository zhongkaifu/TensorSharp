"""Bounded CPU retrieval through an owned embedding-only host, using synthetic TXT.

The primary encoder and KnowledgeStore encoder independently load the same pinned
MiniLM file. Collection scope is retrieval scope, not an authentication boundary.
No chat, remote provider, OCR, generated tools, or user documents are involved.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import time
import urllib.error
import uuid

PINNED_MODEL_SHA256 = "263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed"
MAX_BODY = 64 * 1024
FIXTURES = (
    ("m01.txt", b"A car's engine oil must be changed every eight thousand kilometers.\n"),
    ("m02.txt", b"Garden flowers grow best in moist soil with regular watering and sunshine.\n"),
    ("m03.txt", b"The synthetic foreign collection access code is ORBIT_COBALT_7319.\n"),
)
QUERY = "automobile lubricant maintenance"
LIMITATIONS = ["Grounded chat and answer citations remain untested.",
              "Reranking, OCR, media, persistence across restart, and concurrent mutation remain untested.",
              "Collection IDs are retrieval scope; this is not a tenant authorization test.",
              "The primary CPU backend is API-attested; the independent sidecar CPU path is explicitly configured and source-admitted, not measured by backend telemetry.",
              "One dense paraphrase ranking fixture is not a retrieval-quality benchmark."]


def availability(context):
    if context.get("preset_id") != "embedding-minilm":
        return False, "retrieval_vector_cpu is restricted to the pinned embedding-minilm preset"
    if context.get("backend") != "cpu":
        return False, "retrieval_vector_cpu requires the explicit CPU backend for both encoder instances; no fallback"
    if context.get("model_options", {}).get("embeddings") is not True:
        return False, "retrieval_vector_cpu requires embedding-only host configuration"
    return True, "bounded MiniLM CPU retrieval prerequisites present"


def host_environment(context):
    """Never open or reuse a pre-existing knowledge library, even a suite artifact."""
    root = Path(context["artifact_dir"]).resolve() / "knowledge-store"
    if root.exists():
        raise ValueError("retrieval requires a fresh scenario-local knowledge-store directory")
    root.mkdir(parents=True)
    return {"TS_KNOWLEDGE_DIRECTORY": str(root),
            "TS_KNOWLEDGE_EMBEDDING_MODEL": str(Path(context["model_path"]).resolve()),
            "TS_KNOWLEDGE_EMBEDDING_BACKEND": "CPU",
            "TS_KNOWLEDGE_QUERY_PREFIX": "", "TS_KNOWLEDGE_DOCUMENT_PREFIX": "",
            "TS_KNOWLEDGE_PDF_OCR": "none", "TS_KNOWLEDGE_RERANK": "0"}


def multipart(files):
    """Only small, ASCII bare filenames and supplied synthetic bytes are accepted."""
    boundary = "regression-" + uuid.uuid4().hex
    body = bytearray()
    if not 1 <= len(files) <= 2:
        raise ValueError("fixture multipart must contain one file or the two-file negative fixture")
    for name, data in files:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+\.txt", name) or not isinstance(data, bytes):
            raise ValueError("invalid synthetic multipart fixture")
        body.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                     'Content-Type: text/plain\r\n\r\n').encode("ascii"))
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))
    if len(body) > MAX_BODY:
        raise ValueError("fixture multipart exceeds 64 KiB")
    return bytes(body), "multipart/form-data; boundary=" + boundary


def _read(response):
    data = response.read(MAX_BODY + 1)
    if len(data) > MAX_BODY:
        raise ValueError("knowledge response exceeds 64 KiB capture budget")
    return data


def _upload(http, route, files, result):
    data, mime = multipart(files)
    started = time.monotonic()
    with http.open(route, method="POST", headers={"Content-Type": mime}, raw_body=data) as response:
        return _capture_json(response, route, result, started)


def _capture_json(response, route, result, started):
    from .scenarios import _json_loads
    raw = _read(response)
    record = {"route": route, "status": response.status, "duration_seconds": round(time.monotonic() - started, 4)}
    result["evidence"].setdefault("responses", []).append(record)
    try:
        record["body"] = _json_loads(raw)
    except ValueError:
        record["invalid_json"] = raw.decode("utf-8", "replace")
        raise
    return record["body"]


def run_retrieval(http, model, context, result):
    from .scenarios import _json_loads, _require
    def check(name, condition, expected=None, actual=None):
        _require(result, name, condition, expected, actual)

    ready, why = availability(context)
    check("retrieval CPU/preset admission", ready, True, why)
    path = Path(context["model_path"])
    check("bounded local encoder file", 0 < path.stat().st_size <= 128 * 1024 * 1024)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    check("pinned MiniLM encoder weights", digest == PINNED_MODEL_SHA256, PINNED_MODEL_SHA256, digest)
    embedding_id = "gguf:sha256:" + digest + ":mean-parts-v1:" + hashlib.sha256(b"\0").hexdigest().upper()
    proof = result["evidence"]["retrieval"] = {
        "model_sha256": digest, "embedding_model_id": embedding_id, "resident_encoder_instances": 2,
        "shared_weight_file": str(path.resolve()), "sidecar_backend_requested": "CPU",
        "sidecar_backend_evidence": "explicit environment; managed CPU encoder source path; no public sidecar backend telemetry",
        "knowledge_directory": str(Path(context["artifact_dir"]).resolve() / "knowledge-store"),
        "created_collection_ids": [], "limitations": LIMITATIONS,
        "score_semantics": "reciprocal rank fusion/diversification score, not raw cosine"}
    result["inputs"]["retrieval_fixtures"] = [{"filename": name, "text": data.decode("ascii"),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in FIXTURES]
    result["inputs"]["dense_query"] = QUERY

    # Use a smaller budget than the generic host JSON capture for every knowledge
    # response, including errors handled by _Http's bounded common path.
    def json_request(route, payload=None, method=None):
        started = time.monotonic()
        with http.open(route, payload, method) as response:
            return _capture_json(response, route, result, started)

    def negative(name, status, operation):
        before = len(result["evidence"].get("http_errors", []))
        try:
            returned = operation()
            if hasattr(returned, "close"):
                returned.close()
        except urllib.error.HTTPError as error:
            records = result["evidence"].get("http_errors", [])[before:]
            body = _json_loads(records[-1]["body"]) if records else {}
            check(name, error.code == status and isinstance(body, dict) and isinstance(body.get("error"), str)
                  and bool(body["error"]), {"status": status, "error": "nonempty string"}, {"status": error.code, "body": body})
        else:
            check(name, False, status, "request unexpectedly accepted")

    base = "/api/knowledge"
    check("fresh knowledge library is empty", json_request(base + "/collections") == {"collections": []})
    collections = []
    for suffix in ("local", "foreign"):
        name = "regression-vector-" + suffix
        row = json_request(base + "/collections", {"name": name, "description": "Original synthetic regression fixtures only."}).get("collection", {})
        cid = row.get("id", "")
        check("created collection identity " + suffix, isinstance(cid, str) and re.fullmatch(r"[a-f0-9]{32}", cid) is not None
              and cid not in collections and row.get("name") == name, name, row)
        collections.append(cid)
        proof["created_collection_ids"].append(cid)
    listed = json_request(base + "/collections").get("collections", [])
    check("only owned collections listed", len(listed) == 2 and {r.get("id") for r in listed} == set(collections))
    local, foreign = collections
    documents = []
    for index, (name, data) in enumerate(FIXTURES):
        cid = local if index < 2 else foreign
        docs = _upload(http, base + f"/collections/{cid}/documents", [(name, data)], result).get("documents", [])
        check("one import result " + name, isinstance(docs, list) and len(docs) == 1)
        doc = docs[0]
        check("document identity " + name, isinstance(doc.get("id"), str) and re.fullmatch(r"[a-f0-9]{32}", doc["id"]) is not None
              and doc["id"] not in {d["id"] for d in documents} and doc.get("collectionId") == cid and doc.get("fileName") == name)
        expected = {"contentHash": hashlib.sha256(data).hexdigest(), "sizeBytes": len(data), "version": 1,
                    "chunkCount": 1, "status": "Ready", "embeddingModelId": embedding_id,
                    "extractorId": "tensorsharp-managed-documents-v1"}
        actual = {key: doc.get(key) for key in expected}
        check("indexed metadata and encoder identity " + name, actual == expected
              and all(type(actual[key]) is int for key in ("sizeBytes", "version", "chunkCount")), expected, actual)
        documents.append(doc)
    proof["documents"] = documents

    def listed_documents(cid, ids):
        value = json_request(base + f"/collections/{cid}/documents")
        check("document list exactly matches owned state", value.get("total") == len(ids) and value.get("nextOffset") is None
              and len(value.get("documents", [])) == len(ids) and {d.get("id") for d in value["documents"]} == set(ids), ids, value)
    listed_documents(local, [d["id"] for d in documents[:2]])
    listed_documents(foreign, [documents[2]["id"]])

    def search(query, scope):
        value = json_request(base + "/search", {"query": query, "collectionIds": scope, "limit": 3, "maxCharacters": 4096})
        rows = value.get("results")
        check("bounded search result envelope", isinstance(rows, list) and len(rows) <= 3, "list <= 3", rows)
        for row in rows:
            doc = next((d for d in documents if d["id"] == row.get("documentId")), None)
            check("search result belongs to requested fixture scope", doc is not None and doc["collectionId"] in scope
                  and row.get("collectionId") == doc["collectionId"], scope, row)
            check("search score finite and positive", type(row.get("score")) in (float, int)
                  and math.isfinite(row["score"]) and row["score"] > 0, ">0 finite RRF", row.get("score"))
            validate_row(row, doc)
        return rows

    def validate_row(row, doc):
        expected_text = next(data.decode("ascii").strip() for name, data in FIXTURES if name == doc["fileName"])
        expected = {"reference": f"kb://{doc['collectionId']}/{doc['id']}@1/0", "collectionId": doc["collectionId"],
                    "documentId": doc["id"], "fileName": doc["fileName"], "version": 1, "chunkIndex": 0,
                    "text": expected_text, "status": "Ready", "truncated": False, "modality": "text",
                    "sourceUrl": base + f"/collections/{doc['collectionId']}/documents/{doc['id']}/source?version=1"}
        actual = {key: row.get(key) for key in expected}
        check("exact chunk/source identity and content", actual == expected, expected, actual)

    query_tokens = set(re.findall(r"[a-z0-9]+", QUERY.lower()))
    indexed_tokens = set(re.findall(r"[a-z0-9]+", " ".join(name + " " + data.decode("ascii") for name, data in FIXTURES).lower()))
    proof["dense_query_lexical_overlap"] = sorted(query_tokens & indexed_tokens)
    check("dense fixture has no lexical token match including filenames", not proof["dense_query_lexical_overlap"])
    hits = search(QUERY, [local])
    check("dense paraphrase ranks maintenance document first", bool(hits) and hits[0]["documentId"] == documents[0]["id"], documents[0]["id"], hits)
    target = hits[0]
    read = json_request(base + "/read", {"reference": target["reference"], "collectionIds": [local], "maxCharacters": 4096, "adjacentChunks": 0}).get("results", [])
    check("scoped read returns one chunk", len(read) == 1)
    validate_row(read[0], documents[0])
    check("read score is source-defined zero", type(read[0].get("score")) in (float, int) and read[0]["score"] == 0)
    for doc, (_, data) in zip(documents, FIXTURES):
        route = base + f"/collections/{doc['collectionId']}/documents/{doc['id']}/source?version=1"
        with http.open(route) as response:
            content, headers = _read(response), response.headers
            actual_source = {"route": route, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                             "headers": dict(headers)}
            proof.setdefault("source_downloads", []).append(actual_source)
            check("source defensive download headers", headers.get("Content-Type", "").split(";")[0] == "application/octet-stream"
                  and headers.get("X-Content-Type-Options") == "nosniff" and "no-store" in headers.get("Cache-Control", "")
                  and headers.get("Content-Disposition", "").lower().startswith("attachment"),
                  "application/octet-stream; attachment; nosniff; no-store", dict(headers))
        check("source bytes and content hash " + doc["fileName"], content == data and actual_source["sha256"] == doc["contentHash"],
              {"bytes": len(data), "sha256": doc["contentHash"]}, actual_source)
    check("empty collection scope returns no results", search(QUERY, []) == [])
    foreign_hits = search("ORBIT_COBALT_7319", [foreign])
    check("foreign collection positive control", bool(foreign_hits) and foreign_hits[0]["documentId"] == documents[2]["id"])
    search("ORBIT_COBALT_7319", [local])  # Every returned row must belong to the selected local fixtures.
    negative("wrong-scope reference rejected", 404, lambda: json_request(base + "/read", {"reference": target["reference"], "collectionIds": [foreign]}))
    negative("missing collection scope rejected", 400, lambda: json_request(base + "/search", {"query": QUERY}))
    negative("invalid collection ID rejected", 400, lambda: json_request(base + "/search", {"query": QUERY, "collectionIds": ["invalid"]}))
    negative("overlong query rejected", 413, lambda: json_request(base + "/search", {"query": "x" * 4001, "collectionIds": [local]}))
    negative("two-file multipart rejected atomically", 400, lambda: _upload(http, base + f"/collections/{local}/documents",
        [("rejected1.txt", b"Synthetic first rejected upload."), ("rejected2.txt", b"Synthetic second rejected upload.")], result))
    listed_documents(local, [d["id"] for d in documents[:2]])
    target_route = base + f"/collections/{local}/documents/{documents[0]['id']}"
    check("document deletion succeeds", json_request(target_route, method="DELETE") == {"deleted": True})
    check("document repeated deletion is false", json_request(target_route, method="DELETE") == {"deleted": False})
    listed_documents(local, [documents[1]["id"]])
    remaining = search(QUERY, [local])
    check("deleted document removed from search", all(row["documentId"] != documents[0]["id"] for row in remaining))
    negative("deleted source is unavailable", 404, lambda: http.open(target["sourceUrl"]))
    negative("deleted reference is unavailable", 404, lambda: json_request(base + "/read", {"reference": target["reference"], "collectionIds": [local]}))
    for cid in collections:
        route = base + "/collections/" + cid
        check("collection deletion succeeds", json_request(route, method="DELETE") == {"deleted": True})
        check("collection repeated deletion is false", json_request(route, method="DELETE") == {"deleted": False})
    check("owned collections all removed", json_request(base + "/collections") == {"collections": []})
    negative("deleted collection scope is unavailable", 404, lambda: search(QUERY, [local]))
    proof["completed"] = True
