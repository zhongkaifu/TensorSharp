"""Synthetic protocol fixtures only; no model/backend, downloads or engine build."""
import copy
from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import Mock, patch

from eng.regression import knowledge_retrieval as retrieval
from eng.regression.scenarios import _Http, _server, availability, ScenarioFailure


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, headers=None):
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.headers = headers or {"Content-Type": "application/json"}


class FakeKnowledge:
    def __init__(self, result, digest, fault=None):
        self.result, self.digest, self.fault = result, digest, fault
        self.collections, self.documents, self.bytes, self.calls = {}, {}, {}, []
        self.count, self.deleted = 0, None

    def error(self, route, status):
        self.result["evidence"].setdefault("http_errors", []).append({"route": route, "status": status,
            "body": json.dumps({"error": "Synthetic protocol rejection"})})
        raise urllib.error.HTTPError(route, status, "fixture rejection", {}, None)

    def row(self, doc):
        return {"reference": f"kb://{doc['collectionId']}/{doc['id']}@1/0", "collectionId": doc["collectionId"],
            "documentId": doc["id"], "fileName": doc["fileName"], "version": 1, "chunkIndex": 0,
            "text": self.bytes[doc["id"]].decode().strip(), "score": .02, "status": "Ready", "truncated": False,
            "modality": "text", "sourceUrl": f"/api/knowledge/collections/{doc['collectionId']}/documents/{doc['id']}/source?version=1"}

    def open(self, route, payload=None, method=None, headers=None, *, raw_body=None):
        method = method or ("POST" if payload is not None or raw_body is not None else "GET")
        self.calls.append((route, method, payload, raw_body))
        parts = route.split("/")
        if route == "/api/knowledge/collections":
            if method == "POST":
                self.count += 1
                cid = f"{self.count:032x}"
                self.collections[cid] = {"id": cid, "name": payload["name"]}
                return Response({"collection": self.collections[cid]})
            return Response({"collections": list(self.collections.values())})
        if route.endswith("/search") or route.endswith("/read"):
            scope = payload.get("collectionIds")
            if scope is None or any(not re.fullmatch("[a-f0-9]{32}", value) for value in scope):
                return self.error(route, 400)
            if any(cid not in self.collections for cid in scope):
                return self.error(route, 404)
            if len(payload.get("query", "")) > 4000:
                if self.fault == "accept_query":
                    return Response({"results": []})
                return self.error(route, 413)
            if route.endswith("/read"):
                did = payload["reference"].split("/")[3].split("@")[0]
                doc = self.documents.get(did)
                if doc is None or doc["collectionId"] not in scope and self.fault != "read_leak":
                    return self.error(route, 404)
                row = self.row(doc)
                row["score"] = 0
                return Response({"results": [row]})
            docs = [d for d in self.documents.values() if d["collectionId"] in scope]
            docs.sort(key=lambda d: d["fileName"], reverse=self.fault == "wrong_rank")
            rows = [self.row(doc) for doc in docs]
            if self.fault == "lexical_only" and payload["query"] == retrieval.QUERY:
                rows = []
            if self.fault == "search_leak" and len(docs) == 2:
                rows.append(self.row(list(self.documents.values())[-1]))
            if self.fault == "deleted_search" and self.deleted:
                rows.insert(0, self.row(self.deleted))
            if self.fault == "nonfinite_score" and rows:
                rows[0]["score"] = float("nan")
            return Response({"results": rows})
        if len(parts) >= 5:
            cid = parts[4]
            if len(parts) == 5 and method == "DELETE":
                existed = self.collections.pop(cid, None) is not None
                self.documents = {did: doc for did, doc in self.documents.items() if doc["collectionId"] != cid}
                return Response({"deleted": existed})
            if cid not in self.collections:
                return self.error(route, 404)
            if len(parts) == 6:
                if method == "POST":
                    envelope = BytesParser(policy=default).parsebytes(("Content-Type: " + headers["Content-Type"] + "\r\n\r\n").encode() + raw_body)
                    files = list(envelope.iter_parts())
                    if len(files) != 1:
                        if self.fault != "partial_upload":
                            return self.error(route, 400)
                        self.documents["f" * 32] = {"id": "f" * 32, "collectionId": cid}
                        return self.error(route, 400)
                    self.count += 1
                    did = f"{self.count:032x}"
                    data, name = files[0].get_payload(decode=True), files[0].get_filename()
                    doc = {"id": did, "collectionId": cid, "fileName": name, "contentHash": hashlib.sha256(data).hexdigest(),
                        "sizeBytes": len(data), "version": 1, "chunkCount": 1, "status": "Ready",
                        "embeddingModelId": "gguf:sha256:" + self.digest + ":mean-parts-v1:" + hashlib.sha256(b"\0").hexdigest().upper(),
                        "extractorId": "tensorsharp-managed-documents-v1"}
                    if self.fault == "wrong_hash": doc["contentHash"] = "0" * 64
                    if self.fault == "wrong_encoder": doc["embeddingModelId"] = "different-file"
                    if self.fault == "metadata_only": doc["status"] = "MetadataOnly"
                    self.documents[did], self.bytes[did] = doc, data
                    return Response({"documents": [doc]})
                docs = [doc for doc in self.documents.values() if doc["collectionId"] == cid]
                return Response({"documents": docs, "total": len(docs), "nextOffset": None})
            did = parts[6]
            if method == "DELETE":
                doc = self.documents.pop(did, None)
                if doc: self.deleted = doc
                return Response({"deleted": doc is not None})
            if did not in self.documents:
                return self.error(route, 404)
            data = self.bytes[did]
            if self.fault == "source_corrupt": data += b"wrong"
            headers = {"Content-Type": "application/octet-stream", "Content-Disposition": "attachment; filename=m01.txt",
                       "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
            if self.fault == "source_inline": headers["Content-Disposition"] = "inline"
            return Response(data, headers)
        raise AssertionError((route, method))


class RetrievalContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "all-MiniLM-L6-v2-Q8_0.gguf"
        self.model.write_bytes(b"synthetic fixture, not model weights")
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.context = {"backend": "cpu", "preset_id": "embedding-minilm", "model_options": {"embeddings": True},
                        "model_path": self.model, "artifact_dir": self.root / "artifacts", "server_command": ["not-executed"]}
        self.result = {"inputs": {}, "evidence": {}, "assertions": [], "processes": []}

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, fault=None):
        service = FakeKnowledge(self.result, self.digest, fault)
        with patch.object(retrieval, "PINNED_MODEL_SHA256", self.digest):
            retrieval.run_retrieval(service, self.model.name, self.context, self.result)
        return service

    def test_complete_fixture_lifecycle_and_no_chat(self):
        service = self.execute()
        self.assertTrue(self.result["evidence"]["retrieval"]["completed"])
        self.assertFalse(service.collections)
        self.assertFalse(service.documents)
        self.assertEqual(8, len(self.result["evidence"]["http_errors"]))
        self.assertTrue(all(route.startswith("/api/knowledge/") for route, *_ in service.calls))
        self.assertEqual([], self.result["evidence"]["retrieval"]["dense_query_lexical_overlap"])
        self.assertTrue(all(item["passed"] for item in self.result["assertions"]))

    def test_wrong_metadata_rank_scope_source_and_deletion_fail(self):
        for fault in ("wrong_hash", "wrong_encoder", "metadata_only", "wrong_rank", "lexical_only", "search_leak",
                      "read_leak", "source_corrupt", "source_inline", "deleted_search", "partial_upload", "accept_query", "nonfinite_score"):
            with self.subTest(fault=fault):
                self.result = {"inputs": {}, "evidence": {}, "assertions": [], "processes": []}
                with self.assertRaises((ScenarioFailure, ValueError)):
                    self.execute(fault)

    def test_unpinned_model_fails_before_http(self):
        http = Mock()
        with self.assertRaisesRegex(ScenarioFailure, "pinned MiniLM"):
            retrieval.run_retrieval(http, self.model.name, self.context, self.result)
        http.open.assert_not_called()

    def test_preset_backend_and_embedding_mode_admission(self):
        self.assertTrue(availability({"id": "retrieval_vector_cpu"}, self.context)[0])
        for changes in ({"backend": "ggml_cuda"}, {"backend": "ggml_cpu"}, {"preset_id": "embedding-nomic"},
                        {"model_options": {"embeddings": False}}):
            with self.subTest(changes=changes):
                self.assertFalse(availability({"id": "retrieval_vector_cpu"}, {**self.context, **changes})[0])

    def test_environment_same_file_and_fresh_owned_store(self):
        env = retrieval.host_environment(self.context)
        self.assertEqual(str(self.model.resolve()), env["TS_KNOWLEDGE_EMBEDDING_MODEL"])
        self.assertEqual("CPU", env["TS_KNOWLEDGE_EMBEDDING_BACKEND"])
        self.assertEqual("none", env["TS_KNOWLEDGE_PDF_OCR"])
        self.assertEqual("0", env["TS_KNOWLEDGE_RERANK"])
        self.assertTrue(Path(env["TS_KNOWLEDGE_DIRECTORY"]).resolve().is_relative_to(
            Path(self.context["artifact_dir"]).resolve()))
        with self.assertRaisesRegex(ValueError, "fresh"):
            retrieval.host_environment(self.context)

    def test_multipart_and_response_bounds(self):
        for files in ([("../private.txt", b"x")], [("m.txt", b"x" * retrieval.MAX_BODY)], []):
            with self.assertRaises(ValueError): retrieval.multipart(files)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            retrieval._read(Response(b"x" * (retrieval.MAX_BODY + 1)))
        http = _Http(1, time.monotonic() + 1, self.result)
        with patch.object(http.opener, "open") as opened:
            for body in ("text", b"x" * 65537):
                with self.assertRaises(ScenarioFailure): http.open("/api/knowledge/x", raw_body=body)
            with self.assertRaises(ScenarioFailure): http.open("/api/knowledge/x", {}, raw_body=b"x")
            with self.assertRaises(ScenarioFailure): http.open("//external/x", raw_body=b"x")
            opened.assert_not_called()

    def test_server_integration_requires_embedding_cpu_identity_and_sidecar_env(self):
        for architecture in ("bert", "qwen35"):
            with self.subTest(architecture=architecture):
                artifact = self.root / architecture
                context = {**self.context, "artifact_dir": artifact, "repo": self.root}
                process = Mock()
                process.process.poll.return_value = None
                process.stdout.read.return_value = ("Now listening on: http://127.0.0.1:40000", False)
                process.stderr.read.return_value = ("", False)
                process.snapshot.return_value = {"stdout": "", "stderr": "", "teardown": {"verified": True}}
                http = Mock(base="http://127.0.0.1:40000")
                http.json.side_effect = [{"loaded": self.model.name, "loadedBackend": "cpu", "architecture": architecture,
                                         "embeddingDimensions": 384, "capabilities": ["embedding"]},
                                        {"data": [{"id": self.model.stem}]}]
                with patch("eng.regression.scenarios.ManagedProcess", return_value=process) as launch, \
                     patch("eng.regression.scenarios._Http", return_value=http), \
                     patch("eng.regression.knowledge_retrieval.run_retrieval") as execute:
                    if architecture == "bert":
                        _server({"id": "retrieval_vector_cpu"}, context, self.result)
                        execute.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ScenarioFailure, "MiniLM embedding-only"):
                            _server({"id": "retrieval_vector_cpu"}, context, self.result)
                        execute.assert_not_called()
                self.assertIn("--embeddings", launch.call_args.args[0])
                self.assertEqual(str(self.model.resolve()), launch.call_args.kwargs["env"]["TS_KNOWLEDGE_EMBEDDING_MODEL"])
                self.assertEqual("CPU", launch.call_args.kwargs["env"]["TS_KNOWLEDGE_EMBEDDING_BACKEND"])

    def test_loopback_multipart_transport_and_redirect_refusal(self):
        captured, redirect = [], [False]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, self.headers["Content-Type"], self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(302 if redirect[0] else 200)
                self.send_header("Content-Type", "application/json")
                if redirect[0]: self.send_header("Location", "http://127.0.0.1:1/outside")
                self.end_headers()
                self.wfile.write(b'{}')
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            http = _Http(server.server_address[1], time.monotonic() + 5, self.result)
            retrieval._upload(http, "/api/knowledge/upload", [retrieval.FIXTURES[0]], self.result)
            self.assertIn(retrieval.FIXTURES[0][1], captured[0][2])
            self.assertIn("multipart/form-data; boundary=", captured[0][1])
            self.assertEqual(hashlib.sha256(captured[0][2]).hexdigest(), self.result["inputs"]["requests"][0]["body"]["sha256"])
            redirect[0] = True
            with self.assertRaises(urllib.error.HTTPError) as error:
                retrieval._upload(http, "/api/knowledge/upload", [retrieval.FIXTURES[0]], self.result)
            self.assertEqual(302, error.exception.code)
            self.assertEqual(2, len(captured))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
