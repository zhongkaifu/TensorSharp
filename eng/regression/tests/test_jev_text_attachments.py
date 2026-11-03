"""Pinned local fixture and synthetic response tests; no HTTP/model process is launched."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import jev_text_attachments as attachment
from regression import scenarios

REPO = Path(__file__).resolve().parents[3]
MODEL = "fixture-diffusiongemma.gguf"


def valid_response(case, cached=False):
    answers = {}
    for name, question in case["questions"].items():
        kind, gold = question["type"], case["expected"][name]
        if kind == "noul":
            value = {"type": kind, "noul": float(gold)}
        else:
            labels = list(question["criteria"]) if kind == "choice" else list(map(str, range(len(question["criteria"]))))
            winner = gold if kind == "choice" else str(gold)
            value = {"type": kind, "confidence": 1.0, "probabilities": {label: float(label == winner) for label in labels}}
            if kind == "choice":
                value["choice"] = winner
            else:
                value.update(score=float(gold), legend={str(i): v for i, v in enumerate(question["criteria"])})
        answers[name] = value
    ordered = sorted(case["attachments"], key=lambda item: attachment._FIELDS.index(item["field"]))
    return {"model": MODEL, "answers": answers, "usage": {"input_tokens": 80, "output_tokens": len(answers)},
            "diagnostics": {"engine": "tensorsharp", "steps": 1, "seed": 42,
                "samples": {"policy": "fixed", "n": 1}, "probability_semantics": "conditional_label_softmax_temperature_1", "images": 0,
                "attachments": [{"name": item["path"], "kind": item["kind"], "textCharacters": 100, "imageCount": 0,
                                 "sampled": False, "cacheHit": cached, "warning": None} for item in ordered],
                "timing": {"preprocessing_ms": 1.0, "inference_ms": 2.0, "total_ms": 3.0}}}


class FakeHttp:
    def __init__(self, cases, files, result, mutate=None, invalid_status=422, invalid_type="validation_error", capture=True, response_status=200):
        self.cases = {json.dumps(attachment.make_payload(case, MODEL, files), sort_keys=True): case for case in cases}
        self.result, self.mutate = result, mutate
        self.invalid_status, self.invalid_type = invalid_status, invalid_type
        self.capture, self.response_status = capture, response_status
        self.calls, self.seen = [], set()

    def json(self, route, payload):
        self.calls.append((route, copy.deepcopy(payload)))
        key = json.dumps(payload, sort_keys=True)
        if key in self.cases:
            case = self.cases[key]
            body = valid_response(case, key in self.seen)
            self.seen.add(key)
            if self.mutate:
                self.mutate(body, case, len(self.calls))
            if self.capture:
                self.result["evidence"].setdefault("responses", []).append({"route": route, "status": self.response_status, "body": body})
            return body
        if self.invalid_status == 200:
            return {"accepted": True}
        if self.capture:
            self.result["evidence"].setdefault("http_errors", []).append({"route": route, "status": self.invalid_status,
                "body": json.dumps({"error": {"type": self.invalid_type, "message": "invalid synthetic attachment"}})})
        raise urllib.error.HTTPError("http://127.0.0.1/v1/systemone", self.invalid_status, "synthetic", {}, None)


class AttachmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.files, cls.excluded, cls.manifest_hash = attachment.load_text_fixtures(REPO)

    def setUp(self):
        self.result = {"assertions": [], "inputs": {}, "evidence": {}}
        self.context = {"repo": REPO, "model_path": Path(MODEL), "artifact_dir": REPO / "artifacts/synthetic-not-created"}

    def run_fixture(self, **options):
        http = FakeHttp(self.cases, self.files, self.result, **options)
        attachment.run_text_attachments(http, MODEL, self.context, self.result)
        return http

    def test_all_fourteen_pinned_cases_replay_and_eight_typed_negatives(self):
        http = self.run_fixture()
        evidence = self.result["evidence"]["jev_text_attachments"]
        self.assertEqual(len(self.cases), 14)
        self.assertEqual(len(self.excluded), 5)
        self.assertEqual(evidence["counts"]["passed_fixtures"], 14)
        self.assertEqual(evidence["counts"]["failures"], 0)
        self.assertEqual(len(http.calls), 23)
        self.assertTrue(evidence["cache_replay"]["passed"])
        self.assertTrue(all(row["passed"] for row in self.result["assertions"]))
        self.assertEqual(evidence["manifest_sha256"], self.manifest_hash)
        self.assertTrue(all("data" not in identity for identity in evidence["files"].values()))

    def test_pinned_file_content_or_size_mismatch_fails_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / attachment.MANIFEST
            shutil.copytree((REPO / attachment.MANIFEST).parent, path.parent)
            fixture = path.parent / "ticket.txt"
            original = fixture.read_bytes()
            fixture.write_bytes(b"x" + original[1:])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                attachment.load_text_fixtures(directory)
            fixture.write_bytes(original + b"x")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                attachment.load_text_fixtures(directory)

    def test_windows_crlf_checkout_only_passes_when_canonical_pin_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / attachment.MANIFEST
            shutil.copytree((REPO / attachment.MANIFEST).parent, path.parent)
            fixture = path.parent / "ticket.txt"
            canonical = self.files["ticket.txt"]["data"]
            fixture.write_bytes(canonical.replace(b"\n", b"\r\n"))
            _, files, _, _ = attachment.load_text_fixtures(directory)
            self.assertEqual(files["ticket.txt"]["data"], canonical)
            self.assertEqual(files["ticket.txt"]["normalization"], "git_checkout_crlf_to_lf")
            self.assertNotEqual(files["ticket.txt"]["source_sha256"], files["ticket.txt"]["sha256"])
            fixture.write_bytes(fixture.read_bytes().replace(b"refund", b"rewind"))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                attachment.load_text_fixtures(directory)

    def test_manifest_cannot_hide_media_or_escape_fixture_root(self):
        manifest = json.loads((REPO / attachment.MANIFEST).read_text())
        for change in ({"path": "../ticket.txt"}, {"kind": "image"}, {"field": "audios"}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / attachment.MANIFEST
                path.parent.mkdir(parents=True)
                value = copy.deepcopy(manifest)
                value["cases"][0]["attachments"][0].update(change)
                path.write_text(json.dumps(value))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    attachment.load_text_fixtures(directory)

    def test_duplicate_manifest_keys_and_case_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            attachment._json('{"version":0,"version":1}')
        manifest = json.loads((REPO / attachment.MANIFEST).read_text())
        manifest["cases"][1]["id"] = manifest["cases"][0]["id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / attachment.MANIFEST
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unique"):
                attachment.load_text_fixtures(directory)

    def test_attachment_order_missing_text_and_media_fallback_fail(self):
        case = next(case for case in self.cases if case["id"] == "multi-tickets")
        for change in ("order", "missing", "empty", "images", "cache", "sampled"):
            body = valid_response(case)
            diagnostics = body["diagnostics"]
            if change == "order":
                diagnostics["attachments"].reverse()
            elif change == "missing":
                diagnostics["attachments"].pop()
            elif change == "empty":
                diagnostics["attachments"][0]["textCharacters"] = 0
            elif change == "images":
                diagnostics["images"] = 1
            elif change == "cache":
                diagnostics["attachments"][0]["cacheHit"] = "true"
            else:
                diagnostics["attachments"][0]["sampled"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                attachment.validate_preparation(case, body)

    def test_nonfinite_negative_or_impossible_timings_fail(self):
        for key, value in (("preprocessing_ms", float("nan")), ("inference_ms", -1), ("total_ms", 0), ("total_ms", True)):
            case = self.cases[0]
            body = valid_response(case)
            body["diagnostics"]["timing"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                attachment.validate_preparation(case, body)

    def test_wrong_attachment_decision_retains_failure_and_continues(self):
        def mutate(body, case, call):
            if call == 1:
                body["answers"]["urgent"]["noul"] = 1.0
        with self.assertRaisesRegex(ValueError, "expected label"):
            self.run_fixture(mutate=mutate)
        rows = self.result["evidence"]["jev_text_attachments"]["cases"]
        self.assertEqual(len(rows), 14)
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(all(row["passed"] for row in rows[1:]))

    def test_no_positive_cache_evidence_cannot_pass(self):
        def mutate(body, case, call):
            for item in body["diagnostics"]["attachments"]:
                item["cacheHit"] = False
        with self.assertRaisesRegex(ValueError, "no positive attachment extraction cache hit"):
            self.run_fixture(mutate=mutate)

    def test_cache_hit_without_stable_probabilities_fails(self):
        def mutate(body, case, call):
            if call == 15:
                body["answers"]["urgent"]["noul"] = 0.1
        with self.assertRaisesRegex(ValueError, "probabilities changed"):
            self.run_fixture(mutate=mutate)

    def test_http_success_status_and_current_response_capture_required(self):
        for options in ({"response_status": 201}, {"capture": False}):
            self.result = {"assertions": [], "inputs": {}, "evidence": {}}
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "current HTTP 200"):
                self.run_fixture(**options)

    def test_negative_cases_require_exact_422_validation_error(self):
        for options in ({"invalid_status": 200}, {"invalid_status": 400}, {"invalid_status": 503}, {"invalid_type": "invalid_request_error"}):
            self.result = {"assertions": [], "inputs": {}, "evidence": {}}
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_fixture(**options)
            self.assertTrue(all(not row["passed"] for row in self.result["evidence"]["jev_text_attachments"]["invalid_requests"]))

    def test_negative_body_has_no_ambient_user_file_or_remote_url(self):
        invalid = dict(attachment.invalid_payloads(MODEL))
        self.assertEqual(len(invalid), 8)
        self.assertEqual(invalid["path-traversal"]["files"], [{"file": "../ticket.txt"}])
        self.assertNotIn("http", json.dumps(invalid))

    def test_adapter_requires_server_and_manifest_not_cli(self):
        context = {"repo": REPO, "backend": "ggml_cpu", "preparing_model": True, "model_path": MODEL,
                   "server_command": ["not-executed"], "model_options": {}}
        case = {"id": "jev_text_attachments"}
        self.assertTrue(scenarios.availability(case, context)[0])
        context.pop("server_command")
        context["cli_command"] = ["not-executed"]
        self.assertFalse(scenarios.availability(case, context)[0])


if __name__ == "__main__":
    unittest.main()
