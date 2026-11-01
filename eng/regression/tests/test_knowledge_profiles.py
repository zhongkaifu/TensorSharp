"""Profile contract and protocol faults, without real models, GPUs or builds."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import Mock, patch

from eng.regression import knowledge_profiles as profiles
from eng.regression.scenarios import ScenarioFailure, ScenarioUntested
from eng.regression.tests.test_knowledge_retrieval import FakeKnowledge, Response
from eng.regression.tests.test_jev import response as jev_response
from eng.regression.tests.test_image_generation import png

REPO = Path(__file__).resolve().parents[3]


class GroundedHttp(FakeKnowledge):
    base = "http://127.0.0.1:40000"

    def __init__(self, result, digest, profile, model, fault=None):
        super().__init__(result, digest)
        self.profile, self.model, self.chat_fault, self.source = profile, model, fault, None

    def open(self, route, payload=None, method=None, headers=None, *, raw_body=None):
        self.result["inputs"].setdefault("requests", []).append({"route": route, "body": payload})
        if route == "/v1/chat/completions":
            if self.chat_fault == "timeout": raise TimeoutError("synthetic inference deadline")
            scope = payload.get("knowledgeCollections")
            if not isinstance(scope, list):
                self.result["evidence"].setdefault("http_errors", []).append({"route": route, "status": 400,
                    "body": json.dumps({"error": {"type": "invalid_request_error", "message": "invalid scope"}})})
                raise urllib.error.HTTPError(route, 400, "fixture", {}, None)
            docs = [doc for doc in self.documents.values() if doc["collectionId"] in scope]
            if docs:
                row = self.row(docs[0]); self.source = row["sourceUrl"]
                row.pop("score"); row.pop("text")
                answer = self.profile["expected"]["answer_contains"] + " [Source](" + row["sourceUrl"] + ")"
                sources = [row]
                if self.chat_fault == "wrong_answer": answer = "UNKNOWN " + row["sourceUrl"]
                if self.chat_fault == "no_source": sources = []
                if self.chat_fault == "wrong_source": sources[0]["documentId"] = "f" * 32
                if self.chat_fault == "invented_citation": answer = answer.replace("/documents/", "/documents/invented")
                if self.chat_fault == "external_link": answer += " [Other](https://example.invalid/never-fetch)"
            else:
                answer, sources = "UNKNOWN", []
                if self.chat_fault == "scope_leak": answer = self.profile["expected"]["answer_contains"]
                if self.chat_fault == "scope_leak_case": answer = self.profile["expected"]["answer_contains"].lower()
            return Response({"object": "chat.completion", "model": self.model if self.chat_fault != "wrong_model" else "other.gguf",
                "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                "knowledge_sources": sources, "knowledge_citation_repairs": 0,
                "knowledge_tool_steps": [{"tool": "exec"}] if self.chat_fault == "tools" else [], "knowledge_tool_steps_truncated": False})
        if route.endswith("/search") and self.chat_fault == "bad_excerpt":
            result = super().open(route, payload, method, headers, raw_body=raw_body)
            body = json.loads(result.read())
            body["results"][0]["text"] = "wrong source text"
            return Response(body)
        return super().open(route, payload, method, headers, raw_body=raw_body)


class JevHttp:
    base = "http://127.0.0.1:40000"

    def __init__(self, result, profile, model, fault=None):
        self.result, self.profile, self.model, self.fault = result, profile, model, fault
        self.media_calls = 0

    def open(self, route, payload=None, method=None, headers=None):
        self.result["inputs"].setdefault("requests", []).append({"route": route, "body": payload})
        if self.fault == "timeout": raise TimeoutError("synthetic media deadline")
        media = payload.get("files") or payload.get("videos")
        invalid = payload.get("images") or (media and (media[0].get("data") == "%%%" or media[0]["name"].endswith(".txt")))
        if invalid or payload.get("audios"):
            code, kind = (422, "validation_error") if invalid else (503, "model_unavailable")
            if self.fault == "accept_invalid": return Response({"accepted": True})
            if self.fault == "wrong_error": code = 400
            self.result["evidence"].setdefault("http_errors", []).append({"route": route, "status": code,
                "body": json.dumps({"error": {"type": kind, "message": "synthetic typed rejection"}})})
            raise urllib.error.HTTPError(route, code, "fixture", {}, None)
        gold = self.profile["expected"]["decisions" if media else "control_decisions"]
        body = jev_response(payload["questions"], gold)
        body["model"] = self.model
        body["diagnostics"].update(images=0, attachments=[])
        if media:
            self.media_calls += 1
            kind = self.profile["parameters"]["media_kind"]
            count = self.profile["expected"]["image_count"]
            body["diagnostics"].update(images=count, attachments=[{"name": media[0]["name"], "kind": kind,
                "textCharacters": 0, "imageCount": count, "sampled": kind == "video", "cacheHit": self.media_calls > 1,
                "warning": "Video uses sampled frames." if kind == "video" else None}],
                timing={"preprocessing_ms": 1, "inference_ms": 2, "total_ms": 3})
            if self.fault == "no_images": body["diagnostics"]["images"] = 0
            if self.fault == "wrong_kind": body["diagnostics"]["attachments"][0]["kind"] = "text"
            if self.fault == "not_cached": body["diagnostics"]["attachments"][0]["cacheHit"] = False
            if self.fault == "nonfinite": body["answers"]["signal"]["probabilities"]["green"] = float("nan")
            if self.fault == "wrong_gold":
                body = jev_response(payload["questions"], self.profile["expected"]["control_decisions"])
                body["model"] = self.model
            if self.fault == "changed_replay" and self.media_calls == 2:
                body["answers"]["signal"].update(probabilities={"green": .9, "red": .1}, confidence=.9)
        return Response(body)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "chat.gguf"; self.model.write_bytes(b"synthetic chat")
        self.embedding = self.root / "encoder.gguf"; self.embedding.write_bytes(b"synthetic encoder")
        self.document = self.root / "source.txt"
        self.document.write_bytes(b"The LANTERN-STATION maintenance code is COBALT-7319.\n")
        self.media = self.root / "signal.png"; self.media.write_bytes(png(size=16))
        self.projector = self.root / "vision.safetensors"; self.projector.write_bytes(b"synthetic vision")
        self.profile = {"schema_version": 1, "id": "fixture", "scenario": "retrieval", "preset": "chat", "backend": "cpu",
            "models": {"embedding": "embedding-minilm"},
            "fixtures": {"document": {"path": str(self.document), "sha256": hashlib.sha256(self.document.read_bytes()).hexdigest()}},
            "parameters": {"embedding_backend": "cpu", "question": "What is the LANTERN-STATION maintenance code?", "search_query": "LANTERN-STATION maintenance code"},
            "expected": {"answer_contains": "COBALT-7319", "source_excerpt": "The LANTERN-STATION maintenance code is COBALT-7319."},
            "resources": {"ram_required_gib": 8, "vram_required_gib": 0}}
        self.context = {"repo": REPO, "artifact_dir": self.root / "out", "model_path": self.model, "backend": "cpu",
            "server_command": ["not-executed"], "model_options": {}, "model_family": "qwen35", "scenario_profile": self.profile,
            "profile_models": {"embedding": {"preset": "embedding-minilm", "model_path": self.embedding, "options": {"embeddings": True}}}}
        self.result = {"assertions": [], "evidence": {}, "inputs": {}, "processes": []}

    def tearDown(self): self.temp.cleanup()

    def jev_profile(self, kind="image"):
        if kind == "video":
            path = self.media.with_suffix(".mp4"); path.write_bytes(b"synthetic video")
        elif kind == "audio":
            path = self.media.with_suffix(".wav"); path.write_bytes(b"synthetic audio")
        else: path = self.media
        profile = copy.deepcopy(self.profile)
        profile.update(scenario="jev_media", models={}, fixtures={"media": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
            parameters={"media_kind": kind, "state": "A vehicle is approaching the pictured signal.",
                        "questions": {"signal": {"type": "choice", "instructions": "Which lamp is lit?", "criteria": {"green": "green lamp", "red": "red lamp"}}}},
            expected={"decisions": {"signal": "green"}, "control_decisions": {"signal": "red"}, "image_count": 0 if kind == "audio" else 2 if kind == "video" else 1})
        return profile, path

    def grounded(self, fault=None):
        http = GroundedHttp(self.result, hashlib.sha256(self.embedding.read_bytes()).hexdigest(), self.profile, self.model.name, fault)
        profiles._grounded(http, self.model.name, self.context, self.profile, self.document, self.document.read_bytes(), self.result)
        return http

    def test_grounded_positive_provenance_scope_and_cleanup(self):
        http = self.grounded()
        proof = self.result["evidence"]["grounded_retrieval"]
        self.assertTrue(proof["completed"])
        self.assertEqual(["selected", "empty", "unrelated"], [row["scope"] for row in proof["answers"]])
        self.assertEqual([], list(http.collections))
        self.assertEqual(3, len(self.result["evidence"]["http_errors"]))

    def test_grounded_wrong_oracle_citations_scope_model_tools_and_timeout_fail(self):
        for fault in ("wrong_answer", "bad_excerpt", "no_source", "wrong_source", "invented_citation", "external_link", "scope_leak", "scope_leak_case", "wrong_model", "tools", "timeout"):
            with self.subTest(fault=fault):
                self.result = {"assertions": [], "evidence": {}, "inputs": {}, "processes": []}
                with self.assertRaises((ScenarioFailure, ValueError, TimeoutError)):
                    self.grounded(fault)

    def test_typed_parameters_unknown_keys_and_prompt_leak_refused(self):
        for change in ({"command": "never execute"}, {"embedding_backend": "cuda"}, {"max_tokens": True}, {"question": "COBALT-7319"}):
            profile = copy.deepcopy(self.profile); profile["parameters"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): profiles.validate_profile(profile)
        self.assertEqual(set(), profiles.required_options(self.profile))

    def test_missing_fixture_is_untested_changed_sha_is_failure_before_launch(self):
        self.document.unlink()
        ready, reason = profiles.availability({"id": "retrieval"}, self.context)
        self.assertFalse(ready); self.assertIn("missing", reason)
        with self.assertRaises(ScenarioUntested): profiles.run({"id": "retrieval"}, self.context, self.result)
        self.document.write_bytes(b"different source")
        with patch.object(profiles, "ManagedProcess") as launch, self.assertRaisesRegex(ValueError, "SHA"):
            profiles.run({"id": "retrieval"}, self.context, self.result)
        launch.assert_not_called()

    def test_missing_embedding_role_and_chat_primary_mode(self):
        for change in ({"profile_models": {}}, {"model_options": {"embeddings": True}}):
            with self.subTest(change=change):
                self.assertFalse(profiles.availability({"id": "retrieval"}, {**self.context, **change})[0])

    def test_document_fixture_caps_and_companion_role_hash_identity(self):
        self.document.write_bytes(b"x" * (48 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "49152"): profiles.availability({"id": "retrieval"}, self.context)

    def test_image_and_video_contrast_preparation_replay_and_negatives(self):
        for kind in ("image", "video"):
            with self.subTest(kind=kind):
                profile, path = self.jev_profile(kind)
                result = {"assertions": [], "evidence": {}, "inputs": {}, "processes": []}
                http = JevHttp(result, profile, self.model.name)
                profiles._media(http, self.model.name, self.context, profile, path, path.read_bytes(), result)
                self.assertTrue(result["evidence"]["jev_media_profile"]["completed"])
                self.assertEqual(4, len(result["evidence"]["http_errors"]))
                self.assertEqual(2, http.media_calls)
                self.assertNotIn("data", result["inputs"]["requests"][1]["body"]["videos" if kind == "video" else "files"][0])
                self.assertEqual({"mmproj"}, profiles.required_options(profile))

    def test_jev_wrong_semantics_preparation_replay_nonfinite_and_errors_fail(self):
        profile, path = self.jev_profile()
        for fault in ("no_images", "wrong_kind", "not_cached", "nonfinite", "wrong_gold", "changed_replay", "accept_invalid", "wrong_error", "timeout"):
            with self.subTest(fault=fault):
                result = {"assertions": [], "evidence": {}, "inputs": {}, "processes": []}
                http = JevHttp(result, profile, self.model.name, fault)
                with self.assertRaises((ScenarioFailure, ValueError, TimeoutError)):
                    profiles._media(http, self.model.name, self.context, profile, path, path.read_bytes(), result)

    def test_media_payload_redacted_even_when_open_fails_and_live_payload_preserved(self):
        for error in (TimeoutError("synthetic transport timeout"),
                      urllib.error.HTTPError("http://127.0.0.1/v1/systemone", 503, "synthetic", {}, None)):
            with self.subTest(error=type(error).__name__):
                result = {"assertions": [], "evidence": {}, "inputs": {"requests": [{"route": "/earlier", "body": {"fixture": "keep"}}]}}
                http = Mock(result=result)
                payload = {"files": [{"name": "private.png", "data": "cHJpdmF0ZS1maXh0dXJl"}]}
                original = copy.deepcopy(payload)
                def failed_open(route, body, method):
                    result["inputs"]["requests"].append({"route": route, "body": body})
                    raise error
                http.open.side_effect = failed_open
                with self.assertRaises(type(error)):
                    profiles._json(http, "/v1/systemone", payload)
                self.assertEqual(original, payload)
                captured = result["inputs"]["requests"][-1]["body"]["files"][0]
                self.assertNotIn("data", captured)
                self.assertEqual(hashlib.sha256(original["files"][0]["data"].encode()).hexdigest(), captured["inline_data_capture"]["sha256_of_base64"])
                self.assertEqual({"fixture": "keep"}, result["inputs"]["requests"][0]["body"])

    def test_jev_missing_projector_audio_and_wrong_family_are_untested(self):
        profile, path = self.jev_profile()
        context = {**self.context, "scenario_profile": profile, "model_family": "diffusion-gemma"}
        self.assertFalse(profiles.availability({"id": "jev_media"}, context)[0])
        context["model_options"] = {"mmproj": str(self.projector)}
        self.assertTrue(profiles.availability({"id": "jev_media"}, context)[0])
        self.assertFalse(profiles.availability({"id": "jev_media"}, {**context, "model_family": "qwen35"})[0])
        audio, _ = self.jev_profile("audio")
        ready, why = profiles.availability({"id": "jev_media"}, {**context, "scenario_profile": audio})
        self.assertFalse(ready); self.assertIn("ASR", why)

    def test_jev_oracles_and_question_schema_are_strict(self):
        profile, _ = self.jev_profile()
        for transform in (lambda p: p["parameters"].update(endpoint="https://example.invalid"),
                          lambda p: p["expected"].update(control_decisions=p["expected"]["decisions"]),
                          lambda p: p["expected"].update(image_count=True),
                          lambda p: p["parameters"]["questions"]["signal"].update(depends_on="other"),
                          lambda p: p["expected"]["decisions"].update(signal="unknown")):
            bad = copy.deepcopy(profile); transform(bad)
            with self.assertRaises(ValueError): profiles.validate_profile(bad)

    def test_large_compressed_image_rejected_before_host_load(self):
        import struct
        data = bytearray(self.media.read_bytes())
        struct.pack_into(">II", data, 16, 4096, 4096)
        self.media.write_bytes(data)
        profile, _ = self.jev_profile()
        context = {**self.context, "scenario_profile": profile, "model_family": "diffusion-gemma", "model_options": {"mmproj": str(self.projector)}}
        with patch.object(profiles, "ManagedProcess") as launch, self.assertRaisesRegex(ValueError, "512"):
            profiles.run({"id": "jev_media"}, context, self.result)
        launch.assert_not_called()

    def test_unsupported_png_encoding_is_untested_before_host_load(self):
        for options in ({"depth": 16}, {"interlace": 1}):
            with self.subTest(options=options):
                self.media.write_bytes(png(size=16, **options))
                profile, _ = self.jev_profile()
                context = {**self.context, "scenario_profile": profile, "model_family": "diffusion-gemma", "model_options": {"mmproj": str(self.projector)}}
                with patch.object(profiles, "ManagedProcess") as launch, self.assertRaises(ScenarioUntested):
                    profiles.run({"id": "jev_media"}, context, self.result)
                launch.assert_not_called()
                self.assertIn("unsupported_media_encoding", self.result["evidence"])

    def test_video_geometry_duration_count_timeout_and_cleanup_are_bounded_before_host(self):
        profile, path = self.jev_profile("video")
        context = {**self.context, "scenario_profile": profile, "model_family": "diffusion-gemma", "model_options": {"mmproj": str(self.projector)}}
        metadata = {"streams": [{"codec_type": "video", "width": 320, "height": 240, "nb_read_frames": "16"}], "format": {"duration": "2"}}
        receipt = {"exit_code": 0, "timed_out": False, "output_truncated": False,
                   "teardown": {"verified": True}, "stdout": json.dumps(metadata), "stderr": ""}
        with patch("eng.regression.image_profiles.run_process", return_value=receipt) as probe:
            profiles._preflight_media(profile, context, self.result)
        self.assertLessEqual(probe.call_args.kwargs["timeout_seconds"], 30)
        self.assertIn("file,pipe", probe.call_args.args[0])
        faults = [{"width": 4096}, {"nb_read_frames": "1000000"}, {"duration": "600"}, {"duration": "nan"}]
        receipts = []
        for fault in faults:
            bad = copy.deepcopy(metadata)
            (bad["format"] if "duration" in fault else bad["streams"][0]).update(fault)
            receipts.append({**receipt, "stdout": json.dumps(bad)})
        receipts += [{**receipt, "timed_out": True}, {**receipt, "teardown": {"verified": False}}]
        for failed in receipts:
            with self.subTest(failed=failed), patch("eng.regression.image_profiles.run_process", return_value=failed), \
                 patch.object(profiles.shutil, "which", return_value="synthetic-ffprobe"), patch.object(profiles, "ManagedProcess") as launch:
                with self.assertRaises(ValueError): profiles.run({"id": "jev_media"}, context, self.result)
                launch.assert_not_called()
        with patch.object(profiles.shutil, "which", return_value=None):
            ready, why = profiles.availability({"id": "jev_media"}, context)
            self.assertFalse(ready); self.assertIn("ffprobe", why)

    def owned_run(self, *, fault=None, media=False):
        context = {**self.context, "artifact_dir": self.root / ("run-" + str(fault) + str(media))}
        if media:
            profile, _ = self.jev_profile()
            context.update(scenario_profile=profile, model_family="diffusion-gemma", model_options={"mmproj": str(self.projector)})
        context["env"] = {"TS_JEV_TRANSCRIPTION_URL": "https://never-contact.invalid", "TS_KNOWLEDGE_DIRECTORY": "never-read"}
        process = Mock()
        process.process.poll.return_value = 1 if fault == "early_exit" else None
        process.stdout.read.return_value = ("Now listening on: http://127.0.0.1:40000", False)
        process.stderr.read.return_value = ("", False)
        process.snapshot.return_value = {"stdout": "Backend: Cpu", "stderr": "falling back to CPU" if fault == "fallback" else "",
            "stage": "profile-server", "teardown": {"attempted": True, "verified": fault != "cleanup"}}
        http = Mock(base="http://127.0.0.1:40000")
        if fault == "timeout": http.remaining.side_effect = ScenarioFailure("synthetic deadline")
        identity = {"loaded": self.model.name, "loadedBackend": "ggml_cpu" if fault == "backend" else "cpu",
                    "architecture": "diffusiongemma" if media else "qwen35", "visionReady": fault != "vision",
                    "loadedMmProj": self.projector.name}
        with patch.object(profiles, "ManagedProcess", return_value=process) as launch, patch.object(profiles, "_Http", return_value=http), \
             patch.object(profiles, "_json", side_effect=[identity, {"data": [{"id": self.model.stem}]}]), \
             patch.object(profiles, "_grounded", side_effect=ValueError("body failed") if fault == "body" else None) as grounded, \
             patch.object(profiles, "_media") as media_helper:
            try:
                profiles.run({"id": "jev_media" if media else "retrieval"}, context, self.result)
            finally:
                process.stop.assert_called_once()
                self.assertNotIn("TS_JEV_TRANSCRIPTION_URL", launch.call_args.kwargs["env"])
                self.assertNotEqual("never-read", launch.call_args.kwargs["env"].get("TS_KNOWLEDGE_DIRECTORY"))
        return launch.call_args, grounded, media_helper

    def test_owned_host_identity_no_tools_and_same_verified_embedding_file(self):
        call, grounded, _ = self.owned_run()
        grounded.assert_called_once()
        self.assertEqual(str(self.embedding.resolve()), call.kwargs["env"]["TS_KNOWLEDGE_EMBEDDING_MODEL"])
        self.assertEqual("CPU", call.kwargs["env"]["TS_KNOWLEDGE_EMBEDDING_BACKEND"])
        self.assertIn("--no-skills", call.args[0]); self.assertIn("--no-multi-agent", call.args[0])
        self.assertNotIn("--embeddings", call.args[0])

    def test_owned_media_host_keeps_verified_projector(self):
        call, _, media = self.owned_run(media=True)
        media.assert_called_once()
        self.assertEqual(str(self.projector), call.args[0][call.args[0].index("--mmproj") + 1])

    def test_failed_body_timeout_backend_fallback_and_teardown_cannot_pass(self):
        for fault in ("body", "timeout", "early_exit", "backend", "fallback", "cleanup"):
            with self.subTest(fault=fault), self.assertRaises((ScenarioFailure, ValueError)):
                self.owned_run(fault=fault)
        with self.assertRaises(ScenarioFailure): self.owned_run(fault="vision", media=True)


if __name__ == "__main__": unittest.main()
