"""Synthetic approved oracles, fake HTTP and contained-process receipts only."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import wave
import zlib

from eng.regression import image_profiles as module
from eng.regression.image_generation import png_metadata
from eng.regression.scenarios import ScenarioFailure


def png(size=32, changed=False):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    rows = b"".join(b"\0" + bytes(v for x in range(size) for v in ((x+int(changed)) % 256, y % 256, 40, 255)) for y in range(size))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "image/png"}


class FakeHttp:
    def __init__(self, result, identity, content=None):
        self.base = "http://127.0.0.1:12345"
        self.result, self.identity = result, identity
        self.content = content or png()
        self.reject = True
        self.response_change = None
        self.payloads = []
        self.result_value = {"ok": True, "width": 32, "height": 32, "elapsedSeconds": .1, "url": "/uploads/image.png"}

    def remaining(self):
        return 10

    def json(self, route, payload=None):
        self.payloads.append((route, payload))
        if route == "/api/models":
            return self.identity
        if route == "/v1/models":
            return {"data": [{"id": "model"}]}
        if route == "/v1/embeddings":
            bad = isinstance(payload["input"], dict)
            if bad and self.reject:
                self.error(route, {"type": "invalid_request_error", "param": "input", "message": "invalid media"})
            value = {"object": "list", "model": "model", "usage": {"prompt_tokens": 2, "total_tokens": 2},
                     "data": [{"index": i, "object": "embedding", "embedding": [1., 0.]} for i in range(2)]}
        else:
            bad = payload["width"] % 32 or payload["steps"] < 0 or not payload["prompt"] or payload.get("imagePaths") == [] or payload.get("maskMode") == "invalid-regression-mode" or payload.get("maskPath") == "bad.png"
            if bad and self.reject:
                self.error(route, "invalid image parameters")
            value = dict(self.result_value)
        if self.response_change:
            self.response_change(value)
        return value

    def error(self, route, detail):
        self.result["evidence"].setdefault("http_errors", []).append({"route": route, "body": json.dumps({"error": detail})})
        raise urllib.error.HTTPError(self.base + route, 400, "Bad Request", {}, None)

    def open(self, route):
        return Response(self.content)


class ImageProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = {"assertions": [], "evidence": {}, "inputs": {}, "processes": []}
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"not a real model")
        self.profile = {"id": "image", "scenario": "image_generation", "preset": "qwen-image-2.1", "backend": "ggml_cuda",
                        "parameters": {"prompt": "red cube", "width": 32, "height": 32, "steps": 2, "cfg": 1, "seed": 42},
                        "expected": {"pixel_sha256": png_metadata(png(), 32, 32)["pixel_sha256"]}, "fixtures": {}}
        self.context = {"scenario_profile": self.profile, "repo": Path(__file__).resolve().parents[3], "artifact_dir": self.root,
                        "backend": "ggml_cuda", "model_path": self.model, "server_command": ["synthetic-never-launched"],
                        "model_options": {key: str(self.model) for key in ("qwen-image-vae", "qwen-image-vl", "qwen-image-mmproj", "mmproj")}}
        self.identity = {"loaded": "model.gguf", "loadedBackend": "ggml_cuda", "architecture": "qwen_image"}

    def fixture(self, key, data, suffix=".png"):
        target = self.root / (key + suffix)
        target.write_bytes(data)
        self.profile["fixtures"][key] = {"path": str(target), "sha256": hashlib.sha256(data).hexdigest()}
        return target

    def embedding(self, modality="image"):
        self.profile.update(scenario="embedding_media", parameters={"modality": modality, "text": "synthetic", "dimensions": 2}, expected={})
        self.fixture("media", png())
        oracle = {"schema_version": 1, "model": "model.gguf", "projector": "model.gguf", "modality": modality,
                  "media_sha256": self.profile["fixtures"]["media"]["sha256"], "text": "synthetic", "dimensions": 2,
                  "vector": [1., 0.], "provenance": "synthetic independent test fixture"}
        self.fixture("oracle", json.dumps(oracle).encode(), ".json")
        self.identity.update(architecture="gemma-embedding2", loadedMmProj="model.gguf")
        return oracle

    def editing(self):
        self.profile["scenario"] = "image_editing"
        self.profile["expected"]["min_changed_mean_abs_diff"] = 1
        self.fixture("input", png())
        self.fixture("mask", png())  # Decoder is mocked for lifecycle tests; not a real binary mask.

    def test_schema_accepts_bounded_generation(self):
        module.validate_profile(self.profile)

    def test_schema_rejects_unbounded_or_unknown_parameters(self):
        for key, value in (("width", 1024), ("steps", 51), ("seed", -1), ("cfg", float("nan")), ("command", "anything"), ("prompt", ""), ("height", True)):
            with self.subTest(key=key):
                value_profile = copy.deepcopy(self.profile)
                value_profile["parameters"][key] = value
                with self.assertRaises(ValueError):
                    module.validate_profile(value_profile)

    def test_generation_requires_supplied_pixel_oracle(self):
        self.profile["expected"] = {}
        with self.assertRaises(ValueError):
            module.validate_profile(self.profile)

    def test_embedding_gates_cannot_be_weakened(self):
        self.embedding()
        for expected in ({"max_relative_l2": .01}, {"min_cosine": .99}, {"min_cosine": float("nan")}, {"other": 1}):
            self.profile["expected"] = expected
            with self.assertRaises(ValueError):
                module.validate_profile(self.profile)

    def test_mask_profile_requires_both_input_and_positive_change_bound(self):
        self.editing()
        module.validate_profile(self.profile)
        self.profile["expected"]["min_changed_mean_abs_diff"] = 0
        with self.assertRaises(ValueError):
            module.validate_profile(self.profile)

    def test_missing_profile_and_companion_are_unavailable(self):
        self.assertFalse(module.availability({"id": "image_generation"}, {})[0])
        self.context["model_options"].pop("qwen-image-vae")
        self.assertFalse(module.availability({"id": "image_generation"}, self.context)[0])

    def test_only_consumed_companions_and_roles_are_accepted(self):
        self.assertEqual({"model", "qwen-image-vae", "qwen-image-vl"}, module.required_options(self.profile))
        self.editing()
        self.assertIn("qwen-image-mmproj", module.required_options(self.profile))
        self.profile["models"] = {"unused": "huge-model"}
        with self.assertRaisesRegex(ValueError, "not consumed"):
            module.validate_profile(self.profile)

    def test_turbo_schedule_cannot_silently_use_base_recipe(self):
        self.context["model_options"]["qwen-image-variant"] = "turbo"
        self.assertFalse(module.availability({"id": "image_generation"}, self.context)[0])
        self.profile["parameters"]["steps"] = 8
        self.assertTrue(module.availability({"id": "image_generation"}, self.context)[0])

    def test_changed_fixture_and_oversize_refused(self):
        target = self.fixture("input", png())
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            module._fixture_bytes(self.profile, "input")
        target.write_bytes(b"12345")
        with self.assertRaisesRegex(ValueError, "budget"):
            module._fixture_bytes(self.profile, "input", 4)

    def test_png_geometry_cannot_exceed_budget(self):
        with self.assertRaises(ValueError):
            module._image_input(png(32), 64, 64)
        data = bytearray(png())
        struct.pack_into(">I", data, 16, 4096)
        with self.assertRaisesRegex(ValueError, "geometry"):
            module._image_input(bytes(data))

    def test_generation_positive_and_typed_negatives(self):
        http = FakeHttp(self.result, self.identity)
        module._images(http, self.identity, self.profile, self.context, self.result, None)
        self.assertEqual(len(self.result["evidence"]["http_errors"]), 3)
        self.assertTrue((self.root / "profile-output.png").is_file())
        self.assertTrue(all(x["passed"] for x in self.result["assertions"]))

    def test_wrong_pixels_fail_independent_oracle(self):
        http = FakeHttp(self.result, self.identity, png(changed=True))
        with self.assertRaisesRegex(ScenarioFailure, "oracle"):
            module._images(http, self.identity, self.profile, self.context, self.result, None)

    def test_accepted_bad_request_does_not_pass(self):
        http = FakeHttp(self.result, self.identity)
        http.reject = False
        with self.assertRaisesRegex(ScenarioFailure, "rejected"):
            module._images(http, self.identity, self.profile, self.context, self.result, None)

    def test_cross_origin_output_is_refused(self):
        http = FakeHttp(self.result, self.identity)
        http.result_value["url"] = "https://example.org/private.png"
        with self.assertRaisesRegex(ValueError, "origin"):
            module._images(http, self.identity, self.profile, self.context, self.result, None)

    def test_media_embedding_positive_reference_and_negatives(self):
        self.embedding()
        part = module._media_part(self.profile, self.context, self.result)
        http = FakeHttp(self.result, self.identity)
        module._embeddings(http, self.identity, self.profile, self.context, self.result, part)
        self.assertEqual(2, len(self.result["evidence"]["embedding_reference_comparisons"]))
        self.assertEqual(3, len(self.result["evidence"]["http_errors"]))

    def test_media_embedding_wrong_oracle_identity_fails(self):
        oracle = self.embedding()
        oracle["model"] = "wrong.gguf"
        self.fixture("oracle", json.dumps(oracle).encode(), ".json")
        with self.assertRaisesRegex(ScenarioFailure, "oracle model"):
            module._embeddings(FakeHttp(self.result, self.identity), self.identity, self.profile, self.context, self.result, {})

    def test_nonfinite_wrong_dimensions_and_wrong_reference_fail(self):
        self.embedding()
        for vector in ([float("nan"), 0], [1], [0, 1], [0, 0]):
            http = FakeHttp(self.result, self.identity)
            http.response_change = lambda value, vector=vector: value["data"][0].update(embedding=vector)
            with self.subTest(vector=vector), self.assertRaises((ValueError, ScenarioFailure)):
                module._embeddings(http, self.identity, self.profile, self.context, self.result, {})

    def test_audio_fixture_budget_and_complete_payload(self):
        self.embedding("audio")
        data = io.BytesIO()
        with wave.open(data, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0" * 800)
        self.fixture("media", data.getvalue(), ".wav")
        self.assertEqual("input_audio", module._media_part(self.profile, self.context, self.result)["type"])
        self.fixture("media", data.getvalue()[:-10], ".wav")
        with self.assertRaisesRegex(ValueError, "complete PCM"):
            module._media_part(self.profile, self.context, self.result)

    def test_video_probe_timeout_or_cleanup_refused_before_host(self):
        self.embedding("video")
        self.fixture("media", b"synthetic video", ".mp4")
        for change in ({"timed_out": True}, {"teardown": {"verified": False}}, {"exit_code": 1}):
            receipt = {"timed_out": False, "exit_code": 0, "output_truncated": False, "teardown": {"verified": True}, **change}
            with patch.object(module, "run_process", return_value=receipt), self.assertRaisesRegex(ValueError, "video probe"):
                module._media_part(self.profile, self.context, self.result)

    def test_video_probe_bounded_positive_and_geometry_negative(self):
        self.embedding("video")
        self.fixture("media", b"synthetic video", ".mp4")
        data = {"streams": [{"codec_type": "video", "width": 32, "height": 32, "nb_read_frames": "2"}], "format": {"duration": "1"}}
        def receipt(*args, **kwargs):
            return {"stdout": json.dumps(data), "timed_out": False, "exit_code": 0, "output_truncated": False, "teardown": {"verified": True}}
        with patch.object(module, "run_process", side_effect=receipt):
            self.assertEqual("input_video", module._media_part(self.profile, self.context, self.result)["type"])
            data["streams"][0]["width"] = 4096
            with self.assertRaises(ValueError):
                module._media_part(self.profile, self.context, self.result)

    def lifecycle(self, *, fallback=False, cleanup=True, wrong_backend=False, timeout=False):
        result, identity = self.result, dict(self.identity)
        if wrong_backend:
            identity["loadedBackend"] = "cpu"
        http = FakeHttp(result, identity)
        if timeout:
            http.remaining = lambda: (_ for _ in ()).throw(ScenarioFailure("deadline exceeded"))
        instances = []
        class Process:
            def __init__(self, command, **kwargs):
                port = command[command.index("--port")+1]
                http.base = "http://127.0.0.1:" + port
                self.process = self
                self.stdout = self
                self.stderr = self
                self.stopped = False
                instances.append(self)
            def poll(self):
                return None
            def read(self):
                return "Now listening on: " + http.base, False
            def stop(self):
                self.stopped = True
            def snapshot(self):
                return {"teardown": {"attempted": True, "verified": cleanup}, "stdout": "falling back to cpu" if fallback else "Backend: ggml_cuda", "stderr": ""}
        with patch.object(module, "ManagedProcess", Process), patch("eng.regression.scenarios._Http", return_value=http):
            try:
                module.run({"id": self.profile["scenario"]}, self.context, result)
            finally:
                self.assertTrue(instances[0].stopped)
                self.assertIn("profile_server_shutdown", result["evidence"])

    def test_owned_host_positive_cleanup_and_actual_identity(self):
        self.lifecycle()
        self.assertTrue(all(x["passed"] for x in self.result["assertions"]))

    def test_fallback_never_counts_as_requested_backend_pass(self):
        with self.assertRaisesRegex(ScenarioFailure, "fallback"):
            self.lifecycle(fallback=True)

    def test_wrong_backend_still_cleans_host(self):
        with self.assertRaisesRegex(ScenarioFailure, "actual backend"):
            self.lifecycle(wrong_backend=True)

    def test_deadline_still_cleans_host(self):
        with self.assertRaisesRegex(ScenarioFailure, "deadline"):
            self.lifecycle(timeout=True)

    def test_cleanup_failure_carries_unsafe_receipt(self):
        with self.assertRaisesRegex(ScenarioFailure, "stopped"):
            self.lifecycle(cleanup=False)
        self.assertFalse(self.result["processes"][-1]["teardown"]["verified"])

    def test_editing_positive_noop_negatives_and_fidelity_dispatch(self):
        self.editing()
        http = FakeHttp(self.result, self.identity)
        with patch.object(module, "_mask_data", return_value=b"mask"), patch.object(module, "_solid_mask", return_value=b"solid"), \
             patch.object(module, "_upload_image", side_effect=["input.png", "mask.png", "bad.png", "noop.png"]), patch.object(module, "_fidelity") as fidelity:
            module._images(http, self.identity, self.profile, self.context, self.result, png())
        self.assertEqual(5, len(self.result["evidence"]["http_errors"]))
        self.assertEqual(2, len(self.result["evidence"]["generated_images"]))
        fidelity.assert_called_once()

    def test_fidelity_protected_pixels_cannot_change(self):
        self.editing()
        output = self.root / "profile-output.png"
        output.write_bytes(png())
        record = {"png": str(output), "protected_pixels_changed": 1, "changed_mean_abs_diff": 8}
        receipt = {"exit_code": 0, "timed_out": False, "output_truncated": False, "teardown": {"verified": True}}
        def execute(*args, **kwargs):
            (self.root / "image-fidelity.json").write_text(json.dumps({"edits": [record]}))
            return receipt
        with patch.object(module, "run_process", side_effect=execute), self.assertRaisesRegex(ScenarioFailure, "protected"):
            module._fidelity(self.profile, self.context, self.result, output)

    def test_fidelity_positive_and_selected_change_negative(self):
        self.editing()
        output = self.root / "profile-output.png"
        output.write_bytes(png())
        record = {"png": str(output), "protected_pixels_changed": 0, "changed_mean_abs_diff": 8}
        def execute(*args, **kwargs):
            (self.root / "image-fidelity.json").write_text(json.dumps({"edits": [record]}))
            return {"exit_code": 0, "timed_out": False, "output_truncated": False, "teardown": {"verified": True}}
        with patch.object(module, "run_process", side_effect=execute):
            module._fidelity(self.profile, self.context, self.result, output)
            record["changed_mean_abs_diff"] = 0
            with self.assertRaisesRegex(ScenarioFailure, "change bound"):
                module._fidelity(self.profile, self.context, self.result, output)

    def test_stale_fidelity_report_never_passes(self):
        self.editing()
        target = self.root / "image-fidelity.json"
        target.write_text(json.dumps({"edits": [{"png": "old", "protected_pixels_changed": 0}]}))
        receipt = {"exit_code": 0, "timed_out": False, "output_truncated": False, "teardown": {"verified": True}}
        with patch.object(module, "run_process", return_value=receipt), self.assertRaises(FileNotFoundError):
            module._fidelity(self.profile, self.context, self.result, self.root / "profile-output.png")
        self.assertFalse(target.exists())

    def test_fidelity_timeout_and_unverified_cleanup_fail(self):
        self.editing()
        for change in ({"timed_out": True}, {"teardown": {"verified": False}}):
            receipt = {"exit_code": 0, "timed_out": False, "output_truncated": False, "teardown": {"verified": True}, **change}
            with patch.object(module, "run_process", return_value=receipt), self.assertRaisesRegex(ScenarioFailure, "contained"):
                module._fidelity(self.profile, self.context, self.result, self.root / "profile-output.png")

    def test_report_does_not_persist_media_base64(self):
        self.result["inputs"]["requests"] = [{"body": {"input": [{"content": [{"image_url": {"url": "data:image/png;base64,cHJpdmF0ZQ=="}}]}]}}]
        module._redact_media_requests(self.result)
        self.assertNotIn("cHJpdmF0ZQ", json.dumps(self.result))
        self.assertIn("encoded_sha256", json.dumps(self.result))

    def test_upload_uses_source_defined_file_field_and_hash_receipt(self):
        http = FakeHttp(self.result, self.identity)
        class Opener:
            def open(self, request, timeout):
                return Response(json.dumps({"ok": True, "mediaType": "image", "file": "owned-reference.png", "url": "/uploads/owned-reference.png"}).encode())
        http.opener = Opener()
        self.assertEqual("owned-reference.png", module._upload_image(http, self.result, png()))
        self.assertEqual(hashlib.sha256(png()).hexdigest(), self.result["inputs"]["uploads"][0]["sha256"])

    def test_upload_never_accepts_external_or_traversal_reference(self):
        http = FakeHttp(self.result, self.identity)
        class Opener:
            def open(self, request, timeout):
                return Response(json.dumps({"ok": True, "mediaType": "image", "file": "../private.png"}).encode())
        http.opener = Opener()
        with self.assertRaisesRegex(ScenarioFailure, "confined"):
            module._upload_image(http, self.result, png())


if __name__ == "__main__":
    unittest.main()
