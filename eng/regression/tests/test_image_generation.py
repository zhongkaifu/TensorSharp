"""Synthetic PNG/HTTP fixtures; no model, GPU, network or image libraries."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import urllib.error
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import image_generation as image
from regression.scenarios import ScenarioFailure


def chunk(kind, content):
    return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff)


def png(size=128, *, constant=False, transparent=False, changed=False, filters=(0,), raw=None, compressed=None, header=None):
    if raw is None:
        raw, previous = bytearray(), bytes(size * 4)
        for y in range(size):
            row = bytes(value for x in range(size) for value in (
                70 if constant else (x * 3 + y + int(changed)) % 256,
                50 if constant else (y * 7 + x) % 256,
                20, 0 if transparent else 255))
            kind = filters[y % len(filters)]
            raw.append(kind)
            for i, value in enumerate(row):
                a, b, c = (row[i - 4] if i >= 4 else 0), previous[i], (previous[i - 4] if i >= 4 else 0)
                predictor = 0 if kind == 0 else a if kind == 1 else b if kind == 2 else (a + b) // 2 if kind == 3 else image._paeth(a, b, c) if kind == 4 else 0
                raw.append((value - predictor) & 255)
            previous = row
    head = header or struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    payload = zlib.compress(bytes(raw)) if compressed is None else compressed
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", payload) + chunk(b"IEND", b"")


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "image/png"}


class FakeHttp:
    base = "http://127.0.0.1:4567"

    def __init__(self, result):
        self.result, self.requests, self.downloads, self.generated = result, [], [], 0
        self.url, self.mime, self.reject_invalid = "/uploads/generated.png", "image/png", True
        self.content, self.repeat_changes = png(), False

    def json(self, route, payload):
        self.requests.append(payload)
        error = None
        if not isinstance(payload["width"], int):
            error = "Invalid image-generation parameters: width"
        elif payload["width"] % 32:
            error = "width and height must both be zero (automatic) or positive multiples of 32."
        elif not isinstance(payload["prompt"], str):
            error = "prompt must be a string."
        elif not payload["prompt"]:
            error = "Text-to-image generation requires a nonempty prompt."
        if error and self.reject_invalid:
            self.result["evidence"].setdefault("http_errors", []).append({"route": route, "status": 400, "body": json.dumps({"error": error})})
            raise urllib.error.HTTPError(self.base + route, 400, "bad request", {}, None)
        self.generated += 1
        return {"ok": True, "url": self.url, "width": 128, "height": 128, "elapsedSeconds": 0.01}

    def open(self, route):
        self.downloads.append(route)
        response = Response(png(changed=True) if self.repeat_changes and self.generated > 1 else self.content)
        response.headers = {"Content-Type": self.mime}
        return response


class PngTests(unittest.TestCase):
    def test_all_filters_decode_identically(self):
        values = [image.png_metadata(png(8, filters=(kind,)), 8, 8) for kind in range(5)]
        self.assertEqual(len({value["pixel_sha256"] for value in values}), 1)
        self.assertTrue(all(value["spatially_varied"] and value["has_visible_pixels"] for value in values))

    def test_corrupt_crc_truncation_and_trailing_bytes_rejected(self):
        valid = png(8)
        corrupt = bytearray(valid)
        corrupt[20] ^= 1
        for data in (valid[:-1], valid[:-12], valid + b"trailing", bytes(corrupt), valid[:27]):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                image.png_metadata(data, 8, 8)

    def test_constant_transparent_and_wrong_size_or_format_rejected(self):
        cases = [png(8, constant=True), png(8, transparent=True), png(7),
                 png(8, header=struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)),
                 png(8, filters=(5,))]
        for data in cases:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                image.png_metadata(data, 8, 8)

    def test_inflate_size_bomb_invalid_stream_and_trailing_zlib_rejected(self):
        for compressed in (zlib.compress(b"x" * 100000), b"invalid", zlib.compress(b"x") + b"trailing"):
            with self.subTest(length=len(compressed)), self.assertRaises(ValueError):
                image.png_metadata(png(8, compressed=compressed), 8, 8)

    def test_local_url_accepts_only_owned_origin_root_routes(self):
        base = "http://127.0.0.1:4567"
        for url in ("/uploads/generated.png", base + "/uploads/generated.png"):
            self.assertEqual(image.local_artifact_route(url, base), "/uploads/generated.png")
        for url in ("http://example.com/a", "http://127.0.0.1:9999/a", "//evil/a", "file:///a",
                    "http://user@127.0.0.1:4567/a", "/a#fragment", "/a\n", "\\evil", "relative.png", ""):
            with self.subTest(url=url), self.assertRaises(ValueError):
                image.local_artifact_route(url, base)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.result = {"assertions": [], "inputs": {}, "evidence": {}}
        self.http = FakeHttp(self.result)
        self.context = {"artifact_dir": self.root, "model_options": {}}

    def run_smoke(self):
        image.run_image_smoke(self.http, "fixture", self.context, self.result)

    def test_smoke_checks_negatives_and_both_pngs(self):
        self.run_smoke()
        self.assertEqual(len(self.http.requests), 6)
        self.assertEqual(len(self.http.downloads), 2)
        self.assertEqual(len(self.result["assertions"]), 10)
        self.assertTrue(all(row["passed"] for row in self.result["assertions"]))
        self.assertTrue(self.result["evidence"]["image_seeded_repeat"]["pixel_sha_equal"])
        self.assertFalse(self.result["evidence"]["image_seeded_repeat"]["qualified"])
        self.assertEqual(self.http.requests[-1]["steps"], 2)
        self.assertEqual(len(list(self.root.glob("image-smoke-*.png"))), 2)

    def test_turbo_uses_published_eight_steps(self):
        self.context["model_options"]["qwen-image-variant"] = "turbo"
        self.run_smoke()
        self.assertTrue(all(payload["steps"] == 8 for payload in self.http.requests))

    def test_repeat_difference_is_observed_not_claimed_as_qualified_determinism(self):
        self.http.repeat_changes = True
        self.run_smoke()
        observed = self.result["evidence"]["image_seeded_repeat"]
        self.assertFalse(observed["pixel_sha_equal"])
        self.assertFalse(observed["qualified"])
        self.assertIn("untested", self.result["evidence"]["image_scope"]["semantic_quality"])

    def test_accepted_invalid_request_fails_before_generation_download(self):
        self.http.reject_invalid = False
        with self.assertRaises(ScenarioFailure):
            self.run_smoke()
        self.assertFalse(self.http.downloads)

    def test_external_artifact_or_wrong_mime_fails_without_external_access(self):
        self.http.url = "https://example.com/image.png"
        with self.assertRaises(ScenarioFailure):
            self.run_smoke()
        self.assertFalse(self.http.downloads)
        self.http.url, self.http.mime = "/uploads/generated.png", "text/plain"
        with self.assertRaises(ScenarioFailure):
            self.run_smoke()

    def test_constant_or_truncated_output_never_passes(self):
        for content in (png(constant=True), png()[:-5]):
            self.http.content = content
            with self.subTest(length=len(content)), self.assertRaises(ScenarioFailure):
                self.run_smoke()
            receipt = self.result["evidence"]["generated_images"][0]
            self.assertEqual(receipt["validation"], "incomplete")
            self.assertEqual(Path(receipt["artifact"]).read_bytes(), content)

    def test_required_companions_checked_before_and_after_preparation(self):
        self.assertFalse(image.availability(self.context)[0])
        options = {"qwen-image-vae": str(self.root / "vae"), "qwen-image-vl": str(self.root / "vl")}
        self.context.update(model_options=options, preparing_model=True)
        self.assertTrue(image.availability(self.context)[0])
        self.context["preparing_model"] = False
        self.assertFalse(image.availability(self.context)[0])
        for path in options.values():
            Path(path).write_bytes(b"synthetic companion; not a model")
        self.assertTrue(image.availability(self.context)[0])
        options["qwen-image-variant"] = "unverified"
        self.assertFalse(image.availability(self.context)[0])


if __name__ == "__main__":
    unittest.main()
