#!/usr/bin/env python3
"""Small validator tests using a local HTTP fixture, not model-quality evidence."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


generation = module("validate-release-generation")
audio = module("validate-release-audio")


def picture(kind):
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    if kind != "blank":
        image[:, :16, 0] = 255
        image[:, 16:, 2] = 255
    output = io.BytesIO()
    Image.fromarray(image).save(output, format="JPEG" if kind == "jpeg" else "PNG")
    return output.getvalue()


class FixtureServer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        content = picture(self.path.removeprefix("/uploads/"))
        self.send_response(200)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        kind = body["fixture"]
        result = {"url": "/uploads/" + kind, "width": 32, "height": 16}
        self.send_response(200)
        streaming = self.path.endswith("/stream")
        self.send_header("Content-Type", "text/event-stream; charset=utf-8" if streaming else "application/json")
        self.end_headers()
        if streaming:
            frame = {"imageEdit": True, "step": 1, "total": 2,
                     "image": "data:image/png;base64," + base64.b64encode(picture("valid")).decode()}
            self.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode())
            if kind != "truncated":
                final = {"done": True, **result} if kind != "error" else {"done": True, "error": "fixture failure"}
                self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
        else:
            self.wfile.write(json.dumps({"ok": True, **result}).encode())


class GenerationValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureServer)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = SimpleNamespace(url=f"http://127.0.0.1:{self.server.server_port}", output=Path(self.temp.name),
                                    timeout=10, minimum_psnr=40, minimum_audio_snr=40)

    def run_fixture(self, kind="valid", streaming=False, expected=None, reference=None):
        case = {"id": "fixture", "endpoint": "/api/image-edit" + ("/stream" if streaming else ""),
                "request": {"fixture": kind}, "expected": expected or {"width": 32, "height": 16}}
        return generation.run_case(self.args, case, "fixture", reference)

    def test_valid_png_and_retained_same_seed_reference(self):
        first = self.run_fixture()
        self.assertEqual(first["status"], "passed-structural", first)
        self.args.output = Path(self.temp.name) / "candidate"
        second = self.run_fixture(reference={"cases": [first]})
        self.assertEqual(second["status"], "passed-structural", second)
        self.assertTrue(second["comparison"]["decoded_exact"])
        self.assertEqual(second["semantic_quality"], "pending-separate-visual-review")

    def test_blank_output_fails(self):
        self.assertIn("blank", self.run_fixture("blank")["error"])

    def test_jpeg_disguised_as_png_fails(self):
        self.assertIn("not a PNG", self.run_fixture("jpeg")["error"])

    def test_wrong_dimensions_fail(self):
        self.assertIn("differs", self.run_fixture(expected={"width": 64})["error"])

    def test_unknown_required_metadata_fails(self):
        self.assertIn("Cannot verify", self.run_fixture(expected={"unknown": 1})["error"])

    def test_stream_progress_preview_and_download(self):
        result = self.run_fixture(streaming=True)
        self.assertEqual(result["status"], "passed-structural", result)
        self.assertEqual(len(result["events"]), 2)
        self.assertTrue(Path(result["events"][0]["image"]["path"]).is_file())

    def test_truncated_stream_fails(self):
        result = self.run_fixture("truncated", streaming=True)
        self.assertIn("without a final", result["error"])
        self.assertEqual(len(result["events"]), 1)

    def test_stream_error_fails(self):
        result = self.run_fixture("error", streaming=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["response"]["error"], "fixture failure")

    def test_cli_failure_is_nonzero_and_remaining_work_is_not_run(self):
        cases = [{"id": kind, "endpoint": "/api/image-edit", "request": {"fixture": kind},
                  "expected": {"width": 32, "height": 16}} for kind in ("blank", "valid")]
        plan_path = Path(self.temp.name) / "plan.json"
        plan_path.write_text(json.dumps({"cases": cases}))
        output = Path(self.temp.name) / "cli"
        result = subprocess.run([sys.executable, generation.__file__, "--url", self.args.url,
                                 "--plan", str(plan_path), "--output", str(output),
                                 "--concurrency", "1", "--repeats", "1"],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        report = json.loads((output / "results.json").read_text())
        self.assertFalse(report["run_complete"])
        self.assertEqual(len(report["warmup"]), 1)
        self.assertEqual(report["cases"], [])
        self.assertIn("valid-c1-r0-i0", report["not_run"])


class AudioAnswerValidatorTests(unittest.TestCase):
    def run_answer(self, content, fallback="4821", reason="different"):
        metrics = {"assistant_message": {"content": content, "reasoning_content": reason},
                   "output_text": fallback, "finish_reason": "stop"}
        args = SimpleNamespace(url="fixture", model="fixture", timeout=1)
        with patch.object(audio.engines, "run_openai_chat", return_value=metrics):
            return audio.run_case(args, {}, "code", "fixture")

    def test_real_final_code_passes(self):
        self.assertEqual(self.run_answer("4821", fallback="wrong")["status"], "ok")

    def test_reasoning_only_answer_fails(self):
        result = self.run_answer(None, reason="4821")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["metrics"]["assistant_message"]["reasoning_content"], "4821")

    def test_wrong_final_cannot_pass_with_correct_fallback(self):
        self.assertEqual(self.run_answer("1234")["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
