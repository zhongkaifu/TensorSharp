# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""Tests for the parts that do not need ComfyUI: request bodies, the SSE reader against a
fake server, and launch profiles. Run them with the Python that runs ComfyUI, which has
aiohttp:

    python tests/test_client.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest

from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import client as ts  # noqa: E402
import server_manager as sm  # noqa: E402


class BodyTests(unittest.TestCase):
    def test_zero_values_are_left_to_the_model(self):
        body = ts.build_video_body("a fox", seed=7)
        self.assertEqual(body, {"prompt": "a fox", "generateAudio": True, "seed": 7})

    def test_every_field(self):
        body = ts.build_video_body(
            "a fox", negative_prompt="blur", mode="fl2v", width=640, height=384, frames=22, steps=8, cfg=1.5,
            seed=2**64 - 1, fps=24, generate_audio=False, first_frame_png=b"\x89PNG", end_image="last.png",
            reference_images=["a.png", "b.png"])
        self.assertEqual(body["videoMode"], "fl2v")
        self.assertEqual(body["seed"], (2**64 - 1) % 2**63)
        self.assertTrue(body["image"].startswith("data:image/png;base64,"))
        self.assertEqual(body["endImage"], "last.png")
        self.assertEqual(body["referenceImages"], ["a.png", "b.png"])
        self.assertIs(body["generateAudio"], False)
        self.assertEqual((body["width"], body["height"], body["frames"], body["steps"], body["fps"]), (640, 384, 22, 8, 24))

    def test_refuses_empty_prompt_and_unknown_mode(self):
        with self.assertRaises(ValueError):
            ts.build_video_body("  ")
        with self.assertRaises(ValueError):
            ts.build_video_body("x", mode="v2v")

    def test_image_body(self):
        self.assertEqual(ts.build_image_body("cat", seed=1, image_paths=["r.png"]),
                         {"prompt": "cat", "seed": 1, "imagePaths": ["r.png"]})

    def test_error_envelopes(self):
        self.assertEqual(ts.describe_error({"error": "nope"}), "nope")
        self.assertEqual(ts.describe_error({"error": {"message": "bad"}}), "bad")
        self.assertEqual(ts.describe_error("x", 500), "HTTP 500: x")


class SseTests(unittest.TestCase):
    def test_partial_lines_are_kept(self):
        buf = ts.SseBuffer()
        self.assertEqual(buf.feed(b'data: {"st'), [])
        lines = buf.feed(b'ep": 1}\n\ndata: {"done": true}\n')
        self.assertEqual([ts.parse_sse_data(l) for l in lines], [{"step": 1}, None, {"done": True}])

    def test_non_data_lines(self):
        self.assertIsNone(ts.parse_sse_data(b": keep-alive"))
        self.assertIsNone(ts.parse_sse_data(b"data: [DONE]"))


def _event(payload) -> bytes:
    return ("data: " + json.dumps(payload) + "\n\n").encode()


class FakeServer:
    """Streams scripted events; records whether the client hung up."""

    def __init__(self, script, delay=0.05):
        self.script = script
        self.delay = delay
        self.disconnected = asyncio.Event()

    async def handle(self, request):
        await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            for item in self.script:
                if item == "hang":
                    for _ in range(200):
                        await asyncio.sleep(0.05)
                        await resp.write(b": heartbeat\n\n")
                    continue
                data = _event(item)
                # Split each event in two writes to exercise partial reads.
                await resp.write(data[:7])
                await asyncio.sleep(self.delay)
                await resp.write(data[7:])
        except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
            self.disconnected.set()
        return resp

    async def refuse(self, request):
        return web.json_response({"ok": False, "error": "The loaded model is not a video-generation model."}, status=400)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, server):
        app = web.Application()
        app.router.add_post("/stream", server.handle)
        app.router.add_post("/refuse", server.refuse)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_events_until_done(self):
        url = await self.start(FakeServer([{"videoGen": True, "step": 1, "total": 2},
                                           {"videoGen": True, "step": 2, "total": 2},
                                           {"done": True, "url": "/uploads/v.mp4"}]))
        async with ts.TensorSharpClient(url) as c:
            events = [e async for e in c.stream("/stream", {"prompt": "x"})]
        self.assertEqual([e.get("step") for e in events[:2]], [1, 2])
        self.assertEqual(events[-1]["url"], "/uploads/v.mp4")

    async def test_error_frame_raises(self):
        url = await self.start(FakeServer([{"done": True, "error": "steps contradicts the LoRA"}]))
        async with ts.TensorSharpClient(url) as c:
            with self.assertRaisesRegex(ts.TensorSharpError, "contradicts"):
                [e async for e in c.stream("/stream", {"prompt": "x"})]

    async def test_http_refusal_raises(self):
        url = await self.start(FakeServer([]))
        async with ts.TensorSharpClient(url) as c:
            with self.assertRaisesRegex(ts.TensorSharpError, "not a video-generation model"):
                [e async for e in c.stream("/refuse", {"prompt": "x"})]

    async def test_stream_without_result_raises(self):
        url = await self.start(FakeServer([{"videoGen": True, "step": 1, "total": 2}]))
        async with ts.TensorSharpClient(url) as c:
            with self.assertRaisesRegex(ts.TensorSharpError, "without a result"):
                [e async for e in c.stream("/stream", {"prompt": "x"})]

    async def test_stop_closes_the_connection(self):
        server = FakeServer([{"videoGen": True, "step": 1, "total": 8}, "hang"])
        url = await self.start(server)
        seen = []
        async with ts.TensorSharpClient(url) as c:
            with self.assertRaises(ts.Interrupted):
                async for e in c.stream("/stream", {"prompt": "x"}, should_stop=lambda: bool(seen), poll_seconds=0.1):
                    seen.append(e)
        await asyncio.wait_for(server.disconnected.wait(), timeout=5)
        self.assertEqual(len(seen), 1)


REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))
    return path


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.environ["TS_TEST_DIR"] = self.dir
        os.environ.pop(sm.EXTRA_PROFILES_ENV, None)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("TS_TEST_DIR", None)
        os.environ.pop(sm.EXTRA_PROFILES_ENV, None)

    def test_config_profile_resolves_against_the_checkout(self):
        _write(os.path.join(self.dir, "p.json"), {"config": "config/ltx-2.5-distilled.json", "args": ["--backend", "ggml_cuda"],
                                                  "env": {"A": "${TS_TEST_DIR}"}, "path_prepend": ["${TS_TEST_DIR}"],
                                                  "service_name": "svc", "unknown": 1})
        _write(os.path.join(self.dir, "_private.json"), "{}")
        self.assertEqual(sm.list_profiles([self.dir]), ["p"])
        p = sm.load_profile("p", [self.dir], repo_root=REPO)
        self.assertEqual(p.config, os.path.join(REPO, "config", "ltx-2.5-distilled.json"))
        self.assertEqual(p.model_name(), "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf")
        env = p.environment({"PATH": "base", "OTEL_SERVICE_NAME": "comfyui"})
        self.assertEqual(env["A"], self.dir)
        self.assertEqual(env["PATH"], self.dir + os.pathsep + "base")
        self.assertEqual(env["OTEL_SERVICE_NAME"], "svc")

    def test_relative_model_is_taken_from_the_profile_folder(self):
        _write(os.path.join(self.dir, "m.json"), {"model": "models/a.gguf"})
        p = sm.load_profile("m", [self.dir])
        self.assertEqual(p.model, os.path.join(self.dir, "models", "a.gguf"))
        self.assertEqual(p.model_name(), "a.gguf")

    def test_command(self):
        dll = _write(os.path.join(self.dir, "Server.dll"), "")
        cfg = _write(os.path.join(self.dir, "c.json"), '{ "model": { "path": "${root}/x.gguf" } }')
        p = sm.Profile(name="p", config=cfg, args=["--backend", "ggml_cuda"], server_dll=dll)
        self.assertEqual(p.command("127.0.0.1", 5090),
                         ["dotnet", dll, "--config", cfg, "--host", "127.0.0.1", "--port", "5090", "--backend", "ggml_cuda"])
        self.assertEqual(p.model_name(), "x.gguf")
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            sm.Profile(name="p", config=cfg, server_dll=os.path.join(self.dir, "missing.dll")).command("h", 1)

    def test_refuses_unset_variables_and_empty_profiles(self):
        _write(os.path.join(self.dir, "u.json"), {"config": "${TS_SURELY_UNSET_VAR}/c.json"})
        with self.assertRaisesRegex(ValueError, "TS_SURELY_UNSET_VAR"):
            sm.load_profile("u", [self.dir])
        _write(os.path.join(self.dir, "e.json"), {"args": ["--backend", "cpu"]})
        with self.assertRaisesRegex(ValueError, "neither"):
            sm.load_profile("e", [self.dir])
        with self.assertRaises(FileNotFoundError):
            sm.load_profile("nope", [self.dir])

    def test_extra_profile_folder_comes_first(self):
        _write(os.path.join(self.dir, "ltx25-distilled.json"), {"model": "private.gguf"})
        os.environ[sm.EXTRA_PROFILES_ENV] = self.dir
        self.assertEqual(sm.profile_dirs(), [self.dir, sm.PROFILE_DIR])
        self.assertIn("qwen-image-2.1-turbo", sm.list_profiles())
        self.assertEqual(sm.load_profile("ltx25-distilled").model_name(), "private.gguf")

    def test_server_is_found_in_the_checkout(self):
        self.assertEqual(sm.default_server_dll(REPO),
                         os.path.join(REPO, "TensorSharp.Server.Host", "bin", "TensorSharp.Server.Host.dll"))
        self.assertEqual(sm.REPO_ROOT, REPO)
        os.environ.pop(sm.SERVER_DLL_ENV, None)
        self.assertEqual(sm.Profile(name="p", model="m.gguf").resolved_server_dll(), sm.default_server_dll())
        os.environ[sm.SERVER_DLL_ENV] = "X.dll"
        try:
            self.assertEqual(sm.Profile(name="p", model="m.gguf").resolved_server_dll(), "X.dll")
        finally:
            os.environ.pop(sm.SERVER_DLL_ENV, None)

    def test_hosted_model_matching(self):
        self.assertEqual(sm.hosted_model({"loaded": r"C:\m\a.gguf", "models": ["a.gguf"]}), r"C:\m\a.gguf")
        self.assertEqual(sm.hosted_model({"models": [{"name": "b.gguf"}]}), "b.gguf")
        self.assertTrue(sm.same_model("a.gguf", r"C:\models\A.gguf"))
        self.assertTrue(sm.same_model("a", "a.gguf"))
        self.assertFalse(sm.same_model("", "a.gguf"))
        self.assertFalse(sm.same_model("a.gguf", ""))
        self.assertFalse(sm.same_model("b.gguf", "a.gguf"))

    def test_shipped_profiles_name_presets_in_this_checkout(self):
        names = sm.list_profiles([sm.PROFILE_DIR])
        self.assertEqual(names, ["ltx25-distilled", "qwen-image-2.1-turbo"])
        for name in names:
            p = sm.load_profile(name, [sm.PROFILE_DIR])
            self.assertTrue(os.path.isfile(p.config), p.config)
            self.assertTrue(p.model_name().endswith(".gguf"), name)
            self.assertIn("--backend", p.args, name)

    @unittest.skipUnless(sys.platform.startswith("linux"), "setpriv is Linux-only")
    def test_linux_launch_prefix(self):
        prefix = sm.launch_prefix()
        if prefix:
            self.assertEqual(prefix[1:], ["--pdeathsig", "TERM", "--"])

    def test_no_launch_prefix_elsewhere(self):
        if not sys.platform.startswith("linux"):
            self.assertEqual(sm.launch_prefix(), [])


if __name__ == "__main__":
    unittest.main()
