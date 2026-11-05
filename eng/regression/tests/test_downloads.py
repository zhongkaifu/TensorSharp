"""Small loopback-only payloads: no external hosts, models or GPU are touched."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import ConfigError, FileSpec, flatten, loads_jsonc, read_config
from downloads import DownloadPolicy, PrepareError, prepare_files, probe_files


PAYLOAD = b"GGUF" + bytes(range(256)) * 32
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Windows CI can expose an 8.3 TEMP alias; the resolver canonicalizes it.
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "config" / "fixture.json"
        self.config.parent.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, obj):
        self.config.write_text(json.dumps(obj), encoding="utf-8")

    def test_jsonc_preserves_urls_strings_escaped_quotes_and_commas(self):
        value = loads_jsonc(r'''{ // comment
            "url": "https://host/a//b/*c*/", /* block */
            "s": "x\"//", "comma": ",}", "items": [1,2,],
        }''')
        self.assertEqual(value["url"], "https://host/a//b/*c*/")
        self.assertEqual(value["s"], 'x"//')
        self.assertEqual(value["comma"], ",}")
        self.assertEqual(value["items"], [1, 2])

    def test_invalid_json_and_duplicate_properties_fail(self):
        for text in ('{"x":1,"x":2}', '{"x": NaN}', '/* unclosed', '{"x": "unclosed}'):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                loads_jsonc(text)

    def test_variables_groups_and_optional_files(self):
        self.write({"variables": {"root": "${TENSORSHARP_MODELS:-${base}/models}", "base": ".."},
                    "model": {"path": "${root}/main.gguf", "sha256": SHA,
                              "urls": ["https://host/file"], "files": [
                                  {"path": "${root}/tokenizer.json", "optional": True}]}})
        resolved = read_config(self.config, self.root / "cache", environ={})
        self.assertEqual(resolved.files[0].path, self.root / "cache" / "main.gguf")
        self.assertEqual(resolved.files[1].group, resolved.files[0].group)
        self.assertTrue(resolved.files[1].optional)
        self.assertNotIn("variables", resolved.options)
        self.assertIsInstance(resolved.options["model"], str)

    def test_override_skips_invalid_unused_variable_and_download(self):
        self.write({"model": {"path": "${undefined}/model.gguf", "urls": ["bad"]},
                    "mmproj": {"path": "${also_undefined}"}})
        resolved = read_config(self.config, overrides={"model": str(self.root / "local.gguf"), "mmproj": "none"})
        self.assertEqual(len(resolved.files), 1)
        self.assertEqual(resolved.files[0].path, self.root / "local.gguf")

    def test_unavailable_cannot_be_bypassed_and_is_network_free(self):
        self.write({"$unavailable": "not implemented", "model": {"path": "${missing}"}})
        result = read_config(self.config, overrides={"$unavailable": None})
        self.assertEqual(result.unavailable, "not implemented")
        self.assertEqual(result.files, [])

    def test_cycles_and_undefined_variables_rejected(self):
        for variables in ({"a": "${b}", "b": "${a}"}, {}):
            self.write({"variables": variables, "model": "${a}"})
            with self.assertRaises(ConfigError):
                read_config(self.config, environ={})

    def test_split_override_infers_local_siblings_without_inventing_sources(self):
        self.write({"model": str(self.root / "model-00001-of-00003.gguf")})
        result = read_config(self.config)
        self.assertEqual(len(result.files), 3)
        self.assertTrue(all(not item.urls for item in result.files))
        self.assertEqual(result.files[2].path.name, "model-00003-of-00003.gguf")

    def test_file_schema_rejects_malformed_hash_size_and_shapes(self):
        for extra in ({"sha256": "bad"}, {"urls": "not-list"}, {"files": {}},
                      {"size_bytes": -1}, {"optional": "true"}, {"typo": 1}):
            self.write({"model": {"path": "model.gguf", **extra}})
            with self.subTest(extra=extra), self.assertRaises(ConfigError):
                read_config(self.config)

    def test_shipped_host_presets_parse_without_model_io(self):
        repo = Path(__file__).resolve().parents[3]
        for path in (repo / "config").glob("*.json"):
            with self.subTest(path=path.name):
                parsed = read_config(path, self.root / "models", environ={})
                self.assertIsInstance(flatten(parsed), list)

    def test_environment_defaults_are_validated_raw_metadata_not_downloads_or_process_mutations(self):
        key = "TS_CONFIG_OFFLINE_TEST"
        values = {key: "value", "TS_Q35_NINFER_DRAFT_PTX": "${appBaseDirectory}/cuda_kernels/draft.ptx"}
        self.write({"environment": values})
        with mock.patch.dict(os.environ, {key: "inherited"}):
            parsed = read_config(self.config, environ={})
            self.assertEqual(os.environ[key], "inherited")
        self.assertEqual(parsed.environment_defaults, values)
        self.assertEqual(parsed.options, {})
        self.assertEqual(parsed.files, [])
        self.assertEqual(flatten(parsed), [])

    def test_environment_invalid_shapes_names_values_and_builtin_override_are_rejected(self):
        for invalid in (None, [], "value", {"X": 1}, {"X": False}, {"X": None},
                        {"": "x"}, {"X=Y": "x"}, {"X\x00Y": "x"}, {"X": "nul\x00value"}):
            self.write({"environment": invalid})
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                read_config(self.config, environ={})
        self.write({"variables": {"appBaseDirectory": "/invented"}})
        with self.assertRaises(ConfigError):
            read_config(self.config, environ={})

    def test_shared_ninfer_profiles_preserve_host_only_environment_metadata(self):
        repo = Path(__file__).resolve().parents[3]
        for suffix, context in (("", "40960"), ("-64k", "65536")):
            parsed = read_config(repo / f"config/qwen3.8-27b-ninfer{suffix}.json", self.root / "models", environ={})
            self.assertEqual(parsed.environment_defaults["MAX_CONTEXT"], context)
            self.assertEqual(parsed.environment_defaults["TS_SPEC_ADAPTIVE"], "0")
            self.assertIn("${appBaseDirectory}", parsed.environment_defaults["TS_Q35_NINFER_DRAFT_PTX"])
            self.assertEqual(parsed.options["backend"], "cuda")
            self.assertNotIn("environment", parsed.options)
            self.assertEqual(len(parsed.files), 1)
            self.assertEqual(parsed.files[0].sha256, "74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82")


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                cls.requests.append(("HEAD", self.path, self.headers.get("Range")))
                if self.path == "/auth":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                if self.path != "/unknown":
                    self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()

            def do_GET(self):
                cls.requests.append(("GET", self.path, self.headers.get("Range")))
                if self.path == "/failure":
                    self.send_response(503)
                    self.end_headers()
                    return
                data = PAYLOAD
                offset = 0
                if self.headers.get("Range") and self.path != "/ignore-range":
                    offset = int(self.headers["Range"].split("=")[1].split("-")[0])
                    self.send_response(206)
                    reported_offset = offset + 1 if self.path == "/bad-range" else offset
                    self.send_header("Content-Range", f"bytes {reported_offset}-{len(data)-1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - offset))
                self.end_headers()
                if self.path == "/short":
                    self.wfile.write(data[offset:offset + 1024])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.wfile.write(data[offset:])

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.policy = DownloadPolicy(self.root, len(PAYLOAD) * 5, 0, True,
                                     chunk_bytes=1024, allow_loopback_http=True)
        self.requests.clear()

    def tearDown(self):
        self.temp.cleanup()

    def spec(self, route="/file", **kwargs):
        return FileSpec("model", self.root / "model.gguf", (self.url + route,), kwargs.pop("sha256", SHA), **kwargs)

    def assert_error(self, code, callback):
        with self.assertRaises(PrepareError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def partial(self, spec, data):
        Path(str(spec.path) + ".part").write_bytes(data)
        Path(str(spec.path) + ".part.json").write_text(
            json.dumps({"version": 1, "sha256": spec.sha256, "size_bytes": len(PAYLOAD)}), encoding="utf-8")

    def test_sequential_download_verify_and_cached_reuse(self):
        spec = self.spec()
        records = prepare_files([spec], self.policy)
        self.assertEqual(records[0].status, "downloaded")
        self.assertEqual(records[0].sha256, SHA)
        self.assertEqual(spec.path.read_bytes(), PAYLOAD)
        self.assertFalse(Path(str(spec.path) + ".part").exists())
        self.requests.clear()
        self.assertEqual(prepare_files([spec], self.policy)[0].status, "local")
        self.assertEqual(self.requests, [])

    def test_probe_offline_does_not_hash_or_network(self):
        spec = self.spec()
        spec.path.write_bytes(PAYLOAD)
        with mock.patch("downloads._hash", side_effect=AssertionError("must not hash")):
            result = probe_files([spec], self.policy)
        self.assertEqual(result[0]["status"], "local-unverified")
        self.assertEqual(result[0]["size_bytes"], len(PAYLOAD))
        self.assertEqual(self.requests, [])

    def test_immutable_head_metadata_reused_between_plan_and_prepare(self):
        spec = self.spec()
        first = probe_files([spec], self.policy, network=True)
        self.assertEqual(first[0]["remaining_bytes"], len(PAYLOAD))
        self.partial(spec, PAYLOAD[:2048])
        second = probe_files([spec], self.policy, network=True)
        self.assertEqual(second[0]["remaining_bytes"], len(PAYLOAD) - 2048)
        prepare_files([spec], self.policy)
        self.assertEqual(sum(request[0] == "HEAD" for request in self.requests), 1)
        self.assertEqual(sum(request[0] == "GET" for request in self.requests), 1)

    def test_partial_resume_and_atomic_promotion(self):
        spec = self.spec()
        self.partial(spec, PAYLOAD[:2048])
        result = prepare_files([spec], self.policy)
        self.assertEqual(result[0].sha256, SHA)
        self.assertIn(("GET", "/file", "bytes=2048-"), self.requests)
        self.assertEqual(self.policy.downloaded_bytes, len(PAYLOAD) - 2048)

    def test_completed_partial_verifies_without_get(self):
        spec = self.spec()
        self.partial(spec, PAYLOAD)
        prepare_files([spec], self.policy)
        self.assertFalse(any(request[0] == "GET" for request in self.requests))

    def test_corrupt_cached_file_not_overwritten(self):
        spec = self.spec()
        spec.path.write_bytes(b"GGUF-corrupt")
        self.assert_error("checksum_mismatch", lambda: prepare_files([spec], self.policy))
        self.assertEqual(spec.path.read_bytes(), b"GGUF-corrupt")
        self.assertEqual(self.requests, [])

    def test_corrupt_download_is_retained_as_partial(self):
        spec = self.spec(sha256="f" * 64)
        self.assert_error("checksum_mismatch", lambda: prepare_files([spec], self.policy))
        self.assertFalse(spec.path.exists())
        self.assertTrue(Path(str(spec.path) + ".part").exists())

    def test_budget_and_disk_checks_happen_before_get(self):
        self.policy.max_download_bytes = len(PAYLOAD) - 1
        self.assert_error("download_budget", lambda: prepare_files([self.spec()], self.policy))
        self.assertFalse(any(request[0] == "GET" for request in self.requests))
        self.policy.max_download_bytes = len(PAYLOAD) * 5
        with mock.patch("downloads._free_bytes", return_value=0):
            self.assert_error("disk_budget", lambda: prepare_files([self.spec()], self.policy))

    def test_total_budget_reserves_all_required_files_before_get(self):
        self.policy.max_download_bytes = len(PAYLOAD) + 1
        other = FileSpec("mmproj", self.root / "projector.gguf", (self.url + "/file",), SHA)
        self.assert_error("download_budget", lambda: prepare_files([self.spec(), other], self.policy))
        self.assertFalse(any(request[0] == "GET" for request in self.requests))

    def test_untrusted_partial_is_not_modified(self):
        spec = self.spec()
        partial = Path(str(spec.path) + ".part")
        partial.write_bytes(b"user data")
        self.assert_error("partial_unowned", lambda: prepare_files([spec], self.policy))
        self.assertEqual(partial.read_bytes(), b"user data")

    def test_missing_files_and_sources_explicit(self):
        self.policy.allow_download = False
        self.assert_error("download_disabled", lambda: prepare_files([self.spec()], self.policy))
        self.policy.allow_download = True
        self.assert_error("unverified_source", lambda: prepare_files([self.spec(sha256=None)], self.policy))
        self.assert_error("missing_source", lambda: prepare_files([FileSpec("model", self.root / "x", (), SHA)], self.policy))

    def test_unknown_size_blocks_and_auth_is_actionable(self):
        self.assert_error("unknown_size", lambda: prepare_files([self.spec("/unknown")], self.policy))
        self.assert_error("credentials_or_license", lambda: prepare_files([self.spec("/auth")], self.policy))
        self.assertFalse(any(request[0] == "GET" for request in self.requests))

    def test_optional_auth_failure_is_visible_not_pass(self):
        result = prepare_files([self.spec("/auth", optional=True)], self.policy)
        self.assertEqual(result[0].status, "optional-skipped")
        self.assertIn("credentials_or_license", result[0].reason)

    def test_server_failure_has_stage_and_no_final_file(self):
        self.assert_error("download_failed", lambda: prepare_files([self.spec("/failure")], self.policy))
        self.assertFalse(self.spec().path.exists())

    def test_truncated_transfer_is_not_a_cached_success(self):
        spec = self.spec("/short")
        self.assert_error("partial_file", lambda: prepare_files([spec], self.policy))
        self.assertFalse(spec.path.exists())
        self.assertEqual(Path(str(spec.path) + ".part").stat().st_size, 1024)

    def test_optional_companion_does_not_displace_required_budget(self):
        self.policy.max_download_bytes = len(PAYLOAD)
        optional = FileSpec("mmproj", self.root / "optional.gguf", (self.url + "/file",), SHA, optional=True)
        result = prepare_files([optional, self.spec()], self.policy)
        self.assertEqual({x.status for x in result}, {"optional-skipped", "downloaded"})
        self.assertFalse(optional.path.exists())

    def test_cancel_preserves_resumable_partial_and_releases_lock(self):
        cancelled = False

        def progress(event):
            nonlocal cancelled
            if event["stage"] == "download":
                cancelled = True

        spec = self.spec()
        self.assert_error("cancelled", lambda: prepare_files([spec], self.policy, lambda: cancelled, progress))
        self.assertFalse(spec.path.exists())
        self.assertTrue(Path(str(spec.path) + ".part").exists())
        self.assertFalse(Path(str(spec.path) + ".download.lock").exists())
        self.assertEqual(prepare_files([spec], self.policy)[0].sha256, SHA)

    def test_download_target_escape_rejected_but_local_outside_root_allowed(self):
        outside = self.root.parent / (self.root.name + "-not-created.gguf")
        spec = FileSpec("model", outside, (self.url + "/file",), SHA)
        self.assert_error("unsafe_path", lambda: prepare_files([spec], self.policy))
        self.assertFalse(outside.exists())
        self.assertEqual(self.requests, [])

    def test_pinned_source_and_sha_required_before_network(self):
        self.policy.allow_loopback_http = False
        spec = FileSpec("model", self.root / "m", ("https://huggingface.co/org/repo/resolve/main/a.gguf",), SHA)
        with mock.patch("downloads._open", side_effect=AssertionError("network forbidden")):
            self.assert_error("unverified_source", lambda: prepare_files([spec], self.policy))

    def test_timeout_is_classified_and_never_finalized(self):
        with mock.patch("downloads._open", side_effect=socket.timeout()):
            self.assert_error("timeout", lambda: prepare_files([self.spec()], self.policy))

    def test_restart_without_range_support_obeys_remaining_budget(self):
        spec = self.spec("/ignore-range")
        self.partial(spec, PAYLOAD[:2048])
        self.policy.max_download_bytes = len(PAYLOAD) - 2048
        self.assert_error("download_budget", lambda: prepare_files([spec], self.policy))
        self.assertEqual(Path(str(spec.path) + ".part").read_bytes(), PAYLOAD[:2048])

    def test_wrong_resume_offset_is_rejected_without_modifying_partial(self):
        spec = self.spec("/bad-range")
        self.partial(spec, PAYLOAD[:2048])
        self.assert_error("invalid_range", lambda: prepare_files([spec], self.policy))
        self.assertEqual(Path(str(spec.path) + ".part").read_bytes(), PAYLOAD[:2048])


if __name__ == "__main__":
    unittest.main()
