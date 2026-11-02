"""CPU-only temporary fixtures and mock POSTs; no ignored evidence dependencies."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


TOOL = module(HERE / "qwen38-cua-cold-control.py", "cold_control_tested")
CUA = module(HERE / "qwen38-cua-control.py", "cold_control_frozen_producer")
BENCH = CUA.load_bench()
MOCK_URL = "http://MOCK_ONLY.invalid"


class FakeResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, expected, prompt, cached):
        args = json.dumps(expected, separators=(",", ":"))
        first = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
            "id": "MOCK_ONLY", "function": {"name": "submit_observation", "arguments": args}}]},
            "finish_reason": None}]}
        details = {} if cached is None else {"cached_tokens": cached}
        last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 80,
                "total_tokens": prompt + 80, "prompt_tokens_details": details}}
        self.lines = [("data: " + json.dumps(x) + "\n\n").encode() for x in (first, last)]
        self.lines.append(b"data: [DONE]\n\n")

    def __enter__(self): return self
    def __exit__(self, *args): return None
    def __iter__(self): return iter(self.lines)


class ColdControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        fixture_dir = cls.directory / "fixture"
        (fixture_dir / "images").mkdir(parents=True)
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            # Small literal padding, but actual CPU-generated PNGs, both lane
            # geometries and immutable prescribed histories, including step25.
            fixture, images = CUA.build_fixture(BENCH, steps=26, prefix_chars=96, window=2)
            for name, data in images.items():
                (fixture_dir / "images" / name).write_bytes(data)
            path = fixture_dir / "fixture.json"
            path.write_text(CUA.canonical(fixture) + "\n", encoding="utf-8")
            prepared = {"schema": CUA.VERSION + "/prepared", "sources": fixture["sources"],
                "fixture_sha256": CUA.file_sha(path), "images": fixture["images"]}
            (fixture_dir / "manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
            cls.fixture, cls.fm = CUA.load_fixture(fixture_dir)
            launch = {"engine": "tensorsharp", "mode": "dflash2", "context_tokens": 65536,
                "lanes": 2, "argv": ["dotnet", "MOCK_ONLY.dll", "--backend", "cuda"],
                "working_directory": "/mock/bin", "environment": {
                    "TS_Q35_GREEDY_VERIFY": "0", "TS_PREFIX_CACHE_ISOLATION": "shared"},
                "binaries": [{"path": "/mock/bin/MOCK_ONLY.dll", "sha256": "a" * 64}],
                "models": [{"path": "/mock/model.gguf", "sha256": "b" * 64}],
                "process": {"pid": 1001, "starttime": 500, "boot_id": "MOCK_BOOT"}}
            header = {"kind": "run", "schema": CUA.VERSION, "engine": "tensorsharp",
                "mode": "dflash2", "label": "MOCK_ONLY_NOT_INFERENCE", "url": MOCK_URL,
                "model": "MOCK_MODEL", "lanes": 1, "steps": 26, "sources": CUA.sources(),
                "fixture_manifest": cls.fm, "fixture_path": str(fixture_dir),
                "launch_manifest": launch, "launch_manifest_sha256": CUA.digest(launch),
                "context_capacity_operator_declaration": 65536,
                "cache_profile_operator_declaration": "trusted-single-tenant-shared"}
            c = TOOL.client(BENCH, header, MOCK_URL)
            output = [header]
            for row in cls.fixture["requests"]:
                if (row["scenario"], row["lanes"], row["lane"]) != ("edit", 1, 0): continue
                expected = copy.deepcopy(row["expected"])
                if row["step"] > 0 and row["step"] % 5 == 0:
                    expected["revision"] = f"R{row['step'] // 5 - 1:03d}"
                warm = CUA.measure(BENCH, c, row, 1,
                    lambda request, expected=expected, **kwargs: FakeResponse(expected, 19000, 14680))
                output.append(warm)
            output.append({"kind": "run_end", "complete": True, "qualified": False})
            cls.source = cls.directory / "MOCK_ONLY_REPLAY.jsonl"
            cls.source.write_text("\n".join(CUA.canonical(r) for r in output) + "\n", encoding="utf-8")
        cls.header, cls.row, cls.ref = TOOL.select(cls.source, "edit", 5, 0)

    def setUp(self):
        self.no_network = patch("urllib.request.urlopen", side_effect=AssertionError("unexpected real POST"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def prepared(self, source=None):
        return TOOL.prepare(BENCH, CUA, self.header, source or self.row,
            self.fixture, self.fm, MOCK_URL)

    def test_step5_exact_fields_and_original_wire_reconstructed_without_network(self):
        c, body, checks, identity = self.prepared()
        self.assertTrue(all(checks.values()))
        self.assertEqual(body, self.row["request_body"])
        self.assertEqual(identity["wire_request_body_sha256"], self.row["wire_request_body_sha256"])
        sorted_wire = hashlib.sha256(json.dumps(self.row["request_body"]).encode()).hexdigest()
        self.assertNotEqual(sorted_wire, identity["wire_request_body_sha256"])

    def test_step25_exact_body_reconstruction(self):
        header, row, ref = TOOL.select(self.source, "edit", 25, 0)
        self.assertTrue(all(TOOL.prepare(BENCH, CUA, header, row,
            self.fixture, self.fm, MOCK_URL)[2].values()))

    def test_tamper_canonical_body_rejected_before_post(self):
        row = copy.deepcopy(self.row)
        row["request_body"]["temperature"] = 0.5
        with self.assertRaises(ValueError): self.prepared(row)

    def run_mock(self, cached, wrong_revision=False, prompt=19000):
        c, body, checks, identity = self.prepared()
        calls = []
        expected = copy.deepcopy(self.row["expected"])
        if wrong_revision: expected["revision"] = "WRONG_MOCK"

        def opened(request, **kwargs):
            calls.append(request)
            self.assertEqual(request.data, json.dumps(body).encode())
            return FakeResponse(expected, prompt, cached)

        result = TOOL.observe(BENCH, CUA, c, body, self.row, 1, opened)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result["response"]["decode_tok_s"])
        wire = base64.b64decode(result["raw_transport"]["body_base64"])
        self.assertEqual(hashlib.sha256(wire).hexdigest(), result["raw_transport"]["body_sha256"])
        return result

    def test_one_mock_post_actual_zero_cache_passes_original_checks(self):
        result = self.run_mock(0)
        self.assertTrue(result["complete"])
        self.assertTrue(result["diagnostic_qualified"])
        self.assertEqual(result["quality"]["verdict"], "PASS")
        self.assertEqual(self.row["quality"]["verdict"], "FAIL")

    def test_nonzero_missing_and_bool_cache_cannot_qualify(self):
        for cached in (14680, None, False):
            result = self.run_mock(cached)
            self.assertFalse(result["diagnostic_qualified"])
            self.assertEqual(result["cold_gate"]["verdict"],
                "FAIL" if type(cached) is int else "UNVERIFIED")

    def test_wrong_revision_retained_as_fail_even_if_cold(self):
        result = self.run_mock(0, True)
        self.assertTrue(result["complete"])
        self.assertFalse(result["diagnostic_qualified"])
        self.assertFalse(result["quality"]["checks"]["typed_exact_arguments"])

    def test_prompt_drop_cannot_qualify(self):
        self.assertFalse(self.run_mock(0, prompt=100)["diagnostic_qualified"])

    def test_same_deployment_ignores_timestamps_pid_and_source_status_metadata(self):
        old = self.header["launch_manifest"]
        fresh = copy.deepcopy(old)
        fresh["process"]["pid"] += 1
        fresh["started_at_utc"] = "MOCK_NEW_TIME"
        fresh["engine_source"] = {"tracked_dirty": True}
        fresh["binaries"][0]["mtime_ns"] = 123
        self.assertTrue(all(TOOL.same_deployment(old, fresh)["checks"].values()))

    def test_changed_argv_binary_model_or_math_policy_rejected(self):
        old = self.header["launch_manifest"]
        for change in ("argv", "binaries", "models", "environment"):
            fresh = copy.deepcopy(old)
            if change == "argv": fresh["argv"] += ["--temperature", "1"]
            elif change in ("binaries", "models"): fresh[change][0]["sha256"] = "f" * 64
            else: fresh["environment"]["TS_Q35_GREEDY_VERIFY"] = "1"
            with self.assertRaises(ValueError): TOOL.same_deployment(old, fresh)


if __name__ == "__main__": unittest.main()
