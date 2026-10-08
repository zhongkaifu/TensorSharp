"""Integrity must not be inferred from plausible lengths or download sidecars."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("preflight", Path(__file__).parents[1] / "local-model-preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def gguf(path, payload=b"\0" * 16, split=None, tensors=True):
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    values = {"general.architecture": "test"}
    if split:
        values.update(zip(("split.no", "split.count", "split.tensors.count"), split))
    data = b"GGUF" + struct.pack("<IQQ", 3, int(tensors), len(values))
    for key, value in values.items():
        data += string(key)
        data += struct.pack("<I", 8) + string(value) if isinstance(value, str) else struct.pack("<II", 4, value)
    if tensors:
        data += string(path.stem) + struct.pack("<IQIQ", 1, 4, 0, 0)
    data += b"\0" * (-len(data) % 32)
    path.write_bytes(data + (payload if tensors else b""))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_hash_with_oracle_is_verified(self):
        path = self.root / "complete.gguf"
        gguf(path)
        expected = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.assertEqual("verified", PREFLIGHT.inspect_file(path, expected, True)["status"])
        self.assertEqual("structure_only", PREFLIGHT.inspect_file(path, expected)["status"])

    def test_valid_header_and_full_length_zeros_cannot_fake_checkpoint(self):
        path = self.root / "sparse.gguf"
        gguf(path, b"\x01" * 16)
        expected = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        gguf(path, b"\0" * 16)
        Path(str(path) + ".ranges.json").write_text(json.dumps({"prefix": path.stat().st_size, "completed": [], "total": path.stat().st_size}))
        report = PREFLIGHT.inspect_file(path, expected, True)
        self.assertEqual("invalid", report["status"])
        self.assertIn("SHA-256", report["error"])

    def test_hash_without_independent_oracle_is_identity_only(self):
        path = self.root / "unknown.gguf"
        gguf(path)
        self.assertEqual("identity_only", PREFLIGHT.inspect_file(path, hash_contents=True)["status"])

    def test_truncated_payload_is_invalid(self):
        path = self.root / "truncated.gguf"
        gguf(path, b"\0" * 15)
        self.assertEqual("invalid", PREFLIGHT.inspect_file(path)["status"])

    def test_metadata_only_first_shard_is_valid_but_missing_shard_is_not(self):
        paths = [self.root / f"model-{i:05}-of-00002.gguf" for i in (1, 2)]
        gguf(paths[0], split=(0, 2, 1), tensors=False)
        gguf(paths[1], split=(1, 2, 1))
        reports = [PREFLIGHT.inspect_file(path) for path in paths]
        self.assertEqual(0, reports[0]["structure"]["tensor_count"])
        self.assertTrue(PREFLIGHT.split_checks(reports)[0]["structurally_complete"])
        self.assertFalse(PREFLIGHT.split_checks(reports[:1])[0]["structurally_complete"])

    def test_missing_ranges_are_reported_without_claiming_verified(self):
        path = self.root / "partial.gguf"
        gguf(path)
        Path(str(path) + ".ranges.json").write_text(json.dumps({"prefix": 0, "completed": [], "total": path.stat().st_size}))
        report = PREFLIGHT.inspect_file(path)
        self.assertFalse(report["download_records"][0]["ranges_cover_file"])
        self.assertEqual([0], report["download_records"][0]["missing_chunk_starts"])
        self.assertEqual("structure_only", report["status"])

    def test_safetensors_truncation_is_invalid(self):
        path = self.root / "model.safetensors"
        header = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 15)
        self.assertEqual("invalid", PREFLIGHT.inspect_file(path)["status"])


if __name__ == "__main__":
    unittest.main()
