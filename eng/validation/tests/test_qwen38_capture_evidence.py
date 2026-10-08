import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

tool_dir = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture_evidence", tool_dir / "qwen38_capture_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompletedCaptureEvidence(unittest.TestCase):
    def fixture(self, directory):
        blob, index_path, report_path = (directory / name for name in ("rows.f32", "rows.json", "model.json"))
        data = struct.pack("<fff", 1., 2., 3.)
        blob.write_bytes(data * 2)
        digest = hashlib.sha256(data).hexdigest()
        rows = [{"stage": "prefill" if step == 0 else "decode", "iteration": 0, "warmup": False,
                 "elements": 3, "byte_offset": step * 12, "sha256": digest,
                 "input_tokens": [0, 1] + [2] * step} for step in range(2)]
        index = {"format": "f32le", "data_path": str(blob), "rows": rows}
        report = {"passed": True, "run_complete": True, "error": None, "synthetic": True,
                  "model_sha256": "a" * 64, "model_bytes": 123,
                  "native_sha256": "b" * 64, "managed_assemblies_sha256": {"Models.dll": "c" * 64},
                  "model_geometry": {"architecture": "qwen4exp", "hidden_size": 4, "layers": 2,
                      "heads": 2, "kv_heads": 1, "vocabulary": 3, "context_limit": 64, "kv_dtype": "f16"},
                  "cleanup": {"model_disposed": True, "cache_cleared": True, "reuse_released": True,
                      "native_shutdown": True, "retained_model_owner": False, "retained_scope_owner": False,
                      "scope_detached": False, "errors": []},
                  "requested_options": {"iterations": "1", "warmup": "0", "decode-tokens": "1"},
                  "logit_captures": {**index, "index_path": str(index_path)},
                  "prompt_tokens": [0, 1], "forced_tokens": [2], "decode_mode": "teacher-forced",
                  "runs": [{"iteration": 0, "warmup": False, "prefill_tokens": 2, "decode_tokens": 1,
                            "final_logit_sha256": digest}]}
        self.write(index_path, report_path, index, report)
        return index_path, report_path, index, report

    @staticmethod
    def write(index_path, report_path, index, report):
        index_path.write_text(json.dumps(index), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")

    def test_complete_bound_capture_accepts_every_expected_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, report, _, _ = self.fixture(Path(temporary))
            self.assertEqual(2, module.validate(index, report)["validated_rows"])

    def test_equal_truncated_files_do_not_pass_cli_even_if_available_rows_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ip, rp, index, report = self.fixture(directory)
            index["rows"] = index["rows"][:1]
            report["logit_captures"]["rows"] = index["rows"]
            report["passed"] = report["run_complete"] = False
            Path(index["data_path"]).write_bytes(struct.pack("<fff", 1., 2., 3.))
            self.write(ip, rp, index, report)
            output = directory / "comparison.json"
            result = subprocess.run([sys.executable, str(tool_dir / "qwen38-compare-captures.py"),
                "--left", str(ip), "--right", str(ip), "--left-report", str(rp), "--right-report", str(rp),
                "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(1, result.returncode, result.stderr)
            compared = json.loads(output.read_text())
            self.assertFalse(compared["passed"])
            self.assertFalse(compared["run_complete"])
            self.assertTrue(compared["rows"][0]["passed"])
            self.assertEqual(2, len(compared["errors"]))

    def test_complete_claim_still_requires_all_requested_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            index["rows"] = index["rows"][:1]
            report["logit_captures"]["rows"] = index["rows"]
            self.write(ip, rp, index, report)
            with self.assertRaisesRegex(ValueError, "missing a completed"):
                module.validate(ip, rp)

    def test_histories_must_append_exactly_the_consumed_teacher_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            for changed in ([0, 1], [0, 1, 0], [0, 1, 2, 2]):
                index["rows"][1]["input_tokens"] = changed
                self.write(ip, rp, index, report)
                with self.assertRaisesRegex(ValueError, "histories"):
                    module.validate(ip, rp)

    def test_cleanup_failure_geometry_absence_and_wrong_final_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            variants = []
            cleanup = copy.deepcopy(report); cleanup["cleanup"]["retained_model_owner"] = True; variants.append(cleanup)
            geometry = copy.deepcopy(report); geometry["model_geometry"] = {}; variants.append(geometry)
            final = copy.deepcopy(report); final["runs"][0]["final_logit_sha256"] = "0" * 64; variants.append(final)
            for variant in variants:
                self.write(ip, rp, index, variant)
                with self.assertRaises(ValueError):
                    module.validate(ip, rp)

    def test_real_checkpoint_cannot_use_only_a_path_as_its_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            report["synthetic"] = False
            report["model_path"] = "some-model.gguf"
            self.write(ip, rp, index, report)
            with self.assertRaisesRegex(ValueError, "verified checkpoint identity"):
                module.validate(ip, rp)

    def test_warmup_iterations_are_distinct_and_their_bytes_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            warmup = copy.deepcopy(index["rows"])
            for row in warmup:
                row["iteration"], row["warmup"] = -1, True
            for row in index["rows"]:
                row["byte_offset"] += 24
            index["rows"] = warmup + index["rows"]
            report["logit_captures"]["rows"] = index["rows"]
            first_run = copy.deepcopy(report["runs"][0])
            first_run["iteration"], first_run["warmup"] = -1, True
            report["runs"].insert(0, first_run)
            report["requested_options"]["warmup"] = "1"
            blob = Path(index["data_path"])
            data = blob.read_bytes(); blob.write_bytes(data * 2)
            self.write(ip, rp, index, report)
            self.assertEqual(4, module.validate(ip, rp)["validated_rows"])
            blob.write_bytes(struct.pack("<f", float("nan")) + blob.read_bytes()[4:])
            index["rows"][0]["sha256"] = hashlib.sha256(blob.read_bytes()[:12]).hexdigest()
            self.write(ip, rp, index, report)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                module.validate(ip, rp)

    def test_greedy_length_boundary_requires_consumed_history_and_all_generated_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            ip, rp, index, report = self.fixture(Path(temporary))
            report["decode_mode"] = "greedy"
            report["requested_options"]["decode-tokens"] = "2"
            run = report["runs"][0]
            run["generated_tokens"], run["finish_reason"] = [2, 1], "length"
            self.write(ip, rp, index, report)
            self.assertEqual(2, module.validate(ip, rp)["validated_rows"])
            run["generated_tokens"] = [2]
            self.write(ip, rp, index, report)
            with self.assertRaisesRegex(ValueError, "generated/consumed"):
                module.validate(ip, rp)

    def test_real_checkpoint_identity_is_bound_to_loaded_path_and_compared_by_all_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ip, rp, index, report = self.fixture(directory)
            report["synthetic"] = False
            report["model_path"] = str(directory / "model-00001-of-00002.gguf")
            report["checkpoint_identity"] = {"current_metadata_checked": True,
                "manifest_sha256": "d" * 64, "model_path": report["model_path"],
                "files": [{"sha256": "e" * 64, "bytes": 10}, {"sha256": "f" * 64, "bytes": 20}]}
            self.write(ip, rp, index, report)
            self.assertEqual([("e" * 64, 10), ("f" * 64, 20)], module.validate(ip, rp)["checkpoint"])
            report["checkpoint_identity"]["model_path"] = str(directory / "other.gguf")
            self.write(ip, rp, index, report)
            with self.assertRaisesRegex(ValueError, "loaded model path"):
                module.validate(ip, rp)


if __name__ == "__main__":
    unittest.main()
