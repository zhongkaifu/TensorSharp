import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("compare_strata_perf", Path(__file__).resolve().parents[1] / "compare-strata-perf.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "logits.f32"
        path.write_bytes(bytes(8))
        self.report = dict(passed=True, repeated_final_logits_exact=True, backend="Cuda", architecture="fixture",
            model_bytes=123, max_context=256, prompt_tokens=[1], forced_tokens=[2], quality_checked=True,
            quality_passed=True, quality=[dict(name="math", prompt_tokens=[3], generated_tokens=[42, 0], eos=True, passed=True)],
            managed_assemblies_sha256={"a": "123"}, ptx_sha256={"k": "456"}, native_sha256={},
            logits=dict(path=str(path), rows=2, columns=1, format="little-endian-float32", sha256=module.digest(path), final_logit_sha256="f"),
            source_dimensions={"fixture.expert_count": 16},
            environment={"TENSORSHARP_CUDA_MOE_FUSION": "0"}, fusion={"kernel_available": True, "process_enabled": False},
            runs=[dict(warmup=False, prefill_tokens=1, decode_tokens=1, prefill_ms=10., decode_ms=20., final_logit_sha256="f")])

    def candidate(self):
        candidate = copy.deepcopy(self.report)
        candidate["environment"]["TENSORSHARP_CUDA_MOE_FUSION"] = "1"
        candidate["fusion"]["process_enabled"] = True
        return candidate

    def test_identical(self):
        result = module.compare([self.report], [self.candidate()])
        self.assertTrue(result["passed"])
        self.assertFalse(result["complete_checkpoint_file_identity_checked"])
        self.assertTrue(any("Legacy probe" in note for note in result["limitations"]))

    def add_shards(self):
        self.report.update(model_files=[
            dict(path="C:/models/shard-1.gguf", bytes=123, last_write_utc="2026-10-01T00:00:00Z"),
            dict(path="C:/models/shard-2.gguf", bytes=456, last_write_utc="2026-10-01T00:00:01Z")],
            model_file_count=2, model_declared_file_count=2, model_total_file_bytes=579,
            model_file_identity_incomplete=False)

    def test_complete_shards_are_checked(self):
        self.add_shards()
        result = module.compare([self.report], [self.candidate()])
        self.assertTrue(result["passed"])
        self.assertTrue(result["complete_checkpoint_file_identity_checked"])
        self.assertFalse(any("Legacy probe" in note for note in result["limitations"]))

    def test_changed_additional_shard_fails(self):
        self.add_shards()
        for field, value in (("path", "C:/models/other.gguf"), ("bytes", 457),
                             ("last_write_utc", "2026-10-02T00:00:01Z")):
            with self.subTest(field=field):
                candidate = self.candidate()
                candidate["model_files"][1][field] = value
                candidate["model_total_file_bytes"] = sum(f["bytes"] for f in candidate["model_files"])
                with self.assertRaisesRegex(ValueError, "shard identity mismatch"):
                    module.compare([self.report], [candidate])

    def test_incomplete_or_malformed_shard_identity_fails(self):
        self.add_shards()
        for field, value in (("model_file_identity_incomplete", True), ("model_file_count", 1),
                             ("model_total_file_bytes", 123), ("model_declared_file_count", 3)):
            with self.subTest(field=field):
                candidate = self.candidate()
                candidate[field] = value
                with self.assertRaises(ValueError):
                    module.compare([self.report], [candidate])
        candidate = self.candidate()
        del candidate["model_file_identity_incomplete"]
        with self.assertRaises(ValueError):
            module.compare([self.report], [candidate])

    def test_new_and_legacy_shard_identity_cannot_be_mixed(self):
        legacy = self.candidate()
        self.add_shards()
        with self.assertRaisesRegex(ValueError, "shard identity mismatch"):
            module.compare([self.report], [legacy])

    def test_generated_fixture_paths_and_mtimes_may_change(self):
        self.add_shards()
        self.report.update(synthetic=True, synthetic_model_sha256="fixture-full-checksum")
        candidate = self.candidate()
        candidate["model_files"][1].update(path="C:/generated/shard-2.gguf", last_write_utc="2026-10-02T00:00:01Z")
        self.assertTrue(module.compare([self.report], [candidate])["passed"])
        candidate["synthetic_model_sha256"] = "different-fixture"
        with self.assertRaises(ValueError):
            module.compare([self.report], [candidate])

    def test_regression_fails(self):
        candidate = self.candidate()
        candidate["runs"][0]["decode_ms"] = 24
        self.assertFalse(module.compare([self.report], [candidate])["passed"])

    def test_speedup_and_warmup_exclusion(self):
        candidate = self.candidate()
        candidate["runs"][0]["decode_ms"] = 10
        warmup = dict(warmup=True, prefill_tokens=1, decode_tokens=1, prefill_ms=1000., decode_ms=1000., final_logit_sha256="cold-capture")
        candidate["runs"].append(warmup.copy())
        self.report["runs"].append(warmup.copy())
        result = module.compare([self.report], [candidate])
        self.assertEqual(result["metrics"]["decode_ms"]["speedup_percent"], 100.)

    def test_missing_or_tampered_capture_fails(self):
        path = Path(self.report["logits"]["path"])
        path.write_bytes(bytes(4))
        with self.assertRaises(ValueError):
            module.compare([self.report], [self.report])

    def test_changed_tokens_fail(self):
        candidate = self.candidate()
        candidate["quality"][0]["generated_tokens"] = [43, 0]
        with self.assertRaises(ValueError):
            module.compare([self.report], [candidate])

    def test_wrong_binary_or_workload_fails(self):
        for field, value in (("forced_tokens", [3]), ("ptx_sha256", {"k": "789"}),
                             ("source_dimensions", {"fixture.expert_count": 32})):
            candidate = self.candidate()
            candidate[field] = value
            with self.assertRaises(ValueError):
                module.compare([self.report], [candidate])

    def test_changed_environment_controls_fail(self):
        candidate = self.candidate()
        candidate["environment"]["TS_CUDA_PREFILL_GRAPH_MAX_SEQLEN"] = "1"
        with self.assertRaisesRegex(ValueError, "environment controls"):
            module.compare([self.report], [candidate])

    def test_both_feature_flags_may_select_the_same_arm(self):
        candidate = self.candidate()
        self.report["environment"]["TS_Q4E_PREFILL_COMBINE"] = "0"
        candidate["environment"]["TS_Q4E_PREFILL_COMBINE"] = "1"
        self.assertTrue(module.compare([self.report], [candidate])["passed"])
        candidate["environment"]["TS_Q4E_PREFILL_COMBINE"] = "0"
        with self.assertRaisesRegex(ValueError, "Unexpected TS_Q4E_PREFILL_COMBINE"):
            module.compare([self.report], [candidate])

    def test_wrong_effective_fusion_policy_fails(self):
        candidate = self.candidate()
        candidate["fusion"]["process_enabled"] = False
        with self.assertRaisesRegex(ValueError, "fusion policy"):
            module.compare([self.report], [candidate])

    def test_unbalanced_measured_iterations_fail(self):
        candidate = self.candidate()
        candidate["runs"].append(candidate["runs"][0].copy())
        with self.assertRaisesRegex(ValueError, "iteration count"):
            module.compare([self.report], [candidate])

    def test_nonfinite_timing_fails(self):
        self.report["runs"][0]["prefill_ms"] = float("nan")
        with self.assertRaises(ValueError):
            module.validate(self.report)

    def test_unengaged_kernel_or_wrong_arm_fails(self):
        for field, value in (("fusion", {"kernel_available": False}), ("environment", {"TENSORSHARP_CUDA_MOE_FUSION": "0"})):
            candidate = self.candidate()
            candidate[field] = value
            with self.assertRaises(ValueError):
                module.compare([self.report], [candidate])

    def test_wrong_token_count_fails(self):
        self.report["runs"][0]["decode_tokens"] = 2
        with self.assertRaises(ValueError):
            module.validate(self.report)

    def test_changed_capture_warmup_fails(self):
        candidate = self.candidate()
        candidate["runs"].append(dict(warmup=True, prefill_tokens=1, decode_tokens=1, prefill_ms=10., decode_ms=20., final_logit_sha256="different"))
        with self.assertRaises(ValueError):
            module.compare([self.report], [candidate])

    def test_failed_or_empty_probe_fails(self):
        for field, value in (("passed", False), ("repeated_final_logits_exact", False), ("quality_passed", False), ("runs", [])):
            candidate = copy.deepcopy(self.report)
            candidate[field] = value
            with self.assertRaises(ValueError):
                module.validate(candidate)


if __name__ == "__main__":
    unittest.main()
