"""CPU MoE policies are explicit, constrained, and never guesses about fit."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression import runner
from regression.profiles import annotate_resource_plan, residency_profile


class ResidencyProfileTests(unittest.TestCase):
    def test_cli_requires_both_explicit_peak_requirements(self):
        for options in ([], ["--ram-required-gib", "16"], ["--vram-required-gib", "6"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--cpu-moe", *options])
        args = runner.parse_args(["--cpu-moe", "--ram-required-gib", "16", "--vram-required-gib", "6"])
        self.assertTrue(args.cpu_moe)

    def test_profile_rejects_invalid_requirements_even_outside_cli(self):
        for value in (None, 0, -1, float("nan"), float("inf"), True, "6"):
            for key in ("ram_required_gib", "vram_required_gib"):
                with self.subTest(value=value, key=key):
                    options = {"ram_required_gib": 16, "vram_required_gib": 6, key: value}
                    self.assertFalse(residency_profile(True, **options)["supported"])

    def test_source_supported_family_requires_moe_layout_and_ggml_backend(self):
        for family in ("qwen35", "qwen4exp"):
            for backend in ("ggml_cpu", "ggml_cuda"):
                profile = residency_profile(True, 16, 6, {"family": family, "layout": "moe"}, backend)
                self.assertTrue(profile["supported"])
                self.assertEqual({"TS_CPU_MOE": "1"}, profile["environment"])
                self.assertEqual(backend, profile["backend"])
        for preset, backend in (({"family": "qwen35", "layout": "dense"}, "ggml_cuda"),
                                ({"family": "other", "layout": "moe"}, "ggml_cuda"),
                                ({"family": "qwen4exp", "layout": "moe"}, "cuda"),
                                ({"family": "qwen4exp", "layout": "moe"}, "cpu")):
            self.assertFalse(residency_profile(True, 16, 6, preset, backend)["supported"])

    def test_cpu_backend_does_not_claim_vram_savings(self):
        profile = residency_profile(True, 16, 6, {"family": "qwen4exp", "layout": "moe"}, "ggml_cpu")
        self.assertFalse(profile["vram_requirement_applied"])
        self.assertIn("already", profile["placement_effect"])

    def test_default_requests_no_policy_or_environment_change(self):
        profile = residency_profile()
        self.assertFalse(profile["requested"])
        self.assertFalse(profile["runtime_confirmation_required"])
        self.assertEqual({}, profile["environment"])

    def test_explicit_requirements_do_not_imply_fitting_or_gpu_fraction(self):
        profile = residency_profile(True, 16, 6, {"family": "qwen4exp", "layout": "moe"}, "ggml_cuda")
        plan = annotate_resource_plan({"eligible": False, "reason": "estimated RAM exceeds budget",
            "estimates": {"weight_bytes": 75 * 1024 ** 3, "gpu_weight_fraction": 1}, "warnings": []}, profile)
        self.assertFalse(plan["eligible"])
        self.assertIn("RAM exceeds", plan["reason"])
        self.assertIsNone(plan["estimates"]["gpu_weight_fraction"])
        self.assertEqual(75 * 1024 ** 3, plan["estimates"]["weight_bytes"])
        self.assertEqual("explicit_user_requirements", plan["estimates"]["requirements_source"])

    def test_probe_initializes_offload_before_creating_model_and_emits_attestation(self):
        # Source contract only: this test intentionally does not build/load native dependencies.
        source = (Path(__file__).resolve().parents[2] / "ForcedLogitProbe" / "Program.cs").read_text()
        self.assertLess(source.index("MoeCpuOffloadConfig.ConfigureFromEnvironment();"), source.index("ModelBase.Create("))
        for key in ("cpu_moe_enabled", "all_layers", "cpu_moe_layers", "explicitly_set", "configuration_source"):
            self.assertIn(key + " =", source)


if __name__ == "__main__":
    unittest.main()
