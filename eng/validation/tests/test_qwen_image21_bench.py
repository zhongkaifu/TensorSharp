"""Schedule and command coverage for the matched Qwen-Image-2.1 benchmark."""
import importlib.util
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "qwen-image21-bench.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_bench", MODULE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class QwenImage21BenchmarkTests(unittest.TestCase):
    def fallback_result(self):
        log = ("generating image: 1/1 - seed 42\n"
               " |=> | 1/2 - 2.00s/it\n |====| 2/2 - 2.00s/it\n"
               "sampling completed, taking 4.00s\n"
               "[ERROR] ggml_runner.cpp:873 - wan_vae segment 1/1 (graph) failed during weight preparation\n"
               "[ERROR] vae.hpp:312 - vae decode compute failed\n"
               "[WARN] backend_fit.cpp:502 - VAE decode failed (likely out of memory); retrying with spatial tiling\n"
               " |=> | 1/3 - 1.00s/it\n |==> | 2/3 - 1.00s/it\n |====| 3/3 - 1.00s/it\n"
               "decode_first_stage completed, taking 3.00s\n"
               "generate_image completed in 7.00s\n"
               "save result image 0 to 'output.png' (success)\n")
        return dict(BENCH.parse_log("sd_cpp", log), exit_code=0,
                    image={"exists": True, "width": 1024, "height": 1024})

    def test_recovered_vae_tiling_is_explicit_and_excluded_from_denoise(self):
        result = self.fallback_result()
        self.assertEqual([step["step"] for step in result["steps"]], [1, 2])
        self.assertEqual(BENCH.classify_result(result, "sd_cpp", 2, 1024, 1024), "completed_with_fallback")
        self.assertEqual(len(result["error_lines"]), 3)
        self.assertEqual(result["validation_failures"], [])

    def test_fallback_never_masks_exit_errors_missing_steps_or_invalid_images(self):
        for change in (lambda r: r.update(exit_code=1),
                       lambda r: r.update(timed_out=True),
                       lambda r: r["steps"].pop(),
                       lambda r: r["image"].update(decode_error="bad PNG"),
                       lambda r: r["image"].pop("width"),
                       lambda r: r["image"].update(width=512),
                       lambda r: r["fallback_events"][0].update(recovery_confirmed_in_log=False),
                       lambda r: r["error_lines"].append("[ERROR] unrelated failure")):
            result = self.fallback_result()
            change(result)
            self.assertEqual(BENCH.classify_result(result, "sd_cpp", 2, 1024, 1024), "failed")

    def test_unrecognized_error_cannot_pass(self):
        result = self.fallback_result()
        result["fallback_events"] = []
        self.assertEqual(BENCH.classify_result(result, "sd_cpp", 2, 1024, 1024), "failed")
        result["error_lines"] = []
        self.assertEqual(BENCH.classify_result(result, "sd_cpp", 2, 1024, 1024), "passed")

    def test_cooldown_occurs_only_between_runs_and_reports_actual_elapsed(self):
        with patch.object(BENCH.time, "sleep") as sleep, patch.object(BENCH.time, "perf_counter", side_effect=[10, 70.1]):
            self.assertEqual(BENCH.cooldown_before_run(60, False), 0)
            self.assertEqual(BENCH.cooldown_before_run(0, True), 0)
            sleep.assert_not_called()
            self.assertAlmostEqual(BENCH.cooldown_before_run(60, True), 60.1)
            sleep.assert_called_once_with(60)

    def test_gpu_csv_preserves_device_identity_and_unavailable_values(self):
        devices = BENCH.parse_gpu_sample('0, GPU-a, NVIDIA RTX, 512.5, 16384, 75, 62\n'
                                        '1, GPU-b, Other GPU, 128, 8192, [N/A], [Not Supported]\n')
        self.assertEqual(devices[0]["memory_used_mib"], 512.5)
        self.assertEqual(devices[0]["temperature_c"], 62)
        self.assertEqual(devices[1]["uuid"], "GPU-b")
        self.assertIsNone(devices[1]["utilization_percent"])
        self.assertIsNone(devices[1]["temperature_c"])
        with self.assertRaises(ValueError):
            BENCH.parse_gpu_sample("unexpected output")

    def test_gpu_report_separates_baseline_peaks_and_whole_device_scope(self):
        baseline = BENCH.parse_gpu_sample('0,GPU-a,RTX,500,16000,0,N/A')
        samples = [{"elapsed_seconds": 1, "devices": BENCH.parse_gpu_sample('0,GPU-a,RTX,900,16000,95,70')},
                   {"elapsed_seconds": 2, "devices": BENCH.parse_gpu_sample('0,GPU-a,RTX,800,16000,90,75')}]
        report = BENCH.summarize_gpu_samples(baseline, samples, 1, [])
        self.assertEqual(report["status"], "available")
        self.assertEqual(report["baseline"][0]["memory_used_mib"], 500)
        self.assertEqual(report["peaks"][0]["peak_memory_used_mib"], 900)
        self.assertEqual(report["peaks"][0]["peak_utilization_percent"], 95)
        self.assertEqual(report["peaks"][0]["peak_temperature_c"], 75)
        self.assertIn("not this process", report["scope"])

    def test_gpu_unavailable_report_retains_reason(self):
        report = BENCH.summarize_gpu_samples([], [], 2, ["nvidia-smi missing"])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["peaks"], [])
        self.assertEqual(report["sample_interval_seconds"], 2)
        self.assertEqual(report["errors"], ["nvidia-smi missing"])

    def test_gpu_baseline_alone_does_not_claim_runtime_measurement(self):
        baseline = BENCH.parse_gpu_sample('0,GPU-a,RTX,500,16000,0,40')
        report = BENCH.summarize_gpu_samples(baseline, [], 1, ["runtime query failed"])
        self.assertEqual(report["status"], "baseline_only")
        self.assertEqual(report["runtime_sampling_status"], "unavailable")

    def test_sd_progress_accepts_seconds_and_reciprocal_rates(self):
        log = ("\r  |=> | 1/3 - 4.64s/it\x1b[K\n"
               "\r  |====> | 2/3 - 1.25it/s\x1b[K\n"
               "\r  |======| 3/3 - 2.00it/s\x1b[K\n"
               "  |########| 128/128 - 300.00MB/s\n"
               "sampling completed, taking 5.94s\n")
        actual = BENCH.parse_log("sd_cpp", log)
        self.assertEqual([step["step"] for step in actual["steps"]], [1, 2, 3])
        self.assertEqual([step["seconds"] for step in actual["steps"]], [4.64, .8, .5])
        self.assertEqual(actual["steps"][-1]["reported_unit"], "it/s")
        self.assertEqual(actual["phases_seconds"]["denoise"], 5.94)
        self.assertAlmostEqual(actual["steady_step_mean_seconds"], .65)

    def test_zero_reciprocal_rate_does_not_falsely_confirm_completion(self):
        actual = BENCH.parse_log("sd_cpp", "  |==| 1/1 - 0.00it/s\n")
        self.assertEqual(actual["steps"], [])

    def test_conditioning_diagnostics_ignore_unused_negative_prompt(self):
        args = SimpleNamespace(prompt="A red cube.", negative_prompt="(blur:1.2)", cfg=1,
                               image=[], width=1024, height=1024, ts_extra=[], sd_extra=[], engine="both")
        self.assertEqual(BENCH.comparison_notes(args), [])
        args.cfg = 2
        self.assertTrue(any("prompt-weighting" in note for note in BENCH.comparison_notes(args)))

    def test_edit_diagnostics_distinguish_2k_reference_workload(self):
        args = SimpleNamespace(prompt="Change the cube to blue.", negative_prompt="", cfg=1,
                               image=[Path("reference.png")], width=1024, height=1024,
                               ts_extra=[], sd_extra=[], engine="both")
        notes = BENCH.comparison_notes(args)
        self.assertTrue(any("different filters" in note for note in notes))
        self.assertFalse(any("1 megapixel" in note for note in notes))
        args.width = args.height = 2048
        args.sd_extra = ["--ref-image-args", "vae_input_max_pixels=1048576"]
        notes = BENCH.comparison_notes(args)
        self.assertTrue(any("1 megapixel" in note for note in notes))
        self.assertTrue(any("Extra arguments" in note for note in notes))

    def test_edit_noise_defaults_match_the_comparison_and_never_guess(self):
        resolve = BENCH.resolve_edit_noise
        self.assertIsNone(resolve("t2i", "both", None, None))
        self.assertEqual(resolve("edit", "both", None, None), "seed")
        self.assertEqual(resolve("multi", "sd_cpp", None, ""), "seed")
        self.assertEqual(resolve("edit", "tensorsharp", None, None), "references")
        # An A/B arm's environment override and an explicit flag both win over the default.
        self.assertEqual(resolve("edit", "tensorsharp", None, " Seed "), "seed")
        self.assertEqual(resolve("edit", "both", "references", None), "references")
        self.assertEqual(resolve("edit", "both", "seed", "seed"), "seed")
        for mode, requested, inherited in (("edit", "references", "seed"), ("edit", None, "sd"), ("t2i", "seed", None)):
            with self.subTest(mode=mode, requested=requested, inherited=inherited), self.assertRaises(ValueError):
                resolve(mode, "both", requested, inherited)

    def test_edit_notes_say_whether_the_engines_shared_noise(self):
        args = SimpleNamespace(prompt="Change the cube to blue.", negative_prompt="", cfg=1, image=[Path("reference.png")],
                               width=1024, height=1024, ts_extra=[], sd_extra=[], engine="both", edit_noise="seed")
        self.assertTrue(any("TS_QWEN21_EDIT_NOISE=seed" in note for note in BENCH.comparison_notes(args)))
        args.edit_noise = "references"
        self.assertTrue(any("different noise" in note for note in BENCH.comparison_notes(args)))
        args.engine = "tensorsharp"
        self.assertFalse(any("noise" in note for note in BENCH.comparison_notes(args)))

    def test_schedule_matches_official_golden_vectors(self):
        # The same independent Diffusers golden vectors are used by
        # QwenImage21SamplingTests.FortyStepScheduleMatchesOfficialQwenImage21Scheduler.
        cases = {
            256: (.9843579531, .8282203674, .6143688560, .3408319354, .0601277351),
            4096: (.9869637489, .8528681993, .6566662788, .3819332719, .0678805709),
            8192: (.9892514348, .8756618500, .6988655925, .4275377989, .0776016116),
            16384: (.9926459789, .9116604924, .7724375129, .5205857754, .1022295356),
        }
        for tokens, expected in cases.items():
            with self.subTest(tokens=tokens):
                actual = BENCH.qwen21_sigmas(40, tokens)
                self.assertEqual(len(actual), 41)
                for index, value in zip((1, 10, 20, 30, 38), expected):
                    self.assertLessEqual(abs(actual[index] - value), 2e-7)
                self.assertEqual(actual[-2], struct.unpack("<f", struct.pack("<f", .02))[0])
                self.assertEqual(actual[-1], 0)

    def test_schedule_boundaries_and_monotonicity(self):
        for steps in (1, 2, 4, 25, 40, 100):
            for tokens in (4, 256, 4096, 8192, 16384, 65536):
                values = BENCH.qwen21_sigmas(steps, tokens)
                self.assertEqual(len(values), steps + 1)
                self.assertEqual((values[0], values[-1]), (1, 0))
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        for steps, tokens in ((0, 256), (-1, 256), (40, 0), (40, -1)):
            with self.assertRaises(ValueError):
                BENCH.qwen21_sigmas(steps, tokens)

    def test_reference_command_receives_f32_roundtrippable_sigmas(self):
        args = SimpleNamespace(
            dotnet="dotnet", cli=Path("TensorSharp.Cli.dll"), backend="ggml_cuda",
            prompt="A red cube on a white table.", width=512, height=1024, steps=40,
            cfg=1, seed=42, sd_cli=Path("sd-cli.exe"), sd_backend="cuda0",
            match_sigmas=True, sigma_nodes=None, lora=None, variant=None,
            negative_prompt="", image=[], ts_extra=[], sd_extra=[])
        models = {name: Path(filename) for name, filename in BENCH.MODEL_NAMES.items()}
        commands = BENCH.commands(args, models, Path("ts.png"), Path("sd.png"))
        sd = commands["sd_cpp"]
        text = sd[sd.index("--sigmas") + 1]
        parsed = [struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in text.split(",")]
        self.assertEqual(parsed, BENCH.qwen21_sigmas(40, 2048))
        self.assertNotIn("--sigmas", commands["tensorsharp"])
        self.assertEqual(sd[sd.index("--rng") + 1], "cuda")
        args.match_sigmas = False
        self.assertNotIn("--sigmas", BENCH.commands(args, models, Path("ts.png"), Path("sd.png"))["sd_cpp"])


if __name__ == "__main__":
    unittest.main()
