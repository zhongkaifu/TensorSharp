"""Video profile tests use synthetic media receipts; no models or GPU are loaded."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from eng.regression import video_profiles as video
from eng.regression.scenarios import ScenarioFailure, ScenarioUntested


class VideoProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "synthetic.gguf"
        self.model.write_bytes(b"never loaded")
        self.output = self.root / "artifacts"
        self.output.mkdir()
        disk = patch.object(video.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10 * 1024 ** 3})())
        disk.start()
        self.addCleanup(disk.stop)
        self.context = {"repo": self.root, "artifact_dir": self.output, "cli_command": ["fixture-cli"],
            "model_path": self.model, "backend": "cuda", "model_family": "wan", "timeout_seconds": 7,
            "disk_headroom_gib": 0,
            "model_options": {name: str(self.model) for name in video._COMPANIONS}}
        self.profile = {"id": "synthetic-wan", "scenario": "video_generation", "preset": "wan-video-ti2v-5b",
            "parameters": {"prompt": "A synthetic red toy car.", "width": 64, "height": 64, "frames": 5,
                           "fps": 24, "steps": 4, "cfg": 1, "seed": 42, "sampler": "euler"},
            "fixtures": {}, "expected": {"max_rgb_rmse": 0}}
        self.calls = []
        self.backend = "Cuda"
        self.failure_stage = self.failure = None
        self.bad_pixels = self.bad_audio = False
        self.wrong_frame_count = False
        self.refresh()

    def fixture(self, name, data, suffix=".raw"):
        path = self.root / (name + suffix)
        path.write_bytes(data)
        self.profile["fixtures"][name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}

    def refresh(self):
        p = self.profile["parameters"]
        self.rgb = bytes(range(256)) * (p["width"] * p["height"] * p["frames"] * 3 // 256)
        self.fixture("reference_rgb", self.rgb)
        if self.profile["scenario"] == "video_audio_generation" or p.get("audio"):
            rate = 48000 if self.context["model_family"] == "ltx-2.5" else 32000
            samples = round(p["frames"] / p["fps"] * rate)
            self.pcm = struct.pack("<f", .25) * (samples * 2)
            self.fixture("reference_pcm", self.pcm)
            self.profile["expected"].update(audio_samples_per_channel=samples, max_audio_rmse=0)
        self.context["scenario_profile"] = self.profile

    def execute(self, command, **kwargs):
        stage = kwargs["stage"]
        self.calls.append((command, kwargs))
        stdout = ""
        p = self.profile["parameters"]
        if stage == "video-generate":
            (self.output / "generated.mp4").write_bytes(b"synthetic encoded file, codec mocked")
            if self.profile["scenario"] == "video_audio_generation" or p.get("audio"):
                (self.output / "generated.wav").write_bytes(b"synthetic WAV, codec mocked")
            stdout = "Backend: " + self.backend + "\n"
            for name, marker in (("first_image", "image"), ("end_image", "end"), ("reference_image", "ref-img")):
                if name in self.profile["fixtures"]:
                    stdout += "  " + marker + " : " + self.profile["fixtures"][name]["path"] + "\n"
        elif stage == "video-probe":
            stdout = json.dumps({"streams": [{"codec_type": "video", "width": p["width"], "height": p["height"],
                "nb_read_frames": str(p["frames"] + int(self.wrong_frame_count)), "avg_frame_rate": str(p["fps"]) + "/1"}]})
        elif stage == "video-decode":
            (self.output / "decoded.rgb24").write_bytes(bytes(len(self.rgb)) if self.bad_pixels else self.rgb)
        elif stage == "audio-probe":
            stdout = json.dumps({"streams": [{"codec_type": "audio", "sample_rate": "48000" if self.context["model_family"] == "ltx-2.5" else "32000", "channels": 2}]})
        elif stage == "audio-decode":
            (self.output / "decoded.pcm.f32le").write_bytes(struct.pack("<f", float("nan")) + self.pcm[4:] if self.bad_audio else self.pcm)
        result = {"stage": stage, "command": command, "stdout": stdout, "stderr": "", "exit_code": 0,
                  "timed_out": False, "output_truncated": False, "teardown": {"attempted": True, "verified": True}}
        if stage == self.failure_stage:
            result.update(self.failure)
        return result

    def run_profile(self):
        result = {"assertions": [], "inputs": {}, "evidence": {}, "processes": []}
        with patch.object(video.shutil, "which", side_effect=lambda name: name), patch.object(video, "run_process", side_effect=self.execute):
            video.run({"id": self.profile["scenario"]}, self.context, result)
        return result

    def ready(self):
        with patch.object(video.shutil, "which", side_effect=lambda name: name):
            return video.availability({"id": self.profile["scenario"]}, self.context)

    def test_wan_full_decode_and_gold_are_required(self):
        result = self.run_profile()
        self.assertTrue(all(a["passed"] for a in result["assertions"]))
        self.assertEqual(result["evidence"]["video"]["decoded"]["rmse"], 0)
        self.assertEqual(len(result["processes"]), 3)
        command = self.calls[0][0]
        self.assertIn("--no-audio", command)
        self.assertEqual(command[command.index("--video-frames") + 1], "5")
        self.assertNotIn("--config", command)
        self.assertTrue(all(kwargs["timeout_seconds"] == 7 for _, kwargs in self.calls))
        self.assertIn("-xerror", self.calls[2][0])
        self.assertIn("-fs", self.calls[2][0])

    def test_h3_audio_has_exact_finite_pcm_and_duration(self):
        self.context["model_family"] = "minimax-h3"
        self.profile["scenario"] = "video_audio_generation"
        self.refresh()
        result = self.run_profile()
        self.assertEqual(result["evidence"]["audio"]["decoded"]["rmse"], 0)
        self.assertEqual(result["evidence"]["audio"]["decoded"]["rms"], .25)
        self.assertEqual(len(self.calls), 5)
        self.assertNotIn("--no-audio", self.calls[0][0])

    def test_ltx_exact_backend_schedule_and_audio(self):
        self.context["model_family"] = "ltx-2.5"
        self.profile["scenario"] = "video_audio_generation"
        self.profile["parameters"].update(frames=9, steps=8, sampler="euler_ancestral")
        self.refresh()
        self.assertTrue(self.ready()[0])
        self.assertTrue(self.run_profile()["evidence"]["backend"]["verified"])
        self.assertIn("--video-upscaler", self.calls[0][0])

    def test_reference_fixture_receipt_and_golden_output(self):
        self.profile["scenario"] = "video_reference"
        self.profile["parameters"]["mode"] = "i2v"
        self.fixture("first_image", b"synthetic approved fixture, no image decoder invoked", ".png")
        self.refresh()
        result = self.run_profile()
        self.assertEqual(result["evidence"]["conditioning"]["mode"], "i2v")
        self.assertIn("--image", self.calls[0][0])
        self.assertIn("no face/subject identity", result["evidence"]["conditioning"]["scope"])

    def test_missing_profile_and_codecs_are_unavailable(self):
        self.context.pop("scenario_profile")
        self.assertIn("--scenario-profiles", video.availability({"id": "video_generation"}, self.context)[1])
        self.context["scenario_profile"] = self.profile
        with patch.object(video.shutil, "which", return_value=None):
            self.assertIn("ffmpeg", video.availability({"id": "video_generation"}, self.context)[1])

    def test_unknown_keys_booleans_and_unbounded_values_are_rejected(self):
        changes = [("command", ["shell"]), ("width", True), ("width", 2048), ("fps", 0), ("cfg", float("nan")),
                   ("frames", 122), ("steps", 51), ("seed", -1), ("sampler", "unknown"), ("audio", "yes")]
        for key, value in changes:
            profile = copy.deepcopy(self.profile)
            profile["parameters"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                video.validate_profile(profile)

    def test_unused_model_roles_cannot_trigger_extra_downloads(self):
        self.profile["models"] = {"unused": "large-unrelated-model"}
        with self.assertRaisesRegex(ValueError, "not executed"):
            video.validate_profile(self.profile)

    def test_reference_and_audio_require_approved_gold(self):
        for name in ("reference_rgb",):
            profile = copy.deepcopy(self.profile)
            profile["fixtures"].pop(name)
            with self.assertRaises(ValueError):
                video.validate_profile(profile)
        self.profile["scenario"] = "video_reference"
        self.assertFalse(self.ready()[0])
        self.profile["scenario"] = "video_audio_generation"
        self.assertFalse(self.ready()[0])

    def test_wrong_fixture_hash_stops_before_model_process(self):
        self.profile["fixtures"]["reference_rgb"]["sha256"] = "0" * 64
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(self.calls, [])

    def test_wrong_fixture_size_and_missing_companion_are_unavailable(self):
        path = Path(self.profile["fixtures"]["reference_rgb"]["path"])
        path.write_bytes(b"partial")
        self.assertIn("complete frame-major", self.ready()[1])
        self.refresh()
        self.context["model_options"].pop("video-vae")
        self.assertIn("video-vae", self.ready()[1])

    def test_ltx_alias_never_counts_as_ggml_backend(self):
        self.context.update(model_family="ltx-2.5", backend="ggml_cuda")
        self.profile["parameters"].update(frames=9, steps=8)
        self.refresh()
        self.assertIn("aliases", self.ready()[1])

    def test_required_options_omit_gated_identity_adapter(self):
        self.assertEqual(video.required_options(self.profile), {"video-vae", "video-text-encoder"})
        self.profile["preset"] = "ltx-2.5-distilled"
        self.assertEqual(video.required_options(self.profile), {"video-vae", "video-text-encoder", "video-upscaler", "audio-vae"})

    def test_wan_i2v_only_and_ti2v_geometry_fail_before_execution(self):
        self.profile["preset"] = "wan-video-i2v-a14b"
        self.assertIn("I2V-A14B", self.ready()[1])
        self.profile["preset"] = "wan-video-ti2v-5b"
        self.profile["parameters"]["width"] = 80
        self.refresh()
        self.assertIn("32-pixel", self.ready()[1])

    def test_family_grids_and_ltx_gated_identity_are_explicit(self):
        self.context["model_family"] = "minimax-h3"
        self.profile["parameters"]["frames"] = 9
        self.refresh()
        self.assertIn("17n+5", self.ready()[1])
        self.context["model_family"] = "ltx-2.5"
        self.profile["parameters"].update(steps=8, mode="ref")
        self.profile["scenario"] = "video_reference"
        self.fixture("reference_image", b"mock", ".png")
        self.assertIn("Ingredients", self.ready()[1])

    def test_wrong_backend_fallback_and_timeout_are_failures(self):
        self.backend = "Cpu"
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_timeout_never_decodes_or_passes(self):
        self.failure_stage = "video-generate"
        self.failure = {"timed_out": True, "exit_code": -1}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_backend_fallback_log_invalidates_matching_banner(self):
        self.failure_stage = "video-generate"
        self.failure = {"stderr": "Falling back to CPU"}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_oom_exit_preserves_failure_and_stops(self):
        self.failure_stage = "video-generate"
        self.failure = {"exit_code": 1, "stderr": "CUDA_ERROR_OUT_OF_MEMORY"}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_codec_timeout_never_counts_as_complete_decode(self):
        self.failure_stage = "video-decode"
        self.failure = {"timed_out": True}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 3)

    def test_truncated_generation_output_cannot_attest_backend(self):
        self.failure_stage = "video-generate"
        self.failure = {"output_truncated": True}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_unverified_cleanup_stops_further_codec_launches(self):
        self.failure_stage = "video-generate"
        self.failure = {"containment_failure": True, "teardown": {"attempted": True, "verified": False}}
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 1)

    def test_zero_exit_without_video_is_not_pass(self):
        result = {"assertions": [], "inputs": {}, "evidence": {}, "processes": []}
        def absent(command, **kwargs):
            receipt = self.execute(command, **kwargs)
            (self.output / "generated.mp4").unlink()
            return receipt
        with patch.object(video.shutil, "which", side_effect=lambda x: x), patch.object(video, "run_process", side_effect=absent):
            with self.assertRaises(ScenarioFailure):
                video.run({"id": self.profile["scenario"]}, self.context, result)

    def test_wrong_frame_count_is_not_pass(self):
        self.wrong_frame_count = True
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual(len(self.calls), 2)

    def test_blank_or_incorrect_pixels_fail(self):
        self.bad_pixels = True
        with self.assertRaises(ScenarioFailure):
            self.run_profile()

    def test_finite_but_different_pixels_fail_exact_oracle(self):
        self.fixture("reference_rgb", self.rgb[::-1])
        with self.assertRaises(ScenarioFailure):
            self.run_profile()

    def test_nonfinite_audio_fails_even_after_good_video(self):
        self.context["model_family"] = "minimax-h3"
        self.profile["scenario"] = "video_audio_generation"
        self.refresh()
        self.bad_audio = True
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_profile()

    def test_stale_outputs_never_overwritten(self):
        (self.output / "generated.mp4").write_bytes(b"user old artifact")
        with self.assertRaises(ScenarioFailure):
            self.run_profile()
        self.assertEqual((self.output / "generated.mp4").read_bytes(), b"user old artifact")
        self.assertEqual(self.calls, [])

    def test_low_disk_is_untested_before_execution(self):
        with patch.object(video.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            with self.assertRaises(ScenarioUntested):
                self.run_profile()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
