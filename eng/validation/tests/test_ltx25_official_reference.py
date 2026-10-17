"""Offline identity/scope checks; these tests never load a model runtime."""
import importlib.util
import copy
import hashlib
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ltx25_reference", ROOT / "eng/validation/ltx25-official-denoiser-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class OfficialReferenceScopeTests(unittest.TestCase):
    def test_accepts_only_explicit_forward_precision_and_bounded_depth(self):
        for precision in ["Float32", "Float64"]:
            for blocks in [1, 4]:
                with self.subTest(precision=precision, blocks=blocks):
                    self.assertEqual(reference.validate_probe(dict(completed=True, schedule="forward", output_kind="velocity",
                                                                  frequency_precision=precision, executed_blocks=blocks)), blocks)

    def test_sampled_latents_and_incomplete_or_ambiguous_metadata_are_rejected(self):
        valid = dict(completed=True, schedule="forward", output_kind="velocity", frequency_precision="Float64", executed_blocks=1)
        for changes in [dict(completed=False), dict(schedule="distilled"), dict(output_kind="sampled latent"),
                        dict(schedule=None), dict(frequency_precision=None), dict(frequency_precision="Float16"),
                        dict(executed_blocks=0), dict(executed_blocks=5), dict(executed_blocks=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                reference.validate_probe(dict(valid, **changes))

    def test_full_depth_requires_explicit_residency_override(self):
        full = dict(completed=True, schedule="forward", output_kind="velocity", frequency_precision="Float64", executed_blocks=48)
        with self.assertRaises(ValueError):
            reference.validate_probe(full)
        self.assertEqual(reference.validate_probe(full, 48), 48)
        for limit in [0, 129, True]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                reference.validate_probe(full, limit)


class CapturedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "forward.json"
        self.probe = dict(input_mode="captured", use_reference_lora=False,
                          config=dict(VideoChannels=2, AudioChannels=2, VideoContextDim=3, AudioContextDim=4),
                          tokens=dict(video=2, audio=3, video_context=4, audio_context=5),
                          video_sigma=1.0, audio_sigma=0.725, inputs={})
        for name, shape in dict(video=[2, 2], audio=[3, 2], video_context=[4, 3], audio_context=[5, 4],
                                video_positions=[2, 3, 2], audio_positions=[3, 1, 2], video_timesteps=[2],
                                keyframe_mask=[2]).items():
            values = list(range(math.prod(shape))) if name.endswith("_positions") else [0.0, 1.0] * (math.prod(shape) // 2)
            self.array(name, shape, values)
        self.probe["inputs"].update(audio_timesteps=None, video_denoise_mask=None, clean_video=None)

    def array(self, name, shape, values):
        payload = struct.pack("<" + "f" * len(values), *values)
        path = self.path.parent / (name + ".f32")
        path.write_bytes(payload)
        self.probe["inputs"][name] = dict(path=path.name, sha256=hashlib.sha256(payload).hexdigest(),
                                           shape=shape, dtype="float32-le")

    def test_validates_exact_arrays_and_retains_optional_nulls(self):
        loaded = reference.load_captured_inputs(self.probe, self.path)
        self.assertEqual(struct.unpack("<ff", loaded["video_timesteps"]), (0.0, 1.0))
        self.assertEqual(struct.unpack("<ff", loaded["keyframe_mask"]), (0.0, 1.0))
        self.assertIsNone(loaded["audio_timesteps"])
        self.assertEqual(len(loaded["video_context"]), 4 * 3 * 4)
        self.assertEqual(len(loaded["audio_context"]), 5 * 4 * 4)

    def test_scalar_timestep_override_is_allowed(self):
        self.array("audio_timesteps", [1], [0.25])
        self.assertEqual(struct.unpack("<f", reference.load_captured_inputs(self.probe, self.path)["audio_timesteps"]), (0.25,))

    def test_hash_length_shape_dtype_and_required_fields_are_checked(self):
        cases = [lambda p: p["inputs"]["video"].update(sha256="0" * 64),
                 lambda p: p["inputs"]["video"].update(shape=[1, 4]),
                 lambda p: p["inputs"]["video"].update(shape=[True, 4]),
                 lambda p: p["inputs"]["video"].update(dtype="float16-le"),
                 lambda p: p["inputs"].pop("video_positions"),
                 lambda p: p["inputs"].update(attention_mask=None),
                 lambda p: p["tokens"].update(video=3),
                 lambda p: p["tokens"].update(video=True)]
        for change in cases:
            probe = copy.deepcopy(self.probe)
            change(probe)
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                reference.load_captured_inputs(probe, self.path)
        (self.path.parent / "video.f32").write_bytes(b"1234")
        with self.assertRaisesRegex(ValueError, "byte length"):
            reference.load_captured_inputs(self.probe, self.path)

    def test_values_positions_and_sigma_are_checked_even_with_matching_hash(self):
        for name, shape, values in [("video", [2, 2], [float("nan"), 0, 0, 0]),
                                    ("video_timesteps", [2], [0, 1.1]),
                                    ("video_positions", [2, 3, 2], [1, 0] * 6)]:
            with self.subTest(name=name):
                self.setUp()
                self.array(name, shape, values)
                with self.assertRaises(ValueError):
                    reference.load_captured_inputs(self.probe, self.path)
        self.setUp()
        for value in [float("nan"), -0.1, 1.1, True, None]:
            with self.subTest(sigma=value), self.assertRaises(ValueError):
                reference.load_captured_inputs(dict(self.probe, video_sigma=value), self.path)

    def test_lora_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "LoRA"):
            reference.load_captured_inputs(dict(self.probe, use_reference_lora=True), self.path)

    def test_native_output_integrity_is_checked_before_reference_forward(self):
        self.probe["outputs"], self.probe["output_sha256"] = {}, {}
        for name in ("video", "audio"):
            self.probe["outputs"][name] = str(self.path.parent / (name + ".f32"))
            self.probe["output_sha256"][name] = self.probe["inputs"][name]["sha256"]
        reference.validate_captured_outputs(self.probe)
        self.probe["output_sha256"]["video"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            reference.validate_captured_outputs(self.probe)

    @unittest.skipUnless(os.environ.get("TS_LTX25_REFERENCE"), "Set TS_LTX25_REFERENCE to test unchanged official Modality mapping")
    def test_exact_official_modality_layout_mask_and_timestep_semantics(self):
        import numpy as np
        import torch
        sys.path.insert(0, str(Path(os.environ["TS_LTX25_REFERENCE"]) / "packages/ltx-core/src"))
        from ltx_core.model.transformer.modality import Modality
        payloads = reference.load_captured_inputs(self.probe, self.path)
        video, audio = reference.captured_modalities(self.probe, payloads, np, torch, Modality)
        self.assertEqual(tuple(video.latent.shape), (1, 2, 2))
        self.assertEqual(tuple(audio.context.shape), (1, 5, 4))
        torch.testing.assert_close(video.timesteps, torch.tensor([[0., 1.]]), rtol=0, atol=0)
        torch.testing.assert_close(audio.timesteps, torch.full((1, 3), 0.725), rtol=0, atol=0)
        torch.testing.assert_close(video.keyframes_mask, torch.tensor([[[0.], [1.]]]), rtol=0, atol=0)
        self.assertIsNone(audio.keyframes_mask)
        self.assertIsNone(video.attention_mask)
        self.assertIsNone(video.context_mask)
        expected = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2).transpose(1, 2)
        torch.testing.assert_close(video.positions, expected, rtol=0, atol=0)
        self.array("audio_timesteps", [1], [0.25])
        _, audio = reference.captured_modalities(self.probe, reference.load_captured_inputs(self.probe, self.path), np, torch, Modality)
        torch.testing.assert_close(audio.timesteps, torch.full((1, 3), 0.25), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
