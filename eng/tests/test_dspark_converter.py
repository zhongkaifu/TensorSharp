#!/usr/bin/env python3
"""Small real safetensors -> GGUF checks; no downloaded weights or GPU required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(os.environ.get("DS_CONVERTER", Path(__file__).resolve().parents[1] / "dsv4-dspark-to-gguf.py"))
spec = importlib.util.spec_from_file_location("dspark_converter", SOURCE)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def fixture(v41=True, expert_dtype="I8"):
    cfg = dict(dspark_block_size=5, dspark_noise_token_id=31,
               dspark_target_layer_ids=[1, 2, 3], dspark_markov_rank=32,
               num_hidden_layers=4, num_nextn_predict_layers=2,
               vocab_size=32, n_routed_experts=2, num_experts_per_tok=1)
    if v41:
        cfg.update(model_type="deepseek_v41_text", n_routed_experts=6,
                   num_experts_per_tok=3, dspark_n_routed_experts=2,
                   dspark_num_experts_per_tok=1)
    raw = dict(model_type="deepseek_v41", text_config=cfg,
               quantization_config={"weight_block_size": [32, 32]}) if v41 else cfg
    tensors = {}

    def add(name, shape, dtype="F32", value=1):
        native = {"F32": np.float32, "I8": np.int8, "F8_E8M0": np.uint8}[dtype]
        values = np.full(shape, value, dtype=native)
        tensors[name] = (values.tobytes(), dtype, shape)

    for stage in range(2):
        prefix = f"mtp.{stage}."
        for name in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
                     "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"):
            add(prefix + name + ".weight", [32, 32])
        for name in ("attn.q_norm.weight", "attn.kv_norm.weight", "attn.attn_sink",
                     "attn_norm.weight", "ffn_norm.weight", "hc_attn_fn", "hc_attn_scale",
                     "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            add(prefix + name, [32])
        add(prefix + "ffn.gate.weight", [2, 32])
        add(prefix + "ffn.gate.bias", [2])
        if v41:
            add(prefix + "ffn.gate.bias_vl", [2], value=2)
        for e in range(2):
            for w in ("w1", "w2", "w3"):
                name = prefix + f"ffn.experts.{e}.{w}"
                # Packed byte 0x21 represents .5 followed by 1 in source order.
                add(name + ".weight", [2, 128 if expert_dtype == "I8" else 256],
                    expert_dtype, value=0x21 if expert_dtype == "I8" else .5)
                if expert_dtype == "I8":
                    add(name + ".scale", [2, 8], "F8_E8M0", 127)
    add("mtp.0.main_norm.weight", [32])
    add("mtp.0.main_proj.weight", [32, 96])
    add("mtp.1.norm.weight", [32])
    add("mtp.1.confidence_head.proj.weight", [1, 64])
    if not v41:
        for name in ("hc_head_fn", "hc_head_scale", "hc_head_base"):
            add("mtp.1." + name, [32])
    for name in (("embed", "head") if v41 else ("markov_w1", "markov_w2")):
        add(f"mtp.1.markov_head.{name}.weight", [32, 32])
    return raw, tensors


def write_checkpoint(directory, config, tensors):
    (directory / "config.json").write_text(json.dumps(config))
    header, payload = {}, bytearray()
    for name, (data, dtype, shape) in tensors.items():
        begin = len(payload)
        payload += data
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
    raw = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))


def read_gguf(path):
    """Independent, deliberately narrow GGUF parser for these emitted types."""
    data = path.read_bytes()
    pos = 0

    def take(fmt):
        nonlocal pos
        result = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return result[0] if len(result) == 1 else result

    def string():
        nonlocal pos
        size = take("Q")
        value = data[pos:pos + size].decode()
        pos += size
        return value

    assert take("II") == (0x46554747, 3)
    nt, nk = take("QQ")
    metadata = {}
    for _ in range(nk):
        key, kind = string(), take("I")
        if kind == 8:
            value = string()
        elif kind == 4:
            value = take("I")
        elif kind == 9:
            assert take("I") == 5
            value = [take("i") for _ in range(take("Q"))]
        else:
            raise AssertionError(kind)
        metadata[key] = value
    tensors = {}
    for _ in range(nt):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        kind, offset = take("IQ")
        tensors[name] = dict(dims=dims, kind=kind, offset=offset)
    base = (pos + 31) // 32 * 32
    for value in tensors.values():
        block, size = {0: (1, 4), 8: (32, 34), 10: (256, 84), 30: (1, 2), 39: (32, 17)}[value["kind"]]
        assert value["dims"][0] % block == 0
        size = int(np.prod(value["dims"])) // block * size
        value["payload"] = data[base + value["offset"]:base + value["offset"] + size]
        assert len(value["payload"]) == size
    return metadata, tensors


class MemoryTensors:
    def __init__(self, tensors, block=(128, 128)):
        self.tensors = tensors
        self.fp8_block_shape = block

    def raw(self, name):
        return self.tensors[name]

    def has(self, name):
        return name in self.tensors


class DSparkConverterTests(unittest.TestCase):
    def convert(self, config, tensors, expert_type="mxfp4"):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            write_checkpoint(directory, config, tensors)
            output = directory / "draft.gguf"
            result = subprocess.run([sys.executable, str(SOURCE), "--checkpoint", str(directory),
                                     "--out", str(output), "--expert-type", expert_type],
                                    text=True, capture_output=True, timeout=30)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            return read_gguf(output)

    def test_legacy_v4_artifact_still_converts(self):
        metadata, tensors = self.convert(*fixture(False))
        self.assertEqual("deepseek4-dspark", metadata["general.architecture"])
        self.assertIn("mtp.1.hc_head_fn.weight", tensors)
        self.assertEqual(39, tensors["mtp.0.ffn_gate_exps.weight"]["kind"])

    def test_v41_nested_config_uses_draft_experts_and_distinct_architecture(self):
        metadata, tensors = self.convert(*fixture())
        self.assertEqual("deepseek41-dspark", metadata["general.architecture"])
        self.assertEqual("deepseek41", metadata["dspark.target_architecture"])
        self.assertEqual(2, metadata["dspark.expert_count"])
        self.assertEqual(1, metadata["dspark.expert_used_count"])
        self.assertEqual([1, 2, 3], metadata["dspark.target_layer_ids"])
        self.assertEqual(2, metadata["dspark.stage_count"])
        self.assertEqual([256, 2, 2], tensors["mtp.0.ffn_gate_exps.weight"]["dims"])
        self.assertFalse(any("hc_head" in name for name in tensors))
        self.assertIn("mtp.1.markov_head.markov_w1.weight", tensors)
        self.assertIn("mtp.1.markov_head.markov_w2.weight", tensors)

    def test_mxfp4_payload_preserves_source_nibble_values(self):
        _, tensors = self.convert(*fixture(False))
        tensor = tensors["mtp.0.ffn_gate_exps.weight"]
        blocks = np.frombuffer(tensor["payload"], np.uint8).reshape(-1, 17)
        self.assertTrue(np.all(blocks[:, 0] == 127))
        expected = np.tile([0x11, 0x22], 8)
        np.testing.assert_array_equal(blocks[:, 1:], np.tile(expected, (32, 1)))

    def test_floating_experts_tag_actual_q8_payload(self):
        _, tensors = self.convert(*fixture(False, "F32"))
        tensor = tensors["mtp.0.ffn_gate_exps.weight"]
        self.assertEqual(8, tensor["kind"])
        blocks = np.frombuffer(tensor["payload"], np.uint8).reshape(-1, 34)
        scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)
        values = blocks[:, 2:].view(np.int8) * scales
        np.testing.assert_allclose(values, .5, rtol=.001, atol=0)

    def test_q2_k_tags_and_lengths(self):
        for dtype in ("I8", "F32"):
            with self.subTest(dtype=dtype):
                _, tensors = self.convert(*fixture(False, dtype), "q2_k")
                tensor = tensors["mtp.0.ffn_gate_exps.weight"]
                self.assertEqual(10, tensor["kind"])
                self.assertEqual(336, len(tensor["payload"]))

    def test_mixed_expert_encodings_are_rejected(self):
        config, tensors = fixture(False)
        tensors["mtp.0.ffn.experts.1.w1.weight"] = (np.ones((2, 256), np.float32).tobytes(), "F32", [2, 256])
        with self.assertRaisesRegex(ValueError, "same shape and encoding"):
            converter.stage_tensors(MemoryTensors(tensors), converter.GgufWriter("unused"),
                                    0, config, "mxfp4", lambda _: None)

    def test_mismatched_expert_shapes_are_rejected(self):
        config, tensors = fixture(False, "F32")
        tensors["mtp.0.ffn.experts.1.w1.weight"] = (np.ones((3, 256), np.float32).tobytes(), "F32", [3, 256])
        with self.assertRaisesRegex(ValueError, "same shape and encoding"):
            converter.stage_tensors(MemoryTensors(tensors), converter.GgufWriter("unused"),
                                    0, config, "mxfp4", lambda _: None)

    def test_partial_fp8_scale_tiles_keep_fixed_boundaries(self):
        for block in (32, 128):
            with self.subTest(block=block):
                shape = [block + 1, block + 1]
                values = np.full(shape, 0x38, np.uint8)  # exact FP8 value 1
                scales = np.array([[127, 128], [129, 130]], np.uint8)
                st = MemoryTensors({"x.weight": (values.tobytes(), "F8_E4M3", shape),
                                    "x.scale": (scales.tobytes(), "F8_E8M0", [2, 2])}, (block, block))
                actual = converter.dequant_block_scaled(st, "x.weight")
                expected = np.ones(shape, np.float32)
                expected[:, block:] *= 2
                expected[block:, :] *= 4
                np.testing.assert_array_equal(expected, actual)

    def test_fp8_missing_scale_is_rejected(self):
        st = MemoryTensors({"x.weight": (bytes([0x38]) * 1024, "F8_E4M3", [32, 32])})
        with self.assertRaisesRegex(ValueError, "no block scales"):
            converter.dequant_block_scaled(st, "x.weight")

    def test_fp8_wrong_scale_shape_is_rejected(self):
        st = MemoryTensors({"x.weight": (bytes([0x38]) * 1024, "F8_E4M3", [32, 32]),
                            "x.scale": (bytes([127, 128]), "F8_E8M0", [1, 2])}, (32, 32))
        with self.assertRaisesRegex(ValueError, "scale shape"):
            converter.dequant_block_scaled(st, "x.weight")

    def test_missing_or_extra_experts_are_rejected_before_payload_read(self):
        config, tensors = fixture()
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                index = dict.fromkeys(tensors, "missing.safetensors")
                if mutation == "missing":
                    del index["mtp.0.ffn.experts.1.w1.weight"]
                else:
                    index["mtp.0.ffn.experts.2.w1.weight"] = "missing.safetensors"
                with self.assertRaisesRegex(ValueError, "expert IDs"):
                    converter.resolve_dspark_config(config, index)

    def test_stage_gaps_are_rejected(self):
        config, tensors = fixture()
        del tensors["mtp.0.attn_norm.weight"]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            converter.resolve_dspark_config(config, tensors)

    def test_stage_count_mismatch_is_rejected(self):
        config, tensors = fixture()
        config["text_config"]["num_nextn_predict_layers"] = 3
        with self.assertRaisesRegex(ValueError, "stage count"):
            converter.resolve_dspark_config(config, tensors)

    def test_invalid_draft_metadata_is_rejected(self):
        config, tensors = fixture()
        changes = [("dspark_block_size", 0), ("dspark_block_size", True),
                   ("dspark_markov_rank", -1), ("dspark_noise_token_id", 32),
                   ("dspark_target_layer_ids", []), ("dspark_target_layer_ids", [2, 1]),
                   ("dspark_target_layer_ids", [1, 1]), ("dspark_target_layer_ids", [4]),
                   ("dspark_n_routed_experts", 0), ("dspark_num_experts_per_tok", 3)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(config)
                changed["text_config"][key] = value
                with self.assertRaises(ValueError):
                    converter.resolve_dspark_config(changed, tensors)

    def test_writer_rejects_encoding_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "payload"):
            converter.GgufWriter("unused").add_tensor("expert", [32, 1], 39, bytes(34))

    def test_writer_rejects_cross_row_quantization(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            converter.GgufWriter("unused").add_tensor("projection", [16, 2], 8, bytes(34))


if __name__ == "__main__":
    unittest.main(verbosity=2)
