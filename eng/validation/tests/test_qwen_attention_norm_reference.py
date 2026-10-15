import importlib.util
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('reference', Path(__file__).parents[1] / 'qwen-attention-norm-reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class ReferenceTests(unittest.TestCase):
    def test_position_zero_is_weighted_rms_norm_including_unrotated_tail(self):
        values, weights = [1., 2., 3., 4., 5., 6., 7., 8.], [1., 2., 1., 2.]
        for mode in ('fp64', 'fp32_sequential'):
            result, normalized = ref.transform(values, weights, 0, 2, 10000., 1., 1e-6, mode)
            self.assertEqual(result, normalized)
            self.assertAlmostEqual(result[0], 1 / math.sqrt(7.5 + 1e-6), places=6)
            self.assertAlmostEqual(result[1], 4 / math.sqrt(7.5 + 1e-6), places=6)

    def test_neox_pairs_half_rotary_dimensions_not_adjacent_channels(self):
        result, normalized = ref.transform([1., 2., 3., 4., 5., 6.], [1.] * 6, 1, 4, 10000., 1., 0., 'fp64')
        self.assertAlmostEqual(result[0], normalized[0] * math.cos(1) - normalized[2] * math.sin(1))
        self.assertAlmostEqual(result[2], normalized[0] * math.sin(1) + normalized[2] * math.cos(1))
        self.assertAlmostEqual(result[1], normalized[1] * math.cos(.01) - normalized[3] * math.sin(.01))
        self.assertEqual(result[4:], normalized[4:])

    def test_each_head_has_independent_rms(self):
        result, _ = ref.transform([1., 1., 4., 4.], [1., 2.], 0, 2, 10000., 1., 0., 'fp64')
        self.assertEqual(result, [1., 2., 1., 2.])

    def test_f32_sum_is_explicit_sequential_rounding(self):
        inputs = [4096., 1., 1., 1.]
        actual, _ = ref.transform(inputs, [1.] * 4, 0, 2, 10000., 1., 0., 'fp32_sequential')
        total = 0.
        for value in inputs:
            total = ref.f32(total + ref.f32(value * value))
        scale = ref.f32(1 / math.sqrt(ref.f32(total / 4)))
        self.assertEqual(actual, [ref.f32(scale * value) for value in inputs])

    def test_invalid_geometry_and_nonfinite_are_rejected(self):
        for values, weights, rotary in [([1., 2.], [1., 1.], 3), ([1.], [1., 1.], 2), ([math.nan, 1.], [1., 1.], 2)]:
            with self.assertRaises(ValueError):
                ref.transform(values, weights, 1, rotary, 10000., 1., 1e-6, 'fp64')

    def test_half_nearest_even_and_coordinate_reporting(self):
        midpoint = 1 + 2 ** -11
        self.assertEqual(ref.half_word(midpoint), ref.half_word(1.))
        report = ref.compare([ref.half_word(1.), ref.half_word(2.001953125)], [midpoint, 2.], 2, 2)
        self.assertEqual(report['different_half_elements'], 1)
        self.assertEqual(report['max_half_ulp_distance'], 1)
        self.assertEqual(report['max_absolute_half_difference'], 0.001953125)
        self.assertEqual(report['different_coordinates'][0]['dimension'], 1)
        self.assertEqual(report['different_coordinates'][0]['head'], 0)

    def test_nonfinite_half_is_not_a_pass(self):
        with self.assertRaises(ValueError):
            ref.compare([0x7c00], [1.], 1, 0)

    def test_signed_zero_retains_bit_difference(self):
        result = ref.compare([0x8000], [0.], 1, 0)
        self.assertFalse(result['half_bit_exact'])
        self.assertEqual(result['max_half_ulp_distance'], 0)

    def test_capture_hash_and_size_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'k.bin'
            path.write_bytes(b'ab')
            with self.assertRaises(ValueError):
                ref.verify_capture({'path': str(path), 'size': 2, 'fnv1a64': '0'})
            with self.assertRaises(ValueError):
                ref.verify_capture({'path': str(path), 'size': 3, 'fnv1a64': '0'})

    def test_truncated_header_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 0))
            with self.assertRaises(ValueError):
                ref.headers(path, {'blk.39.attn_k_norm.weight'})

    def test_invalid_shard_count_is_rejected_before_file_access(self):
        for count in ('00000', '99999'):
            with self.assertRaisesRegex(ValueError, 'shard count'):
                ref.norms(f'does-not-exist-00001-of-{count}.gguf', 39)

    def test_small_real_format_header_exports_exact_norm_bytes(self):
        def string(value):
            raw = value.encode('utf-8')
            return struct.pack('<Q', len(raw)) + raw
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            metadata = string('qwen4exp.attention.key_length') + struct.pack('<II', 4, 2)
            names = [f'blk.39.attn_{part}_norm.weight' for part in ('k', 'q')]
            descriptors = b''.join(string(name) + struct.pack('<IQIQ', 1, 2, 0, offset)
                                   for name, offset in zip(names, (0, 32)))
            header = b'GGUF' + struct.pack('<IQQ', 3, 2, 1) + metadata + descriptors
            padding = bytes((-len(header)) % 32)
            k, q = struct.pack('<ff', 1., 2.), struct.pack('<ff', 3., 4.)
            path.write_bytes(header + padding + k + bytes(24) + q)
            _, found, weights, _ = ref.norms(path, 39)
            self.assertEqual(weights[names[0]], k)
            self.assertEqual(weights[names[1]], q)
            self.assertEqual(found[names[0]]['absolute_offset'], len(header + padding))
            self.assertEqual(found[names[1]]['sha256'], hashlib.sha256(q).hexdigest())
            self.assertEqual(ref.sha(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
