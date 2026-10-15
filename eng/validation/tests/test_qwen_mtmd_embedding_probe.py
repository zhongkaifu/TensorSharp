"""Model-independent contract tests; no native/model/device initialization."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('qwen_mtmd_embedding_probe', Path(__file__).parents[1] / 'qwen-mtmd-embedding-probe.py')
TOOL = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TOOL)


class CaptureContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.plan = {'Rows': 2, 'Dimension': 3, 'Gpu': True, 'ImageToken': 7, 'ExpectedExpandedIds': [1,7,7,2]}
        self.native = {'Schema': 'qwen-mtmd-native-capture.v2', 'Completed': True, 'VocabularyOnly': False,
                       'ModelNoAlloc': True, 'ModelLoadMode': 'none', 'ModelLazyMode': 'off', 'LanguageModelExecuted': False,
                       'ImageRows': 2, 'Dimension': 3, 'GpuRequested': True, 'FlashAttentionRequested': True,
                       'Warmup': False, 'AllFinite': True, 'PerformanceQualified': False,
                       'AnswerQualityQualified': False, 'ImageStartToken': 1}
        TOOL.save(self.output / 'native-result.json', self.native)
        TOOL.save(self.output / 'expanded-ids.json', [1,7,7,2])
        TOOL.save(self.output / 'positions-thw.json', [0,0,0,1,1,1,1,1,2,3,3,3])
        TOOL.save(self.output / 'image-positions-thw.json', [1,1,1,1,1,2])
        (self.output / 'projected.f32').write_bytes(struct.pack('<6f', 1,2,3,4,5,6))

    def test_finite_exact_capture_is_execution_observation(self):
        capture = TOOL.inspect_payload(self.output, self.plan)
        self.assertTrue(capture['ExactExpandedInputIds'])
        self.assertFalse(capture['Native']['AnswerQualityQualified'])
        self.assertEqual(24, capture['Projection']['Bytes'])

    def test_nonfinite_payload_refused_even_when_native_claims_finite(self):
        (self.output / 'projected.f32').write_bytes(struct.pack('<6f', 1,2,3,4,float('nan'),6))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_truncated_projection_refused(self):
        (self.output / 'projected.f32').write_bytes(b'\0' * 20)
        with self.assertRaisesRegex(ValueError, 'byte count'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_different_expanded_input_refused(self):
        TOOL.save(self.output / 'expanded-ids.json', [1,7,7,3])
        with self.assertRaisesRegex(ValueError, 'expanded native IDs'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_wrong_position_slice_refused(self):
        TOOL.save(self.output / 'image-positions-thw.json', [1,1,2,1,1,1])
        with self.assertRaisesRegex(ValueError, 'position slice'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_wrong_native_policy_refused(self):
        self.native['LanguageModelExecuted'] = True; TOOL.save(self.output / 'native-result.json', self.native)
        with self.assertRaisesRegex(ValueError, 'metadata-only no-alloc'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_real_lm_weight_allocation_policy_refused(self):
        self.native['ModelNoAlloc'] = False; TOOL.save(self.output / 'native-result.json', self.native)
        with self.assertRaisesRegex(ValueError, 'metadata-only no-alloc'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_mmap_lm_load_policy_refused(self):
        self.native['ModelLoadMode'] = 'mmap'; TOOL.save(self.output / 'native-result.json', self.native)
        with self.assertRaisesRegex(ValueError, 'metadata-only no-alloc'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_wrong_native_geometry_refused(self):
        self.native['Dimension'] = 4; TOOL.save(self.output / 'native-result.json', self.native)
        with self.assertRaisesRegex(ValueError, 'policy/geometry'):
            TOOL.inspect_payload(self.output, self.plan)

    def test_pin_mutation_refused(self):
        path = self.output / 'pinned.txt'; path.write_text('old')
        pinned = TOOL.pin(path); path.write_text('new')
        with self.assertRaisesRegex(ValueError, 'changed'):
            TOOL.verify_pins([pinned])

    def test_existing_output_refused(self):
        with self.assertRaisesRegex(ValueError, 'Fresh'):
            TOOL.fresh(self.output)


if __name__ == '__main__':
    unittest.main()
