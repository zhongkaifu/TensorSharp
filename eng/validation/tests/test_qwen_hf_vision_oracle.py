"""Header/schema mapping checks only. Never import torch, HF, gguf, or a model."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('qwen_hf_oracle',Path(__file__).parents[1] / 'qwen-hf-vision-oracle.py')
TOOL = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TOOL)


class MappingContract(unittest.TestCase):
    def setUp(self):
        self.mapping = TOOL.directory_mapping(TOOL.CONFIG)
        self.directory = [{'name':row['Source'],'dimensions':list(reversed(row['SourceShape'])),
                           'type':0,'stored_bytes':TOOL.math.prod(row['SourceShape'])*4} for row in self.mapping]

    def test_exact_parameter_coverage_and_temporal_pair(self):
        self.assertEqual(334,len(self.mapping))
        self.assertEqual(333,len({row['Destination'] for row in self.mapping}))
        pair = [row for row in self.mapping if row['Destination'] == 'patch_embed.proj.weight']
        self.assertEqual(['v.patch_embd.weight','v.patch_embd.weight.1'],[row['Source'] for row in pair])
        self.assertEqual([0,1],[row['TemporalSlice'] for row in pair])
        self.assertEqual([1152,3,16,16],pair[0]['SourceShape'])
        by_name = TOOL.validate_directory(self.directory,self.mapping)
        self.assertEqual([4608,2560],by_name['mm.2.weight']['dimensions'])

    def test_unexpected_tensor_cannot_be_silently_dropped(self):
        self.directory.append({'name':'unexpected.weight','dimensions':[1],'type':0,'stored_bytes':4})
        with self.assertRaisesRegex(ValueError,'Missing/unexpected'):
            TOOL.validate_directory(self.directory,self.mapping)

    def test_missing_tensor_refused(self):
        with self.assertRaisesRegex(ValueError,'Missing/unexpected'):
            TOOL.validate_directory(self.directory[:-1],self.mapping)

    def test_duplicate_tensor_refused(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            TOOL.validate_directory(self.directory+[self.directory[0]],self.mapping)

    def test_shape_axis_swap_refused(self):
        row = next(row for row in self.directory if row['name'] == 'mm.2.weight')
        row['dimensions'].reverse()
        with self.assertRaisesRegex(ValueError,'shape mismatch'):
            TOOL.validate_directory(self.directory,self.mapping)

    def test_unsupported_quantization_refused(self):
        self.directory[0]['type'] = 1
        with self.assertRaisesRegex(ValueError,'F32/BF16'):
            TOOL.validate_directory(self.directory,self.mapping)

    def test_payload_extent_refused(self):
        self.directory[0]['stored_bytes'] -= 4
        with self.assertRaisesRegex(ValueError,'byte count'):
            TOOL.validate_directory(self.directory,self.mapping)

    def test_bf16_is_explicit_and_byte_counted(self):
        row = self.directory[0]; row['type'] = 30; row['stored_bytes'] //= 2
        TOOL.validate_directory(self.directory,self.mapping)


if __name__ == '__main__':
    unittest.main()
