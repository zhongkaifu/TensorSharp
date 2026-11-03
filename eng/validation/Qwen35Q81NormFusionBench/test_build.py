import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('norm_fusion_build',HERE/'build.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

class SourceAndGeometry(unittest.TestCase):
    def test_original_norm_exact_reverse_gate(self):
        generated,norm=B.generate(B.SOURCE.read_text())
        self.assertEqual(B.sha(norm.encode()),B.NORM_SHA)
        self.assertIn('((volatile float*)y)[row * width + i] = xr[i] * scale * w[i];',generated)
        self.assertIn('((const volatile float*)y)[row * width + b * 32 + lane]',generated)
        self.assertEqual(generated.count(B.SYMBOL+'('),1)
        self.assertNotIn('ts_q35_add_rmsnorm_f32(',generated)
    def test_changed_norm_arithmetic_refused(self):
        source=B.SOURCE.read_text();start=source.index('extern "C" __global__ void ts_q35_add_rmsnorm_f32(')
        changed=source[:start]+source[start:].replace('ss += v * v;','ss += 0.5f * v * v;',1)
        with self.assertRaises(ValueError):B.generate(changed)
    def test_changed_helper_refused(self):
        with self.assertRaises(ValueError):B.generate(B.SOURCE.read_text().replace('for (int off = 16;','for (int off = 8;',1))
    def test_all_component_writes_unique_complete(self):
        for width in (5120,6144,17408):
            for rows in range(1,9):
                actual=B.geometry(rows,width)
                self.assertEqual(len(actual),rows*width)
                self.assertEqual(len(set(actual)),len(actual))
                self.assertEqual(set(actual),{(r,b,l)for r in range(rows)for b in range(width//32)for l in range(32)})
    def test_small_partial_warp_iteration_geometry(self):
        for width in (32,64,96,480,544,1056):
            values=B.geometry(1,width)
            self.assertEqual(len(values),width)
            self.assertEqual({b for _,b,_ in values},set(range(width//32)))
    def test_unsupported_geometry_refused(self):
        for rows,width in ((0,5120),(9,5120),(1,0),(1,33)):
            with self.assertRaises(ValueError):B.geometry(rows,width)
    def test_implicit_fastmath_refused_before_creation(self):
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as tmp:
            target=Path(tmp)/'new'
            with patch.dict(os.environ,{'NVCC_APPEND_FLAGS':'--use_fast_math'}):
                with self.assertRaises(SystemExit):B.main(['--out',str(target)])
            self.assertFalse(target.exists())
    def test_generate_only_has_no_compilation_or_gpu_permission(self):
        import json
        with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as tmp:
            target=Path(tmp)/'new'
            with patch.dict(os.environ,{'NVCC_PREPEND_FLAGS':'','NVCC_APPEND_FLAGS':''}):B.main(['--out',str(target)])
            data=json.loads((target/'build.json').read_text())
            self.assertFalse(data['complete']);self.assertFalse(data['compiled']);self.assertEqual(data['quality'],'GPU_NOT_RUN')
            self.assertFalse((target/'kernels.cubin').exists())

if __name__=='__main__':unittest.main()
