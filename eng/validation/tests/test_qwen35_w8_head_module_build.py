import contextlib,hashlib,importlib.util,io,json,re,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('w8_head_module_builder',HERE/'build-qwen35-w8-head-module.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
class Tests(unittest.TestCase):
 def test_original_closure_all_twelve_bytes_and_modes(self):
  files=B.inventory();self.assertEqual(len(files),15);self.assertEqual(B.sha(B.NATIVE/'official-source-manifest.json'),B.CLOSURE_SHA)
 def test_only_original_global_instantiations_T1_through33_and_exact_cast(self):
  s=(B.NATIVE/'tensorsharp_w8g32_head.cu').read_text();self.assertEqual(list(map(int,re.findall(r'TS_W8_INSTANTIATE\((\d+)\)',s))),list(range(1,34)))
  self.assertIn('template __global__ void w8_small_t_mma_kernel<',s);self.assertEqual(s.count('extern "C" __global__'),1);self.assertIn('__float2bfloat16_rn(input[i])',s)
  for field in('sizeof(CUlaunchAttributeValue) == 64','sizeof(CUlaunchAttribute) == 72','sizeof(CUlaunchConfig) == 56','sizeof(W8ContiguousOutput) == 16','sizeof(W8SmallTMmaStoreEpilogue) == 1'):self.assertIn(field,s)
 def test_generate_only_no_native_no_marker(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   out=Path(t)/'generated'
   with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(B.main(['--out',str(out)]),0)
   state=json.loads((out/'manifest.json').read_text());self.assertFalse(state['complete']);self.assertEqual(state['status'],'GENERATED_ONLY_NATIVE_AND_GPU_NOT_RUN');self.assertNotIn('artifacts',state)
   for n,h in state['source'].items():self.assertEqual(B.sha(out/'source'/n),h)
 def test_paths_and_reuse_refused(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   p=Path(t)
   for bad in(p,p/'..'/'other',Path('/tmp/not-an-owned-module-output')):
    with self.assertRaises(ValueError):B.evidence(bad)
   (p/'alias').symlink_to(p,target_is_directory=True)
   with self.assertRaises(ValueError):B.evidence(p/'alias/out')
 def test_actual_resource_fields_required_and_spills_recorded(self):
  names=['ts_w8g32_cast_f32_bf16']+[f'original_w8_small_t_mma_kernel_T{n}'for n in range(1,34)]
  text=''.join("ptxas info : Compiling entry function '"+n+"'\nptxas info : Function properties\n 8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\nptxas info : Used 72 registers\n"for n in names)
  r=B.resources(text);self.assertEqual(len(r),34);self.assertEqual(r[names[0]]['spill_stores'],4)
  for bad in(text[:text.rfind('ptxas info : Compiling')],text+text[:text.index("ptxas info : Compiling entry function '",5)],text.replace('Used 72 registers','no actual fields',1)):
   with self.assertRaises(ValueError):B.resources(bad)
if __name__=='__main__':unittest.main()
