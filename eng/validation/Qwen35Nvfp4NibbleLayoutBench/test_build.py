import copy,hashlib,importlib.util,json,os,random,shutil,struct,subprocess,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load(HERE/'build.py','nibble_CPU_build');R=load(HERE/'run.py','nibble_CPU_run')
def pack_word(q):
 lo=(q&15)|((q>>4)&0xf0)|((q>>8)&0xf00)|((q>>12)&0xf000)
 hi=((q>>4)&15)|((q>>8)&0xf0)|((q>>12)&0xf00)|((q>>16)&0xf000)
 return lo|(hi<<16)
def reference(data):
 if len(data)%4:raise ValueError('whole words')
 result=bytearray(len(data))
 for word in range(len(data)//4):
  for j in range(4):
   byte=data[word*4+j];result[word*4+j//2]|=(byte&15)<<(4*(j%2));result[word*4+2+j//2]|=(byte>>4)<<(4*(j%2))
 return bytes(result)
def inverse(data):
 if len(data)%4:raise ValueError('whole words')
 result=bytearray(len(data))
 for word in range(len(data)//4):
  for j in range(4):result[word*4+j]=((data[word*4+j//2]>>(4*(j%2)))&15)|(((data[word*4+2+j//2]>>(4*(j%2)))&15)<<4)
 return bytes(result)
def perm(a,b,s):
 both=a|(b<<32);return sum(((both>>(((s>>(4*i))&7)*8))&255)<<(8*i)for i in range(4))
def six(q):
 table=(0x03020100,0x0c080604,0xfdfeff00,0xf4f8fafc);select=0x32103210|((q&0x88888888)>>1);out=[]
 for shift in(0,16):out.append(perm(perm(*table[:2],q>>shift),perm(*table[2:],q>>shift),select>>shift))
 return out

def prep(s,phase='strict',role='NOT_TIMING'):
 return dict(event='prepack_case_pass',phase=phase,plane_role=role,shape=R.name(s),codes_bytes=s['out_dim']*(s['k']//2),extra_candidate_allocation_bytes=s['out_dim']*(s['k']//2),steady_codes_bytes_delta=0,all_code_bytes=True,inverse_original_bytes=True,dual_poison_AA=True,raw_codes_unchanged=True,scales_unchanged=True,guards=True,outside_inference_timing=True,load_once_pass1_ms=.1,load_once_pass2_ms=.09)
def fixture(bench=True,cache=False):
 c=B.chosen('all');m=dict(arm_selection='all',candidates=c);events=[dict(event='device',name='CPU_MOCK_NOT_DEVICE',sms=170,registers_per_sm=65536,driver=13020,runtime=13020)]
 for v in [dict(symbol=B.ORACLE,rows_per_warp=4,block_threads=128),*c]:events.append(dict(event='driver_resources',symbol=v['symbol'],rows_per_warp=4,block_threads=128,registers=172,local_bytes=0,static_shared_bytes=0,max_block_threads=256,active_ctas=2,binary_version=120))
 for event,sym in [('quantizer_resources',B.QUANT),('prepack_resources',B.PACK)]:events.append(dict(event=event,symbol=sym,registers=32,local_bytes=0,static_shared_bytes=0,max_block_threads=1024,active_ctas=4,block_threads=256))
 events += [dict(event='production_rows_query',actual_rows_per_warp=4,expected_rows_per_warp=4,guards=True),prep(dict(k=64,out_dim=1,pattern=5,alpha_bits=0x80000000),'pilot'),dict(event='driver_launch_gate',launchable=True,full_quality_plan_passed=False)]
 for s in B.SEALED.plan():
  events.append(prep(s))
  events.append(dict(event='strict_case_pass',symbol=c[0]['symbol'],shape=R.name(s),rows_per_warp=4,block_threads=128,k=s['k'],out_dim=s['out_dim'],pattern=s['pattern'],alpha_bits=s['alpha_bits'],full_output_bits=True,oracle_AA=True,candidate_AA=True,guards=True,actual_output_argmax_equal=True,outputs=8*s['out_dim'],actual_output_argmax=[0]*8))
 geometry=B.sweep_geometry(100663296,170);sweeps=0
 if cache:events.append(dict(event='cache_sweep_setup',mode='all_data_l2_eviction_control',method='volatile_asm_ld_global_cg_u32_grid_stride_per_block_u64_sum_and_count',physical_cold_guaranteed=False,excluded_from_event_interval=True,registers=20,local_bytes=0,static_shared_bytes=128,active_ctas=6,max_block_threads=1024,**geometry))
 if bench:
  for s in B.TIMED:
   v=c[0];events.append(prep(s,'pipeline',s['role']));events.append(dict(event='pipeline_case_pass',symbol=v['symbol'],plane_role=s['role'],shape=R.name(s),Q81_dual_poison_AA=True,full_output_bits_AA_guards=True))
   for round in range(10):
    if cache:
     for half in('A1','B1','B2','A2'):
      sweeps+=1;events.append(dict(event='cache_sweep',index=sweeps,candidate=v['symbol'],measured_symbol=B.ORACLE if half.startswith('A')else v['symbol'],shape=R.name(s),plane_role=s['role'],round=round,half=half,driver_l2_bytes=geometry['driver_l2_bytes'],sweep_bytes=geometry['sweep_bytes'],words_read=geometry['words'],checksum=geometry['words']*B.SWEEP_VALUE,guards=True,checksum_verified=True,synchronized_before_event=True,excluded_from_event_interval=True))
    events.append(dict(event='ABBA_gpu_ms',symbol=v['symbol'],plane_role=s['role'],shape=R.name(s),round=round,rows_per_warp=4,block_threads=128,k=s['k'],out_dim=s['out_dim'],role=v['role'],scope='complete_original_q81_quant_plus_gemv',iterations_per_sample=1 if cache else 3,fullbits_each_half=True,A1_oracle_ms=1.,B1_candidate_ms=1.,B2_candidate_ms=1.,A2_oracle_ms=1.))
 result=dict(complete=True,status='STRICT_OUTPUT_BITS_AND_ARGMAX_PASS',activation_rows=8,strict_cases_per_candidate=96,strict_candidate_comparisons=96,candidates=1,timed_shapes_per_candidate=6 if bench else 0,timed_unique_geometries=5,scope='complete_original_q81_quant_plus_gemv',rounds=10,iterations_per_sample=1 if cache else 3,cache_sweep=cache,cache_mode='all_data_l2_eviction_control'if cache else'warm_repeated_matrix',cache_sweeps=sweeps,prepack_cases=103 if bench else 97,load_once_prepack_excluded_from_inference_events=True,A4_prefill_layout_compatible=False)
 return events,result,m
HOST_CUDA_STUB='#pragma once\n#include <cstddef>\nusing CUresult=int;using CUdevice=int;using CUmodule=void*;using CUfunction=void*;\nconstexpr int CUDA_SUCCESS=0;\nenum {CU_FUNC_ATTRIBUTE_NUM_REGS,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,CU_FUNC_ATTRIBUTE_BINARY_VERSION,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE};\nCUresult cuGetErrorString(CUresult,const char**);CUresult cuModuleGetFunction(CUfunction*,CUmodule,const char*);CUresult cuFuncGetAttribute(int*,int,CUfunction);CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int*,CUfunction,int,std::size_t);CUresult cuModuleLoad(CUmodule*,const char*);CUresult cuModuleUnload(CUmodule);CUresult cuDeviceGet(CUdevice*,int);CUresult cuDeviceGetAttribute(int*,int,CUdevice);CUresult cuInit(unsigned int);\nCUresult cuLaunchKernel(CUfunction,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,void*,void**,void**);\n'
HOST_RUNTIME_STUB='#pragma once\n#include <cstddef>\nusing cudaError_t=int;using cudaEvent_t=void*;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;\nstruct cudaDeviceProp {char name[256];int major,minor,multiProcessorCount,regsPerMultiprocessor;};\nconst char* cudaGetErrorString(cudaError_t);cudaError_t cudaMalloc(void**,std::size_t);cudaError_t cudaMemset(void*,int,std::size_t);cudaError_t cudaFree(void*);cudaError_t cudaMemcpy(void*,const void*,std::size_t,int);cudaError_t cudaDeviceSynchronize();cudaError_t cudaSetDevice(int);cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);cudaError_t cudaDriverGetVersion(int*);cudaError_t cudaRuntimeGetVersion(int*);cudaError_t cudaEventCreate(cudaEvent_t*);cudaError_t cudaEventRecord(cudaEvent_t,void* =nullptr);cudaError_t cudaEventSynchronize(cudaEvent_t);cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);cudaError_t cudaEventDestroy(cudaEvent_t);\n'
class Nibble(unittest.TestCase):
 def validate(self,events,result,m,bench=True,cache=False):return R.validate_observation(events,result,m,bench,10,1 if cache else 3,cache)
 def test_01_known_word_little_endian(self):self.assertEqual(pack_word(0x76543210),0x75316420);self.assertEqual(reference(bytes.fromhex('10325476')),bytes.fromhex('20643175'))
 def test_02_all_positions_all16_codes(self):
  for pos in range(8):
   for code in range(16):
    q=code<<(4*pos);data=struct.pack('<I',q);self.assertEqual(reference(data),struct.pack('<I',pack_word(q)));self.assertEqual(inverse(reference(data)),data)
 def test_03_exhaustive_16bit_halves_and_signs(self):
  for half in range(65536):
   q=half|(half^0xdead)<<16;old=six(q);old=(perm(old[0],old[1],0x6420),perm(old[0],old[1],0x7531));self.assertEqual(tuple(six(pack_word(q))),old)
 def test_04_independent_signed_E2_table(self):
  table=[0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
  for q in [0,0xffffffff,0x88888888,*[random.Random(i).getrandbits(32)for i in range(1000)]]:
   got=six(pack_word(q));actual=[x if x<128 else x-256 for v in got for x in struct.pack('<I',v)];expected=[table[(q>>(4*j))&15]for j in(0,2,4,6,1,3,5,7)];self.assertEqual(actual,expected)
 def test_05_multiple_rows_K_and_no_cross_word(self):
  for k,m in[(64,1),(128,17),(1984,5),(2112,3)]:
   raw=bytes((i*73+(i//4)*17)%256 for i in range(m*k//2));packed=reference(raw);self.assertEqual(len(packed),len(raw));self.assertEqual(inverse(packed),raw)
   for word in range(len(raw)//4):self.assertEqual(packed[word*4:word*4+4],struct.pack('<I',pack_word(struct.unpack_from('<I',raw,word*4)[0])))
 def test_06_each_bit_mutation_is_rejected(self):
  raw=bytes(range(64));packed=reference(raw)
  for at in range(0,len(packed),4):
   for bit in range(32):
    bad=bytearray(packed);bad[at+bit//8]^=1<<(bit%8);self.assertNotEqual(inverse(bad),raw)
 def test_07_negative_zero_raw_code_is_not_dropped(self):
  raw=bytes.fromhex('80888808');packed=reference(raw);self.assertEqual(inverse(packed),raw);self.assertNotEqual(packed,reference(bytes(4)))
 def test_08_swapped_words_and_wrong_permutation_rejected(self):
  raw=bytes(range(16));p=reference(raw);self.assertNotEqual(inverse(p[4:8]+p[:4]+p[8:]),raw);self.assertNotEqual(inverse(raw),raw)
 def test_09_layout_requires_whole_words(self):
  for n in(1,2,3,5,7):
   with self.assertRaises(ValueError):reference(bytes(n))
   with self.assertRaises(ValueError):inverse(bytes(n))
 def test_10_original_math_and_prefix_exact(self):
  source=B.SOURCE.read_text();generated=B.generate(source,B.CANDIDATES);prefix=B.SEALED.oracle_prefix(source);self.assertTrue(generated.startswith('#include <cuda_runtime.h>\n'+prefix));d,t,p=B.candidate_parts(source);self.assertEqual(t.replace('void ts_bench_nvfp4_nibble(','void ts_q35_gemv_nvfp4(').replace('ts_bench_e2m1x8_nibble(','ts_q35_e2m1x8('),B.F.template(source));self.assertEqual(d.count('__byte_perm('),3);self.assertEqual(B.decoder(source).count('__byte_perm('),5)
  for text in('acc[n][j] += ds[j] * xs * (float)sumi;','const float v = ts_q35_warp_sum(acc[n][j]);','ds[j] = ts_q35_ue4m3(dw[j] >> (8 * s));'):self.assertIn(text,t)
  self.assertNotIn('__launch_bounds__',generated);self.assertNotIn('maxrregcount',str(B.FLAGS))
 def test_11_source_quantizer_scale_mutations_rejected(self):
  source=B.SOURCE.read_text()
  for old,new in [('127.0f','126.0f'),('return 0.5f * v;','return v;'),('0x6420','0x6421'),('acc[n][j] += ds[j] * xs','acc[n][j] += xs * ds[j]')]:
   self.assertIn(old,source)
   with self.assertRaises(ValueError):B.generate(source.replace(old,new,1),B.CANDIDATES)
 def test_12_original_runner_reverse_exact(self):
  s=(HERE/'bench.cu').read_text();d=json.loads((HERE/'runner_delta.json').read_text())
  for x in reversed(d['deltas']):self.assertEqual(s.count(x['after']),1);s=s.replace(x['after'],x['before'],1)
  self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),d['old_source_sha256']);self.assertEqual(s,(HERE.with_name('Qwen35Dp4aLaunchGeometryBench')/'bench.cu').read_text())
 def test_13_real_host_CPP_pack_kernel_and_reference(self):
  cc=shutil.which('clang++')or shutil.which('g++')
  if not cc:self.skipTest('host C++ unavailable; not NVCC')
  kernel=json.loads((HERE/'kernel_delta.json').read_text())['pack_kernel'];source='#include <cstdint>\n#include <vector>\n#include <cstdio>\n#define __global__\n#define __restrict__\nstruct Dim{long long x;}; Dim blockIdx,threadIdx,blockDim;\n'+kernel+'\n#include "layout.h"\nint main(){ std::vector<uint8_t> raw(256*4);for(size_t i=0;i<raw.size();i++)raw[i]=uint8_t(i*73+i/4);std::vector<uint32_t>a(raw.size()/4),b(a.size());for(size_t i=0;i<a.size();i++)a[i]=uint32_t(raw[i*4])|(uint32_t(raw[i*4+1])<<8)|(uint32_t(raw[i*4+2])<<16)|(uint32_t(raw[i*4+3])<<24);blockDim.x=256;blockIdx.x=0;for(int i=0;i<257;i++){threadIdx.x=i;ts_bench_nvfp4_nibble_pack(a.data(),b.data(),a.size());}auto p=PackNibbleReference(raw);for(size_t i=0;i<p.size();i++)if(p[i]!=uint8_t(b[i/4]>>(8*(i%4))))return 1;if(UnpackNibbleReference(p)!=raw)return 2;return 0;}\n'
  with tempfile.TemporaryDirectory()as td:
   td=Path(td);(td/'test.cpp').write_text(source);code,out=B.command([cc,'-std=c++17','-O2','-I',str(HERE),str(td/'test.cpp'),'-o',str(td/'test')],60);self.assertEqual(code,0,out);code,out=B.command([str(td/'test')],60);self.assertEqual(code,0,out)
 def test_14_complete_strict_and_pipeline_counts(self):
  for bench in(False,True):e,r,m=fixture(bench);v=self.validate(e,r,m,bench);self.assertEqual(v['strict_comparisons'],96);self.assertEqual(v['prepack_cases'],103 if bench else 97);self.assertEqual(v['ABBA_events'],60 if bench else 0)
 def test_15_pack_payload_byte_inverse_poison_must_pass(self):
  e,r,m=fixture()
  for field in('all_code_bytes','inverse_original_bytes','dual_poison_AA','raw_codes_unchanged','scales_unchanged','guards','outside_inference_timing'):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']=='prepack_case_pass')[field]=False
   with self.assertRaises(ValueError):self.validate(bad,r,m)
 def test_16_missing_duplicate_pack_and_wrong_bytes_rejected(self):
  e,r,m=fixture();i=next(i for i,x in enumerate(e)if x['event']=='prepack_case_pass')
  for bad in(e[:i]+e[i+1:],e+[copy.deepcopy(e[i])]):
   with self.assertRaises(ValueError):self.validate(bad,r,m)
  for field,value in(('codes_bytes',16),('steady_codes_bytes_delta',4),('load_once_pass1_ms',0),('load_once_pass2_ms',float('nan'))):
   bad=copy.deepcopy(e);bad[i][field]=value
   with self.assertRaises(ValueError):self.validate(bad,r,m)
 def test_17_prepack_and_oracle_driver_spill_rejected(self):
  e,r,m=fixture()
  for kind,field,val in [('prepack_resources','local_bytes',4),('prepack_resources','active_ctas',0),('driver_resources','registers',256),('quantizer_resources','static_shared_bytes',1)]:
   bad=copy.deepcopy(e);next(x for x in bad if x['event']==kind)[field]=val
   with self.assertRaises(ValueError):self.validate(bad,r,m)
 def test_18_original_strict_missing_or_relaxed_cannot_pass(self):
  e,r,m=fixture()
  for field in('full_output_bits','candidate_AA','oracle_AA','guards','actual_output_argmax_equal'):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']=='strict_case_pass')[field]=False
   with self.assertRaises(ValueError):self.validate(bad,r,m)
  with self.assertRaises(ValueError):self.validate([x for x in e if x['event']!='strict_case_pass'],r,m)
 def test_19_pipeline_and_all_ABBA_halves_required(self):
  e,r,m=fixture();self.assertEqual(self.validate(e,r,m)['pipeline_comparisons'],6)
  for kind,field,value in [('pipeline_case_pass','Q81_dual_poison_AA',False),('ABBA_gpu_ms','fullbits_each_half',False),('ABBA_gpu_ms','B1_candidate_ms',0),('ABBA_gpu_ms','A1_oracle_ms',float('inf'))]:
   bad=copy.deepcopy(e);next(x for x in bad if x['event']==kind)[field]=value
   with self.assertRaises(ValueError):self.validate(bad,r,m)
  with self.assertRaises(ValueError):self.validate(e[:-1],r,m)
 def test_20_sweep_exact240_and_no_cold_claim(self):
  e,r,m=fixture(cache=True);v=self.validate(e,r,m,True,True);self.assertEqual(v['cache_sweeps'],240)
  for field,value in(('checksum',0),('synchronized_before_event',False),('excluded_from_event_interval',False)):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']=='cache_sweep')[field]=value
   with self.assertRaises(ValueError):self.validate(bad,r,m,True,True)
 def test_21_cpu_rowtail_geometry_and_no_extra_weight_bytes(self):
  for m in(1,7,15,16,17,16480,34817):
   grid,b,w=B.geometry(m,4,128);rows=[(i*4+warp)*4+j for i in range(grid)for warp in range(4)for j in range(4)if(i*4+warp)*4+j<m];self.assertEqual(rows,list(range(m)));self.assertEqual((b,w),(128,16))
  self.assertEqual(len(B.SEALED.plan()),96);self.assertEqual(len(B.TIMED),6);self.assertEqual(len({(x['k'],x['out_dim'])for x in B.TIMED}),5)
 def test_22_actual_runner_host_syntax_only(self):
  cc=shutil.which('clang++')or shutil.which('g++')
  if not cc:self.skipTest('host compiler unavailable; not NVCC')
  with tempfile.TemporaryDirectory()as td:
   td=Path(td);(td/'cuda.h').write_text(HOST_CUDA_STUB);(td/'cuda_runtime.h').write_text(HOST_RUNTIME_STUB);(td/'plan.h').write_text(B.SEALED.plan_header(B.SEALED.plan()));(td/'timed.h').write_text(B.timed_header());(td/'variants.h').write_text(B.variants_header(B.CANDIDATES));(td/'cache-sweep.h').write_bytes((HERE/'cache-sweep.h').read_bytes());(td/'layout.h').write_bytes((HERE/'layout.h').read_bytes());code,out=B.command([cc,'-std=c++17','-x','c++','-fsyntax-only','-I',str(td),str(HERE/'bench.cu')],60);self.assertEqual(code,0,out)
if __name__=='__main__':unittest.main()
