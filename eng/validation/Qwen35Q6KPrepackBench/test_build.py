import copy,contextlib,hashlib,importlib.util,io,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def load(name):
 s=importlib.util.spec_from_file_location('q6_CPU_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load('build');R=load('run')
DRIVER='''#pragma once
#include <cstddef>
using CUresult=int;using CUdevice=int;using CUmodule=void*;using CUfunction=void*;
constexpr int CUDA_SUCCESS=0;
enum{CU_FUNC_ATTRIBUTE_NUM_REGS,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE};
CUresult cuGetErrorString(CUresult,const char**);CUresult cuModuleGetFunction(CUfunction*,CUmodule,const char*);CUresult cuFuncGetAttribute(int*,int,CUfunction);CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int*,CUfunction,int,std::size_t);CUresult cuModuleLoad(CUmodule*,const char*);CUresult cuModuleUnload(CUmodule);CUresult cuDeviceGet(CUdevice*,int);CUresult cuDeviceGetAttribute(int*,int,CUdevice);CUresult cuInit(unsigned int);CUresult cuLaunchKernel(CUfunction,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,void*,void**,void**);
'''
RUNTIME='''#pragma once
#include <cstddef>
using cudaError_t=int;using cudaEvent_t=void*;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct cudaDeviceProp{char name[256];int multiProcessorCount;};
const char* cudaGetErrorString(cudaError_t);cudaError_t cudaMalloc(void**,std::size_t);cudaError_t cudaMemset(void*,int,std::size_t);cudaError_t cudaMemsetAsync(void*,int,std::size_t);cudaError_t cudaFree(void*);cudaError_t cudaMemcpy(void*,const void*,std::size_t,int);cudaError_t cudaDeviceSynchronize();cudaError_t cudaSetDevice(int);cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);cudaError_t cudaDriverGetVersion(int*);cudaError_t cudaRuntimeGetVersion(int*);cudaError_t cudaEventCreate(cudaEvent_t*);cudaError_t cudaEventRecord(cudaEvent_t,void* =nullptr);cudaError_t cudaEventSynchronize(cudaEvent_t);cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);cudaError_t cudaEventDestroy(cudaEvent_t);
'''
CPP='''#include "checks.h"
#include <cassert>
#include <iostream>
int main(){size_t tested=0;for(int rows:{1,63,65})for(int k:{256,512,5120}){
 std::vector<uint8_t> raw(Q6Check::RawBytes(rows,k),0);std::vector<int> logical(size_t(rows)*k);
 for(int r=0;r<rows;r++)for(int sb=0;sb<k/256;sb++){
  uint8_t* b=raw.data()+(size_t(r)*(k/256)+sb)*210;
  // Independent encoder: write all 64 possible unsigned codes across all half/segment/lane coordinates.
  for(int h=0;h<2;h++)for(int m=0;m<4;m++)for(int l=0;l<32;l++){
   int j=128*h+32*m+l,q=(r*7+sb*11+j)%64;logical[size_t(r)*k+sb*256+j]=q-32;
   b[h*64+(m%2)*32+l]|=uint8_t((q&15)<<(m>=2?4:0));b[128+h*32+l]|=uint8_t((q>>4)<<(2*m));
  }
  const int ss[]={-128,-127,-1,0,1,127};for(int j=0;j<16;j++)b[192+j]=uint8_t(ss[(r+sb+j)%6]);
  const uint16_t ds[]={0,0x8000,1,0x8001,0x03ff,0x0400,0x7bff,0x7c00,0xfc00,0x7e01,0xfe23};uint16_t d=ds[(r+sb)%11];std::memcpy(b+208,&d,2);
 }
 auto packed=Q6Check::Pack(raw,rows,k);Q6Check::Verify(raw,packed,rows,k);
 for(size_t i=0;i<logical.size();i++){assert(int((int8_t)packed[i])==logical[i]);tested++;}
 size_t count=size_t(rows)*k;for(int r=0;r<rows;r++)for(int sb=0;sb<k/256;sb++){
  const uint8_t* b=raw.data()+(size_t(r)*(k/256)+sb)*210;assert(std::memcmp(packed.data()+count+size_t(r)*(k/16)+sb*16,b+192,16)==0);assert(std::memcmp(packed.data()+count+size_t(rows)*(k/16)+(size_t(r)*(k/256)+sb)*2,b+208,2)==0);
 }
 for(size_t at:{size_t(0),count,count+size_t(rows)*(k/16)}){auto bad=packed;bad[at]^=1;bool rejected=false;try{Q6Check::Verify(raw,bad,rows,k);}catch(...){rejected=true;}assert(rejected);}
 auto bad=packed;std::swap(bad[0],bad[8]);bool rejected=false;try{Q6Check::Verify(raw,bad,rows,k);}catch(...){rejected=true;}assert(rejected);
 }std::cout<<"CPU logical elements checked="<<tested<<"; metadata bits and 36 mutations rejected\\n";}
'''
def fixture(bench=True,byte_planes=False,aligned_blocks=False):
 events=[dict(event='device',name='MOCK_NOT_GPU',sms=170,driver=13020,runtime=13020)]
 names={'ts_q35_quantize_q81_f32','ts_bench_q6_prepack','ts_q35_gemv_q6k_1','ts_bench_gemv_q6packed_1'}|{f'{p}_{n}'for p in('ts_q35_gemm_q6k','ts_bench_gemm_q6packed')for n in(1,2,4,8)}
 for s in sorted(names):events.append(dict(event='driver_resources',symbol=s,block_threads=256 if s in('ts_q35_quantize_q81_f32','ts_bench_q6_prepack')else 128,registers=64,local_bytes=0,static_shared_bytes=1024,active_ctas=1,max_block_threads=1024))
 def pack(name):
  fields=[int(x[1:])for x in name.split('-')];k,m,n,p=fields
  return dict(event='prepack_pass',shape=name,raw_bytes=m*(k//256)*210,packed_bytes=m*(k//256)*(210 if byte_planes else 212 if aligned_blocks else 274),extra_bytes=m*(k//256)*(0 if byte_planes else 2 if aligned_blocks else 64),load_once_gpu_ms=.1,prepack_GPU_ms_two_poison_passes=[.2,.1],load_time_scope='prepack_kernel_only_second_pass_not_full_model_load',whole_bytes_CPU_decoder=True,dual_poison=True,guards=True)
 for s in sorted(R.shapes(byte_planes,aligned_blocks)):events.extend([pack(s),dict(event='strict_pass',shape=s,full_bits=True,dual_poison_AA=True,guards=True,split_k=1)])
 if bench:
  for n in R.timed_ns(byte_planes,aligned_blocks):
   name=f'k5120-m248320-n{n}-p0';events.extend([pack(name),dict(event='pipeline_pass',shape=name,full_bits_AA_guards=True)])
   for r in range(10):events.append(dict(event='ABBA_gpu_ms',shape=name,n=n,round=r,A1_oracle_ms=1.,B1_candidate_ms=1.,B2_candidate_ms=1.,A2_oracle_ms=1.,fullbits_each_half=True,scope='original_Q81_quant_plus_zero_if_needed_plus_Q6_matmul',split_k=1))
 result=dict(complete=True,strict_pass=True,layout=R.layout(byte_planes,aligned_blocks),strict_cases=len(R.shapes(byte_planes,aligned_blocks)),timed_shapes=len(R.timed_ns(byte_planes,aligned_blocks))if bench else 0,rounds=10,iterations=3,cache_sweep=False,cache_sweeps=0)
 return events,result
class Tests(unittest.TestCase):
 def test_exact_source_oracle_and_math_expressions(self):
  source=B.SOURCE.read_text();prefix,q=B.section(source);s=B.generate(source);self.assertTrue(s.startswith('#include <cuda_runtime.h>\n'+prefix+q))
  for line in('acc[n] += ws * xd[(long long)n * (k >> 5) + (v >> 5)] * (float)ts_q35_dp4a(wq[m], xi, 0);','acc[nt][0] += (float)c[0] * s0 * x0;','acc[nt][3] += (float)c[3] * s1 * x1;'):
   self.assertEqual(s.count(line),2)
  with self.assertRaises(ValueError):B.generate(source.replace('acc[nt][0] +=','acc[nt][0] -=',1))
 def test_actual_host_independent_decode_and_mutation_gate(self):
  compiler=shutil.which('clang++')or shutil.which('g++')
  if not compiler:self.skipTest('host compiler unavailable; not CPU PASS')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'test.cpp').write_text(CPP);r=subprocess.run([compiler,'-std=c++17','-O2','-I',str(HERE),str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);r=subprocess.run([str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('36 mutations',r.stdout)
 def test_host_runner_syntax_with_CUDA_API_stubs_NOT_NVCC(self):
  compiler=shutil.which('clang++')or shutil.which('g++')
  if not compiler:self.skipTest('host compiler unavailable')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'cuda.h').write_text(DRIVER);(p/'cuda_runtime.h').write_text(RUNTIME)
   for defines in([],['-DTS_Q6_BYTE_PLANES=1'],['-DTS_Q6_ALIGNED_BLOCKS=1']):
    r=subprocess.run([compiler,'-std=c++17','-fsyntax-only','-Werror=format','-Werror=format-extra-args',*defines,'-I',str(p),'-x','c++',str(HERE/'bench.cu')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_byte_planes_actual_host_all_bytes_inverse_and_mutations(self):
  compiler=shutil.which('clang++')or shutil.which('g++')
  if not compiler:self.skipTest('host compiler unavailable')
  insert='''auto planes=Q6Check::BytePlanes(raw,rows,k);assert(planes.size()==raw.size());Q6Check::Verify(raw,planes,rows,k,true);size_t blocks=size_t(rows)*(k/256);std::vector<uint8_t> restore(raw.size());
 for(size_t b=0;b<blocks;b++)for(int at=0;at<210;at++){size_t index=at<128?b*128+at:at<192?blocks*128+b*64+at-128:at<208?blocks*192+b*16+at-192:blocks*208+b*2+at-208;restore[b*210+at]=planes[index];}assert(restore==raw);
 for(size_t at:{size_t(0),blocks*128,blocks*192,blocks*208}){auto bad=planes;bad[at]^=1;bool rejected=false;try{Q6Check::Verify(raw,bad,rows,k,true);}catch(...){rejected=true;}assert(rejected);}
 '''
  cpp=CPP.replace('auto packed=Q6Check::Pack(raw,rows,k);',insert+'auto packed=Q6Check::Pack(raw,rows,k);')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'test.cpp').write_text(cpp);r=subprocess.run([compiler,'-std=c++17','-O2','-I',str(HERE),str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);r=subprocess.run([str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_byte_planes_generator_only_changes_addresses_and_aligned_decode_loads(self):
  source=B.SOURCE.read_text();prefix,q=B.section(source);s=B.generate(source,True)
  self.assertTrue(s.startswith('#include <cuda_runtime.h>\n#define TS_Q6_BYTE_PLANES 1\n'+prefix+q))
  c=s[s.index('// Private byte-plane'):s.index('// Independent integer-only preprocessing.')]
  for value in('const uint32_t a = *(const uint32_t*)(ql + l0);','const uint32_t b = *(const uint32_t*)(ql + 32 + l0);','const uint32_t c = *(const uint32_t*)(qh + l0);'):
   self.assertEqual(c.count(value),2)
  self.assertEqual(c.count('acc[nt][0] += (float)c[0] * s0 * x0;'),1)
  for m in(1,63,65):
   for k in(256,512,5120):
    blocks=m*(k//256)
    for block in range(blocks):
     for h in(0,1):
      for l0 in range(0,32,4):
       for at in(block*128+h*64+l0,block*128+h*64+32+l0,blocks*128+block*64+h*32+l0):self.assertEqual(at%4,0)
 def test_byte_planes_extended_matrix_keeps_all_old91_and_rejects_wrong_layout_or_missing_tail(self):
  self.assertEqual(len(R.shapes()),91);self.assertEqual(len(R.shapes(True)),166);self.assertTrue(R.shapes()<R.shapes(True))
  e,r=fixture(byte_planes=True);self.assertEqual(R.validate(e,r,True,10,3,False,True)['ABBA'],100)
  with self.assertRaises(ValueError):R.validate(e,r,True,10,3,False)
  for n in(7,14,28,9,17,33):
   bad=[x for x in e if not(x.get('event')=='strict_pass'and f'-n{n}-'in x.get('shape',''))]
   with self.assertRaises(ValueError):R.validate(bad,r,True,10,3,False,True)
 def test_aligned_blocks_actual_host_all210_bytes_padding_zero_inverse_and_mutations(self):
  compiler=shutil.which('clang++')or shutil.which('g++')
  if not compiler:self.skipTest('host compiler unavailable')
  insert='''auto aligned=Q6Check::AlignedBlocks(raw,rows,k);size_t blocks=size_t(rows)*(k/256);assert(aligned.size()==blocks*212);Q6Check::Verify(raw,aligned,rows,k,false,true);std::vector<uint8_t> inverse(raw.size());
 for(size_t b=0;b<blocks;b++){std::memcpy(inverse.data()+b*210,aligned.data()+b*212,210);assert(aligned[b*212+210]==0&&aligned[b*212+211]==0);}assert(inverse==raw);
 for(size_t at:{size_t(0),size_t(128),size_t(192),size_t(208),size_t(210),size_t(211)}){auto bad=aligned;bad[at]^=1;bool rejected=false;try{Q6Check::Verify(raw,bad,rows,k,false,true);}catch(...){rejected=true;}assert(rejected);}
 '''
  cpp=CPP.replace('auto packed=Q6Check::Pack(raw,rows,k);',insert+'auto packed=Q6Check::Pack(raw,rows,k);')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'test.cpp').write_text(cpp);r=subprocess.run([compiler,'-std=c++17','-O2','-I',str(HERE),str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);r=subprocess.run([str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_aligned_source_reverse_restores_original_whole_q6_section(self):
  source=B.SOURCE.read_text();prefix,q=B.section(source);s=B.generate(source,False,True)
  marker='// Private byte-plane/212-stride loads and aligned integer decode only.\n';candidate=s.split(marker,1)[1].split('// Independent integer-only preprocessing.',1)[0]
  for name,offset in [('a','ql + l0'),('b','ql + 32 + l0'),('c','qh + l0')]:
   after=f'const uint32_t {name} = *(const uint32_t*)({offset});';before=f'const uint32_t {name} = *(const uint16_t*)({offset}) | ((uint32_t)*(const uint16_t*)({offset} + 2) << 16);';self.assertEqual(candidate.count(after),2);candidate=candidate.replace(after,before)
  candidate=candidate.replace('* 212;','* 210;').replace('ts_bench_gemv_q6packed','ts_q35_gemv_q6k').replace('ts_bench_gemm_q6packed','ts_q35_gemm_q6k').replace('ts_bench_mma_s8','ts_q35_mma_s8').replace('TS_BENCH_GEMV_Q6K','TS_Q35_GEMV_Q6K').replace('TS_BENCH_GEMM_Q6K','TS_Q35_GEMM_Q6K')
  self.assertEqual(candidate,q)
  for r in(0,1,62,64):
   for k in(256,512,5120):
    for sb in range(k//256):
     for h in(0,1):
      for l0 in range(0,32,4):
       for off in(64*h+l0,64*h+32+l0,128+32*h+l0):self.assertEqual(((r*(k//256)+sb)*212+off)%4,0)
  self.assertIn('v.packed&&AlignedBlocks?212:210',(HERE/'bench.cu').read_text());self.assertEqual(248320*(5120//256)*2,9932800);self.assertEqual(9932800/2**20,9.47265625)
 def test_aligned_matrix_and_mutual_layout_exclusion(self):
  e,r=fixture(aligned_blocks=True);self.assertEqual(R.validate(e,r,True,10,3,False,False,True)['ABBA'],100);self.assertEqual(R.shapes(False,True),R.shapes(True))
  with self.assertRaises(ValueError):R.validate(e,r,True,10,3,False,True)
  with self.assertRaises(ValueError):B.generate(B.SOURCE.read_text(),True,True)
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   out=Path(t)/'forbidden'
   with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):B.main(['--out',str(out),'--byte-planes','--aligned-blocks','--generate-only'])
   self.assertFalse(out.exists())
 def test_complete_mock_and_missing_duplicate_guard_rejected(self):
  e,r=fixture();self.assertEqual(R.validate(e,r,True,10,3,False)['strict'],91)
  for mode in('missing','duplicate','guard','pipeline','inf','late'):
   a=copy.deepcopy(e);i=next(i for i,x in enumerate(a)if x['event']=='strict_pass');j=next(i for i,x in enumerate(a)if x['event']=='ABBA_gpu_ms')
   if mode=='missing':a.pop(i)
   elif mode=='duplicate':a.append(a[i])
   elif mode=='guard':a[i]['guards']=False
   elif mode=='pipeline':a=[x for x in a if x['event']!='pipeline_pass']
   elif mode=='inf':a[j]['B1_candidate_ms']=float('inf')
   else:a.append(a.pop(i))
   with self.subTest(mode=mode),self.assertRaises(ValueError):R.validate(a,r,True,10,3,False)
 def test_model_byte_budget_and_actual_shape_source(self):
  m,k=248320,5120;self.assertEqual(m*k//256*210,1042944000);self.assertEqual(m*k//256*274,1360793600);self.assertEqual((274-210)*m*k//256/2**20,303.125)
  s=(HERE/'bench.cu').read_text();self.assertIn('Shape s{5120,248320,n,0}',s);self.assertIn('Quant();void* w=',s)
 def test_generate_only_no_compiler_and_no_permit(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   with patch.object(B,'command',side_effect=AssertionError('compiler called')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(B.main(['--out',str(Path(t)/'g'),'--generate-only']),0)
   m=json.loads((Path(t)/'g/manifest.json').read_text());self.assertFalse(m['complete']);self.assertEqual(m['gpu_quality'],'NOT_RUN');self.assertFalse((Path(t)/'g/offline-permit.json').exists())
   with patch.object(B,'command',side_effect=AssertionError('compiler called')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(B.main(['--out',str(Path(t)/'bp'),'--generate-only','--byte-planes']),0)
   m=json.loads((Path(t)/'bp/manifest.json').read_text());self.assertEqual(m['layout'],'byte_planes_210');self.assertEqual(m['strict_cases'],166);self.assertEqual(m['model_shape']['increment_bytes'],0)
   with patch.object(B,'command',side_effect=AssertionError('compiler called')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(B.main(['--out',str(Path(t)/'aligned'),'--generate-only','--aligned-blocks']),0)
   m=json.loads((Path(t)/'aligned/manifest.json').read_text());self.assertEqual(m['layout'],'aligned_blocks_212');self.assertEqual(m['strict_cases'],166);self.assertEqual(m['model_shape']['increment_bytes'],9932800)
 def test_default_plan_no_GPU_and_sweep_forces_one_iteration(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   with patch.object(R.B,'command',side_effect=AssertionError('GPU called')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(R.main(['--build',t,'--name','plan','--bench','--cache-sweep']),0)
   m=json.loads((Path(t)/'plan/run.json').read_text());self.assertFalse(m['complete']);self.assertFalse(m['gpu_invoked']);self.assertEqual(m['iterations'],1)
 def test_actual_resources_require_complete_and_no_spill(self):
  names=['ts_q35_quantize_q81_f32','ts_q35_gemv_q6k_1','ts_bench_gemv_q6packed_1','ts_bench_q6_prepack','ts_bench_q6_cache_sweep']+[f'{p}_{n}'for p in('ts_q35_gemm_q6k','ts_bench_gemm_q6packed')for n in(1,2,4,8)]
  log='\n'.join("ptxas info    : Compiling entry function '%s' for 'sm_120a'\nptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used 64 registers, 2048 bytes smem"%n for n in names)
  self.assertEqual(len(B.resources(log)),13)
  with self.assertRaises(ValueError):B.resources(log.replace('0 bytes spill stores','4 bytes spill stores',1))
if __name__=='__main__':unittest.main()
