import contextlib,io,copy,hashlib,importlib.util,json,math,os,shutil,struct,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('w8_run_tests',HERE/'run.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);B=R.B
CPP=r'''#include "reference.h"
#include <cassert>
#include <iostream>
int main(){size_t checked=0,mutations=0;
 for(int rows:{1,15,16,17,33})for(int k:{32,64,160,512}){
  W8Layout l(rows,k);assert(l.kpad%128==0&&l.scales_at%256==0);
  std::vector<uint8_t>w(l.total,0x5a);std::vector<int> visits(l.total,0);
  for(int r=0;r<rows;r++)for(unsigned c=0;c<l.kpad;c++){
   size_t flat=size_t(r)*l.kpad+c;w[flat]=uint8_t(flat%256);assert(l.Code(r,c)==flat);visits[l.Code(r,c)]++;
   assert(SignedCode(w[flat])==(int(flat%256)<128?int(flat%256):int(flat%256)-256));checked++;
  }
  for(int r=0;r<rows;r++)for(unsigned g=0;g<l.groups;g++){
   uint16_t bits=uint16_t((r*l.groups+g)%0x7c00);size_t at=l.scales_at+2*(size_t(r)*l.groups+g);
   w[at]=uint8_t(bits);w[at+1]=uint8_t(bits>>8);assert(l.Scale(r,g)==at&&ReadHalf(w,at)==bits);visits[at]++;visits[at+1]++;
  }
  for(size_t at=0;at<w.size();at++)assert(visits[at]==int(at<l.codes||at>=l.scales_at));
  std::vector<uint16_t>x(k);for(int c=0;c<k;c++)x[c]=Bf16(float(c%7-3));
  std::vector<double>ref,ab;std::vector<uint16_t>f32;CpuReference(w,l,x,1,ref,ab,f32);
  for(int r=0;r<rows;r++){double z=0,a=0;for(int c=0;c<k;c++){
    auto at=l.scales_at+2*(size_t(r)*(l.kpad/32)+c/32);uint16_t h=w[at]|uint16_t(w[at+1])<<8;
    // Independent flat encoder interpretation, without Layout.Code/Scale helpers.
    double p=double(int8_t(w[size_t(r)*l.kpad+c]))*HalfValue(h)*double(BValue(x[c]));z+=p;a+=std::abs(p);
   }assert(ref[r]==z&&ab[r]==a);
  }
  std::vector<int8_t> logical(size_t(rows)*l.kpad);std::vector<uint16_t> scaleBits(size_t(rows)*l.groups);
  for(size_t i=0;i<logical.size();i++)logical[i]=int8_t(uint8_t(i%256));
  for(size_t i=0;i<scaleBits.size();i++)scaleBits[i]=uint16_t(i%0x7c00);
  VerifyLayout(w,l,logical,scaleBits);std::vector<int8_t> dc;std::vector<uint16_t> ds;DecodeLayout(w,l,dc,ds);assert(dc==logical&&ds==scaleBits);
  // Independent inverse emits exactly the input planes, including original align padding.
  auto restored=w;for(size_t i=0;i<dc.size();i++)restored[i]=uint8_t(dc[i]);for(size_t i=0;i<ds.size();i++){restored[l.scales_at+2*i]=uint8_t(ds[i]);restored[l.scales_at+2*i+1]=uint8_t(ds[i]>>8);}assert(restored==w);
  for(auto at:{uint64_t(0),l.scales_at,l.total-1}){auto bad=w;bad[at]^=1;bool reject=false;try{VerifyLayout(bad,l,logical,scaleBits);}catch(...){reject=true;}assert(reject);mutations++;}
 }
 for(unsigned h=0;h<65536;h++){
  if((h&0x7c00)==0x7c00){bool fail=false;try{HalfValue(h);}catch(...){fail=true;}assert(fail);continue;}
  auto v=HalfValue(h);assert(std::isfinite(v));if(!(h&0x7fff))assert(std::signbit(v)==bool(h&0x8000));
 }
 assert(Bf16(1.00390625f)==0x3f80);assert(Bf16(1.01171875f)==0x3f82);assert(Bf16(-0.0f)==0x8000);
 for(float bad:{FValue(0x7f800000),FValue(0x7fc00001),FValue(0x00010000),FValue(0x7f7fffff)}){bool f=false;try{Bf16(bad);}catch(...){f=true;}assert(f);}
 for(unsigned b=0;b<65536;b++){if((b&0x7f80)==0x7f80)continue;assert(RoundBf16Reference(BValue(b))==b);}
 for(unsigned b=0x0080;b<0x7f7f;b+=31){double a=BValue(b),z=BValue(b+1),mid=(a+z)/2;assert(RoundBf16Reference(mid)==((b&1)?b+1:b));}
 std::vector<double> ref={1,-2,0},ab={1,2,0};std::vector<uint16_t>good={0x3f80,0xc000,0};CheckOutput(good,ref,ab,5120,true);
 for(int i=0;i<3;i++)for(auto v:{uint16_t(0x7fc0),uint16_t(0x7f80),uint16_t(0x4000)}){auto bad=good;bad[i]=v;bool rejected=false;try{CheckOutput(bad,ref,ab,5120,true);}catch(...){rejected=true;}assert(rejected);mutations++;}
 auto bad=good;bad[2]=0x8000;bool rejected=false;try{CheckOutput(bad,ref,ab,5120,true);}catch(...){rejected=true;}assert(rejected);mutations++;
 assert(Allowed(10,30,5120)>0.039&&Allowed(10,30,5120)<0.041);
 std::cout<<"logical_codes="<<checked<<" mutations="<<mutations<<" half_domain=65536 BF16_roundtrip=65280\n";
}'''
RUNTIME=r'''#pragma once
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#define __global__ __attribute__((global))
using cudaError_t=int;using cudaEvent_t=void*;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaDevAttrL2CacheSize=3;
struct dim3{unsigned x,y,z;constexpr dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
struct cudaUUID_t{char bytes[16];};struct cudaDeviceProp{char name[256];cudaUUID_t uuid;int multiProcessorCount;};
struct cudaFuncAttributes{std::size_t sharedSizeBytes,localSizeBytes;int maxThreadsPerBlock,numRegs;};
const char*cudaGetErrorString(cudaError_t);cudaError_t cudaFree(void*);cudaError_t cudaMemset(void*,int,std::size_t);cudaError_t cudaMemcpy(void*,const void*,std::size_t,int);cudaError_t cudaDeviceSynchronize();cudaError_t cudaGetLastError();cudaError_t cudaDeviceGetAttribute(int*,int,int);cudaError_t cudaGetDevice(int*);cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);cudaError_t cudaDriverGetVersion(int*);cudaError_t cudaRuntimeGetVersion(int*);cudaError_t cudaEventCreate(cudaEvent_t*);cudaError_t cudaEventRecord(cudaEvent_t,void*=nullptr);cudaError_t cudaEventSynchronize(cudaEvent_t);cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);cudaError_t cudaEventDestroy(cudaEvent_t);
template<class T>cudaError_t cudaMalloc(T**,std::size_t);cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*,const void*);cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,const void*,int,std::size_t);cudaError_t cudaLaunchKernel(const void*,dim3,dim3,void**,std::size_t=0,void*=nullptr);
extern "C" cudaError_t cudaConfigureCall(dim3,dim3,std::size_t=0,void*=nullptr);extern "C" cudaError_t cudaSetupArgument(const void*,std::size_t,std::size_t);extern "C" cudaError_t cudaLaunch(const void*);
'''.replace('void*=','void* =')
STUBS=r'''#pragma once
#include "cuda_runtime.h"
struct __nv_bfloat16{uint16_t x;};struct __half{uint16_t x;};
extern "C" __global__ void ts_w8_cast(const float*,__nv_bfloat16*,int64_t);
extern "C" __global__ void ts_w8_reference(const int8_t*,const __half*,const __nv_bfloat16*,double*,double*,int,int,int,int);
extern "C" __global__ void ts_w8_simt(const int8_t*,const __half*,const __nv_bfloat16*,__nv_bfloat16*,int,int,int,int);
extern "C" __global__ void ts_w8_sweep(const uint32_t*,uint64_t,uint64_t*);
#define P(N) extern "C" __global__ void ts_w8_tc_##N(const int8_t*,const __half*,const __nv_bfloat16*,__nv_bfloat16*,int,int,int);
P(1) P(7) P(8) P(14) P(16) P(28) P(32) P(33)
#undef P
cudaError_t OfficialAttributes(int,cudaFuncAttributes*,int*,int*);cudaError_t LaunchOfficial(int,const __nv_bfloat16*,const int8_t*,const __half*,__nv_bfloat16*,cudaStream_t=nullptr);
'''
def fixture(bench=True,sweep=False):
 e=[dict(event='device',name='MOCK_NOT_GPU',uuid='0'*32,sms=170,driver=13020,runtime=13020)]
 names={'cast','reference_fp64','simt','sweep'}|{f'{p}_{t}'for p in('owned','official')for t in R.TS}
 e.extend(dict(event='resources',symbol=n,registers=80,local_bytes=0,shared_bytes=0,active_ctas=1,block_threads=128)for n in sorted(names))
 if sweep:e.append(dict(event='sweep_setup',driver_l2_bytes=256,sweep_bytes=512,physical_cold_guaranteed=False,method='volatile_asm_ld_global_cg_u32_checksum_count',all_data_evicted_control=True))
 def strict(s,dump):
  rows,k,t,p=s;return dict(event='strict',shape=R.name(s),elements=rows*t,independent_cpu_fp64=rows!=248320,cpu_fp32_bits=rows!=248320,numeric_bound=True,exact_integer_case=p==0,fullbits_AA=True,dual_poison=True,guards=True,official=rows==248320,dumped=dump)
 for s in R.plan():e.append(strict(s,True))
 calls=0
 if bench:
  for t in R.TS:
   s=(248320,5120,t,3);e.append(strict(s,False))
   for pair in R.PAIRS:
    for r in range(10):
     if sweep:
      for half in('A1','B1','B2','A2'):
       calls+=1;e.append(dict(event='sweep',shape=R.name(s),pair=pair,round=r,half=half,index=calls,words_read=128,checksum=128*0x5a5a5a5a,before_event=True,synchronized=True,guards=True))
     e.append(dict(event='ABBA',shape=R.name(s),pair=pair,round=r,A1_ms=1,B1_ms=1,B2_ms=1,A2_ms=1,fullbits_each_half=True,scope=R.SCOPE))
 result=dict(complete=True,numeric_pass=True,strict_cases=32,timed_shapes=8 if bench else 0,rounds=10,iterations=1,cache_sweep=sweep,sweeps=calls,model_quality='NOT_RUN');return e,result

class Tests(unittest.TestCase):
 def test_actual_CPU_layout_rounding_full_domain_and_negative_checker(self):
  c=shutil.which('clang++')or shutil.which('g++')
  if not c:self.skipTest('CPU compiler unavailable, not a passing scenario')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'test.cpp').write_text(CPP);r=subprocess.run([c,'-std=c++17','-O2','-ffp-contract=off','-I',str(HERE),str(p/'test.cpp'),'-o',str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);r=subprocess.run([str(p/'test')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('mutations=70',r.stdout)
 def test_host_runner_syntax_CUDA_declarations_only_NOT_NVCC(self):
  c=shutil.which('clang++')
  if not c:self.skipTest('clang unavailable')
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'cuda_runtime.h').write_text(RUNTIME);(p/'cuda.h').write_text('#pragma once\n');(p/'kernel-stubs.h').write_text(STUBS)
   src=(HERE/'bench.cu').read_text().replace('#include "kernels.cuh"','#include "kernel-stubs.h"').replace('#include "official.h"','')
   (p/'host.cu').write_text(src);r=subprocess.run([c,'-std=c++17','--cuda-host-only','-nocudainc','-nocudalib','-fsyntax-only','-I',str(p),'-I',str(HERE),str(p/'host.cu')],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_exact_preregistered_contract_and_official_closure(self):
  self.assertEqual(B.digest(HERE/'numeric-contract.json'),B.CONTRACT_SHA);self.assertEqual(B.digest(HERE/'official-source-manifest.json'),B.CLOSURE_SHA);self.assertEqual(len(B.tool_inventory()),23)
 def test_MMA_lane_mapping_unique_stores_and_valid_loads(self):
  for t in R.TS:
   for rows in(1,15,16,17,33):
    stores=[]
    for block in range((rows+15)//16):
     for lane in range(32):
      g,q=lane//4,lane%4
      for nt in range((t+7)//8):
       for row in(block*16+g,block*16+g+8):
        for token in(nt*8+2*q,nt*8+2*q+1):
         if row<rows and token<t:stores.append(token*rows+row)
    self.assertEqual(sorted(stores),list(range(t*rows)))
  for k in(32,160,512,5120):
   visits=[0]*(k//32)
   for warp in range(4):
    for group in range(warp,k//32,4):
     visits[group]+=1
     for h in range(2):
      for q in range(4):
       c=group*32+h*16+2*q
       for off in(0,1,8,9):self.assertLess(c+off,k)
   self.assertEqual(set(visits),{1})
 def test_owned_independent_source_no_dynamic_parallel_or_hidden_fastmath(self):
  s=(HERE/'kernels.cuh').read_text();self.assertIn('__device__ __forceinline__ void W8Tc',s);self.assertNotIn('__global__ void W8Tc',s);self.assertEqual(s.count('mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32'),1);self.assertIn('s<S;s++',s);self.assertNotIn('--use_fast_math',B.FLAGS);self.assertIn('--ftz=false',B.FLAGS)
 def test_plan_real8_smalltails_and_primary_official(self):
  self.assertEqual(len(R.plan()),32);self.assertEqual(len(set(R.shapemap())),32);self.assertEqual({s[2]for s in R.plan()if s[0]==248320},set(R.TS));self.assertIn('owned_vs_official_primary',R.PAIRS)
 def test_strict_warm_and_sweep_exact_matrices(self):
  for bench,sweep in((False,False),(True,False),(True,True)):
   e,r=fixture(bench,sweep);o=R.validate(e,r,bench,10,1,sweep);self.assertEqual(o['strict_unique'],32);self.assertEqual(o['ABBA'],160 if bench else 0);self.assertEqual(o['sweeps'],640 if sweep else 0)
 def test_missing_duplicate_or_late_strict_rejected(self):
  e,r=fixture()
  for bad in(e[1:],e+[e[22]],e[:22]+e[23:]):
   with self.assertRaises(ValueError):R.validate(bad,r,True,10,1,False)
  bad=copy.deepcopy(e);strict=next(x for x in bad if x['event']=='strict');bad.remove(strict);bad.append(strict)
  with self.assertRaises(ValueError):R.validate(bad,r,True,10,1,False)
 def test_unavailable_nonfinite_resource_or_output_gate_rejected(self):
  e,r=fixture()
  bad_result=copy.deepcopy(r);bad_result['complete']=1
  with self.assertRaises(ValueError):R.validate(e,bad_result,True,10,1,False)
  for event,key,val in(('strict','guards',False),('strict','cpu_fp32_bits',False),('strict','numeric_bound',False),('resources','active_ctas',0),('resources','registers',True),('ABBA','A1_ms',float('nan')),('ABBA','fullbits_each_half',False)):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']==event)[key]=val
   with self.assertRaises(ValueError):R.validate(bad,r,True,10,1,False)
 def test_wrong_scope_or_primary_pair_rejected(self):
  e,r=fixture()
  for key,val in(('scope','matmul_only'),('pair','vs_Q6'),('round',10)):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']=='ABBA')[key]=val
   with self.assertRaises(ValueError):R.validate(bad,r,True,10,1,False)
 def test_sweep_mutations_rejected(self):
  e,r=fixture(True,True)
  for event,key,val in(('sweep_setup','sweep_bytes',256),('sweep_setup','physical_cold_guaranteed',True),('sweep','words_read',127),('sweep','checksum',0),('sweep','before_event',False),('sweep','half','A3')):
   bad=copy.deepcopy(e);next(x for x in bad if x['event']==event)[key]=val
   with self.assertRaises(ValueError):R.validate(bad,r,True,10,1,True)
 def test_data_path_and_no_GPU_default_plan(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   p=Path(t);data=Path('/dev/shm')/('w8-cpu-plan-'+p.name)
   with contextlib.redirect_stdout(io.StringIO()),patch.object(B.P,'command',side_effect=AssertionError('must not spawn')):
    self.assertEqual(R.main(['--build',str(p),'--name','plan','--data',str(data)]),0)
   self.assertFalse(data.exists());self.assertFalse(json.loads((p/'plan/run.json').read_text())['gpu_invoked'])
  for bad in('/tmp/w8','/dev/shm/../shm/w8'):
   with self.assertRaises(ValueError):R.data_path(Path(bad))
 def test_offline_generate_only_no_subprocess_and_source_exact(self):
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   p=Path(t)/'generated'
   with contextlib.redirect_stdout(io.StringIO()),patch.object(B.P,'command',side_effect=AssertionError('generate-only cannot compile')):self.assertEqual(B.main(['--out',str(p)]),0)
   m=json.loads((p/'manifest.json').read_text());self.assertFalse(m['complete']);self.assertEqual(m['source'],m['tools']);self.assertFalse((p/'offline-permit.json').exists())
 def test_resource_parser_preserves_spill_facts_and_requires_all20(self):
  names=['ts_w8_cast','ts_w8_reference','ts_w8_simt','ts_w8_sweep']+[f'ts_w8_tc_{t}'for t in R.TS]+[f'_official_w8_small_t_mma_kernel_{t}'for t in R.TS]
  text=''.join("ptxas info    : Compiling entry function '"+n+"' for 'sm_120a'\nptxas info    : Function properties for "+n+"\n    8 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\nptxas info    : Used 80 registers, 2048 bytes smem\n"for n in names)
  out=B.resources(text);self.assertEqual(len(out),20);self.assertEqual(out[names[0]]['spill_stores'],4)
  with self.assertRaises(ValueError):B.resources(text.split("ptxas info    : Compiling entry function",1)[1])
 def test_process_registration_exception_signals_only_own_group(self):
  proc=unittest.mock.Mock(pid=417,returncode=None);proc.communicate.side_effect=OSError('registration/wait failure')
  with patch.object(B.P.subprocess,'Popen',return_value=proc),patch.object(B.P.os,'getpgid',return_value=417),patch.object(B.P.os,'killpg')as kill:
   with self.assertRaises(OSError):B.P.command(['MOCK'],1)
   kill.assert_called_once_with(417,B.P.signal.SIGTERM);proc.wait.assert_called_once()
 def test_persisted_full_array_checker_and_input_negative_controls(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);shape=(17,32,1,0);d=p/R.name(shape);d.mkdir()
   for name,kind,values in(('reference.f64','d',[0.]*17),('sumAbs.f64','d',[0.]*17),('input.f32','f',[0.]*32),('input.bf16','H',[0]*32),('simt.bf16','H',[0]*17),('owned.bf16','H',[0]*17)):
    a=R.array.array(kind,values)
    with(d/name).open('wb')as f:a.tofile(f)
   with patch.object(R,'shapemap',return_value={R.name(shape):shape}):
    self.assertEqual(len(R.audit_arrays(p)),6)
    for name,content in(('owned.bf16',struct.pack('<H',0x3f80)+bytes(32)),('input.bf16',struct.pack('<H',1)+bytes(62)),('reference.f64',struct.pack('<d',float('nan'))+bytes(128))):
     old=(d/name).read_bytes();(d/name).write_bytes(content)
     with self.assertRaises(ValueError):R.audit_arrays(p)
     (d/name).write_bytes(old)
    (d/'extra').write_bytes(b'x')
    with self.assertRaises(ValueError):R.audit_arrays(p)
 def test_numeric_bound_is_fixed_and_cancellation_aware(self):
  self.assertAlmostEqual(R.allowed(10,30,5120),0.03946302061602821,places=5)
  self.assertGreater(R.allowed(0,100000,5120),R.allowed(0,1,5120));self.assertEqual(B.CONTRACT_SHA,'03fc1a0681beb3a4a0d9ebfdde3980d670b49534008db7e3284c3bc551875f7e')
if __name__=='__main__':unittest.main()
