import collections,contextlib,hashlib,importlib.util,io,json,math,os,re,shutil,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import build;import run

def ue(x):
 e=(x>>3)&15;m=x&7
 return 0.0 if x in (127,255) else math.ldexp(float(m),-9) if e==0 else math.ldexp(1+m/8,e-7)
def bf(x):
 b=struct.unpack('<I',struct.pack('<f',x))[0];return ((b+0x7fff+((b>>16)&1))>>16)&65535
def bfval(x):return struct.unpack('<f',struct.pack('<I',bf(x)<<16))[0]
def perm(l):
 s=(l>>4)&1;h=(l>>3)&1;q=(l>>1)&3;j=l&1
 return (l&~31)+16*(q>>1)+4*(q&1)+2*s+h+8*j
TABLE=(0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6)
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):(build.ROOT/'artifacts').mkdir(exist_ok=True)
 def test_all_weight_codes_scales_bf16_exact_factor_two(self):
  for scale in range(256):
   for code in range(16):
    w=TABLE[code]*ue(scale);self.assertEqual(bfval(w),w)
    doubled=int(TABLE[code]*2);self.assertEqual(doubled*(ue(scale)*.5),w)
  self.assertEqual(ue(0x81),ue(1));self.assertEqual(ue(127),0);self.assertEqual(ue(255),0)
 def test_permutation_bijection_full_fragment_and_scale_groups(self):
  for base in (0,32,64,96):self.assertEqual(sorted(perm(base+i) for i in range(32)),list(range(base,base+32)))
  for s in range(2):
   for h in range(2):
    for q in range(4):
     for j in range(2):
      p=perm(16*s+8*h+2*q+j);byte=q*4+2*s+h
      self.assertEqual(p,(byte//8)*16+byte%8+8*j);self.assertEqual(p//16,q//2)
  # PTX C-fragment ownership: every valid (token,weight-row) once, including both tails.
  for n in (1,2,4,8,16):
   for rows in (1,17,63,64,65):
    owned=[]
    for block in range((rows+63)//64):
     for warp in range(4):
      for lane in range(32):
       g,q=lane//4,lane%4
       for tile in range(2):
        for h in range(2):
         for j in range(2):
          token=tile*8+q*2+j;row=block*64+warp*16+g+8*h
          if token<n and row<rows:owned.append((token,row))
    self.assertEqual(len(owned),n*rows);self.assertEqual(set(owned),{(t,r) for t in range(n) for r in range(rows)})
 def test_small_logical_matmul_independent_vs_fragment_and_mutation(self):
  n,k,m=8,128,17;codes=bytes((i*73+(i//7)*31)%256 for i in range(m*k//2));scales=bytes(1+(i*37)%126 for i in range(m*k//16));x=[[bfval((t*131+(j*j*17)%233-100)/128) for j in range(k)] for t in range(n)]
  def weight(r,c):
   byte=codes[r*k//2+(c//16)*8+c%8];return TABLE[(byte>>(4 if c%16>=8 else 0))&15]*ue(scales[r*k//16+c//16])
  mismatches=0
  for t in range(n):
   for r in range(m):
    ref=math.fsum(weight(r,c)*x[t][c] for c in range(k));candidate=math.fsum(weight(r,perm(l))*x[t][perm(l)] for l in range(k));wrong=math.fsum(weight(r,perm(l))*x[t][l] for l in range(k))
    self.assertEqual(ref,candidate);mismatches+=ref!=wrong
  self.assertGreater(mismatches,n*m//2)
 def test_plan_supported_shapes_tails_and_nonunit_pilot(self):
  p=build.plan();self.assertEqual(len(p),78);self.assertEqual(sum(r['timed'] for r in p),12);self.assertEqual({r['n'] for r in p},{1,2,4,8,16})
  ids=[(r['k'],r['m'],r['n'],r['pattern'],struct.pack('<f',r['alpha'])) for r in p];self.assertEqual(len(set(ids)),len(ids))
  self.assertTrue(any(r['k']==128 and r['m']==17 and r['n']==8 and r['pattern']==1 for r in p));self.assertTrue(any(r['m']<64 for r in p));self.assertTrue(any(r['m']%64 for r in p))
 def test_split_no_empty_slice_and_full_cover_actual_sms(self):
  for sms in (1,80,170):
   for p in build.plan():
    z=1;total=p['k']//128
    while ((p['m']+63)//64)*z<2*sms and total//(z*2)>=2:z*=2
    per=(total+z-1)//z;ranges=[(s*per,min(total,(s+1)*per)) for s in range(z)]
    self.assertTrue(all(a<b for a,b in ranges));self.assertEqual([i for a,b in ranges for i in range(a,b)],list(range(total)));self.assertLessEqual(z,256)
 def test_default_generate_no_compiler_no_marker_and_no_overwrite(self):
  with tempfile.TemporaryDirectory(dir=build.ROOT/'artifacts') as tmp:
   out=Path(tmp)/'new'
   with mock.patch.object(build.subprocess,'run',side_effect=AssertionError('compiler')):self.assertEqual(build.main(['--out',str(out)]),0)
   m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['complete']);self.assertEqual(m['GPU'],'NOT_RUN');self.assertFalse((out/'w4a16-build-ready.json').exists());self.assertEqual(m['strict_cases'],78)
   with self.assertRaises(ValueError):build.main(['--out',str(out)])
 def test_generate_prefix_exact_and_native_source_unchanged(self):
  s=build.SOURCE.read_text();p=build.prefix(s);self.assertTrue(s.startswith(p));self.assertEqual(build.sha(build.SOURCE),build.SOURCE_SHA)
  c=(HERE/'skinny.cuh').read_text();self.assertIn('2.0f*ts_q35_ue4m3(rawScale)',c);self.assertIn('reference_f64',c)
 def test_hidden_flags_rejected_before_output(self):
  with tempfile.TemporaryDirectory(dir=build.ROOT/'artifacts') as tmp,mock.patch.dict(os.environ,{'NVCC_APPEND_FLAGS':'--use_fast_math'}):
   out=Path(tmp)/'never'
   with self.assertRaises(ValueError):build.main(['--out',str(out)])
   self.assertFalse(out.exists())
 def test_compile_version_failure_no_ready(self):
  with tempfile.TemporaryDirectory(dir=build.ROOT/'artifacts') as tmp,mock.patch.object(build.subprocess,'run',return_value=subprocess.CompletedProcess(['nvcc'],2,'','fail')) as call:
   out=Path(tmp)/'bad';self.assertEqual(build.main(['--out',str(out),'--compile']),1);self.assertEqual(call.call_count,1);self.assertFalse((out/'w4a16-build-ready.json').exists())
 def test_resource_record_no_arbitrary_register_or_twocta_reject(self):
  log="ptxas info : Compiling entry function 'ts_wm16_weightmajor' for 'sm_120a'\n256 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\nUsed 240 registers\n"
  r=build.resource(log);self.assertEqual(r['registers'],240);self.assertEqual(r['stack_bytes'],256);self.assertNotIn('qualified',r)
  with self.assertRaises(ValueError):build.resource('')
 def test_gpu_plan_no_process_or_creation(self):
  with mock.patch.object(run.subprocess,'Popen',side_effect=AssertionError('GPU')),mock.patch.object(Path,'mkdir',side_effect=AssertionError('write')),contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(run.main(['--build','/tmp/not-built','--name','plan']),0)
  self.assertEqual(json.loads(out.getvalue())['GPU'],'NOT_RUN')
 def valid_events(self):
  p=build.plan();events=[dict(event='driver_resources',active_ctas=1,actual_block=128,dynamic_shared_bytes=39936),dict(event='device',sms=170),dict(event='cache_control',mode='warm',driver_l2_bytes=96*1024*1024,sweep_bytes=0,blocks=0,threads=256,constant_word=16843009,method='disabled',physical_cold_guaranteed=False)]
  for r in p:
   name='k%d-m%d-n%d-p%d-a%d'%(r['k'],r['m'],r['n'],r['pattern'],struct.unpack('<I',struct.pack('<f',r['alpha']))[0]);events.append(dict(event='strict_case_pass',shape=name,full_output_saved=True,AA=True,guards=True,relative_l2=0.0,mutation_checked=r['k']==128 and r['m']==17 and r['n']==8 and r['pattern']==1))
  result=dict(schema='q35-w4a16-weightmajor-result/1',complete=True,quality_passed=True,old_A8_equivalence_claim=False,strict_cases=78,timed_shapes=0,rounds=5,iterations=3,cache_mode='warm',cache_sweep_calls=0,driver_l2_bytes=96*1024*1024,sweep_bytes=0)
  return result,events
 def test_run_admission_counts_duplicate_numeric_and_mutation_failclosed(self):
  r,e=self.valid_events();self.assertEqual(run.validate(r,e,False,5,3)['strict_events'],78)
  for mutate in ('duplicate','l2','bool','mutation','driver'):
   rr=json.loads(json.dumps(r));ee=json.loads(json.dumps(e))
   if mutate=='duplicate':ee[-1]=ee[-2]
   elif mutate=='l2':ee[-1]['relative_l2']=.01
   elif mutate=='bool':ee[-1]['relative_l2']=False
   elif mutate=='mutation':next(x for x in ee if x.get('mutation_checked'))['mutation_checked']=False
   else:ee[0]['active_ctas']=0
   with self.assertRaises(ValueError):run.validate(rr,ee,False,5,3)
 def test_complete_host_runner_cpp_syntax_without_cuda_or_link(self):
  compiler=shutil.which('clang++') or shutil.which('g++');self.assertIsNotNone(compiler)
  runtime="""#pragma once
#include <cstddef>
using cudaError_t=int;using cudaEvent_t=void*;constexpr int cudaSuccess=0;enum cudaMemcpyKind{cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost};struct cudaDeviceProp{char name[256];int multiProcessorCount,major,minor;};
const char*cudaGetErrorString(int);int cudaMalloc(void**,size_t);int cudaFree(void*);int cudaMemset(void*,int,size_t);int cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind);int cudaDeviceSynchronize();int cudaSetDevice(int);int cudaGetDeviceProperties(cudaDeviceProp*,int);int cudaEventCreate(cudaEvent_t*);int cudaEventRecord(cudaEvent_t,void* = nullptr);int cudaEventSynchronize(cudaEvent_t);int cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);int cudaEventDestroy(cudaEvent_t);
"""
  driver="""#pragma once
#include <cstddef>
using CUresult=int;using CUmodule=void*;using CUfunction=void*;using CUdevice=int;constexpr int CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=38;int cuDeviceGet(CUdevice*,int);int cuDeviceGetAttribute(int*,int,CUdevice);constexpr int CUDA_SUCCESS=0;enum{CU_FUNC_ATTRIBUTE_NUM_REGS,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,CU_FUNC_ATTRIBUTE_BINARY_VERSION};int cuGetErrorString(int,const char**);int cuModuleLoad(CUmodule*,const char*);int cuModuleGetFunction(CUfunction*,CUmodule,const char*);int cuModuleUnload(CUmodule);int cuFuncGetAttribute(int*,int,CUfunction);int cuOccupancyMaxActiveBlocksPerMultiprocessor(int*,CUfunction,int,size_t);int cuLaunchKernel(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**);int cuInit(unsigned);
"""
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);(d/'cuda_runtime.h').write_text(runtime);(d/'cuda.h').write_text(driver);(d/'plan.h').write_text(build.header(build.plan()))
   result=subprocess.run([compiler,'-x','c++','-std=c++17','-fsyntax-only','-I',str(d),'-I',str(HERE),str(HERE/'bench.cu')],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
 def test_original_fp64_checker_strict_and_fixture_math_unchanged(self):
  cu=(HERE/'skinny.cuh').read_text();reference=cu[cu.index('// Independent logical-index FP64 oracle.'):cu.index('// SA16_EVICTION_SWEEP_BEGIN')]
  self.assertEqual(hashlib.sha256(reference.encode()).hexdigest(),'9f6d2a68bf11281572bd42752e75e6a3f6b3a9511b7b524f0357e988f77fdac1')
  self.assertEqual(build.sha(HERE/'checks.h'),'73a2c08439fdc9e4818112c2b33c734c5041889617f16fbe6feab73e7d3ff15c')
  host=(HERE/'bench.cu').read_text();fixture=host[host.index('struct Fixture {'):host.index('void Bits(')].replace('(a.n!=1&&a.n!=2','(a.n!=2')
  self.assertEqual(hashlib.sha256(fixture.encode()).hexdigest(),'d65b93f6267e8a558a134144e1a83fcd6520705ba2f0b097435992d07ad8e454')
  self.assertEqual(hashlib.sha256(host[host.index('Expected Strict('):host.index('float Time(')].encode()).hexdigest(),'1e86ec94516e67dedb2f5e35bd7abb41b1ab7cd34cee71ef83d8b7f536ee227d')
 def test_cache_sweep_cli_forces_single_iteration_default_warm_unchanged(self):
  with contextlib.redirect_stdout(io.StringIO()) as output:self.assertEqual(run.main(['--build','/tmp/unused','--name','plan','--bench','--cache-sweep']),0)
  p=json.loads(output.getvalue());self.assertEqual(p['cache_mode'],'eviction-control');self.assertEqual(p['argv'][p['argv'].index('--iterations')+1],'1');self.assertIn('--cache-sweep',p['argv'])
  for args in (['--cache-sweep'],['--bench','--cache-sweep','--iterations','2']):
   with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):run.main(['--build','/tmp/unused','--name','plan',*args])
  with contextlib.redirect_stdout(io.StringIO()) as output:run.main(['--build','/tmp/unused','--name','warm'])
  p=json.loads(output.getvalue());self.assertEqual(p['cache_mode'],'warm');self.assertNotIn('--cache-sweep',p['argv']);self.assertEqual(p['argv'][p['argv'].index('--iterations')+1],'3')
 def test_full_read_gridstride_coverage_and_two_actual_l2_sizes(self):
  for l2 in (1,1024,64*1024*1024,96*1024*1024):
   for sms in (1,170):
    size=((l2*2+255)//256)*256;words=size//4;workers=sms*4*256;self.assertGreaterEqual(size,2*l2)
    counts=[1+(words-1-i)//workers if i<words else 0 for i in range(workers)];self.assertEqual(sum(counts),words)
    for i in (0,min(workers-1,words-1)):
     if counts[i]:self.assertLess(i+(counts[i]-1)*workers,words);self.assertGreaterEqual(i+counts[i]*workers,words)
    q,r=divmod(words,workers);expected=r*(((q+1)*16843009)&0xffffffff)+(workers-r)*((q*16843009)&0xffffffff)
    self.assertEqual(sum((n*16843009)&0xffffffff for n in counts),expected)
 def test_sweep_source_and_event_boundary_before_timer_not_inside_iterations(self):
  source=(HERE/'skinny.cuh').read_text();self.assertIn('asm volatile("ld.global.cg.u32',source);self.assertIn(': "memory")',source);self.assertIn('sum+=value',source);self.assertIn('checksum[id]=sum',source)
  host=(HERE/'bench.cu').read_text();timer=host[host.index('float Time('):host.index('int main(')];self.assertLess(timer.index('sweep.Run'),timer.index('cudaEventRecord(start)'));self.assertLess(timer.index('cudaEventRecord(start)'),timer.index('for(int i=0;i<iterations;i++)'));self.assertNotIn('sweep.Run',timer[timer.index('for(int i=0;i<iterations;i++)'):timer.index('cudaEventRecord(end)')])
  self.assertIn('cuDeviceGetAttribute(&actual,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE',host);self.assertIn('Rt(cudaDeviceSynchronize());auto values=',host);self.assertIn('if(values[i]!=want)',host)
 def cache_events(self):
  result,events=self.valid_events();l2=96*1024*1024;sweep=l2*2;blocks=170*4;workers=blocks*256;q,r=divmod(sweep//4,workers);checksum=r*(((q+1)*16843009)&0xffffffff)+(workers-r)*((q*16843009)&0xffffffff)
  events[2].update(mode='eviction-control',sweep_bytes=sweep,blocks=blocks,method='volatile_asm_ld_global_cg_u32')
  timed=[x for x in build.plan() if x['timed']]
  def name(x):return 'k%d-m%d-n%d-p%d-a%d'%(x['k'],x['m'],x['n'],x['pattern'],struct.unpack('<I',struct.pack('<f',x['alpha']))[0])
  for row in timed:
   # Timed fixture is re-gated just before that shape's intervals.
   events.append(dict(event='strict_case_pass',shape=name(row),full_output_saved=True,AA=True,guards=True,relative_l2=0.0,mutation_checked=False))
   for scope in ('matrix_kernel_only','full_pack_kernel_combine'):
    for rd in range(5):
     for half in ('A1','B1','B2','A2'):events.append(dict(event='cache_sweep',shape=name(row),scope=scope,round=rd,half=half,mode='eviction-control',driver_l2_bytes=l2,sweep_bytes=sweep,checksum=checksum,expected_checksum=checksum,checksum_words=workers,all_checksums_equal=True,synchronized_before_event=True,physical_cold_guaranteed=False))
     events.append(dict(event='ABBA_gpu_ms',shape=name(row),scope=scope,round=rd,cache_mode='eviction-control',iterations_per_sample=1,A1_legacy_ms=1.,B1_bf16_ms=1.,B2_bf16_ms=1.,A2_legacy_ms=1.,full_output_postsample_bits=True))
  result.update(timed_shapes=12,iterations=1,cache_mode='eviction-control',sweep_bytes=sweep,cache_sweep_calls=480)
  return result,events
 def test_eviction_event_complete_matrix_and_missing_duplicate_checksum_refused(self):
  r,e=self.cache_events();self.assertEqual(run.validate(r,e,True,5,1,True)['cache_sweeps'],480)
  for mode in ('missing','duplicate','checksum','expected','size','iteration','order','early'):
   rr=json.loads(json.dumps(r));ee=json.loads(json.dumps(e));at=next(i for i,x in enumerate(ee) if x.get('event')=='cache_sweep')
   if mode=='missing':ee.pop(at)
   elif mode=='duplicate':ee.insert(at,ee[at])
   elif mode=='checksum':ee[at]['checksum']+=1
   elif mode=='expected':ee[at]['checksum']+=1;ee[at]['expected_checksum']+=1
   elif mode=='size':ee[2]['sweep_bytes']=ee[2]['driver_l2_bytes'];rr['sweep_bytes']=ee[2]['sweep_bytes']
   elif mode=='iteration':rr['iterations']=2
   elif mode=='order':ee[at],ee[at+1]=ee[at+1],ee[at]
   else:ee.insert(0,ee.pop(at))
   with self.assertRaises(ValueError):run.validate(rr,ee,True,5,1,True)
 def test_weight_major_A_B_fragment_coordinates_and_half_MMA_work(self):
  # CUDA13.2 PTX m16n8k16: A registers = rows g/g+8, cols2q/{2q+8}; B=K rows2q/{2q+8}, colg.
  for step in range(2):
   for lane in range(32):
    g,q=lane//4,lane%4
    for h in range(2):
     for j in range(2):
      l=step*16+8*h+2*q+j;p=perm(l);byte=4*q+2*step+h
      self.assertEqual(p,(byte//8)*16+byte%8+8*j);self.assertEqual(p//16,q//2)
    for reg in range(4):
     self.assertEqual(g+(8 if reg in (1,3)else 0),(g,g+8,g,g+8)[reg])
  self.assertEqual(4*2*2,16) # original N8: four ch,2weighttiles,2Ksteps.
  self.assertEqual(4*2*1,8) # new N8: four ch,2Ksteps,1tokenblock.
  src=(HERE/'skinny.cuh').read_text();self.assertIn('int r0=warp*16+g,r1=r0+8',src);self.assertIn('int token=tokenBlock*8+q*2,row=row0+warp*16+g+8*h',src);self.assertIn('tokenBlock*8<n',src);self.assertNotIn('for(int tile=0;tile<2;tile++)',src)
 def test_actual_integer_decode_helper_full4096_signed_zero_bits(self):
  compiler=shutil.which('clang++')or shutil.which('g++');self.assertIsNotNone(compiler)
  source=(HERE/'skinny.cuh').read_text();start=source.index('__host__ __device__ uint16_t sa16_weight_bits_int');end=source.index('__device__ uint32_t sa16_weight_pair',start);helper=source[start:end].replace('__host__ __device__','')
  test=r'''#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
HELPER
int main(){const float table[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};int tested=0;for(unsigned sc=0;sc<256;sc++)for(unsigned c=0;c<16;c++){int e=(sc>>3)&15,m=sc&7;float scale=(sc==127||sc==255)?0:e==0?std::ldexp(float(m),-9):std::ldexp(1.f+m/8.f,e-7);volatile float w=table[c]*scale;float value=w;uint32_t bits;std::memcpy(&bits,&value,4);uint16_t want=uint16_t((bits+0x7fffu+((bits>>16)&1))>>16);auto actual=sa16_weight_bits_int(c,uint8_t(sc));if(actual!=want){std::printf("sc%u code%u got%x want%x\n",sc,c,actual,want);return 1;}tested++;}if(tested!=4096)return 2;if(sa16_weight_bits_int(9,0)!=0x8000||sa16_weight_bits_int(8,0)!=0)return 3;return 0;}
'''.replace('HELPER',helper)
  with tempfile.TemporaryDirectory()as tmp:
   tmp=Path(tmp);(tmp/'test.cpp').write_text(test);cmd=[compiler,'-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',str(tmp/'test.cpp'),'-o',str(tmp/'test')];c=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(c.returncode,0,c.stderr);p=subprocess.run([str(tmp/'test')],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stdout+p.stderr)
 def test_integer_decode_explicit_separate_build_not_implicit(self):
  with tempfile.TemporaryDirectory(dir=build.ROOT/'artifacts')as tmp:
   for on in (False,True):
    out=Path(tmp)/str(on);self.assertEqual(build.main(['--out',str(out)]+(['--integer-decode']if on else [])),0);m=json.loads((out/'manifest.json').read_text());self.assertIs(m['integer_decode'],on);self.assertEqual(m['flags'],build.FLAGS+(['-DSA16_INTEGER_DECODE=1']if on else []));self.assertFalse(m['complete'])
 def test_original67_fixtures_retained_additional11_N1_only(self):
  p=build.plan();self.assertEqual(hashlib.sha256(json.dumps(p[:67],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'2592c543f937d455988314f926f3076ad458459fe7fc224a6d112214b5e9cbba');self.assertTrue(all(x['n']!=1 for x in p[:67]));self.assertEqual(len(p[67:]),11);self.assertTrue(all(x['n']==1 and not x['timed']for x in p[67:]));self.assertEqual(sum(x['timed']for x in p),12)
 def test_actual_host_cpp_checker_and_guard_mutations(self):
  compiler=shutil.which('clang++') or shutil.which('g++');self.assertIsNotNone(compiler)
  source=r'''#include "checks.h"
int main(){std::vector<float>a={1,2,-3};std::vector<double>r={1,2,-3},ab={1,2,3};if(!CheckNumeric(a,r,ab,128,2).pass)return 1;a[1]*=1.01f;if(CheckNumeric(a,r,ab,128,2).pass)return 2;a={1,2,-3};a[0]=NAN;if(CheckNumeric(a,r,ab,128,2).pass)return 3;uint32_t before[2]={9,9},after[2]={9,9};RequireGuard(before,after,2,9);before[1]=8;try{RequireGuard(before,after,2,9);return 4;}catch(const std::runtime_error&){}before[1]=9;after[0]=8;try{RequireGuard(before,after,2,9);return 5;}catch(const std::runtime_error&){}for(int l=0;l<128;l++){int p=HostPerm(l);if(p/32!=l/32)return 6;}return 0;}'''
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);(d/'test.cpp').write_text(source);c=subprocess.run([compiler,'-std=c++17','-O2','-I',str(HERE),str(d/'test.cpp'),'-o',str(d/'test')],capture_output=True,text=True);self.assertEqual(c.returncode,0,c.stderr);self.assertEqual(subprocess.run([str(d/'test')]).returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
