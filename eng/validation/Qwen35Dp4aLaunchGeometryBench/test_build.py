import copy,hashlib,importlib.util,json,os,shutil,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load(HERE/'build.py','geometry_CPU_build');R=load(HERE/'run.py','geometry_CPU_run')
def fixture(bench=True,cache=False,choice='all'):
 c=B.chosen(choice);m=dict(arm_selection=choice,candidates=c);events=[dict(event='device',name='CPU_MOCK_NOT_DEVICE',sms=170,registers_per_sm=65536,driver=13020,runtime=13020)]
 for v in [dict(symbol=B.ORACLE,rows_per_warp=4,block_threads=128),*c]:events.append(dict(event='driver_resources',symbol=v['symbol'],rows_per_warp=v['rows_per_warp'],block_threads=v['block_threads'],registers=172,local_bytes=0,static_shared_bytes=0,max_block_threads=256,active_ctas=1,binary_version=120))
 events += [dict(event='quantizer_resources',symbol=B.QUANT,registers=32,local_bytes=0,static_shared_bytes=0,max_block_threads=1024,active_ctas=4,block_threads=256),dict(event='production_rows_query',actual_rows_per_warp=4,expected_rows_per_warp=4,guards=True),dict(event='driver_launch_gate',launchable=True,full_quality_plan_passed=False)]
 for s in B.SEALED.plan():
  for v in c:events.append(dict(event='strict_case_pass',symbol=v['symbol'],shape=R.name(s),rows_per_warp=v['rows_per_warp'],block_threads=v['block_threads'],k=s['k'],out_dim=s['out_dim'],pattern=s['pattern'],alpha_bits=s['alpha_bits'],full_output_bits=True,oracle_AA=True,candidate_AA=True,guards=True,actual_output_argmax_equal=True,outputs=8*s['out_dim'],actual_output_argmax=[0]*8))
 geometry=B.sweep_geometry(100663296,170);sweeps=0
 if cache:events.append(dict(event='cache_sweep_setup',mode='all_data_l2_eviction_control',method='volatile_asm_ld_global_cg_u32_grid_stride_per_block_u64_sum_and_count',physical_cold_guaranteed=False,excluded_from_event_interval=True,registers=20,local_bytes=0,static_shared_bytes=128,active_ctas=6,max_block_threads=1024,**geometry))
 if bench:
  for s in B.TIMED:
   for v in c:
    events.append(dict(event='pipeline_case_pass',symbol=v['symbol'],plane_role=s['role'],shape=R.name(s),Q81_dual_poison_AA=True,full_output_bits_AA_guards=True))
    for round in range(5):
     if cache:
      for half in('A1','B1','B2','A2'):
       sweeps+=1;events.append(dict(event='cache_sweep',index=sweeps,candidate=v['symbol'],measured_symbol=B.ORACLE if half.startswith('A')else v['symbol'],shape=R.name(s),plane_role=s['role'],round=round,half=half,driver_l2_bytes=geometry['driver_l2_bytes'],sweep_bytes=geometry['sweep_bytes'],words_read=geometry['words'],checksum=geometry['words']*B.SWEEP_VALUE,guards=True,checksum_verified=True,synchronized_before_event=True,excluded_from_event_interval=True))
     events.append(dict(event='ABBA_gpu_ms',symbol=v['symbol'],plane_role=s['role'],shape=R.name(s),round=round,rows_per_warp=v['rows_per_warp'],block_threads=v['block_threads'],k=s['k'],out_dim=s['out_dim'],role=v['role'],scope='complete_original_q81_quant_plus_gemv',iterations_per_sample=1 if cache else 3,fullbits_each_half=True,A1_oracle_ms=1.,B1_candidate_ms=1.,B2_candidate_ms=1.,A2_oracle_ms=1.))
 result=dict(complete=True,status='STRICT_OUTPUT_BITS_AND_ARGMAX_PASS',activation_rows=8,strict_cases_per_candidate=96,strict_candidate_comparisons=96*len(c),candidates=len(c),timed_shapes_per_candidate=6 if bench else 0,timed_unique_geometries=5,scope='complete_original_q81_quant_plus_gemv',rounds=5,iterations_per_sample=1 if cache else 3,cache_sweep=cache,cache_mode='all_data_l2_eviction_control'if cache else'warm_repeated_matrix',cache_sweeps=sweeps)
 return events,result,m
HOST_CUDA_STUB='#pragma once\n#include <cstddef>\nusing CUresult=int;using CUdevice=int;using CUmodule=void*;using CUfunction=void*;\nconstexpr int CUDA_SUCCESS=0;\nenum {CU_FUNC_ATTRIBUTE_NUM_REGS,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,CU_FUNC_ATTRIBUTE_BINARY_VERSION,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE};\nCUresult cuGetErrorString(CUresult,const char**);CUresult cuModuleGetFunction(CUfunction*,CUmodule,const char*);CUresult cuFuncGetAttribute(int*,int,CUfunction);CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int*,CUfunction,int,std::size_t);CUresult cuModuleLoad(CUmodule*,const char*);CUresult cuModuleUnload(CUmodule);CUresult cuDeviceGet(CUdevice*,int);CUresult cuDeviceGetAttribute(int*,int,CUdevice);CUresult cuInit(unsigned int);\nCUresult cuLaunchKernel(CUfunction,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,unsigned int,void*,void**,void**);\n'
HOST_RUNTIME_STUB='#pragma once\n#include <cstddef>\nusing cudaError_t=int;using cudaEvent_t=void*;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;\nstruct cudaDeviceProp {char name[256];int major,minor,multiProcessorCount,regsPerMultiprocessor;};\nconst char* cudaGetErrorString(cudaError_t);cudaError_t cudaMalloc(void**,std::size_t);cudaError_t cudaMemset(void*,int,std::size_t);cudaError_t cudaFree(void*);cudaError_t cudaMemcpy(void*,const void*,std::size_t,int);cudaError_t cudaDeviceSynchronize();cudaError_t cudaSetDevice(int);cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);cudaError_t cudaDriverGetVersion(int*);cudaError_t cudaRuntimeGetVersion(int*);cudaError_t cudaEventCreate(cudaEvent_t*);cudaError_t cudaEventRecord(cudaEvent_t,void* =nullptr);cudaError_t cudaEventSynchronize(cudaEvent_t);cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);cudaError_t cudaEventDestroy(cudaEvent_t);\n'
class Geometry(unittest.TestCase):
 def test_only_six_uncovered_combinations_and_full_row_coverage(self):
  self.assertEqual([(c['rows_per_warp'],c['block_threads'])for c in B.CANDIDATES],[(r,b)for r in(1,2,4)for b in(64,256)])
  for r,b in[(4,128),*((c['rows_per_warp'],c['block_threads'])for c in B.CANDIDATES)]:
   for m in(1,7,8,15,16,17,31,33,5119,5120,5121,16480,34817):
    grid,threads,width=B.geometry(m,r,b);got=[(block*(threads//32)+warp)*r+j for block in range(grid)for warp in range(threads//32)for j in range(r)if(block*(threads//32)+warp)*r+j<m];self.assertEqual(got,list(range(m)));self.assertEqual(width,r*b//32)
 def test_original_prefix_quantizer_and_all_arithmetic_are_unchanged(self):
  source=B.SOURCE.read_text();prefix=B.SEALED.oracle_prefix(source);out=B.generate(source,B.CANDIDATES)
  self.assertTrue(out.startswith('#include <cuda_runtime.h>\n'+prefix));self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),B.SOURCE_SHA)
  for c in B.CANDIDATES:self.assertIn('ts_q35_gemv_nvfp4<8,'+str(c['rows_per_warp'])+'>',out);self.assertIn(c['symbol']+'(',out)
  wrappers=out[len('#include <cuda_runtime.h>\n'+prefix):].removesuffix(B.F.SWEEP_KERNEL);self.assertNotIn('__launch_bounds__',wrappers);self.assertNotIn('__ldcs',wrappers);self.assertNotIn('maxrregcount',B.FLAGS)
  for old,new in[('acc[n][j] += ds[j] * xs * (float)sumi;','acc[n][j] += xs * ds[j] * (float)sumi;'),('return 0.5f * v;','return v;'),('127.0f','126.0f')]:
   if old not in source:continue
   with self.assertRaises(ValueError):B.generate(source.replace(old,new,1),B.CANDIDATES)
 def test_frozen_runner_reverse_delta_and_original_strict_checks(self):
  before=(B.OLD/'bench.cu').read_text();now=(HERE/'bench.cu').read_text();delta=json.loads((HERE/'runner_delta.json').read_text())
  for x in reversed(delta['deltas']):self.assertEqual(now.count(x['after']),1);now=now.replace(x['after'],x['before'],1)
  self.assertEqual(now,before);self.assertEqual(hashlib.sha256(now.encode()).hexdigest(),delta['old_source_sha256']);self.assertEqual(len(B.SEALED.plan()),96)
 def test_pipeline_uses_original_quantizer_inside_each_timed_Issue(self):
  s=(HERE/'bench.cu').read_text();issue=s[s.index('    void Issue('):s.index('    std::vector<uint32_t> Read()')]
  self.assertLess(issue.index('if(input)Quantize()'),issue.index('cuLaunchKernel(variant.function'));self.assertIn('cuLaunchKernel(owner->quantizer,(8*k+255)/256,1,1,256',s);self.assertIn('pass?0x5a:0xa5',s);self.assertIn('q!=expectedQ||d!=expectedD',s)
  time=s[s.index('static float VerifiedTime'):s.index('int main(')];self.assertLess(time.index('sweep->Before'),time.index('cudaEventRecord(begin)'));self.assertIn('for(int i=0;i<iterations;i++)f.Issue(variant)',time);self.assertIn('EqualOutput(oracle,f.Read()',time)
 def test_six_roles_only_five_unique_geometries(self):
  self.assertEqual(len(B.TIMED),6);self.assertEqual(len({(s['k'],s['out_dim'])for s in B.TIMED}),5);self.assertEqual(len({s['role']for s in B.TIMED}),6)
  self.assertEqual([x['role']for x in B.TIMED if(x['k'],x['out_dim'])==(6144,5120)],['gdn-out','attn-out'])
 def test_complete_warm_and_sweep_mock_matrices(self):
  for cache in(False,True):
   events,result,m=fixture(cache=cache);v=R.validate_observation(events,result,m,True,5,1 if cache else 3,cache);self.assertEqual(v['strict_comparisons'],576);self.assertEqual(v['pipeline_comparisons'],36);self.assertEqual(v['ABBA_events'],180);self.assertEqual(v['cache_sweeps'],720 if cache else 0)
 def test_missing_duplicate_96case_false_guard_or_partial_pipeline_rejected(self):
  events,result,m=fixture();i=next(i for i,r in enumerate(events)if r['event']=='strict_case_pass');j=next(i for i,r in enumerate(events)if r['event']=='pipeline_case_pass')
  for mode in('missing','duplicate','guard','pipeline','Q81_AA'):
   bad=copy.deepcopy(events)
   if mode=='missing':bad.pop(i)
   elif mode=='duplicate':bad.append(bad[i])
   elif mode=='guard':bad[i]['guards']=False
   elif mode=='pipeline':bad.pop(j)
   else:bad[j]['Q81_dual_poison_AA']=False
   with self.subTest(mode=mode),self.assertRaises(ValueError):R.validate_observation(bad,result,m,True,5,3)
 def test_actual_block_resource_and_scope_timing_must_match(self):
  events,result,m=fixture();i=next(i for i,r in enumerate(events)if r['event']=='ABBA_gpu_ms')
  for mode in('threads','active','local','scope','role','nan','missing'):
   bad=copy.deepcopy(events)
   if mode=='threads':bad[2]['block_threads']=128
   elif mode=='active':bad[1]['active_ctas']=0
   elif mode=='local':bad[1]['local_bytes']=8
   elif mode=='scope':bad[i]['scope']='matrix_only'
   elif mode=='role':bad[i]['plane_role']='wrong'
   elif mode=='nan':bad[i]['B1_candidate_ms']=float('nan')
   else:bad.pop(i)
   with self.subTest(mode=mode),self.assertRaises(ValueError):R.validate_observation(bad,result,m,True,5,3)
 def test_driver_and_ptxas_spill_or_stack_reject_naturally(self):
  names=[B.ORACLE,B.QUANT,*(c['symbol']for c in B.CANDIDATES)];log='\n'.join("Compiling entry function '%s' for 'sm_120a'\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nUsed 172 registers"%s for s in names)
  self.assertTrue(all(v['offline_resource_qualified']for v in B.parse_resources(log,B.CANDIDATES).values()))
  for before,after in[('0 bytes stack','8 bytes stack'),('0 bytes spill stores','8 bytes spill stores'),('0 bytes spill loads','8 bytes spill loads')]:self.assertFalse(all(v['offline_resource_qualified']for v in B.parse_resources(log.replace(before,after,1),B.CANDIDATES).values()))
 def test_explicit_single_geometry_not_automatic_pruning(self):
  for c in B.CANDIDATES:
   e,r,m=fixture(choice=c['id']);v=R.validate_observation(e,r,m,True,5,3);self.assertEqual(v['strict_comparisons'],96);self.assertEqual(v['ABBA_events'],30)
  with self.assertRaises(ValueError):R.selected(dict(arm_selection='all',candidates=[B.CANDIDATES[0]]))
 def test_cache_roles_unique_and_checksum_order_interval_required(self):
  events,result,m=fixture(cache=True);i=next(i for i,r in enumerate(events)if r['event']=='cache_sweep')
  for mode in('role','checksum','inside','size','drop','order'):
   bad=copy.deepcopy(events)
   if mode=='role':bad[i]['plane_role']='attn-out'
   elif mode=='checksum':bad[i]['checksum']+=1
   elif mode=='inside':bad[i]['excluded_from_event_interval']=False
   elif mode=='size':bad[i]['sweep_bytes']-=256
   elif mode=='drop':bad.pop(i)
   else:bad[i:i+2]=list(reversed(bad[i:i+2]))
   with self.subTest(mode=mode),self.assertRaises(ValueError):R.validate_observation(bad,result,m,True,5,1,True)
 def test_generate_only_has_no_compiler_call_or_permit(self):
  (B.ROOT/'artifacts').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   out=Path(t)/'new'
   with patch.object(B,'command',side_effect=AssertionError('generate-only compiler call')),patch.dict(os.environ,{'NVCC_PREPEND_FLAGS':'','NVCC_APPEND_FLAGS':''}):self.assertEqual(B.main(['--out',str(out),'--generate-only']),0)
   m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['complete']);self.assertEqual(m['planned_strict_comparisons'],576);self.assertEqual(m['planned_ABBA_events'],180);self.assertFalse((out/B.PERMIT).exists());self.assertEqual(json.loads((out/'plan.json').read_text()),B.SEALED.plan())
   with self.assertRaises(FileNotFoundError):R.verify_build(out)
 def test_default_plan_and_sweep_iterations_are_not_GPU(self):
  (B.ROOT/'artifacts').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=B.ROOT/'artifacts')as t:
   with patch.object(B,'command',side_effect=AssertionError('plan started GPU')):self.assertEqual(R.main(['--build',t,'--name','plan','--bench','--cache-sweep']),0)
   state=json.loads((Path(t)/'plan/run.json').read_text());self.assertEqual(state['iterations'],1);self.assertFalse(state['complete']);self.assertFalse(state['gpu_invoked'])
   with self.assertRaises(SystemExit):R.main(['--build',t,'--name','bad','--bench','--cache-sweep','--iterations','3'])
 def test_registration_failure_keeps_only_owned_cleanup(self):
  proc=Mock(pid=27183,returncode=None);proc.communicate.return_value=('',None)
  with patch.object(B.F.subprocess,'Popen',return_value=proc),patch.object(B.F.os,'getpgid',return_value=27183),patch.object(B.F.os,'killpg')as kill:
   with self.assertRaises(OSError):B.command(['unused'],1,on_started=lambda p:(_ for _ in()).throw(OSError('registration save')))
   kill.assert_called_once_with(27183,signal.SIGTERM)
 def test_host_CPP_syntax_with_stub_declarations_is_not_NVCC_or_GPU(self):
  compiler=shutil.which('clang++')or shutil.which('g++')
  if not compiler:self.skipTest('No host C++ compiler: this syntax coverage is unavailable, not passing native/GPU validation')
  with tempfile.TemporaryDirectory()as t:
   out=Path(t);(out/'cuda.h').write_text(HOST_CUDA_STUB);(out/'cuda_runtime.h').write_text(HOST_RUNTIME_STUB);(out/'plan.h').write_text(B.SEALED.plan_header(B.SEALED.plan()));(out/'timed.h').write_text(B.timed_header());(out/'variants.h').write_text(B.variants_header(B.CANDIDATES))
   code,text=B.command([compiler,'-std=c++17','-fsyntax-only','-Werror=format','-Werror=format-extra-args','-x','c++','-I',str(out),str(HERE/'bench.cu')],30);self.assertEqual(code,0,text)
if __name__=='__main__':unittest.main()
