"""CPU-only mathematics/layout/source/admission tests. Never nvcc/CUDA/SSH."""
import contextlib,copy,importlib.util,io,json,os,shutil,subprocess,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
build=module('pipeline_build_cpu',HERE/'build.py');run=module('pipeline_run_cpu',HERE/'run.py')
def record(symbol,regs=128,stack=0,stores=0,loads=0,smem=16):return f"ptxas info : Compiling entry function '{symbol}' for 'sm_120a'\nptxas info : Function properties for {symbol}\n {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads\nptxas info : Used {regs} registers, {smem} bytes smem\n"
def resources(**kw):return ''.join(record(v[k],**kw)for v in build.VARIANTS.values()for k in ('symbol','fused_symbol'))
def observed(variant=1,mode='bench'):
 res=mode=='resources-only';bench=mode=='bench';events=[dict(event='device',name='CPU_MOCK_NOT_GPU',cc_major=12,cc_minor=0,sms=170,shared_per_sm=102400,shared_per_cta_optin=101376,driver=13020,runtime=13020),dict(event='shared_size_query',guards=True,actual_bytes=[build.shared(v)for v in build.VARIANTS.values()])]
 for i in (0,variant):
  v=build.VARIANTS[i]
  for fused in (False,True):events.append(dict(event='driver_resources',variant=i,swiglu=fused,symbol=v['fused_symbol'if fused else'symbol'],registers=128,local_bytes=24,static_shared_bytes=16,dynamic_shared_bytes=build.shared(v),total_shared_bytes=16+build.shared(v),block_threads=256,max_block_threads=256,max_dynamic_shared_bytes=build.shared(v),active_ctas=1))
 if not res:
  for s in build.plan():events.append(dict(event='strict_case_pass',variant=variant,shape=build.shape_name(s),swiglu=s['swiglu'],output_words=s['rows']*(s['out_dim']//2 if s['swiglu']else s['out_dim']),**{k:True for k in ('full_output_bits','complete_write_dual_poison','oracle_AA','candidate_AA','guards','all_inputs_unchanged')}))
 if bench:
  for s in build.plan():
   if s['timed']:
    for r in range(5):events.append(dict(event='ABBA_gpu_ms',variant=variant,shape=build.shape_name(s),swiglu=s['swiglu'],rows=s['rows'],out_dim=s['out_dim'],k=s['k'],round=r,iterations_per_sample=3,fullbits_each_half=True,all_new_work_included=True,A1_oracle_ms=1.0,B1_candidate_ms=1.1,B2_candidate_ms=1.2,A2_oracle_ms=1.3))
 result=dict(complete=True,quality_pass=not res,status='DRIVER_LAUNCHABILITY_ONLY_FULLBITS_NOT_RUN'if res else'FULLBITS_GUARDS_COVERAGE_AA_PASS',variant=variant,split_k=1,rounds=5,iterations_per_sample=3,strict_cases=0 if res else len(build.plan()),timed_shapes=6 if bench else 0,timing_events=30 if bench else 0)
 return result,events

class MathematicsTests(unittest.TestCase):
 def test_original_prefix_is_verbatim_and_private_delta_reverses_exactly(self):
  text=build.SOURCE.read_text();g=build.generate(text);prefix=build.oracle_prefix(text);original=build.template(prefix)
  self.assertEqual(g['production-prefix.cuh'],prefix)
  s=g['single-stage.cuh'].replace('void ts_bench_private_gemm_w4a4_s1(','void ts_q35_gemm_w4a4(',1).replace(build.NEW_SCHEDULE,build.OLD_SCHEDULE,1).replace('\n    static_assert(STAGES == 1, "private single-stage schedule only");','',1)
  self.assertEqual(s,original)
  for i,v in build.VARIANTS.items():
   if i:
    for fused in ('true','false'):self.assertIn(f'<{v["bn"]}, 4, 2, {v["stages"]}, {v["bk"]}, true, {fused}>',g[f'variant{i}.cuh'])
  self.assertIn('y[(long long)c * ff + r / 16 * 8 + r % 16] = ts_q35_silu(gate) * up;',s)
  self.assertFalse(any('maxrregcount'in f or 'fast_math'in f for f in build.FLAGS))
 def test_changed_original_math_boundary_or_MMA_refused(self):
  source=build.SOURCE.read_text()
  for old,new in [('rowScale[c] * alpha','alpha * rowScale[c]'),('kstep < BK / 64','kstep < BK / 128'),(build.BOUNDARY,build.BOUNDARY.replace('Layer kernels.','Layer changed.'))]:
   with self.assertRaises(ValueError):build.generate(source.replace(old,new))
 def test_global_k64_accumulation_order_same_for_both_BK(self):
  for k in (256,512,768,1280,5120,17408):
   oracle=list(range(0,k,64))
   for v in build.VARIANTS.values():self.assertEqual([stage*v['bk']+step*64 for stage in range(k//v['bk'])for step in range(v['bk']//64)],oracle)
 def test_every_output_unique_for_plain_and_fused_tail_tiles(self):
  for v in build.VARIANTS.values():
   bn=v['bn']
   for fused,out_dim in ((False,129),(False,257),(True,144),(True,272)):
    rows=129;width=out_dim//2 if fused else out_dim;seen=Counter()
    for by in range((out_dim+127)//128):
     for bx in range((rows+bn-1)//bn):
      for warp in range(8):
       wm=warp%4;wn=warp//4
       for lane in range(32):
        g=lane//4;q=lane%4
        for i in range(2):
         for j in range(bn//16):
          for e in range(2 if fused else 4):
           r=by*128+wm*32+i*16+g+(8 if e>=2 else 0);c=bx*bn+wn*(bn//2)+j*8+q*2+(e&1)
           if r<out_dim and c<rows:
            rr=r//16*8+r%16 if fused else r
            if fused:self.assertLess(r+8,out_dim)
            seen[(c,rr)]+=1
    self.assertEqual(len(seen),rows*width);self.assertEqual(set(seen.values()),{1});self.assertEqual(set(seen),{(r,c)for r in range(rows)for c in range(width)})
 def test_cp_loads_scale_words_alignment_and_no_stage_oob(self):
  for v in build.VARIANTS.values():
   bk=v['bk'];srow=bk//16;crow=bk//2+16
   for rows in (128,v['bn']):
    coverage=Counter()
    for tid in range(256):
     for i in range(tid,rows*(bk//32),256):
      r,c=divmod(i,bk//32);self.assertEqual((r*crow+c*16)%16,0)
      for z in range(16):coverage[(r,c*16+z)]+=1
    self.assertEqual(len(coverage),rows*bk//2);self.assertEqual(set(coverage.values()),{1})
    for r in range(rows):
     self.assertEqual(r*srow%(16 if srow==16 else 8),0)
     for step in range(bk//64):self.assertLessEqual(step*4+4,srow)
   for k in (256,512,17408):
    for stage in range(k//bk):
     self.assertLessEqual(stage*(bk//2)+bk//2,k//2);self.assertLessEqual(stage*srow+srow,k//16)
 def test_fixed_plan_real_six_shapes_and_two_epilogues(self):
  p=build.plan();self.assertEqual(len(p),131);timed=[s for s in p if s['timed']];self.assertEqual(len(timed),6);self.assertEqual(sum(s['swiglu']for s in timed),2);self.assertTrue(all(s['pattern']==6 for s in timed))
  self.assertTrue(all(s['rows']>=512 and s['k']%256==0 for s in timed));self.assertTrue({129,512,2048,4096}<={s['rows']for s in p});self.assertTrue(all(not s['swiglu']or s['out_dim']%16==0 for s in p));self.assertEqual({s['swiglu']for s in p},{True,False})
  self.assertEqual([build.shared(v)for v in build.VARIANTS.values()],[81920,40960,30720,50688,33792])
 def test_host_fixture_distribution_guards_nan_bits_and_dual_poison(self):
  compiler=shutil.which('c++')
  if not compiler:self.skipTest('host C++ compiler absent; not CUDA validation')
  code=r'''
#include "fixtures.h"
#include "checks.h"
#include <cassert>
#include <array>
int main(){
 Shape s{129,144,768,0,0x3f800000u,false,true};PipelineInputs x(s);std::array<int,16>codes{};
 for(auto v:x.wc){codes[v&15]++;codes[v>>4]++;}for(int n:codes)assert(n>0);
 std::array<int,127>scale{};for(auto v:x.ws){assert(v<127);scale[v]++;}for(int n:scale)assert(n>0);
 for(int p=0;p<7;p++){s.pattern=p;PipelineInputs f(s);for(auto b:f.scale)assert((b&0x7f800000u)!=0x7f800000u);for(auto b:f.ws)assert(b<127);for(auto b:f.xs)assert(b<127);}
 s.pattern=6;PipelineInputs moderate(s);std::array<int,64>moderateScales{};for(auto b:moderate.ws){assert(b<64);moderateScales[b]++;}for(int n:moderateScales)assert(n>0);for(auto b:moderate.scale)assert(b>=0x39800000u&&b<0x3a800000u);
 s.pattern=1;PipelineInputs f(s);assert(f.scale[0]==0&&f.scale[1]==0x80000000u);assert(f.scale[4]==0x3f000000u);
 std::vector<uint32_t>a{7,0x7fa12345u,0xffc12345u,0,0x80000000u,0x7f800000u,7},b=a;
 PipelineGuards(a,5,1,7);assert(PipelineDifference(a,b)==a.size());b[3]=0x80000000u;assert(PipelineDifference(a,b)==3);
 b=a;b[2]^=1;assert(PipelineDifference(a,b)==2);b=a;b[0]=8;bool rejected=false;try{PipelineGuards(b,5,1,7);}catch(...){rejected=true;}assert(rejected);
 // A legitimate output may equal a poison; it is accepted if both runs write it.
 b=a;assert(PipelineDifference(a,b)==a.size());b[1]=0xff812345u;assert(PipelineDifference(a,b)==1);
}
'''
  with tempfile.TemporaryDirectory()as t:
   t=Path(t);(t/'host.cpp').write_text(code);subprocess.run([compiler,'-std=c++17','-I',str(HERE),str(t/'host.cpp'),'-o',str(t/'host')],check=True,capture_output=True);subprocess.run([str(t/'host')],check=True,capture_output=True)

class AdmissionTests(unittest.TestCase):
 def setUp(self):
  base=build.ROOT/'artifacts/qwen38-regression/fp4-pipeline-cpu-tests';base.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def invoke(self,out,*extra):
  with contextlib.redirect_stdout(io.StringIO()):return build.main(['--out',str(out),*extra])
 def test_generate_only_never_compiles_or_grants_marker(self):
  out=self.root/'generated'
  with patch.object(build.subprocess,'run',side_effect=AssertionError('unexpected compiler/GPU')):self.assertEqual(self.invoke(out,'--generate-only'),0)
  m=json.loads((out/'manifest.json').read_text());self.assertFalse(m['complete']);self.assertTrue(m['generation_complete']);self.assertEqual(m['gpu_quality'],'NOT_RUN');self.assertFalse((out/build.PERMIT).exists())
 def test_existing_directory_and_implicit_precision_flags_refused(self):
  out=self.root/'existing';out.mkdir();(out/'manifest.json').write_text('keep')
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):self.invoke(out,'--generate-only')
  self.assertEqual((out/'manifest.json').read_text(),'keep')
  for n in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS'):
   out=self.root/n
   with patch.dict(os.environ,{n:'--use_fast_math'}),patch.object(build.subprocess,'run',side_effect=AssertionError('compiler')):self.assertEqual(self.invoke(out,'--generate-only'),1)
   self.assertFalse((out/build.PERMIT).exists())
 def test_resources_are_facts_spills_not_hidden_and_all_ten_required(self):
  r=build.parse_resources(resources(stack=32,stores=12,loads=16));self.assertEqual(len(r),10);self.assertEqual(r[build.ORACLE]['spill_store_bytes'],12)
  for bad in ('',resources()+record(build.ORACLE),resources().replace('Used 128 registers','unknown registers')):
   with self.assertRaises(ValueError):build.parse_resources(bad)
 def fake_compiler(self,fail=''):
  def f(cmd,**kw):
   if cmd[-1]=='--version':return subprocess.CompletedProcess(cmd,0,'CPU MOCK NOT CUDA EVIDENCE','')
   out=Path(cmd[cmd.index('-o')+1])
   if fail=='runner'and out.name==build.RUNNER:return subprocess.CompletedProcess(cmd,1,'','injected failure')
   out.write_bytes(b'CPU_MOCK_NOT_EXECUTABLE')
   return subprocess.CompletedProcess(cmd,0,resources()if'-cubin'in cmd else'','')
  return f
 def test_mock_complete_build_hash_gate_then_corruption_rejected(self):
  out=self.root/'mock'
  with patch.object(build.subprocess,'run',side_effect=self.fake_compiler()),patch.object(build.shutil,'which',return_value=None):self.assertEqual(self.invoke(out),0)
  run.verify_build(out);(out/'variant3.cuh').write_text('changed')
  with self.assertRaises(ValueError):run.verify_build(out)
 def test_failed_runner_compilation_has_no_marker(self):
  out=self.root/'failed'
  with patch.object(build.subprocess,'run',side_effect=self.fake_compiler('runner')),patch.object(build.shutil,'which',return_value=None):self.assertEqual(self.invoke(out),1)
  self.assertFalse((out/build.PERMIT).exists())
 def test_complete_strict_bench_and_resources_only_are_distinct(self):
  for mode in ('strict','bench','resources-only'):
   r,e=observed(mode=mode);g=run.validate_observation(r,e,1,mode);self.assertEqual(g['quality_fullbits'],'NOT_RUN'if mode=='resources-only'else'PASS')
 def test_missing_duplicate_wrong_epi_or_nan_timing_fails_closed(self):
  r,e=observed()
  for mutate in ('missing','duplicate','epilogue','poison','nan','resource','split','device_missing','device_duplicate'):
   a,b=copy.deepcopy(r),copy.deepcopy(e)
   if mutate=='missing':b.pop()
   elif mutate=='duplicate':b.append(next(z for z in b if z['event']=='strict_case_pass'))
   elif mutate=='epilogue':next(z for z in b if z['event']=='strict_case_pass')['swiglu']=True
   elif mutate=='poison':next(z for z in b if z['event']=='strict_case_pass')['complete_write_dual_poison']=False
   elif mutate=='nan':next(z for z in b if z['event']=='ABBA_gpu_ms')['B1_candidate_ms']=float('nan')
   elif mutate=='resource':next(z for z in b if z['event']=='driver_resources')['active_ctas']=0
   elif mutate=='split':a['split_k']=2
   elif mutate=='device_missing':b=[z for z in b if z['event']!='device']
   elif mutate=='device_duplicate':b.append(dict(b[0]))
   with self.subTest(mutate=mutate),self.assertRaises(ValueError):run.validate_observation(a,b,1,'bench')
if __name__=='__main__':unittest.main()
