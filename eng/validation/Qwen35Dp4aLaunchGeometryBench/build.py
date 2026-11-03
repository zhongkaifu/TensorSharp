#!/usr/bin/env python3
"""Offline N8 launch-geometry experiment; no production math/precision changes."""
import argparse,hashlib,importlib.util,json,os,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=HERE.with_name('Qwen35Dp4aScheduleBenchR3')
SPEC=importlib.util.spec_from_file_location('frozen_schedule_r3_geometry_helpers',OLD/'build.py');F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
FROZEN={p:F.digest(OLD/p)for p in('build.py','run.py','bench.cu','cache-sweep.h','runner_cache_delta.json')}
EXPECTED_FROZEN={'build.py':'eb99b4f723ac5ea62d21eeca33073be5f64f485093de8f0240624297725e1fa8','run.py':'322fcd555fafebb9e45acaf86a79b58bb2af810a3a6cdc76dd6697283ef6f372','bench.cu':'ee761f4273e86211836ee047729b2a7ea273d18ebe44585b1be319e4fc5069c2','cache-sweep.h':'8935cdfaf81a2230f7dc82e31a7944351f48053e33b70dac54436fb6ef4124a5','runner_cache_delta.json':'bbb37a82b2d3ffce25dc1daf75e7621c29d289c17218c79997d7eb6e28f711a4'}
INT8_PINS={'build.py':'190ee0fff7deb1840355fa4c140b6a1be1660a1baed46743e389691e943448e8','bench.cu':'1bba40af143acce35a813a170914e674f9fc24fafa39a96ae9d4e9cefcbafb35','run.py':'1e50db22be13933d5adfb988820e6f683dda0b28b6946f515f6d12c95490b6d9','int8mma.cuh':'f3e3db7963cc94ffe20b08d544e15f90325ae9d4f58bf2efb815c19264c9604a'}
SEALED=F.SEALED;SOURCE=F.SOURCE;FLAGS=list(F.FLAGS);ORACLE=F.ORACLE;QUANT='ts_q35_quantize_q81_f32';SWEEP_SYMBOL=F.SWEEP_SYMBOL;SWEEP_VALUE=F.SWEEP_VALUE
CANDIDATES=[dict(id=f'r{r}-b{b}',symbol=f'ts_bench_dp4a_geometry_r{r}_b{b}',rows_per_warp=r,block_threads=b,role='rows_and_launch_only')for r in(1,2,4)for b in(64,256)]
TIMED=[dict(role=role,k=k,out_dim=m,pattern=0,alpha_bits=0x3f800000,timed=True)for role,k,m in [('gdn-in',5120,16480),('attn-in',5120,14336),('gdn-out',6144,5120),('attn-out',6144,5120),('gate-up',5120,34816),('down',17408,5120)]]
RUNNER='dp4a-launch-geometry-bench';PERMIT='dp4a-launch-geometry-offline.json';TOOLS=('build.py','run.py','bench.cu','cache-sweep.h','runner_delta.json')
ARTIFACTS=('kernels.cu','kernels.cubin','kernels.ptx','plan.json','plan.h','timed.json','timed.h','variants.h','cache-sweep.h',RUNNER)
SOURCE_SHA='28ede74605b4edc798744933ca17cf4a18858bb52fb09080f4686459b6fc40b0';QUANT_SOURCE_SHA='d6868491abfc8aca338f30578dc2c78ac7a5cc2d68231b4e725227acd70d7d02'
digest=F.digest;save=F.save;command=F.command;sweep_geometry=F.sweep_geometry;evidence_dir=F.evidence_dir
def check_frozen():
 F.check_frozen()
 if any(digest(F.BASE.FROZEN/n)!=h for n,h in INT8_PINS.items()):raise ValueError('sealed original96 fixture/flags helper changed')
 if any(digest(OLD/n)!=h for n,h in EXPECTED_FROZEN.items())or any(digest(OLD/n)!=h for n,h in FROZEN.items()):raise ValueError('sealed helper changed')
def fixture_pins():return {**F.fixture_pins(),**{str((OLD/n).relative_to(ROOT)):digest(OLD/n)for n in FROZEN}}
def chosen(choice):
 if choice=='all':return list(CANDIDATES)
 matches=[x for x in CANDIDATES if x['id']==choice]
 if len(matches)!=1:raise ValueError('unknown bounded geometry')
 return matches
def geometry(m,r,b):
 if type(m)is not int or m<=0 or(r,b)not in{(x['rows_per_warp'],x['block_threads'])for x in CANDIDATES}|{(4,128)}:raise ValueError('unsupported geometry')
 width=r*(b//32);return ((m+width-1)//width,b,width)
def variants_header(candidates):return 'std::array<Variant,'+str(len(candidates))+'> candidates{{'+','.join('{'+str(v['rows_per_warp'])+','+str(v['block_threads'])+',"'+v['symbol']+'"}'for v in candidates)+'}};\n'
def timed_header():return 'struct TimedRole { const char* role; Shape shape; };\nstatic const TimedRole TimedPlan[] = {\n'+',\n'.join(' {"'+x['role']+'",{'+str(x['k'])+','+str(x['out_dim'])+',0,0x3f800000u,true}}'for x in TIMED)+'\n};\n'
def generate(source,candidates):
 if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('frozen original oracle/quantizer source changed')
 start=source.index('extern "C" __global__ void ts_q35_quantize_q81_f32(');end=source.index('// ---------------------------------------------------------------------------',start+1)
 if hashlib.sha256(source[start:end].encode()).hexdigest()!=QUANT_SOURCE_SHA:raise ValueError('quantizer arithmetic changed')
 prefix=SEALED.oracle_prefix(source);F.template(source)
 wrappers=[]
 for x in candidates:
  wrappers.append('extern "C" __global__ void '+x['symbol']+'(const uint8_t* __restrict__ codes,const uint8_t* __restrict__ scales,int outDim,int k,const int8_t* __restrict__ xq,const float* __restrict__ xd,float alpha,float* __restrict__ y)\n{ ts_q35_gemv_nvfp4<8,'+str(x['rows_per_warp'])+'>(codes,scales,outDim,k,xq,xd,alpha,y); }\n')
 return '#include <cuda_runtime.h>\n'+prefix+'\n// Only independent wrapper/template instantiation and launch geometry differ.\n'+''.join(wrappers)+F.SWEEP_KERNEL
def parse_resources(log,candidates):
 result=F.parse_resources(log,(ORACLE,QUANT,*(x['symbol']for x in candidates)))
 blocks={ORACLE:128,QUANT:256,**{x['symbol']:x['block_threads']for x in candidates}}
 for name,x in result.items():x['block_threads']=blocks[name]
 return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--generate-only',action='store_true');p.add_argument('--nvcc',default='nvcc');p.add_argument('--cuobjdump',default='cuobjdump');p.add_argument('--timeout',type=int,default=600);p.add_argument('--arms',choices=('all',*(x['id']for x in CANDIDATES)),default='all');a=p.parse_args(argv)
 check_frozen()
 if any(os.environ.get(k,'').strip()for k in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):p.error('implicit compiler flags forbidden')
 out=F.evidence_dir(a.out)
 if out.exists()or out.is_symlink()or a.timeout<1:p.error('fresh ignored output and positive timeout required')
 candidates=chosen(a.arms);out.mkdir(parents=True);state=dict(schema='q35-dp4a-launch-geometry-build/1',complete=False,generation_complete=False,source=str(a.source.resolve()),source_sha256=digest(a.source),tools={n:digest(HERE/n)for n in TOOLS},frozen_fixture_tools=fixture_pins(),flags=FLAGS,activation_rows=8,oracle=dict(symbol=ORACLE,rows_per_warp=4,block_threads=128),oracle_is_freshly_recompiled_NOT_production_BBD=True,arm_selection=a.arms,candidates=candidates,strict_cases=96,timed_roles=TIMED,timed_unique_geometries=5,primary_timing_scope='complete_original_q81_quant_plus_gemv',supports_cache_sweep=True,cache_sweep_kernel=SWEEP_SYMBOL,quantizer=QUANT,gpu_quality='NOT_RUN',gpu_timing='NOT_RUN',hardware_counters='NOT_RUN',production_integration=False,build_environment={k:os.environ.get(k)for k in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CUDA_HOME','CUDA_PATH')})
 save(out/'manifest.json',state)
 try:
  source=a.source.read_text();prefix=SEALED.oracle_prefix(source);generated=generate(source,candidates);plan=SEALED.plan()
  (out/'kernels.cu').write_text(generated);save(out/'plan.json',plan);(out/'plan.h').write_text(SEALED.plan_header(plan));save(out/'timed.json',TIMED);(out/'timed.h').write_text(timed_header());(out/'variants.h').write_text(variants_header(candidates));(out/'cache-sweep.h').write_bytes((HERE/'cache-sweep.h').read_bytes())
  state.update(generation_complete=True,generated_sha256=digest(out/'kernels.cu'),quantizer_source_sha256=QUANT_SOURCE_SHA,unchanged_oracle_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),planned_strict_comparisons=96*len(candidates),planned_ABBA_events=6*len(candidates)*5)
  if a.generate_only:state['status']='GENERATED_ONLY_GPU_NOT_RUN';save(out/'manifest.json',state);print(json.dumps(state,indent=2));return 0
  code,text=command([a.nvcc,'--version'],30);(out/'nvcc-version.log').write_text(text);state['nvcc_version']=text
  if code:raise RuntimeError('nvcc version failed')
  common=[a.nvcc,'-std=c++17','-arch=sm_120a',*FLAGS]
  for kind,cmd in [('cubin',common+['-cubin','-Xptxas=-v,-warn-spills',str(out/'kernels.cu'),'-o',str(out/'kernels.cubin')]),('ptx',common+['-ptx',str(out/'kernels.cu'),'-o',str(out/'kernels.ptx')])]:
   code,text=command(cmd,a.timeout);(out/(kind+'.build.log')).write_text(text);state[kind+'_command']=cmd;state[kind+'_exit_code']=code
   if code:raise RuntimeError(kind+' build failed')
   if kind=='cubin':state['resources']=parse_resources(text,candidates);state['sweep_resources']=F.parse_sweep_resources(text)
  if not all(x['offline_resource_qualified']for x in state['resources'].values())or not state['sweep_resources']['offline_resource_qualified']:raise RuntimeError('actual no-stack/no-spill policy rejected; no permit, no auto-prune')
  dump=shutil.which(a.cuobjdump)
  if dump:
   code,text=command([dump,'--dump-sass',str(out/'kernels.cubin')],a.timeout);(out/'kernels.sass.txt').write_text(text)
   if code:raise RuntimeError('SASS extraction failed')
  cmd=[a.nvcc,'-std=c++17','-O3','-I',str(out),str(HERE/'bench.cu'),'-lcuda','-o',str(out/RUNNER)];code,text=command(cmd,a.timeout);(out/'runner.build.log').write_text(text);state.update(runner_command=cmd,runner_exit_code=code)
  if code:raise RuntimeError('runner build failed')
  if digest(a.source)!=state['source_sha256']or state['tools']!={n:digest(HERE/n)for n in TOOLS}or state['frozen_fixture_tools']!=fixture_pins():raise RuntimeError('source/tools changed')
  state.update(complete=True,status='OFFLINE_BUILT_DRIVER_AND_FULLBITS_NOT_RUN',artifacts={n:digest(out/n)for n in ARTIFACTS});save(out/'manifest.json',state);save(out/PERMIT,dict(schema='q35-dp4a-launch-geometry-offline/1',manifest_sha256=digest(out/'manifest.json'),artifacts=state['artifacts']));print(json.dumps(state,indent=2));return 0
 except BaseException as error:
  state.update(complete=False,status='BUILD_REJECTED_GPU_NOT_RUN',error=repr(error));save(out/'manifest.json',state)
  if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state,indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
