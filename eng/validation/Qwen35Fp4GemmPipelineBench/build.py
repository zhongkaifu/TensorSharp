#!/usr/bin/env python3
"""Offline-only FP4 GEMM pipeline build/source generation; never initializes CUDA."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SOURCE=ROOT/'TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu'
PREFIX_SHA='effe5b928684d6bef4a1713cbf34c3323d327a5e33c2ecd1c1b80dad9fae7167'
TEMPLATE_SHA='64eff289e45cbe0eba9b75ef247d8e5afa59e7642d7be4db9af2ee93cface34b'
ORACLE='ts_q35_gemm_w4a4_128';RUNNER='fp4-gemm-pipeline-bench';PERMIT='fp4-gemm-offline-built.json'
VARIANTS={0:dict(symbol=ORACLE,bn=128,stages=2,bk=256),1:dict(symbol='ts_bench_fp4_128_s1_bk256',bn=128,stages=1,bk=256),2:dict(symbol='ts_bench_fp4_64_s1_bk256',bn=64,stages=1,bk=256),3:dict(symbol='ts_bench_fp4_64_s3_bk128',bn=64,stages=3,bk=128),4:dict(symbol='ts_bench_fp4_64_s2_bk128',bn=64,stages=2,bk=128)}
for _variant in VARIANTS.values():_variant['fused_symbol']=_variant['symbol']+'_swiglu'
TOOLS=('build.py','run.py','bench.cu','checks.h','fixtures.h')
FLAGS=['-O3','--extra-device-vectorization','--ftz=false','--prec-div=true','--prec-sqrt=true','--fmad=true','-lineinfo']
BOUNDARY='// ===========================================================================\n// Layer kernels. Activations are F32 rows [tokens][width].'
TEMPLATE_START='template <int BN, int WM, int WN, int STAGES, int BK = TS_Q35_BK, bool COLS_FAST = false, bool SWIGLU = false>'
OLD_SCHEDULE='''        ts_q35_cp_wait<STAGES - 2>();
        __syncthreads();
        // The slot this fills was consumed last iteration, which every warp has finished.
        const int next = it + STAGES - 1;
        if (next < nStages)
            load(next % STAGES, sBegin + next);
        ts_q35_cp_commit();'''
NEW_SCHEDULE='''        // Private STAGES1 path: every prior bank reader finished before refill.
        if constexpr (STAGES == 1)
        {
            __syncthreads();
            load(0, sBegin + it);
            ts_q35_cp_commit();
            ts_q35_cp_wait<0>();
            __syncthreads();
        }
        else
        {
'''+OLD_SCHEDULE+'''
        }'''

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def evidence_dir(p):
 p=p.resolve()
 if not any(p.is_relative_to(ROOT/r)for r in ('artifacts','docs/validation')):raise ValueError('fresh output must be inside ignored artifacts/ or docs/validation/')
 return p
def shared(v):return (128+v['bn'])*(v['bk']//2+16+v['bk']//16)*v['stages']
def oracle_prefix(source):
 if source.count(BOUNDARY)!=1:raise ValueError('production prefix boundary changed')
 p=source[:source.index(BOUNDARY)]
 if hashlib.sha256(p.encode()).hexdigest()!=PREFIX_SHA:raise ValueError('reviewed production GEMM prefix changed')
 return p
def template(prefix):
 a=prefix.index(TEMPLATE_START);b=prefix.index('\n#define TS_Q35_GEMM(',a);s=prefix[a:b]
 if hashlib.sha256(s.encode()).hexdigest()!=TEMPLATE_SHA or s.count(OLD_SCHEDULE)!=1:raise ValueError('reviewed template/schedule changed')
 return s
def generate(source):
 prefix=oracle_prefix(source);original=template(prefix)
 private=original.replace('void ts_q35_gemm_w4a4(', 'void ts_bench_private_gemm_w4a4_s1(',1).replace(OLD_SCHEDULE,NEW_SCHEDULE,1)
 anchor='    static_assert(WM * WN == 8, "eight warps");'
 private=private.replace(anchor,anchor+'\n    static_assert(STAGES == 1, "private single-stage schedule only");',1)
 # Reverse the sole scheduling/name/assert delta to prove all math/epilogue bytes remain original.
 restored=private.replace('void ts_bench_private_gemm_w4a4_s1(','void ts_q35_gemm_w4a4(',1).replace(NEW_SCHEDULE,OLD_SCHEDULE,1).replace('\n    static_assert(STAGES == 1, "private single-stage schedule only");','',1)
 if restored!=original:raise ValueError('private template changes arithmetic')
 files={'production-prefix.cuh':prefix,'single-stage.cuh':private}
 for ident,v in VARIANTS.items():
  if not ident:continue
  function='ts_bench_private_gemm_w4a4_s1'if v['stages']==1 else'ts_q35_gemm_w4a4'
  files[f'variant{ident}.cuh']=''
  for fused in (False,True):
   name=v['fused_symbol']if fused else v['symbol']
   files[f'variant{ident}.cuh']+=f'''// Private pipeline arm {ident}; original plain/fused epilogue, grid.z=1.
extern "C" __global__ __launch_bounds__(256) void {name}(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters)
{{ {function}<{v['bn']}, 4, 2, {v['stages']}, {v['bk']}, true, {str(fused).lower()}>(wCodes,wScales,outDim,k,xCodes,xScales,rowScale,m,alpha,y,partial,counters); }}
'''
 files['kernels.cu']='#include <cuda_runtime.h>\n'+''.join('#include "'+n+'"\n'for n in files)
 files['kernels.cu']+='extern "C" __global__ void ts_bench_fp4_smem_query(int* out) {\n'+''.join(f' out[{i}]=sizeof(TsQ35GemmSmem<{v["bn"]},4,2,{v["stages"]},{v["bk"]}>);\n'for i,v in VARIANTS.items())+'}\n'
 return files

def plan():
 p=[]
 def add(rows,out,k,pattern=0,alpha=0x3f800000,timed=False,swiglu=False):p.append(dict(rows=rows,out_dim=out,k=k,pattern=pattern,alpha_bits=alpha,timed=timed,swiglu=swiglu))
 for out,k,rows in ((16480,5120,512),(16480,5120,4096),(34816,5120,512),(34816,5120,4096),(5120,17408,512),(5120,17408,4096)):add(rows,out,k,pattern=6,timed=True,swiglu=out==34816)
 for i,rows in enumerate((129,512,2048,4096)):add(rows,129,256,i)
 for out in (1,127,128,129,255,257):add(129,out,512,2)
 for k in (256,512,768,1280,5120,17408):add(129,131,k,3)
 for pattern in range(6):
  for alpha in (0,0x80000000,0x3f000000,0xbf400000,0x3f800001,0x00800000,1,0x7f7fffff):
   add(129,129,768,pattern,alpha)
   add(129,144,768,pattern,alpha,swiglu=True)
 for out in (16,32,48,128,144,256,272):add(129,out,256,pattern=4,alpha=0x3f000000,swiglu=True)
 # All timed shapes also exercise the other original epilogue, untimed.
 for case in p[:6]:
  peer=dict(case,swiglu=not case['swiglu'],timed=False);p.append(peer)
 if len({shape_name(x)for x in p})!=len(p):raise ValueError('duplicate fixture')
 return p

def shape_name(s):return f'r{s["rows"]}-o{s["out_dim"]}-k{s["k"]}-p{s["pattern"]}-a{s["alpha_bits"]}-f{int(s["swiglu"])}'
def variant_header():return '// Generated pipeline descriptors.\nstatic Variant Variants[] = {\n'+',\n'.join(f' {{{i},{v["bn"]},{v["stages"]},{v["bk"]},{shared(v)},{{"{v["symbol"]}","{v["fused_symbol"]}"}},{{nullptr,nullptr}}}}'for i,v in VARIANTS.items())+'\n};\n'

def plan_header(p):return '// Generated fixed synthetic plan.\nstatic const Shape StrictPlan[] = {\n'+',\n'.join(f' {{{s["rows"]},{s["out_dim"]},{s["k"]},{s["pattern"]},0x{s["alpha_bits"]:08x}u,{str(s["timed"]).lower()},{str(s["swiglu"]).lower()}}}'for s in p)+'\n};\n'
def parse_resources(log):
 out={}
 for ident,v in VARIANTS.items():
  for fused in (False,True):
   name=v['fused_symbol']if fused else v['symbol'];records=re.findall(rf'Compiling entry function [\'\"]{name}[\'\"][\s\S]*?(?=Compiling entry function|\Z)',log)
   if len(records)!=1:raise ValueError('missing/duplicate ptxas record: '+name)
   body=records[0];r=re.search(r'Used (\d+) registers',body);s=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body);sm=re.search(r'(\d+) bytes smem',body)
   if not r or not s:raise ValueError('incomplete ptxas resources: '+name)
   out[name]=dict(registers_per_thread=int(r[1]),stack_bytes=int(s[1]),spill_store_bytes=int(s[2]),spill_load_bytes=int(s[3]),static_shared_bytes=int(sm[1])if sm else 0,dynamic_shared_bytes=shared(v),block_threads=256,swiglu=fused,resource_policy='FACTS_ONLY_DRIVER_ACTIVE_CTA_GE1_REQUIRED; stack/spills not hidden or forced away')
 return out

def artifact_names():return ('kernels.cubin','kernels.ptx','kernels.cu','production-prefix.cuh','single-stage.cuh',*(f'variant{i}.cuh'for i in range(1,5)),'plan.json','plan.h','variants.h',RUNNER)
def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--generate-only',action='store_true');ap.add_argument('--nvcc',default='nvcc');ap.add_argument('--cuobjdump',default='cuobjdump');ap.add_argument('--arch',choices=('sm_120a','sm_120f'),default='sm_120a');ap.add_argument('--timeout',type=int,default=600);a=ap.parse_args(argv)
 try:
  out=evidence_dir(a.out)
  if out.exists()or a.timeout<=0:raise ValueError('output must be NEW and timeout positive')
 except ValueError as e:ap.error(str(e))
 out.mkdir(parents=True,exist_ok=False)
 m=dict(schema='qwen35-fp4-pipeline-build/1',complete=False,generation_complete=False,source=str(a.source.resolve()),tools={n:digest(HERE/n)for n in TOOLS},environment={n:os.environ.get(n)for n in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CUDA_HOME','CUDA_PATH')},flags=FLAGS,arch=a.arch,variants={str(i):dict(v,wm=4,wn=2,block_threads=256,cols_fast=True,epilogues=["plain","swiglu"],dynamic_shared_bytes=shared(v))for i,v in VARIANTS.items()},split_k=1,resource_policy='record actual registers/shared/local/stack/spills; driver activeCTA>=1; no imposed register cap/no minblocks hint',cuda_compilation='NOT_RUN',gpu_quality='NOT_RUN',gpu_timing='NOT_RUN',production_integrated=False,commands=[])
 save(out/'manifest.json',m)
 def command(name,cmd):
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=a.timeout);p=out/(name+'.log');p.write_text(r.stdout+r.stderr);m['commands'].append(dict(argv=cmd,exit_code=r.returncode,log_sha256=digest(p)));save(out/'manifest.json',m)
  if r.returncode:raise RuntimeError(name+' failed')
  return r.stdout+r.stderr
 try:
  if any(os.environ.get(n,'').strip()for n in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):raise ValueError('unset implicit NVCC flags')
  text=a.source.read_text();m['source_sha256']=digest(a.source);generated=generate(text)
  for name,s in generated.items():(out/name).write_text(s)
  p=plan();(out/'plan.json').write_text(json.dumps(p,indent=2)+'\n');(out/'plan.h').write_text(plan_header(p));(out/'variants.h').write_text(variant_header());m.update(generation_complete=True,generated_sha256={n:digest(out/n)for n in generated},prefix_sha256=PREFIX_SHA,template_sha256=TEMPLATE_SHA,strict_cases=len(p),timed_shapes=sum(s['timed']for s in p),plan_sha256=digest(out/'plan.json'))
  save(out/'manifest.json',m)
  if a.generate_only:m['status']='SOURCE_ONLY_NOT_COMPILED_NOT_GPU_VALIDATED';save(out/'manifest.json',m);print(json.dumps(m,indent=2));return 0
  m['nvcc_version']=command('nvcc-version',[a.nvcc,'--version']);nvcc=shutil.which(a.nvcc)
  if nvcc:m['nvcc_sha256']=digest(nvcc)
  flags=[a.nvcc,'-std=c++17','-arch='+a.arch,*FLAGS];m['cuda_compilation']='RUN'
  log=command('kernels-build',flags+['-cubin','-Xptxas=-v,-warn-spills',str(out/'kernels.cu'),'-o',str(out/'kernels.cubin')]);m['resources']=parse_resources(log)
  command('ptx-build',flags+['-ptx',str(out/'kernels.cu'),'-o',str(out/'kernels.ptx')])
  dump=shutil.which(a.cuobjdump)
  if dump:command('sass',[dump,'--dump-sass',str(out/'kernels.cubin')])
  command('runner-build',[a.nvcc,'-std=c++17','-O3','-I',str(out),'-I',str(HERE),str(HERE/'bench.cu'),'-lcuda','-o',str(out/RUNNER)])
  if digest(a.source)!=m['source_sha256']or m['tools']!={n:digest(HERE/n)for n in TOOLS}:raise ValueError('source/tool changed during offline build')
  m.update(complete=True,status='OFFLINE_COMPILED_DRIVER_AND_FULLBITS_NOT_RUN',artifacts={n:digest(out/n)for n in artifact_names()});save(out/'manifest.json',m)
  save(out/PERMIT,dict(schema='qwen35-fp4-pipeline-offline/1',manifest_sha256=digest(out/'manifest.json'),artifacts=m['artifacts'],quality='NOT_RUN'))
  print(json.dumps(m,indent=2));return 0
 except (OSError,ValueError,RuntimeError,subprocess.SubprocessError)as e:m.update(complete=False,error=str(e),status='OFFLINE_FAILED_NO_GPU_QUALIFICATION');save(out/'manifest.json',m);print(json.dumps(m,indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
