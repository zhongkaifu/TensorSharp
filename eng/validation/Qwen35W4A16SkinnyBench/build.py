#!/usr/bin/env python3
"""Owned W4A16 skinny-TC experiment. Default only generates source; --compile never runs GPU."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu'
SOURCE_SHA='28ede74605b4edc798744933ca17cf4a18858bb52fb09080f4686459b6fc40b0'
TOOLS=('build.py','run.py','bench.cu','skinny.cuh','checks.h')
ARTIFACTS=('kernels.cu','kernels.cubin','kernels.ptx','plan.h','plan.json','skinny-bench')
FLAGS=['-O3','--extra-device-vectorization','--ftz=false','--prec-div=true','--prec-sqrt=true','--fmad=true','-lineinfo']
INSPIRATION=dict(url='https://github.com/Oceania2018/velox-inferer/blob/b67f7d4565f5335988c91869cf0590b1e05998cd/crates/velox-variant/kernels/rtx5090_skinny16.cu',ref='b67f7d4565f5335988c91869cf0590b1e05998cd',scope='Algorithm inspiration: four-stage raw-weight streaming, BF16 register decode, skinny tensor core tile. Independent TensorSharp implementation, TS-specific nibble permutation/UE4M3 and FP64 oracle.',license_note='Pinned public tree contained no LICENSE/COPYING entry. No upstream implementation is copied/included; no upstream license grant is asserted.')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def prefix(source):
 marker="// Activation rows -> NVFP4 planes + per-row scale, ggml-cuda's quantize_mmq_nvfp4 numerics."
 if source.count(marker)!=1:raise ValueError('baseline boundary differs')
 p=source[:source.index(marker)]
 for s in ['TS_Q35_GEMV(8, TS_Q35_GEMV_ROWS(8))','acc[n][j] += ds[j] * xs * (float)sumi;','return 0.5f * v;']:
  if s not in p:raise ValueError('unchanged A8 oracle differs')
 return p
def plan():
 cases=[]
 def add(k,m,n=8,p=0,a=1.0,t=False):cases.append(dict(k=k,m=m,n=n,pattern=p,alpha=a,timed=t))
 for k in (5120,6144,17408):
  for m in (5120,14336,16480,34816):add(k,m,p=6,t=True)
 for n in (2,4,8,16):
  for k in (128,384,640):
   for m in (1,17,65):add(k,m,n,p=(k//128+m+n)%6)
  add(5120,14335,n,p=1);add(5120,16481,n,p=4)
 for p in range(6):add(256,63,p=p)
 for a in (0.0,-0.0,.5,-1.0):add(128,17,p=5,a=a)
 add(128,17,p=1)
 return cases
def header(cases):return '// Generated fixed plan.\nstatic const Shape StrictPlan[]={\n'+',\n'.join(' {%d,%d,%d,%d,%sf,%s}'%(r['k'],r['m'],r['n'],r['pattern'],str(r['alpha']),'true' if r['timed'] else 'false') for r in cases)+'\n};\n'
def evidence(path):
 p=path.absolute()
 if Path(os.path.normpath(p))!=p or p.exists() or p.is_symlink() or any(a.is_symlink() for a in p.parents):raise ValueError('fresh normalized output without symlink ancestors required')
 if not any(p.is_relative_to(r) for r in (ROOT/'artifacts',ROOT/'docs/validation')):raise ValueError('output must be ignored checkout evidence')
 return p
def resource(log):
 record=re.findall(r"Compiling entry function ['\"]ts_sa16_skinny['\"][\s\S]*?(?=Compiling entry function|\Z)",log)
 if len(record)!=1:raise ValueError('one actual candidate resource record required')
 r=record[0];reg=re.search(r'Used (\d+) registers',r);stack=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',r)
 if not reg or not stack:raise ValueError('actual register/stack/spill data missing')
 sm=re.search(r'(\d+) bytes smem',r)
 return dict(registers=int(reg[1]),stack_bytes=int(stack[1]),spill_stores=int(stack[2]),spill_loads=int(stack[3]),static_shared_bytes=int(sm[1]) if sm else 0,dynamic_shared_bytes=39936,block_threads=128,policy='Record natural resources. Actual driver activeCTA>=1 required before numerical gate; no arbitrary 128regs/2CTA or maxrregcount policy.')
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--compile',action='store_true');p.add_argument('--nvcc',default='nvcc');p.add_argument('--arch',default='sm_120a',choices=('sm_120','sm_120a'));p.add_argument('--cuobjdump',default='cuobjdump');a=p.parse_args(argv)
 if any(os.environ.get(k,'').strip() for k in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):raise ValueError('hidden NVCC flags refused')
 out=evidence(a.out);out.mkdir(parents=True);m=dict(schema='q35-w4a16-skinny-build/1',complete=False,generation_complete=False,GPU='NOT_RUN',source=str(a.source.resolve()),source_sha256=sha(a.source),tools={n:sha(HERE/n) for n in TOOLS},flags=FLAGS,arch=a.arch,build_environment={k:os.environ.get(k) for k in ('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CUDA_HOME','CUDA_PATH')},inspiration=INSPIRATION,old_A8_equivalence_claim=False,numeric_contract='BF16 RNE input × exact GGUF NVFP4 decoded W; F32 MMA + fixed-order split reduction; independent FP64 full-output reference.',timing_controls=dict(default='warm',eviction='--bench --cache-sweep with iterations1; per-half >=2driverL2 asm volatile ld.global.cg.u32 +memoryclobber/checksum/sync outsideevents; fullpipeline is primarycontrol; no physicalcold guarantee'),predeclared_gate=dict(relative_l2_limit=2e-5,per_output_gamma='t=(2K+2splits+8)*2^-24; bound=t/(1-t)*sumAbsProducts+2^-140',candidate_AA='full bits',guards='all allocations + dual output poison',swizzle_mutation='must FAIL'),production_integrated=False)
 try:
  if m['source_sha256']!=SOURCE_SHA:raise ValueError('unchanged production source SHA differs')
  original=prefix(a.source.read_text());generated='#include <cuda_runtime.h>\n'+original+'\n'+(HERE/'skinny.cuh').read_text();(out/'kernels.cu').write_text(generated)
  cases=plan();save(out/'plan.json',cases);(out/'plan.h').write_text(header(cases));m.update(generation_complete=True,strict_cases=len(cases),timed_shapes=sum(x['timed'] for x in cases),original_prefix_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=sha(out/'kernels.cu'))
  if not a.compile:m['status']='GENERATED_GPU_NOT_RUN';save(out/'manifest.json',m);return 0
  def call(cmd,name):
   r=subprocess.run(list(map(str,cmd)),capture_output=True,text=True);(out/name).write_text(r.stdout+r.stderr)
   m.setdefault('commands',[]).append(dict(argv=list(map(str,cmd)),exit_code=r.returncode,log=name,log_sha256=sha(out/name)))
   if r.returncode:raise RuntimeError(name+' failed')
   return r.stdout+r.stderr
  m['nvcc_version']=call([a.nvcc,'--version'],'nvcc-version.log');common=[a.nvcc,'-std=c++17','-arch='+a.arch,*FLAGS]
  log=call(common+['-cubin','-Xptxas=-v,-warn-spills',out/'kernels.cu','-o',out/'kernels.cubin'],'kernels.build.log');m['actual_resources']=resource(log)
  call(common+['-ptx',out/'kernels.cu','-o',out/'kernels.ptx'],'ptx.build.log')
  if shutil.which(a.cuobjdump):call([a.cuobjdump,'--dump-sass',out/'kernels.cubin'],'kernels.sass.txt')
  call([a.nvcc,'-std=c++17','-O3','-I',out,'-I',HERE,HERE/'bench.cu','-lcuda','-o',out/'skinny-bench'],'runner.build.log')
  m.update(complete=True,status='COMPILED_GPU_NOT_RUN',artifacts={n:sha(out/n) for n in ARTIFACTS});save(out/'manifest.json',m);save(out/'w4a16-build-ready.json',dict(schema='q35-w4a16-skinny-ready/1',manifest_sha256=sha(out/'manifest.json'),artifacts=m['artifacts'],GPU='NOT_RUN'));return 0
 except Exception as error:m.update(complete=False,status='BUILD_ERROR_NO_GPU_AUTHORIZATION',error=repr(error));save(out/'manifest.json',m);return 1
if __name__=='__main__':raise SystemExit(main())
