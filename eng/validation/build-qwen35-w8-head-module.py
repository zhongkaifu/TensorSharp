#!/usr/bin/env python3
"""Generate-only by default. Explicit offline NVCC build; never launches a GPU."""
import argparse,hashlib,importlib.util,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];NATIVE=ROOT/'TensorSharp.Backends.Cuda/native/w8'
HELPER=ROOT/'eng/validation/Qwen35W8G32HeadBench/process.py'
HELPER_SHA='1cc878e7f6792b2886de71e4d572ea63dc7e65e78aae33eb048591788c18d8b0'
CLOSURE_SHA='66b48ffe40fc820ac4cfab5c9cfcc32ee93042514c1944dafd7a16a44f2e7c9c'
FLAGS=['-std=c++17','-arch=sm_120a','-O3','--fmad=true','--ftz=false','-lineinfo']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def evidence(p):
 p=Path(p).absolute()
 if p!=Path(os.path.normpath(p))or p.resolve()!=p or p.exists()or not any(x in p.parents for x in(ROOT/'artifacts',ROOT/'docs/validation')):raise ValueError('fresh nonalias ignored output required')
 return p
def inventory():
 if sha(HELPER)!=HELPER_SHA or sha(NATIVE/'official-source-manifest.json')!=CLOSURE_SHA:raise ValueError('exact process helper/original closure changed')
 m=json.loads((NATIVE/'official-source-manifest.json').read_text());files={}
 for row in m['files']:
  p=Path(row['path'])
  if p.is_absolute()or '..'in p.parts or str(p)in files or NATIVE.joinpath(p).resolve()!=NATIVE.joinpath(p):raise ValueError('unsafe/duplicate/alias original header path')
  f=NATIVE/p
  if f.stat().st_size!=row['bytes']or sha(f)!=row['sha256']or(f.stat().st_mode&0o777)!=int(row['mode'],8):raise ValueError('original closure bytes/mode changed')
  files[str(p)]=row['sha256']
 actual={str(p.relative_to(NATIVE))for p in(NATIVE/'upstream').rglob('*')if p.is_file()or p.is_symlink()}
 if actual!=set(files):raise ValueError('extra/missing original header/license')
 return files|{n:sha(NATIVE/n)for n in('tensorsharp_w8g32_head.cu','official-source-manifest.json','README.md')}
def resources(text):
 found={}
 for name,body in re.findall(r"ptxas info\s+: Compiling entry function '([^']+)'(.*?)(?=ptxas info\s+: Compiling entry function|\Z)",text,re.S):
  regs=re.search(r'Used (\d+) registers',body);spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body)
  if not regs or not spill or name in found:raise ValueError('actual compiler resource fields absent/duplicate')
  found[name]=dict(registers=int(regs[1]),stack=int(spill[1]),spill_stores=int(spill[2]),spill_loads=int(spill[3]))
 if len(found)!=34 or 'ts_w8g32_cast_f32_bf16'not in found or sum('w8_small_t_mma_kernel'in x for x in found)!=33:raise ValueError('34 actual native entry resources required')
 return found

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--compile',action='store_true');p.add_argument('--nvcc',default='/usr/local/cuda-13.2/bin/nvcc');p.add_argument('--timeout',type=int,default=1200);a=p.parse_args(argv)
 if a.timeout<=0 or any(os.environ.get(k,'').strip()for k in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):p.error('positive timeout and no implicit nvcc flags required')
 out=evidence(a.out);files=inventory();out.mkdir(parents=True);state=dict(schema='qwen35-original-W8-head-module-build/1',complete=False,status='GENERATED_ONLY_NATIVE_AND_GPU_NOT_RUN',source=files,flags=FLAGS,helper_sha256=HELPER_SHA,closure_sha256=CLOSURE_SHA,model_quality='NOT_RUN',old_Q35_or_ggml_integrated=False)
 save(out/'manifest.json',state)
 try:
  source=out/'source'
  for n in files:dst=source/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((NATIVE/n).read_bytes())
  state['builder_sha256']=sha(__file__)
  if not a.compile:save(out/'manifest.json',state);print(json.dumps(state));return 0
  s=importlib.util.spec_from_file_location('W8_frozen_process',HELPER);P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
  code,text=P.command([a.nvcc,'--version'],30);(out/'nvcc-version.log').write_text(text)
  if code:raise RuntimeError('actual nvcc unavailable')
  state['nvcc_version']=text;state['commands']=[]
  for kind in('fatbin','ptx'):
   target=out/('tensorsharp_w8g32_head.'+kind);cmd=[a.nvcc,*FLAGS,'-'+kind,*(['-Xptxas=-v,-warn-spills']if kind=='fatbin'else[]),'-I',str(source/'upstream/kernel/vendor/src'),str(source/'tensorsharp_w8g32_head.cu'),'-o',str(target)]
   code,text=P.command(cmd,a.timeout);(out/(kind+'.log')).write_text(text);state['commands'].append(dict(argv=cmd,exit_code=code,log_sha256=sha(out/(kind+'.log'))));save(out/'manifest.json',state)
   if code:raise RuntimeError('actual offline '+kind+' failed')
   if kind=='fatbin':state['resources']=resources(text)
  (out/'tensorsharp_w8g32_head.fatbin.sha256').write_text(sha(out/'tensorsharp_w8g32_head.ptx')+'\n')
  if inventory()!=files or any(sha(source/n)!=h for n,h in files.items())or sha(__file__)!=state['builder_sha256']:raise ValueError('native/build inputs changed')
  state.update(complete=True,status='OFFLINE_COMPILED_CSHARP_GPU_NOT_RUN',artifacts={p.name:sha(p)for p in out.iterdir()if p.is_file()and p.name!='manifest.json'});save(out/'manifest.json',state);print(json.dumps(state));return 0
 except BaseException as e:
  state.update(complete=False,status='BUILD_NOT_PASS',error=repr(e));save(out/'manifest.json',state)
  if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state));return 1
if __name__=='__main__':raise SystemExit(main())
