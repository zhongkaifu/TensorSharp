#!/usr/bin/env python3
"""Offline standalone W8G32 build. Default generate-only; never launches a GPU."""
import argparse,hashlib,importlib.util,json,os,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('w8_process',HERE/'process.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
FLAGS=['-std=c++17','-arch=sm_120a','-O3','--fmad=true','--ftz=false','-lineinfo']
RUNNER='w8-g32-head-bench'
CONTRACT_SHA='03fc1a0681beb3a4a0d9ebfdde3980d670b49534008db7e3284c3bc551875f7e'
CLOSURE_SHA='66b48ffe40fc820ac4cfab5c9cfcc32ee93042514c1944dafd7a16a44f2e7c9c'
OWN=('build.py','run.py','process.py','bench.cu','kernels.cuh','reference.h','official.h','test_build.py','USAGE.txt','numeric-contract.json','official-source-manifest.json')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def evidence(p):
 p=Path(p).absolute()
 if p!=Path(os.path.normpath(p))or p.resolve()!=p or not any(x in p.parents for x in(ROOT/'artifacts',ROOT/'docs/validation')):raise ValueError('fresh nonalias ignored evidence path required')
 return p
def tool_inventory():
 if digest(HERE/'numeric-contract.json')!=CONTRACT_SHA or digest(HERE/'official-source-manifest.json')!=CLOSURE_SHA:raise ValueError('preregistered contract or official closure changed')
 m=json.loads((HERE/'official-source-manifest.json').read_text());paths=set()
 for r in m['files']:
  p=Path(r['path'])
  if p.is_absolute()or '..'in p.parts or str(p)in paths or len(p.parts)<2 or p.parts[0]!='upstream':raise ValueError('unsafe/duplicate official closure path')
  paths.add(str(p));f=HERE/p
  if f.is_symlink()or not f.is_file()or f.stat().st_size!=r['bytes']or (f.stat().st_mode&0o777)!=int(r['mode'],8)or digest(f)!=r['sha256']:raise ValueError('exact official bytes missing or changed')
 actual={str(p.relative_to(HERE))for p in(HERE/'upstream').rglob('*')if p.is_file()or p.is_symlink()}
 if actual!=paths:raise ValueError('official closure extra/missing file')
 return {n:digest(HERE/n)for n in(*OWN,*sorted(paths))}
def resources(text):
 out={}
 for name,body in re.findall(r"ptxas info\s+: Compiling entry function '([^']+)'(.*?)(?=ptxas info\s+: Compiling entry function|\Z)",text,re.S):
  r=re.search(r'Used (\d+) registers',body);s=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body);sm=re.search(r'(\d+) bytes smem',body)
  if not r or not s or name in out:raise ValueError('missing/duplicate actual register/stack/spill evidence')
  out[name]=dict(registers=int(r[1]),stack=int(s[1]),spill_stores=int(s[2]),spill_loads=int(s[3]),static_shared=int(sm[1])if sm else 0)
 own={'ts_w8_cast','ts_w8_reference','ts_w8_simt','ts_w8_sweep'}|{f'ts_w8_tc_{t}'for t in(1,7,8,14,16,28,32,33)}
 if not own.issubset(out)or len(out)!=20 or sum('w8_small_t_mma_kernel'in n for n in out)!=8:raise ValueError('actual 12 owned plus 8 official kernel resources required')
 if any(not 0<r['registers']<=255 for r in out.values()):raise ValueError('invalid registers')
 return out

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--compile',action='store_true');p.add_argument('--nvcc',default='/usr/local/cuda-13.2/bin/nvcc');p.add_argument('--timeout',type=int,default=600);a=p.parse_args(argv)
 if any(os.environ.get(k,'').strip()for k in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):p.error('implicit compiler flags forbidden')
 out=evidence(a.out)
 if out.exists()or a.timeout<=0:p.error('fresh output and positive timeout required')
 tools=tool_inventory();out.mkdir(parents=True);state=dict(schema='q35-w8-g32-build/1',complete=False,status='SOURCE_ONLY_GPU_NOT_RUN',flags=FLAGS,tools=tools,contract_sha256=CONTRACT_SHA,official_closure_sha256=CLOSURE_SHA,official_scope='exact_e600_headers_NOT_actual_Ignis_runtime',model_quality='NOT_RUN',production_integration=False)
 save(out/'manifest.json',state)
 try:
  # Freeze the full translation unit/closure beside the ELF. It is the exact input
  # compiled by nvcc, independently hashable after build and GPU execution.
  for name in tools:
   dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((HERE/name).read_bytes())
  state['source']={n:digest(out/'source'/n)for n in tools}
  if not a.compile:save(out/'manifest.json',state);print(json.dumps(state));return 0
  code,t=P.command([a.nvcc,'--version'],30);(out/'nvcc-version.log').write_text(t)
  if code:raise RuntimeError('nvcc version failed')
  state['nvcc_version']=t;cmd=[a.nvcc,*FLAGS,'-Xptxas=-v,-warn-spills','-I',str(out/'source'),'-I',str(out/'source/upstream/kernel/vendor/src'),str(out/'source/bench.cu'),'-o',str(out/RUNNER)]
  state['command']=cmd;save(out/'manifest.json',state);code,t=P.command(cmd,a.timeout);(out/'build.log').write_text(t);state['exit_code']=code
  if code:raise RuntimeError('offline compiler failed')
  state['resources']=resources(t)
  if tool_inventory()!=tools or state['source']!={n:digest(out/'source'/n)for n in tools}:raise ValueError('build inputs changed')
  state.update(complete=True,status='OFFLINE_COMPILED_GPU_NOT_RUN',artifacts={RUNNER:digest(out/RUNNER),'build.log':digest(out/'build.log')});save(out/'manifest.json',state)
  save(out/'offline-permit.json',dict(manifest_sha256=digest(out/'manifest.json'),artifacts=state['artifacts']));print(json.dumps(state));return 0
 except BaseException as e:
  state.update(complete=False,status='BUILD_NOT_PASS_GPU_NOT_RUN',error=repr(e));save(out/'manifest.json',state)
  if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state));return 1
if __name__=='__main__':raise SystemExit(main())
