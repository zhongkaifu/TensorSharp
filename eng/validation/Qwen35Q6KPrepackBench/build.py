#!/usr/bin/env python3
"""Offline Q6_K exact integer predecode experiment. Never launches a GPU kernel."""
import argparse,hashlib,json,os,re,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
SOURCE=ROOT/'TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu'
SOURCE_SHA='28ede74605b4edc798744933ca17cf4a18858bb52fb09080f4686459b6fc40b0'
Q6_SHA='4a76e503ced34d2fa2b72933e64eadef22dbecd69ae9e0bd6edb98f2b80cba89'
FLAGS=['-O3','--extra-device-vectorization']; RUNNER='q6k-prepack-bench'
TOOLS=('build.py','run.py','bench.cu','checks.h','cache-sweep.h')
ARTIFACTS=('kernels.cu','kernels.cubin','kernels.ptx',RUNNER)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def evidence(p):
 p=Path(p).absolute()
 if p!=Path(os.path.normpath(p)) or p.resolve()!=p or not any(x in p.parents for x in(ROOT/'artifacts',ROOT/'docs/validation')):raise ValueError('fresh nonalias ignored evidence path required')
 return p
def command(argv,timeout,stdout=None):
 """Own new-session group only; registration is inside cleanup scope, child unmasks."""
 proc=None;oldmask=None;handlers={};f=stdout
 def interrupted(s,frame):raise KeyboardInterrupt('signal '+str(s))
 try:
  for sig in(signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
  oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
  proc=subprocess.Popen(argv,stdout=f or subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask))
  signal.pthread_sigmask(signal.SIG_SETMASK,oldmask);oldmask=None
  if f:return proc.wait(timeout=timeout),''
  text=proc.communicate(timeout=timeout)[0];return proc.returncode,text
 except BaseException:
  if proc is not None and proc.returncode is None:
   for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
   try:
    if os.getpgid(proc.pid)==proc.pid:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   except ProcessLookupError:proc.wait()
  raise
 finally:
  if oldmask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
  for sig,handler in handlers.items():signal.signal(sig,handler)
def section(source):
 if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('original production source changed')
 a=source.rfind('// ---------------------------------------------------------------------------',0,source.index('// LM head: Q6_K'))
 q=source[a:source.index('// dst[r] = src[idx[r]]',a)]
 if hashlib.sha256(q.encode()).hexdigest()!=Q6_SHA:raise ValueError('original Q6 oracle changed')
 return source[:source.index('// ---------------------------------------------------------------------------\n// ggml block_nvfp4')],q
def replace_once(s,a,b):
 if s.count(a)!=1:raise ValueError('source delta does not match exactly once: '+a[:60])
 return s.replace(a,b,1)
def generate(source,byte_planes=False,aligned_blocks=False):
 if byte_planes and aligned_blocks:raise ValueError('layout options are mutually exclusive')
 prefix,q=section(source);c=q
 # Keep all integer MMA, floating products, K traversal, barriers, and split-K stores.
 c=c.replace('ts_q35_gemv_q6k','ts_bench_gemv_q6packed').replace('ts_q35_gemm_q6k','ts_bench_gemm_q6packed').replace('ts_q35_mma_s8','ts_bench_mma_s8')
 c=c.replace('TS_Q35_GEMV_Q6K','TS_BENCH_GEMV_Q6K').replace('TS_Q35_GEMM_Q6K','TS_BENCH_GEMM_Q6K')
 if aligned_blocks:
  c=replace_once(c,'const uint8_t* sb = row + (long long)sbIdx * 210;','const uint8_t* sb = row + (long long)sbIdx * 212;')
  c=replace_once(c,'const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 210;','const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 212;')
 if byte_planes:
  c=replace_once(c,'const uint8_t* sb = row + (long long)sbIdx * 210;\n        const uint8_t* ql = sb + 64 * h;\n        const uint8_t* qh = sb + 128 + 32 * h;\n        const int8_t* sc = (const int8_t*)(sb + 192) + 8 * h;\n        const float d = __half2float(*(const half*)(sb + 208));',
 '''const long long totalBlocks = (long long)outDim * (k / 256), block = (long long)r * (k / 256) + sbIdx;
        const uint8_t* ql = w + block * 128 + 64 * h;
        const uint8_t* qh = w + totalBlocks * 128 + block * 64 + 32 * h;
        const int8_t* sc = (const int8_t*)(w + totalBlocks * 192 + block * 16) + 8 * h;
        const float d = __half2float(((const half*)(w + totalBlocks * 208))[block]);''')
  c=replace_once(c,'const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 210;\n                const uint8_t* ql = blk + 64 * h;\n                const uint8_t* qh = blk + 128 + 32 * h;',
 '''const long long totalBlocks = (long long)outDim * (k / 256), block = (long long)gr * (k / 256) + sb;
                const uint8_t* ql = w + block * 128 + 64 * h;
                const uint8_t* qh = w + totalBlocks * 128 + block * 64 + 32 * h;''')
  c=replace_once(c,'const float d = __half2float(*(const half*)(blk + 208));\n                    const int8_t* sc = (const int8_t*)(blk + 192) + 8 * h;',
 '''const float d = __half2float(((const half*)(w + totalBlocks * 208))[block]);
                    const int8_t* sc = (const int8_t*)(w + totalBlocks * 192 + block * 16) + 8 * h;''')
 if byte_planes or aligned_blocks:
  for name,offset in [('a','ql + l0'),('b','ql + 32 + l0'),('c','qh + l0')]:
   old=f'const uint32_t {name} = *(const uint16_t*)({offset}) | ((uint32_t)*(const uint16_t*)({offset} + 2) << 16);'
   new=f'const uint32_t {name} = *(const uint32_t*)({offset});'
   if c.count(old)!=2:raise ValueError('expected same aligned load replacement in both consumers')
   c=c.replace(old,new)
  define='TS_Q6_BYTE_PLANES'if byte_planes else'TS_Q6_ALIGNED_BLOCKS'
  return '#include <cuda_runtime.h>\n#define '+define+' 1\n'+prefix+q+'\n// Private byte-plane/212-stride loads and aligned integer decode only.\n'+c+(HERE/'prepack.cuh').read_text()
 c=replace_once(c,'const uint8_t* row = w + (long long)r * rowBytes;','const uint8_t* row = w + (long long)r * k;')
 c=replace_once(c,'const uint8_t* sb = row + (long long)sbIdx * 210;\n        const uint8_t* ql = sb + 64 * h;\n        const uint8_t* qh = sb + 128 + 32 * h;\n        const int8_t* sc = (const int8_t*)(sb + 192) + 8 * h;\n        const float d = __half2float(*(const half*)(sb + 208));',
 '''const int8_t* sc = (const int8_t*)(w + (long long)outDim * k) + (long long)r * (k / 16) + sbIdx * 16 + 8 * h;
        const half* dp = (const half*)(w + (long long)outDim * k + (long long)outDim * (k / 16));
        const float d = __half2float(dp[(long long)r * (k / 256) + sbIdx]);''')
 a=c.index('            const uint32_t a =',c.index('ts_bench_gemv_q6packed'));b=c.index('#pragma unroll\n            for (int m = 0;',a)
 old=c[a:b];new='            int wq[4];\n#pragma unroll\n            for (int m = 0; m < 4; ++m) wq[m] = *(const int*)(row + v0 + 32 * m + l0);\n'
 c=c[:a]+new+c[b:]
 a=c.index('                const uint8_t* blk =');b=c.index('\n                if (l0 == 0)',a)
 c=c[:a]+'''                const int8_t* packed = (const int8_t*)w + (long long)gr * k + sb * 256 + 128 * h;
#pragma unroll
                for (int m = 0; m < 4; ++m) q[m] = *(const uint32_t*)(packed + 32 * m + l0);'''+c[b:]
 c=replace_once(c,'const float d = __half2float(*(const half*)(blk + 208));\n                    const int8_t* sc = (const int8_t*)(blk + 192) + 8 * h;',
 '''const half* dp = (const half*)(w + (long long)outDim * k + (long long)outDim * (k / 16));
                    const float d = __half2float(dp[(long long)gr * (k / 256) + sb]);
                    const int8_t* sc = (const int8_t*)(w + (long long)outDim * k) + (long long)gr * (k / 16) + sb * 16 + 8 * h;''')
 # Save a machine-auditable delta; oracle body above is copied byte-for-byte.
 extra=(HERE/'prepack.cuh').read_text()
 return '#include <cuda_runtime.h>\n'+prefix+q+'\n// Private exact code-plane loading candidate; original oracle above.\n'+c+extra
def resources(text):
 rows={}
 for name,body in re.findall(r"ptxas info\s+: Compiling entry function '([^']+)'(.*?)(?=ptxas info\s+: Compiling entry function|\Z)",text,re.S):
  r=re.search(r'Used (\d+) registers',body);s=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',body);sm=re.search(r'(\d+) bytes smem',body)
  if not r or not s:raise ValueError('missing actual ptxas registers/stack/spill '+name)
  rows[name]=dict(registers=int(r[1]),stack=int(s[1]),spill_stores=int(s[2]),spill_loads=int(s[3]),static_shared=int(sm[1])if sm else 0)
 expected={'ts_q35_quantize_q81_f32','ts_q35_gemv_q6k_1','ts_bench_gemv_q6packed_1','ts_bench_q6_prepack','ts_bench_q6_cache_sweep'}|{f'{prefix}_{n}'for prefix in('ts_q35_gemm_q6k','ts_bench_gemm_q6packed')for n in(1,2,4,8)}
 if set(rows)!=expected:raise ValueError('missing/extra compiled entry resources')
 if any(not 0<x['registers']<=255 or any(x[k]for k in('stack','spill_stores','spill_loads'))for x in rows.values()):raise ValueError('no-stack/no-spill resource policy rejected')
 return rows
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--generate-only',action='store_true');choice=p.add_mutually_exclusive_group();choice.add_argument('--byte-planes',action='store_true');choice.add_argument('--aligned-blocks',action='store_true');p.add_argument('--nvcc',default='nvcc');p.add_argument('--timeout',type=int,default=600);a=p.parse_args(argv)
 if any(os.environ.get(k,'').strip()for k in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):p.error('implicit NVCC flags forbidden')
 out=evidence(a.out)
 if out.exists()or a.timeout<=0:p.error('fresh output/positive timeout required')
 out.mkdir(parents=True);tools={n:digest(HERE/n)for n in(*TOOLS,'prepack.cuh')};state=dict(schema='q35-q6-prepack-build/1',complete=False,source=str(a.source.resolve()),source_sha256=digest(a.source),oracle_Q6_sha256=Q6_SHA,tools=tools,flags=FLAGS,oracle_scope='freshly_recompiled_original_NOT_actual_BBD',gpu_quality='NOT_RUN',production_integration=False,model_shape=dict(vocab=248320,k=5120,original_bytes=1042944000,packed_bytes=1360793600,increment_bytes=317849600,increment_MiB=303.125,old_plus_packed_bytes_in_this_micro=2403737600,new_allocation_during_conversion_bytes=1360793600,production_memory_fit='NOT_TESTED'))
 state.update(layout='byte_planes_210'if a.byte_planes else'int8_planes_274',strict_cases=166 if a.byte_planes else 91,timed_shapes=10 if a.byte_planes else 7,runner_defines=['TS_Q6_BYTE_PLANES=1']if a.byte_planes else[])
 if a.byte_planes:state['model_shape'].update(packed_bytes=1042944000,increment_bytes=0,increment_MiB=0,old_plus_packed_bytes_in_this_micro=2085888000,new_allocation_during_conversion_bytes=1042944000)
 if a.aligned_blocks:
  state.update(layout='aligned_blocks_212',strict_cases=166,timed_shapes=10,runner_defines=['TS_Q6_ALIGNED_BLOCKS=1'])
  state['model_shape'].update(packed_bytes=1052876800,increment_bytes=9932800,increment_MiB=9.47265625,old_plus_packed_bytes_in_this_micro=2095820800,new_allocation_during_conversion_bytes=1052876800)
 save(out/'manifest.json',state)
 try:
  (out/'kernels.cu').write_text(generate(a.source.read_text(),a.byte_planes,a.aligned_blocks));state['generated_sha256']=digest(out/'kernels.cu')
  if a.generate_only:state['status']='GENERATED_ONLY_GPU_NOT_RUN';save(out/'manifest.json',state);print(json.dumps(state));return 0
  code,t=command([a.nvcc,'--version'],30);(out/'nvcc-version.log').write_text(t)
  if code:raise RuntimeError('nvcc --version failed')
  state['nvcc_version']=t
  for kind in('cubin','ptx'):
   cmd=[a.nvcc,'-std=c++17','-arch=sm_120a',*FLAGS,'-'+kind,*(['-Xptxas=-v,-warn-spills']if kind=='cubin'else[]),str(out/'kernels.cu'),'-o',str(out/('kernels.'+kind))]
   code,t=command(cmd,a.timeout);(out/(kind+'.build.log')).write_text(t);state[kind+'_command']=cmd
   if code:raise RuntimeError(kind+' compile failed')
   if kind=='cubin':state['resources']=resources(t)
  cmd=[a.nvcc,'-std=c++17','-O3',*['-D'+x for x in state['runner_defines']],'-I',str(HERE),str(HERE/'bench.cu'),'-lcuda','-o',str(out/RUNNER)];code,t=command(cmd,a.timeout);(out/'runner.build.log').write_text(t);state['runner_command']=cmd
  if code:raise RuntimeError('runner compile failed')
  if digest(a.source)!=state['source_sha256']or tools!={n:digest(HERE/n)for n in tools}:raise ValueError('source/tools changed during build')
  state.update(complete=True,status='OFFLINE_BUILT_GPU_NOT_RUN',artifacts={n:digest(out/n)for n in ARTIFACTS});save(out/'manifest.json',state);save(out/'offline-permit.json',dict(manifest_sha256=digest(out/'manifest.json'),artifacts=state['artifacts']));print(json.dumps(state));return 0
 except BaseException as e:
  state.update(complete=False,error=repr(e),status='BUILD_REJECTED_GPU_NOT_RUN');save(out/'manifest.json',state)
  if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state));return 1
if __name__=='__main__':raise SystemExit(main())
