#!/usr/bin/env python3
"""Default plan, explicit --execute for the private synthetic GPU runner."""
import argparse,importlib.util,json,math,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('q6_prepack_build',HERE/'build.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
NS=(1,2,4,8,16,32,64)
def strict_ns(byte_planes=False,aligned_blocks=False):return NS+((7,14,28,9,17,33)if byte_planes or aligned_blocks else())
def timed_ns(byte_planes=False,aligned_blocks=False):return NS+((7,14,28)if byte_planes or aligned_blocks else())
def layout(byte_planes=False,aligned_blocks=False):
 if byte_planes and aligned_blocks:raise ValueError('layout options are mutually exclusive')
 return 'byte_planes_210'if byte_planes else'aligned_blocks_212'if aligned_blocks else'int8_planes_274'
def shapes(byte_planes=False,aligned_blocks=False):return {f'k{k}-m{65 if p%2 else 63}-n{n}-p{p}'for n in strict_ns(byte_planes,aligned_blocks)for p in range(4)for k in(256,512,5120)}|{f'k5120-m248320-n{n}-p0'for n in timed_ns(byte_planes,aligned_blocks)}
def verify_build(d,byte_planes=False,aligned_blocks=False):
 m=json.loads((d/'manifest.json').read_text());p=json.loads((d/'offline-permit.json').read_text())
 if m.get('schema')!='q35-q6-prepack-build/1'or m.get('complete')is not True or m.get('flags')!=B.FLAGS or m.get('source_sha256')!=B.SOURCE_SHA or B.digest(m['source'])!=B.SOURCE_SHA:raise ValueError('not the actual pinned offline build')
 if m.get('layout')!=layout(byte_planes,aligned_blocks)or m.get('strict_cases')!=len(shapes(byte_planes,aligned_blocks))or m.get('timed_shapes')!=len(timed_ns(byte_planes,aligned_blocks))or m.get('runner_defines')!=(['TS_Q6_BYTE_PLANES=1']if byte_planes else['TS_Q6_ALIGNED_BLOCKS=1']if aligned_blocks else[]):raise ValueError('explicit selected layout/build plan mismatch')
 if m.get('tools')!={n:B.digest(HERE/n)for n in(*B.TOOLS,'prepack.cuh')}:raise ValueError('tools changed')
 if set(m.get('artifacts',{}))!=set(B.ARTIFACTS)or any(B.digest(d/n)!=v for n,v in m['artifacts'].items())or p!={'manifest_sha256':B.digest(d/'manifest.json'),'artifacts':m['artifacts']}:raise ValueError('build artifact/real marker changed')
 if m.get('generated_sha256')!=B.digest(d/'kernels.cu')or(d/'kernels.cu').read_text()!=B.generate(Path(m['source']).read_text(),byte_planes,aligned_blocks):raise ValueError('compiled generated arithmetic changed')
 actual=B.resources((d/'cubin.build.log').read_text())
 if actual!=m.get('resources'):raise ValueError('actual ptxas resource record changed')
 return m
def validate(events,result,bench,rounds,iterations,sweep,byte_planes=False,aligned_blocks=False):
 ns=timed_ns(byte_planes,aligned_blocks);shape_ids=shapes(byte_planes,aligned_blocks)
 if result.get('complete')is not True or result.get('strict_pass')is not True or result.get('layout')!=layout(byte_planes,aligned_blocks)or result.get('strict_cases')!=len(shape_ids)or result.get('timed_shapes')!=(len(ns)if bench else 0)or result.get('rounds')!=rounds or result.get('iterations')!=iterations or result.get('cache_sweep')is not sweep:raise ValueError('incomplete result matrix')
 groups={}
 allowed={'device','driver_resources','strict_pass','prepack_pass','pipeline_pass','ABBA_gpu_ms','cache_sweep_setup','cache_sweep','atomic_AA_DIAGNOSTIC_NOT_EQUIVALENCE_PASS'}
 for r in events:
  if not isinstance(r,dict)or r.get('event')not in allowed:raise ValueError('unknown event')
  groups.setdefault(r['event'],[]).append(r)
 if len(groups.get('device',[]))!=1:raise ValueError('actual device missing/duplicate')
 device=groups['device'][0]
 if not isinstance(device.get('name'),str)or not device['name']or any(type(device.get(k))is not int or device[k]<=0 for k in('sms','driver','runtime')):raise ValueError('invalid actual device metadata')
 observed=groups.get('driver_resources',[])
 expected={'ts_q35_quantize_q81_f32','ts_bench_q6_prepack','ts_q35_gemv_q6k_1','ts_bench_gemv_q6packed_1'}|{f'{p}_{n}'for p in('ts_q35_gemm_q6k','ts_bench_gemm_q6packed')for n in(1,2,4,8)}
 if len(observed)!=12 or{r.get('symbol')for r in observed}!=expected:raise ValueError('driver resource matrix incomplete')
 for r in observed:
  if any(type(r.get(k))is not int for k in('registers','local_bytes','static_shared_bytes','active_ctas','max_block_threads','block_threads'))or not 0<r['registers']<=255 or r['local_bytes']!=0 or r['active_ctas']<1 or r['max_block_threads']<r['block_threads']:raise ValueError('driver resource rejected')
 strict=groups.get('strict_pass',[])
 if len(strict)!=len(shape_ids)or {r.get('shape')for r in strict}!=shape_ids or any(any(r.get(k)is not True for k in('full_bits','dual_poison_AA','guards'))or r.get('split_k')!=1 for r in strict):raise ValueError('whole strict bit/AA/guard matrix incomplete')
 packs=groups.get('prepack_pass',[])
 for r in packs:
  shape=re.fullmatch(r'k(\d+)-m(\d+)-n(\d+)-p(\d+)',r.get('shape',''))
  if shape is None:raise ValueError('invalid prepack shape')
  k,m,n,p=map(int,shape.groups());passes=r.get('prepack_GPU_ms_two_poison_passes')
  if any(r.get(f)is not True for f in('whole_bytes_CPU_decoder','dual_poison','guards'))or r.get('raw_bytes')!=m*(k//256)*210 or r.get('packed_bytes')!=m*(k//256)*(210 if byte_planes else 212 if aligned_blocks else 274)or r.get('extra_bytes')!=m*(k//256)*(0 if byte_planes else 2 if aligned_blocks else 64)or not isinstance(passes,list)or len(passes)!=2 or any(type(x)not in(int,float)or not math.isfinite(x)or x<=0 for x in passes)or r.get('load_once_gpu_ms')!=passes[1]or r.get('load_time_scope')!='prepack_kernel_only_second_pass_not_full_model_load':raise ValueError('packed payload/load cost evidence invalid')
 from collections import Counter
 expected_packs=Counter(shape_ids)
 if bench:expected_packs.update(f'k5120-m248320-n{n}-p0'for n in ns)
 if Counter(r.get('shape')for r in packs)!=expected_packs:raise ValueError('pack matrix incomplete')
 pipeline=groups.get('pipeline_pass',[])
 if len(pipeline)!=(len(ns)if bench else 0)or{r.get('shape')for r in pipeline}!=({f'k5120-m248320-n{n}-p0'for n in ns}if bench else set())or any(r.get('full_bits_AA_guards')is not True for r in pipeline):raise ValueError('timed full pipeline gate incomplete')
 times=groups.get('ABBA_gpu_ms',[]);ids=set()
 for r in times:
  key=(r.get('n'),r.get('round'))
  if key in ids or key[0]not in ns or type(key[1])is not int or not 0<=key[1]<rounds or r.get('shape')!=f'k5120-m248320-n{key[0]}-p0'or r.get('scope')!='original_Q81_quant_plus_zero_if_needed_plus_Q6_matmul'or r.get('fullbits_each_half')is not True or r.get('split_k')!=1:raise ValueError('bad/duplicate ABBA identity')
  if any(type(r.get(k))not in(int,float)or not math.isfinite(r[k])or r[k]<=0 for k in('A1_oracle_ms','B1_candidate_ms','B2_candidate_ms','A2_oracle_ms')):raise ValueError('invalid event interval')
  ids.add(key)
 if ids!=({(n,r)for n in ns for r in range(rounds)}if bench else set()):raise ValueError('ABBA matrix incomplete')
 first_time=next((i for i,r in enumerate(events)if r.get('event')=='ABBA_gpu_ms'),len(events))
 if any(r.get('event')=='strict_pass'for r in events[first_time:]):raise ValueError('timing preceded complete strict matrix')
 for i,r in enumerate(events):
  if r.get('event')=='ABBA_gpu_ms'and not any(x.get('event')=='pipeline_pass'and x.get('shape')==r['shape']for x in events[:i]):raise ValueError('timed shape not gated')
 setup=groups.get('cache_sweep_setup',[]);sweeps=groups.get('cache_sweep',[])
 if not sweep:
  if setup or sweeps or result.get('cache_sweeps')!=0:raise ValueError('warm mode unexpectedly swept')
 else:
  if len(setup)!=1:raise ValueError('cache setup missing/duplicate')
  s=setup[0];l2=s.get('driver_l2_bytes');size=s.get('sweep_bytes')
  if type(l2)is not int or l2<=0 or type(size)is not int or size<2*l2 or size%256 or s.get('words')!=size//4 or s.get('method')!='volatile_asm_ld_global_cg_u32_grid_stride_per_block_u64_sum_and_count'or s.get('physical_cold_guaranteed')is not False or s.get('excluded_from_event_interval')is not True:raise ValueError('actual cache control invalid')
  ids=set()
  for i,r in enumerate(sweeps,1):
   key=(r.get('shape'),r.get('round'),r.get('half'))
   if key in ids or r.get('index')!=i or r.get('words_read')!=size//4 or r.get('checksum')!=(size//4)*0x5a5a5a5a or r.get('driver_l2_bytes')!=l2 or r.get('sweep_bytes')!=size or any(r.get(k)is not True for k in('guards','checksum_verified','synchronized_before_event','excluded_from_event_interval')):raise ValueError('cache sweep checksum/identity incomplete')
   ids.add(key)
  expected={(f'k5120-m248320-n{n}-p0',r,h)for n in ns for r in range(rounds)for h in('A1','B1','B2','A2')}
  if ids!=expected or result.get('cache_sweeps')!=len(expected):raise ValueError('cache sweep half matrix incomplete')
 return dict(layout=layout(byte_planes,aligned_blocks),strict=len(shape_ids),ABBA=len(times),cache_sweeps=len(sweeps),actual_BBD_and_real_model='NOT_RUN',production_atomic_equivalence='NOT_ESTABLISHED')
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',required=True,type=Path);p.add_argument('--name',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--bench',action='store_true');choice=p.add_mutually_exclusive_group();choice.add_argument('--byte-planes',action='store_true');choice.add_argument('--aligned-blocks',action='store_true');p.add_argument('--cache-sweep',action='store_true');p.add_argument('--rounds',type=int,default=10);p.add_argument('--iterations',type=int);p.add_argument('--timeout',type=int,default=900);p.add_argument('--atomic-aa',action='store_true');a=p.parse_args(argv)
 if a.iterations is None:a.iterations=1 if a.cache_sweep else 3
 if Path(a.name).name!=a.name or a.name in('.','..')or a.rounds<10 or min(a.iterations,a.timeout)<1 or(a.cache_sweep and(not a.bench or a.iterations!=1)):p.error('bounded fresh name; >=10 rounds; sweep needs bench/iterations1')
 d=B.evidence(a.build);out=d/a.name;out.mkdir(exist_ok=False);state=dict(schema='q35-q6-prepack-run/1',complete=False,gpu_invoked=False,status='PLAN_GPU_NOT_RUN',layout=layout(a.byte_planes,a.aligned_blocks),bench=a.bench,cache_sweep=a.cache_sweep,atomic_AA_diagnostic_only=a.atomic_aa,strict_cases=len(shapes(a.byte_planes,a.aligned_blocks)),rounds=a.rounds,iterations=a.iterations,production_integration=False);B.save(out/'run.json',state)
 if not a.execute:print(json.dumps(state));return 0
 try:
  m=verify_build(d,a.byte_planes,a.aligned_blocks);before=B.digest(d/'manifest.json');cmd=[str(d/B.RUNNER),str(d),str(out),'--bench'if a.bench else'--strict','--rounds',str(a.rounds),'--iterations',str(a.iterations)]+(['--cache-sweep']if a.cache_sweep else[])+(['--atomic-aa']if a.atomic_aa else[]);state.update(command=cmd,gpu_invoked=True,build_manifest_sha256=before);B.save(out/'run.json',state)
  with(out/'runner.jsonl').open('x')as f:code,_=B.command(cmd,a.timeout,stdout=f)
  state['exit_code']=code
  if code:raise ValueError('runner failed; raw preserved')
  result=json.loads((out/'result.json').read_text());events=[json.loads(x)for x in(out/'runner.jsonl').read_text().splitlines()if x.strip()];observed=validate(events,result,a.bench,a.rounds,a.iterations,a.cache_sweep,a.byte_planes,a.aligned_blocks);verify_build(d,a.byte_planes,a.aligned_blocks)
  if B.digest(d/'manifest.json')!=before:raise ValueError('build changed during GPU run')
  state.update(complete=True,status='STRICT_SYNTHETIC_BITS_PASS_NOT_MODEL_GATE',observations=observed,runner_log_sha256=B.digest(out/'runner.jsonl'));B.save(out/'run.json',state);print(json.dumps(state));return 0
 except BaseException as e:
  state.update(complete=False,status='NOT_PASS',error=repr(e));B.save(out/'run.json',state)
  if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state));return 1
if __name__=='__main__':raise SystemExit(main())
