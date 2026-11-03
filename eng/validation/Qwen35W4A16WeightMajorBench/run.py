#!/usr/bin/env python3
"""PLAN by default. Explicit --execute requires an operator-owned exclusive GPU window."""
import argparse,collections,json,math,os,signal,subprocess
from pathlib import Path
import build

def verify(directory):
 m=json.loads((directory/'manifest.json').read_text());p=json.loads((directory/'w4a16-build-ready.json').read_text())
 if m.get('schema')!='q35-w4a16-weightmajor-build/1' or m.get('complete') is not True or p.get('schema')!='q35-w4a16-weightmajor-ready/1' or p.get('manifest_sha256')!=build.sha(directory/'manifest.json'):raise ValueError('no matching completed build')
 if set(m.get('tools',{}))!=set(build.TOOLS) or set(m.get('artifacts',{}))!=set(build.ARTIFACTS) or p.get('artifacts')!=m['artifacts']:raise ValueError('incomplete tool/artifact identity')
 if m['source_sha256']!=build.SOURCE_SHA or build.sha(m['source'])!=build.SOURCE_SHA:raise ValueError('production oracle source changed')
 for n,digest in m['tools'].items():
  if build.sha(build.HERE/n)!=digest:raise ValueError('prototype changed: '+n)
 for n,digest in m['artifacts'].items():
  if build.sha(directory/n)!=digest:raise ValueError('build artifact changed: '+n)
 if type(m.get('integer_decode'))is not bool or m.get('flags')!=build.FLAGS+(['-DSA16_INTEGER_DECODE=1'] if m['integer_decode'] else []) or m.get('strict_cases')!=len(build.plan()):raise ValueError('build flags/plan changed')
 return m

def validate(result,events,bench,rounds,iterations,cache_sweep=False):
 plan=build.plan();timed=[x for x in plan if x['timed']];mode='eviction-control' if cache_sweep else 'warm'
 if cache_sweep and (not bench or iterations!=1):raise ValueError('eviction-control needs bench and iterations1')
 if result.get('cache_mode')!=mode:raise ValueError('actual timing cache mode differs')
 if result.get('schema')!='q35-w4a16-weightmajor-result/1' or result.get('complete') is not True or result.get('quality_passed') is not True or result.get('old_A8_equivalence_claim') is not False:raise ValueError('original numeric gate did not pass')
 if (result.get('strict_cases'),result.get('timed_shapes'),result.get('rounds'),result.get('iterations'))!=(len(plan),len(timed) if bench else 0,rounds,iterations):raise ValueError('complete plan counts differ')
 drivers=[e for e in events if e.get('event')=='driver_resources'];devices=[e for e in events if e.get('event')=='device']
 if len(drivers)!=1 or type(drivers[0].get('active_ctas')) is not int or drivers[0]['active_ctas']<1 or drivers[0].get('actual_block')!=128 or drivers[0].get('dynamic_shared_bytes')!=39936 or len(devices)!=1 or type(devices[0].get('sms')) is not int or devices[0]['sms']<1:raise ValueError('actual driver launch/resource evidence incomplete')
 controls=[e for e in events if e.get('event')=='cache_control']
 if len(controls)!=1 or controls[0].get('mode')!=mode or controls[0].get('physical_cold_guaranteed') is not False:raise ValueError('actual cache control metadata missing')
 control=controls[0];l2=control.get('driver_l2_bytes');sweep_bytes=control.get('sweep_bytes')
 if type(l2)is not int or type(sweep_bytes)is not int or l2<0 or sweep_bytes<0 or result.get('driver_l2_bytes')!=l2 or result.get('sweep_bytes')!=sweep_bytes:raise ValueError('invalid cache capacity metadata')
 if cache_sweep:
  if l2<=0 or sweep_bytes<2*l2 or sweep_bytes%256 or control.get('method')!='volatile_asm_ld_global_cg_u32' or control.get('threads')!=256 or control.get('constant_word')!=16843009 or type(control.get('blocks'))is not int or control['blocks']!=devices[0]['sms']*4:raise ValueError('eviction sweep is not actual twoL2 full read')
 elif sweep_bytes!=0 or control.get('blocks')!=0 or control.get('method')!='disabled':raise ValueError('warm mode unexpectedly sweeps')
 def ident(x):return x['k'],x['m'],x['n'],x['pattern'],float(x['alpha']).hex()
 # Native shape names encode alpha bits; use the serialized plan identity in their exact generated order.
 import struct
 def name(x):return 'k%d-m%d-n%d-p%d-a%d'%(x['k'],x['m'],x['n'],x['pattern'],struct.unpack('<I',struct.pack('<f',x['alpha']))[0])
 strict=[e for e in events if e.get('event')=='strict_case_pass'];expected=collections.Counter(name(x) for x in plan+(timed if bench else []))
 initial=[(i,e) for i,e in enumerate(events) if e.get('event')=='strict_case_pass'][:len(plan)];first_timing=min((i for i,e in enumerate(events) if e.get('event')in('cache_sweep','ABBA_gpu_ms')),default=len(events))
 if len(initial)!=len(plan) or collections.Counter(e.get('shape') for _,e in initial)!=collections.Counter(name(x) for x in plan) or any(i>=first_timing for i,_ in initial):raise ValueError('complete initial strict plan must precede every timed half')
 if collections.Counter(e.get('shape') for e in strict)!=expected:raise ValueError('missing/duplicate/unexpected strict cases')
 if any(any(e.get(k) is not True for k in ('full_output_saved','AA','guards')) or not isinstance(e.get('relative_l2'),(int,float)) or isinstance(e['relative_l2'],bool) or not math.isfinite(e['relative_l2']) or e['relative_l2']>2e-5 for e in strict):raise ValueError('original strict evidence missing')
 if sum(e.get('mutation_checked') is True for e in strict)!=1:raise ValueError('swizzle/scale mutation gate missing')
 rows=[e for e in events if e.get('event')=='ABBA_gpu_ms'];want={(name(x),scope,r) for x in timed for scope in ('matrix_kernel_only','full_pack_kernel_combine') for r in range(rounds)} if bench else set()
 if len(rows)!=len(want) or {(e.get('shape'),e.get('scope'),e.get('round')) for e in rows}!=want:raise ValueError('ABBA set incomplete or duplicate')
 for e in rows:
  if e.get('cache_mode')!=mode or type(e.get('iterations_per_sample'))is not int or e.get('iterations_per_sample')!=iterations:raise ValueError('timed mode/iteration mismatch')
  if e.get('full_output_postsample_bits') is not True or any(type(e.get(k)) not in (int,float) or not math.isfinite(e[k]) or e[k]<=0 for k in ('A1_legacy_ms','B1_bf16_ms','B2_bf16_ms','A2_legacy_ms')):raise ValueError('invalid timing/full bits')
 sweeps=[(i,e) for i,e in enumerate(events) if e.get('event')=='cache_sweep'];expected_sweeps={(shape,scope,r,half) for shape,scope,r in want for half in ('A1','B1','B2','A2')} if cache_sweep else set()
 if len(sweeps)!=len(expected_sweeps) or {(e.get('shape'),e.get('scope'),e.get('round'),e.get('half')) for _,e in sweeps}!=expected_sweeps or type(result.get('cache_sweep_calls'))is not int or result.get('cache_sweep_calls')!=len(sweeps):raise ValueError('missing/duplicate per-half sweeps')
 if cache_sweep:
  workers=control['blocks']*256;q,r=divmod(sweep_bytes//4,workers);expected_checksum=r*(((q+1)*16843009)&0xffffffff)+(workers-r)*((q*16843009)&0xffffffff)
 else:expected_checksum=0
 timed_index={(e['shape'],e['scope'],e['round']):i for i,e in enumerate(events) if e.get('event')=='ABBA_gpu_ms'}
 for i,e in sweeps:
  if e.get('mode')!=mode or e.get('driver_l2_bytes')!=l2 or e.get('sweep_bytes')!=sweep_bytes or e.get('all_checksums_equal') is not True or e.get('synchronized_before_event') is not True or e.get('physical_cold_guaranteed') is not False or type(e.get('checksum')) is not int or type(e.get('expected_checksum')) is not int or e['checksum']!=expected_checksum or e['expected_checksum']!=expected_checksum or e.get('checksum_words')!=control['blocks']*256:raise ValueError('invalid full-read checksum/sync sweep evidence')
  if i>=timed_index[(e['shape'],e['scope'],e['round'])]:raise ValueError('sweep follows its timed half')
 for key in want:
  order=[(i,e['half']) for i,e in sweeps if (e['shape'],e['scope'],e['round'])==key]
  if cache_sweep and [half for _,half in order]!=['A1','B1','B2','A2']:raise ValueError('per-half sweep order differs')
 return dict(strict_events=len(strict),ABBA_samples=len(rows),driver=drivers[0],device=devices[0],cache_control=control,cache_sweeps=len(sweeps),primary_eviction_scope='full_pack_kernel_combine; matrix-only also evicts prepared inputs')

def owned(argv,log,timeout):
 p=None;prior={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
 def stop(sig,frame):
  for s in prior:signal.signal(s,signal.SIG_IGN)
  raise KeyboardInterrupt('own GPU micro runner interrupted')
 for s in prior:signal.signal(s,stop)
 try:
  with log.open('w') as out:
   mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(prior))
   try:p=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,mask))
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   return p.wait(timeout=timeout)
 except BaseException:
  if p is not None and p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise
 finally:
  for s,h in prior.items():signal.signal(s,h)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--bench',action='store_true');p.add_argument('--rounds',type=int,default=5);p.add_argument('--iterations',type=int);p.add_argument('--cache-sweep',action='store_true');p.add_argument('--timeout',type=int,default=1800);a=p.parse_args(argv)
 a.iterations=(1 if a.cache_sweep else 3) if a.iterations is None else a.iterations
 if a.cache_sweep and (not a.bench or a.iterations!=1):p.error('--cache-sweep requires --bench and iterations1')
 if Path(a.name).name!=a.name or a.name in ('.','..') or min(a.rounds,a.iterations,a.timeout)<1:p.error('positive bounds and one fresh result name required')
 directory=a.build.resolve();out=directory/a.name;command=[str(directory/'weightmajor-bench'),str(directory),str(out),'--bench' if a.bench else '--strict','--rounds',str(a.rounds),'--iterations',str(a.iterations)]+(['--cache-sweep'] if a.cache_sweep else [])
 if not a.execute:print(json.dumps(dict(execute=False,GPU='NOT_RUN',argv=command,cache_mode='eviction-control' if a.cache_sweep else 'warm',operator_exclusive_lease_required=True)));return 0
 manifest=verify(directory)
 if out.exists() or out.is_symlink():raise ValueError('fresh result directory required')
 out.mkdir();state=dict(schema='q35-w4a16-weightmajor-run/1',complete=False,quality_passed=False,command=command,build_manifest_sha256=build.sha(directory/'manifest.json'),GPU='INVOKED_NOT_VALIDATED');save=lambda:build.save(out/'run.json',state);save()
 try:
  code=owned(command,out/'runner.jsonl',a.timeout);state['exit_code']=code;observed=json.loads((out/'result.json').read_text());state['original_result']=observed
  if code:raise ValueError('GPU runner failed; no timing qualification')
  events=[json.loads(line) for line in (out/'runner.jsonl').read_text().splitlines() if line.startswith('{')];state['validation']=validate(observed,events,a.bench,a.rounds,a.iterations,a.cache_sweep)
  # Bind every full numeric output; no claims of quality beyond this synthetic BF16 contract.
  state['outputs']={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=build.sha(f)) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='run.json'}
  state.update(complete=True,quality_passed=True,GPU='NUMERIC_PASS_NOT_MODEL_QUALITY',production_integrated=False);save();return 0
 except BaseException as e:state.update(complete=False,quality_passed=False,error=repr(e));save();return 1
if __name__=='__main__':raise SystemExit(main())
