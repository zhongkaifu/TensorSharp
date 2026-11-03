#!/usr/bin/env python3
"""Default non-GPU plan. Explicit execution requires the exact offline ELF/closure."""
import argparse,array,collections,importlib.util,json,math,os,re,struct
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('w8_build',HERE/'build.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
TS=(1,7,8,14,16,28,32,33)
PAIRS=('owned_vs_simt_functional','owned_vs_official_primary')
SCOPE='F32_to_BF16_cast_plus_W8_matmul_plus_BF16_store'
def plan():return [(17,32,t,0)for t in TS]+[(33,160,t,1)for t in TS]+[(15,512,t,2)for t in TS]+[(248320,5120,t,3)for t in TS]
def name(s):r,k,t,p=s;return f'r{r}-k{k}-t{t}-p{p}'
def shapemap():return {name(s):s for s in plan()}
def data_path(p):
 p=Path(p).absolute()
 if p!=Path(os.path.normpath(p))or p.resolve()!=p or Path('/dev/shm')not in p.parents or p.exists():raise ValueError('fresh physical /dev/shm data path required (no FUSE measurement writes)')
 return p
def verify_build(d):
 m=json.loads((d/'manifest.json').read_text());permit=json.loads((d/'offline-permit.json').read_text());tools=B.tool_inventory()
 if m.get('schema')!='q35-w8-g32-build/1'or m.get('complete')is not True or m.get('flags')!=B.FLAGS or m.get('tools')!=tools or m.get('contract_sha256')!=B.CONTRACT_SHA or m.get('official_closure_sha256')!=B.CLOSURE_SHA:raise ValueError('actual pinned offline build required')
 if m.get('source')!={n:B.digest(d/'source'/n)for n in tools}or m['source']!=tools:raise ValueError('compiled source differs')
 expected={B.RUNNER,'build.log'}
 if set(m.get('artifacts',{}))!=expected or any(B.digest(d/n)!=h for n,h in m['artifacts'].items())or permit!={'manifest_sha256':B.digest(d/'manifest.json'),'artifacts':m['artifacts']}:raise ValueError('artifact/real offline permit changed')
 if m.get('resources')!=B.resources((d/'build.log').read_text()):raise ValueError('actual compiler resource log changed')
 return m

def validate(events,result,bench,rounds,iterations,sweep):
 expected_shapes=shapemap();groups=collections.defaultdict(list)
 for i,e in enumerate(events):
  if not isinstance(e,dict)or e.get('event')not in ('device','resources','strict','ABBA','sweep_setup','sweep'):raise ValueError('unknown/failure GPU event')
  groups[e['event']].append((i,e))
 if any(type(result.get(k))is not bool for k in('complete','numeric_pass','cache_sweep'))or any(type(result.get(k))is not int for k in('strict_cases','timed_shapes','rounds','iterations','sweeps')):raise ValueError('typed result fields required')
 if result!={'complete':True,'numeric_pass':True,'strict_cases':32,'timed_shapes':8 if bench else 0,'rounds':rounds,'iterations':iterations,'cache_sweep':sweep,'sweeps':8*2*rounds*4 if sweep else 0,'model_quality':'NOT_RUN'}:raise ValueError('incomplete/unexpected result matrix')
 if len(groups['device'])!=1:raise ValueError('one actual device required')
 dev=groups['device'][0][1]
 if not isinstance(dev.get('name'),str)or not dev['name']or not re.fullmatch('[0-9a-f]{32}',dev.get('uuid',''))or any(type(dev.get(k))is not int or dev[k]<=0 for k in('sms','driver','runtime')):raise ValueError('invalid actual hardware')
 needed={'cast','reference_fp64','simt','sweep'}|{f'{kind}_{t}'for kind in('owned','official')for t in TS};resources=groups['resources']
 if len(resources)!=20 or {r.get('symbol')for _,r in resources}!=needed:raise ValueError('20 actual driver resources required')
 for _,r in resources:
  if any(type(r.get(k))is not int or r[k]<0 for k in('registers','local_bytes','shared_bytes','active_ctas','block_threads'))or not 0<r['registers']<=255 or r['active_ctas']<1 or r['block_threads']<1:raise ValueError('driver launchability failed')
 strict=groups['strict'];gates={}
 for i,r in strict:
  key=(r.get('shape'),r.get('dumped'))
  if key in gates or key[0]not in expected_shapes or type(key[1])is not bool:raise ValueError('duplicate/invalid strict identity')
  rows,k,t,p=expected_shapes[key[0]];real=rows==248320
  if r.get('elements')!=rows*t or r.get('official')is not real or r.get('independent_cpu_fp64')is not(not real)or r.get('cpu_fp32_bits')is not(not real)or r.get('exact_integer_case')is not(p==0)or any(r.get(x)is not True for x in('numeric_bound','fullbits_AA','dual_poison','guards')):raise ValueError('incomplete preregistered strict gate')
  gates[key]=i
 want={(s,True)for s in expected_shapes}|({(name((248320,5120,t,3)),False)for t in TS}if bench else set())
 if set(gates)!=want:raise ValueError('strict shape matrix incomplete')
 seen=set()
 for i,r in groups['ABBA']:
  key=(r.get('shape'),r.get('pair'),r.get('round'))
  if key in seen or key[0]not in expected_shapes or expected_shapes[key[0]][0]!=248320 or key[1]not in PAIRS or type(key[2])is not int or not 0<=key[2]<rounds or r.get('scope')!=SCOPE or r.get('fullbits_each_half')is not True:raise ValueError('bad/duplicate ABBA identity')
  if any(type(r.get(k))not in(int,float)or not math.isfinite(r[k])or r[k]<=0 for k in('A1_ms','B1_ms','B2_ms','A2_ms')):raise ValueError('invalid GPU event interval')
  if any(gates[(s,True)]>=i for s in expected_shapes)or gates.get((key[0],False),i)>=i:raise ValueError('timing preceded full strict/shape gate')
  seen.add(key)
 want_times={(name((248320,5120,t,3)),p,r)for t in TS for p in PAIRS for r in range(rounds)}if bench else set()
 if seen!=want_times:raise ValueError('ABBA matrix incomplete')
 sweeps=groups['sweep'];setups=groups['sweep_setup']
 if not sweep:
  if sweeps or setups:raise ValueError('warm mode swept')
 else:
  if len(setups)!=1:raise ValueError('missing/duplicate sweep setup')
  s=setups[0][1];l2=s.get('driver_l2_bytes');size=s.get('sweep_bytes')
  if type(l2)is not int or l2<=0 or type(size)is not int or size<2*l2 or size%256 or s.get('physical_cold_guaranteed')is not False or s.get('method')!='volatile_asm_ld_global_cg_u32_checksum_count'or s.get('all_data_evicted_control')is not True:raise ValueError('invalid actual L2 control')
  seen=set()
  for index,(i,r)in enumerate(sweeps,1):
   key=(r.get('shape'),r.get('pair'),r.get('round'),r.get('half'))
   if key in seen or r.get('index')!=index or r.get('words_read')!=size//4 or r.get('checksum')!=(size//4)*0x5a5a5a5a or any(r.get(k)is not True for k in('before_event','synchronized','guards')):raise ValueError('sweep checksum/identity failed')
   timing=next((j for j,e in groups['ABBA']if(e.get('shape'),e.get('pair'),e.get('round'))==key[:3]),-1)
   if timing<=i:raise ValueError('sweep not before paired event')
   seen.add(key)
  if seen!={(*x,h)for x in want_times for h in('A1','B1','B2','A2')}:raise ValueError('sweep half matrix incomplete')
 return dict(strict_unique=32,strict_executions=len(strict),ABBA=len(groups['ABBA']),sweeps=len(sweeps),model_quality='NOT_RUN',actual_Ignis_binary='NOT_TESTED',TS_Q6_equivalence='NOT_CLAIMED')

def allowed(ref,ab,k):
 n=k//32+64;g32=(n*2**-24)/(1-n*2**-24);g64=(k*2**-53)/(1-k*2**-53);e=(g32+g64)*ab+n*2**-149
 return e+2**-8*(abs(ref)+e)+2**-134

def read_array(p,kind,count):
 n={'d':8,'f':4,'H':2}[kind]
 if p.is_symlink()or p.stat().st_size!=count*n:raise ValueError('full array size/alias mismatch: '+str(p))
 x=array.array(kind)
 with p.open('rb')as f:x.fromfile(f,count)
 return x

def audit_arrays(data):
 """Independent fixed-bound recheck; AA/guards remain producer observations."""
 bf=[struct.unpack('<f',struct.pack('<I',b<<16))[0]for b in range(65536)];inventory={}
 expected_dirs=set(shapemap())
 if {p.name for p in data.iterdir()}!=expected_dirs:raise ValueError('missing/extra persisted shape')
 for key,(rows,k,t,pattern)in shapemap().items():
  d=data/key;count=rows*t;files={'reference.f64':('d',count),'sumAbs.f64':('d',count),'input.f32':('f',t*k),'input.bf16':('H',t*k),'simt.bf16':('H',count),'owned.bf16':('H',count)}
  if rows==248320:files['official.bf16']=('H',count)
  if d.is_symlink()or not d.is_dir()or {p.name for p in d.iterdir()}!=set(files):raise ValueError('full output set mismatch')
  a={n:read_array(d/n,*s)for n,s in files.items()};ref=a['reference.f64'];ab=a['sumAbs.f64']
  for f,b in zip(a['input.f32'],a['input.bf16']):
   bits=struct.unpack('<I',struct.pack('<f',f))[0];rounding=((bits+0x7fff+((bits>>16)&1))>>16)&65535
   if not math.isfinite(f)or rounding!=b or (b&0x7f80)==0x7f80 or((b&0x7f80)==0 and b&127):raise ValueError('persisted input cast unsupported/different')
  if any(not math.isfinite(r)or not math.isfinite(s)or s<0 for r,s in zip(ref,ab)):raise ValueError('invalid persisted FP64 reference')
  for f in ('simt.bf16','owned.bf16',*(['official.bf16']if rows==248320 else [])):
   for b,r,s in zip(a[f],ref,ab):
    v=bf[b]
    if not math.isfinite(v)or abs(v-r)>allowed(r,s,k)or(pattern==0 and v!=r):raise ValueError('persisted output violates preregistered bound/exact subset')
  for f in files:inventory[str((d/f).relative_to(data))]=dict(bytes=(d/f).stat().st_size,sha256=B.digest(d/f))
 return inventory

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--bench',action='store_true');p.add_argument('--cache-sweep',action='store_true');p.add_argument('--rounds',type=int,default=10);p.add_argument('--iterations',type=int,default=1);p.add_argument('--timeout',type=int,default=1800);a=p.parse_args(argv)
 if Path(a.name).name!=a.name or a.name in('.','..')or a.rounds<10 or min(a.iterations,a.timeout)<1 or(a.cache_sweep and(not a.bench or a.iterations!=1)):p.error('fresh bounded name, >=10 rounds, sweep requires bench/iterations1')
 d=B.evidence(a.build);data=data_path(a.data);out=d/a.name;out.mkdir(exist_ok=False)
 cmd=[str(d/B.RUNNER),'--out',str(out),'--data',str(data),'--rounds',str(a.rounds),'--iterations',str(a.iterations)]+(['--bench']if a.bench else[])+(['--cache-sweep']if a.cache_sweep else[])
 state=dict(schema='q35-w8-g32-run/1',complete=False,status='PLAN_GPU_NOT_RUN',gpu_invoked=False,command=cmd,data=str(data),model_quality='NOT_RUN',production_integration=False);B.save(out/'run.json',state)
 if not a.execute:print(json.dumps(state));return 0
 try:
  verify_build(d);before=B.digest(d/'manifest.json');state.update(gpu_invoked=True,status='RUNNING',build_manifest_sha256=before);B.save(out/'run.json',state)
  with(out/'runner.jsonl').open('x')as f:code,_=B.P.command(cmd,a.timeout,stdout=f)
  state['exit_code']=code
  if code:raise ValueError('GPU producer failed; original raw/data preserved')
  result=json.loads((out/'result.json').read_text());events=[json.loads(x)for x in(out/'runner.jsonl').read_text().splitlines()if x.strip()]
  observed=validate(events,result,a.bench,a.rounds,a.iterations,a.cache_sweep);arrays=audit_arrays(data);verify_build(d)
  if B.digest(d/'manifest.json')!=before:raise ValueError('build changed during execution')
  B.save(out/'full-array-inventory.json',arrays);state.update(complete=True,status='SYNTHETIC_W8_NUMERIC_PASS_NOT_MODEL_QUALITY',observed=observed,array_inventory_sha256=B.digest(out/'full-array-inventory.json'),runner_sha256=B.digest(out/'runner.jsonl'));B.save(out/'run.json',state);print(json.dumps(state));return 0
 except BaseException as e:
  state.update(complete=False,status='NOT_PASS',error=repr(e));B.save(out/'run.json',state)
  if isinstance(e,(KeyboardInterrupt,SystemExit)):raise
  print(json.dumps(state));return 1
if __name__=='__main__':raise SystemExit(main())
