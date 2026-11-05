#!/usr/bin/env python3
"""Controlled same-S4 CUA repeats v2: single-factor splitK profiles; original frozen scoring, PLAN by default.

Explicit splitK0 is a separately labelled math policy, not proof of deterministic
execution or quality. Every text/media replay uses a fresh owned server.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,re,signal,socket,subprocess,sys,time,types,urllib.request
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_CONTROLLER_SHA='e87b3b979aff057b2cf416fb7918c1bf2b612eef6587b089f3523a81afc553b3'
S4_BUILD_SHA='cc01b83179bdafec573212bd4db11f352ebc9c4110d9193d3d41048b87d29b16'
S4_CATALOG_SHA='5d3cad25a7be77a850804e2ecf0804b589cb8c83ea0fb103d291faf67b9be7d2'
S4_RUNTIME_SHA='ed750f55d7b8b12dfa10908b42a54b1d4eec97d5eef31c4601cf60920296cb54'
S4_SAMPLER_SHA='b0fca5c7c1cb54586e29345c40e9adad226c519665923cdaddeb2f18ab2e5c53'
S4_RUNTIME_DLL_SHA='630b88357150223137bca79591314118f054d560742fee8bcc979897811aeea3'
SPLIT_ENV=('TS_Q35_DRAFT_SPLITK','TS_Q35_GEMM_SPLITK')
SPLIT_POLICIES={
 'production-unset':dict(zip(SPLIT_ENV,(None,None))),
 'draft-zero':dict(zip(SPLIT_ENV,('0',None))),
 'target-zero':dict(zip(SPLIT_ENV,(None,'0'))),
 'splitk-zero':dict(zip(SPLIT_ENV,('0','0'))),
}
SPLITS=tuple(SPLIT_POLICIES)
SPLIT_LABELS={'production-unset':'splitkunset','draft-zero':'draft0-NEWMATH','target-zero':'target0-NEWMATH','splitk-zero':'bothsplitk0-NEWMATH'}
MATRIX_SHA='41217710ffc12a3384aae47f87be59479d27165e100c1ca79afb40d8681668e8'
AUDIT_SHA='a2a559852be77bb83451723d532584799b153b116ff51bb0be77a4ed05e86bab'
TOGGLES=('TS_TOKEN_SAMPLER_SIMD','TS_Q35_SKINNY_FFN_A4')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def module(path,name,pin):
 require(sha(path)==pin,'frozen helper changed: '+str(path));s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def configurations(value):
 require(isinstance(value,dict)and set(value)=={'schema','profiles','comparisons'}and value['schema']=='qwen38-cua-causality-jobs/1','exact versioned profiles/comparisons object required')
 profiles=value['profiles'];pairs=value['comparisons']
 require(isinstance(profiles,list)and 2<=len(profiles)<=8,'bounded2..8 fresh profiles required')
 for p in profiles:
  require(isinstance(p,dict)and set(p)=={'name','cpu_simd','splitk_policy'},'only explicit name/CPU/splitK policy allowed; memory and all other ALT fixed0')
  require(isinstance(p['name'],str)and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,63}',p['name'])is not None and type(p['cpu_simd'])is str and p['cpu_simd']in('0','1')and p['splitk_policy']in SPLITS,'invalid profile identity or explicit policy')
 by={p['name']:p for p in profiles};require(len(by)==len(profiles),'duplicate profile name')
 require(isinstance(pairs,list)and 1<=len(pairs)<=28,'bounded directed comparisons required');seen=set();used=set()
 for p in pairs:
  require(isinstance(p,dict)and set(p)=={'left','right'}and type(p['left'])is str and type(p['right'])is str and p['left']in by and p['right']in by and p['left']!=p['right'],'comparison must bind two distinct existing profiles')
  key=(p['left'],p['right']);require(key not in seen,'duplicate directed comparison');seen.add(key);used.update(key)
  a,b=by[p['left']],by[p['right']];require(a['cpu_simd']==b['cpu_simd']or a['splitk_policy']==b['splitk_policy'],'change CPU or splitK in a directed comparison, never both')
 require(used==set(by),'every measured profile must participate in an explicit comparison')
 return value

def jobs(config,selection):
 return [dict(p,name=p['name']+('-media'if media else'-text'),configuration=p['name'],env={'TS_TOKEN_SAMPLER_SIMD':p['cpu_simd'],'TS_Q35_SKINNY_FFN_A4':'0'},media=media,scenarios=['edit_media']if media else['append','truncate','edit'],steps=6 if media else 30,expected_requests=12 if media else 180,context=65536,lanes=2,mode='dflash2')for media in(False,True)if selection=='both'or selection==('media'if media else'text')for p in config['profiles']]

def replay_path(B,cua,harness,work,name):
 require(isinstance(name,str)and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,63}-(text|media)',name)is not None,'bounded fresh profile/kind required')
 root=harness/'artifacts/qwen38-causal-n8-20261102'/work.name;raw=root/name/'replay.jsonl';B.normalized(raw)
 require(raw.is_relative_to(harness/'artifacts/qwen38-causal-n8-20261102')and not raw.exists()and cua.evidence_path(str(raw))==raw.absolute(),'fresh actual frozen-producer evidence path required')
 return raw

def verify_S4(build,catalog,build_path,catalog_path):
 require(sha(build_path)==S4_BUILD_SHA and sha(catalog_path)==S4_CATALOG_SHA,'only actual immutable reviewed S4 build/catalog permitted')
 require(build.get('complete')is True and build.get('source_catalog_sha256')==S4_CATALOG_SHA and len(catalog)==4445 and len(build.get('new_runtime',[]))==415,'actual S4 complete4445/415 required')
 require(hashlib.sha256(json.dumps(build['new_runtime'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==S4_RUNTIME_SHA,'full415 S4 runtime catalog differs')
 require(next((r['sha256']for r in catalog if r['path']=='TensorSharp.Runtime/TokenSampler.cs'),None)==S4_SAMPLER_SHA and next((r['sha256']for r in build['new_runtime']if r['path']=='TensorSharp.Runtime.dll'),None)==S4_RUNTIME_DLL_SHA,'actual defaultOFF sampler source/runtime differs')
def audit_bound_model(A,model,*args):
 # Same frozen audit bytecode/checks; explicit model is observed from /v1/models,
 # never guessed. Do not mutate the frozen module's globals or call its main.
 fn=types.FunctionType(A.audit_cua.__code__,dict(A.audit_cua.__globals__,MODEL_ID=model),A.audit_cua.__name__,A.audit_cua.__defaults__,A.audit_cua.__closure__)
 return fn(*args)
def pair_fields(A,left,right):
 result=A.pair_cua(left,right)
 def rows(p):return{tuple(r[k]for k in('scenario','lanes','lane','step')):r for r in map(json.loads,Path(p).read_text().splitlines())if r.get('kind')=='request'}
 a,b=rows(left),rows(right);fields=[]
 for key in sorted(a):
  require(a[key]['expected']==b[key]['expected'],'expected fixture differs between jobs')
  correctness=[];values=[]
  for r in(a[key],b[key]):
   calls=r['response'].get('tool_calls')or[];parsed=calls[0].get('parsed')if len(calls)==1 and calls[0].get('name')=='submit_observation'else None
   values.append(parsed);correctness.append({k:isinstance(parsed,dict)and k in parsed and type(parsed[k])is type(v)and parsed[k]==v for k,v in r['expected'].items()})
  fields.append(dict(id=list(key),expected=a[key]['expected'],parsed=values,typed_field_checks=[*correctness],true_to_false=[k for k,v in correctness[0].items()if v is True and correctness[1][k]is False],false_to_true=[k for k,v in correctness[0].items()if v is False and correctness[1][k]is True]))
 result.update(typed_fields_diagnostic_only=fields,typed_field_true_to_false=sum(len(x['true_to_false'])for x in fields),typed_fields_do_NOT_rescore_original_verdict=True)
 return result
def environment(B,a,job,directory):
 env=B.environment(a,job,directory);env.update(TS_PREFIX_CACHE_ISOLATION='shared')
 for key,value in SPLIT_POLICIES[job['splitk_policy']].items():
  if value is None:env.pop(key,None)
  else:env[key]=value
 env['MANIFEST_ENV_ALLOWLIST']=','.join(sorted(set(env['MANIFEST_ENV_ALLOWLIST'].split(','))|{'MAX_CONTEXT','TS_SCHED_MAX_RUNNING_SEQS','TS_PREFIX_CACHE_ISOLATION',*TOGGLES,*SPLIT_ENV,'TS_Q35_MIXED_FP4_VERIFY'}))
 return env
def manifest_gate(B,a,m,owner,env,job,label,before):
 require(m.get('status')=='ready'and m.get('engine')=='tensorsharp'and m.get('mode')=='dflash2'and m.get('label')==label and m.get('context_tokens')==65536 and m.get('lanes')==2 and m.get('working_directory')==str(a.runtime),'actual ready context/cap/runtime differs')
 require(all(m.get(k,{}).get('path')==str(a.source)for k in('engine_source','tensorsharp_source')),'manifest source differs from built source')
 require(all(m.get('process',{}).get(k)==owner.get(k)and owner.get(k)is not None for k in('pid','starttime','boot_id','run_id')),'manifest process owner differs')
 actual=m.get('environment',{})
 require(all(actual.get(k)==env[k]for k in(*B.ALT,*TOGGLES,'TS_Q35_UBATCH','TS_PREFIX_CACHE_ISOLATION','TENSORSHARP_PREFIX_CACHE_DIR','TENSORSHARP_UPLOAD_DIR','TENSORSHARP_LOG_DIR','TS_STUDIO_DIRECTORY','ASPNETCORE_WEBROOT')),'actual sampling/ALT/cache/path env differs')
 require(actual.get('MAX_CONTEXT')=='65536'and actual.get('TS_SCHED_MAX_RUNNING_SEQS')=='2'and m.get('sequence_admission_configuration',{}).get('configured_running_sequence_limit')=='2'and all(k in actual and actual[k]==SPLIT_POLICIES[job['splitk_policy']][k]for k in SPLIT_ENV),'actual cap/context or explicitly selected splitK math policy differs')
 require('TS_Q35_MIXED_FP4_VERIFY' in actual and actual['TS_Q35_MIXED_FP4_VERIFY'] is None,'mixed FP4 verify must remain original UNSET policy; capture is not an override')
 require(m.get('gpu',{}).get('devices',[{}])[0].get('uuid')==B.UUID and m.get('binaries')and m.get('models'),'actual GPU/model/binary enrichment missing')
 argv=m.get('argv',[])
 for flag,value in[('--backend','cuda'),('--model',str(a.models/'Qwen3.8-27B-NVFP4-MTP-N4_0.gguf')),('--mmproj',str(a.models/'mmproj-BF16.gguf')),('--draft-model',str(a.models/'Qwen3.8-27B-DFlash2-Q4_K_M.gguf')),('--spec-draft','7'),('--spec-pmin','0')]:require(argv.count(flag)==1 and argv[argv.index(flag)+1]==value,'actual native argv differs: '+flag)
 runtime=B.served_binary_map(a.runtime,before['runtime'])
 for item in m['binaries']:
  p=Path(item['path']);v={'sha256':item['sha256'],'size_bytes':item['size_bytes']};require(sha(p)==v['sha256']and p.stat().st_size==v['size_bytes'],'actual manifest binary bytes differ')
  require(runtime.get(str(p))==v or(str(p)=='/root/dotnet/dotnet'and v=={'sha256':before['dotnet']['sha256'],'size_bytes':before['dotnet']['bytes']}),'unbound runtime provider')
 observed={x['path']for x in m['binaries']};rows={x['path']:x for x in before['runtime']}
 for name in('TensorSharp.Server.Host.dll','TensorSharp.Runtime.dll','TensorSharp.Models.dll','TensorSharp.Backends.Cuda.dll','cuda_kernels/tensorsharp_q35_kernels.fatbin','libGgmlOps.so'):
  require(name in rows,'required served runtime artifact missing: '+name);row=rows[name];require((row.get('symlink_target')or str(a.runtime/name))in observed,'required actual binary/provider not captured: '+name)
 require('Model: qwen35 on the direct-CUDA engine'in Path(m['log']).read_text(errors='replace'),'actual CUDA route marker missing')
def cleanup_owned(B,a,attempt,record,manifest,started,env,command):
 if record.exists():
  require(not record.is_symlink(),'aliased ownership record');owner=B.read(record);require(started and owner.get('manifest')==str(manifest),'foreign process record; refuse cleanup')
  if'process_record'in attempt:require(owner==attempt['process_record']['value'],'owned process record changed')
  attempt['cleanup_record']=dict(path=str(record),sha256=sha(record),value=owner);stop=command(attempt,'cleanup',[sys.executable,a.harness/'eng/qwen38-engine-process.py','stop','--record',record,'--timeout','30'],45,env);require(stop['complete']and not record.exists(),'owned process cleanup failed')
 else:require('process_record'not in attempt,'previously observed owned record vanished');attempt['cleanup_record']=dict(status='ABSENT_AFTER_STARTUP_FAILURE_NO_SIGNAL_SENT',startup_attempted=started)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for key in('source','runtime','build-campaign','jobs-json'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--source-catalog',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--work',type=Path,default=Path('/dev/shm/ts-qwen38-causal-n8-r1/production-AA-r1'));p.add_argument('--selection',choices=('both','text','media'),default='both');p.add_argument('--lease',type=Path,default=Path('/dev/shm/ts-qwen38-causal-n8-r1/control/lease.json'));p.add_argument('--harness',type=Path,default=Path('/root/ts-harness-root-frozen'));p.add_argument('--matrix-helper',type=Path,default=HERE/'qwen38_vm_guard.py');p.add_argument('--audit-helper',type=Path,default=HERE/'qwen38_cua_guard.py');p.add_argument('--bench-python',type=Path,default=Path('/root/venvs/bench/bin/python'));p.add_argument('--models',type=Path,default=Path('/dev/shm/qwen38-models'));p.add_argument('--tokenizer',type=Path,default=Path('/workspace/hf/Qwen3.8-27B-NVFP4-nvidia/tokenizer.json'));p.add_argument('--haystack',type=Path,default=Path('/workspace/ts-eval/artifacts/haystack.txt'));p.add_argument('--model-manifest',type=Path,default=Path('/workspace/ts-eval/artifacts/combined-model-manifest.json'));p.add_argument('--port',type=int,default=5088);p.add_argument('--startup-timeout',type=int,default=330);p.add_argument('--replay-timeout',type=int,default=1800);a=p.parse_args(argv)
 require(0<a.port<65536 and a.port!=8081 and min(a.startup_timeout,a.replay_timeout)>0,'bounded budgets/port; nginx8081 reserved')
 B=module(a.matrix_helper,'ignis_CUA_matrix_helpers',MATRIX_SHA);A=module(a.audit_helper,'ignis_CUA_frozen_audit_only',AUDIT_SHA);config=configurations(read(a.jobs_json));selected=jobs(config,a.selection);a.source_catalog=a.source_catalog or a.build_campaign.parent/'source-catalog.json'
 state=dict(schema='qwen38-controlled-S4-CUA-causality/2',complete=False,execution_complete=False,quality_passed=False,plan_only=not a.execute,controller_sha256=sha(__file__),jobs=selected,source=str(a.source),runtime=str(a.runtime),build_campaign=str(a.build_campaign),work=str(a.work),attempts=[],pairings={},cache_profile='trusted-single-tenant-shared',server_cap=2,CTX=65536,UBATCH=4096,client_lanes=2,splitK_policy_assignments=SPLIT_POLICIES,splitK_profiles=config['profiles'],directed_comparisons=config['comparisons'],all_splitK_overrides_are_new_math_policies_not_determinism_proof=True,causal_attribution='NOT_ESTABLISHED_BY_REPLAY',dynamic_batch_math='Original mixed FP4 verify UNSET and dynamic verify window remain unchanged; splitK0 is not full determinism',baseline_controller_sha256=BASE_CONTROLLER_SHA,frozen_CUA=A.CUA_SHA,complete_is_NOT_quality_PASS=True,independent_SSE_semantics_reparse=False,raw_scope='Complete exact bytes consumed by frozen producer through DONE/EOF hash/size verified; parsed fields trusted to hash-bound frozen producer.',FFN_status='REJECTED_AFTER_ACTUAL_QUALITY_REGRESSION_NO_FFN_ARM',production_deployment=False)
 if not a.execute:print(json.dumps(state,indent=2));return 0
 B.normalized(a.source);B.normalized(a.runtime);B.work_scope(a.work,(a.source,a.runtime,a.harness,a.models,a.build_campaign,a.source_catalog,a.jobs_json,a.matrix_helper,a.audit_helper));require(a.work.is_relative_to('/dev/shm')and a.work!=Path('/dev/shm')and not a.work.exists(),'fresh owned /dev/shm work required')
 require(all(sha(a.harness/k)==v for k,v in B.PINS.items()),'frozen 4e/966/7e/fbd differs');cua,bench,fixtures=A.fixture_contract(types.SimpleNamespace(load=B.module),a.harness,a.selection!='text')
 for x in a.runtime.rglob('*'):
  if x.is_symlink():require(x.is_file()and not x.is_dir(),'missing/directory runtime provider alias not inventoried')
 build=B.read(a.build_campaign);catalog=B.read(a.source_catalog);verify_S4(build,catalog,a.build_campaign,a.source_catalog)
 lock_path=Path('/root/ts-eval-gpu.lock');B.normalized(lock_path);held=lock_path.open('a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.work.mkdir(parents=True);control=a.work/'control';control.mkdir();campaign=control/'campaign.json';replay_root=None
 def persist():B.save(campaign,state)
 def fixture_files():
  # Reuse the frozen loader's full image/source verification again at postcheck.
  checked,_,loaded=A.fixture_contract(types.SimpleNamespace(load=B.module),a.harness,a.selection!='text')
  values=A.fixture_snapshot_cua(a.harness,a.selection!='text');values[str(checked.GENERATOR)]=sha(checked.GENERATOR)
  for directory,fixture in loaded.values():
   for name in fixture['images']:values[str(directory/'images'/name)]=sha(directory/'images'/name)
  return values
 def snapshot():
  return dict(source=B.source_snapshot(a.source,catalog),runtime=B.inventory(a.runtime),ggml=B.dependency(a.source),fixtures=fixture_files(),tools={str(x):sha(x)for x in(Path(__file__),a.matrix_helper,a.audit_helper,a.jobs_json,a.build_campaign,a.source_catalog)},harness={k:sha(a.harness/k)for k in B.PINS},dotnet=dict(path='/root/dotnet/dotnet',bytes=Path('/root/dotnet/dotnet').stat().st_size,sha256=sha('/root/dotnet/dotnet')),models={n:dict(path=str(a.models/n),bytes=(a.models/n).stat().st_size,sha256=sha(a.models/n))for n in B.MODEL_PINS},inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in(a.tokenizer,a.haystack,a.model_manifest)},hardware=dict(host=socket.gethostname(),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text()))
 previous={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)}
 def interrupted(s,frame):raise KeyboardInterrupt('owned CUA controller signal '+str(s))
 for s in previous:signal.signal(s,interrupted)
 def command(attempt,name,argv,timeout,env):
  if name!='cleanup':state['last_lease']=B.require_lease(a.lease,timeout)
  row=dict(name=name,argv=list(map(str,argv)),timeout=timeout,complete=False);attempt['commands'].append(row);persist();log=Path(attempt['directory'])/(name+'.log');child=None;oldmask=None
  try:
   with log.open('xb')as f:
    oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
    try:child=subprocess.Popen(row['argv'],cwd=a.source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask));row['launcher_pid']=child.pid;persist()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask);oldmask=None
    try:child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:row['timed_out']=True
  finally:
   if child is not None and child.returncode is None:
    try:
     if os.getpgid(child.pid)==child.pid:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=2)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    except ProcessLookupError:child.wait()
   row.update(exit_code=None if child is None else child.returncode,complete=child is not None and child.returncode==0 and not row.get('timed_out'),log_sha256=sha(log)if log.exists()else None);persist()
  return row
 try:
  state['lease_pre']=B.require_lease(a.lease,a.startup_timeout+a.replay_timeout+90);B.idle();state['GPU_pre']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);require(state['GPU_pre'].splitlines()[0].split(',')[0].strip()=='0'and B.UUID in state['GPU_pre'].splitlines()[0],'actual visible GPU0/UUID differs');state['before']=snapshot();require(state['before']['runtime']==build['new_runtime']and all(state['before']['models'][k]['sha256']==v for k,v in B.MODEL_PINS.items()),'actual new runtime/model differs from build')
  replay_root=replay_path(B,cua,a.harness,a.work,selected[0]['name']).parent.parent;require(not replay_root.exists(),'fresh replay namespace required');replay_root.mkdir(parents=True);state['local_replay_root']=str(replay_root);persist();model=None
  for job in selected:
   state['last_arm_lease']=B.require_lease(a.lease,a.startup_timeout+a.replay_timeout+90);B.idle();directory=control/job['name'];directory.mkdir();env=environment(B,a,job,directory)
   for key in('TENSORSHARP_PREFIX_CACHE_DIR','TENSORSHARP_UPLOAD_DIR','TENSORSHARP_LOG_DIR','TS_STUDIO_DIRECTORY','ASPNETCORE_WEBROOT'):Path(env[key]).mkdir()
   raw=replay_path(B,cua,a.harness,a.work,job['name']);raw.parent.mkdir();label='vm2-causal-n8-r2-'+job['name']+'-cpu'+job['cpu_simd']+'-'+SPLIT_LABELS[job['splitk_policy']];manifest=directory/'server/manifests'/('tensorsharp-dflash2-'+label+'.json');record=directory/'server/pids'/('tensorsharp-'+str(a.port)+'.json');attempt=dict(job=job,label=label,directory=str(directory),raw=str(raw),commands=[],complete=False,quality_passed=False,environment=env);state['attempts'].append(attempt);persist();started=False
   try:
    with socket.socket()as sock:require(sock.connect_ex(('127.0.0.1',a.port))!=0,'occupied unknown server port; never killed')
    started=True;start=command(attempt,'start',['bash',a.harness/'eng/qwen38-engine-servers.sh','start','tensorsharp','dflash2',label],a.startup_timeout+30,env);require(start['complete'],'server startup failed');launch=B.read(manifest);owner=B.read(record);require(owner.get('manifest')==str(manifest),'record does not bind this launch');manifest_gate(B,a,launch,owner,env,job,label,state['before']);attempt.update(launch_manifest=dict(path=str(manifest),sha256=sha(manifest),value=launch),process_record=dict(path=str(record),sha256=sha(record),value=owner))
    with urllib.request.urlopen('http://127.0.0.1:'+str(a.port)+'/v1/models',timeout=10)as response:available=json.load(response)['data']
    require(isinstance(available,list)and len(available)==1 and isinstance(available[0].get('id'),str)and available[0]['id'],'unique actual model ID required');actual=available[0]['id'];require(model is None or actual==model,'model ID changed between same-runtime jobs');model=actual;attempt['actual_API_model_id']=model
    fixture_dir,fixture=fixtures[job['media']];cmd=[a.bench_python,a.harness/'eng/validation/qwen38-cua-control.py','run','--fixtures',fixture_dir,'--engine','tensorsharp','--url','http://127.0.0.1:'+str(a.port),'--model',model,'--manifest',manifest,'--label',label,'--mode','dflash2','--lanes','2','--context-capacity','65536','--cache-profile','trusted-single-tenant-shared','--scenarios',','.join(job['scenarios']),'--timeout','120','--out',raw];done=command(attempt,'replay',cmd,a.replay_timeout,env);audit=audit_bound_model(A,model,raw,job,label,launch,attempt['launch_manifest']['sha256'],fixture,cua,bench,done['exit_code']);attempt.update(audit=audit,complete=audit['execution_complete'],quality_passed=audit['original_quality_and_measurement_qualified'])
   except Exception as e:attempt.update(error=repr(e),complete=False,quality_passed=False)
   finally:
    deferred=[];cleanup_handlers={s:signal.signal(s,lambda signum,frame:deferred.append(signum))for s in previous}
    try:
     cleanup_owned(B,a,attempt,record,manifest,started,env,command)
     B.idle(12);attempt['GPU_idle_after_cleanup']=True
     if manifest.exists():attempt['final_manifest']=dict(path=str(manifest),sha256=sha(manifest),value=B.read(manifest))
    except BaseException as e:attempt.update(complete=False,quality_passed=False,cleanup_error=repr(e));persist();raise
    finally:
     for s,handler in cleanup_handlers.items():signal.signal(s,handler)
     attempt['deferred_cleanup_signals']=deferred;persist()
    if deferred:raise KeyboardInterrupt('signal deferred until owned cleanup')
  for media in(False,True):
   arms={x['job']['configuration']:x for x in state['attempts']if x['job']['media']is media}
   if not arms:continue
   for comparison in config['comparisons']:
    left,right=arms[comparison['left']],arms[comparison['right']]
    if left['complete']and right['complete']:
     paired=pair_fields(A,left['raw'],right['raw']);paired.update(left_configuration=comparison['left'],right_configuration=comparison['right'],left_CPU=left['job']['cpu_simd'],right_CPU=right['job']['cpu_simd'],left_splitK=left['job']['splitk_policy'],right_splitK=right['job']['splitk_policy'],comparison_type='repeat-AA'if(left['job']['cpu_simd'],left['job']['splitk_policy'])==(right['job']['cpu_simd'],right['job']['splitk_policy'])else'CPU-AB'if left['job']['splitk_policy']==right['job']['splitk_policy']else'NEW-MATH-POLICY',no_causal_or_speed_claim=True);state['pairings'][comparison['left']+'->'+comparison['right']+(':media'if media else':text')]=paired
  state['execution_complete']=len(state['attempts'])==len(selected)and all(x['complete']for x in state['attempts']);expected_pairs=len(config['comparisons'])*(2 if a.selection=='both'else 1);state['comparison_inputs_identical']=len(state['pairings'])==expected_pairs and all(x['all_input_identities_match']for x in state['pairings'].values());state['complete']=state['execution_complete']and state['comparison_inputs_identical'];state['no_new_original_check_or_field_regressions']=state['complete']and all(x['original_check_true_to_false']==0 and x['typed_field_true_to_false']==0 for x in state['pairings'].values());state['quality_passed']=state['complete']and all(x['quality_passed']for x in state['attempts'])and state['no_new_original_check_or_field_regressions']
 except BaseException as e:state.update(complete=False,quality_passed=False,error=repr(e))
 finally:
  try:
   if'before'in state:state['after']=snapshot();require(state['before']==state['after'],'source/runtime/provider/model/fixture/tools/build changed');state['identities_unchanged']=True
   state['lease_post']=B.require_lease(a.lease,0)
  except BaseException as e:state.update(complete=False,quality_passed=False,postcheck_error=repr(e))
  persist()
  for s,handler in previous.items():signal.signal(s,handler)
  fcntl.flock(held,fcntl.LOCK_UN);held.close()
 print(json.dumps({k:state.get(k)for k in('complete','execution_complete','quality_passed','comparison_inputs_identical','error','postcheck_error')},indent=2));return 0 if state['complete']else 1
if __name__=='__main__':raise SystemExit(main())
