#!/usr/bin/env python3
"""Explicit later GPU entry point. One variant per fresh result; no bypass gate."""
from __future__ import annotations
import argparse,importlib.util,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fp4_pipeline_build_for_run',HERE/'build.py');build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)

def verify_build(directory):
 m=json.loads((directory/'manifest.json').read_text());permit=json.loads((directory/build.PERMIT).read_text())
 expected={str(i):dict(v,wm=4,wn=2,block_threads=256,cols_fast=True,epilogues=['plain','swiglu'],dynamic_shared_bytes=build.shared(v))for i,v in build.VARIANTS.items()}
 if(m.get('schema')!='qwen35-fp4-pipeline-build/1'or m.get('complete')is not True or m.get('flags')!=build.FLAGS or m.get('split_k')!=1 or m.get('variants')!=expected or m.get('strict_cases')!=len(build.plan())or m.get('timed_shapes')!=6 or permit.get('schema')!='qwen35-fp4-pipeline-offline/1'or permit.get('quality')!='NOT_RUN'or permit.get('manifest_sha256')!=build.digest(directory/'manifest.json')):raise ValueError('missing/rejected/incompatible offline build; not GPU quality permission')
 artifacts=m.get('artifacts');resources=m.get('resources');symbols={v[k]for v in build.VARIANTS.values()for k in ('symbol','fused_symbol')}
 if not isinstance(artifacts,dict)or set(artifacts)!=set(build.artifact_names())or permit.get('artifacts')!=artifacts:raise ValueError('artifact identity incomplete')
 for name,value in artifacts.items():
  if Path(name).name!=name or build.digest(directory/name)!=value:raise ValueError('changed built artifact '+name)
 if m.get('tools')!={n:build.digest(HERE/n)for n in build.TOOLS}or build.digest(m['source'])!=m.get('source_sha256'):raise ValueError('source/tool changed; rebuild required')
 if not isinstance(resources,dict)or set(resources)!=symbols:raise ValueError('missing per-epilogue actual ptxas evidence')
 for v in build.VARIANTS.values():
  for key in ('symbol','fused_symbol'):
   r=resources[v[key]]
   if(r.get('dynamic_shared_bytes')!=build.shared(v)or r.get('block_threads')!=256 or r.get('swiglu')!=(key=='fused_symbol')or any(type(r.get(k))is not int or r[k]<0 for k in ('registers_per_thread','stack_bytes','spill_store_bytes','spill_load_bytes','static_shared_bytes'))):raise ValueError('invalid actual resource record')
 p=json.loads((directory/'plan.json').read_text())
 if p!=build.plan()or(directory/'plan.h').read_text()!=build.plan_header(p)or(directory/'variants.h').read_text()!=build.variant_header():raise ValueError('shape/variant plan changed')
 generated=build.generate(Path(m['source']).read_text())
 if any((directory/n).read_text()!=s for n,s in generated.items()):raise ValueError('generated source differs from reviewed original template/schedule')
 return m

def positive(x):return isinstance(x,(int,float))and not isinstance(x,bool)and math.isfinite(x)and x>0

def validate_observation(result,events,variant,mode):
 resonly=mode=='resources-only';bench=mode=='bench';plan=build.plan();expected={build.shape_name(s):s for s in plan};timed={n:s for n,s in expected.items()if s['timed']}
 if any(type(result.get(k))is not int for k in ('variant','split_k','rounds','iterations_per_sample','strict_cases','timed_shapes','timing_events')):raise ValueError('runner integer identity/count types invalid')
 if(result.get('complete')is not True or result.get('quality_pass')is not(not resonly)or result.get('status')!=('DRIVER_LAUNCHABILITY_ONLY_FULLBITS_NOT_RUN'if resonly else'FULLBITS_GUARDS_COVERAGE_AA_PASS')or result.get('variant')!=variant or result.get('split_k')!=1 or result.get('rounds')!=5 or result.get('iterations_per_sample')!=3 or result.get('strict_cases')!=(0 if resonly else len(plan))or result.get('timed_shapes')!=(6 if bench else 0)or result.get('timing_events')!=(30 if bench else 0)):raise ValueError('incomplete actual runner result')
 devices=[e for e in events if e.get('event')=='device']
 if len(devices)!=1 or not isinstance(devices[0].get('name'),str)or not devices[0]['name']or any(type(devices[0].get(k))is not int or devices[0][k]<=0 for k in ('cc_major','sms','shared_per_sm','shared_per_cta_optin','driver','runtime'))or type(devices[0].get('cc_minor'))is not int or devices[0]['cc_minor']<0:raise ValueError('missing/duplicate/invalid hardware evidence')
 queries=[e for e in events if e.get('event')=='shared_size_query']
 if len(queries)!=1 or queries[0].get('guards')is not True or queries[0].get('actual_bytes')!=[build.shared(v)for v in build.VARIANTS.values()]:raise ValueError('guarded runtime shared sizeof query failed')
 resources=[e for e in events if e.get('event')=='driver_resources'];wanted={(i,f)for i in (0,variant)for f in (False,True)};actual=[]
 for e in resources:
  i=e.get('variant');f=e.get('swiglu');actual.append((i,f));v=build.VARIANTS.get(i)
  if(type(i)is not int or v is None or type(f)is not bool or e.get('symbol')!=v['fused_symbol'if f else'symbol']or e.get('block_threads')!=256 or e.get('dynamic_shared_bytes')!=build.shared(v)or type(e.get('active_ctas'))is not int or e['active_ctas']<1 or type(e.get('max_block_threads'))is not int or e['max_block_threads']<256 or type(e.get('max_dynamic_shared_bytes'))is not int or e['max_dynamic_shared_bytes']<build.shared(v)or any(type(e.get(k))is not int or e[k]<0 for k in ('registers','local_bytes','static_shared_bytes'))or e.get('total_shared_bytes')!=e['static_shared_bytes']+e['dynamic_shared_bytes']):raise ValueError('actual driver launchability gate failed')
 if len(actual)!=len(set(actual))or set(actual)!=wanted:raise ValueError('missing/duplicate plain/fused driver resources')
 strict=[e for e in events if e.get('event')=='strict_case_pass'];names=[]
 for e in strict:
  n=e.get('shape');names.append(n);s=expected.get(n)
  if(s is None or e.get('variant')!=variant or e.get('swiglu')!=s['swiglu']or e.get('output_words')!=s['rows']*(s['out_dim']//2 if s['swiglu']else s['out_dim'])or any(e.get(k)is not True for k in ('full_output_bits','complete_write_dual_poison','oracle_AA','candidate_AA','guards','all_inputs_unchanged'))):raise ValueError('strict case identity/fullbits/coverage/AA failed')
 if len(names)!=len(set(names))or set(names)!=(set()if resonly else set(expected)):raise ValueError('strict plan missing/duplicate/unexpected cases')
 abba=[e for e in events if e.get('event')=='ABBA_gpu_ms'];pairs=[]
 for e in abba:
  n=e.get('shape');r=e.get('round');s=timed.get(n);pairs.append((n,r))
  if(s is None or type(r)is not int or r not in range(5)or e.get('variant')!=variant or e.get('iterations_per_sample')!=3 or e.get('swiglu')!=s['swiglu']or any(e.get(k)!=s[k]for k in ('rows','out_dim','k'))or e.get('fullbits_each_half')is not True or e.get('all_new_work_included')is not True or not all(positive(e.get(k))for k in ('A1_oracle_ms','B1_candidate_ms','B2_candidate_ms','A2_oracle_ms'))):raise ValueError('ABBA half identity/fullbits/positive timing failed')
 if len(pairs)!=len(set(pairs))or set(pairs)!=({(n,r)for n in timed for r in range(5)}if bench else set()):raise ValueError('ABBA set incomplete/duplicate/unexpected')
 if any(e.get('event')not in ('device','shared_size_query','driver_resources','strict_case_pass','ABBA_gpu_ms')for e in events):raise ValueError('unexpected runner diagnostic/failure event')
 return dict(driver_launchability=True,quality_fullbits='NOT_RUN'if resonly else'PASS',strict_cases=len(strict),timing_events=len(abba),per_epilogue_driver_resources=resources)

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',type=Path,required=True);ap.add_argument('--name',required=True);ap.add_argument('--variant',type=int,choices=(1,2,3,4),required=True);g=ap.add_mutually_exclusive_group();g.add_argument('--bench',action='store_true');g.add_argument('--resources-only',action='store_true');ap.add_argument('--timeout',type=int,default=1200);a=ap.parse_args(argv)
 directory=build.evidence_dir(a.build)
 if Path(a.name).name!=a.name or a.name in('.','..')or a.timeout<=0:ap.error('simple fresh name and positive timeout required')
 out=directory/a.name;out.mkdir(exist_ok=False);mode='bench'if a.bench else'resources-only'if a.resources_only else'strict';state=dict(schema='qwen35-fp4-pipeline-run/1',complete=False,quality_pass=False,variant=a.variant,mode=mode,gpu_invoked=False,started_utc=datetime.now(timezone.utc).isoformat(),hardware_counters='NOT_RUN',production_integrated=False)
 save=lambda:build.save(out/'run.json',state);save()
 try:
  m=verify_build(directory);command=[str(directory/build.RUNNER),str(directory),str(out),'--'+mode,str(a.variant)];state.update(command=command,build_manifest_sha256=build.digest(directory/'manifest.json'),offline_resource_facts=m['resources'],gpu_invoked=True);save()
  with(out/'runner.jsonl').open('w')as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
  state['exit_code']=r.returncode
  if r.returncode:raise ValueError('native runner rejected/failed; no timing qualification')
  observed=json.loads((out/'result.json').read_text());events=[json.loads(s)for s in(out/'runner.jsonl').read_text().splitlines()if s.strip()];gate=validate_observation(observed,events,a.variant,mode)
  verify_build(directory);state.update(complete=True,quality_pass=not a.resources_only,result=observed,validated=gate,runner_log_sha256=build.digest(out/'runner.jsonl'),status='DRIVER_ONLY_NOT_QUALITY_PASS'if a.resources_only else'FULLBITS_PASS_TIMED'if a.bench else'FULLBITS_PASS_NOT_TIMED')
 except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError)as e:state.update(complete=False,quality_pass=False,status='REJECTED_NOT_VALIDATED',error=str(e))
 state['finished_utc']=datetime.now(timezone.utc).isoformat();save();print(json.dumps(state,indent=2));return 0 if state['complete']else 1
if __name__=='__main__':raise SystemExit(main())
