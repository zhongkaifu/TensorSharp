#!/usr/bin/env python3
"""Default-plan entrypoint; --execute alone permits private synthetic GPU gates."""
import argparse,datetime,importlib.util,json,math,os,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('dp4a_schedule_build',HERE/'build.py');B=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(B)
def selected(manifest):
    choice=manifest.get('arm_selection')
    expected=B.CANDIDATES if choice=='all'else[B.CANDIDATES[('bounds','stream','combo').index(choice)]]if choice in('bounds','stream','combo')else None
    if expected is None or manifest.get('candidates')!=expected:raise ValueError('unknown/changed candidate identities')
    return expected
def verify_build(directory):
    m=json.loads((directory/'manifest.json').read_text());p=json.loads((directory/B.PERMIT).read_text());candidates=selected(m);symbols={B.ORACLE,*(v['symbol']for v in candidates)}
    if(m.get('schema')!='q35-dp4a-schedule-build/1'or m.get('complete')is not True or m.get('activation_rows')!=8 or m.get('strict_cases')!=96 or m.get('timed_shapes')!=12 or m.get('flags')!=B.FLAGS or m.get('oracle')!=dict(symbol=B.ORACLE,rows_per_warp=4,block_threads=128)or m.get('oracle_is_freshly_recompiled_NOT_production_BBD')is not True):raise ValueError('missing actual complete offline build')
    if p.get('schema')!='q35-dp4a-schedule-offline/1'or p.get('manifest_sha256')!=B.digest(directory/'manifest.json')or p.get('artifacts')!=m.get('artifacts'):raise ValueError('missing/changed real offline marker')
    if set(m.get('artifacts',{}))!=set(B.ARTIFACTS)or any(B.digest(directory/n)!=h for n,h in m['artifacts'].items()):raise ValueError('built artifacts changed')
    if m.get('tools')!={n:B.digest(HERE/n)for n in B.TOOLS}or m.get('frozen_fixture_tools')!=B.fixture_pins()or B.digest(m['source'])!=m['source_sha256']:raise ValueError('tool/fixture/production source changed')
    resources=m.get('resources',{})
    if set(resources)!=symbols or any(r.get('offline_resource_qualified')is not True or not(0<r.get('registers_per_thread',0)<=255)or any(r.get(k)!=0 for k in('stack_bytes','spill_store_bytes','spill_load_bytes','static_shared_bytes'))for r in resources.values()):raise ValueError('actual offline resources rejected')
    plan=json.loads((directory/'plan.json').read_text())
    if plan!=B.SEALED.plan()or(directory/'plan.h').read_text()!=B.SEALED.plan_header(plan):raise ValueError('sealed fixture plan changed')
    expected='std::array<Variant,'+str(len(candidates))+'> candidates{{'+','.join('{4,128,"'+v['symbol']+'"}'for v in candidates)+'}};\n'
    if(directory/'variants.h').read_text()!=expected:raise ValueError('compiled candidate list changed')
    return m
def validate_observation(events,result,manifest,bench,rounds,iterations):
    candidates=selected(manifest);symbols={B.ORACLE,*(v['symbol']for v in candidates)};by_name={v['symbol']:v for v in candidates};plan=B.SEALED.plan();names={name(s):s for s in plan}
    if len(names)!=96:raise ValueError('invalid sealed fixture identity')
    if result.get('complete')is not True or result.get('status')!='STRICT_OUTPUT_BITS_AND_ARGMAX_PASS'or result.get('activation_rows')!=8 or result.get('strict_cases_per_candidate')!=96 or result.get('strict_candidate_comparisons')!=96*len(candidates)or result.get('candidates')!=len(candidates)or result.get('timed_shapes_per_candidate')!=(12 if bench else 0)or result.get('rounds')!=rounds or result.get('iterations_per_sample')!=iterations:raise ValueError('incomplete strict/result matrix')
    groups={}
    for row in events:
        if not isinstance(row,dict)or row.get('event')not in('device','driver_resources','production_rows_query','driver_launch_gate','strict_case_pass','ABBA_gpu_ms'):raise ValueError('unknown runner event')
        groups.setdefault(row['event'],[]).append(row)
    device=groups.get('device',[])
    if len(device)!=1 or not isinstance(device[0].get('name'),str)or not device[0]['name']or any(type(device[0].get(k))is not int or device[0][k]<=0 for k in('sms','registers_per_sm','driver','runtime')):raise ValueError('missing actual device metadata')
    resources=groups.get('driver_resources',[])
    if len(resources)!=len(symbols)or{r.get('symbol')for r in resources}!=symbols:raise ValueError('missing/duplicate driver resource observations')
    for r in resources:
        if any(type(r.get(k))is not int for k in('registers','local_bytes','static_shared_bytes','max_block_threads','active_ctas'))or r.get('rows_per_warp')!=4 or r.get('block_threads')!=128 or not(0<r.get('registers',0)<=255)or r.get('local_bytes')!=0 or r.get('static_shared_bytes')!=0 or r.get('max_block_threads',0)<128 or r.get('active_ctas',0)<1:raise ValueError('actual driver resource rejection')
    query=groups.get('production_rows_query',[]);launch=groups.get('driver_launch_gate',[])
    if len(query)!=1 or query[0].get('actual_rows_per_warp')!=4 or query[0].get('expected_rows_per_warp')!=4 or query[0].get('guards')is not True or len(launch)!=1 or launch[0].get('launchable')is not True or launch[0].get('full_quality_plan_passed')is not False:raise ValueError('missing real launch/metadata pilot')
    actual=set()
    for r in groups.get('strict_case_pass',[]):
        key=(r.get('symbol'),r.get('shape'));s=names.get(key[1])
        if key in actual or key[0]not in by_name or s is None or any(r.get(k)is not True for k in('full_output_bits','oracle_AA','candidate_AA','guards','actual_output_argmax_equal')):raise ValueError('bad strict observation')
        if r.get('rows_per_warp')!=4 or r.get('block_threads')!=128 or r.get('k')!=s['k']or r.get('out_dim')!=s['out_dim']or r.get('pattern')!=s['pattern']or r.get('alpha_bits')!=s['alpha_bits']or r.get('outputs')!=8*s['out_dim']:raise ValueError('strict shape mismatch')
        ids=r.get('actual_output_argmax')
        if not isinstance(ids,list)or len(ids)!=8 or any(type(i)is not int or not 0<=i<s['out_dim']for i in ids):raise ValueError('argmax observations incomplete')
        actual.add(key)
    if actual!={(symbol,n)for symbol in by_name for n in names}:raise ValueError('strict missing/duplicate case')
    observed=set();timed={n:s for n,s in names.items()if s['timed']}
    for r in groups.get('ABBA_gpu_ms',[]):
        key=(r.get('symbol'),r.get('shape'),r.get('round'));s=timed.get(key[1]);v=by_name.get(key[0])
        if key in observed or not bench or s is None or v is None or type(key[2])is not int or not 0<=key[2]<rounds:raise ValueError('bad/duplicate timing cell')
        if r.get('rows_per_warp')!=4 or r.get('block_threads')!=128 or r.get('role')!=v['role']or r.get('k')!=s['k']or r.get('out_dim')!=s['out_dim']or r.get('iterations_per_sample')!=iterations or r.get('fullbits_each_half')is not True:raise ValueError('timed identity/gate differs')
        if any(type(r.get(k))not in(float,int)or not math.isfinite(r[k])or r[k]<=0 for k in('A1_oracle_ms','B1_candidate_ms','B2_candidate_ms','A2_oracle_ms')):raise ValueError('invalid event interval')
        observed.add(key)
    expected={(symbol,n,r)for symbol in by_name for n in timed for r in range(rounds)}if bench else set()
    if observed!=expected:raise ValueError('incomplete timed matrix')
    return dict(strict_comparisons=len(actual),ABBA_events=len(observed),driver_resources=resources,occupancy_observations_not_speedup_claim=True)
def name(s):return f"k{s['k']}-m{s['out_dim']}-p{s['pattern']}-alpha{s['alpha_bits']}"
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',required=True,type=Path);ap.add_argument('--name',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--bench',action='store_true');ap.add_argument('--rounds',type=int,default=5);ap.add_argument('--iterations',type=int,default=3);ap.add_argument('--timeout',type=int,default=300);a=ap.parse_args(argv)
    if Path(a.name).name!=a.name or a.name in('.','..')or min(a.rounds,a.iterations,a.timeout)<1:ap.error('single fresh name and positive bounds required')
    directory=B.evidence_dir(a.build);out=directory/a.name;out.mkdir(exist_ok=False)
    state=dict(schema='q35-dp4a-schedule-run/1',complete=False,gpu_invoked=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hardware_counters='NOT_RUN',production_integrated=False)
    save=lambda:B.save(out/'run.json',state);save()
    if not a.execute:state['status']='PLAN_ONLY_GPU_NOT_RUN';save();print(json.dumps(state));return 0
    try:
        manifest=verify_build(directory);before=B.digest(directory/'manifest.json');cmd=[str(directory/B.RUNNER),str(directory),str(out),'--bench'if a.bench else'--strict','--acts','8','--rounds',str(a.rounds),'--iterations',str(a.iterations)]
        state.update(command=cmd,timeout_seconds=a.timeout,build_manifest_sha256=before,build_manifest=manifest,status='GPU_RUNNER_REQUESTED_NOT_PASS');save()
        with(out/'runner.jsonl').open('x')as log:
            def registered(process):state.update(gpu_invoked=True,owned_pid=process.pid);save()
            exit_code,_=B.command(cmd,a.timeout,stdout=log,on_started=registered)
        result=json.loads((out/'result.json').read_text());events=[json.loads(l)for l in(out/'runner.jsonl').read_text().splitlines()if l.strip()]
        state.update(exit_code=exit_code,result=result,runner_log_sha256=B.digest(out/'runner.jsonl'))
        if exit_code!=0:raise ValueError('private runner failed; raw retained')
        observed=validate_observation(events,result,manifest,a.bench,a.rounds,a.iterations)
        verify_build(directory)
        if before!=B.digest(directory/'manifest.json'):raise ValueError('build identity changed while running')
        state.update(complete=True,status='STRICT_PASS_SELECTED_SYNTHETIC_ARMS',observations=observed,source_artifact_pre_post_verified=True);save();print(json.dumps(state,indent=2));return 0
    except BaseException as error:
        state.update(complete=False,status='REJECTED_NOT_VALIDATED',error=repr(error));save()
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
        print(json.dumps(state,indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
