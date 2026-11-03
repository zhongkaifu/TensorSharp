#!/usr/bin/env python3
"""Default-plan entrypoint; --execute alone permits private synthetic GPU gates."""
import argparse,datetime,importlib.util,json,math,os,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('q81_shared_tile_build',HERE/'build.py');B=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(B)
def selected(manifest):
    expected=B.chosen(manifest.get('arm_selection'))
    if manifest.get('candidates')!=expected:raise ValueError('changed bounded geometries')
    return expected
def verify_build(directory):
    B.check_frozen()
    m=json.loads((directory/'manifest.json').read_text());p=json.loads((directory/B.PERMIT).read_text());candidates=selected(m);symbols={B.ORACLE,B.QUANT,*(v['symbol']for v in candidates)}
    if(m.get('schema')!='q35-q81-shared-tile-build/1'or m.get('complete')is not True or m.get('activation_rows')!=8 or m.get('strict_cases')!=96 or m.get('timed_roles')!=B.TIMED or m.get('timed_unique_geometries')!=5 or m.get('primary_timing_scope')!='complete_original_q81_quant_plus_gemv' or m.get('flags')!=B.FLAGS or m.get('oracle')!=dict(symbol=B.ORACLE,rows_per_warp=4,block_threads=128)or m.get('oracle_is_freshly_recompiled_NOT_production_BBD')is not True or m.get('supports_cache_sweep')is not True or m.get('cache_sweep_kernel')!=B.SWEEP_SYMBOL or m.get('candidate_shared_bytes')!=B.SHARED_BYTES or m.get('staging_tile_blocks')!=32 or m.get('staging_block_stride_bytes')!=80 or m.get('planned_rounds')!=10):raise ValueError('missing actual complete offline build')
    if p.get('schema')!='q35-q81-shared-tile-offline/1'or p.get('manifest_sha256')!=B.digest(directory/'manifest.json')or p.get('artifacts')!=m.get('artifacts'):raise ValueError('missing/changed real offline marker')
    if set(m.get('artifacts',{}))!=set(B.ARTIFACTS)or any(B.digest(directory/n)!=h for n,h in m['artifacts'].items()):raise ValueError('built artifacts changed')
    if m.get('tools')!={n:B.digest(HERE/n)for n in B.TOOLS}or m.get('frozen_fixture_tools')!=B.fixture_pins()or B.digest(m['source'])!=m['source_sha256']:raise ValueError('tool/fixture/production source changed')
    if m['source_sha256']!=B.SOURCE_SHA or m.get('quantizer_source_sha256')!=B.QUANT_SOURCE_SHA:raise ValueError('original oracle/quantizer freeze differs')
    resources=m.get('resources',{})
    if set(resources)!=symbols or any(r.get('offline_resource_qualified')is not True or not(0<r.get('registers_per_thread',0)<=255)or any(r.get(k)!=0 for k in('stack_bytes','spill_store_bytes','spill_load_bytes'))or r.get('static_shared_bytes')!=B.expected_shared(symbol)for symbol,r in resources.items()):raise ValueError('actual offline resources rejected')
    sweep=m.get('sweep_resources',{})
    if sweep.get('offline_resource_qualified')is not True or sweep.get('block_threads')!=256 or not(0<sweep.get('registers_per_thread',0)<=255)or any(sweep.get(k)!=0 for k in('stack_bytes','spill_store_bytes','spill_load_bytes')):raise ValueError('actual sweep resources rejected')
    plan=json.loads((directory/'plan.json').read_text())
    if plan!=B.SEALED.plan()or(directory/'plan.h').read_text()!=B.SEALED.plan_header(plan):raise ValueError('sealed fixture plan changed')
    expected=B.variants_header(candidates)
    if(directory/'variants.h').read_text()!=expected:raise ValueError('compiled candidate list changed')
    if json.loads((directory/'timed.json').read_text())!=B.TIMED or(directory/'timed.h').read_text()!=B.timed_header():raise ValueError('six-role timed plan changed')
    return m
def validate_observation(events,result,manifest,bench,rounds,iterations,cache_sweep=False):
    if rounds!=10:raise ValueError('bounded experiment requires exactly10 rounds')
    candidates=selected(manifest);symbols={B.ORACLE,*(v['symbol']for v in candidates)};by_name={v['symbol']:v for v in candidates};plan=B.SEALED.plan();names={name(s):s for s in plan}
    if len(names)!=96:raise ValueError('invalid sealed fixture identity')
    if result.get('complete')is not True or result.get('status')!='STRICT_OUTPUT_BITS_AND_ARGMAX_PASS'or result.get('activation_rows')!=8 or result.get('strict_cases_per_candidate')!=96 or result.get('strict_candidate_comparisons')!=96*len(candidates)or result.get('candidates')!=len(candidates)or result.get('timed_shapes_per_candidate')!=(6 if bench else 0)or result.get('timed_unique_geometries')!=5 or result.get('scope')!='complete_original_q81_quant_plus_gemv'or result.get('rounds')!=rounds or result.get('iterations_per_sample')!=iterations:raise ValueError('incomplete strict/result matrix')
    mode='all_data_l2_eviction_control'if cache_sweep else'warm_repeated_matrix'
    if result.get('cache_sweep')is not cache_sweep or result.get('cache_mode')!=mode or(cache_sweep and(not bench or iterations!=1)):raise ValueError('cache mode/iteration contract differs')
    groups={}
    for row in events:
        if not isinstance(row,dict)or row.get('event')not in('device','driver_resources','production_rows_query','driver_launch_gate','strict_case_pass','ABBA_gpu_ms','cache_sweep_setup','cache_sweep','quantizer_resources','pipeline_case_pass'):raise ValueError('unknown runner event')
        groups.setdefault(row['event'],[]).append(row)
    device=groups.get('device',[])
    if len(device)!=1 or not isinstance(device[0].get('name'),str)or not device[0]['name']or any(type(device[0].get(k))is not int or device[0][k]<=0 for k in('sms','registers_per_sm','driver','runtime')):raise ValueError('missing actual device metadata')
    resources=groups.get('driver_resources',[])
    if len(resources)!=len(symbols)or{r.get('symbol')for r in resources}!=symbols:raise ValueError('missing/duplicate driver resource observations')
    layout={B.ORACLE:(4,128),**{x['symbol']:(x['rows_per_warp'],x['block_threads'])for x in candidates}}
    for r in resources:
        shape=layout.get(r.get('symbol'));expected_rows,expected_threads=shape if shape else(0,0)
        if any(type(r.get(k))is not int for k in('registers','local_bytes','static_shared_bytes','max_block_threads','active_ctas'))or r.get('rows_per_warp')!=expected_rows or r.get('block_threads')!=expected_threads or not(0<r.get('registers',0)<=255)or r.get('local_bytes')!=0 or r.get('static_shared_bytes')!=B.expected_shared(r.get('symbol')) or r.get('max_block_threads',0)<expected_threads or r.get('active_ctas',0)<1:raise ValueError('actual driver resource rejection')
    q=groups.get('quantizer_resources',[])
    if len(q)!=1 or any(type(q[0].get(k))is not int for k in('registers','local_bytes','static_shared_bytes','active_ctas','max_block_threads','block_threads'))or q[0].get('symbol')!=B.QUANT or q[0].get('block_threads')!=256 or not 0<q[0].get('registers',0)<=255 or q[0].get('local_bytes')!=0 or q[0].get('static_shared_bytes')!=0 or q[0].get('active_ctas',0)<1 or q[0].get('max_block_threads',0)<256:raise ValueError('actual original quantizer resources rejected')
    query=groups.get('production_rows_query',[]);launch=groups.get('driver_launch_gate',[])
    if len(query)!=1 or query[0].get('actual_rows_per_warp')!=4 or query[0].get('expected_rows_per_warp')!=4 or query[0].get('guards')is not True or len(launch)!=1 or launch[0].get('launchable')is not True or launch[0].get('full_quality_plan_passed')is not False:raise ValueError('missing real launch/metadata pilot')
    actual=set()
    for r in groups.get('strict_case_pass',[]):
        key=(r.get('symbol'),r.get('shape'));s=names.get(key[1])
        if key in actual or key[0]not in by_name or s is None or any(r.get(k)is not True for k in('full_output_bits','oracle_AA','candidate_AA','guards','actual_output_argmax_equal')):raise ValueError('bad strict observation')
        if r.get('rows_per_warp')!=by_name[key[0]]['rows_per_warp']or r.get('block_threads')!=by_name[key[0]]['block_threads'] or r.get('k')!=s['k']or r.get('out_dim')!=s['out_dim']or r.get('pattern')!=s['pattern']or r.get('alpha_bits')!=s['alpha_bits']or r.get('outputs')!=8*s['out_dim']:raise ValueError('strict shape mismatch')
        ids=r.get('actual_output_argmax')
        if not isinstance(ids,list)or len(ids)!=8 or any(type(i)is not int or not 0<=i<s['out_dim']for i in ids):raise ValueError('argmax observations incomplete')
        actual.add(key)
    if actual!={(symbol,n)for symbol in by_name for n in names}:raise ValueError('strict missing/duplicate case')
    observed=set();timed={s['role']:s for s in B.TIMED};pipeline=set()
    for r in groups.get('pipeline_case_pass',[]):
        key=(r.get('symbol'),r.get('plane_role'));s=timed.get(key[1])
        if key in pipeline or not bench or key[0]not in by_name or s is None or r.get('shape')!=name(s)or r.get('Q81_dual_poison_AA')is not True or r.get('full_output_bits_AA_guards')is not True:raise ValueError('pipeline fullbits/quantizer A/A gate missing or invalid')
        pipeline.add(key)
    if pipeline!=({(v,n)for v in by_name for n in timed}if bench else set()):raise ValueError('pipeline six-role matrix incomplete')
    for r in groups.get('ABBA_gpu_ms',[]):
        key=(r.get('symbol'),r.get('plane_role'),r.get('round'));s=timed.get(key[1]);v=by_name.get(key[0])
        if key in observed or not bench or s is None or v is None or type(key[2])is not int or not 0<=key[2]<rounds:raise ValueError('bad/duplicate timing cell')
        if r.get('rows_per_warp')!=v['rows_per_warp']or r.get('block_threads')!=v['block_threads']or r.get('shape')!=name(s)or r.get('scope')!='complete_original_q81_quant_plus_gemv' or r.get('role')!=v['role']or r.get('k')!=s['k']or r.get('out_dim')!=s['out_dim']or r.get('iterations_per_sample')!=iterations or r.get('fullbits_each_half')is not True:raise ValueError('timed identity/gate differs')
        if any(type(r.get(k))not in(float,int)or not math.isfinite(r[k])or r[k]<=0 for k in('A1_oracle_ms','B1_candidate_ms','B2_candidate_ms','A2_oracle_ms')):raise ValueError('invalid event interval')
        observed.add(key)
    expected={(symbol,n,r)for symbol in by_name for n in timed for r in range(rounds)}if bench else set()
    if observed!=expected:raise ValueError('incomplete timed matrix')
    setups=groups.get('cache_sweep_setup',[]);sweeps=groups.get('cache_sweep',[])
    if not cache_sweep:
        if setups or sweeps or result.get('cache_sweeps')!=0:raise ValueError('warm mode unexpectedly swept')
    else:
        if len(setups)!=1:raise ValueError('missing/duplicate actual cache setup')
        setup=setups[0];l2=setup.get('driver_l2_bytes')
        if type(l2)is not int or not 0<l2<2**31:raise ValueError('invalid actual driver L2 size')
        geometry=B.sweep_geometry(l2,device[0]['sms'])
        if any(setup.get(k)!=v for k,v in geometry.items())or setup.get('mode')!=mode or setup.get('method')!='volatile_asm_ld_global_cg_u32_grid_stride_per_block_u64_sum_and_count'or setup.get('physical_cold_guaranteed')is not False or setup.get('excluded_from_event_interval')is not True:raise ValueError('cache sweep actual size/method differs')
        if any(type(setup.get(k))is not int for k in('registers','local_bytes','static_shared_bytes','active_ctas','max_block_threads'))or not 0<setup['registers']<=255 or setup['local_bytes']!=0 or setup['active_ctas']<1 or setup['max_block_threads']<256:raise ValueError('sweep driver resources rejected')
        expected_sweeps={(v,n,r,half)for v,n,r in expected for half in('A1','B1','B2','A2')};actual_sweeps=set()
        for index,row in enumerate(sweeps,1):
            key=(row.get('candidate'),row.get('plane_role'),row.get('round'),row.get('half'))
            measured=B.ORACLE if key[3]in('A1','A2')else key[0]
            if key not in expected_sweeps or row.get('shape')!=name(timed[key[1]])or key in actual_sweeps or row.get('index')!=index or row.get('measured_symbol')!=measured or row.get('driver_l2_bytes')!=l2 or row.get('sweep_bytes')!=geometry['sweep_bytes']or row.get('words_read')!=geometry['words']or row.get('checksum')!=geometry['words']*B.SWEEP_VALUE or any(row.get(k)is not True for k in('guards','checksum_verified','synchronized_before_event','excluded_from_event_interval')):raise ValueError('missing/duplicate/failed full-read sweep half')
            actual_sweeps.add(key)
        if actual_sweeps!=expected_sweeps or result.get('cache_sweeps')!=len(expected_sweeps):raise ValueError('incomplete cache-sweep matrix')
        first_setup=next(i for i,r in enumerate(events)if r['event']=='cache_sweep_setup')
        if any(r['event']=='strict_case_pass'for r in events[first_setup+1:]):raise ValueError('sweep setup preceded complete strict gate')
        for index,row in enumerate(events):
            if row['event']!='ABBA_gpu_ms':continue
            preceding=events[index-4:index]
            if len(preceding)!=4 or any(r['event']!='cache_sweep'or(r['candidate'],r['plane_role'],r['round'])!=(row['symbol'],row['plane_role'],row['round'])for r in preceding)or[r['half']for r in preceding]!=['A1','B1','B2','A2']:raise ValueError('ABBA lacks four immediately preceding controlled halves')
    first_timing=next((i for i,r in enumerate(events)if r['event']=='ABBA_gpu_ms'),len(events))
    if any(r['event']=='strict_case_pass'for r in events[first_timing:]):raise ValueError('timing started before all96 strict cases')
    for i,r in enumerate(events):
        if r['event']=='ABBA_gpu_ms'and not any(x['event']=='pipeline_case_pass'and x['symbol']==r['symbol']and x['plane_role']==r['plane_role']for x in events[:i]):raise ValueError('timing preceded matching complete quant+gemv strict gate')
    return dict(pipeline_comparisons=len(pipeline),timing_scope='complete_original_q81_quant_plus_gemv',timed_role_count=6,timed_unique_geometries=5,strict_comparisons=len(actual),ABBA_events=len(observed),driver_resources=resources,cache_mode=mode,cache_sweep_setup=setups[0]if setups else None,cache_sweeps=len(sweeps),occupancy_observations_not_speedup_claim=True)
def name(s):return f"k{s['k']}-m{s['out_dim']}-p{s['pattern']}-alpha{s['alpha_bits']}"
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--build',required=True,type=Path);ap.add_argument('--name',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--bench',action='store_true');ap.add_argument('--cache-sweep',action='store_true');ap.add_argument('--rounds',type=int,default=10);ap.add_argument('--iterations',type=int);ap.add_argument('--timeout',type=int,default=300);a=ap.parse_args(argv)
    if a.iterations is None:a.iterations=1 if a.cache_sweep else 3
    if a.cache_sweep and(not a.bench or a.iterations!=1):ap.error('--cache-sweep requires --bench and iterations1; no warm repeat within a half')
    if Path(a.name).name!=a.name or a.name in('.','..')or min(a.rounds,a.iterations,a.timeout)<1 or a.rounds!=10:ap.error('single fresh name and positive bounds required')
    directory=B.evidence_dir(a.build);out=directory/a.name;out.mkdir(exist_ok=False)
    state=dict(schema='q35-q81-shared-tile-run/1',complete=False,gpu_invoked=False,cache_sweep=a.cache_sweep,cache_mode='all_data_l2_eviction_control'if a.cache_sweep else'warm_repeated_matrix',iterations=a.iterations,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hardware_counters='NOT_RUN',production_integrated=False)
    save=lambda:B.save(out/'run.json',state);save()
    if not a.execute:state['status']='PLAN_ONLY_GPU_NOT_RUN';save();print(json.dumps(state));return 0
    try:
        manifest=verify_build(directory);before=B.digest(directory/'manifest.json');cmd=[str(directory/B.RUNNER),str(directory),str(out),'--bench'if a.bench else'--strict','--acts','8','--rounds',str(a.rounds),'--iterations',str(a.iterations)]
        if a.cache_sweep:cmd.append('--cache-sweep')
        state.update(command=cmd,timeout_seconds=a.timeout,build_manifest_sha256=before,build_manifest=manifest,status='GPU_RUNNER_REQUESTED_NOT_PASS');save()
        with(out/'runner.jsonl').open('x')as log:
            def registered(process):state.update(gpu_invoked=True,owned_pid=process.pid);save()
            exit_code,_=B.command(cmd,a.timeout,stdout=log,on_started=registered)
        result=json.loads((out/'result.json').read_text());events=[json.loads(l)for l in(out/'runner.jsonl').read_text().splitlines()if l.strip()]
        state.update(exit_code=exit_code,result=result,runner_log_sha256=B.digest(out/'runner.jsonl'))
        if exit_code!=0:raise ValueError('private runner failed; raw retained')
        observed=validate_observation(events,result,manifest,a.bench,a.rounds,a.iterations,a.cache_sweep)
        verify_build(directory)
        if before!=B.digest(directory/'manifest.json'):raise ValueError('build identity changed while running')
        state.update(complete=True,status='STRICT_PASS_SELECTED_SYNTHETIC_ARMS',observations=observed,source_artifact_pre_post_verified=True);save();print(json.dumps(state,indent=2));return 0
    except BaseException as error:
        state.update(complete=False,status='REJECTED_NOT_VALIDATED',error=repr(error));save()
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
        print(json.dumps(state,indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
