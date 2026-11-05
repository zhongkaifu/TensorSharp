#!/usr/bin/env python3
"""One exact-body cold CUA diagnostic against an ALREADY started fresh server.

Never starts/stops a server, discovers a model, warms a cache, or edits fixtures.
--prepare-only is CPU/file-only. Execution makes exactly one POST. complete is
execution completeness, NOT quality PASS; actual explicit cached_tokens==0 and
unchanged prompt usage are mandatory for diagnostic qualification.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def select(path, scenario, step, lane, lanes=None):
    data = path.read_bytes(); entries = [(i,json.loads(x)) for i,x in enumerate(data.decode().splitlines(),1) if x.strip()]
    starts = [(i,r) for i,r in entries if r.get('kind') == 'run']
    ends = [(i,r) for i,r in entries if r.get('kind') == 'run_end']
    if len(starts)!=1 or len(ends)!=1 or ends[0][1].get('complete') is not True or starts[0][0]>=ends[0][0]:
        raise ValueError('source must be one completed CUA replay; selected quality FAIL is allowed')
    header = starts[0][1]; lanes = header['lanes'] if lanes is None else lanes
    matches = [(i,r) for i,r in entries if r.get('kind')=='request' and
               (r.get('scenario'),r.get('lanes'),r.get('lane'),r.get('step'))==(scenario,lanes,lane,step)]
    if len(matches)!=1 or not starts[0][0]<matches[0][0]<ends[0][0]:
        raise ValueError('selected request is missing, duplicate or outside the source run')
    if header['lanes']!=lanes: raise ValueError('source header/request lanes disagree')
    return header,matches[0][1],dict(path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),raw_line=matches[0][0])


def client(bench, header, url):
    # A nonempty original model ID prevents Client.discover_model HTTP traffic.
    if not header.get('model'): raise ValueError('source model routing ID is required')
    return bench.Client(types.SimpleNamespace(engine=header['engine'],url=url,model=header['model'],
        extra_body=None,metrics_url='',tokenizer=None,server_log=None,timeout=120,
        temperature=0,top_p=1,sampling_seed=1234))


def prepare(bench, cua, header, source, fixture, fixture_manifest, url):
    if header['sources']!=cua.sources() or fixture_manifest['sources']!=header['sources']:
        raise ValueError('frozen producer/client/generator identity differs from source')
    if fixture_manifest['fixture_sha256']!=header['fixture_manifest']['fixture_sha256']:
        raise ValueError('this is not the original prepared fixture')
    identity = tuple(source[k] for k in ('scenario','lanes','lane','step'))
    matches = [r for r in fixture['requests'] if tuple(r[k] for k in ('scenario','lanes','lane','step'))==identity]
    if len(matches)!=1: raise ValueError('original fixture identity is not unique')
    row=matches[0];common=row['common'];original=source['request_body'];c=client(bench,header,url)
    # JSONL canonicalizes dictionary order. Restore the ORIGINAL frozen builder
    # order, but reject ANY field/value difference before using its ordered body.
    body=c.body(common['messages'],common['max_tokens'],{k:v for k,v in common.items() if k not in ('messages','max_tokens')})
    wire_bytes=json.dumps(body).encode('utf-8') # exactly bench.stream_chat encoding
    checks=dict(all_request_fields_unchanged=body==original,
        common_sha256=cua.digest(common)==source['common_sha256']==row['common_sha256'],
        full_body_sha256=cua.digest(body)==source['full_body_sha256']==cua.digest(original),
        canonical_input_sha256=cua.digest({k:v for k,v in body.items() if k!='model'})==source['canonical_input_sha256'],
        normalized_diagnostic_sha256=cua.digest(bench.request_identity(body))==source['frozen_normalized_request_sha256'],
        exact_original_wire_bytes_sha256=hashlib.sha256(wire_bytes).hexdigest()==source['wire_request_body_sha256'],
        expected_unchanged=row['expected']==source['expected'])
    if not all(checks.values()): raise ValueError('request provenance mismatch: '+json.dumps(checks))
    return c,body,checks,dict(common_sha256=source['common_sha256'],full_body_sha256=cua.digest(body),
        canonical_input_sha256=cua.digest({k:v for k,v in body.items() if k!='model'}),
        wire_request_body_sha256=hashlib.sha256(wire_bytes).hexdigest(),wire_request_body_bytes=len(wire_bytes))


def same_deployment(original, fresh):
    def files(value):
        found={}
        def walk(v):
            if isinstance(v,dict):
                if 'path'in v and 'sha256'in v:
                    path,digest=v['path'],v['sha256']
                    if not isinstance(path,str)or not isinstance(digest,str)or len(digest)!=64:
                        raise ValueError('missing file SHA identity')
                    if path in found and found[path]!=digest:raise ValueError('conflicting file identity')
                    found[path]=digest
                for x in v.values():walk(x)
            elif isinstance(v,list):
                for x in v:walk(x)
        walk(value)
        if not found:raise ValueError('empty deployment file inventory')
        return found
    oldenv=original.get('environment')or{};newenv=fresh.get('environment')or{}
    fixed={'KV_CACHE_DTYPE','MAX_CONTEXT','TS_SCHED_MAX_RUNNING_SEQS','TS_PREFIX_CACHE_ISOLATION','TS_BACKEND','CUDA_VISIBLE_DEVICES'}
    keys=sorted(k for k in set(oldenv)|set(newenv)|fixed if k in fixed or k.startswith(('TS_Q35_','TS_CUDA_','GGML_')))
    checks={k:original.get(k)==fresh.get(k) for k in ('engine','mode','argv','working_directory','context_tokens','lanes')}
    checks.update(binaries_path_sha256=files(original.get('binaries'))==files(fresh.get('binaries')),
        models_path_sha256=files(original.get('models'))==files(fresh.get('models')),
        math_environment=all(oldenv.get(k)==newenv.get(k)for k in keys))
    if not all(checks.values()):raise ValueError('fresh deployment differs: '+json.dumps(checks))
    return dict(checks=checks,math_environment={k:oldenv.get(k)for k in keys},
        ignored_metadata='timestamps,source Git status/metadata,PID/starttime/run_id,logs/output/prefix-cache directory metadata; argv remains exact')


def observe(bench,cua,c,body,source,timeout,opener=None):
    obs,wire=cua.captured_stream(bench,c.url,body,timeout,opener)
    try:result=c.finish(obs)
    except Exception as error:
        result=dict(error=f'frozen_client_normalization: {type(error).__name__}: {error}',
            http_status=obs.get('http_status'),finish_reason=obs.get('finish_reason'),
            content=''.join(obs.get('content')or[]),reasoning=''.join(obs.get('reasoning')or[]),
            reasoning_chars=sum(len(s)for s in obs.get('reasoning')or[]),tool_calls=[],
            ttft_s=None,total_s=obs['t_end']-obs['t0'])
    result['raw_usage']=obs['usage'];result['multiple_choices']=obs.get('multiple_choices',False)
    result['decode_tok_s']=None;result['decode_unavailable_reason']='CUA_tool_parser_stream_not_raw_target_token_stream'
    result.pop('legacy_completion_minus_one_decode_tok_s',None)
    quality=cua.validate(bench,result,wire,source['expected'],body['max_tokens'],body['tools'][0]['function']['parameters'])
    usage=obs.get('usage')or{};details=usage.get('prompt_tokens_details')or{};cached=details.get('cached_tokens')
    old_usage=source.get('raw_observation',{}).get('usage')or{}
    source_prompt=old_usage.get('prompt_tokens')
    cold=dict(actual_cached_tokens=cached,source='raw_observation.usage.prompt_tokens_details.cached_tokens',
        verdict='UNVERIFIED' if type(cached)is not int else 'PASS' if cached==0 else 'FAIL',
        actual_prompt_tokens=usage.get('prompt_tokens'),source_prompt_tokens=source_prompt,
        prompt_tokens_unchanged=type(source_prompt)is int and source_prompt>0 and usage.get('prompt_tokens')==source_prompt,
        exact_wire_sha256=wire.get('request_body_sha256')==source['wire_request_body_sha256'])
    complete=(wire.get('done')is True and wire.get('closed_normally')is True and result.get('http_status')==200
        and not result.get('error') and result.get('finish_reason')in('stop','tool_calls','function_call'))
    qualified=(complete and cold['verdict']=='PASS' and cold['prompt_tokens_unchanged'] and cold['exact_wire_sha256']
        and quality['latency_qualified']is True)
    return dict(raw_observation=obs,raw_transport=wire,response=result,quality=quality,cold_gate=cold,
        execution_complete=complete,complete=complete,diagnostic_qualified=qualified)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--harness-root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--fixtures',type=Path);ap.add_argument('--url');ap.add_argument('--manifest',type=Path)
    ap.add_argument('--scenario',default='edit');ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--lane',type=int,default=0);ap.add_argument('--lanes',type=int)
    ap.add_argument('--timeout',type=float,default=120);ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args(argv)
    if not math.isfinite(args.timeout)or args.timeout<=0:ap.error('positive finite timeout required')
    if not args.prepare_only and (not args.url or not args.manifest):ap.error('one POST requires explicit --url and fresh-server --manifest')
    out=args.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x'):pass # reserve; never overwrite earlier diagnostic evidence
    state=dict(schema='qwen38-cua-cold-control/1',kind='single_request_diagnostic',complete=False,
        execution_complete=False,diagnostic_qualified=False,prepare_only=args.prepare_only,
        script_sha256=sha(__file__),started_utc=datetime.now(timezone.utc).isoformat(),http_posts=0,
        limits=['Operator supplies an already fresh server; this tool never starts/stops/warms it.',
            'One exact historical body, not a new fixture,30-step replay,independent run or cache implementation proof.',
            'Original quality remains original; cold checks use the same frozen validator without score changes.',
            'No tool decode tok/s; freshness qualification requires explicit actual cached_tokens=0.'])
    def save():
        tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(out)
    save()
    try:
        cua=load(args.harness_root/'eng/validation/qwen38-cua-control.py','cold_frozen_cua');bench=cua.load_bench()
        header,source,ref=select(args.replay,args.scenario,args.step,args.lane,args.lanes)
        fixtures=args.fixtures or Path(header['fixture_path']);fixture,fm=cua.load_fixture(fixtures)
        c,body,checks,identity=prepare(bench,cua,header,source,fixture,fm,args.url or header['url'])
        state.update(source=ref,source_identity={k:source[k]for k in('scenario','lanes','lane','step','scope')},
            source_quality=source['quality'],source_response=source['response'],source_raw_usage=source['raw_observation']['usage'],
            source_launch_manifest=header['launch_manifest'],source_launch_manifest_sha256=header['launch_manifest_sha256'],
            sources=header['sources'],fixture_sha256=fm['fixture_sha256'],request_body=body,
            prepared_request_wire_body_base64=base64.b64encode(json.dumps(body).encode('utf-8')).decode(),
            identity=identity,identity_checks=checks,wire_reconstruction='same frozen Client.body + original fixture; full field and exact original wire SHA enforced',status='PREPARED_NOT_SENT')
        save()
        if args.prepare_only:return 0
        deployment=json.loads(args.manifest.read_text());cua.validate_deployment(deployment,types.SimpleNamespace(
            engine=header['engine'],context_capacity=header['context_capacity_operator_declaration'],lanes=1,
            cache_profile=header['cache_profile_operator_declaration']),fixture)
        state['same_deployment_gate']=same_deployment(header['launch_manifest'],deployment);save()
        old=header['launch_manifest'].get('process')or{};new=deployment.get('process')or{}
        keys=('pid','starttime','boot_id')
        if not all(new.get(k)is not None for k in keys)or tuple(old.get(k)for k in keys)==tuple(new.get(k)for k in keys):
            raise ValueError('fresh process identity must be explicit and different from original replay server')
        state.update(deployment_manifest=deployment,deployment_manifest_sha256=sha(args.manifest),
            fresh_process_identity_differs=True,status='REQUEST_IN_FLIGHT',http_posts=1);save()
        state.update(observe(bench,cua,c,body,source,args.timeout));state['status']='COLD_QUALIFIED'if state['diagnostic_qualified']else'COLD_OR_QUALITY_UNQUALIFIED'
    except Exception as error:
        state.update(error=f'{type(error).__name__}: {error}',status='ERROR_OR_UNVERIFIED',complete=False,diagnostic_qualified=False)
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();state['complete_is_NOT_quality_PASS']=True;save()
    print(json.dumps({k:state.get(k)for k in('status','complete','diagnostic_qualified','cold_gate','quality','identity_checks')},indent=2))
    return 0 if state['diagnostic_qualified'] else 1


if __name__=='__main__':raise SystemExit(main())
