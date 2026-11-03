"""Literal frozen helper subset for controlled CUA validation; no executable main."""
import base64, hashlib, json, types
from pathlib import Path
ORIGIN_SHA256='bcb1601a5cca35c96d4ed4fc3d427e9fe73386d4863809df8cf9037faf8f7c4a'
FROZEN_FUNCTION_AST_SHA256={'sha': '75bdf9c01d35098ea82a48e9b6cd2253c0b13b1993e267ef403d0c353dbc9e7f', 'fixture_snapshot_cua': '868736c7d9ab441b426b0d3b66b7f0c29c81fed6387e29f9dbfc4c316fb9c91b', 'fixture_contract': '801fd117e07bb6b2e1c82c117308814c6ec7b9ef0f71e4c8ba81b66a8d9156b6', 'response_hash': '4aedf92446ca42846fde89931d62173fe218ce89c0cdb8244f62f14ea78a45c3', 'audit_cua': '245d0304d969ac0b8c746940f5dfd4a2b52cad594e4f8912311ed82bb69ee91c', 'pair_cua': '0412ba0bb61ba0290cdc3b48cb4175b274df416d52fb0e3b50b22979765ee2df'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


CUA_SHA='ab25261e8f3bcdc4ed2dfee8c010d6802b57cd6fb29a881daf63536a344eaf69'


MODEL_ID='Qwen3.8-27B-NVFP4-MTP-N4_0'


FIXTURES={False:('cua-public-v1','2b84088f7d87bb430898458f938640f052915c70a02ac666ba8cdd4fd4003251','0d9229e106d57b57d56ef114e38d21cd03f109c12cc2cdc63e36af0dd7ec2d85'),True:('cua-media-probe-v1','fe7af7ac47a53f9ee08e0077d6ba0de4fd327d818b78a001237768efddcebef2','75f73a999850e0c3021b5511d1e72e0e69a853663f418a3a490ab091a7290183')}


QUALITY={'http_success','single_choice','stream_complete','no_hidden_thinking','one_correct_tool','typed_exact_arguments','no_freeform_answer'}


MEASUREMENT={'actual_server_usage','finite_client_timings','ttft_within_request'}


def fixture_snapshot_cua(harness,media):
 result={str(harness/'artifacts/qwen38-regression'/name/leaf):sha(harness/'artifacts/qwen38-regression'/name/leaf)for enabled,(name,_,_)in FIXTURES.items()if not enabled or media for leaf in('fixture.json','manifest.json')}
 result.update({str(harness/'eng'/name):sha(harness/'eng'/name)for name in('qwen38-engine-bench.py','qwen38-engine-servers.sh','qwen38-engine-process.py','qwen38-engine-report.py','validation/qwen38-cua-control.py')});return result


def fixture_contract(b,harness,media):
 tool=harness/'eng/validation/qwen38-cua-control.py'
 if sha(tool)!=CUA_SHA:raise ValueError('frozen CUA producer changed')
 cua=b.load(tool,'Main_Q81_frozen_CUA');bench=cua.load_bench();result={}
 for enabled,(name,manifest_hash,fixture_hash)in FIXTURES.items():
  if enabled and not media:continue
  directory=harness/'artifacts/qwen38-regression'/name
  if sha(directory/'manifest.json')!=manifest_hash or sha(directory/'fixture.json')!=fixture_hash:raise ValueError('prepared common fixture changed')
  fixture,_=cua.load_fixture(directory)
  if type(fixture['steps'])is not int or fixture['steps']!=(6 if enabled else 30)or fixture['scenarios']!=(['edit_media']if enabled else['append','truncate','edit'])or fixture['declared_context_capacity']!=65536:raise ValueError('exact CUA protocol/context changed')
  result[enabled]=(directory,fixture)
 return cua,bench,result


def response_hash(response):
 value=dict(content=response.get('content')or'',reasoning=response.get('reasoning')or'',tool_calls=[dict(name=t['name'],arguments=t['parsed']if t.get('parsed')is not None else t['arguments'])for t in response.get('tool_calls',[])])
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def audit_cua(path,job,label,launch,manifest_sha,fixture,cua,bench,exit_code):
 rows=[json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()];headers=[r for r in rows if r.get('kind')=='run'];ends=[r for r in rows if r.get('kind')=='run_end'];requests=[r for r in rows if r.get('kind')=='request'];waves=[r for r in rows if r.get('kind')=='wave'];summaries=[r for r in rows if r.get('kind')=='case_summary']
 expected={(s,2,l,t)for s in job['scenarios']for l in range(2)for t in range(job['steps'])};ids=[tuple(r.get(k)for k in('scenario','lanes','lane','step'))for r in requests];wid=[tuple(r.get(k)for k in('scenario','lanes','step'))for r in waves]
 if type(exit_code)is not int or exit_code not in(0,1)or len(headers)!=1 or len(ends)!=1 or not rows or rows[0]is not headers[0]or rows[-1]is not ends[0]or any(r.get('kind')not in('run','run_end','request','wave','case_summary')for r in rows):raise ValueError('complete single-run CUA transport required')
 header=headers[0];end=ends[0]
 fields=dict(schema=cua.VERSION,engine='tensorsharp',mode='dflash2',model=MODEL_ID,label=label,lanes=2,steps=job['steps'],selected_scenarios=job['scenarios'],launch_manifest_sha256=manifest_sha,launch_manifest=launch,context_capacity_operator_declaration=65536,cache_profile_operator_declaration='trusted-single-tenant-shared',cache_comparison_group='trusted-single-tenant-prefix',protocol=fixture['protocol'])
 if any(header.get(k)!=v for k,v in fields.items())or header.get('sources')!=fixture['sources']or header.get('fixture_manifest',{}).get('fixture_sha256')!=FIXTURES[job['media']][2]or header.get('settings')!=dict(temperature=0,top_p=1,seed=1234,thinking=False,tool_choice='auto',max_tokens=192,warmup='NONE',prior_history='prescribed'):raise ValueError('actual CUA header/fixture/deployment/settings differ')
 if any(not isinstance(s,str)or any(type(v)is not int for v in(n,l,t))for s,n,l,t in ids)or len(ids)!=len(set(ids))or set(ids)!=expected:raise ValueError('missing/duplicate/invalid request identity')
 if any(not isinstance(s,str)or type(n)is not int or type(t)is not int for s,n,t in wid)or len(wid)!=len(set(wid))or set(wid)!={(s,2,t)for s in job['scenarios']for t in range(job['steps'])}:raise ValueError('missing/duplicate/invalid wave identity')
 lookup={(r['scenario'],r['lanes'],r['lane'],r['step']):r for r in fixture['requests']};client=bench.Client(types.SimpleNamespace(engine='tensorsharp',url=header['url'],model=MODEL_ID,extra_body=None,metrics_url='',tokenizer=None,server_log=None,timeout=120,temperature=0,top_p=1,sampling_seed=1234));details=[]
 for key,r in zip(ids,requests):
  prepared=lookup[key];common=prepared['common'];body=client.body(common['messages'],common['max_tokens'],{k:v for k,v in common.items()if k not in('messages','max_tokens')});wire=json.dumps(body).encode();wire_sha=hashlib.sha256(wire).hexdigest();transport=r['raw_transport'];response=r['response'];q=r['quality']
  if r['expected']!=prepared['expected']or r['common_sha256']!=prepared['common_sha256']or r['request_body']!=body or r['full_body_sha256']!=cua.digest(body)or r['canonical_input_sha256']!=cua.digest({k:v for k,v in body.items()if k!='model'})or r['frozen_normalized_request_sha256']!=cua.digest(bench.request_identity(body))or r['wire_request_body_sha256']!=wire_sha or transport.get('request_body_sha256')!=wire_sha or transport.get('request_body_bytes')!=len(wire):raise ValueError('prepared/full/canonical/actual wire input identity differs')
  payload=base64.b64decode(transport['body_base64'],validate=True)
  if hashlib.sha256(payload).hexdigest()!=transport['body_sha256']or len(payload)!=transport['bytes']or response_hash(response)!=response['response_sha256']or hashlib.sha256((response.get('content')or'').encode()).hexdigest()!=response['content_sha256']:raise ValueError('complete raw response identity differs')
  if set(q['checks'])!=QUALITY or set(q['measurement_checks'])!=MEASUREMENT or q!=cua.validate(bench,response,transport,r['expected'],body['max_tokens'],body['tools'][0]['function']['parameters']):raise ValueError('original seven quality/three measurement checks changed')
  if response.get('decode_tok_s')is not None or response.get('raw_usage')!=r['raw_observation'].get('usage'):raise ValueError('CUA token/timing semantics changed')
  details.append(dict(id=list(key),expected=r['expected'],checks=q['checks'],measurement_checks=q['measurement_checks'],quality=q['verdict'],measurement=q['measurement_verdict'],common_sha256=r['common_sha256'],canonical_input_sha256=r['canonical_input_sha256'],full_body_sha256=r['full_body_sha256'],wire_request_body_sha256=wire_sha,response_sha256=response['response_sha256'],raw_response_sha256=transport['body_sha256'],actual_usage=response.get('raw_usage'),cached_tokens=response.get('cached_tokens'),ttft_s=response.get('ttft_s'),total_s=response.get('total_s')))
 if len(summaries)!=len(job['scenarios'])or len({(s.get('scenario'),s.get('lanes'))for s in summaries})!=len(summaries):raise ValueError('case summaries incomplete or duplicated')
 for scenario,summary in zip(job['scenarios'],summaries):
  original=cua.summarise([r for r in requests if r['scenario']==scenario],[w for w in waves if w['scenario']==scenario],2*job['steps']);original.update(kind='case_summary',scenario=scenario,lanes=2)
  if summary!=original:raise ValueError('original summary/withheld latency semantics changed')
 qualified=all(s['latency_qualified']for s in summaries)
 if end.get('complete')is not True or end.get('qualified')is not qualified or end.get('summaries')!=summaries:raise ValueError('run end does not match complete original cases')
 return dict(path=str(path),sha256=sha(path),execution_complete=True,original_quality_and_measurement_qualified=qualified,expected_requests=len(expected),observed_requests=len(requests),original_quality_pass=sum(r['quality']['verdict']=='PASS'for r in requests),original_measurement_pass=sum(r['quality']['measurement_verdict']=='PASS'for r in requests),case_summaries=summaries,all_original_checks_preserved=True,all_fixture_and_request_identities_verified=True,actual_usage_all_requests=details,decode_tok_s=None,CI='NONE;30 prescribed sequential waves are correlated, not independent runs')


def pair_cua(left,right):
 def read(path):return {tuple(r[k]for k in('scenario','lanes','lane','step')):r for r in map(json.loads,Path(path).read_text().splitlines())if r.get('kind')=='request'}
 a,b=read(left),read(right)
 if set(a)!=set(b):raise ValueError('cannot pair missing/different request identities')
 observations=[]
 for key in sorted(a):
  x,y=a[key],b[key];old={**x['quality']['checks'],**x['quality']['measurement_checks']};new={**y['quality']['checks'],**y['quality']['measurement_checks']};identities={k:x[k]==y[k]for k in('common_sha256','canonical_input_sha256','full_body_sha256','wire_request_body_sha256','expected')}
  observations.append(dict(id=list(key),input_matches=identities,all_inputs_match=all(identities.values()),original_checks_true_to_false=[k for k,v in old.items()if v is True and new.get(k)is False],original_checks_false_to_true=[k for k,v in old.items()if v is False and new.get(k)is True],all_check_transitions={k:[v,new.get(k)]for k,v in old.items()},response_sha256=[x['response']['response_sha256'],y['response']['response_sha256']],raw_response_sha256=[x['raw_transport']['body_sha256'],y['raw_transport']['body_sha256']],parsed=[[(c.get('name'),c.get('parsed'))for c in r['response'].get('tool_calls',[])]for r in(x,y)]))
 return dict(quality_equivalence_claim=False,request_pairs=len(observations),all_input_identities_match=all(x['all_inputs_match']for x in observations),original_check_true_to_false=sum(len(x['original_checks_true_to_false'])for x in observations),observations=observations,limits='No rescoring or netting improvements against regressions. Prescribed common history; exact input/wire identities retained. One fresh replay per profile, not independent repeated experiments.')


