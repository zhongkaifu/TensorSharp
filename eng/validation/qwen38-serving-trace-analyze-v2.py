#!/usr/bin/env python3
"""Offline serving metadata joins with schema2 dedicated-file closure. Never changes original CUA scores or infers causality."""
import argparse,base64,collections,hashlib,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREFIX='[q35-serving-trace] '
SCHEMA='tensorsharp-q35-serving-trace/1'
FILE_SCHEMA='tensorsharp-q35-serving-trace/2'
QCHECKS={'http_success','no_freeform_answer','no_hidden_thinking','one_correct_tool','single_choice','stream_complete','typed_exact_arguments'}
MCHECKS={'actual_server_usage','finite_client_timings','ttft_within_request'}
HASHES=('common_sha256','canonical_input_sha256','full_body_sha256','wire_request_body_sha256','frozen_normalized_request_sha256')
class Pairs(list):pass

def require(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def strict(pairs):
 d={}
 for k,v in pairs:
  require(k not in d,'duplicate JSON key: '+k);d[k]=v
 return d
def parse(s,pairs=False):return json.loads(s,object_pairs_hook=Pairs if pairs else strict,parse_constant=lambda x:(_ for _ in()).throw(ValueError('nonfinite JSON '+x)))
def vals(obj,k):return [v for key,v in obj if key==k]if isinstance(obj,Pairs)else []
def one(obj,k):
 v=vals(obj,k);return v[0]if len(v)==1 else None
def loc(path,line,pointer=''):return dict(path=str(path),line=line,json_pointer=pointer)
def lines(path,issues,kind):
 for n,b in enumerate(path.read_bytes().splitlines(),1):
  try:yield n,b.decode('utf-8')
  except UnicodeError:issues.append(dict(kind='invalid_UTF8_'+kind,location=loc(path,n)))
def identity(r):
 require(isinstance(r.get('scenario'),str)and all(type(r.get(k))is int for k in('lanes','lane','step')),'invalid original request identity')
 return (r['scenario'],r['lanes'],r['lane'],r['step'])
def kid(k):return '/'.join(map(str,k))
def file_input(x):
 require(isinstance(x,dict)and set(x)>={'path','producer_path'},'log input needs explicit local path and producer_path')
 p=Path(x['path']);require(p.is_file()and not p.is_symlink(),'regular local input required: '+str(p));return p,x['producer_path']
def normalized_identity(body):
 b=json.loads(json.dumps(body));b.pop('model',None);ids={}
 for m in b.get('messages',[]):
  for c in m.get('tool_calls')or []:
   if c.get('id')is not None:ids.setdefault(c['id'],f'tool_{len(ids)}');c['id']=ids[c['id']]
  if m.get('tool_call_id')is not None:ids.setdefault(m['tool_call_id'],f'tool_{len(ids)}');m['tool_call_id']=ids[m['tool_call_id']]
 return b

def replay(path,attempt,issues):
 rows=[];heads=[];ends=[]
 for n,line in lines(path,issues,'replay'):
  try:r=parse(line);require(isinstance(r,dict),'replay JSON object required')
  except (ValueError,TypeError)as e:issues.append(dict(kind='corrupt_replay',location=loc(path,n),error=str(e)));continue
  if r.get('kind')=='run':heads.append(r)
  elif r.get('kind')=='run_end':ends.append(r)
  elif r.get('kind')=='request':
   try:
    k=identity(r);q=r.get('quality',{});require(set(q.get('checks',{}))==QCHECKS and set(q.get('measurement_checks',{}))==MCHECKS,'original7+3 missing/extra');require(all(type(x)is bool for x in [*q['checks'].values(),*q['measurement_checks'].values()]),'original checks not typed bool')
    headers=r.get('raw_transport',{}).get('headers',[]);ids=[v for key,v in headers if isinstance(key,str)and key.lower()=='x-tensorsharp-request-id'];http=ids[0]if len(ids)==1 and isinstance(ids[0],str)and ids[0]else None
    body=r.get('request_body');require(isinstance(body,dict),'original body unavailable');checks=dict(full_body=digest(body)==r.get('full_body_sha256'),canonical_input=digest({k:v for k,v in body.items()if k!='model'})==r.get('canonical_input_sha256'),normalized=digest(normalized_identity(body))==r.get('frozen_normalized_request_sha256'),reported_wire_sha=r.get('wire_request_body_sha256')==r.get('raw_transport',{}).get('request_body_sha256'))
    require(all(checks.values()),'original input identity inconsistent');wire=r['raw_transport'];payload=base64.b64decode(wire['body_base64'],validate=True);require(len(payload)==wire.get('bytes')and hashlib.sha256(payload).hexdigest()==wire.get('body_sha256'),'original response wire bytes differ')
    row=dict(id=kid(k),key=list(k),replay_location=loc(path,n),http_request_id=http,input_hashes={x:r.get(x)for x in HASHES},input_identity_checks=checks,quality=q,expected=r.get('expected',{}),response=r.get('response',{}),paths=[],issues=[],engine_sequence_id=None)
    row['_body']=body
    if http is None:row['issues'].append('missing_or_duplicate_HTTP_response_header')
    if any(not isinstance(r.get(x),str)or not re.fullmatch('[a-f0-9]{64}',r[x])for x in HASHES):row['issues'].append('missing_original_hash')
    rows.append(row)
   except (ValueError,TypeError,KeyError)as e:issues.append(dict(kind='unavailable_request',location=loc(path,n),error=str(e)))
 if len(heads)!=1 or len(ends)!=1 or ends[0].get('complete')is not True:issues.append(dict(kind='missing_or_incomplete_replay_run_markers'))
 if len(heads)==1:
  h=heads[0]
  if h.get('label')!=attempt.get('label')or h.get('launch_manifest_sha256')!=attempt.get('launch_manifest',{}).get('sha256'):issues.append(dict(kind='campaign_header_identity_differs'))
 expected=attempt.get('job',{}).get('expected_requests')
 if type(expected)is not int or len(rows)!=expected:issues.append(dict(kind='request_count_differs',expected=expected,observed=len(rows)))
 if len({r['id']for r in rows})!=len(rows):issues.append(dict(kind='duplicate_request_identity'))
 return rows

def structured(inputs,http_ids,issues):
 mappings=collections.defaultdict(list);duplicates=[]
 for x in inputs:
  p,_=file_input(x)
  for n,line in lines(p,issues,'structured'):
   if not line.strip():continue
   try:
    r=parse(line,True);require(isinstance(r,Pairs),'structured JSON object required')
    scope=one(r,'scope');props=one(r,'props');scope_ids=vals(scope,'RequestId')
    if len(scope_ids)>1:duplicates.append(dict(location=loc(p,n,'/scope/RequestId'),all_values=scope_ids))
    if one(r,'template')!='q35.serving.request engine_sequence_id={EngineSequenceId}':continue
    sequence=one(props,'EngineSequenceId');require(isinstance(sequence,str)and sequence,'mapping EngineSequenceId missing/duplicate')
    matched=[v for v in scope_ids if isinstance(v,str)and v in http_ids]
    if len(matched)!=1:issues.append(dict(kind='mapping_HTTP_scope_missing_or_ambiguous',location=loc(p,n),scope_values=scope_ids));continue
    mappings[matched[0]].append(dict(sequence=sequence,location=loc(p,n),scope_RequestId_candidates=scope_ids))
   except (ValueError,TypeError)as e:issues.append(dict(kind='corrupt_structured_JSON',location=loc(p,n),error=str(e)))
 return mappings,duplicates

def trace(inputs,issues):
 events=[];last={};ids=set()
 for x in inputs:
  p,_=file_input(x)
  for n,line in lines(p,issues,'stderr'):
   if PREFIX not in line:continue
   if not line.startswith(PREFIX):issues.append(dict(kind='interleaved_trace_prefix',location=loc(p,n)));continue
   try:
    r=parse(line[len(PREFIX):]);require(isinstance(r,dict)and r.get('schema')==SCHEMA,'trace schema differs');engine=r.get('engine_id');index=r.get('event_index');dropped=r.get('previously_dropped_events');require(isinstance(engine,str)and engine and type(index)is int and index>0 and type(dropped)is int and dropped>=0 and isinstance(r.get('utc'),str)and isinstance(r.get('metadata'),dict),'trace identity/types missing');require(r.get('kind')in('sequence_slot_binding','run_groups','verify_commit_completed','head_host_return_completed'),'unknown trace event')
    require((engine,index)not in ids,'duplicate engine/event_index');ids.add((engine,index))
    if index!=last.get(engine,0)+1:issues.append(dict(kind='missing_or_out_of_order_trace_index',location=loc(p,n),engine_id=engine,previous=last.get(engine,0),actual=index))
    last[engine]=index
    if dropped:issues.append(dict(kind='trace_dropped_events_observed',location=loc(p,n),count=dropped))
    events.append(dict(event=r,location=loc(p,n)))
   except (ValueError,TypeError)as e:issues.append(dict(kind='corrupt_trace_JSON',location=loc(p,n),error=str(e)))
 if not events:issues.append(dict(kind='no_trace_records'))
 return events

def file_trace(inputs,issues):
 """Plain JSONL only. Never repairs lines or substitutes stderr for a configured file."""
 events=[];statuses=[]
 for entry in inputs:
  p,producer=file_input(entry);start=len(issues);records=[];engine=None
  for n,line in lines(p,issues,'trace_file'):
   try:
    require(bool(line.strip()),'blank line in dedicated trace file')
    r=parse(line);require(isinstance(r,dict)and r.get('schema')==FILE_SCHEMA,'dedicated trace schema differs')
    require(isinstance(r.get('engine_id'),str)and r['engine_id']and type(r.get('event_index'))is int and r['event_index']>0 and type(r.get('previously_dropped_events'))is int and r['previously_dropped_events']>=0 and isinstance(r.get('utc'),str)and isinstance(r.get('metadata'),dict),'trace identity/types missing')
    require(r.get('kind')in('trace_opened','trace_closed','sequence_slot_binding','run_groups','verify_commit_completed','head_host_return_completed'),'unknown trace event')
    if engine is None:engine=r['engine_id']
    require(r['engine_id']==engine,'multiple engines in one dedicated trace file')
    require(r['event_index']==len(records)+1,'missing/duplicate/out-of-order dedicated trace index')
    if r['previously_dropped_events']:issues.append(dict(kind='trace_dropped_events_observed',location=loc(p,n),count=r['previously_dropped_events']))
    item=dict(event=r,location=loc(p,n));records.append(item);events.append(item)
   except (ValueError,TypeError)as ex:issues.append(dict(kind='corrupt_or_incomplete_dedicated_trace',location=loc(p,n),error=str(ex)))
  try:
   require(len(records)>=2,'dedicated trace opened/closed markers missing')
   first,last=records[0]['event'],records[-1]['event']
   require(first['kind']=='trace_opened'and last['kind']=='trace_closed','dedicated trace first/terminal markers missing')
   require(sum(x['event']['kind']=='trace_opened'for x in records)==1 and sum(x['event']['kind']=='trace_closed'for x in records)==1,'duplicate dedicated trace markers')
   opened=first['metadata'];closed=last['metadata']
   require(set(opened)=={'sink','file_path','terminal_marker_required'}and opened['sink']=='dedicated_file'and opened['file_path']==producer and opened['terminal_marker_required']is True,'opening marker does not bind actual configured file')
   require(set(closed)=={'reason','last_event_index','last_forward_id','last_successful_forward_id','active_forward_id','dropped_events'},'terminal metadata fields differ')
   require(all(type(closed[k])is int and closed[k]>=0 for k in('last_event_index','last_forward_id','last_successful_forward_id','dropped_events')),'terminal counters invalid')
   require(closed['last_event_index']==last['event_index']-1,'terminal previous index differs')
   require(closed['active_forward_id']is None,'forward still active at trace close')
   require(closed['reason']in('engine_dispose_finally','trace_disposed'),'abnormal/constructor-failed trace close')
   require(closed['dropped_events']==last['previously_dropped_events']==0,'terminal reports dropped diagnostic writes')
   forwards=[x['event']['metadata']for x in records if x['event']['kind']=='run_groups']
   ids=[x.get('forward_id')for x in forwards]
   require(all(type(x)is int for x in ids)and len(ids)==closed['last_forward_id']and all(x==i for i,x in enumerate(ids,1)),'terminal forward sequence missing or inconsistent')
   expected_success=forwards[-1]['forward_id']if forwards and forwards[-1].get('outcome')=='host_submission_returned'else 0
   require(closed['last_successful_forward_id']==expected_success,'terminal successful-forward marker differs')
  except (ValueError,TypeError,KeyError)as ex:issues.append(dict(kind='dedicated_trace_not_closed_cleanly',path=str(p),error=str(ex)))
  statuses.append(dict(path=str(p),producer_path=producer,engine_id=engine,records=len(records),closed_cleanly=len(issues)==start,scope='host_metadata_file_only_NOT_complete_GPU_or_per_kernel_capture'))
 if not events:issues.append(dict(kind='no_trace_records'))
 return events,statuses

def join(rows,events,mappings,issues):
 by_http=collections.defaultdict(list);by_seq=collections.defaultdict(list)
 for r in rows:
  if r['http_request_id']:by_http[r['http_request_id']].append(r)
 for http,rs in by_http.items():
  if len(rs)!=1:
   for r in rs:r['issues'].append('HTTP_header_ID_reused_by_multiple_replay_requests')
   continue
  r=rs[0];ms=mappings.get(http,[])
  if len(ms)!=1:r['issues'].append('missing_or_multiple_HTTP_to_engine_mappings');continue
  seq=ms[0]['sequence'];r['engine_sequence_id']=seq;r['paths'].append(dict(kind='HTTP_to_engine_mapping',**ms[0]));by_seq[seq].append(r)
 for seq,rs in by_seq.items():
  if len(rs)!=1:
   for r in rs:r['issues'].append('engine_sequence_ID_maps_multiple_HTTP_requests')
 bindings={};forwards={};unmapped=[]
 def targets(seq,truncated,where):
  if truncated is not False or not isinstance(seq,str):
   unmapped.append(dict(location=where,engine_sequence_id=seq,truncated=truncated))
   if seq is not None:issues.append(dict(kind='truncated_or_unknown_trace_identifier',location=where))
   return []
  rs=by_seq.get(seq,[])
  if len(rs)!=1:unmapped.append(dict(location=where,engine_sequence_id=seq,truncated=False));return []
  return rs
 def slot_key(engine,slot):
  require(type(slot.get('slot_id'))is int and slot['slot_id']>0 and type(slot.get('binding_generation'))is int and slot['binding_generation']>=0,'invalid slot identity')
  return(engine,slot['slot_id'],slot['binding_generation'])
 for item in events:
  e=item['event'];m=e['metadata'];where=item['location'];engine=e['engine_id'];kind=e['kind']
  try:
   if kind in('trace_opened','trace_closed'):continue
   if kind=='sequence_slot_binding':
    key=slot_key(engine,m);require(key[2]>0 and all(type(m.get(k))is int for k in('npast','rope_delta','block_table_version'))and type(m.get('request_id_truncated'))is bool and m.get('binding_status')in('mapped','primary_unmapped'),'binding metadata incomplete');require(key not in bindings,'duplicate binding generation');bindings[key]=m
    for r in targets(m.get('engine_sequence_id'),m.get('request_id_truncated'),where):r['paths'].append(dict(kind='slot_binding',engine_id=engine,event_index=e['event_index'],location=where,metadata=m))
   elif kind=='run_groups':
    fid=m.get('forward_id');require(type(fid)is int and fid>0 and(engine,fid)not in forwards,'invalid/duplicate forward ID');before=m.get('groups_before');after=m.get('groups_after');require(isinstance(before,list)and isinstance(after,list)and len(before)==len(after),'group snapshots unavailable');require(type(m.get('total_rows'))is int and m['total_rows']>0 and type(m.get('layer_branch_observed'))is bool and type(m.get('verify_graph_replay_delta'))is int and m.get('outcome')in('host_submission_returned','failed')and m.get('device_completion')=='NOT_QUERIED_BY_TRACE','forward geometry/branch types');require(isinstance(m.get('layer_branch'),str)and(m['layer_branch_observed']or m['layer_branch']=='NOT_OBSERVED_ON_HOST'),'branch unavailable');members=[]
    for gi,g in enumerate(before):
     require(g.get('kind')in('prefill','decode','verify')and all(type(g.get(k))is int for k in('rows_per_sequence','padded_rows','real_rows'))and isinstance(g.get('slots'),list),'group metadata invalid')
     require(g.get('group_ordinal')==gi and g.get('sequences')==len(g['slots'])and g['padded_rows']==g['rows_per_sequence']*len(g['slots'])and g['real_rows']==sum(z.get('real_rows',-1)for z in g['slots']),'group geometry inconsistent');ag=after[gi];require(isinstance(ag.get('slots'),list)and len(g['slots'])==len(ag['slots']),'slot snapshots differ')
     for si,slot in enumerate(g['slots']):
      key=slot_key(engine,slot);require(type(slot.get('real_rows'))is int and 0<=slot['real_rows']<=g['rows_per_sequence'],'slot real rows unavailable');end=ag['slots'][si];require(slot_key(engine,end)==key,'slot identity changed within forward');require(slot.get('engine_sequence_id')==end.get('engine_sequence_id')and slot.get('request_id_truncated')is end.get('request_id_truncated'),'slot owner changed within forward');require(all(type(slot.get(k))is int and type(end.get(k))is int for k in('npast','rope_delta','block_table_version')),'slot positions unavailable')
      for r in targets(slot.get('engine_sequence_id'),slot.get('request_id_truncated'),where):
       bound=bindings.get(key)
       if bound is None or bound.get('request_id_truncated')is not False or bound.get('engine_sequence_id')!=r['engine_sequence_id']:r['issues'].append('forward_slot_binding_record_missing_or_differs')
       path=dict(kind='run_groups',engine_id=engine,event_index=e['event_index'],location=where,slot_pointer=f'/metadata/groups_before/{gi}/slots/{si}',forward_id=fid,total_rows=m['total_rows'],group_kind=g['kind'],rows_per_sequence=g['rows_per_sequence'],group_padded_rows=g['padded_rows'],group_real_rows=g['real_rows'],slot_before=slot,slot_after=end,layer_branch_observed=m['layer_branch_observed'],layer_branch=m.get('layer_branch'),graph_replay_delta=m.get('verify_graph_replay_delta'),outcome=m.get('outcome'),device_completion=m.get('device_completion'));r['paths'].append(path);members.append(r)
       if m['outcome']=='failed':r['issues'].append('host_forward_failed')
    require(sum(g['padded_rows']for g in before)==m['total_rows'],'total rows differ from original groups');forwards[(engine,fid)]={id(r):r for r in members}.values()
   elif kind=='verify_commit_completed':
    require(type(m.get('window'))is int and m['window']>0 and isinstance(m.get('slots'),list)and m.get('completion')=='existing_sync_returned','commit metadata invalid')
    for si,slot in enumerate(m['slots']):
     key=slot_key(engine,slot);fid=slot.get('verify_forward_id');bound=bindings.get(key);require(type(fid)is int and all(type(slot.get(k))is int for k in('kept','npast_before','npast_after')),'commit fields unavailable')
     if bound is None:unmapped.append(dict(kind='commit_binding_record_missing',location=where,slot_id=slot['slot_id']))
     require(0<=slot['kept']<=m['window']and slot['npast_after']==slot['npast_before']+slot['kept'],'commit positions/kept inconsistent')
     for r in targets(slot.get('engine_sequence_id'),bound.get('request_id_truncated')if bound else None,where):
      if bound.get('engine_sequence_id')!=r['engine_sequence_id']:r['issues'].append('commit_binding_record_differs')
      if fid<=0 or not any(x is r for x in forwards.get((engine,fid),[])):r['issues'].append('commit_verify_forward_missing_or_unmapped')
      r['paths'].append(dict(kind='verify_commit_completed',engine_id=engine,event_index=e['event_index'],location=where,slot_pointer=f'/metadata/slots/{si}',window=m['window'],metadata=slot,completion=m.get('completion')))
   else:
    fid=m.get('forward_id');require(type(fid)is int and all(type(m.get(k))is int for k in('rows','vocab','logits_bytes_copied','argmax_bytes_copied')),'head fields unavailable');require(m['rows']>0 and m['vocab']>0 and m['logits_bytes_copied']in(0,m['rows']*m['vocab']*4)and m['argmax_bytes_copied']in(0,m['rows']*4)and m.get('completion')=='existing_sync_and_requested_host_copies_returned','head completion/bytecounts inconsistent');owners=list(forwards.get((engine,fid),[]))
    if not owners:unmapped.append(dict(kind='head_forward_unmapped',location=where,forward_id=fid))
    for r in owners:r['paths'].append(dict(kind='head_forward_association',engine_id=engine,event_index=e['event_index'],location=where,metadata=m,association_scope='forward_only_NOT_per_request_logit_rows',candidate_requests=len(owners)))
  except (ValueError,TypeError,KeyError)as ex:issues.append(dict(kind='trace_metadata_unusable',location=where,error=str(ex)))
 for r in rows:
  kinds=[x['kind']for x in r['paths']]
  for k in('slot_binding','run_groups','head_forward_association'):
   if k not in kinds:r['issues'].append('missing_'+k)
  verify={(x['engine_id'],x['forward_id'])for x in r['paths']if x.get('group_kind')=='verify'}
  heads={(x['engine_id'],x['metadata']['forward_id'])for x in r['paths']if x['kind']=='head_forward_association'}
  commits={(x['engine_id'],x['metadata']['verify_forward_id'])for x in r['paths']if x['kind']=='verify_commit_completed'}
  r['observed_verify_forwards_missing_head']=[list(x)for x in sorted(verify-heads)]
  r['observed_verify_forwards_missing_commit']=[list(x)for x in sorted(verify-commits)]
  if verify-heads:r['issues'].append('observed_verify_forward_missing_head')
  if verify-commits:r['issues'].append('observed_verify_forward_missing_commit')
  r['minimum_metadata_joins_available']=not r['issues']
 return unmapped

def fields(r):
 response=r['response'];expected=r['expected'];calls=response.get('tool_calls')
 if response.get('error'):return {k:None for k in expected}
 if not isinstance(calls,list)or len(calls)!=1:return {k:False for k in expected}
 try:a=parse(calls[0].get('arguments',''));require(isinstance(a,dict),'arguments not object')
 except (ValueError,TypeError):return {k:False for k in expected}
 return {k:type(a.get(k))is type(v)and a[k]==v for k,v in expected.items()}
def route_sequence(r):return [{k:x.get(k)for k in('total_rows','group_kind','rows_per_sequence','group_padded_rows','group_real_rows','layer_branch_observed','layer_branch','graph_replay_delta','slot_before','slot_after')}for x in r['paths']if x['kind']=='run_groups']
def configuration(profile):
 m=profile['producer_identity'].get('launch_manifest')or {};env=m.get('environment',{});rows=lambda key:sorted((Path(x.get('path','')).name,x.get('sha256'),x.get('size_bytes',x.get('bytes')))for x in m.get(key,[]))
 return dict(flags={k:v for k,v in env.items()if k!='TS_Q35_SERVING_TRACE_FILE'and(k.startswith(('TS_Q35_','TS_TOKEN_SAMPLER_'))or k in('TS_PREFIX_CACHE_ISOLATION','GGML_CUDA_PDL'))},geometry={k:m.get(k)for k in('engine','mode','context_tokens','lanes')},binaries=rows('binaries'),models=rows('models'),GPU_UUIDs=sorted(x.get('uuid')for x in(m.get('gpu',{}).get('devices',[]))if isinstance(x.get('uuid'),str)))
def compare(left,right):
 l={r['id']:r for r in left['requests']};r={r['id']:r for r in right['requests']};pairs=[]
 for key in sorted(set(l)&set(r)):
  a,b=l[key],r[key];match={h:isinstance(a['input_hashes'][h],str)and re.fullmatch('[a-f0-9]{64}',a['input_hashes'][h])is not None and a['input_hashes'][h]==b['input_hashes'][h]for h in HASHES};body_equal=canonical(a['_body'])==canonical(b['_body']);checks=[]
  for group in('checks','measurement_checks'):
   for k in sorted(a['quality'][group]):
    x,y=a['quality'][group][k],b['quality'][group][k]
    if x is True and y is False:checks.append(dict(group=group,field=k,left=True,right=False))
  af,bf=fields(a),fields(b);ft=[dict(field=k,left=True,right=False)for k in af if af[k]is True and bf.get(k)is False]
  ar,br=route_sequence(a),route_sequence(b)
  # Strip only engine-local slot IDs/ownership labels from diagnostic route comparison.
  def geometry(rs):return [{k:v for k,v in x.items()if k not in('slot_before','slot_after')}for x in rs]
  def positions(rs):return [{side:{k:v for k,v in x[side].items()if k in('npast','rope_delta','block_table_version','real_rows')}for side in('slot_before','slot_after')}for x in rs]
  pairs.append(dict(id=key,input_matches=match,full_body_equal=body_equal,all_input_identities_match=body_equal and all(match.values()),original_true_to_false=checks,typed_field_true_to_false=ft,minimum_metadata_joins_both=a['minimum_metadata_joins_available']and b['minimum_metadata_joins_available'],observed_geometry_sequence_equal=(geometry(ar)==geometry(br))if a['minimum_metadata_joins_available']and b['minimum_metadata_joins_available']else None,observed_position_sequence_equal=(positions(ar)==positions(br))if a['minimum_metadata_joins_available']and b['minimum_metadata_joins_available']else None,left_forward_count=len(ar),right_forward_count=len(br),left_issues=a['issues'],right_issues=b['issues'],causal_attribution='NOT_ESTABLISHED'))
 ac,bc=configuration(left),configuration(right);config=dict(actual_GPU_UUIDs_equal=ac['GPU_UUIDs']==bc['GPU_UUIDs']if ac['GPU_UUIDs']and bc['GPU_UUIDs']else None,actual_flags_equal=ac['flags']==bc['flags']and bool(ac['flags']),actual_geometry_equal=ac['geometry']==bc['geometry']and all(v is not None for v in ac['geometry'].values()),binary_inventory_equal=ac['binaries']==bc['binaries']if ac['binaries']and bc['binaries']else None,model_inventory_equal=ac['models']==bc['models']if ac['models']and bc['models']else None)
 return dict(left=left['name'],right=right['name'],configuration_comparison=config,AA_identity_confirmed=all(v is True for v in config.values()),missing_left=sorted(set(r)-set(l)),missing_right=sorted(set(l)-set(r)),pairs=pairs,original_check_TTF=sum(len(x['original_true_to_false'])for x in pairs),typed_field_TTF=sum(len(x['typed_field_true_to_false'])for x in pairs),no_netting=True,no_causal_or_performance_claim=True)

def analyze(campaign,profiles):
 require(isinstance(profiles,dict)and isinstance(profiles.get('profiles'),list)and 0<len(profiles['profiles'])<=8 and isinstance(profiles.get('comparisons',[]),list),'explicit bounded profiles JSON required');attempts=campaign.get('attempts',[]);result=[];names=set()
 for spec in profiles['profiles']:
  name=spec.get('name');require(isinstance(name,str)and name and name not in names,'unique profile names required');names.add(name);matches=[a for a in attempts if a.get('job',{}).get('name')==spec.get('campaign_attempt')];require(len(matches)==1,'one actual campaign attempt required');attempt=matches[0];issues=[];raw=Path(spec['replay']);require(raw.is_file()and not raw.is_symlink(),'regular original replay required');expected=attempt.get('audit',{}).get('sha256')
  require(isinstance(expected,str)and sha(raw)==expected,'original replay does not match actual campaign audit SHA')
  logs=spec.get('structured_logs',[]);stderr=spec.get('stderr_logs',[]);trace_logs=spec.get('trace_logs',[]);require(all(isinstance(x,list)for x in(logs,stderr,trace_logs)),'explicit log lists required')
  actual_file=attempt.get('environment',{}).get('TS_Q35_SERVING_TRACE_FILE');manifest_env=attempt.get('launch_manifest',{}).get('value',{}).get('environment',{})
  if actual_file is not None:
   require(isinstance(actual_file,str)and Path(actual_file).is_absolute()and os.path.normpath(actual_file)==actual_file,'actual trace file must be normalized absolute path')
   require(manifest_env.get('TS_Q35_SERVING_TRACE_FILE')==actual_file and manifest_env.get('TS_Q35_SERVING_TRACE')==attempt.get('environment',{}).get('TS_Q35_SERVING_TRACE')=='1','dedicated trace actual environment and launch manifest differ')
   require(len(trace_logs)==1,'configured dedicated trace requires exactly one explicit trace_logs input')
  else:require(not trace_logs and manifest_env.get('TS_Q35_SERVING_TRACE_FILE')is None,'trace_logs requires matching actual file configuration')
  inputs=[dict(path=str(raw),producer_path=attempt.get('raw'),sha256=sha(raw),bytes=raw.stat().st_size)]
  for kind,entries in(('structured',logs),('stderr',stderr),('trace',trace_logs)):
   for entry in entries:
    p,producer=file_input(entry);inputs.append(dict(path=str(p),producer_path=producer,sha256=sha(p),bytes=p.stat().st_size))
    if kind=='stderr':require(producer==attempt.get('launch_manifest',{}).get('value',{}).get('log'),'stderr producer path differs from actual launch')
    elif kind=='trace':require(producer==actual_file,'dedicated trace producer path differs from actual file environment')
    else:require(Path(producer).parent==Path(attempt.get('environment',{}).get('TENSORSHARP_LOG_DIR','__missing__')),'structured producer log outside actual log directory')
  require(len({x['path']for x in inputs})==len(inputs),'duplicate input file paths')
  rows=replay(raw,attempt,issues);maps,duplicates=structured(logs,{r['http_request_id']for r in rows if r['http_request_id']},issues);closure=[]
  if actual_file is not None:events,closure=file_trace(trace_logs,issues)
  else:events=trace(stderr,issues)
  unmapped=join(rows,events,maps,issues)
  if attempt.get('environment',{}).get('TS_Q35_SERVING_TRACE')!='1':issues.append(dict(kind='trace_not_explicitly_enabled_in_actual_environment'))
  for inp in inputs:
   if Path(inp['path']).stat().st_size!=inp['bytes']or sha(inp['path'])!=inp['sha256']:issues.append(dict(kind='input_changed_during_offline_analysis',path=inp['path']))
  if issues:
   for row in rows:row['issues'].append('profile_capture_or_identity_incomplete');row['minimum_metadata_joins_available']=False
  result.append(dict(name=name,summary=dict(observed_requests=len(rows),original_request_quality_pass=sum(all(r['quality']['checks'].values())for r in rows),original_request_measurement_pass=sum(all(r['quality']['measurement_checks'].values())for r in rows),minimum_joinable_requests=sum(r['minimum_metadata_joins_available']for r in rows),trace_records=len(events),unmapped_trace_records=len(unmapped),capture_issues=len(issues)),producer_identity=dict(controller_sha256=campaign.get('controller_sha256'),source=campaign.get('source'),runtime=campaign.get('runtime'),build_campaign=campaign.get('build_campaign'),launch_manifest_sha256=attempt.get('launch_manifest',{}).get('sha256'),launch_manifest=attempt.get('launch_manifest',{}).get('value'),actual_environment=attempt.get('environment')), campaign_attempt=spec['campaign_attempt'],producer_complete=attempt.get('complete'),producer_quality_passed=attempt.get('quality_passed'),inputs=inputs,requests=rows,trace_events=events,issues=issues,structured_duplicate_scope_RequestId=duplicates,unmapped_trace_records=unmapped,minimum_all_requests_joinable=bool(rows)and all(r['minimum_metadata_joins_available']for r in rows),no_corrupt_or_dropped_observed=not issues,trace_capture_complete=('DEDICATED_HOST_METADATA_STREAM_CLOSED_ONLY'if closure and all(x['closed_cleanly']for x in closure)and not issues else 'NOT_PROVEN_DEDICATED_FILE_INCOMPLETE')if actual_file is not None else 'NOT_PROVEN_NO_TERMINAL_TRACE_FLUSH_MARKER',dedicated_trace_file_status=closure,independent_request_wire_reconstruction=False,independent_response_SSE_reparse=False))
 comparisons=[];by={x['name']:x for x in result}
 for pair in profiles.get('comparisons',[]):require(pair.get('left')in by and pair.get('right')in by and pair['left']!=pair['right'],'explicit directed comparison invalid');comparisons.append(compare(by[pair['left']],by[pair['right']]))
 for p in result:
  for row in p['requests']:row.pop('_body',None)
 return dict(schema='qwen38-serving-trace-offline-analysis/2',analysis_completed=True,profiles=result,comparisons=comparisons,original_scores_changed=False,quality_or_performance_pass_NOT_CREATED=True,causal_attribution='NOT_ESTABLISHED',limitations=['TraceON changes CPU timing/batching; separate diagnostic profile, not performance data.','Missing/interleaved/invalid records are not repaired.','Structured scope duplicate RequestId values retained; join uses exact original HTTP header match only.','Head ownership is forward-only; mixed per-request logit rows are not observed.','Legacy stderr has no terminal marker. Schema2 can close the observed host metadata stream only; it never proves complete GPU/kernel/state capture.','No GPU state/logits/first-token hashes or actual Snapshot/Restore operations are observed.','Request wire equality uses frozen-producer reported hashes; canonical full bodies and response wire bytes independently checked.'])
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--profiles-json',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv);out=a.out.absolute();require(out==Path(os.path.normpath(out))and out.resolve()==out and not out.exists()and any(x in out.parents for x in(ROOT/'artifacts',ROOT/'docs/validation')),'fresh nonalias ignored output required');before={str(x):dict(sha256=sha(x),bytes=x.stat().st_size)for x in(a.campaign,a.profiles_json)};c=parse(a.campaign.read_text());profiles=parse(a.profiles_json.read_text());report=analyze(c,profiles);require(before=={str(x):dict(sha256=sha(x),bytes=x.stat().st_size)for x in(a.campaign,a.profiles_json)},'campaign/profiles changed during offline analysis');report['input_manifests']=before;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x')as f:f.write(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 print(json.dumps(dict(analysis_completed=True,profiles=len(report['profiles']),comparisons=len(report['comparisons']),original_scores_changed=False,causal_attribution='NOT_ESTABLISHED')));return 0
if __name__=='__main__':raise SystemExit(main())
