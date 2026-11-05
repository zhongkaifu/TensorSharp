import base64,copy,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
S=Path(__file__).resolve().parents[1]/'qwen38-serving-trace-analyze-v2.py';spec=importlib.util.spec_from_file_location('serving_trace_analyze',S);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def fixture(root,n='A',wrong=False,branch='native_default_W4A8_Q81'):
 d=root/n;d.mkdir();raw=d/'replay.jsonl';structured=d/'structured.jsonl';stderr=d/'server.log';seq='chat-'+n;http='http-'+n;body={'model':'m','messages':[]};payload=b'data: [DONE]\n\n'
 q={'checks':{k:True for k in A.QCHECKS},'measurement_checks':{k:True for k in A.MCHECKS},'verdict':'PASS','measurement_verdict':'PASS','latency_qualified':True}
 if wrong:q['checks']['typed_exact_arguments']=False;q.update(verdict='FAIL',latency_qualified=False)
 r=dict(kind='request',scenario='append',lanes=2,lane=1,step=5,expected={'code':'1001'},request_body=body,common_sha256=A.digest('fixture'),full_body_sha256=A.digest(body),canonical_input_sha256=A.digest({'messages':[]}),frozen_normalized_request_sha256=A.digest({'messages':[]}),wire_request_body_sha256='b'*64,quality=q,response={'error':None,'tool_calls':[{'name':'submit_observation','arguments':json.dumps({'code':'1661'if wrong else'1001'})}]},raw_transport=dict(headers=[['X-TensorSharp-Request-Id',http]],request_body_sha256='b'*64,body_base64=base64.b64encode(payload).decode(),bytes=len(payload),body_sha256=hashlib.sha256(payload).hexdigest()))
 header=dict(kind='run',label=n,launch_manifest_sha256='a'*64);end=dict(kind='run_end',complete=True)
 raw.write_text('\n'.join(json.dumps(x)for x in(header,r,end))+'\n')
 # The original logger can emit duplicate scope RequestId keys. Preserve both;
 # the exact response-header candidate is the valid join, not last-key wins.
 structured.write_text('{"template":"q35.serving.request engine_sequence_id={EngineSequenceId}","props":{"EngineSequenceId":'+json.dumps(seq)+'},"scope":{"RequestId":"aspnet-long","RequestId":'+json.dumps(http)+'}}\n')
 def slot(np=7):return dict(slot_id=1,binding_generation=1,engine_sequence_id=seq,request_id_truncated=False,binding_status='mapped',real_rows=8,npast=np,rope_delta=0,block_table_version=2)
 group=dict(group_ordinal=0,kind='verify',sequences=1,rows_per_sequence=8,padded_rows=8,real_rows=8,ragged_counts_present=False,slots=[slot()])
 metas=[('sequence_slot_binding',slot()),('run_groups',dict(forward_id=1,total_rows=8,tap_first=0,groups_before=[group],groups_after=[copy.deepcopy(group)],verify_decode_Q81_enabled=False,layer_branch_observed=True,layer_branch=branch,verify_graph_replay_delta=0,outcome='host_submission_returned',failure_type=None,device_completion='NOT_QUERIED_BY_TRACE')),('head_host_return_completed',dict(forward_id=1,rows=8,vocab=100,logits_bytes_copied=3200,argmax_bytes_copied=0,completion='existing_sync_and_requested_host_copies_returned')),('verify_commit_completed',dict(window=8,slots=[dict(slot_id=1,binding_generation=1,engine_sequence_id=seq,verify_forward_id=1,kept=4,npast_before=7,npast_after=11)],completion='existing_sync_returned'))]
 events=[dict(schema=A.SCHEMA,kind=k,engine_id=n,event_index=i,utc='2026-11-03T00:00:00Z',previously_dropped_events=0,metadata=m)for i,(k,m)in enumerate(metas,1)]
 stderr.write_text('normal stderr\n'+''.join(A.PREFIX+json.dumps(e)+'\n'for e in events))
 attempt=dict(job={'name':n,'expected_requests':1},label=n,raw='/remote/'+n+'/replay.jsonl',complete=True,quality_passed=not wrong,audit={'sha256':A.sha(raw)},environment={'TS_Q35_SERVING_TRACE':'1','TENSORSHARP_LOG_DIR':'/remote/'+n+'/logs'},launch_manifest={'sha256':'a'*64,'value':{'log':'/remote/'+n+'/server.log'}})
 entry=dict(name=n,campaign_attempt=n,replay=str(raw),structured_logs=[{'path':str(structured),'producer_path':'/remote/'+n+'/logs/server.jsonl'}],stderr_logs=[{'path':str(stderr),'producer_path':'/remote/'+n+'/server.log'}]);return attempt,entry,events
def dedicated(root,n='A'):
 attempt,entry,events=fixture(root,n)
 path=root/n/'trace.jsonl';producer='/remote/'+n+'/trace.jsonl'
 attempt['environment']['TS_Q35_SERVING_TRACE_FILE']=producer
 attempt['launch_manifest']['value']['environment']={k:attempt['environment'][k]for k in('TS_Q35_SERVING_TRACE','TS_Q35_SERVING_TRACE_FILE')}
 entry['trace_logs']=[dict(path=str(path),producer_path=producer)]
 rows=[dict(schema=A.FILE_SCHEMA,kind='trace_opened',engine_id=n,event_index=1,utc='2026-11-03T00:00:00Z',previously_dropped_events=0,metadata=dict(sink='dedicated_file',file_path=producer,terminal_marker_required=True))]
 for x in events:x['schema']=A.FILE_SCHEMA;x['event_index']+=1;rows.append(x)
 rows.append(dict(schema=A.FILE_SCHEMA,kind='trace_closed',engine_id=n,event_index=len(rows)+1,utc='2026-11-03T00:00:00Z',previously_dropped_events=0,metadata=dict(reason='engine_dispose_finally',last_event_index=len(rows),last_forward_id=1,last_successful_forward_id=1,active_forward_id=None,dropped_events=0)))
 path.write_text(''.join(json.dumps(x)+'\n'for x in rows));return attempt,entry,rows
class Tests(unittest.TestCase):
 def test_real_schema_duplicate_scope_join_and_original_TTF_not_netting(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,pa,_=fixture(p,'A');b,pb,_=fixture(p,'B',True,branch='native_default_W4A4');r=A.analyze({'attempts':[a,b]},dict(profiles=[pa,pb],comparisons=[{'left':'A','right':'B'}]));c=r['comparisons'][0]
   self.assertEqual(c['original_check_TTF'],1);self.assertEqual(c['typed_field_TTF'],1);self.assertTrue(c['pairs'][0]['all_input_identities_match']);self.assertFalse(c['pairs'][0]['observed_geometry_sequence_equal']);self.assertTrue(r['profiles'][0]['minimum_all_requests_joinable']);self.assertEqual(len(r['profiles'][0]['structured_duplicate_scope_RequestId']),1);self.assertEqual(r['profiles'][1]['requests'][0]['quality']['verdict'],'FAIL');self.assertFalse(r['original_scores_changed']);self.assertEqual(r['profiles'][0]['trace_capture_complete'],'NOT_PROVEN_NO_TERMINAL_TRACE_FLUSH_MARKER')
 def test_corrupt_interleave_index_missing_dropped_and_truncated_incomplete(self):
  for mutation in('interleave','broken_json','missing_event','dropped','truncated','missing_last_verify_head'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as t:
    p=Path(t);a,e,events=fixture(p);log=Path(e['stderr_logs'][0]['path'])
    if mutation=='interleave':log.write_text(log.read_text().replace(A.PREFIX,'interleaved '+A.PREFIX,1))
    elif mutation=='broken_json':log.write_text(log.read_text()+A.PREFIX+'{"bad":\n')
    elif mutation=='missing_event':log.write_text(''.join(A.PREFIX+json.dumps(x)+'\n'for x in events[1:]))
    elif mutation=='dropped':events[-1]['previously_dropped_events']=1;log.write_text(''.join(A.PREFIX+json.dumps(x)+'\n'for x in events))
    elif mutation=='missing_last_verify_head':
     extra=copy.deepcopy(events[1]);extra['event_index']=5;extra['metadata']['forward_id']=2;events.append(extra);log.write_text(''.join(A.PREFIX+json.dumps(x)+'\n'for x in events))
    else:
     events[0]['metadata']['request_id_truncated']=True;events[1]['metadata']['groups_before'][0]['slots'][0]['request_id_truncated']=True;log.write_text(''.join(A.PREFIX+json.dumps(x)+'\n'for x in events))
    r=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertFalse(r['profiles'][0]['minimum_all_requests_joinable'])
 def test_missing_mapping_and_ambiguous_HTTP_scope_never_guess(self):
  for content in('', '{"template":"q35.serving.request engine_sequence_id={EngineSequenceId}","props":{"EngineSequenceId":"chat-A"},"scope":{}}\n'):
   with tempfile.TemporaryDirectory()as t:
    p=Path(t);a,e,_=fixture(p);Path(e['structured_logs'][0]['path']).write_text(content);r=A.analyze({'attempts':[a]},dict(profiles=[e]));row=r['profiles'][0]['requests'][0];self.assertIsNone(row['engine_sequence_id']);self.assertFalse(row['minimum_metadata_joins_available'])
 def test_source_bound_replay_and_log_paths_required(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,e,_=fixture(p)
   for bad in('sha','stderr','structured'):
    aa=copy.deepcopy(a);ee=copy.deepcopy(e)
    if bad=='sha':aa['audit']['sha256']='f'*64
    elif bad=='stderr':ee['stderr_logs'][0]['producer_path']='/some/other/log'
    else:ee['structured_logs'][0]['producer_path']='/unrelated/log'
    with self.assertRaises(ValueError):A.analyze({'attempts':[aa]},dict(profiles=[ee]))
 def test_trace_duplicate_JSON_key_and_duplicate_event_rejected_as_incomplete(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,e,_=fixture(p);f=Path(e['stderr_logs'][0]['path']);old=f.read_text();f.write_text(old+old);r=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertFalse(r['profiles'][0]['no_corrupt_or_dropped_observed'])
   f.write_text(old+A.PREFIX+'{"schema":"'+A.SCHEMA+'","schema":"other"}\n');r=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertFalse(r['profiles'][0]['minimum_all_requests_joinable'])
 def test_changed_input_still_reports_descriptive_TTF(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,pa,_=fixture(p,'A');b,pb,_=fixture(p,'B',True);f=Path(pb['replay']);rs=[json.loads(x)for x in f.read_text().splitlines()];r=rs[1];r['request_body']['messages']=[{'role':'user','content':'changed'}];r['full_body_sha256']=A.digest(r['request_body']);r['canonical_input_sha256']=A.digest({'messages':r['request_body']['messages']});r['frozen_normalized_request_sha256']=r['canonical_input_sha256'];f.write_text(''.join(json.dumps(x)+'\n'for x in rs));b['audit']['sha256']=A.sha(f);out=A.analyze({'attempts':[a,b]},dict(profiles=[pa,pb],comparisons=[{'left':'A','right':'B'}]));pair=out['comparisons'][0]['pairs'][0];self.assertFalse(pair['all_input_identities_match']);self.assertEqual(len(pair['original_true_to_false']),1)
 def test_Head_multiple_request_forward_not_claimed_per_request_rows(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,e,events=fixture(p);raw=Path(e['replay']);rs=[json.loads(x)for x in raw.read_text().splitlines()];second=copy.deepcopy(rs[1]);second['lane']=0;second['raw_transport']['headers']=[['X-TensorSharp-Request-Id','http-second']];rs.insert(2,second);raw.write_text(''.join(json.dumps(x)+'\n'for x in rs));a['audit']['sha256']=A.sha(raw);a['job']['expected_requests']=2
   with Path(e['structured_logs'][0]['path']).open('a')as f:f.write(json.dumps({'template':'q35.serving.request engine_sequence_id={EngineSequenceId}','props':{'EngineSequenceId':'chat-second'},'scope':{'RequestId':'http-second'}})+'\n')
   binding=copy.deepcopy(events[0]);binding['metadata'].update(slot_id=2,engine_sequence_id='chat-second');events.insert(1,binding)
   forward=events[2]['metadata'];forward['total_rows']=16
   for stage in('groups_before','groups_after'):
    g=forward[stage][0];slot=copy.deepcopy(g['slots'][0]);slot.update(slot_id=2,engine_sequence_id='chat-second');g['slots'].append(slot);g.update(sequences=2,padded_rows=16,real_rows=16)
   events[3]['metadata'].update(rows=16,logits_bytes_copied=6400);commit=copy.deepcopy(events[4]['metadata']['slots'][0]);commit.update(slot_id=2,engine_sequence_id='chat-second');events[4]['metadata']['slots'].append(commit)
   for i,x in enumerate(events,1):x['event_index']=i
   Path(e['stderr_logs'][0]['path']).write_text(''.join(A.PREFIX+json.dumps(x)+'\n'for x in events));out=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertTrue(out['profiles'][0]['minimum_all_requests_joinable'])
   for row in out['profiles'][0]['requests']:
    head=next(x for x in row['paths']if x['kind']=='head_forward_association');self.assertEqual(head['association_scope'],'forward_only_NOT_per_request_logit_rows');self.assertEqual(head['candidate_requests'],2)
 def test_missing_trace_or_env_not_passed_as_no_difference(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,e,_=fixture(p);a['environment']['TS_Q35_SERVING_TRACE']='0';Path(e['stderr_logs'][0]['path']).write_text('no trace\n');out=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertFalse(out['profiles'][0]['minimum_all_requests_joinable']);self.assertFalse(out['profiles'][0]['no_corrupt_or_dropped_observed'])
 def test_dedicated_file_markers_join_and_do_not_consume_interleaved_stderr(self):
  with tempfile.TemporaryDirectory()as t:
   a,e,_=dedicated(Path(t));Path(e['stderr_logs'][0]['path']).write_text('not JSON '+A.PREFIX+'corrupt legacy log\n');r=A.analyze({'attempts':[a]},dict(profiles=[e]));p=r['profiles'][0]
   self.assertTrue(p['minimum_all_requests_joinable']);self.assertTrue(p['dedicated_trace_file_status'][0]['closed_cleanly']);self.assertEqual(p['trace_capture_complete'],'DEDICATED_HOST_METADATA_STREAM_CLOSED_ONLY');self.assertFalse(r['original_scores_changed'])
 def test_dedicated_file_requires_exact_actual_manifest_and_environment_path(self):
  for mutation in('foreign','missing','manifest','environment','unset'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as t:
    a,e,_=dedicated(Path(t))
    if mutation=='foreign':e['trace_logs'][0]['producer_path']='/different/trace.jsonl'
    elif mutation=='missing':e['trace_logs']=[]
    elif mutation=='manifest':a['launch_manifest']['value']['environment']['TS_Q35_SERVING_TRACE_FILE']='/different/trace.jsonl'
    elif mutation=='environment':a['environment']['TS_Q35_SERVING_TRACE']='0'
    else:a['environment'].pop('TS_Q35_SERVING_TRACE_FILE')
    with self.assertRaises(ValueError):A.analyze({'attempts':[a]},dict(profiles=[e]))
 def test_dedicated_closure_missing_corrupt_dropped_or_active_is_not_complete(self):
  for mutation in('no_open','no_close','gap','dropped','active','ctor_failure','bad_terminal','extra_engine','reordered','interleave','after_close','forward_missing'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as t:
    a,e,rows=dedicated(Path(t));p=Path(e['trace_logs'][0]['path'])
    if mutation=='no_open':rows=rows[1:]
    elif mutation=='no_close':rows=rows[:-1]
    elif mutation=='gap':rows.pop(2)
    elif mutation=='dropped':rows[-1]['previously_dropped_events']=rows[-1]['metadata']['dropped_events']=1
    elif mutation=='active':rows[-1]['metadata']['active_forward_id']=2
    elif mutation=='ctor_failure':rows[-1]['metadata']['reason']='constructor_failed'
    elif mutation=='bad_terminal':rows[-1]['metadata']['last_event_index']=20
    elif mutation=='extra_engine':rows[2]['engine_id']='other'
    elif mutation=='reordered':rows[2],rows[3]=rows[3],rows[2]
    elif mutation=='after_close':x=copy.deepcopy(rows[1]);x['event_index']=len(rows)+1;rows.append(x)
    elif mutation=='forward_missing':rows[-1]['metadata']['last_forward_id']=2
    p.write_text(''.join(json.dumps(x)+'\n'for x in rows))
    if mutation=='interleave':p.write_text('foreign text'+p.read_text())
    r=A.analyze({'attempts':[a]},dict(profiles=[e]));self.assertFalse(r['profiles'][0]['minimum_all_requests_joinable']);self.assertFalse(r['profiles'][0]['dedicated_trace_file_status'][0]['closed_cleanly']);self.assertNotEqual(r['profiles'][0]['trace_capture_complete'],'DEDICATED_HOST_METADATA_STREAM_CLOSED_ONLY')
 def test_file_paths_are_recorded_but_do_not_make_otherwise_same_flags_different(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);a,pa,_=dedicated(p,'A');b,pb,_=dedicated(p,'B');r=A.analyze({'attempts':[a,b]},dict(profiles=[pa,pb],comparisons=[dict(left='A',right='B')]))
   self.assertTrue(r['comparisons'][0]['configuration_comparison']['actual_flags_equal']);self.assertNotEqual(r['profiles'][0]['inputs'][-1]['producer_path'],r['profiles'][1]['inputs'][-1]['producer_path'])
if __name__=='__main__':unittest.main()
