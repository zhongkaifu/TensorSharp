"""Model-free actual token log extraction, identity and raw baseline plans."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('server_token_oracle',Path(__file__).parents[1]/'prepare-server-token-oracle.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture():
    request={'sessionId':'session-a','messages':[{'role':'user','content':'请详细介绍最终幻想7'}],
        'think':False,'temperature':0,'topP':1,'topK':0,'repeatPenalty':1,'seed':17,'maxTokens':3}
    row={'name':'serial-0','request_id':'http-a','session':{'sessionId':'session-a'},'completed':True,'status':200,
        'request':request,'text':'original answer','terminal':{'sessionId':'session-a','done':True,'aborted':False,
            'error':None,'promptTokens':2,'tokenCount':3,'truncated':True}}
    def event(second,template,props=None,name=None):
        return {'ts':f'2026-10-11T00:00:0{second}.0000000+00:00','scope':{'RequestId':'http-a'},
            'template':template,'props':props or {},'event':{'name':name}}
    records=[event(0,'start',{'Method':'POST','Path':'/api/chat'},'HttpRequestStarted'),
        event(1,'generation.validation.input internal={InternalRequestId}',{'InternalRequestId':'internal-a','SessionId':'session-a',
            'Architecture':'qwen4exp','HistoryCompacted':False,'Thinking':False,'MaxNewTokens':3,'Tokens':'11,22'}),
        event(2,'generation.timings internal={InternalRequestId}',{'InternalRequestId':'internal-a','SessionId':'session-a',
            'EngineCompletionAvailable':True,'PromptTokens':2,'EngineOutputTokens':3,'GeneratedTokens':3,'FinishReason':'max_tokens'}),
        event(3,'generation.validation.output internal={InternalRequestId}',{'InternalRequestId':'internal-a','FinishReason':'max_tokens',
            'ComputedTokens':4,'Tokens':'31,32,33'}),
        event(4,'done',{'Method':'POST','Path':'/api/chat','StatusCode':200},'HttpRequestCompleted')]
    return {'serial':[row]},records


def identities():
    bins={name:'a'*64 for name in ('GgmlOps.dll','TensorSharp.Chat.dll','TensorSharp.Runtime.dll',
        'TensorSharp.Models.dll','TensorSharp.AgentHost.dll','TensorSharp.Server.Host.dll')}
    audit={'cleanupComplete':True,'passed':True,'errors':[],'binariesBefore':bins,'binariesAfter':dict(bins),
        'childEnvironment':{'TS_VALIDATION_TOKEN_TRACE':'1'},'owned':{'pid':30,
            'startTicks':621355968000000000+module.instant('2026-10-10T23:59:59Z')//100,'executable':'/app/server.exe'},
        'command':['/app/server.exe','--model','/models/model.gguf','--port','52176'],'loadedNative':'/app/GgmlOps.dll',
        'startedUtc':'2026-10-10T23:59:58Z','finishedUtc':'2026-10-11T00:00:06Z',
        'modelFiles':[{'FullName':'/models/model.gguf','Length':10,'LastWriteTimeUtc':'2026-10-11T00:00:00.1234567Z'}]}
    model={'loaded':'model.gguf','architecture':'qwen4exp','loadedBackend':'ggml_cuda'}
    baseline={'source':{'path':'/llama','revision':'clean'},'deployment':{'files':{'llama-server.exe':{'path':'/llama/server.exe','sha256':'b'*64}}},
        'checkpoint':{'shards':[{'path':'/models/model.gguf','bytes':10,'mtime_ns':module.instant('2026-10-11T00:00:00.1234567Z'),'sha256':'c'*64}]},
        'expected_checkpoint_eos_ids':[99], 'futureCommand':['/llama/server.exe','--host','127.0.0.1','--port','1234','--ctx-size','512',
            '--model','/models/model.gguf','--parallel','1','--spec-type','none'], 'environment':{'clear_prefixes':['TS_'],'set':{}}}
    return audit,model,baseline


def serial_three_fixture():
    original,events=fixture(); audit,model,baseline=identities()
    report={'started_utc':'2026-10-11T00:00:00Z','model':model,
        'args':{'url':'http://127.0.0.1:52176','server_log':'/app/logs/tensorsharp-server-20261011.jsonl'},
        'tool_sha256':'d'*64,'concurrent_report_sha256':'e'*64,
        'server_log':{'path':'/app/logs/tensorsharp-server-20261011.jsonl'},'serial':[]}
    audit.update(serialThreeCheckerSha=report['tool_sha256'],concurrentReportSha=report['concurrent_report_sha256'])
    records=[]
    for index,prompt in enumerate(('请详细介绍最终幻想7','请详细介绍时间简史','请详细介绍最终幻想8')):
        row=copy.deepcopy(original['serial'][0])
        http,session,internal=(f'http-{index}',f'session-{index}',f'internal-{index}')
        row.update(name=f'serial-{index}',request_id=http,session={'sessionId':session})
        row['request'].update(sessionId=session)
        row['request']['messages'][0]['content']=prompt
        row['terminal']['sessionId']=session
        report['serial'].append(row)
        group=copy.deepcopy(events)
        for event in group:
            event['scope']['RequestId']=http
            if 'InternalRequestId' in event['props']: event['props']['InternalRequestId']=internal
            if 'SessionId' in event['props']: event['props']['SessionId']=session
        records.extend(group)
    return audit,model,baseline,report,records


def three_chat_fixture():
    audit,model,baseline,report,serial=serial_three_fixture()
    report.pop('concurrent_report_sha256')
    report.update(concurrent=report.pop('serial'), passed=True, completed=True, cancel_at=12.5,
                  final_queue={'busy':False,'pending_requests':0})
    report['args']['expect_stall']=False
    report['tool_sha256']=hashlib.sha256(module.THREE_CHAT_CHECKER.read_bytes()).hexdigest()
    audit['threeChatCheckerSha']=report['tool_sha256']
    audit['childEnvironment']['TS_BATCHED_FUSED_TRACE']='1'
    report['concurrent'][0].update(completed=False,client_cancelled_at=12.5)
    records=[]
    for index,row in enumerate(report['concurrent']):
        row['name']=f'chat-{index}'
        records.extend(copy.deepcopy(serial[index*5:index*5+2]))
        records.append({'ts':'2026-10-11T00:00:01Z','scope':{'RequestId':row['request_id']},
            'template':'generation.sequence.bound',
            'props':{'InternalRequestId':f'internal-{index}','SessionId':row['session']['sessionId']}})
    def batch(native,ids,position):
        return {'ts':'2026-10-11T00:00:01Z','scope':{},'template':'batch.fused.accepted',
            'props':{'SequenceRequestIdsJson':ids,'PositionsBeforeJson':[position]*len(ids),
                'PositionsAfterJson':[position+1]*len(ids),'ActiveSequenceCount':len(ids),
                'NativeAcceptedExecutionId':native,'ManagedPublishCompleted':True,'NativeGraphCount':1,
                'NativeNodeCount':11,'NativeHostSegmentCount':0,'NativeExecutionKind':0}}
    records.extend([batch(1,['internal-0','internal-1','internal-2'],2),
                    batch(2,['internal-0','internal-1','internal-2'],3)])
    cancelled=copy.deepcopy(serial[2:5])
    cancelled[0]['props']['FinishReason']='cancelled'
    cancelled[1]['props'].update(FinishReason='cancelled',ComputedTokens=5)
    records.extend(cancelled)
    records.extend([batch(3,['internal-1','internal-2'],4),batch(4,['internal-1','internal-2'],5)])
    for index in (1,2):
        report['concurrent'][index]['request']['maxTokens']=5
        report['concurrent'][index]['terminal']['tokenCount']=5
        group=copy.deepcopy(serial[index*5+2:index*5+5])
        # The accepted N3→N2 path supplies four AR steps after the first token.
        records[index*3+1]['props']['MaxNewTokens']=5
        group[0]['props'].update(EngineOutputTokens=5,GeneratedTokens=5)
        group[1]['props'].update(Tokens='31,32,33,34,35',ComputedTokens=6)
        records.extend(group)
    spec=importlib.util.spec_from_file_location('three_chat_fixture_checker',module.THREE_CHAT_CHECKER)
    checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    report['native']=checker.native_proof(report['concurrent'],records)
    assert report['native']['passed'],report['native']
    report['cancel_server_evidence']=[cancelled[0]]
    return audit,model,baseline,report,records


def write_three_chat_run(folder):
    audit,model,base,report,records=three_chat_fixture()
    run=Path(folder)/'run'; (run/'three-chat').mkdir(parents=True)
    log_path=run/'three-chat/server-log.jsonl'
    log_path.write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
    report['server_log']['sha256']=hashlib.sha256(log_path.read_bytes()).hexdigest()
    for path,data in ((run/'launch-audit.json',audit),(run/'initial-model.json',model),
        (run/'three-chat/report.json',report),(Path(folder)/'baseline.json',base)):
        path.write_text(json.dumps(data),encoding='utf-8')
    return run,Path(folder)/'baseline.json'


class ExtractionTests(unittest.TestCase):
    def test_actual_tokens_not_retokenized_visible_text(self):
        report,records=fixture(); row=module.extract_requests(report,records,'serial',[99])[0]
        self.assertEqual([11,22],row['prompt']['tokens']); self.assertEqual([31,32,33],row['output']['tokens'])
        self.assertEqual(module.token_hash([11,22]),row['prompt']['i32le_sha256'])
        self.assertFalse(row['output']['natural_eos']); self.assertEqual('original answer',row['original_visible_text'])
        self.assertEqual('terminal_output_not_forwarded',row['output']['ar_commit_convention']['name'])

    def test_exact_ar_commit_conventions_cover_length_and_natural_eos(self):
        for computed,forwarded in ((4,False),(5,True)):
            for natural in (False,True):
                with self.subTest(computed=computed,natural=natural):
                    report,records=fixture(); records[3]['props']['ComputedTokens']=computed
                    if natural:
                        records[3]['props'].update(Tokens='31,32,99',FinishReason='eos')
                        records[2]['props'].update(FinishReason='eos',GeneratedTokens=2)
                        report['serial'][0]['terminal'].update(tokenCount=2,truncated=False)
                    row=module.extract_requests(report,records,'serial',[99])[0]
                    commit=row['output']['ar_commit_convention']
                    self.assertIs(forwarded,commit['terminal_output_forwarded'])
                    self.assertEqual(int(not forwarded),commit['unforwarded_output_tokens'])
                    self.assertEqual(natural,row['output']['natural_eos'])
                    self.assertEqual('prompt_count + output_count' + ('' if forwarded else ' - 1'),
                                     commit['computed_tokens_formula'])

    def test_ar_commit_counts_have_no_extra_or_missing_token_tolerance(self):
        # P=2,O=3 permits exactly 4 or 5: P+O+1 and P+O-2 fail.
        for computed in (0,1,2,3,6,7,True,None,4.0,'4'):
            report,records=fixture(); records[3]['props']['ComputedTokens']=computed
            with self.subTest(computed=computed),self.assertRaises(ValueError):
                module.extract_requests(report,records,'serial',[99])

    def test_one_output_token_has_two_exact_commit_conventions(self):
        for computed in (2,3):
            report,records=fixture()
            report['serial'][0]['request']['maxTokens']=1
            report['serial'][0]['terminal']['tokenCount']=1
            records[1]['props']['MaxNewTokens']=1
            records[2]['props'].update(EngineOutputTokens=1,GeneratedTokens=1)
            records[3]['props'].update(Tokens='31',ComputedTokens=computed)
            row=module.extract_requests(report,records,'serial',[99])[0]
            self.assertEqual(computed==3,row['output']['ar_commit_convention']['terminal_output_forwarded'])

    def test_natural_eos_keeps_engine_eos_excludes_published_token(self):
        report,records=fixture(); records[3]['props'].update(Tokens='31,32,99',FinishReason='eos')
        records[2]['props'].update(FinishReason='eos',GeneratedTokens=2)
        report['serial'][0]['terminal'].update(tokenCount=2,truncated=False)
        self.assertTrue(module.extract_requests(report,records,'serial',[99])[0]['output']['natural_eos'])

    def test_invalid_csv_and_nonfinite_json_fail_closed(self):
        for value in ('','1,','1,,2','1,NaN','1,1.0','1,-2','1,2147483648','1, 2','01,2','1,True',True,'١,2'):
            with self.subTest(value=value),self.assertRaises(ValueError): module.token_csv(value)
        for raw in ('{"x":NaN}','{"x":Infinity}','{"x":1e999}','{"x":1,"x":2}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError): module.strict_json(raw)

    def test_missing_duplicate_or_extra_child_events_fail(self):
        for mutate in (lambda r:r.pop(1),lambda r:r.insert(2,copy.deepcopy(r[1])),
                       lambda r:r.__setitem__(slice(1,4),[r[3],r[2],r[1]])):
            report,records=fixture(); mutate(records)
            with self.assertRaises(ValueError): module.extract_requests(report,records,'serial',[99])

    def test_unrelated_http_event_is_not_attributed(self):
        report,records=fixture(); other=copy.deepcopy(records[1]); other['scope']['RequestId']='other'; records.insert(1,other)
        self.assertEqual(1,len(module.extract_requests(report,records,'serial',[99])))

    def test_compaction_thinking_id_and_counts_are_strict(self):
        mutations=[(1,'HistoryCompacted',True),(1,'HistoryCompacted',0),(1,'Thinking',True),(1,'SessionId','child'),
            (1,'Architecture','other'),(1,'MaxNewTokens',2),(2,'PromptTokens',3),(2,'EngineOutputTokens',2),
            (2,'EngineCompletionAvailable',False),(2,'GeneratedTokens',2),(3,'ComputedTokens',6),
            (3,'InternalRequestId','other'),(3,'FinishReason','repetition'),(3,'Tokens','99,32,33')]
        for index,key,value in mutations:
            report,records=fixture(); records[index]['props'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): module.extract_requests(report,records,'serial',[99])

    def test_request_media_cancelled_terminal_and_count_fail(self):
        for mutate in (lambda r:r['request']['messages'][0].update(images=['image']),lambda r:r.update(completed=False),
            lambda r:r['terminal'].update(aborted=True),lambda r:r['terminal'].update(promptTokens=4),
            lambda r:r['terminal'].update(tokenCount=4),lambda r:r['terminal'].update(sessionId='other'),
            lambda r:r['request'].update(temperature=.1)):
            report,records=fixture(); mutate(report['serial'][0])
            with self.assertRaises(ValueError): module.extract_requests(report,records,'serial',[99])

    def test_reversed_timestamp_or_non_utc_rejected(self):
        report,records=fixture(); records[3]['ts']=records[0]['ts']
        with self.assertRaises(ValueError): module.extract_requests(report,records,'serial',[99])
        with self.assertRaises(ValueError): module.instant('2026-10-11T00:00:00')


class IdentityTests(unittest.TestCase):
    def test_three_chat_keeps_cancelled_parent_proof_and_selects_completed_sibling(self):
        audit,model,_,report,records=three_chat_fixture()
        bound=module.bind_run_evidence(audit,model,report,records,'concurrent','three-chat')
        proof=bound['three_chat_capture']
        self.assertEqual(3,len(proof['original_requests']))
        self.assertEqual(2,proof['n3_steps']); self.assertEqual(2,proof['n2_sibling_steps'])
        self.assertEqual('cancelled',proof['first_chat_cancellation']['props']['FinishReason'])
        selected=module.extract_capture_requests(report,records,'concurrent',[99],'three-chat','chat-2')
        self.assertEqual(['chat-2'],[row['name'] for row in selected])
        self.assertEqual('请详细介绍最终幻想8',selected[0]['request']['messages'][0]['content'])

    def test_three_chat_cannot_export_cancelled_parent_or_select_legacy_schema(self):
        _,_,_,report,records=three_chat_fixture()
        for name in (None,'chat-0','other'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                module.extract_capture_requests(report,records,'concurrent',[99],'three-chat',name)
        with self.assertRaises(ValueError):
            module.extract_capture_requests(report,records,'concurrent',[99],'concurrency','chat-2')

    def test_three_chat_schema_checker_native_and_cancel_proof_fail_closed(self):
        audit,model,_,report,records=three_chat_fixture()
        mutations=[lambda a,r,ls:r['concurrent'].pop(),
            lambda a,r,ls:r['concurrent'][0].update(name='chat-1'),
            lambda a,r,ls:r['concurrent'][0].update(request_id=r['concurrent'][1]['request_id']),
            lambda a,r,ls:r['concurrent'][0]['session'].update(sessionId='foreign'),
            lambda a,r,ls:r.update(passed=False),lambda a,r,ls:r.update(completed=False),
            lambda a,r,ls:r['args'].update(expect_stall=True),
            lambda a,r,ls:a.update(threeChatCheckerSha='f'*64),
            lambda a,r,ls:r.update(tool_sha256='f'*64),
            lambda a,r,ls:a['childEnvironment'].pop('TS_BATCHED_FUSED_TRACE'),
            lambda a,r,ls:r['native'].update(n3_steps=3),
            lambda a,r,ls:ls[9]['props'].update(NativeGraphCount=2),
            lambda a,r,ls:r.update(cancel_server_evidence=[]),
            lambda a,r,ls:r['concurrent'][0].update(completed=True),
            lambda a,r,ls:r['concurrent'][0].update(client_cancelled_at=None),
            lambda a,r,ls:r['final_queue'].update(pending_requests=1),
            lambda a,r,ls:ls[0]['props'].update(Path='/api/foreign'),
            lambda a,r,ls:ls[1].update(ts='2026-10-11T00:00:07Z'),
            lambda a,r,ls:ls.insert(2,copy.deepcopy(ls[1]))]
        for mutate in mutations:
            a,r,ls=copy.deepcopy((audit,report,records)); mutate(a,r,ls)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                module.bind_run_evidence(a,model,r,ls,'concurrent','three-chat')
        with self.assertRaises(ValueError):
            module.bind_run_evidence(audit,model,report,records,'serial','three-chat')

    def test_three_chat_selection_does_not_hide_bad_unselected_sibling(self):
        for mutate in (lambda r,ls:r['concurrent'][1]['terminal'].update(aborted=True),
            lambda r,ls:r['concurrent'][1]['request']['messages'][0].update(images=['media']),
            lambda r,ls:ls[4]['props'].update(HistoryCompacted=True),
            lambda r,ls:ls[17]['props'].update(Tokens='99,32,33,34,35')):
            _,_,_,report,records=three_chat_fixture(); mutate(report,records)
            with self.assertRaises(ValueError):
                module.extract_capture_requests(report,records,'concurrent',[99],'three-chat','chat-2')

    def test_prepare_three_chat_binds_all_rows_before_one_fresh_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,baseline=write_three_chat_run(tmp); out=Path(tmp)/'out'
            with mock.patch.object(module.subprocess,'check_output',side_effect=['clean\n','']):
                result=module.prepare(run,baseline,'concurrent',out,512,52178,'three-chat','chat-2')
            self.assertEqual(1,result['requests'])
            exported=json.loads((out/'tokens.json').read_text(encoding='utf-8'))
            self.assertEqual('chat-2',exported['request_name'])
            self.assertEqual(3,len(exported['identity']['log_lifetime_binding']['three_chat_capture']['original_requests']))
            self.assertEqual(1,len(list(out.glob('plan-*.json'))))
            plan=json.loads((out/'plan-0.json').read_text(encoding='utf-8'))
            self.assertEqual('free_text',plan['case']); self.assertIsNone(plan['semantic_pass'])

    def test_prepare_three_chat_rejects_duplicate_json_even_in_unselected_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,baseline=write_three_chat_run(tmp); out=Path(tmp)/'out'
            log_path=run/'three-chat/server-log.jsonl'
            raw=log_path.read_text(encoding='utf-8').replace('"RequestId": "http-0"',
                '"RequestId": "asp-net-original", "RequestId": "http-0"',1)
            log_path.write_text(raw,encoding='utf-8')
            report_path=run/'three-chat/report.json'; report=json.loads(report_path.read_text(encoding='utf-8'))
            report['server_log']['sha256']=hashlib.sha256(log_path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Duplicate JSON member'):
                module.prepare(run,baseline,'concurrent',out,512,52178,'three-chat','chat-2')
            self.assertFalse(out.exists())

    def test_serial_three_exact_requests_keep_all_identity_and_lifetime_guards(self):
        audit,model,_,report,records=serial_three_fixture()
        bound=module.bind_run_evidence(audit,model,report,records,'serial','serial-three')
        self.assertEqual(report['args']['url'],bound['http_endpoint'])
        exported=module.extract_requests(report,records,'serial',[99])
        self.assertEqual(3,len(exported))
        self.assertEqual('请详细介绍最终幻想8',exported[2]['request']['messages'][0]['content'])

    def test_serial_three_schema_and_checker_pins_fail_closed(self):
        audit,model,_,report,records=serial_three_fixture()
        mutations=[lambda a,r,ls:r['serial'].pop(),lambda a,r,ls:r.pop('tool_sha256'),
            lambda a,r,ls:a.update(serialThreeCheckerSha='f'*64),
            lambda a,r,ls:a.update(concurrentReportSha='f'*64),
            lambda a,r,ls:r.update(concurrent_report_sha256='not-a-hash'),
            lambda a,r,ls:r.pop('args'),
            lambda a,r,ls:r['args'].update(url='http://127.0.0.1:52177'),
            lambda a,r,ls:r['server_log'].update(path='/other/logs/tensorsharp-server-20261011.jsonl'),
            lambda a,r,ls:ls[-1].update(ts='2026-10-11T00:00:07Z')]
        for mutate in mutations:
            a,r,ls=copy.deepcopy((audit,report,records)); mutate(a,r,ls)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                module.bind_run_evidence(a,model,r,ls,'serial','serial-three')
        with self.assertRaises(ValueError):
            module.bind_run_evidence(audit,model,report,records,'concurrent','serial-three')

    def test_serial_three_does_not_fall_back_to_legacy_fused_log_pin(self):
        audit,model,_,report,records=serial_three_fixture()
        report['fused_evidence']={'server_log':report.pop('server_log')}
        with self.assertRaises(ValueError):
            module.bind_run_evidence(audit,model,report,records,'serial','serial-three')

    def test_log_is_bound_to_actual_deployment_lifetime_and_endpoint(self):
        audit,model,_=identities(); report,records=fixture()
        report.update(started_utc='2026-10-11T00:00:00Z',model=model,
            args={'url':'http://127.0.0.1:52176','server_log':'/app/logs/tensorsharp-server-20261011.jsonl'},
            fused_evidence={'server_log':{'path':'/app/logs/tensorsharp-server-20261011.jsonl'}})
        self.assertEqual(report['args']['url'],module.bind_run_evidence(audit,model,report,records,'serial')['http_endpoint'])
        mutations=[lambda a,r,ls:r['fused_evidence']['server_log'].update(path='/other/logs/tensorsharp-server-20261011.jsonl'),
            lambda a,r,ls:r['args'].update(server_log='/app/logs/other.jsonl'),
            lambda a,r,ls:r['args'].update(url='http://127.0.0.1:52177'),
            lambda a,r,ls:r.update(started_utc='2026-10-10T23:59:58Z'),
            lambda a,r,ls:a.update(finishedUtc='2026-10-11T00:00:02Z'),
            lambda a,r,ls:ls[1].update(ts='2026-10-10T23:59:58Z'),
            lambda a,r,ls:r.update(model={'loaded':'wrong','architecture':'qwen4exp','loadedBackend':'ggml_cuda'})]
        for mutate in mutations:
            a,r,ls=copy.deepcopy((audit,report,records)); mutate(a,r,ls)
            with self.assertRaises(ValueError): module.bind_run_evidence(a,model,r,ls,'serial')

    def test_actual_snapshot_model_and_process_are_required(self):
        audit,model,baseline=identities(); self.assertIn('binaries',module.identity_gates(audit,model,baseline))
        for mutate in (lambda a:a.update(cleanupComplete=False),lambda a:a['binariesAfter'].update({'GgmlOps.dll':'d'*64}),
            lambda a:a['childEnvironment'].clear(),lambda a:a.update(loadedNative='/other/GgmlOps.dll'),
            lambda a:a['owned'].update(pid=True),lambda a:a['modelFiles'][0].update(Length=11),
            lambda a:a['modelFiles'][0].update(LastWriteTimeUtc='2026-10-11T00:00:00Z')):
            bad=copy.deepcopy(audit); mutate(bad)
            with self.assertRaises(ValueError): module.identity_gates(bad,model,baseline)

    def test_failed_quality_suite_is_retained_not_relabelled_pass(self):
        audit,model,baseline=identities(); audit.update(passed=False,errors=['semantic checker failed'])
        value=module.identity_gates(audit,model,baseline)
        self.assertFalse(value['wrapper_passed']); self.assertEqual(audit['errors'],value['wrapper_errors'])

    def test_plan_is_bound_free_text_and_preserves_placement(self):
        report,records=fixture(); row=module.extract_requests(report,records,'serial',[99])[0]
        _,_,base=identities(); base['old_semantic_gate']='obsolete'
        plan=module.make_plan(base,row,{'path':'tokens.json','sha256':'e'*64},0,256,52178)
        self.assertEqual('free_text',plan['case']); self.assertIsNone(plan['semantic_pass'])
        self.assertTrue(plan['record_only']); self.assertNotIn('old_semantic_gate',plan)
        self.assertEqual([11,22],plan['futureRequest']['body']['prompt'])
        self.assertEqual(17,plan['futureRequest']['body']['seed'])
        self.assertEqual('2048',plan['futureCommand'][plan['futureCommand'].index('--ctx-size')+1])
        self.assertEqual(base['checkpoint'],plan['checkpoint'])
        self.assertEqual(row['output']['ar_commit_convention'],plan['reference_output']['ar_commit_convention'])

    def test_prepare_writes_fresh_pinned_export_without_model_launch(self):
        report,records=fixture(); audit,model,base=identities()
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'; (run/'concurrency').mkdir(parents=True)
            log='\n'.join(json.dumps(r) for r in records)+'\n'
            (run/'concurrency/fused-server-log.jsonl').write_text(log,encoding='utf-8')
            report.update(started_utc='2026-10-11T00:00:00Z',model=model,
                args={'url':'http://127.0.0.1:52176','server_log':'/app/logs/tensorsharp-server-20261011.jsonl'})
            report['fused_evidence']={'server_log':{'sha256':hashlib.sha256((run/'concurrency/fused-server-log.jsonl').read_bytes()).hexdigest(),
                'path':'/app/logs/tensorsharp-server-20261011.jsonl'}}
            for path,data in ((run/'launch-audit.json',audit),(run/'initial-model.json',model),
                (run/'concurrency/report.json',report),(Path(tmp)/'baseline.json',base)):
                path.write_text(json.dumps(data),encoding='utf-8')
            out=Path(tmp)/'out'
            with mock.patch.object(module.subprocess,'check_output',side_effect=['clean\n','']):
                result=module.prepare(run,Path(tmp)/'baseline.json','serial',out)
            self.assertEqual(1,result['requests']); self.assertTrue((out/'plan-0.json').is_file())
            with self.assertRaisesRegex(ValueError,'Fresh'): module.prepare(run,Path(tmp)/'baseline.json','serial',out)
            self.assertIsNone(json.loads((out/'tokens.json').read_text(encoding='utf-8'))['semantic_pass'])

    def test_prepare_serial_three_uses_its_actual_log_and_emits_three_plans(self):
        audit,model,base,report,records=serial_three_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'; (run/'serial-three').mkdir(parents=True)
            log='\n'.join(json.dumps(r) for r in records)+'\n'
            log_path=run/'serial-three/server-log.jsonl'
            log_path.write_text(log,encoding='utf-8')
            report['server_log']['sha256']=hashlib.sha256(log_path.read_bytes()).hexdigest()
            for path,data in ((run/'launch-audit.json',audit),(run/'initial-model.json',model),
                (run/'serial-three/report.json',report),(Path(tmp)/'baseline.json',base)):
                path.write_text(json.dumps(data),encoding='utf-8')
            out=Path(tmp)/'out'
            with mock.patch.object(module.subprocess,'check_output',side_effect=['clean\n','']):
                result=module.prepare(run,Path(tmp)/'baseline.json','serial',out,512,52178,'serial-three')
            self.assertEqual(3,result['requests'])
            exported=json.loads((out/'tokens.json').read_text(encoding='utf-8'))
            self.assertEqual('serial-three',exported['case_directory'])
            self.assertEqual(3,len(list(out.glob('plan-*.json'))))
            self.assertEqual(512,json.loads((out/'request-2.json').read_text(encoding='utf-8'))['n_predict'])
            self.assertIsNone(exported['semantic_pass'])

    def test_prepare_serial_three_digest_mutation_is_rejected_before_export(self):
        audit,model,base,report,records=serial_three_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'run'; (run/'serial-three').mkdir(parents=True)
            (run/'serial-three/server-log.jsonl').write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
            report['server_log']['sha256']='f'*64
            for path,data in ((run/'launch-audit.json',audit),(run/'initial-model.json',model),
                (run/'serial-three/report.json',report),(Path(tmp)/'baseline.json',base)):
                path.write_text(json.dumps(data),encoding='utf-8')
            out=Path(tmp)/'out'
            with self.assertRaisesRegex(ValueError,'digest differs'):
                module.prepare(run,Path(tmp)/'baseline.json','serial',out,512,52178,'serial-three')
            self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()
