"""No model or GPU: complete-distribution conditioning, arithmetic and ownership."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('prefix_logits',Path(__file__).parents[1]/'qwen38-prefix-logits.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture():
    body=dict(prompt=[0,1],n_predict=2,temperature=0.,samplers=['temperature'],top_k=0,top_p=1.,min_p=0.,
        repeat_penalty=1.,repeat_last_n=0,presence_penalty=0.,frequency_penalty=0.,dry_multiplier=0.,
        cache_prompt=False,return_tokens=True,n_probs=3,post_sampling_probs=False,stream=False,ignore_eos=False,stop=[])
    llama=dict(futureCommand=['server','--host','127.0.0.1','--port','52178','--parallel','1','--ctx-size','32','--spec-type','none'],
        futureRequest=dict(endpoint='http://127.0.0.1:52178/completion',body=body),
        prompt=dict(tokens=[0,1],count=2),expected_checkpoint_eos_ids=[99],
        source=dict(path='missing',revision='abc'),deployment=dict(files={}),checkpoint={})
    return dict(llama=llama,tensorsharp=dict(cache_budgets_mb=[0,8192]),
        diagnostic=dict(vocabulary=3,first_token=2,candidate_tokens=[0,1],context=32,
                        expected_backend_sampling=False,baseline_llama_tokens=[2,1]))


def response():
    values=[[-3.,-2.,-1.],[-2.,-1.,-3.]]
    settings=copy.deepcopy(fixture()['llama']['futureRequest']['body'])
    settings.update(backend_sampling=False,grammar='',logit_bias=[],**{'speculative.types':'none,none'})
    return dict(tokens=[2,1],tokens_evaluated=2,tokens_predicted=2,stop_type='limit',truncated=False,
        generation_settings=settings,
        completion_probabilities=[dict(id=token,top_logprobs=[dict(id=i,logprob=value) for i,value in enumerate(row)])
                                  for token,row in zip([2,1],values)])


class PlanTests(unittest.TestCase):
    def test_raw_quality_runner_stays_strict_and_diagnostic_is_separate(self):
        p=fixture();before=copy.deepcopy(p)
        self.assertEqual('/completion',tool.validate_plan(p).path)
        self.assertEqual(before,p)
        with self.assertRaises(ValueError):tool.raw.validate_plan(p['llama'],p['llama']['futureRequest']['body'])

    def test_unreviewed_sampling_or_added_constraint_fails(self):
        for key,value in [('grammar','root ::= "x"'),('temperature',False),('n_probs',2),
                          ('n_predict',3),('post_sampling_probs',True),('cache_prompt',True)]:
            p=fixture();p['llama']['futureRequest']['body'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):tool.validate_plan(p)

    def test_prompt_context_or_first_token_mutation_fails(self):
        for change in ('prompt','context','first','candidates','backend','extra_arm'):
            p=fixture()
            if change=='prompt':p['llama']['futureRequest']['body']['prompt']=[0,1,2]
            if change=='context':p['diagnostic']['context']=31
            if change=='first':p['diagnostic']['first_token']=True
            if change=='candidates':p['diagnostic']['candidate_tokens']=[0,0]
            if change=='backend':p['diagnostic']['expected_backend_sampling']=True
            if change=='extra_arm':p['tensorsharp']['cache_budgets_mb'].append(1024)
            with self.subTest(change=change),self.assertRaises(ValueError):tool.validate_plan(p)


class DistributionTests(unittest.TestCase):
    def test_complete_two_rows_allow_cap_without_semantic_eos_claim(self):
        self.assertEqual([[-3.,-2.,-1.],[-2.,-1.,-3.]],tool.full_distributions(response(),fixture()))

    def test_wrong_consumed_first_or_counts_stop_or_backend_rejected(self):
        for key,value in [('tokens',[1,1]),('tokens',[True,1]),('tokens',[2]),('tokens_evaluated',3),
            ('tokens_predicted',1),('stop_type','eos'),('truncated',True)]:
            r=response();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):tool.full_distributions(r,fixture())
        for key,value in [('backend_sampling',True),('post_sampling_probs',True),('n_probs',2),
                          ('grammar','root ::= "x"'),('logit_bias',[[0,-10]]),('speculative.types','mtp'),
                          ('temperature',False),('samplers',['top_k','temperature'])]:
            r=response();r['generation_settings'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):tool.full_distributions(r,fixture())

    def test_incomplete_duplicate_out_of_range_boolean_ids_rejected(self):
        for kind in ('missing','duplicate','outside','boolean','row_id'):
            r=response();rows=r['completion_probabilities']
            if kind=='missing':rows[0]['top_logprobs'].pop()
            if kind=='duplicate':rows[0]['top_logprobs'][1]['id']=0
            if kind=='outside':rows[0]['top_logprobs'][1]['id']=3
            if kind=='boolean':rows[0]['top_logprobs'][1]['id']=True
            if kind=='row_id':rows[0]['id']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):tool.full_distributions(r,fixture())

    def test_nonfinite_clipped_boolean_and_positive_values_rejected(self):
        for value in (float('nan'),float('inf'),float('-inf'),-3.402823e38,-1e20,True,.001):
            r=response();r['completion_probabilities'][0]['top_logprobs'][0]['logprob']=value
            with self.subTest(value=value),self.assertRaises(ValueError):tool.full_distributions(r,fixture())

    def test_selected_id_must_be_argmax_and_prediction_rows_complete(self):
        r=response();r['completion_probabilities'][1]['top_logprobs'][0]['logprob']=-.1
        with self.assertRaisesRegex(ValueError,'argmax'):tool.full_distributions(r,fixture())
        r=response();r['completion_probabilities'].pop()
        with self.assertRaisesRegex(ValueError,'prediction rows'):tool.full_distributions(r,fixture())

    def test_common_offset_removes_normalization_without_hiding_margin_difference(self):
        self.assertEqual(0,tool.relative_difference([7.,8.,9.],[-3.,-2.,-1.])['max_abs_after_common_token_offset'])
        changed=tool.relative_difference([7.,9.,9.],[-3.,-2.,-1.])
        self.assertEqual(1,changed['max_abs_after_common_token_offset'])
        self.assertAlmostEqual(1/math.sqrt(5),changed['relative_l2_after_common_token_offset'])
        self.assertEqual(-1,tool.row_summary([-3.,-2.,-1.],[0,1])['candidate_margin'])
        pairwise=tool.relative_difference([-1.,2.,2.],[0.,1.,2.])
        self.assertEqual(1,pairwise['max_abs_after_common_token_offset'])
        self.assertEqual(2,pairwise['max_pairwise_margin_error'])

    def test_f64_storage_is_explicitly_normalized_and_preserves_api_values_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);values=tool.full_distributions(response(),fixture())
            tool.write_distributions(d,values,[0,1],2)
            index=tool.load(d/'rows.json');blob=(d/'logprobs.f64').read_bytes()
            self.assertEqual('normalized-logprob-f64le',index['format'])
            self.assertEqual([0,1,2],index['rows'][1]['input_tokens'])
            self.assertEqual(48,len(blob));self.assertEqual(tuple(values[1]),struct.unpack('<ddd',blob[24:]))
            self.assertEqual(hashlib.sha256(blob[24:]).hexdigest(),index['rows'][1]['sha256'])


def ts_fixture(directory):
    p=fixture();p['llama']['expected_checkpoint_eos_ids']=[99]
    p['tensorsharp']['deployment']=dict(files={})
    for name in ('GgmlOps.dll','TensorSharp.Models.dll'):
        f=directory/name;f.write_bytes(name.encode())
        p['tensorsharp']['deployment']['files'][name]=dict(path=str(f),sha256=tool.bench.sha(f))
    model=directory/'model.gguf';model.write_bytes(b'model')
    shard=dict(path=str(model),sha256='a'*64,bytes=5)
    p['llama']['checkpoint']['shards']=[copy.deepcopy(shard)]
    logdir=directory/'logits';logdir.mkdir()
    blob=logdir/'rows.f32';index_path=logdir/'rows.json';report_path=directory/'probe.json'
    blocks=[struct.pack('<fff',1,2,3),struct.pack('<fff',1,3,2)];blob.write_bytes(b''.join(blocks))
    rows=[dict(stage='prefill' if i==0 else 'decode',iteration=0,warmup=False,input_tokens=[0,1]+([2] if i else []),
        elements=3,byte_offset=i*12,sha256=hashlib.sha256(block).hexdigest()) for i,block in enumerate(blocks)]
    index=dict(format='f32le',data_path=str(blob),rows=rows)
    pins=p['tensorsharp']['deployment']['files']
    report=dict(passed=True,run_complete=True,error=None,synthetic=False,model_path=str(model),backend='GgmlCuda',placement='host',
        native_path=pins['GgmlOps.dll']['path'],native_sha256=pins['GgmlOps.dll']['sha256'],
        managed_assemblies_sha256={pins['TensorSharp.Models.dll']['path']:pins['TensorSharp.Models.dll']['sha256']},
        checkpoint_identity=dict(current_metadata_checked=True,manifest_sha256='b'*64,model_path=str(model),files=[shard]),
        model_geometry=dict(architecture='qwen4exp',hidden_size=4,layers=2,heads=2,kv_heads=1,vocabulary=3,context_limit=32,kv_dtype='f16'),
        cleanup=dict(model_disposed=True,cache_cleared=True,reuse_released=True,native_shutdown=True,
            retained_model_owner=False,retained_scope_owner=False,errors=[]),
        requested_options={'warmup':'0','iterations':'1','decode-tokens':'1','require-cache':'0'},
        cache_stats=dict(Calls=0,Hits=0,Misses=0,ReservedBytes=0,BudgetBytes=0),
        environment={'TS_HOST_MOE_EXPERT_CACHE_MB':'0'},logit_captures=dict(**index,index_path=str(index_path)),
        prompt_tokens=[0,1],forced_tokens=[2],decode_mode='teacher-forced',
        runs=[dict(iteration=0,warmup=False,prefill_tokens=2,decode_tokens=1,final_logit_sha256=rows[1]['sha256'])])
    tool.save(index_path,index);tool.save(report_path,report)
    return p,index,report


class TensorCaptureTests(unittest.TestCase):
    def test_single_cold_decode_proves_activation_without_claiming_warm_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p,_,r=ts_fixture(d)
            r['environment']['TS_HOST_MOE_EXPERT_CACHE_MB']='8192'
            r['cache_stats']=dict(Calls=2,Hits=0,Misses=20,ReservedBytes=1024,BudgetBytes=8192<<20)
            tool.save(d/'probe.json',r)
            self.assertEqual(2,tool.validate_ts(d,p,8192)[0]['validated_rows'])
            for key,value in [('Calls',0),('Misses',0),('ReservedBytes',0),('BudgetBytes',0),('Hits',True)]:
                bad=copy.deepcopy(r);bad['cache_stats'][key]=value;tool.save(d/'probe.json',bad)
                with self.assertRaises(ValueError):tool.validate_ts(d,p,8192)
            bad=copy.deepcopy(r);bad['requested_options']['require-cache']='1';tool.save(d/'probe.json',bad)
            with self.assertRaises(ValueError):tool.validate_ts(d,p,8192)

    def test_real_history_identity_and_cleanup_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p,_,_=ts_fixture(d)
            identity,values=tool.validate_ts(d,p,0)
            self.assertEqual(2,identity['validated_rows']);self.assertEqual([1,3,2],list(values[1]))

    def test_teacher_history_native_managed_checkpoint_cleanup_and_geometry_fail_closed(self):
        for kind in ('history','native','managed','checkpoint','cleanup','geometry','extra_row','chunk','budget'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp);p,index,r=ts_fixture(d)
                if kind=='history':index['rows'][1]['input_tokens']=[0,1,0]
                if kind=='native':r['native_sha256']='0'*64
                if kind=='managed':r['managed_assemblies_sha256']={}
                if kind=='checkpoint':p['llama']['checkpoint']['shards'][0]['sha256']='c'*64
                if kind=='cleanup':r['cleanup']['native_shutdown']=False
                if kind=='geometry':r['model_geometry']['context_limit']=31
                if kind=='extra_row':index['rows'].append(copy.deepcopy(index['rows'][1]))
                if kind=='chunk':r['requested_options']['prefill-chunk']='8'
                if kind=='budget':r['environment']['TS_HOST_MOE_EXPERT_CACHE_MB']='8192'
                tool.save(d/'probe.json',r);tool.save(d/'logits/rows.json',index)
                with self.assertRaises(ValueError):tool.validate_ts(d,p,0)


class LifetimeTests(unittest.TestCase):
    def test_loaded_identity_failure_reaps_an_actual_owned_cpu_child(self):
        p=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);model=directory/'model.gguf';model.write_bytes(b'model')
            manifest=directory/'integrity.json';manifest.write_text('{}')
            p['llama']['checkpoint']=dict(integrity_manifest=str(manifest),integrity_manifest_sha256=tool.bench.sha(manifest),
                shards=[dict(path=str(model),bytes=5,mtime_ns=model.stat().st_mtime_ns,sha256='a'*64)])
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            command=[sys.executable,'-c','import time; time.sleep(60)','--host','127.0.0.1','--port',str(port),
                     '--parallel','1','--ctx-size','32','--spec-type','none','--model',str(model)]
            p['llama']['futureCommand']=command
            p['llama']['futureRequest']['endpoint']=f'http://127.0.0.1:{port}/completion'
            p['llama']['deployment']['files']={Path(sys.executable).name:dict(path=sys.executable,sha256=tool.bench.sha(sys.executable))}
            p['llama']['environment']=dict(clear_prefixes=[],set={})
            sampler=mock.Mock();sampler.rows=[];sampler.errors=[];sampler.thread.is_alive.return_value=False
            process=mock.Mock();process.net_connections.return_value=[mock.Mock(status=tool.bench.psutil.CONN_LISTEN,
                laddr=mock.Mock(ip='127.0.0.1',port=port))]
            children=[];real_popen=subprocess.Popen
            def start(*args,**kwargs):
                child=real_popen(*args,**kwargs);children.append(child);return child
            with (mock.patch.object(tool.subprocess,'check_output',side_effect=['abc\n','']),
                  mock.patch.object(tool.subprocess,'Popen',side_effect=start),
                  mock.patch.object(tool.bench,'Sampler',return_value=sampler),
                  mock.patch.object(tool.bench.psutil,'Process',return_value=process),
                  mock.patch.object(tool.raw.owned,'request',return_value={}),
                  mock.patch.object(tool.raw,'observe_modules',side_effect=ValueError('loaded path mismatch'))):
                result=tool.run_llama(p,directory/'evidence',5)
            self.assertEqual(1,len(children));self.assertIsNotNone(children[0].poll())
            self.assertTrue(result['owned_server_exited']);self.assertFalse(result['cleanup_errors'])
            self.assertFalse(result['passed']);self.assertIn('loaded path mismatch',result['error'])

    def test_failed_preflight_preserves_terminal_evidence_without_launching_server(self):
        p=fixture()
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        p['llama']['futureCommand'][p['llama']['futureCommand'].index('--port')+1]=str(port)
        p['llama']['futureRequest']['endpoint']=f'http://127.0.0.1:{port}/completion'
        with (tempfile.TemporaryDirectory() as tmp,
              mock.patch.object(tool.subprocess,'check_output',side_effect=ValueError('source mismatch')),
              mock.patch.object(tool.subprocess,'Popen') as launch):
            out=Path(tmp)/'llama';result=tool.run_llama(p,out,3)
            launch.assert_not_called();self.assertFalse(result['passed']);self.assertTrue(result['owned_server_exited'])
            self.assertFalse(result['eos_validated']);self.assertFalse(result['semantic_quality_validated'])
            self.assertIn('source mismatch',tool.load(out/'execution.json')['error'])

    def test_duplicate_json_object_keys_rejected_before_model_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json';path.write_text('{"tokens":[2,1],"tokens":[1,2]}')
            with self.assertRaises(ValueError):tool.load(path)

    def test_deployment_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p,_,_=ts_fixture(d)
            tool.raw.deployment_hashes(p['tensorsharp'])
            (d/'GgmlOps.dll').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Deployment changed'):tool.raw.deployment_hashes(p['tensorsharp'])


if __name__=='__main__':unittest.main()
